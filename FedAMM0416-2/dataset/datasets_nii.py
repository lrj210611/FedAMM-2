import os

try:
    import nibabel as nib
except ModuleNotFoundError:  # pragma: no cover - nibabel is not needed for the npy smoke-test path
    nib = None
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .data_utils import pkload
from .rand import Uniform
from .transforms import (CenterCrop, Compose, Flip, GaussianBlur, Identity,
                         Noise, Normalize, NumpyType, Pad, RandCrop,
                         RandCrop3D, RandomFlip, RandomIntensityChange,
                         RandomRotion, RandSelect, Rot90)
import glob
import random

HGG = []
LGG = []
for i in range(0, 260):
    HGG.append(str(i).zfill(3))
for i in range(336, 370):
    HGG.append(str(i).zfill(3))
for i in range(260, 336):
    LGG.append(str(i).zfill(3))

mask_array = np.array([[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]])
full_mask_array = np.array([True, True, True, True])
mask_valid_array = np.array([[False, False, True, False],
            [False, True, True, False],
            [True, True, False, True],
            [True, True, True, True]])


def _load_case_names(data_file_path):
    if data_file_path is None:
        raise ValueError('data_file_path is required for Brats full-modal labeled dataset')
    ext = os.path.splitext(data_file_path)[1].lower()
    if ext == '.csv':
        excel_data = pd.read_csv(data_file_path)
        if 'data_name' not in excel_data.columns:
            raise ValueError(f'"data_name" column is required in csv file: {data_file_path}')
        datalist = excel_data['data_name'].astype(str).tolist()
    else:
        with open(data_file_path, 'r') as f:
            datalist = [i.strip() for i in f.readlines()]
    datalist = [name for name in datalist if name]
    datalist.sort()
    return datalist


def _pad_volume_to_patch_size(volume, patch_size):
    """将体数据补到不小于 patch_size，避免共享裁剪时出现负索引。"""
    pad_width = []
    for dim_idx, target_size in enumerate(patch_size):
        current_size = volume.shape[dim_idx]
        if current_size >= target_size:
            pad_width.append((0, 0))
            continue
        total_pad = target_size - current_size
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        pad_width.append((pad_before, pad_after))

    if volume.ndim == 4:
        pad_width.append((0, 0))
    if any(before > 0 or after > 0 for before, after in pad_width):
        volume = np.pad(volume, pad_width, mode='constant')
    return volume


def _sample_shared_patch(volume, patch_size):
    """
    从同一个 volume 中采样共享空间 patch。
    teacher-student 的伪标签学习要求 weak/strong 在同一空间中逐 voxel 对齐，
    因此必须先共享裁剪，再分别做不改变空间坐标的强度增强。
    """
    volume = _pad_volume_to_patch_size(volume, patch_size)
    starts = []
    for dim_idx, patch_dim in enumerate(patch_size):
        max_start = volume.shape[dim_idx] - patch_dim
        start = random.randint(0, max_start) if max_start > 0 else 0
        starts.append(start)
    h_start, w_start, z_start = starts
    h_size, w_size, z_size = patch_size
    return volume[
        h_start:h_start + h_size,
        w_start:w_start + w_size,
        z_start:z_start + z_size,
        ...
    ]


def _apply_random_intensity_change(volume, shift, scale):
    """按通道做轻量强度扰动，不改变空间结构。"""
    shift_factor = np.random.uniform(
        low=-shift,
        high=shift,
        size=(1, volume.shape[1], 1, 1, volume.shape[4]),
    )
    scale_factor = np.random.uniform(
        low=1.0 - scale,
        high=1.0 + scale,
        size=(1, volume.shape[1], 1, 1, volume.shape[4]),
    )
    return volume * scale_factor + shift_factor


def _apply_random_multiplicative_noise(volume, sigma):
    """乘性噪声增强，保持 weak/strong 的空间坐标不变。"""
    noise = np.exp(sigma * np.random.randn(1, 1, 1, 1, volume.shape[4])).astype(np.float32)
    return volume * noise


def _apply_random_gamma(volume, gamma_min=0.7, gamma_max=1.5):
    """对每个模态单独做 gamma 变化，只改变强度分布。"""
    volume = volume.copy()
    gamma = np.random.uniform(gamma_min, gamma_max)
    for channel_idx in range(volume.shape[4]):
        channel = volume[0, ..., channel_idx]
        valid_mask = channel != 0
        if not np.any(valid_mask):
            continue
        values = channel[valid_mask]
        value_min = values.min()
        value_max = values.max()
        if value_max <= value_min:
            continue
        normalized = (values - value_min) / (value_max - value_min)
        channel[valid_mask] = np.power(normalized, gamma) * (value_max - value_min) + value_min
        volume[0, ..., channel_idx] = channel
    return volume


def _build_weak_view_from_patch(shared_patch):
    """weak view 只允许 identity 或轻强度扰动，不能破坏空间对齐。"""
    weak_patch = shared_patch.copy()
    if random.random() < 0.5:
        weak_patch = _apply_random_intensity_change(weak_patch, shift=0.05, scale=0.05)
    return weak_patch.astype(np.float32, copy=False)


def _build_strong_view_from_patch(shared_patch):
    """strong view 使用强度类增强，但不允许再次随机裁剪或几何变换。"""
    strong_patch = shared_patch.copy()
    if random.random() < 0.8:
        strong_patch = _apply_random_intensity_change(strong_patch, shift=0.1, scale=0.1)
    if random.random() < 0.5:
        strong_patch = _apply_random_multiplicative_noise(strong_patch, sigma=0.1)
    if random.random() < 0.5:
        strong_patch = _apply_random_gamma(strong_patch, gamma_min=0.7, gamma_max=1.5)
    return strong_patch.astype(np.float32, copy=False)


class Brats_loadall_train_nii_pdt(Dataset):
    def __init__(self, transforms='', root=None, modal='all', num_cls=4, train_file='train.txt'):
        data_file_path = os.path.join(train_file)
        with open(data_file_path, 'r') as f:
            datalist = [i.strip() for i in f.readlines()]
        datalist.sort()

        volpaths = []
        for dataname in datalist:
            volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))

        self.volpaths = volpaths
        self.transforms = eval(transforms or 'Identity()')
        self.names = datalist
        self.num_cls = num_cls
        if modal == 'flair':
            self.modal_ind = np.array([0])
        elif modal == 't1ce':
            self.modal_ind = np.array([1])
        elif modal == 't1':
            self.modal_ind = np.array([2])
        elif modal == 't2':
            self.modal_ind = np.array([3])
        elif modal == 'all':
            self.modal_ind = np.array([0,1,2,3])

    def __getitem__(self, index):

        volpath = self.volpaths[index]
        name = self.names[index]
        
        x = np.load(volpath)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath)
        x, y = x[None, ...], y[None, ...]

        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
        _, H, W, Z = np.shape(y)
        y = np.reshape(y, (-1))
        one_hot_targets = np.eye(self.num_cls)[y]
        yo = np.reshape(one_hot_targets, (1, H, W, Z, -1))
        yo = np.ascontiguousarray(yo.transpose(0, 4, 1, 2, 3))

        x = x[:, self.modal_ind, :, :, :]

        x = torch.squeeze(torch.from_numpy(x), dim=0)
        yo = torch.squeeze(torch.from_numpy(yo), dim=0)

        mask_idx = np.random.choice(15, 1)
        mask = torch.squeeze(torch.from_numpy(mask_array[mask_idx]), dim=0)
        return x, yo, mask, name

    def __len__(self):
        return len(self.volpaths)

class Brats_loadall_train_nii_idt(Dataset):
    def __init__(self, transforms='', root=None, modal='all', num_cls=4, mask_type='idt', train_file=None):
        self.excel_path = train_file
        excel_data = pd.read_csv(self.excel_path)
        data_name = excel_data['data_name']
        data_name_list = data_name.values.tolist()
        mask_id = excel_data['mask_id']
        mask_id_list = mask_id.values.tolist()
        samples_mask = excel_data['mask']
        samples_mask_list = samples_mask.values.tolist()
        pos_mask_id = excel_data['pos_mask_ids']
        pos_mask_id_list = pos_mask_id.values.tolist()

        volpaths = []
        for dataname in data_name_list:
            volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))

        self.volpaths = volpaths
        self.transforms = eval(transforms or 'Identity()')
        self.names = data_name_list
        self.mask_ids = mask_id_list
        self.samples_masks = samples_mask_list
        self.pos_mask_ids = pos_mask_id_list
        self.num_cls = num_cls
        self.mask_type = mask_type
        if modal == 'flair':
            self.modal_ind = np.array([0])
        elif modal == 't1ce':
            self.modal_ind = np.array([1])
        elif modal == 't1':
            self.modal_ind = np.array([2])
        elif modal == 't2':
            self.modal_ind = np.array([3])
        elif modal == 'all':
            self.modal_ind = np.array([0,1,2,3])

    def __getitem__(self, index):

        volpath = self.volpaths[index]
        name = self.names[index]
        if self.mask_type == 'idt':
            mask_idx = np.array([self.mask_ids[index]])
        elif self.mask_type == 'idt_drop':
            mask_idx = np.random.choice(eval(self.pos_mask_ids[index]),1)
        elif self.mask_type == 'pdt':
            mask_idx = np.random.choice(15, 1)

        x = np.load(volpath)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath)
        x, y = x[None, ...], y[None, ...]

        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
        _, H, W, Z = np.shape(y)
        y = np.reshape(y, (-1))
        one_hot_targets = np.eye(self.num_cls)[y]
        yo = np.reshape(one_hot_targets, (1, H, W, Z, -1))
        yo = np.ascontiguousarray(yo.transpose(0, 4, 1, 2, 3))

        x = x[:, self.modal_ind, :, :, :]
        x = torch.squeeze(torch.from_numpy(x), dim=0)
        yo = torch.squeeze(torch.from_numpy(yo), dim=0)

        mask = torch.squeeze(torch.from_numpy(mask_array[mask_idx]), dim=0)
        return x, yo, mask, name

    def __len__(self):
        return len(self.volpaths)


class Brats_loadall_labeled_full_nii(Dataset):
    def __init__(self, transforms='', root=None, modal='all', num_cls=4, train_file=None):
        datalist = _load_case_names(train_file)

        volpaths = []
        for dataname in datalist:
            volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))

        self.volpaths = volpaths
        self.transforms = eval(transforms or 'Identity()')
        self.names = datalist
        self.num_cls = num_cls
        if modal == 'flair':
            self.modal_ind = np.array([0])
        elif modal == 't1ce':
            self.modal_ind = np.array([1])
        elif modal == 't1':
            self.modal_ind = np.array([2])
        elif modal == 't2':
            self.modal_ind = np.array([3])
        elif modal == 'all':
            self.modal_ind = np.array([0,1,2,3])

    def __getitem__(self, index):
        volpath = self.volpaths[index]
        name = self.names[index]

        x = np.load(volpath)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath)
        x, y = x[None, ...], y[None, ...]

        x, y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))
        _, H, W, Z = np.shape(y)
        y = np.reshape(y, (-1))
        one_hot_targets = np.eye(self.num_cls)[y]
        yo = np.reshape(one_hot_targets, (1, H, W, Z, -1))
        yo = np.ascontiguousarray(yo.transpose(0, 4, 1, 2, 3))

        x = x[:, self.modal_ind, :, :, :]
        x = torch.squeeze(torch.from_numpy(x), dim=0)
        yo = torch.squeeze(torch.from_numpy(yo), dim=0)
        mask = torch.from_numpy(full_mask_array.copy())

        return x, yo, mask, name

    def __len__(self):
        return len(self.volpaths)


class Brats_loadall_unlabeled_missing_nii(Dataset):
    def __init__(
        self,
        weak_transforms='',
        strong_transforms='',
        root=None,
        modal='all',
        train_file=None,
        mask_type='idt',
        patch_size=(80, 80, 80),
    ):
        self.excel_path = train_file
        excel_data = pd.read_csv(self.excel_path)
        data_name_list = excel_data['data_name'].astype(str).tolist()
        self.mask_ids = excel_data['mask_id'].tolist() if 'mask_id' in excel_data.columns else [14] * len(data_name_list)
        self.pos_mask_ids = excel_data['pos_mask_ids'].tolist() if 'pos_mask_ids' in excel_data.columns else [str([14])] * len(data_name_list)

        self.volpaths = [os.path.join(root, 'vol', dataname + '_vol.npy') for dataname in data_name_list]
        # 保留原有参数签名以兼容 train.py，但 Module 2 的无标注数据流不能继续复用
        # 含随机裁剪/旋转的 transforms。teacher-student pseudo-label learning 必须保证
        # weak/strong 在同一空间 patch 上逐 voxel 对齐，否则 consistency 和 pseudo CE/Dice
        # 都不成立。因此这里统一采用“共享空间裁剪 + 仅强度增强”的内部数据流。
        self.weak_transforms = weak_transforms
        self.strong_transforms = strong_transforms
        self.names = data_name_list
        self.mask_type = mask_type
        self.patch_size = tuple(patch_size)
        if modal == 'flair':
            self.modal_ind = np.array([0])
        elif modal == 't1ce':
            self.modal_ind = np.array([1])
        elif modal == 't1':
            self.modal_ind = np.array([2])
        elif modal == 't2':
            self.modal_ind = np.array([3])
        elif modal == 'all':
            self.modal_ind = np.array([0,1,2,3])

    def __getitem__(self, index):
        volpath = self.volpaths[index]
        name = self.names[index]

        if self.mask_type == 'idt':
            mask_idx = np.array([self.mask_ids[index]])
        elif self.mask_type == 'idt_drop':
            mask_idx = np.random.choice(eval(self.pos_mask_ids[index]), 1)
        else:
            mask_idx = np.random.choice(15, 1)

        x = np.load(volpath).astype(np.float32, copy=False)
        shared_patch = _sample_shared_patch(x, self.patch_size)[None, ...]
        weak_x = _build_weak_view_from_patch(shared_patch)
        strong_x = _build_strong_view_from_patch(shared_patch)

        weak_x = np.ascontiguousarray(weak_x.transpose(0, 4, 1, 2, 3))
        strong_x = np.ascontiguousarray(strong_x.transpose(0, 4, 1, 2, 3))
        weak_x = weak_x[:, self.modal_ind, :, :, :]
        strong_x = strong_x[:, self.modal_ind, :, :, :]

        weak_x = torch.squeeze(torch.from_numpy(weak_x), dim=0)
        strong_x = torch.squeeze(torch.from_numpy(strong_x), dim=0)
        mask = torch.squeeze(torch.from_numpy(mask_array[mask_idx]), dim=0)
        mask_id = int(mask_idx[0])

        return weak_x, strong_x, mask, mask_id, name

    def __len__(self):
        return len(self.volpaths)

class Brats_loadall_test_nii(Dataset):
    def __init__(self, transforms='', root=None, modal='all', test_file='test.txt'):
        data_file_path = os.path.join(test_file)
        with open(data_file_path, 'r') as f:
            datalist = [i.strip() for i in f.readlines()]
        datalist.sort()
        volpaths = []
        for dataname in datalist:
            volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))
        self.volpaths = volpaths
        self.transforms = eval(transforms or 'Identity()')
        self.names = datalist
        if modal == 'flair':
            self.modal_ind = np.array([0])
        elif modal == 't1ce':
            self.modal_ind = np.array([1])
        elif modal == 't1':
            self.modal_ind = np.array([2])
        elif modal == 't2':
            self.modal_ind = np.array([3])
        elif modal == 'all':
            self.modal_ind = np.array([0,1,2,3])

    def __getitem__(self, index):

        volpath = self.volpaths[index]
        name = self.names[index]
        x = np.load(volpath)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath).astype(np.uint8)
        x, y = x[None, ...], y[None, ...]
        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
        y = np.ascontiguousarray(y)

        x = x[:, self.modal_ind, :, :, :]
        x = torch.squeeze(torch.from_numpy(x), dim=0)
        y = torch.squeeze(torch.from_numpy(y), dim=0)

        return x, y, name

    def __len__(self):
        return len(self.volpaths)


class Brats_loadall_val_nii(Dataset):
    def __init__(self, transforms='', root=None, modal='all', num_cls=4, train_file='val.txt'):
        data_file_path = os.path.join(train_file)
        with open(data_file_path, 'r') as f:
            datalist = [i.strip() for i in f.readlines()]
        datalist.sort()

        volpaths = []
        for dataname in datalist:
            volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))

        self.volpaths = volpaths
        self.transforms = eval(transforms or 'Identity()')
        self.names = datalist
        self.num_cls = num_cls
        if modal == 'flair':
            self.modal_ind = np.array([0])
        elif modal == 't1ce':
            self.modal_ind = np.array([1])
        elif modal == 't1':
            self.modal_ind = np.array([2])
        elif modal == 't2':
            self.modal_ind = np.array([3])
        elif modal == 'all':
            self.modal_ind = np.array([0,1,2,3])

    def __getitem__(self, index):

        volpath = self.volpaths[index]
        name = self.names[index]
        
        x = np.load(volpath)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath)
        x, y = x[None, ...], y[None, ...]

        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
        y = np.ascontiguousarray(y.astype(np.uint8))

        x = x[:, self.modal_ind, :, :, :]

        x = torch.squeeze(torch.from_numpy(x), dim=0)
        y = torch.squeeze(torch.from_numpy(y), dim=0)

        # Validation for R_sup should be deterministic. Use full-modal mask by default,
        # and let the caller override it through feature_mask when needed.
        mask = torch.from_numpy(full_mask_array.copy())
        return x, y, mask, name

    def __len__(self):
        return len(self.volpaths)
