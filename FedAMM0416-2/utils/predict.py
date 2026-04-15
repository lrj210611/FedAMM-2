import logging
import os
import time
from unittest.mock import patch

from matplotlib.colors import ListedColormap
try:
    import nibabel as nib
except ModuleNotFoundError:  # pragma: no cover - nibabel is not needed for the npy smoke-test path
    nib = None
import numpy as np
import scipy.misc
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
try:
    from medpy.metric import hd95
except ModuleNotFoundError:  # pragma: no cover - hd95 is unused in the current smoke-test path
    hd95 = None
import csv
# from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

cudnn.benchmark = True

path = os.path.dirname(__file__)

patch_size = 80


def softmax_output_dice_class4(output, target):
    eps = 1e-8
    #######label1########
    o1 = (output == 1).float()
    t1 = (target == 1).float()
    intersect1 = torch.sum(2 * (o1 * t1), dim=(1,2,3)) + eps
    denominator1 = torch.sum(o1, dim=(1,2,3)) + torch.sum(t1, dim=(1,2,3)) + eps
    ncr_net_dice = intersect1 / denominator1

    o2 = (output == 2).float()
    t2 = (target == 2).float()
    intersect2 = torch.sum(2 * (o2 * t2), dim=(1,2,3)) + eps
    denominator2 = torch.sum(o2, dim=(1,2,3)) + torch.sum(t2, dim=(1,2,3)) + eps
    edema_dice = intersect2 / denominator2

    o3 = (output == 3).float()
    t3 = (target == 3).float()
    intersect3 = torch.sum(2 * (o3 * t3), dim=(1,2,3)) + eps
    denominator3 = torch.sum(o3, dim=(1,2,3)) + torch.sum(t3, dim=(1,2,3)) + eps
    enhancing_dice = intersect3 / denominator3

    ####post processing:
    if torch.sum(o3) < 500:
       o4 = o3 * 0.0
    else:
       o4 = o3
    t4 = t3
    intersect4 = torch.sum(2 * (o4 * t4), dim=(1,2,3)) + eps
    denominator4 = torch.sum(o4, dim=(1,2,3)) + torch.sum(t4, dim=(1,2,3)) + eps
    enhancing_dice_postpro = intersect4 / denominator4

    o_whole = o1 + o2 + o3 
    t_whole = t1 + t2 + t3 
    intersect_whole = torch.sum(2 * (o_whole * t_whole), dim=(1,2,3)) + eps
    denominator_whole = torch.sum(o_whole, dim=(1,2,3)) + torch.sum(t_whole, dim=(1,2,3)) + eps
    dice_whole = intersect_whole / denominator_whole

    o_core = o1 + o3
    t_core = t1 + t3
    intersect_core = torch.sum(2 * (o_core * t_core), dim=(1,2,3)) + eps
    denominator_core = torch.sum(o_core, dim=(1,2,3)) + torch.sum(t_core, dim=(1,2,3)) + eps
    dice_core = intersect_core / denominator_core

    dice_separate = torch.cat((torch.unsqueeze(ncr_net_dice, 1), torch.unsqueeze(edema_dice, 1), torch.unsqueeze(enhancing_dice, 1)), dim=1)
    dice_evaluate = torch.cat((torch.unsqueeze(dice_whole, 1), torch.unsqueeze(dice_core, 1), torch.unsqueeze(enhancing_dice, 1), torch.unsqueeze(enhancing_dice_postpro, 1)), dim=1)

    return dice_separate.cpu().numpy(), dice_evaluate.cpu().numpy()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _prepare_dense_target(target):
    if target.dim() == 5 and target.size(1) > 1:
        target = torch.argmax(target, dim=1)
    elif target.dim() == 5 and target.size(1) == 1:
        target = target[:, 0]
    return target.long()


def validate_dice_softmax(
        val_loader,
        model,
        feature_mask=None,
        device='cuda',
        max_samples=None,
        log_prefix=None,
        ):
    start_time = time.time()
    was_training_mode = model.training
    model.eval()
    model.to(device)
    vals_dice_evaluation = AverageMeter()
    one_tensor = torch.ones(1, patch_size, patch_size, patch_size).float().to(device)
    num_cls = 4
    class_evaluation = 'whole', 'core', 'enhancing', 'enhancing_postpro'

    previous_is_training = getattr(model, 'is_training', None)
    model.is_training = False

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            if max_samples is not None and i >= max_samples:
                break

            target = _prepare_dense_target(data[1].to(device))
            x = data[0].to(device)
            names = data[-1]
            if feature_mask is not None:
                mask = torch.from_numpy(np.array(feature_mask))
                mask = torch.unsqueeze(mask, dim=0).repeat(len(names), 1)
            else:
                mask = data[2]
            mask = mask.to(device)
            _, _, H, W, Z = x.size()

            h_cnt = np.int_(np.ceil((H - patch_size) / (patch_size * (1 - 0.5))))
            h_idx_list = range(0, h_cnt)
            h_idx_list = [h_idx * np.int_(patch_size * (1 - 0.5)) for h_idx in h_idx_list]
            h_idx_list.append(H - patch_size)

            w_cnt = np.int_(np.ceil((W - patch_size) / (patch_size * (1 - 0.5))))
            w_idx_list = range(0, w_cnt)
            w_idx_list = [w_idx * np.int_(patch_size * (1 - 0.5)) for w_idx in w_idx_list]
            w_idx_list.append(W - patch_size)

            z_cnt = np.int_(np.ceil((Z - patch_size) / (patch_size * (1 - 0.5))))
            z_idx_list = range(0, z_cnt)
            z_idx_list = [z_idx * np.int_(patch_size * (1 - 0.5)) for z_idx in z_idx_list]
            z_idx_list.append(Z - patch_size)

            weight1 = torch.zeros(1, 1, H, W, Z).float().to(device)
            for h in h_idx_list:
                for w in w_idx_list:
                    for z in z_idx_list:
                        weight1[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size] += one_tensor
            weight = weight1.repeat(len(names), num_cls, 1, 1, 1)

            pred = torch.zeros(len(names), num_cls, H, W, Z).float().to(device)
            for h in h_idx_list:
                for w in w_idx_list:
                    for z in z_idx_list:
                        x_input = x[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size]
                        pred_part = model(x_input, mask)
                        pred[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size] += pred_part
            pred = pred / weight
            pred = pred[:, :, :H, :W, :Z]
            pred = torch.argmax(pred, dim=1)

            _, scores_evaluation = softmax_output_dice_class4(pred, target)
            for k, name in enumerate(names):
                vals_dice_evaluation.update(scores_evaluation[k])
                if log_prefix is not None:
                    msg = '{} Subject {}/{}, {:>20}, '.format(
                        log_prefix, i + 1, min(len(val_loader), max_samples or len(val_loader)), name
                    )
                    msg += ', '.join(['{}: {:.4f}'.format(metric_name, metric_value) for metric_name, metric_value in zip(class_evaluation, scores_evaluation[k])])
                    logging.info(msg)

    if previous_is_training is not None:
        model.is_training = previous_is_training
    model.cpu()
    if was_training_mode:
        model.train()
    else:
        model.eval()

    logging.info(
        'validation time: {:.2f} minutes, avg_dice={}'.format(
            (time.time() - start_time) / 60,
            vals_dice_evaluation.avg,
        )
    )
    return vals_dice_evaluation.avg, class_evaluation

def test_dice_softmax(
        test_loader,
        model,
        dataname = 'BraTS/BRATS2020',
        feature_mask=None,
        mask_name=None,
        csv_name=None,
        device='cuda',
        method_name=None,
        output_dir=None
        ):
    start_time = time.time()
    H, W, T = 240, 240, 155
    model.eval()
    model.to(device)
    vals_dice_evaluation = AverageMeter()
    vals_separate = AverageMeter()
    one_tensor = torch.ones(1, patch_size, patch_size, patch_size).float().to(device)
    num_cls = 4
    class_evaluation= 'whole', 'core', 'enhancing', 'enhancing_postpro'
    class_separate = 'ncr_net', 'edema', 'enhancing'
    

    #创建一个字典用来保存最好结果的name和dice_score
    best_dice_score = {"name":None,"dice_score":0}
    for i, data in enumerate(test_loader):
        target = data[1].to(device)
        x = data[0].to(device)
        names = data[-1]
        if feature_mask is not None:
            mask = torch.from_numpy(np.array(feature_mask))
            mask = torch.unsqueeze(mask, dim=0).repeat(len(names), 1)
        else:
            mask = data[2]
        mask = mask.to(device)
        _, _, H, W, Z = x.size()
        #########get h_ind, w_ind, z_ind for sliding windows
        h_cnt = np.int_(np.ceil((H - patch_size) / (patch_size * (1 - 0.5))))
        h_idx_list = range(0, h_cnt)
        h_idx_list = [h_idx * np.int_(patch_size * (1 - 0.5)) for h_idx in h_idx_list]
        h_idx_list.append(H - patch_size)

        w_cnt = np.int_(np.ceil((W - patch_size) / (patch_size * (1 - 0.5))))
        w_idx_list = range(0, w_cnt)
        w_idx_list = [w_idx * np.int_(patch_size * (1 - 0.5)) for w_idx in w_idx_list]
        w_idx_list.append(W - patch_size)

        z_cnt = np.int_(np.ceil((Z - patch_size) / (patch_size * (1 - 0.5))))
        z_idx_list = range(0, z_cnt)
        z_idx_list = [z_idx * np.int_(patch_size * (1 - 0.5)) for z_idx in z_idx_list]
        z_idx_list.append(Z - patch_size)

        #####compute calculation times for each pixel in sliding windows
        weight1 = torch.zeros(1, 1, H, W, Z).float().to(device)
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    weight1[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size] += one_tensor
        weight = weight1.repeat(len(names), num_cls, 1, 1, 1)

        #####evaluation
        pred = torch.zeros(len(names), num_cls, H, W, Z).float().to(device)
        model.is_training=False
        
        for h in h_idx_list:
            for w in w_idx_list:
                for z in z_idx_list:
                    x_input = x[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size]
                    pred_part = model(x_input, mask)
                    pred[:, :, h:h+patch_size, w:w+patch_size, z:z+patch_size] += pred_part
        pred = pred / weight
        b = time.time()
        pred = pred[:, :, :H, :W, :Z]
        pred = torch.argmax(pred, dim=1)

        if method_name is not None:
            for k, name in enumerate(names):
                visualize_and_save(x[k].cpu().numpy(), target[k].cpu().numpy(), pred[k].cpu().numpy(), slice_index=75, output_dir=output_dir , name=name, mask_name=mask_name, method_name=method_name)


        scores_separate, scores_evaluation = softmax_output_dice_class4(pred, target)

        for k, name in enumerate(names):
            msg = 'Subject {}/{}, {}/{}'.format((i+1), len(test_loader), (k+1), len(names))
            msg += '{:>20}, '.format(name)

            vals_separate.update(scores_separate[k])
            vals_dice_evaluation.update(scores_evaluation[k])
            msg += 'DSC: '
            msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, scores_evaluation[k])])
            file = open(csv_name, "a+")
            csv_writer = csv.writer(file)
            csv_writer.writerow([name,scores_evaluation[k][0], scores_evaluation[k][1], scores_evaluation[k][2],scores_evaluation[k][3],(scores_evaluation[k][0]+scores_evaluation[k][1]+scores_evaluation[k][2])/3])
            file.close()
            #msg += ',' + ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_separate, scores_separate[k])])
            if best_dice_score["dice_score"] < scores_evaluation[k][0]+scores_evaluation[k][1]+scores_evaluation[k][2]+scores_evaluation[k][3]:
                best_dice_score["name"] = name
                best_dice_score["dice_score"] = scores_evaluation[k][0]+scores_evaluation[k][1]+scores_evaluation[k][2]+scores_evaluation[k][3]
            logging.info(msg)
    msg = 'Average scores:'
    msg += ' DSC: '
    msg += ', '.join(['{}: {:.4f}'.format(k, v) for k, v in zip(class_evaluation, vals_dice_evaluation.avg)])
    msg += ' avg:{}'.format((vals_dice_evaluation.avg[0]+vals_dice_evaluation.avg[1]+vals_dice_evaluation.avg[2])/3)
    print(msg)
    logging.info(best_dice_score)
    logging.info(msg)
    logging.info('evaluation time: {:.2f} minutes'.format((time.time() - start_time)/60))
    model.train()
    return vals_dice_evaluation.avg,class_evaluation



def visualize_and_save(image, label, prediction, slice_index, output_dir, name,mask_name,method_name ):
    output_dir = output_dir+"/"+name
    os.makedirs(output_dir, exist_ok=True)

    label_slice = label[0, :, :, slice_index] if label.ndim == 4 else label[:, :, slice_index]
    prediction_slice = prediction[0, :, :, slice_index] if prediction.ndim == 4 else prediction[:, :, slice_index]

    label_slice = np.where(label_slice > 0, label_slice, 0)
    prediction_slice = np.where(prediction_slice > 0, prediction_slice, 0)

    label_colors = ['black', 'red', 'green', 'blue']
    label_cmap = ListedColormap(label_colors)
    # 创建一个图形窗口
    # 显示原始图像切片
    if method_name == 'FedAvg':
        visualize_four_images(image, ['flair', 't1', 't1ce', 't2'], slice_index, output_dir, name)
        label_slice = np.rot90(label_slice, k=-1)
        print("label_slice.shape",label_slice.shape)
        ##将四个原始图像按照2行2列显示出来，并在下方标注模态信息
        plt.figure(figsize=(1, 1))
        plt.imshow(label_slice, cmap=label_cmap, alpha=1.0,interpolation='nearest')
        #plt.title('Label Slice')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'label.png'),  bbox_inches='tight',pad_inches=0,dpi=300)
        #plt.savefig(os.path.join(output_dir, f'label_vec.svg'), format='svg', bbox_inches='tight',pad_inches=0,dpi=300)
        plt.close()

    prediction_slice = np.rot90(prediction_slice, k=-1)
    print("prediction_slice.shape",prediction_slice.shape)
    # 显示预测结果切片
    plt.figure(figsize=(1, 1))
    plt.imshow(prediction_slice, cmap=label_cmap, alpha=1.0,interpolation='nearest')
    #plt.title('Prediction Slice')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, f'{mask_name}_{method_name}_prediction.png'), bbox_inches='tight', pad_inches=0,dpi=300)
    plt.close()


def visualize_four_images(images,modalities,slice_index, output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    

    for i, modality in enumerate(modalities):
        # 选择一个切片进行可视化
        image_slice = images[i, :, :, slice_index] if images.ndim == 4 else images[:, :, slice_index]
        # 创建一个图形窗口
        image_slice = np.rot90(image_slice, k=-1)
        print("image_slice.shape",image_slice.shape)
        plt.figure(figsize=(1, 1))
        # 显示图像
        plt.imshow(image_slice, cmap='gray', interpolation='nearest')
        plt.axis('off')

        # 调整子图的布局
        plt.tight_layout()

        # 保存图像
        plt.savefig(os.path.join(output_dir, f'{name}_{modality}_{slice_index}.png'), bbox_inches='tight', pad_inches=0,dpi=300)
        plt.close()

