import ast
import pandas as pd
import torch.backends.cudnn as cudnn
import csv
import torch
import os
import random
import numpy as np
import time
import copy
import importlib
from torch import nn
from tqdm import tqdm
from datetime import datetime
import logging
from torch.utils.data import DataLoader
from dataset.data_utils import init_fn

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:  # pragma: no cover - smoke-test fallback when tensorboard is unavailable
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass

from models import rfnet
from utils.fl_utils import avg_local_weights,avg_encoder_weights
from utils.lr_scheduler import LR_Scheduler
from utils import criterions
from dataset.datasets_nii import (
    Brats_loadall_train_nii_idt,
    Brats_loadall_test_nii,
    Brats_loadall_val_nii,
    Brats_loadall_labeled_full_nii,
    Brats_loadall_unlabeled_missing_nii,
)
from options import args_parser
from utils.predict import test_dice_softmax,AverageMeter,validate_dice_softmax
from utils.fedmass_anchor import (
    aggregate_global_anchor_bank,
    compute_anchor_supervision,
    finalize_local_anchor_state,
    init_local_anchor_state,
    normalize_anchor_bank,
    update_local_anchor_state,
)
from utils.fedmass_pseudo import (
    compute_pseudo_filtering,
    create_ema_teacher,
    update_ema_teacher,
)
from utils.fedmass_missing_proto import (
    aggregate_global_missing_proto_bank,
    compute_missing_pattern_alignment_loss,
    count_valid_pattern_classes,
    finalize_local_missing_proto_state,
    init_local_missing_proto_state,
    normalize_missing_proto_bank,
    update_local_missing_proto_state,
)
from utils.fedmass_reliability import (
    build_reliability_aggregation,
    init_reliability_state,
    restore_reliability_state,
    update_client_sup_history,
)

from multiprocessing import Pool

setproctitle = importlib.util.find_spec("setproctitle")
if setproctitle is not None:
    importlib.import_module("setproctitle").setproctitle("donot use 0123 gpu!")

###modality missing mask
masks_test = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]
masks_torch = torch.from_numpy(np.array(masks_test))
mask_name = ['t2', 't1c', 't1', 'flair',
            't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
            'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
            'flairt1cet1t2']

masks_all = [True, True, True, True]
masks_all_torch = torch.from_numpy(np.array(masks_all))


def _parse_mask_value(mask_value):
    if isinstance(mask_value, str):
        mask_value = ast.literal_eval(mask_value)
    mask_list = [bool(flag) for flag in mask_value]
    if len(mask_list) != len(masks_all):
        raise ValueError('Expected {} modalities, got {}'.format(len(masks_all), len(mask_list)))
    return mask_list


def _infer_mask_id(mask_value):
    mask_list = _parse_mask_value(mask_value)
    try:
        return masks_test.index(mask_list)
    except ValueError as exc:
        raise ValueError('Unknown modality mask: {}'.format(mask_list)) from exc


def summarize_split_stream(train_file, default_mask=None):
    csv_data = pd.read_csv(train_file)
    sample_count = len(csv_data)
    if 'mask' in csv_data.columns:
        mask_values = [_parse_mask_value(mask_value) for mask_value in csv_data['mask'].tolist()]
    elif 'mask_id' in csv_data.columns:
        mask_values = [list(masks_test[int(mask_id)]) for mask_id in csv_data['mask_id'].tolist()]
    else:
        if default_mask is None:
            raise ValueError('mask column is required for split stats: {}'.format(train_file))
        default_mask = _parse_mask_value(default_mask)
        mask_values = [list(default_mask) for _ in range(sample_count)]

    if 'mask_id' in csv_data.columns:
        mask_ids = [int(mask_id) for mask_id in csv_data['mask_id'].tolist()]
    else:
        mask_ids = [_infer_mask_id(mask_value) for mask_value in mask_values]

    modal_num = np.zeros(4, dtype=np.float32)
    mask_id_count = [0] * len(masks_test)
    for mask_value, mask_id in zip(mask_values, mask_ids):
        modal_num += np.array(mask_value, dtype=np.float32)
        mask_id_count[mask_id] += 1

    return {
        'source': train_file,
        'sample_count': sample_count,
        'modal_num': modal_num,
        'mask_id_count': mask_id_count,
    }


def merge_stream_summaries(*stream_summaries):
    merged_sources = []
    merged_sample_count = 0
    merged_modal_num = np.zeros(4, dtype=np.float32)
    merged_mask_id_count = [0] * len(masks_test)
    for stream_summary in stream_summaries:
        if stream_summary is None:
            continue
        merged_sources.append(stream_summary['source'])
        merged_sample_count += stream_summary['sample_count']
        merged_modal_num += stream_summary['modal_num']
        for mask_id, count in enumerate(stream_summary['mask_id_count']):
            merged_mask_id_count[mask_id] += count
    return {
        'source': ' + '.join(merged_sources),
        'sample_count': merged_sample_count,
        'modal_num': merged_modal_num,
        'mask_id_count': merged_mask_id_count,
    }


def log_stream_summary(client_idx, stream_name, stream_summary):
    logging.info(
        'Client-{} {} stats from {} with samples={}, Mod.Flair-{:d}, Mod.T1c-{:d}, Mod.T1-{:d}, Mod.T2-{:d}, mask_id_count={}'.format(
            client_idx + 1,
            stream_name,
            stream_summary['source'],
            int(stream_summary['sample_count']),
            int(stream_summary['modal_num'][0]),
            int(stream_summary['modal_num'][1]),
            int(stream_summary['modal_num'][2]),
            int(stream_summary['modal_num'][3]),
            stream_summary['mask_id_count'],
        )
    )


def local_training(
    args,
    device_id,
    mask,
    dataloader,
    model,
    client_idx,
    round,
    cluster_centers,
    client_mask_id_proportions,
    anchor_dataloader=None,
    unlabeled_dataloader=None,
    teacher_model=None,
    global_anchor_bank=None,
    global_missing_proto_bank=None,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    print(f"Training on GPU{device_id}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fedmass_training_mode = bool(getattr(args, 'use_fedmass_training', False))
    lr_schedule = LR_Scheduler(args.lr, args.c_rounds)
    model.train()
    model = model.to(device)
    global_missing_proto_bank = normalize_missing_proto_bank(global_missing_proto_bank)
    if args.enable_pseudo_filtering:
        if teacher_model is None:
            # 临时实现：当外部没有持久化 teacher state 时，在本地训练开始时创建 EMA teacher。
            # 后续若实现 Module 3，可将 teacher 状态持久化到 client state 中。
            teacher_model = create_ema_teacher(model)
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
    start = time.time()
    

    # Set Optimizer for the local model update
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'adam':
        train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay':args.weight_decay}]
        optimizer = torch.optim.Adam(train_params,  betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
    elif args.optimizer == 'adamw':
        train_params = [{'params': model.parameters(), 'lr': args.lr, 'weight_decay':args.weight_decay}]
        optimizer = torch.optim.AdamW(train_params,  betas=(0.9, 0.999), eps=1e-08, amsgrad=True)
    
    # writer.add_scalar('lr_lc', step_lr, global_step=round)
    # logging.info('############# client_{} local training ############'.format(client_idx+1))

    iter_per_epoch = len(dataloader)
    train_iter = iter(dataloader)
    joint_anchor_batch = bool(fedmass_training_mode and args.enable_anchor_supervision)
    anchor_iter = None if joint_anchor_batch else (iter(anchor_dataloader) if anchor_dataloader is not None else None)
    unlabeled_iter = iter(unlabeled_dataloader) if unlabeled_dataloader is not None else None

    ### Supervised stream statistics
    supervised_train_file = getattr(args, 'supervised_train_file', args.train_file)[client_idx+1]
    supervised_stream_summary = summarize_split_stream(
        supervised_train_file,
        default_mask=masks_all if fedmass_training_mode else None,
    )
    log_stream_summary(
        client_idx,
        'FedMASS supervised labeled-full stream' if fedmass_training_mode else 'legacy supervised stream',
        supervised_stream_summary,
    )
    unlabeled_train_file = args.unlabeled_train_file.get(client_idx+1) if args.enable_pseudo_filtering else None
    if fedmass_training_mode and unlabeled_train_file is not None:
        unlabeled_stream_summary = summarize_split_stream(unlabeled_train_file)
        log_stream_summary(client_idx, 'FedMASS unlabeled missing stream', unlabeled_stream_summary)
    logging.info(
        'Client-{} local training path: {} (supervised_source={}, unlabeled_source={}, anchor_batch_reuse={})'.format(
            client_idx + 1,
            'FedMASS' if fedmass_training_mode else 'legacy FedAMM',
            supervised_train_file,
            unlabeled_train_file if unlabeled_train_file is not None else 'None',
            joint_anchor_batch,
        )
    )
    modal_num = torch.tensor(supervised_stream_summary['modal_num'], requires_grad=False, device=device).float()
    
    modal_weight = torch.tensor((1,1,1,1), requires_grad=False, device=device).float()
    modal_weight = (iter_per_epoch/modal_num).to(device=device).float()
    imb_beta = torch.tensor((1,1,1,1), requires_grad=False, device=device).float()
    eta = 0.01
    eta_ext = 1.5
    local_full_anchor_state = init_local_anchor_state(args.num_class) if args.enable_anchor_supervision else None
    local_unlabeled_proto_state = None
    local_unlabeled_proto_count = torch.zeros(args.num_class, dtype=torch.float32)
    local_missing_proto_state = None
    pseudo_active = False

    for epoch in range(args.local_ep):
        step_lr = lr_schedule(optimizer, round)
        #writer.add_scalar('lr', step_lr, global_step=(epoch+1))
        epoch_fuse_losses = torch.zeros(1).cpu().float()
        epoch_sep_losses = torch.zeros(1).cpu().float()
        epoch_prm_losses = torch.zeros(1).cpu().float()
        epoch_kl_losses = torch.zeros(1).cpu().float()
        epoch_proto_losses = torch.zeros(1).cpu().float()
        epoch_global_losses = torch.zeros(1).cpu().float()
        epoch_anchor_total_losses = torch.zeros(1).cpu().float()
        epoch_anchor_seg_losses = torch.zeros(1).cpu().float()
        epoch_anchor_sep_losses = torch.zeros(1).cpu().float()
        epoch_anchor_kd_losses = torch.zeros(1).cpu().float()
        epoch_anchor_proto_losses = torch.zeros(1).cpu().float()
        epoch_anchor_prm_losses = torch.zeros(1).cpu().float()
        epoch_pseudo_total_losses = torch.zeros(1).cpu().float()
        epoch_pseudo_ce_losses = torch.zeros(1).cpu().float()
        epoch_pseudo_dice_losses = torch.zeros(1).cpu().float()
        epoch_pseudo_anchor_losses = torch.zeros(1).cpu().float()
        epoch_pseudo_missing_proto_losses = torch.zeros(1).cpu().float()
        epoch_pseudo_missing_proto_active_pairs = torch.zeros(1).cpu().float()
        epoch_pseudo_mean_confidence = torch.zeros(1).cpu().float()
        epoch_pseudo_selected_ratio = torch.zeros(1).cpu().float()
        epoch_pseudo_anchor_agreement_ratio = torch.zeros(1).cpu().float()
        epoch_pseudo_consistency_ratio = torch.zeros(1).cpu().float()
        # epoch_dist_losses = torch.zeros(1).cpu().float()
        epoch_losses = torch.zeros(1).cpu().float()
        epoch_sep_m = torch.zeros(4).cpu().float()
        epoch_kl_m = torch.zeros(4).cpu().float()
        epoch_proto_m = torch.zeros(4).cpu().float()
        epoch_dist_m = torch.zeros(4).cpu().float()

        b = time.time()
        client_gt = []
        for i in range(iter_per_epoch):
            step = (i+1) + epoch*iter_per_epoch
            ###Data load
            try:
                data = next(train_iter)
            except:
                train_iter = iter(dataloader)
                data = next(train_iter)
            x, target, mask, name = data
            x = x.to(device=device, non_blocking=True)
            target = target.to(device=device, non_blocking=True)
            mask_id = masks_test.index(mask[0].tolist())
            mask = mask.to(device=device, non_blocking=True)

            model.is_training = True

            kl_loss_m = torch.zeros(4, device=device).float()
            sep_loss_m = torch.zeros(4, device=device).float()
            proto_loss_m = torch.zeros(4, device=device).float()
            dist_m = torch.zeros(4, device=device).float()
            prm_loss = torch.zeros(1, device=device).float()
            # fuse_loss = torch.zeros(1).cuda().float()
            rp_iter = torch.zeros(4, device=device).float()

            cluster_mask = tuple(mask.cpu().numpy().flatten().tolist())
            fuse_pred, prm_loss_bs, sep_loss_m_bs, kl_loss_m_bs, proto_loss_m_bs, dist_m_bs,gt = model(x, mask, target=target, temp=args.temp)
            
            client_gt.append((cluster_mask,gt.detach().cpu()))
            
            global_loss = torch.zeros(1, device=device).float()
            if cluster_centers and cluster_centers[cluster_mask] is not None:
                cluster_gt = cluster_centers[cluster_mask]
                cluster_gt = cluster_gt.to(device=device).float()
                global_loss = torch.mean((cluster_gt-gt)**2, dim=1)
                non_zero_mask = torch.any(gt != 0, dim=1)  # 找出第一个维度上不全为 0 的行
                global_loss = global_loss[non_zero_mask].sum()
                global_loss = global_loss * client_mask_id_proportions[mask_id]
                logging.info('global_loss:{:.4f}'.format(global_loss.item()))
            
            ###Loss compute
            # fuse_cross_loss = criterions.softmax_weighted_loss(fuse_pred, target, num_cls=num_cls)
            # fuse_dice_loss = criterions.dice_loss(fuse_pred, target, num_cls=num_cls)
            # fuse_loss += fuse_cross_loss + fuse_dice_loss
            fuse_loss_bs = criterions.softmax_weighted_loss_bs(fuse_pred, target, num_cls=args.num_class) + criterions.dice_loss_bs(fuse_pred, target, num_cls=args.num_class)
            fuse_loss = torch.sum(fuse_loss_bs)
            prm_loss = torch.sum(prm_loss_bs)
            
            # masks_sum = torch.clamp(torch.sum(mask, dim=0).float(), min=0.005, max=args.batch_size)
            sep_loss_m = torch.sum(sep_loss_m_bs*mask, dim=0)
            kl_loss_m = torch.sum(kl_loss_m_bs*mask, dim=0)
            proto_loss_m = torch.sum(proto_loss_m_bs*mask, dim=0)
            dist_m = torch.sum(dist_m_bs*mask, dim=0)

            for bs in range(x.size(0)):
                dist_avg_bs = sum(dist_m_bs[bs])/sum(mask[bs])
                rp_iter += mask[bs]*(dist_m_bs[bs]/dist_avg_bs-1)
            rp_mask = rp_iter > 0

            kl_loss = (imb_beta * modal_weight * kl_loss_m).sum()
            proto_loss = (rp_mask * modal_weight * proto_loss_m).sum()
            # dist_loss = (rp_mask * imb_beta * modal_weight * dist_m).sum()

            ## warmup with shared sep-decoder like rfnet
            if round < args.region_fusion_start_epoch:
                sep_loss = (imb_beta * modal_weight * sep_loss_m).sum()
                loss = fuse_loss * 0.0 + sep_loss + prm_loss * 0.0 + kl_loss * 0.0 + proto_loss * 0.0
            else:
                sep_loss = (rp_mask * imb_beta * modal_weight * sep_loss_m).sum()
                loss = fuse_loss + sep_loss + prm_loss + kl_loss * 0.5 + proto_loss * 0.1 + global_loss

            # ## without warmup and without shared sep-decoder
            # sep_loss = (rp_mask * imb_beta * modal_weight * sep_loss_m).sum()
            # loss = fuse_loss + sep_loss * 0.0 + prm_loss + kl_loss * 0.5 + proto_loss * 0.1

            anchor_metrics = None
            if args.enable_anchor_supervision and round >= args.anchor_warmup_rounds:
                anchor_x, anchor_target, anchor_name = None, None, None
                if joint_anchor_batch:
                    anchor_x, anchor_target, anchor_name = x, target, name
                elif anchor_iter is not None:
                    try:
                        anchor_data = next(anchor_iter)
                    except StopIteration:
                        anchor_iter = iter(anchor_dataloader)
                        anchor_data = next(anchor_iter)
                    anchor_x, anchor_target, _, anchor_name = anchor_data
                    anchor_x = anchor_x.to(device=device, non_blocking=True)
                    anchor_target = anchor_target.to(device=device, non_blocking=True)
                if anchor_x is not None:
                    anchor_metrics = compute_anchor_supervision(args, model, anchor_x, anchor_target)
                    loss = loss + anchor_metrics['loss']
                    local_full_anchor_state = update_local_anchor_state(
                        local_full_anchor_state,
                        anchor_metrics['anchors'],
                        anchor_metrics['anchor_valid_mask'],
                    )

            pseudo_metrics = None
            if (
                args.enable_pseudo_filtering
                and unlabeled_iter is not None
                and round >= args.pseudo_warmup_rounds
            ):
                try:
                    unlabeled_data = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_dataloader)
                    unlabeled_data = next(unlabeled_iter)
                weak_x, strong_x, pseudo_mask, pseudo_mask_id, pseudo_name = unlabeled_data
                weak_x = weak_x.to(device=device, non_blocking=True)
                strong_x = strong_x.to(device=device, non_blocking=True)
                pseudo_mask = pseudo_mask.to(device=device, non_blocking=True)

                pseudo_metrics = compute_pseudo_filtering(
                    student_model=model,
                    teacher_model=teacher_model,
                    weak_batch=weak_x,
                    strong_batch=strong_x,
                    mask=pseudo_mask,
                    pseudo_mask_id=pseudo_mask_id,
                    global_anchor_bank=global_anchor_bank,
                    args=args,
                    num_classes=args.num_class,
                    num_patterns=len(masks_test),
                )
                pseudo_active = True
                missing_proto_global_loss = loss.new_zeros(())
                missing_proto_active_count = 0
                if global_missing_proto_bank is not None:
                    missing_proto_global_loss, missing_proto_active_count = compute_missing_pattern_alignment_loss(
                        pseudo_metrics.get('missing_pattern_prototypes'),
                        global_missing_proto_bank,
                    )
                    if missing_proto_global_loss is None:
                        missing_proto_global_loss = loss.new_zeros(())
                        missing_proto_active_count = 0
                    if missing_proto_active_count > 0:
                        global_loss = global_loss + missing_proto_global_loss
                        loss = loss + missing_proto_global_loss
                pseudo_metrics['loss_dict']['missing_proto_loss'] = missing_proto_global_loss.detach()
                pseudo_metrics['stats']['missing_proto_active_count'] = torch.tensor(
                    float(missing_proto_active_count),
                    device=loss.device,
                )
                loss = loss + args.pseudo_loss_weight * pseudo_metrics['total_loss']
                if torch.any(pseudo_metrics['valid_classes']):
                    if local_unlabeled_proto_state is None:
                        local_unlabeled_proto_state = torch.zeros_like(pseudo_metrics['unlabeled_prototypes'].detach().cpu())
                    proto_cpu = pseudo_metrics['unlabeled_prototypes'].detach().cpu()
                    valid_cpu = pseudo_metrics['valid_classes'].detach().cpu()
                    local_unlabeled_proto_state[valid_cpu] += proto_cpu[valid_cpu]
                    local_unlabeled_proto_count[valid_cpu] += 1
                if pseudo_metrics.get('missing_pattern_prototypes') is not None:
                    if local_missing_proto_state is None:
                        local_missing_proto_state = init_local_missing_proto_state(
                            num_patterns=len(masks_test),
                            num_cls=args.num_class,
                        )
                    local_missing_proto_state = update_local_missing_proto_state(
                        local_missing_proto_state,
                        pseudo_metrics['missing_pattern_prototypes'],
                    )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if args.enable_pseudo_filtering and teacher_model is not None:
                teacher_model = update_ema_teacher(model, teacher_model, args.ema_decay)

            epoch_losses += (loss/iter_per_epoch).detach().cpu()
            epoch_fuse_losses += (fuse_loss/iter_per_epoch).detach().cpu()
            epoch_prm_losses += (prm_loss/iter_per_epoch).detach().cpu()
            epoch_sep_losses += (sep_loss/iter_per_epoch).detach().cpu()
            epoch_kl_losses += (kl_loss/iter_per_epoch).detach().cpu()
            epoch_proto_losses += (proto_loss/iter_per_epoch).detach().cpu()
            # epoch_dist_losses += (dist_loss/iter_per_epoch).detach().cpu()
            epoch_global_losses += (global_loss/iter_per_epoch).detach().cpu()
            if anchor_metrics is not None:
                epoch_anchor_total_losses += (anchor_metrics['loss']/iter_per_epoch).detach().cpu()
                epoch_anchor_seg_losses += (anchor_metrics['seg_loss']/iter_per_epoch).detach().cpu()
                epoch_anchor_sep_losses += (anchor_metrics['sep_loss']/iter_per_epoch).detach().cpu()
                epoch_anchor_kd_losses += (anchor_metrics['kd_loss']/iter_per_epoch).detach().cpu()
                epoch_anchor_proto_losses += (anchor_metrics['proto_loss']/iter_per_epoch).detach().cpu()
                epoch_anchor_prm_losses += (anchor_metrics['prm_loss']/iter_per_epoch).detach().cpu()
            if pseudo_metrics is not None:
                epoch_pseudo_total_losses += (pseudo_metrics['loss_dict']['total_loss']/iter_per_epoch).detach().cpu()
                epoch_pseudo_ce_losses += (pseudo_metrics['loss_dict']['ce_loss']/iter_per_epoch).detach().cpu()
                epoch_pseudo_dice_losses += (pseudo_metrics['loss_dict']['dice_loss']/iter_per_epoch).detach().cpu()
                epoch_pseudo_anchor_losses += (pseudo_metrics['loss_dict']['anchor_loss']/iter_per_epoch).detach().cpu()
                epoch_pseudo_missing_proto_losses += (pseudo_metrics['loss_dict']['missing_proto_loss']/iter_per_epoch).detach().cpu()
                epoch_pseudo_missing_proto_active_pairs += pseudo_metrics['stats']['missing_proto_active_count'].detach().cpu()
                epoch_pseudo_mean_confidence += (pseudo_metrics['stats']['mean_confidence']/iter_per_epoch).detach().cpu()
                epoch_pseudo_selected_ratio += (pseudo_metrics['stats']['selected_ratio']/iter_per_epoch).detach().cpu()
                epoch_pseudo_anchor_agreement_ratio += (pseudo_metrics['stats']['anchor_agreement_ratio']/iter_per_epoch).detach().cpu()
                epoch_pseudo_consistency_ratio += (pseudo_metrics['stats']['consistency_ratio']/iter_per_epoch).detach().cpu()

            if args.mask_type == 'idt':
                epoch_kl_m += (kl_loss_m/modal_num).detach().cpu()
                epoch_sep_m += (sep_loss_m/modal_num).detach().cpu()
                epoch_proto_m += (proto_loss_m/modal_num).detach().cpu()
                epoch_dist_m += (dist_m/modal_num).detach().cpu()
            else:
                epoch_kl_m += (kl_loss_m/iter_per_epoch).detach().cpu()
                epoch_sep_m += (sep_loss_m/iter_per_epoch).detach().cpu()
                epoch_proto_m += (proto_loss_m/iter_per_epoch).detach().cpu()
                epoch_dist_m += (dist_m/iter_per_epoch).detach().cpu()

            msg = 'Epoch {}/{}, Iter {}/{}, Loss {:.4f}, '.format((epoch+1), args.local_ep, (i+1), iter_per_epoch, loss.item())
            msg += 'fuse_loss:{:.4f}, prm_loss:{:.4f}, '.format(fuse_loss.item(), prm_loss.item())
            msg += 'sep_loss:{:.4f}, '.format(sep_loss.item())
            msg += 'kl_loss:{:.4f}, proto_loss:{:.4f},'.format(kl_loss.item(), proto_loss.item())
            msg += 'seplist:[{:.4f},{:.4f},{:.4f},{:.4f}] '.format(sep_loss_m[0].item(), sep_loss_m[1].item(), sep_loss_m[2].item(), sep_loss_m[3].item())
            msg += 'kllist:[{:.4f},{:.4f},{:.4f},{:.4f}] '.format(kl_loss_m[0].item(), kl_loss_m[1].item(), kl_loss_m[2].item(), kl_loss_m[3].item())
            msg += 'protolist:[{:.4f},{:.4f},{:.4f},{:.4f}] '.format(proto_loss_m[0].item(), proto_loss_m[1].item(), proto_loss_m[2].item(), proto_loss_m[3].item())
            msg += 'distlist:[{:.4f},{:.4f},{:.4f},{:.4f}] '.format(dist_m[0].item(), dist_m[1].item(), dist_m[2].item(), dist_m[3].item())
            for bs_n in range(x.size(0)):
                msg += '{:>20}, '.format(name[bs_n])
            msg += 'kl_w[{:.2f},{:.2f},{:.2f},{:.2f}] '.format(modal_weight[0].item(), modal_weight[1].item(), modal_weight[2].item(), modal_weight[3].item())
            if anchor_metrics is not None and ((i + 1) % max(args.anchor_log_interval, 1) == 0 or i == 0):
                msg += 'anchor_total:{:.4f}, anchor_seg:{:.4f}, anchor_kd:{:.4f}, anchor_proto:{:.4f}, '.format(
                    anchor_metrics['loss'].item(),
                    anchor_metrics['seg_loss'].item(),
                    anchor_metrics['kd_loss'].item(),
                    anchor_metrics['proto_loss'].item(),
                )
                for anchor_bs_n in range(anchor_x.size(0)):
                    msg += 'anchor_{:>20}, '.format(anchor_name[anchor_bs_n])
            if pseudo_metrics is not None and ((i + 1) % max(args.pseudo_log_interval, 1) == 0 or i == 0):
                msg += 'pseudo_total:{:.4f}, pseudo_ce:{:.4f}, pseudo_dice:{:.4f}, pseudo_anchor:{:.4f}, pseudo_miss_align:{:.4f}, pseudo_miss_pairs:{:d}, '.format(
                    pseudo_metrics['loss_dict']['total_loss'].item(),
                    pseudo_metrics['loss_dict']['ce_loss'].item(),
                    pseudo_metrics['loss_dict']['dice_loss'].item(),
                    pseudo_metrics['loss_dict']['anchor_loss'].item(),
                    pseudo_metrics['loss_dict']['missing_proto_loss'].item(),
                    int(pseudo_metrics['stats']['missing_proto_active_count'].item()),
                )
                msg += 'pseudo_conf:{:.4f}, pseudo_sel:{:.4f}, pseudo_anchor_agree:{:.4f}, pseudo_cons:{:.4f}, '.format(
                    pseudo_metrics['stats']['mean_confidence'].item(),
                    pseudo_metrics['stats']['selected_ratio'].item(),
                    pseudo_metrics['stats']['anchor_agreement_ratio'].item(),
                    pseudo_metrics['stats']['consistency_ratio'].item(),
                )
            logging.info(msg)
        b_train = time.time()
        logging.info('train time per epoch: {}'.format(b_train - b))

        epoch_dist_avg = (sum(epoch_dist_m)/4.0).cpu().float()
        rp_epoch = ((epoch_dist_avg - epoch_dist_m) / epoch_dist_avg)
        if round < args.region_fusion_start_epoch:
            imb_beta = imb_beta.to(device=device)
        else:
            if round % 100 == 0:
                eta = eta * eta_ext
            imb_beta = imb_beta.cpu() - eta * rp_epoch
            imb_beta = torch.clamp(imb_beta, min=0.1, max=4.0)
            imb_beta = 2 * imb_beta / (sum(imb_beta**2)**(0.5))
            imb_beta = imb_beta.to(device=device)


        logging.info('epoch_global_losses:{:.4f}'.format(epoch_global_losses.item()))
        if args.enable_anchor_supervision and local_full_anchor_state is not None:
            logging.info('epoch_anchor_total_losses:{:.4f}'.format(epoch_anchor_total_losses.item()))
        if args.enable_pseudo_filtering:
            logging.info('epoch_pseudo_total_losses:{:.4f}'.format(epoch_pseudo_total_losses.item()))
            logging.info(
                'epoch_pseudo_missing_proto_losses:{:.4f}, epoch_pseudo_missing_proto_active_pairs:{:.0f}'.format(
                    epoch_pseudo_missing_proto_losses.item(),
                    epoch_pseudo_missing_proto_active_pairs.item(),
                )
            )

        logging.info('rp_epoch:[{:.4f},{:.4f},{:.4f},{:.4f}]'.format(rp_epoch[0].item(), rp_epoch[1].item(), rp_epoch[2].item(), rp_epoch[3].item()))
        logging.info('imb_beta:[{:.4f},{:.4f},{:.4f},{:.4f}]'.format(imb_beta[0].item(), imb_beta[1].item(), imb_beta[2].item(), imb_beta[3].item()))

        epoch_loss = {'epoch_losses':epoch_losses, 'epoch_fuse_losses':epoch_fuse_losses, 'epoch_prm_losses':epoch_prm_losses, 'epoch_sep_losses':epoch_sep_losses,
                        'epoch_kl_losses':epoch_kl_losses, 'epoch_proto_losses':epoch_proto_losses, 'epoch_kl_m':epoch_kl_m, 'epoch_sep_m':epoch_sep_m,
                        'epoch_proto_m':epoch_proto_m, 'epoch_dist_m':epoch_dist_m, 'rp_epoch':rp_epoch, 'epoch_global_losses':epoch_global_losses,
                        'epoch_anchor_total_losses': epoch_anchor_total_losses, 'epoch_anchor_seg_losses': epoch_anchor_seg_losses,
                        'epoch_anchor_sep_losses': epoch_anchor_sep_losses, 'epoch_anchor_kd_losses': epoch_anchor_kd_losses,
                        'epoch_anchor_proto_losses': epoch_anchor_proto_losses, 'epoch_anchor_prm_losses': epoch_anchor_prm_losses,
                        'epoch_pseudo_total_losses': epoch_pseudo_total_losses, 'epoch_pseudo_ce_losses': epoch_pseudo_ce_losses,
                        'epoch_pseudo_dice_losses': epoch_pseudo_dice_losses, 'epoch_pseudo_anchor_losses': epoch_pseudo_anchor_losses,
                        'epoch_pseudo_missing_proto_losses': epoch_pseudo_missing_proto_losses,
                        'epoch_pseudo_missing_proto_active_pairs': epoch_pseudo_missing_proto_active_pairs,
                        'epoch_pseudo_mean_confidence': epoch_pseudo_mean_confidence, 'epoch_pseudo_selected_ratio': epoch_pseudo_selected_ratio,
                        'epoch_pseudo_anchor_agreement_ratio': epoch_pseudo_anchor_agreement_ratio, 'epoch_pseudo_consistency_ratio': epoch_pseudo_consistency_ratio, "lr":step_lr}
    
    msg = 'client_{} local training total time: {:.4f} hours'.format(client_idx+1, (time.time() - start)/3600)
    print(msg)
    logging.info(msg)
    model = model.cpu()
    teacher_state = None
    if teacher_model is not None:
        teacher_state = copy.deepcopy(teacher_model.cpu().state_dict())

    local_full_anchors = finalize_local_anchor_state(local_full_anchor_state)
    local_unlabeled_prototypes = None
    if local_unlabeled_proto_state is not None:
        local_unlabeled_prototypes = local_unlabeled_proto_state.clone()
        counts = local_unlabeled_proto_count.unsqueeze(1).clamp_min(1.0)
        local_unlabeled_prototypes = local_unlabeled_prototypes / counts
        zero_mask = local_unlabeled_proto_count == 0
        if torch.any(zero_mask):
            local_unlabeled_prototypes[zero_mask] = 0
    local_missing_pattern_prototypes = finalize_local_missing_proto_state(local_missing_proto_state)

    extra_outputs = {
        'local_full_anchors': local_full_anchors,
        'local_unlabeled_prototypes': local_unlabeled_prototypes,
        'local_missing_pattern_prototypes': local_missing_pattern_prototypes,
        'teacher_state': teacher_state,
        'pseudo_active': pseudo_active,
        'pseudo_stats': {
            'mean_confidence': epoch_loss['epoch_pseudo_mean_confidence'],
            'selected_ratio': epoch_loss['epoch_pseudo_selected_ratio'],
            'anchor_agreement_ratio': epoch_loss['epoch_pseudo_anchor_agreement_ratio'],
            'consistency_ratio': epoch_loss['epoch_pseudo_consistency_ratio'],
        },
    }
    return model,epoch_loss,client_gt,extra_outputs


def log_client_train(writer, client_i, local_losses, round):
    
    writer.add_scalar('Client_{}/epoch_losses'.format(client_i+1), local_losses['epoch_losses'].item(), global_step=round)
    writer.add_scalar('Client_{}/epoch_fuse_losses'.format(client_i+1), local_losses['epoch_fuse_losses'].item(), global_step=round)
    writer.add_scalar('Client_{}/epoch_prm_losses'.format(client_i+1), local_losses['epoch_prm_losses'].item(), global_step=round)
    writer.add_scalar('Client_{}/epoch_sep_losses'.format(client_i+1), local_losses['epoch_sep_losses'].item(), global_step=round)
    writer.add_scalar('Client_{}/epoch_kl_losses'.format(client_i+1), local_losses['epoch_kl_losses'].item(), global_step=round)
    writer.add_scalar('Client_{}/epoch_proto_losses'.format(client_i+1), local_losses['epoch_proto_losses'].item(), global_step=round)
    writer.add_scalar('Client_{}/epoch_global_losses'.format(client_i+1), local_losses['epoch_global_losses'].item(), global_step=round)
    writer.add_scalar('Client_{}/epoch_anchor_total_losses'.format(client_i+1), local_losses['epoch_anchor_total_losses'].item(), global_step=round)
    writer.add_scalar('Client_{}/epoch_anchor_seg_losses'.format(client_i+1), local_losses['epoch_anchor_seg_losses'].item(), global_step=round)
    writer.add_scalar('Client_{}/epoch_anchor_sep_losses'.format(client_i+1), local_losses['epoch_anchor_sep_losses'].item(), global_step=round)
    writer.add_scalar('Client_{}/epoch_anchor_kd_losses'.format(client_i+1), local_losses['epoch_anchor_kd_losses'].item(), global_step=round)
    writer.add_scalar('Client_{}/epoch_anchor_proto_losses'.format(client_i+1), local_losses['epoch_anchor_proto_losses'].item(), global_step=round)
    writer.add_scalar('Client_{}/epoch_anchor_prm_losses'.format(client_i+1), local_losses['epoch_anchor_prm_losses'].item(), global_step=round)
    writer.add_scalar('Client_{}/pseudo_total_loss'.format(client_i+1), local_losses['epoch_pseudo_total_losses'].item(), global_step=round)
    writer.add_scalar('Client_{}/pseudo_ce_loss'.format(client_i+1), local_losses['epoch_pseudo_ce_losses'].item(), global_step=round)
    writer.add_scalar('Client_{}/pseudo_dice_loss'.format(client_i+1), local_losses['epoch_pseudo_dice_losses'].item(), global_step=round)
    writer.add_scalar('Client_{}/pseudo_anchor_loss'.format(client_i+1), local_losses['epoch_pseudo_anchor_losses'].item(), global_step=round)
    writer.add_scalar('Client_{}/pseudo_missing_proto_loss'.format(client_i+1), local_losses['epoch_pseudo_missing_proto_losses'].item(), global_step=round)
    writer.add_scalar('Client_{}/pseudo_missing_proto_active_pairs'.format(client_i+1), local_losses['epoch_pseudo_missing_proto_active_pairs'].item(), global_step=round)
    writer.add_scalar('Client_{}/pseudo_mean_confidence'.format(client_i+1), local_losses['epoch_pseudo_mean_confidence'].item(), global_step=round)
    writer.add_scalar('Client_{}/pseudo_selected_ratio'.format(client_i+1), local_losses['epoch_pseudo_selected_ratio'].item(), global_step=round)
    writer.add_scalar('Client_{}/pseudo_anchor_agreement_ratio'.format(client_i+1), local_losses['epoch_pseudo_anchor_agreement_ratio'].item(), global_step=round)
    writer.add_scalar('Client_{}/pseudo_consistency_ratio'.format(client_i+1), local_losses['epoch_pseudo_consistency_ratio'].item(), global_step=round)
    writer.add_scalar('Client_{}/lr'.format(client_i+1), local_losses['lr'].item(), global_step=round)
    for m in range(4):
        writer.add_scalar('Client_{}/kl_m{}'.format(client_i+1, m), local_losses['epoch_kl_m'][m].item(), global_step=round)
        writer.add_scalar('Client_{}/sep_m{}'.format(client_i+1, m), local_losses['epoch_sep_m'][m].item(), global_step=round)
        writer.add_scalar('Client_{}/proto_m{}'.format(client_i+1, m), local_losses['epoch_proto_m'][m].item(), global_step=round)
        writer.add_scalar('Client_{}/dist_m{}'.format(client_i+1, m), local_losses['epoch_dist_m'][m].item(), global_step=round)
        writer.add_scalar('Client_{}/rp_m{}'.format(client_i+1, m), local_losses['rp_epoch'][m].item(), global_step=round)


def _format_optional_metric(value):
    if value is None:
        return 'N/A'
    return '{:.4f}'.format(float(value))


def uploadLCweightsandGLBupdate(server_model,local_weights,client_mask_proportions_sum,client_modal_weight,model_clients,aggregation_weights=None):
    if aggregation_weights is None:
        client_count = len(local_weights)
        model_weights = [1.0 / float(client_count) for _ in range(client_count)]
        encoder_weights = {
            'flair': client_modal_weight.T[0].tolist(),
            't1ce': client_modal_weight.T[1].tolist(),
            't1': client_modal_weight.T[2].tolist(),
            't2': client_modal_weight.T[3].tolist(),
        }
    else:
        model_weights = aggregation_weights['model_weights']
        encoder_weights = aggregation_weights['encoder_weights']

    glb_w = avg_local_weights(local_weights[0], local_weights[1], local_weights[2], local_weights[3],model_weights)
    server_model.load_state_dict(glb_w)
    flair_encoder = avg_encoder_weights(model_clients[0].flair_encoder.state_dict(), model_clients[1].flair_encoder.state_dict(), model_clients[2].flair_encoder.state_dict(), model_clients[3].flair_encoder.state_dict(),encoder_weights['flair'])
    t1ce_encoder = avg_encoder_weights(model_clients[0].t1ce_encoder.state_dict(), model_clients[1].t1ce_encoder.state_dict(), model_clients[2].t1ce_encoder.state_dict(), model_clients[3].t1ce_encoder.state_dict(),encoder_weights['t1ce'])
    t1_encoder = avg_encoder_weights(model_clients[0].t1_encoder.state_dict(), model_clients[1].t1_encoder.state_dict(), model_clients[2].t1_encoder.state_dict(), model_clients[3].t1_encoder.state_dict(),encoder_weights['t1'])
    t2_encoder = avg_encoder_weights(model_clients[0].t2_encoder.state_dict(), model_clients[1].t2_encoder.state_dict(), model_clients[2].t2_encoder.state_dict(), model_clients[3].t2_encoder.state_dict(),encoder_weights['t2'])
    server_model.flair_encoder.load_state_dict(flair_encoder)
    server_model.t1ce_encoder.load_state_dict(t1ce_encoder)
    server_model.t1_encoder.load_state_dict(t1_encoder)
    server_model.t2_encoder.load_state_dict(t2_encoder)
    return server_model

def downloadGLBweights(server_model, model_clients):
    for i in range(len(model_clients)):
        model_clients[i].load_state_dict(server_model.state_dict())
    return model_clients

if __name__ == '__main__':

    args = args_parser()
    
    args.train_transforms = 'Compose([RandCrop3D((80,80,80)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
    args.test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'
    
    timestamp = datetime.now().strftime("%m%d%H%M")
    args.save_path = args.save_root + '/' + str(args.version)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    args.modelfile_path = os.path.join(args.save_path, 'model_files')
    if not os.path.exists(args.modelfile_path):
        os.makedirs(args.modelfile_path)
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        filename=args.save_path + '/fl_log.txt')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logging.getLogger('').addHandler(console)
    
    writer = SummaryWriter(os.path.join(args.save_path, 'TBlog'))
    
    ########## setting seed for deterministic
    if args.deterministic:
        # cudnn.enabled = False
        # cudnn.benchmark = False
        # cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    ########## setting device and gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.device_ids = list(map(int,args.device_ids.split(',')))
    args.local_devices = args.device_ids

    ########## setting global and local model
    server_model = rfnet.Model(num_cls=args.num_class)
    server_model.mask_type = args.mask_type
    best_dices = [0.0, 0.0, 0.0, 0.0]
    best_dice = 0.0
    cluster_centers = None
    global_anchor_bank = None
    global_missing_proto_bank = None
    reliability_state = init_reliability_state(args.client_num)
    if args.reload_from_checkpoint:
        ckpt = torch.load(args.checkpoint_path + '/last.pth', map_location='cpu')
        server_model.load_state_dict(ckpt["server"])
        args.start_round = ckpt['round']
        best_dice = ckpt['best_dice']
        best_dices = ckpt['best_dices']
        cluster_centers = ckpt['cluster_centers']
        reliability_state = restore_reliability_state(ckpt.get('reliability_state'), args.client_num)
        logging.info(
            'Reload reliability state from round {} with scores {}'.format(
                reliability_state.get('last_round', -1),
                reliability_state.get('last_normalized_scores'),
            )
        )
        if args.enable_anchor_supervision:
            global_anchor_bank = normalize_anchor_bank(ckpt.get('global_anchor_bank'))
            if global_anchor_bank is not None:
                logging.info(
                    'Reload global anchor bank with {} valid classes'.format(
                        int(global_anchor_bank['valid_mask'].sum().item())
                    )
                )
        if args.enable_pseudo_filtering:
            global_missing_proto_bank = normalize_missing_proto_bank(ckpt.get('global_missing_proto_bank'))
            if global_missing_proto_bank is not None:
                logging.info(
                    'Reload global missing-pattern prototype bank with {} valid pattern-class entries'.format(
                        count_valid_pattern_classes(global_missing_proto_bank)
                    )
                )
        print("load best result: {}, {}, {}, {}.".format(best_dice, best_dices[0], best_dices[1], best_dices[2]))


    lr_schedule = LR_Scheduler(args.lr, args.c_rounds)
    ########## FL setting ##########
    # define dataset, model, optimizer for each clients 
    dataloader_clients, anchorloader_clients, unlabeledloader_clients, validloader_clients, testloader_clients = [], [], [], [], []
    model_clients = []
    teacher_clients = []
    optimizer_clients = []
    client_counts, client_weights = [], []     ### FedAvg Setting
    modal_list = ['flair', 't1ce', 't1', 't2']
    logging.info(str(args))
    fedmass_training_mode = bool(getattr(args, 'use_fedmass_training', False))
    logging.info(
        'FedMASS split supervision mode: {} (split_dir={}, anchor_enabled={}, pseudo_enabled={})'.format(
            fedmass_training_mode,
            getattr(args, 'fedmass_split_dir', None),
            args.enable_anchor_supervision,
            args.enable_pseudo_filtering,
        )
    )
    client_modal_weight = []
    mask_id_count_list = []
    fedmass_client_modal_weight = []
    fedmass_supervised_mask_id_count_list = []
    client_mask_id_proportions = []
    for client_idx in range(args.client_num):
        lc_train_file = args.train_file[client_idx+1]
        supervised_train_file = getattr(args, 'supervised_train_file', args.train_file)[client_idx+1]
        if fedmass_training_mode:
            data_set = Brats_loadall_labeled_full_nii(
                transforms=args.train_transforms,
                root=args.datapath,
                num_cls=args.num_class,
                train_file=supervised_train_file,
            )
            logging.info(
                'Client-{} : FedMASS main supervised labeled-full dataset length {} (source: {})'.format(
                    client_idx+1, len(data_set), supervised_train_file
                )
            )
        else:
            data_set = Brats_loadall_train_nii_idt(
                transforms=args.train_transforms,
                root=args.datapath,
                num_cls=args.num_class,
                train_file=lc_train_file,
            )
            logging.info(
                'Client-{} : legacy main supervised dataset length {} (source: {})'.format(
                    client_idx+1, len(data_set), lc_train_file
                )
            )
        data_loader = DataLoader(dataset=data_set, batch_size=args.batch_size,
                                pin_memory=True, shuffle=True, worker_init_fn=init_fn)
        dataloader_clients.append(data_loader)
        if args.enable_anchor_supervision:
            if fedmass_training_mode:
                anchorloader_clients.append(None)
                logging.info(
                    'Client-{} : Module1 anchor supervision reuses the FedMASS labeled-full supervised batch'.format(
                        client_idx+1
                    )
                )
            else:
                anchor_train_file = args.anchor_train_file.get(client_idx+1)
                if anchor_train_file is None:
                    anchor_train_file = lc_train_file
                anchor_batch_size = args.anchor_batch_size if args.anchor_batch_size > 0 else args.batch_size
                anchor_set = Brats_loadall_labeled_full_nii(
                    transforms=args.train_transforms,
                    root=args.datapath,
                    num_cls=args.num_class,
                    train_file=anchor_train_file,
                )
                anchor_loader = DataLoader(
                    dataset=anchor_set,
                    batch_size=anchor_batch_size,
                    pin_memory=True,
                    shuffle=True,
                    worker_init_fn=init_fn,
                )
                anchorloader_clients.append(anchor_loader)
                logging.info(
                    'Client-{} : legacy Module1 extra labeled dataset length {} (source: {})'.format(
                        client_idx+1, len(anchor_set), anchor_train_file
                    )
                )
        else:
            anchorloader_clients.append(None)
        net = copy.deepcopy(server_model)   # .to(device)  # .to(args.device)
        model_clients.append(net)
        unlabeled_train_file = None
        if args.enable_pseudo_filtering:
            unlabeled_train_file = args.unlabeled_train_file.get(client_idx+1)
            if unlabeled_train_file is None:
                unlabeled_train_file = lc_train_file
            unlabeled_set = Brats_loadall_unlabeled_missing_nii(
                weak_transforms=args.test_transforms,
                strong_transforms=args.train_transforms,
                root=args.datapath,
                train_file=unlabeled_train_file,
                mask_type=args.mask_type,
            )
            unlabeled_loader = DataLoader(
                dataset=unlabeled_set,
                batch_size=args.batch_size,
                pin_memory=True,
                shuffle=True,
                worker_init_fn=init_fn,
            )
            unlabeledloader_clients.append(unlabeled_loader)
            teacher_clients.append(create_ema_teacher(net))
            logging.info(
                'Client-{} : {} unlabeled dataset length {} (source: {})'.format(
                    client_idx+1,
                    'FedMASS missing-modal' if fedmass_training_mode else 'Module2',
                    len(unlabeled_set),
                    unlabeled_train_file,
                )
            )
        else:
            unlabeledloader_clients.append(None)
            teacher_clients.append(None)
        if fedmass_training_mode:
            supervised_stream_summary = summarize_split_stream(
                supervised_train_file,
                default_mask=masks_all,
            )
            unlabeled_stream_summary = summarize_split_stream(unlabeled_train_file) if unlabeled_train_file is not None else None
            combined_train_summary = merge_stream_summaries(supervised_stream_summary, unlabeled_stream_summary)
            log_stream_summary(client_idx, 'FedMASS supervised labeled-full stream', supervised_stream_summary)
            if unlabeled_stream_summary is not None:
                log_stream_summary(client_idx, 'FedMASS unlabeled missing stream', unlabeled_stream_summary)
            log_stream_summary(client_idx, 'FedMASS combined train stream for aggregation', combined_train_summary)
            fedmass_client_modal_weight.append(combined_train_summary['modal_num'])
            fedmass_supervised_mask_id_count_list.append(supervised_stream_summary['mask_id_count'])
    ####15种mask的分布
        imb_mr_csv_data = None if fedmass_training_mode else pd.read_csv(args.train_file[client_idx+1])

        clinet_modal_num = np.zeros(4, dtype=np.float32)  # 初始化为零的NumPy数组
        for sample_mask in ([] if fedmass_training_mode else imb_mr_csv_data['mask']):
            clinet_modal_num += np.array(eval(sample_mask), dtype=np.float32)  # 将字符串转换为NumPy数组并累加
        if not fedmass_training_mode:
            client_modal_weight.append(clinet_modal_num)

        mask_id_count = [0] * 15
        for mask_id in ([] if fedmass_training_mode else imb_mr_csv_data['mask_id']):
            mask_id_count[mask_id] += 1
        if not fedmass_training_mode:
            logging.info('Legacy train_file Mask ID Count: {}'.format(mask_id_count))
            mask_id_count_list.append(mask_id_count)
    mask_id_count_source = fedmass_supervised_mask_id_count_list if fedmass_training_mode else mask_id_count_list
    total = [sum(x) for x in zip(*mask_id_count_source)]
    for mask_id_count in mask_id_count_source:
        client_mask_id_proportions.append([mask_id_count[i] / total[i] if total[i] != 0 else 0 for i in range(len(mask_id_count))])
    client_mask_proportions_sum = [sum(client_mask_id_proportions[i])/len(client_mask_id_proportions[i]) for i in range(len(client_mask_id_proportions))]
    logging.info(f'client_mask_proportions_sum: {client_mask_proportions_sum}')
    logging.info(f'client_mask_id_proportions: {client_mask_id_proportions}')
    ####15种mask的分布
    
    client_modal_weight = np.array(fedmass_client_modal_weight if fedmass_training_mode else client_modal_weight)
    all_client_modal_weight = client_modal_weight.sum(axis=0)
    all_client_modal_weight = np.where(all_client_modal_weight == 0, 1, all_client_modal_weight)
    client_modal_weight = client_modal_weight / all_client_modal_weight

    logging.info(f'client_modal_weight: {client_modal_weight}')


    valid_set = Brats_loadall_val_nii(transforms=args.test_transforms, root=args.datapath, train_file=args.valid_file)
    valid_loader = DataLoader(dataset=valid_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    
    test_set = Brats_loadall_test_nii(transforms=args.test_transforms, root=args.datapath, test_file=args.valid_file)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    logging.info('the length of Brats dataset is {} : {}'.format(len(valid_set), len(test_set)))

    #validloader_clients.append(valid_loader)
    #testloader_clients.append(test_loader)
          
    ########## FL Training ##########
    for round in tqdm(range(args.start_round, args.c_rounds+1)):
        start = time.time()
        ##### local training
        local_weights, local_losses, local_protos = [], {}, {}
        client_full_anchor_list = []
        client_unlabeled_proto_list = []
        client_missing_pattern_proto_list = []
        client_reliability_payloads = []
        logging.info(f'\n | Global Training Round : {round} |')
        if args.enable_anchor_supervision:
            if global_anchor_bank is None:
                logging.info('Round {} global anchor bank is empty before local training'.format(round))
            else:
                logging.info(
                    'Round {} global anchor bank ready with {} valid classes'.format(
                        round,
                        int(global_anchor_bank['valid_mask'].sum().item()),
                    )
                )
        if args.enable_pseudo_filtering:
            if global_missing_proto_bank is None:
                logging.info('Round {} global missing-pattern prototype bank is empty before local training'.format(round))
            else:
                logging.info(
                    'Round {} global missing-pattern prototype bank ready with {} valid pattern-class entries'.format(
                        round,
                        count_valid_pattern_classes(global_missing_proto_bank),
                    )
                )
        start = time.time()

        result = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if args.use_multiprocessing:
            ctx = torch.multiprocessing.get_context("spawn")
            pool = ctx.Pool(args.client_num)
        for client_i in range(args.client_num):
            if args.use_multiprocessing:
                result.append(pool.apply_async(local_training, args=(
                    args,
                    args.local_devices[client_i],
                    masks_torch[client_i],
                    dataloader_clients[client_i],
                    model_clients[client_i],
                    client_i,
                    round,
                    cluster_centers,
                    client_mask_id_proportions[client_i],
                    anchorloader_clients[client_i],
                    unlabeledloader_clients[client_i],
                    teacher_clients[client_i],
                    global_anchor_bank,
                    global_missing_proto_bank,
                )))
            else:
                result.append(local_training(
                    args,
                    args.local_devices[client_i],
                    masks_torch[client_i],
                    dataloader_clients[client_i],
                    model_clients[client_i],
                    client_i,
                    round,
                    cluster_centers,
                    client_mask_id_proportions[client_i],
                    anchorloader_clients[client_i],
                    unlabeledloader_clients[client_i],
                    teacher_clients[client_i],
                    global_anchor_bank,
                    global_missing_proto_bank,
                ))
        if args.use_multiprocessing:
            pool.close()
            pool.join()
        
        logging.info("local client training: {}".format(time.time() - start))
        client_gt_list = []
        for client_i, i in enumerate(result):
            if args.use_multiprocessing:
                m, loss,client_gt,extra_outputs = i.get()
            else:
                m, loss,client_gt,extra_outputs = i
            local_weights.append(copy.deepcopy(m.state_dict()))
            local_losses = copy.deepcopy(loss)
            model_clients[client_i] = m              
            log_client_train(writer, client_i, local_losses, round)
            client_gt_list+=client_gt
            client_full_anchors = extra_outputs.get('local_full_anchors')
            client_full_anchor_list.append(client_full_anchors)
            if client_full_anchors is not None:
                valid_classes = int(torch.any(client_full_anchors != 0, dim=1).sum().item())
                logging.info('Client-{} Module1 anchors updated for {} classes'.format(client_i+1, valid_classes))
            client_unlabeled_prototypes = extra_outputs.get('local_unlabeled_prototypes')
            client_unlabeled_proto_list.append(client_unlabeled_prototypes)
            if client_unlabeled_prototypes is not None:
                valid_pseudo_classes = int(torch.any(client_unlabeled_prototypes != 0, dim=1).sum().item())
                logging.info('Client-{} Module2 unlabeled prototypes updated for {} classes'.format(client_i+1, valid_pseudo_classes))
            client_missing_pattern_prototypes = extra_outputs.get('local_missing_pattern_prototypes')
            client_missing_pattern_proto_list.append(client_missing_pattern_prototypes)
            if args.enable_pseudo_filtering:
                logging.info(
                    'Client-{} missing-pattern prototypes uploaded for {} valid pattern-class entries'.format(
                        client_i+1,
                        count_valid_pattern_classes(client_missing_pattern_prototypes),
                    )
                )
            client_reliability_payloads.append({
                'pseudo_stats': extra_outputs.get('pseudo_stats'),
                'pseudo_active': extra_outputs.get('pseudo_active', False),
                'local_unlabeled_prototypes': client_unlabeled_prototypes,
            })
            teacher_state = extra_outputs.get('teacher_state')
            if args.enable_pseudo_filtering and teacher_state is not None and teacher_clients[client_i] is not None:
                teacher_clients[client_i].load_state_dict(teacher_state)


        #cluster_centers = criterions.cluster_and_select(client_gt_list)
        new_cluster_centers = criterions.group_cluster_and_select(client_gt_list,masks_test)
        cluster_centers = criterions.EMA_cls_Fs(cluster_centers, new_cluster_centers)
        #criterions.test_clustering(client_gt_list, cluster_centers)    

        if (round+1)%args.round_per_train == 0:
            logging.info('-'*20 + 'Client Validation For R_sup' + '-'*20)
            max_sup_val_samples = args.sup_val_max_samples if args.sup_val_max_samples > 0 else None
            for client_i in range(args.client_num):
                sup_dice_score, _ = validate_dice_softmax(
                    valid_loader,
                    model_clients[client_i],
                    feature_mask=masks_all,
                    device=args.device,
                    max_samples=max_sup_val_samples,
                    log_prefix='Client-{} R_sup validation'.format(client_i+1),
                )
                sup_dice_avg = float((sup_dice_score[0] + sup_dice_score[1] + sup_dice_score[2]) / 3.0)
                reliability_state = update_client_sup_history(
                    reliability_state,
                    client_i,
                    sup_dice_avg,
                    args.sup_history_window,
                )
                sup_history = reliability_state['client_sup_history'][client_i]
                sup_var = reliability_state['client_last_sup_var'][client_i]
                logging.info(
                    'Client-{} supervised validation dice(avg3)={:.4f}, window_len={}, variance={}'.format(
                        client_i+1,
                        sup_dice_avg,
                        len(sup_history),
                        _format_optional_metric(sup_var),
                    )
                )
                writer.add_scalar('Client_{}/sup_val_dice'.format(client_i+1), sup_dice_avg, global_step=round)
                if sup_var is not None:
                    writer.add_scalar('Client_{}/sup_val_variance'.format(client_i+1), float(sup_var), global_step=round)

        reliability_info = build_reliability_aggregation(
            client_payloads=client_reliability_payloads,
            global_anchor_bank=global_anchor_bank,
            client_modal_weight=client_modal_weight,
            reliability_state=reliability_state,
        )
        reliability_state = reliability_info['reliability_state']
        reliability_state['last_round'] = round
        logging.info(
            'Reliability aggregation summary: R_sup_enabled={}, using_global_anchor_bank={}'.format(
                reliability_state.get('sup_enabled', False),
                global_anchor_bank is not None,
            )
        )
        for detail in reliability_info['client_details']:
            logging.info(
                'Client-{} reliability raw={}, norm={}, sup_dice={}, sup_window={}, sup_var={}, R_sup={}, R_pl={}, R_align={}, '
                'fallback={}, model_w={}, encoder_w=[flair:{:.4f}, t1ce:{:.4f}, t1:{:.4f}, t2:{:.4f}]'.format(
                    detail['client_idx'] + 1,
                    _format_optional_metric(detail['raw_score']),
                    _format_optional_metric(detail['normalized_score']),
                    _format_optional_metric(detail['sup_last_dice']),
                    detail['sup_history_length'],
                    _format_optional_metric(detail['sup_variance']),
                    'disabled' if not detail['sup_enabled'] else _format_optional_metric(detail['components']['r_sup']),
                    _format_optional_metric(detail['components']['r_pl']),
                    _format_optional_metric(detail['components']['r_align']),
                    detail['used_fallback'],
                    _format_optional_metric(detail['model_weight']),
                    detail['encoder_weights']['flair'],
                    detail['encoder_weights']['t1ce'],
                    detail['encoder_weights']['t1'],
                    detail['encoder_weights']['t2'],
                )
            )

        # global Aggre and Fusion
        server_model = uploadLCweightsandGLBupdate(
            server_model,
            local_weights,
            client_mask_proportions_sum,
            client_modal_weight,
            model_clients,
            aggregation_weights={
                'model_weights': reliability_info['model_weights'],
                'encoder_weights': reliability_info['encoder_weights'],
            },
        )
        downloadGLBweights(server_model, model_clients)
        if args.enable_anchor_supervision:
            global_anchor_bank, anchor_bank_stats = aggregate_global_anchor_bank(
                client_full_anchor_list,
                prev_anchor_bank=global_anchor_bank,
                ema=args.anchor_bank_ema,
            )
            if anchor_bank_stats['updated_classes'] > 0:
                msg = (
                    'Server global anchor bank updated: contributing_clients={}/{}, '
                    'updated_classes={}, total_valid_classes={}, ema={:.3f}'
                ).format(
                    anchor_bank_stats['num_contributing_clients'],
                    anchor_bank_stats['num_clients_total'],
                    anchor_bank_stats['updated_classes'],
                    anchor_bank_stats['total_valid_classes'],
                    args.anchor_bank_ema,
                )
                if anchor_bank_stats['reinitialized']:
                    msg += ' (reinitialized due to shape mismatch)'
                logging.info(msg)
            else:
                logging.info(
                    'Server global anchor bank skipped: contributing_clients={}/{}, total_valid_classes={}'.format(
                        anchor_bank_stats['num_contributing_clients'],
                        anchor_bank_stats['num_clients_total'],
                        anchor_bank_stats['total_valid_classes'],
                    )
                )
        if args.enable_pseudo_filtering:
            global_missing_proto_bank, missing_proto_stats = aggregate_global_missing_proto_bank(
                client_missing_pattern_proto_list,
                prev_proto_bank=global_missing_proto_bank,
                ema=args.missing_proto_bank_ema,
                client_weights=reliability_info['reliability_scores'],
            )
            if missing_proto_stats['updated_pattern_classes'] > 0:
                msg = (
                    'Server global missing-pattern prototype bank updated: contributing_clients={}/{}, '
                    'updated_pattern_classes={}, total_valid_pattern_classes={}, ema={:.3f}'
                ).format(
                    missing_proto_stats['num_contributing_clients'],
                    missing_proto_stats['num_clients_total'],
                    missing_proto_stats['updated_pattern_classes'],
                    missing_proto_stats['total_valid_pattern_classes'],
                    args.missing_proto_bank_ema,
                )
                if missing_proto_stats['reinitialized']:
                    msg += ' (reinitialized due to shape mismatch)'
                logging.info(msg)
            else:
                logging.info(
                    'Server global missing-pattern prototype bank skipped: contributing_clients={}/{}, '
                    'total_valid_pattern_classes={}'.format(
                        missing_proto_stats['num_contributing_clients'],
                        missing_proto_stats['num_clients_total'],
                        missing_proto_stats['total_valid_pattern_classes'],
                    )
                )
        ##### Eval the model after aggregation and 10 round
        if (round+1)%args.round_per_train == 0:# and round>200:
            logging.info('-'*20 + 'Test All the Models per 10 round'+ '-'*20)
            test_dice_score = AverageMeter()
            #test_hd95_score = AverageMeter()
            csv_name = os.path.join(args.save_path, '{}.csv'.format('rfnet'))
            with torch.no_grad():
                file = open(csv_name, "a+")
                csv_writer = csv.writer(file)
                csv_writer.writerow(['WT Dice', 'TC Dice', 'ET Dice','ETPro Dice', 'WT HD95', 'TC HD95', 'ET HD95' 'ETPro HD95'])
                file.close()
                for i, mask in enumerate(masks_test[::-1]):
                    logging.info('{}'.format(mask_name[::-1][i]))
                    file = open(csv_name, "a+")
                    csv_writer = csv.writer(file)
                    csv_writer.writerow([mask_name[::-1][i]])
                    file.close()
                    dice_score, class_evaluation = test_dice_softmax(
                                    test_loader,
                                    server_model,
                                    dataname = args.dataname,
                                    feature_mask = mask,
                                    mask_name = mask_name[::-1][i],
                                    csv_name = csv_name,
                                    device = args.device
                                    )
                    for clev in range(len(class_evaluation)):
                        writer.add_scalar('{}/Eval_dice_{}'.format(mask_name[::-1][i], class_evaluation[clev]), dice_score[clev], round)
                        #writer.add_scalar('Eval_hd95_{}_{}'.format(mask_name[::-1][i], class_evaluation[clev]), hd95_score[clev], round)
                    test_dice_score.update(dice_score)
                    #test_hd95_score.update(hd95_score)

                logging.info('Avg Dice scores: {} avg:{}'.format(test_dice_score.avg,(test_dice_score.avg[0]+test_dice_score.avg[1]+test_dice_score.avg[2])/3))
                #logging.info('Avg HD95 scores: {}'.format(test_hd95_score.avg))
                for clev in range(len(class_evaluation)):
                    writer.add_scalar('Eval_AvgDice/{}'.format(class_evaluation[clev]), test_dice_score.avg[clev], round)

        logging.info('*'*10+'FL train a round total time: {:.4f} hours'.format((time.time() - start)/3600)+'*'*10)
        #for c in range(args.client_num):
        #    print("bbbbbbbbbbbbbbb", model_clients[c].decoder_fuse.d3_c1.conv.weight[10,10,1,1])
        if (round+1)%args.round_per_train == 0:
            torch.save({
            'round': round + 1,
            'server': server_model.state_dict(),
            'cluster_centers': cluster_centers,
            'global_anchor_bank': global_anchor_bank,
            'global_missing_proto_bank': global_missing_proto_bank,
            'reliability_state': reliability_state,
            'best_dice': best_dice,
            'best_dices': best_dices
            }, args.modelfile_path + '/last.pth')
            
    writer.close()    
