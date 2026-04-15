import argparse
import os


def _build_client_split_file_dict(split_dir, client_num, filename_pattern):
    file_dict = {}
    for client_idx in range(1, client_num + 1):
        file_dict[client_idx] = os.path.join(
            split_dir,
            filename_pattern.format(client_id=client_idx),
        )
    return file_dict


def resolve_fedmass_split_files(args):
    """
    根据 FedMASS split 目录自动装配 Module 1 / Module 2 的 client 文件路径。

    规则：
    1. 当 args.fedmass_split_dir 为 None 时，不修改现有配置，保持旧行为；
    2. 当 enable_anchor_supervision=True 时，若 anchor_train_file[client_id] 为空，
       则自动指向 client_{id}_labeled_full.csv；
    3. 当 enable_pseudo_filtering=True 时，若 unlabeled_train_file[client_id] 为空，
       则自动指向 client_{id}_unlabeled_missing.csv；
    4. 只要启用了对应模块且声明了 fedmass_split_dir，就要求目标文件实际存在，
       避免静默回退到旧训练集而偏离 FedMASS 设定。
    """
    args.use_fedmass_training = False
    args.supervised_train_file = dict(args.train_file)
    split_dir = getattr(args, 'fedmass_split_dir', None)
    if not split_dir:
        return args

    split_dir = os.path.abspath(split_dir)
    args.fedmass_split_dir = split_dir
    fedmass_modules_enabled = bool(args.enable_anchor_supervision or args.enable_pseudo_filtering)
    # When enabled, train.py should switch the main supervised stream from the
    # legacy args.train_file csv to the strict FedMASS labeled-full split.
    args.use_fedmass_training = bool(split_dir and fedmass_modules_enabled)

    labeled_candidates = _build_client_split_file_dict(
        split_dir=split_dir,
        client_num=args.client_num,
        filename_pattern='client_{client_id}_labeled_full.csv',
    )
    unlabeled_candidates = _build_client_split_file_dict(
        split_dir=split_dir,
        client_num=args.client_num,
        filename_pattern='client_{client_id}_unlabeled_missing.csv',
    )

    if fedmass_modules_enabled:
        for client_idx in range(1, args.client_num + 1):
            if args.anchor_train_file.get(client_idx) is None:
                args.anchor_train_file[client_idx] = labeled_candidates[client_idx]
            if not os.path.isfile(args.anchor_train_file[client_idx]):
                raise FileNotFoundError(
                    f'Module 1 labeled split file not found for client {client_idx}: '
                    f'{args.anchor_train_file[client_idx]}'
                )
            args.supervised_train_file[client_idx] = args.anchor_train_file[client_idx]

    if args.enable_pseudo_filtering:
        for client_idx in range(1, args.client_num + 1):
            if args.unlabeled_train_file.get(client_idx) is None:
                args.unlabeled_train_file[client_idx] = unlabeled_candidates[client_idx]
            if not os.path.isfile(args.unlabeled_train_file[client_idx]):
                raise FileNotFoundError(
                    f'Module 2 unlabeled split file not found for client {client_idx}: '
                    f'{args.unlabeled_train_file[client_idx]}'
                )

    return args


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
    parser.add_argument('--datapath', default='/root/autodl-tmp/BRATS2020_Training_none_npy/', type=str)
    parser.add_argument('--dataname', default='BRATS2020', type=str)
    parser.add_argument('--chose_modal', default='all', type=str)
    parser.add_argument('--num_class', default=4, type=int)
    parser.add_argument('--save_root', default='results', type=str)
    parser.add_argument('--reload_from_checkpoint', action='store_true', default=False, help='reload from checkpoint')
    parser.add_argument('--checkpoint_path', default='./results/debug', type=str)
    parser.add_argument('--pretrain', default=None, type=str)
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--momentum', default=0.5, type=float)
    parser.add_argument('--verbose', default=True)
    parser.add_argument('--visualize', default=True)
    parser.add_argument('--deterministic', default=True)
    parser.add_argument('--seed', default=42, type=int)


    # Settings
    parser.add_argument('--temp', default=4.0, type=float, help='knowledge-distillation temperature')
    parser.add_argument('--mask_type', default='idt', type=str, help='training settings: pdt idt or idt_drop')
    parser.add_argument('--round_per_train', type=int, default=3, help="validate the model per X rounds")
    parser.add_argument('--sup_history_window', default=5, type=int,
                        help='sliding-window size for client supervised validation stability in R_sup')
    parser.add_argument('--sup_val_max_samples', default=4, type=int,
                        help='maximum number of validation samples per client when updating R_sup; <=0 means full validation set')
    parser.add_argument('--region_fusion_start_epoch', default=0, type=int, help='warm-up epochs used in rfnet')
    parser.add_argument('--use_multiprocessing', default=False, help='whether use multiprocessing')
    parser.add_argument('--enable_anchor_supervision', action='store_true', default=False,
                        help='enable FedMASS Module 1 full-modal anchor supervision')
    parser.add_argument('--anchor_warmup_rounds', default=0, type=int,
                        help='start Module 1 after the given communication round')
    parser.add_argument('--anchor_batch_size', default=0, type=int,
                        help='batch size for Module 1 labeled full-modal loader, 0 means reuse batch_size')
    parser.add_argument('--anchor_loss_weight', default=1.0, type=float,
                        help='overall scaling for Module 1 loss')
    parser.add_argument('--anchor_lambda_seg', default=1.0, type=float,
                        help='weight for full-modal segmentation loss in Module 1')
    parser.add_argument('--anchor_lambda_sep', default=0.0, type=float,
                        help='weight for unimodal supervised loss in Module 1')
    parser.add_argument('--anchor_lambda_kd', default=1.0, type=float,
                        help='weight for full-to-unimodal KD loss in Module 1')
    parser.add_argument('--anchor_lambda_proto', default=1.0, type=float,
                        help='weight for prototype distillation loss in Module 1')
    parser.add_argument('--anchor_lambda_prm', default=1.0, type=float,
                        help='weight for PRM supervision loss in Module 1')
    parser.add_argument('--anchor_bank_ema', default=0.9, type=float,
                        help='EMA momentum for server-side global full-modal anchor bank')
    parser.add_argument('--anchor_log_interval', default=50, type=int,
                        help='log Module 1 metrics every N local iterations')
    parser.add_argument('--enable_pseudo_filtering', action='store_true', default=False,
                        help='enable FedMASS Module 2 mask-aware pseudo-label filtering')
    parser.add_argument('--ema_decay', default=0.99, type=float,
                        help='EMA teacher momentum for Module 2')
    parser.add_argument('--pseudo_loss_weight', default=0.1, type=float,
                        help='overall scaling for Module 2 unlabeled loss')
    parser.add_argument('--pseudo_warmup_rounds', default=0, type=int,
                        help='start Module 2 after the given communication round')
    parser.add_argument('--pseudo_conf_base', default=0.7, type=float,
                        help='base confidence threshold tau_0 for Module 2')
    parser.add_argument('--pseudo_conf_gamma', default=0.2, type=float,
                        help='mask-aware threshold increment gamma for Module 2')
    parser.add_argument('--pseudo_temperature', default=0.5, type=float,
                        help='softmax temperature for anchor similarity in Module 2')
    parser.add_argument('--pseudo_consistency_eps', default=0.5, type=float,
                        help='teacher-student consistency threshold for Module 2')
    parser.add_argument('--pseudo_lambda_ce', default=1.0, type=float,
                        help='weight for pseudo CE loss in Module 2')
    parser.add_argument('--pseudo_lambda_dice', default=1.0, type=float,
                        help='weight for pseudo Dice loss in Module 2')
    parser.add_argument('--pseudo_lambda_anchor', default=0.1, type=float,
                        help='weight for unlabeled anchor alignment loss in Module 2')
    parser.add_argument('--pseudo_log_interval', default=10, type=int,
                        help='log Module 2 metrics every N local iterations')
    parser.add_argument('--missing_proto_bank_ema', default=0.9, type=float,
                        help='EMA momentum for server-side missing-pattern prototype bank')

    # FL Settings
    parser.add_argument('--gpus', default='0', help="To use cuda, set to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--c_rounds', type=int, default=300, help="number of rounds of training and communication")
    parser.add_argument('--start_round', type=int, default=0, help="number of rounds of training and communication")

    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--global_ep', type=int, default=1, help="the number of global epochs: E")
    parser.add_argument('--client_num', type=int, default=4, help="number of users: K")
    parser.add_argument('--iid', type=int, default=1, help='Default set to IID. Set to 0 for non-IID.')
    # files
    
    
    parser.add_argument('--train_file', type=dict, 
                default={ 
                1:"./datalist/dir_brats_split/maskid_dir_1024_0.001/client_part_1_imb.csv", 
                2:"./datalist/dir_brats_split/maskid_dir_1024_0.001/client_part_2_imb.csv", 
                3:"./datalist/dir_brats_split/maskid_dir_1024_0.001/client_part_3_imb.csv", 
                4:"./datalist/dir_brats_split/maskid_dir_1024_0.001/client_part_4_imb.csv"})
    parser.add_argument('--anchor_train_file', type=dict,
                default={
                1:None,
                2:None,
                3:None,
                4:None})
    parser.add_argument('--unlabeled_train_file', type=dict,
                default={
                1:None,
                2:None,
                3:None,
                4:None})
    parser.add_argument('--fedmass_split_dir', type=str, default=None,
                help='directory that stores client_k_labeled_full.csv and client_k_unlabeled_missing.csv')
    parser.add_argument('--valid_file', type=str, default="./datalist/BRATS2020_Training_none_npy/val.txt")
    
    
    parser.add_argument('--test_file', type=str, default="./datalist/BRATS2020_Training_none_npy/test.txt")
    parser.add_argument("--device_ids", type=str, default='0,0,0,0')

    # 说明
    parser.add_argument('--version', type=str, default='debug', help='to explain the experiment set up')

    args = parser.parse_args()
    args = resolve_fedmass_split_files(args)
    return args
