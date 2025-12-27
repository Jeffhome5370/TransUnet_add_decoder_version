import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.vit_seg_modeling import TransUNet_TransformerDecoder
from trainer import trainer_synapse

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--num_queries', type=int,
                    default=20, help='number of queries for transformer decoder')
parser.add_argument('--add_decoder', type=int,
                    default=0, help='1 for add transformer decoder or just transformer encoder')
parser.add_argument('--exp_name', type=str, default=None, help='Name of the experiment run')
args = parser.parse_args()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'Synapse': {
            'root_path': '../data/Synapse/train_npz',
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
        },
    }
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.is_pretrain = True
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path+'_'+str(args.max_iterations)[0:2]+'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    # net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    # net.load_from(weights=np.load(config_vit.pretrained_path))
    
    #trainer = {'Synapse': trainer_synapse,}
    #trainer[dataset_name](args, net, snapshot_path)

    # -------------------------------------------------------
    # [新增 2] 初始化 WandB
    # -------------------------------------------------------
    # 如果沒有指定 exp_name，就用預設的 exp 字串
    run_name = args.exp_name if args.exp_name else args.exp
    
    wandb.init(
        project="TransUNet_Experiments",  # 專案名稱 (自己取)
        name=run_name,                    # 這次實驗的名稱
        config=args,                      # 自動把所有 args 存成超參數表
        reinit=True
    )
    # -------------------------------------------------------
    # A. 實例化原始 TransUNet (目的是為了載入 R50+ViT 的權重)
    original_net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    
    # B. 載入 ImageNet21k 預訓練權重
    original_net.load_from(weights=np.load(config_vit.pretrained_path))
    print("Pretrained R50+ViT weights loaded.")

    if (args.add_decoder):
        # C. 將原始模型包裝進 Decoder-only 架構
        # 這會自動凍結 original_net 的參數，並初始化新的 Decoder
        net = TransUNet_TransformerDecoder(
            original_model=original_net,
            num_classes=args.num_classes,
            num_queries=args.num_queries, # 使用 argument 傳入的值
            num_decoder_layers=3
        ).cuda()
    
        # Log 資訊確認
        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in net.parameters())
        print(f"Model constructed. Encoder frozen.")
        print(f"Total Params: {total_params/1e6:.2f}M, Trainable Params: {trainable_params/1e6:.2f}M")
    else:
        net = original_net
    # -----------------------------------------------------------------

    trainer = {'Synapse': trainer_synapse,}
    trainer[dataset_name](args, net, snapshot_path)