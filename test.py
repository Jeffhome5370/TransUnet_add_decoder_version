import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F  # 新增: 用於 Upsample
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
from utils import test_single_volume
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.vit_seg_modeling import TransUNet_TransformerDecoder # 新增: 引入 Decoder 類別
from utils import DiceLoss

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='../data/Synapse/test_vol_h5', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')

parser.add_argument('--max_iterations', type=int,default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=30, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='R50-ViT-B_16', help='select one vit model') # 修正預設值

parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')

parser.add_argument('--add_decoder', type=int, default=0, help='1 for add transformer decoder')
parser.add_argument('--num_queries', type=int, default=20, help='number of queries for transformer decoder')
parser.add_argument('--exp_name', type=str, default='BTCV', help='experiment name') # 讓路徑匹配更容易
parser.add_argument('--decoder_layer', type=int, default=3, help='Numbers of Transformer decoder')
parser.add_argument('--decoder_stride', type=int, default=4, help='stride/downsample rate for transformer decoder (e.g., 2, 4, 8)')
parser.add_argument('--split', type=str, default='test', choices=['train','val','test'])
args = parser.parse_args()

CLASS_LABELS = {
    1: "Aorta",
    2: "Gallbladder",
    3: "Kidney(L)",
    4: "Kidney(R)",
    5: "Liver",
    6: "Pancreas",
    7: "Spleen",
    8: "Stomach",
}

def inference(args, model, test_save_path=None):
    dice_loss = DiceLoss(args.num_classes).cuda()
    val_dice_sum = 0.0
    n = 0
    db_test = Synapse_dataset(
        base_dir=args.volume_path,
        list_dir=args.list_dir,
        split=args.split
    )

    testloader = DataLoader(
        db_test,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )

    logging.info(f"{len(testloader)} {args.split} slices")
    model.eval()

    dice_dict = {c: [] for c in range(1, args.num_classes)}
    '''
    with torch.no_grad():
        for i_batch, batch in tqdm(enumerate(testloader)):
            image = batch["image"].cuda()   # (B,1,H,W)
            label = batch["label"].cuda()   # (B,H,W)
            case_name = batch["case_name"][0] if "case_name" in batch else str(i_batch)

            raw_outputs = model(image)

            # ===== debug：只印第一個 batch =====
            if i_batch == 0:
                print("case:", case_name)
                print("label unique:", torch.unique(label).cpu().tolist()[:50])

                print("type(raw_outputs):", type(raw_outputs))
                if isinstance(raw_outputs, (tuple, list)):
                    print("len(raw_outputs):", len(raw_outputs))
                    for k, o in enumerate(raw_outputs):
                        if torch.is_tensor(o):
                            print(f"  out[{k}] shape={tuple(o.shape)} "
                                  f"min={float(o.min()):.4g} max={float(o.max()):.4g} mean={float(o.mean()):.4g}")
                        else:
                            print(f"  out[{k}] type={type(o)}")
                else:
                    print("raw_outputs shape:", tuple(raw_outputs.shape))
            # ================================

            # 找到真正的 segmentation logits (B,C,H,W)
            if isinstance(raw_outputs, (tuple, list)):
                logits = None
                for o in raw_outputs:
                    if torch.is_tensor(o) and o.dim() == 4 and o.size(1) == args.num_classes:
                        logits = o
                        break
                if logits is None:
                    raise RuntimeError("Cannot find segmentation logits (B,C,H,W) in raw_outputs.")
            else:
                logits = raw_outputs

            if (i_batch == 0):
                bg = logits[:, 0]              # (1,H,W)
                fg = logits[:, 1:]             # (1,8,H,W)
                print("bg min/max/mean:", float(bg.min()), float(bg.max()), float(bg.mean()))
                print("fg max/mean:", float(fg.max()), float(fg.mean()))
            pred = torch.argmax(logits, dim=1)  # (B,H,W)

            if i_batch == 0:
                print("pred unique :", torch.unique(pred).cpu().tolist()[:50])
                print("pred fg ratio:", float((pred > 0).float().mean().item()))

            # ✅ 這段就是你剛剛註解掉的核心：一定要放回來
            for cls in range(1, args.num_classes):
                gt = (label == cls)
                pd = (pred == cls)

                if gt.sum() == 0:
                    continue

                intersect = (gt & pd).sum().float()
                dice = (2 * intersect) / (gt.sum() + pd.sum() + 1e-5)
                dice_dict[cls].append(dice.item())

    logging.info("===== Per-class Dice (slice-level) =====")
    for cls, name in CLASS_LABELS.items():
        if len(dice_dict[cls]) == 0:
            logging.info(f"{name:12s}: N/A (no GT slices)")
        else:
            logging.info(f"{name:12s}: Dice = {np.mean(dice_dict[cls]):.4f} (n={len(dice_dict[cls])})")

    return "Testing Finished!"
    '''
    with torch.no_grad():
        for i_batch, batch in tqdm(enumerate(testloader)):
            image = batch["image"].cuda()
            label = batch["label"].cuda()

            raw = model(image)
            if isinstance(raw, (tuple, list)):
                val_out = raw[-1]
            else:
                val_out = raw

            # trainer validation 的做法：先確保每像素 sum=1
            val_out = val_out + 1e-7
            val_out = val_out / (val_out.sum(dim=1, keepdim=True).clamp_min(1e-6))

            # label shape 對齊
            if label.dim() == 4 and label.size(1) == 1:
                label_ce = label.squeeze(1)
            else:
                label_ce = label

            v_loss_dice = dice_loss(val_out, label_ce, softmax=False)
            val_dice = 1.0 - float(v_loss_dice.item())
            val_dice_sum += val_dice
            n += 1

    print("Trainer-style mean dice:", val_dice_sum / max(n, 1))




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
    '''
    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': '../data/Synapse/test_npz',  # ← 重點
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
            'z_spacing': 1,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True
    '''
    dataset_name = args.dataset
    args.is_pretrain = True
    # name the same snapshot defined in train script!
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    if dataset_name == 'ACDC':  # using max_epoch instead of iteration to control training duration
        snapshot_path = snapshot_path + '_' + str(args.max_iterations)[0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
    
    base_model = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()
    if args.add_decoder:
        print("Add Transformer Decoder...")
        net = TransUNet_TransformerDecoder(
            original_model=base_model,   # <--- 關鍵修改：傳入實例化好的模型
            num_classes=args.num_classes,
            num_queries=args.num_queries,
            img_size= args.img_size,
            decoder_stride=args.decoder_stride,
            num_decoder_layers=args.decoder_layer # 如果您訓練時有改層數，這裡也要加，預設是3
        ).cuda()
    else:
        print("Loading Original TransUNet(only transformer Encoder)...")
        net = base_model

    snapshot = os.path.join(snapshot_path, 'best_model.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
    #-----------------------------------------------------
    ckpt = torch.load(snapshot, map_location="cpu")
    
    # 去掉 module. 前綴（如果有 DataParallel）
    new_ckpt = { (k[7:] if k.startswith("module.") else k): v for k, v in ckpt.items() }

    ret = net.load_state_dict(new_ckpt, strict=True)
    print("Loaded:", snapshot)
    print("Missing:", len(ret.missing_keys), "Unexpected:", len(ret.unexpected_keys))
    print("Missing sample:", ret.missing_keys[:20])
    print("Unexpected sample:", ret.unexpected_keys[:20])

    #----------------------------------------------------------------
    
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = './test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    if args.is_savenii:
        args.test_save_dir = '../predictions'
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    inference(args, net, test_save_path)


