'''
輸出結果
===== Pixel Counts / Ratios =====
Total pixels: 463732736
class  0: count=   443106950 | ratio=0.95552224
class  1: count=      729993 | ratio=0.00157417
class  2: count=      187946 | ratio=0.00040529
class  3: count=     1061963 | ratio=0.00229003
class  4: count=     1064177 | ratio=0.00229481
class  5: count=    11575064 | ratio=0.02496064
class  6: count=      539063 | ratio=0.00116244
class  7: count=     2604912 | ratio=0.00561727
class  8: count=     2862668 | ratio=0.00617310
'''
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root_path", type=str, default='../data/Synapse/train_npz', help="Synapse dataset root path")
    p.add_argument("--list_dir", type=str, default='./lists/lists_Synapse', help="Synapse list_dir (same as trainer)")
    p.add_argument("--num_classes", type=int, default=9, help="include background, e.g. 9")
    p.add_argument("--split", type=str, default="train_split", help="train_split / val / etc.")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", action="store_true")
    return p.parse_args()

@torch.no_grad()
def main():
    args = parse_args()

    # NOTE: 統計分佈時，建議不要用 RandomGenerator 這種隨機增強，避免比例被裁切/縮放扭曲
    from datasets.dataset_synapse import Synapse_dataset
    ds = Synapse_dataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split=args.split,
        transform=None
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    counts = torch.zeros(args.num_classes, dtype=torch.long)
    total_pixels = 0

    for batch in tqdm(dl, desc=f"Counting pixels ({args.split})"):
        label = batch["label"]  # (B,1,H,W) or (B,H,W) or sometimes (H,W)

        if torch.is_tensor(label) is False:
            label = torch.as_tensor(label)

        # unify to (B,H,W)
        if label.dim() == 2:
            label = label.unsqueeze(0)
        if label.dim() == 4 and label.size(1) == 1:
            label = label.squeeze(1)
        label = label.long()

        # count
        for c in range(args.num_classes):
            counts[c] += (label == c).sum().item()
        total_pixels += label.numel()

    ratios = counts.float() / max(total_pixels, 1)

    print("\n===== Pixel Counts / Ratios =====")
    print(f"Total pixels: {total_pixels}")
    for c in range(args.num_classes):
        print(f"class {c:2d}: count={counts[c].item():12d} | ratio={ratios[c].item():.8f}")

    # ---------- suggested weights ----------
    eps = 1e-12
    p = ratios.clamp_min(eps)

    # (1) Median Frequency Balancing (通常 segmentation 很好用)
    # median over foreground only (exclude background)
    fg = p[1:]
    median_fg = fg.median()
    w_median = (median_fg / p).clamp(0.2, 5.0)
    w_median[0] = 0.3  # background 通常別太小也別太大，先給 0.3 當穩定起手式

    # (2) Sqrt Inverse Frequency (更保守、更不容易爆)
    w_sqrt = torch.sqrt(1.0 / p).clamp(0.2, 5.0)
    w_sqrt[0] = 0.3

    print("\n===== Suggested class weights =====")
    print("[MedianFreq] (clamp 0.2~5.0, bg=0.3):")
    print("torch.tensor([" + ", ".join(f"{x:.4f}" for x in w_median.tolist()) + "])")

    print("\n[SqrtInv] (clamp 0.2~5.0, bg=0.3):")
    print("torch.tensor([" + ", ".join(f"{x:.4f}" for x in w_sqrt.tolist()) + "])")

    print("\nDone.")

if __name__ == "__main__":
    main()
