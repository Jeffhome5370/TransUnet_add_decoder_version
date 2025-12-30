import os
import random
import numpy as np
import matplotlib.pyplot as plt

TEST_NPZ_DIR = r"../data/Synapse/test_npz"     # 改成你的
VAL_NPZ_DIR  = r"../data/Synapse/train_npz"    # 或 val_npz，改成你的
OUT_DIR      = r"./debug_vis"
N_SAMPLES    = 12  # 想看幾張

os.makedirs(OUT_DIR, exist_ok=True)

def list_npz(d):
    return sorted([os.path.join(d, f) for f in os.listdir(d) if f.endswith(".npz")])

def load_npz(p):
    d = np.load(p)
    img = d["image"].astype(np.float32)
    lab = d["label"]
    # 允許 label 是 float 但內容應為整數
    lab = lab.astype(np.int32)
    return img, lab

def save_vis(npz_path, prefix):
    img, lab = load_npz(npz_path)

    # 基本統計（順便寫在檔名旁邊，方便你對照）
    mn, mx, mean, std = float(img.min()), float(img.max()), float(img.mean()), float(img.std())
    uniq = np.unique(lab)
    uniq_str = "-".join(map(str, uniq[:10])) + ("..." if len(uniq) > 10 else "")

    base = os.path.splitext(os.path.basename(npz_path))[0]
    tag = f"{prefix}_{base}_mn{mn:.2f}_mx{mx:.2f}_mu{mean:.2f}_sd{std:.2f}_lab{uniq_str}"
    tag = tag.replace(":", "_").replace("\\", "_").replace("/", "_")

    # 影像（自動對比拉伸顯示）
    plt.figure()
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.title(f"{prefix} {base}\nmin/max={mn:.2f}/{mx:.2f} mean/std={mean:.2f}/{std:.2f}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{tag}_img.png"), dpi=150)
    plt.close()

    # 疊圖：label > 0 的區域用透明顏色覆蓋
    plt.figure()
    plt.imshow(img, cmap="gray")
    mask = (lab > 0).astype(np.float32)
    plt.imshow(mask, alpha=0.35)
    plt.axis("off")
    plt.title(f"{prefix} {base} overlay\nlabel uniq: {uniq[:12]}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{tag}_overlay.png"), dpi=150)
    plt.close()

if __name__ == "__main__":
    test_files = list_npz(TEST_NPZ_DIR)
    val_files  = list_npz(VAL_NPZ_DIR)

    random.seed(0)
    random.shuffle(test_files)
    random.shuffle(val_files)

    for p in test_files[:N_SAMPLES]:
        save_vis(p, "TEST")

    for p in val_files[:N_SAMPLES]:
        save_vis(p, "VAL")

    print(f"Saved to: {os.path.abspath(OUT_DIR)}")
