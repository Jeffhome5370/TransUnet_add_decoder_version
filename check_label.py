import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import os
import scipy.ndimage

# 1. 設定標籤對應
class_labels = {
    0: "Background",
    1: "Aorta (主動脈)",
    2: "Gallbladder (膽囊)",
    3: "Kidney_L (左腎)",
    4: "Kidney_R (右腎)",
    5: "Liver (肝臟)",
    6: "Pancreas (胰臟)",
    7: "Spleen (脾臟)",
    8: "Stomach (胃)"
}

# 2. 定義每個類別的固定顏色
colors = [
    'black',   # 0
    'red',     # 1
    'green',   # 2
    'blue',    # 3
    'yellow',  # 4
    'cyan',    # 5
    'magenta', # 6
    'orange',  # 7
    'purple'   # 8
]
custom_cmap = ListedColormap(colors)

# 3. 設定路徑
data_path = r"../data/Synapse/train_npz" 
files = os.listdir(data_path)

target_file = None
print("正在搜尋包含多個器官的切片...")

for filename in files:
    data = np.load(os.path.join(data_path, filename))
    # ▼▼▼▼▼▼▼ 修正點在這裡 ▼▼▼▼▼▼▼
    # 強制轉為 int 型態，避免 TypeError
    label = data['label'].astype(int) 
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
    
    unique_classes = np.unique(label)
    
    if len(unique_classes) == 9:
        target_file = filename
        print(f"找到目標檔案: {filename}")
        print(f"包含類別: {unique_classes}")
        break

if target_file:
    data = np.load(os.path.join(data_path, target_file))
    image = data['image']
    # ▼▼▼▼▼▼▼ 這裡也要轉型 ▼▼▼▼▼▼▼
    label = data['label'].astype(int)
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    # --- 開始畫圖 ---
    plt.figure(figsize=(12, 8))

    # 左圖
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original CT Image")
    plt.axis('off')

    # 右圖
    plt.subplot(1, 2, 2)
    plt.imshow(label, cmap=custom_cmap, vmin=0, vmax=8, interpolation='nearest')
    plt.title("Ground Truth Segmentation")
    plt.axis('off')

    # --- 製作圖例與標註 ---
    present_classes = np.unique(label)
    legend_patches = []
    
    for c_id in present_classes:
        # 確保 c_id 是 int (雖然上面轉過了，雙重保險)
        c_id = int(c_id)
        
        if c_id == 0: continue 
        
        patch = mpatches.Patch(color=colors[c_id], label=f"{c_id}: {class_labels[c_id]}")
        legend_patches.append(patch)
        
        # 標註文字
        mask = (label == c_id)
        coords = scipy.ndimage.center_of_mass(mask)
        center_y, center_x = coords
        
        plt.text(center_x, center_y, class_labels[c_id].split(' ')[0], 
                 color='white', 
                 fontsize=10, 
                 fontweight='bold',
                 ha='center', va='center',
                 bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))

    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()
    plt.show()

else:
    print("找不到包含足夠器官的切片。")