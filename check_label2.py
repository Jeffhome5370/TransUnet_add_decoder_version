import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import os
import scipy.ndimage

# 1. 設定標籤對應
class_labels = {
    0: "Background",
    1: "Aorta",
    2: "Gallbladder",
    3: "Kidney_L",
    4: "Kidney_R",
    5: "Liver",
    6: "Pancreas",
    7: "Spleen",
    8: "Stomach"
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

count = 0
total = 0
for filename in files:
    data = np.load(os.path.join(data_path, filename))
    # ▼▼▼▼▼▼▼ 修正點在這裡 ▼▼▼▼▼▼▼
    # 強制轉為 int 型態，避免 TypeError
    label = data['label'].astype(int) 
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
    
    unique_classes = np.unique(label)
    if 2 in unique_classes:
        count += 1
    total += 1
print(f"{count} / {total}")