import os
import random
import numpy as np

# 設定路徑
list_dir = './lists/lists_Synapse'
original_train_file = os.path.join(list_dir, 'train.txt')

# 設定切分比例 (例如: 80% 訓練, 20% 驗證)
val_ratio = 0.2 
seed = 1234

def split_dataset():
    # 1. 讀取原始 train.txt
    if not os.path.exists(original_train_file):
        print(f"Error: 找不到 {original_train_file}")
        return

    with open(original_train_file, 'r') as f:
        lines = f.readlines()
    
    # 去除空白
    data_list = [line.strip() for line in lines if line.strip()]
    total_samples = len(data_list)
    print(f"原始訓練集總數: {total_samples}")

    # 2. 打亂順序
    random.seed(seed)
    random.shuffle(data_list)

    # 3. 切分
    val_size = int(total_samples * val_ratio)
    val_list = data_list[:val_size]
    train_list = data_list[val_size:]

    print(f"新的訓練集數量: {len(train_list)}")
    print(f"新的驗證集數量: {len(val_list)}")

    # 4. 存檔
    with open(os.path.join(list_dir, 'train_split.txt'), 'w') as f:
        f.write('\n'.join(train_list))
    
    with open(os.path.join(list_dir, 'val.txt'), 'w') as f:
        f.write('\n'.join(val_list))
        
    print("切分完成！已產生 train_split.txt 與 val.txt")

if __name__ == '__main__':
    split_dataset()