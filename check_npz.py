import numpy as np
import os

def inspect_npz_data(npz_path):
    if not os.path.exists(npz_path):
        print(f"錯誤: 找不到檔案 {npz_path}")
        return

    print(f"\n正在檢查檔案: {os.path.basename(npz_path)}")
    print("=" * 40)

    try:
        data = np.load(npz_path)
        keys = list(data.files)
        print(f"檔案包含 Keys: {keys}")

        # --- 檢查 Image ---
        if 'image' in data:
            img = data['image']
            print(f"\n[Image 分析]")
            print(f"  Shape (維度): {img.shape}")
            print(f"  Dtype (類型): {img.dtype}")
            print(f"  Min   (最小值): {img.min()}")
            print(f"  Max   (最大值): {img.max()}")
            print(f"  Mean  (平均值): {img.mean():.4f}")
            
            # 檢查是否為空 (全是0)
            if img.min() == 0 and img.max() == 0:
                print("  警告: 影像全是 0 (全黑)！")
        else:
            print("\n[Image] 警告: 找不到 'image' key")

        # --- 檢查 Label ---
        if 'label' in data:
            lbl = data['label']
            print(f"\n[Label 分析]")
            print(f"  Shape (維度): {lbl.shape}")
            print(f"  Unique Classes (包含類別): {np.unique(lbl)}")
        else:
            print("\n[Label] 警告: 找不到 'label' key")
            
        print("=" * 40)

    except Exception as e:
        print(f"讀取失敗: {e}")

if __name__ == "__main__":
    # ==========================================
    # 請將這裡換成你電腦裡真實的 .npz 路徑
    # 例如: r"C:\path\to\Synapse\train_npz\case0005_slice050.npz"
    # ==========================================
    file_path = r"..\data\Synapse\train_npz\case0005_slice000.npz" 
    
    inspect_npz_data(file_path)