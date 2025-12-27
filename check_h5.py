import h5py
import numpy as np

h5_path = r"..\data\Synapse\test_vol_h5\case0001.npy.h5"   # <<< 確認路徑正確

with h5py.File(h5_path, "r") as f:
    print("=" * 50)
    print("H5 file:", h5_path)
    print("Keys in file:", list(f.keys()))
    print("=" * 50)

    for key in f.keys():
        data = f[key][:]
        print(f"[{key}]")
        print("  shape:", data.shape)
        print("  dtype:", data.dtype)
        print("  min / max:", data.min(), data.max())
        print("  unique (first 20):", np.unique(data)[:20])
        print("  foreground ratio:", (data > 0).mean())
        print("-" * 50)
