import os
import h5py
import numpy as np
'''
h5_dir = "../data/Synapse/test_vol_h5"
save_dir = "../data/Synapse/test_npz"
os.makedirs(save_dir, exist_ok=True)

for fname in sorted(os.listdir(h5_dir)):
    if not fname.endswith(".h5"):
        continue

    vol_name = fname.replace(".npy.h5", "")
    h5_path = os.path.join(h5_dir, fname)

    with h5py.File(h5_path, "r") as f:
        image = f["image"][:]   # (D,H,W)
        label = f["label"][:]

    for i in range(image.shape[0]):
        img_slice = image[i].astype(np.float32)
        lab_slice = label[i].astype(np.uint8)

        slice_name = f"{vol_name}_slice_{i:04d}.npz"
        np.savez(
            os.path.join(save_dir, slice_name),
            image=img_slice,
            label=lab_slice
        )

    print(f"{vol_name} done.")
'''
#----------------check------------------
x = np.load("../data/Synapse/test_npz/case0001_slice_0080.npz")
print(x["image"].shape, x["image"].min(), x["image"].max())
print(np.unique(x["label"]))