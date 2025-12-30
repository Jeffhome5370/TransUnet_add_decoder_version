import os

npz_dir = "../data/Synapse/test_npz"     # 你的 npz 資料夾
list_dir = "./lists/lists_Synapse"       # test.txt 要放的地方
os.makedirs(list_dir, exist_ok=True)

out_path = os.path.join(list_dir, "test.txt")

names = []
for fname in os.listdir(npz_dir):
    if fname.endswith(".npz"):
        names.append(fname.replace(".npz", ""))

# 保持順序穩定（很重要，debug 時會救命）
names.sort()

with open(out_path, "w") as f:
    for name in names:
        f.write(name + "\n")

print(f"Done! {len(names)} entries written to {out_path}")
