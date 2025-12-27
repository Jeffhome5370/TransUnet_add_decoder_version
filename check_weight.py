import torch

# 換成你訓練出來的權重路徑
pth_path = r"..\model\TU_Synapse512\TU_pretrain_R50-ViT-B_16_skip3_epo1_bs24_512\epoch_0.pth" 

state_dict = torch.load(pth_path)

print("=== 檢查權重 Keys ===")
has_decoder = False
for key in state_dict.keys():
    # 檢查是否有 decoder 相關的關鍵字
    if "decoder_layers" in key or "query_embed" in key:
        print(f"Found Decoder Key: {key} | Shape: {state_dict[key].shape}")
        has_decoder = True
        
if has_decoder:
    print("\n✅ 確認：權重檔中包含 Transformer Decoder 的參數！")
else:
    print("\n❌ 警告：沒找到 Decoder 參數，請檢查 save 邏輯。")