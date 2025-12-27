import torch
import torch.nn as nn
from vit_seg_modeling import VisionTransformer, CONFIGS as CONFIGS_ViT_seg

# 這裡放入我們之前討論過的 Transformer Decoder 類別
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        # 1. Masked Cross-Attention (根據 Figure 1，這層排在最前面)
        # 負責將 Query 與 CNN 特徵 (features) 結合，並套用 Mask
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model) # Cross-Attn 後的 Norm
        
        # 2. Multi-Head Self-Attention (MHSA) (根據 Figure 1，這層排在中間)
        # 負責 Query 之間的資訊交換
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model) # Self-Attn 後的 Norm
        
        # 3. Feed Forward Network (MLP)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = nn.LayerNorm(d_model) # MLP 後的 Norm

    def forward(self, query, key_value, current_mask=None):
        """
        順序修正: Cross-Attn -> Self-Attn -> MLP
        """
        
        # --- Step 1: Masked Cross-Attention (First) ---
        # 這是論文圖示的第一個黃色區塊 "Masked attention"
        # query: (B, Q, C)
        # key_value: (B, L, C) -> CNN Features
        
        # 準備 Mask (Bias)
        attn_bias = None
        if current_mask is not None:
            # 根據論文 Eq. 7 製作 Mask (背景處填入極小負值)
            # current_mask 形狀: (B, Q, H, W) -> 需轉為與 attention map 對應的形狀
            # 注意: 這裡簡化示意，實作時需注意 head 維度
            attn_bias = torch.zeros_like(current_mask)
            attn_bias[current_mask < 0.5] = -1e9 #-10^9

        # 實作 Pre-LN 或 Post-LN (圖示為 Layer Norm 接在 Attention 後)
        # 這裡採用常見的 Pre-LN 結構增強穩定性，或依圖示順序實作
        # 根據圖示箭頭: Query -> Masked Attn -> LayerNorm -> MHSA ...
        
        # Residual Connection (圖中雖未畫出 Cross 的外圍殘差，但深度網路通常需要)
        residual = query
        
        # Cross Attention
        # query 是 Q, key_value 是 K, V
        q_cross, _ = self.cross_attn(query, key_value, key_value, attn_mask=attn_bias)
        query = residual + q_cross
        query = self.norm1(query) # 圖示中的第一個 Layer Norm

        # --- Step 2: Multi-Head Self-Attention (Second) ---
        # 這是論文圖示的第二個黃色區塊 "MHSA"
        residual = query
        q_self, _ = self.self_attn(query, query, query)
        query = residual + q_self
        query = self.norm2(query) # 圖示中的第二個 Layer Norm

        # --- Step 3: MLP (Third) ---
        residual = query
        q_mlp = self.linear2(self.dropout(F.relu(self.linear1(query))))
        query = residual + q_mlp
        query = self.norm3(query) # MLP 後的 Layer Norm

        return query

class TransUNet_TransformerDecoder(nn.Module):
    def __init__(self, original_model, num_classes, num_queries=20, num_decoder_layers=3):
        super().__init__()
        # 1. 繼承原有的 Encoder (ResNet + ViT)
        self.transformer = original_model.transformer
        
        # 鎖定 Encoder 參數 (Freeze)
        for param in self.transformer.parameters():
            param.requires_grad = False
            
        # 取得 Hidden Size (通常 ViT-Base 是 768)
        self.hidden_size = original_model.config.hidden_size
        
        # 2. 新增 Transformer Decoder (替換掉原本的 CNN Decoder)
        # 論文 Part IV: Mask Classification [cite: 205-207]
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, self.hidden_size)
        
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(self.hidden_size, nhead=8) 
            for _ in range(num_decoder_layers)
        ])
        
        self.class_head = nn.Linear(self.hidden_size, num_classes)
        self.mask_projector = nn.Linear(self.hidden_size, self.hidden_size) # 用於計算 Mask

    def forward(self, x):
        # A. 使用凍結的 Encoder 提取特徵
        # 注意: 確保這裡調用的是 features extraction 而不是完整的 forward
        if x.size(1) == 1:
            x = x.repeat(1,3,1,1)
            
        # 根據 vit_seg_modeling 的實作，通常 transformer 輸出 (logits, hidden_states)
        # 我們需要 hidden_states (B, N, C)
        x, hidden_states = self.transformer.embeddings(x) 
        encoded_feats, _ = self.transformer.encoder(x, hidden_states)  # (B, N, C)
        
        # B. Transformer Decoder 流程
        B = encoded_feats.shape[0]
        
        # 初始化 Queries
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1) # (B, Q, C)
        
        # 初始 Mask (Z^0)
        # 這裡簡化計算，直接拿 queries 和 key 做點積
        # 論文 Eq. 4 [cite: 165-166]
        logits = torch.bmm(queries, encoded_feats.transpose(1, 2))
        current_mask = torch.sigmoid(logits)
        
        refined_masks = []
        
        # Iterative Refinement
        for layer in self.decoder_layers:
            queries = layer(queries, encoded_feats, current_mask)
            
            # 更新 Mask
            # 這裡可以加上一個 Projector 讓維度對齊
            mask_embed = self.mask_projector(queries)
            logits = torch.bmm(mask_embed, encoded_feats.transpose(1, 2))
            current_mask = torch.sigmoid(logits)
            
            # 需將 Mask Reshape 回空間維度以便計算 Loss (B, Q, H/P, W/P)
            # 假設 Patch size 16, 輸入 224 -> 14x14
            # 這裡需要根據你的 input size 動態調整
            spatial_dim = int(encoded_feats.shape[1]**0.5) 
            refined_masks.append(logits.view(B, self.num_queries, spatial_dim, spatial_dim))
            
        # 分類頭
        class_logits = self.class_head(queries)
        
        return class_logits, refined_masks