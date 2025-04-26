import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ViT(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, num_classes, patch_size=8):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.height = 64
        self.width = 64
        dropout = 0.1

        self.patch_embed = PatchEmbedding(patch_size=patch_size, embed_dim=d_model, in_channels=3)

        self.num_patches = (self.height // patch_size) * (self.width // patch_size)

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, d_model)
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches+1, d_model)
        )



        def set_layer(i):
            if num_layers == 6:
                if i == 0 or i == 1:
                    return 1
                if i == 2 or i == 3:
                    return 2
                if i == 4:
                    return 3
                else:
                    return None
            
            if num_layers == 8:
                if i == 0 or i == 1:
                    return 1
                if i == 2 or i == 3:
                    return 2
                if i == 4 or i == 6:
                    return 3
                else:
                    return None
        
        self.layers = nn.ModuleList([
            TransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                num_patches=self.num_patches,
                layer_index=i,
                local_window = set_layer(i)  # (4x4 patch region)
                #local_window=None  # (4x4 patch region)
            ) for i in range(num_layers)
        ])

        #classification head
        self.norm = nn.LayerNorm(d_model, eps=1e-06)
        self.fc = nn.Linear(d_model, num_classes)

        self.dropout = nn.Dropout(dropout)
        self.classifier_dropout = nn.Dropout(dropout)

        self.apply(_init_vit_weights)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        batch_size = x.size(0)
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        #(B, Num_patches, D)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # (B, 1, D)
        x = torch.cat([cls_tokens, x], dim=1) #(B, patches+1, D)

        x = x + self.pos_embed #(B, patches+1, D)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # Layer norm before classification
        cls_token = x[:, 0]  # Extract cls_token (B, D)
        cls_token = self.classifier_dropout(cls_token) 
        out = self.fc(cls_token)  # (B, num_classes)

        return out

class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, num_patches, layer_index, local_window=None):
        super().__init__()
        dropout = 0.1
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.attn_dim = d_model
        self.num_heads = num_heads
        self.num_patches = num_patches

        # Official MultiheadAttention (batch_first=True for (B, seq_len, D))
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True  # (B, seq_len, D) instead of (seq_len, B, D)
        )

        # Normalization
        self.norm1 = nn.LayerNorm(d_model, eps=1e-06)
        self.dropout1 = nn.Dropout(dropout)

        # Feed-forward
        self.fc1 = nn.Linear(d_model, d_model*4)
        self.fc2 = nn.Linear(d_model*4, d_model)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-06)
        self.dropout2 = nn.Dropout(dropout)

        self.attn_dropout = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

        if local_window is not None:
            self.register_buffer("attn_mask", self._build_local_mask(local_window))
        else:
            self.attn_mask = None


    def forward(self, x):
        # Pre-norm
        x_norm = self.norm1(x)



        attn_mask = self.attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask.to(x.device)  # ensure it's on the correct device

        #PyTorch's MultiheadAttention
        attn_output, _ = self.self_attn(
            query=x_norm,  # (B, seq_len, D)
            key=x_norm,    # (B, seq_len, D)
            value=x_norm,  # (B, seq_len, D)
            need_weights=False,
            attn_mask=attn_mask
        )

        # Residual connection
        x = x + self.dropout1(attn_output)

        # Feed-forward
        ff_input = self.norm2(x)
        x = x + self.dropout2(self.fc2(self.dropout_ffn(F.gelu(self.fc1(ff_input)))))

        return x
    
    def _build_local_mask(self, window_radius):
        grid_size = int(np.sqrt(self.num_patches))
        mask = torch.full((self.num_patches + 1, self.num_patches + 1), float('-inf'))  # +1 for CLS

        for i in range(self.num_patches):
            y_i, x_i = divmod(i, grid_size)
            for j in range(self.num_patches):
                y_j, x_j = divmod(j, grid_size)
                if abs(x_i - x_j) <= window_radius and abs(y_i - y_j) <= window_radius:
                    mask[i + 1, j + 1] = 0  # patch-to-patch

        mask[0, :] = 0  # CLS attends to all
        mask[:, 0] = 0  # All attend to CLS

        return mask  # shape: (seq_len, seq_len), dtype: float
    
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, embed_dim, in_channels=3):
        super().__init__()

        '''
        Instead of a single conv, we do multiple layers of early convolutions.
        '''

        #a conv is an easy way to get a linear projection of image patches
        #you could do the same by rearanging a tensor of image sections and then using an nn.linear layer
        #I think they are mathematically equivalent
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)  # (B, D, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, D)
        return x
    
def _init_vit_weights(m: nn.Module, std: float = 0.02):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.trunc_normal_(m.weight, std=std)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)