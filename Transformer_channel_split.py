import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class ViT(nn.Module):
    def __init__(self, d_model, num_layers, num_heads, num_classes, patch_size=32):
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


        
        self.layers = nn.ModuleList([
            TransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                num_patches=self.num_patches
            ) for _ in range(num_layers)
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
    def __init__(self, d_model, num_heads, num_patches):
        super().__init__()
        dropout = 0.1
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # --- Split parameters ---
        self.split_ratio = 0.5  # 50% attention, 50% conv
        self.d_attn = int(d_model * self.split_ratio)
        self.d_conv = d_model - self.d_attn

        self.num_heads = num_heads
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.d_attn,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Conv over token embeddings (1D conv over sequence)
        self.conv_proj = nn.Sequential(
            nn.Conv1d(self.d_conv, self.d_conv, kernel_size=3, padding=1, groups=1),
            nn.GELU(),
            nn.Conv1d(self.d_conv, self.d_conv, kernel_size=3, padding=1),
        )

        self.pre_split_fc = nn.Linear(d_model, d_model)
        self.post_split_fc = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-06)
        self.dropout1 = nn.Dropout(dropout)

        self.fc1 = nn.Linear(d_model, d_model * 4)
        self.fc2 = nn.Linear(d_model * 4, d_model)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-06)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, x):
        # Pre split MLP layer
        x_pre_split = self.pre_split_fc(x)

        # norm and split
        x_norm = self.norm1(x_pre_split)  # (B, seq_len, D)
        x_attn, x_conv = torch.split(x_norm, [self.d_attn, self.d_conv], dim=2)

        # Attention path
        attn_out, _ = self.self_attn(x_attn, x_attn, x_attn, need_weights=False)

        # Conv path: reshape to (B, C, seq_len) for Conv1d
        conv_in = x_conv.transpose(1, 2)
        conv_out = self.conv_proj(conv_in)
        conv_out = conv_out.transpose(1, 2)  # Back to (B, seq_len, C)

        # Recombine and post split linear layer
        x_combined = self.post_split_fc(torch.cat([attn_out, conv_out], dim=2))
        x = x_pre_split + self.dropout1(x_combined)

        # FFN
        ff_input = self.norm2(x)
        x = x + self.dropout2(self.fc2(self.dropout_ffn(F.gelu(self.fc1(ff_input)))))
        return x

    
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