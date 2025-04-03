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

        nn.init.trunc_normal_(self.pos_embed, std=0.1)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        self.layers = nn.ModuleList([
            TransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                num_patches=self.num_patches
            ) for _ in range(num_layers)
        ])

        #classification head
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)

        self.dropout = nn.Dropout(dropout)
        self.embedding_dropout = nn.Dropout(dropout)
        self.classifier_dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size = x.size(0)
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        #(B, Num_patches, D)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1) # (B, 1, D)
        x = torch.cat([cls_tokens, x], dim=1) #(B, patches+1, D)

        x = self.embedding_dropout(x)
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
        self.d_k = d_model // num_heads
        self.attn_dim = d_model
        self.num_heads = num_heads

        # Attention parameters
        self.W_q = nn.Linear(self.attn_dim, self.attn_dim)
        self.W_k = nn.Linear(self.attn_dim, self.attn_dim)
        self.W_v = nn.Linear(self.attn_dim, self.attn_dim)
        self.W_o = nn.Linear(d_model, d_model)

        #xavier init is best for attention weights I think
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)

        # Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Feed-forward
        self.fc1 = nn.Linear(d_model, d_model*4)
        self.fc2 = nn.Linear(d_model*4, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.attn_dropout = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, x):
        B, num_patches, D = x.shape

        #pre-norm
        x_norm = self.norm1(x)

        # Multi-head attention
        Q = self.split_heads(self.W_q(x_norm))
        K = self.split_heads(self.W_k(x_norm))
        V = self.split_heads(self.W_v(x_norm))
        attn_out = self.scaled_dot_product_attention(Q, K, V)
        attn_out = self.combine_heads(attn_out)
        attn_out = self.W_o(attn_out)

        x = x + self.dropout1(attn_out)
        ff_input = self.norm2(x)  # Apply norm to residual output
        x = x + self.dropout2(self.fc2(self.dropout_ffn(F.gelu(self.fc1(ff_input)))))

        return x
    
    def split_heads(self, x):
        B, P, D = x.size()
        return x.view(B, P, self.num_heads, self.d_k).permute(0, 2, 1, 3)
    
    def combine_heads(self, x):
        B, H, P, d_k = x.size()
        return x.permute(0, 2, 1, 3).reshape(B, P, -1)
    
    def scaled_dot_product_attention(self, Q, K, V):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn_probs = self.attn_dropout(torch.softmax(attn_scores, dim=-1))
        return torch.matmul(attn_probs, V)



class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, embed_dim, in_channels=3):
        super().__init__()
        #a conv is an easy way to get a linear projection of image patches
        #you could do the same by rearanging a tensor of image sections and then using an nn.linear layer
        #I think they are mathematically equivalent
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)  # (B, D, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, D)
        return x