import torch
import torch.nn as nn

class MiniViT(nn.Module):
    input_size = 128

    def __init__(self, num_classes=6, img_size=128, patch_size=16, in_c=3, dim=192, depth=6,
                 heads=6, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        assert img_size % patch_size == 0
        n_tokens = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_c, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens + 1, dim))
        self.pos_drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
                                           dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)      # B, N, D
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        x = self.encoder(self.pos_drop(x))
        return self.head(self.norm(x[:, 0]))
