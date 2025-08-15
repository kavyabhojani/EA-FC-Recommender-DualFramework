import torch
import torch.nn as nn
import torch.nn.functional as F

class NCF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=64, mlp_dims=(128,64)):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        mlp_layers = []
        in_dim = emb_dim * 2
        for d in mlp_dims:
            mlp_layers += [nn.Linear(in_dim, d), nn.ReLU()]
            in_dim = d
        self.mlp = nn.Sequential(*mlp_layers)
        self.out = nn.Linear(in_dim, 1)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, users, items):
        u = self.user_emb(users)
        v = self.item_emb(items)
        x = torch.cat([u, v], dim=-1)
        x = self.mlp(x)
        logits = self.out(x).squeeze(-1)
        return logits

def bce_loss(logits, labels):
    return F.binary_cross_entropy_with_logits(logits, labels)
