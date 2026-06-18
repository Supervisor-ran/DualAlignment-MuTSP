import torch.nn as nn
import torch

class CrossModalAttention(nn.Module):
    def __init__(self, input_dims, embed_dim):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Linear(dim, embed_dim) for dim in input_dims
        ])
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, modal_inputs):
        modal_inputs = [x.to(self.device) for x in modal_inputs]
        projected = [proj(x) for proj, x in zip(self.projections, modal_inputs)]
        stacked = torch.stack(projected, dim=1)
        M, B, T, E = stacked.shape[1], stacked.shape[0], stacked.shape[2], stacked.shape[3]
        fusion_outputs = []
        for t in range(T):
            x_t = stacked[:, :, t, :]
            q = self.query(x_t[:, 0:1, :])
            k = self.key(x_t)
            v = self.value(x_t)
            scores = self.softmax((q @ k.transpose(-2, -1)) / E**0.5)
            fused = (scores @ v).squeeze(1)
            fusion_outputs.append(fused)
        return torch.stack(fusion_outputs, dim=1)