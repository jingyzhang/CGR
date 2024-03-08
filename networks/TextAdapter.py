import torch
import torch.nn as nn

class MultiTextAdapter(nn.Module):
    def __init__(self, text_dim=768, hidden_dim=None,gamma_init_value=1e-4, num_classes=3):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = text_dim // 2
        self.text_dim = text_dim
        self.gamma = nn.ParameterList([nn.Parameter(torch.ones(text_dim) * gamma_init_value, requires_grad=True) for _ in range(num_classes)])
        self.fc = nn.ModuleList([nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, text_dim)
        ) for _ in range(num_classes)])
       

    def forward(self, text_embed, domain_label):
        fc = self.fc[domain_label]
        gamma = self.gamma[domain_label]
        text_embed_after = fc(text_embed)
        text_embed = text_embed + gamma * text_embed_after
        return text_embed