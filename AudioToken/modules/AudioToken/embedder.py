import torch.nn as nn
# from modules.FGA.atten import Atten
from ..FGA.atten import Atten

class FGAEmbedder(nn.Module):
    def __init__(self, input_size=768*3, output_size=768):
        super(FGAEmbedder, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, output_size)
        self.gelu = nn.GELU()
        self.fga = Atten(util_e=[output_size], pairwise_flag=False)

    def forward(self, audio_embs):
        audio_embs = self.fc1(audio_embs)
        audio_embs = self.gelu(audio_embs)
        audio_embs = self.fc2(audio_embs)
        attend = self.fga([audio_embs])[0]
        return attend

class Clap_Embedder(nn.Module):
    def __init__(self, input_size=512, output_size=768):
        super(Clap_Embedder, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, output_size)
        self.gelu = nn.GELU()
        self.fga = Atten(util_e=[output_size], pairwise_flag=False)

    def forward(self, audio_embs):
        audio_embs = self.fc1(audio_embs)
        audio_embs = self.gelu(audio_embs)
        audio_embs = self.fc2(audio_embs)
        attend = self.fga([audio_embs])[0]
        return attend


import torch
import torch.nn as nn
import torch.nn.functional as F

class Clap_Embedder2(nn.Module):
    def __init__(self, input_size=512, hidden_size=2304, output_size=768):
        super(Clap_Embedder2, self).__init__()

        # ===== Project CLAP → higher dim (match FGA capacity) =====
        self.input_proj = nn.Linear(input_size, hidden_size)

        # ===== Single Residual Block (enough for CLAP) =====
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, hidden_size)

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.05)

        # ===== Output projection to CLIP space =====
        self.fc_out = nn.Linear(hidden_size, output_size)

        # ===== Attention refinement (same as your pipeline) =====
        self.fga = Atten(util_e=[output_size], pairwise_flag=False)

    def forward(self, audio_embs):
        # ===== Step 1: Expand CLAP embedding =====
        x = self.input_proj(audio_embs)   # [B, 512] → [B, 2304]

        # ===== Residual Block =====
        residual = x
        x = self.norm(x)
        x = self.fc(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = x + residual

        # ===== Project to CLIP token space =====
        x = self.fc_out(x)   # → [B, 768]

        # ===== Normalize (CRITICAL for CLIP alignment) =====
        x = F.normalize(x, dim=-1)

        # ===== Attention refinement =====
        x = self.fga([x])[0]

        return x