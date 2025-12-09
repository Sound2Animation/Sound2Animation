"""Audio Encoder - Multi-head attention for mel spectrogram"""
import math
import torch
import torch.nn as nn


class AudioEncoder(nn.Module):
    def __init__(self, n_mels: int = 128, d_model: int = 256, n_heads: int = 4, n_layers: int = 4):
        super().__init__()
        self.mel_proj = nn.Linear(n_mels, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=1000)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, dropout=0.1, batch_first=True)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: [B, n_mels, T] mel spectrogram
        Returns:
            [B, T, d_model] audio features (no downsampling)
        """
        x = mel.permute(0, 2, 1)  # [B, T, n_mels]
        x = self.mel_proj(x)  # [B, T, d_model]
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.shape[1]]
        return self.dropout(x)
