"""Trajectory Diffusion Model with Cross-Attention for Audio"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int = None, n_heads: int = 4, head_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        context_dim = context_dim or query_dim
        inner_dim = n_heads * head_dim
        self.n_heads = n_heads
        self.scale = head_dim ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        M = context.shape[1]
        h = self.n_heads

        q = self.to_q(x).view(B, N, h, -1).permute(0, 2, 1, 3)
        k = self.to_k(context).view(B, M, h, -1).permute(0, 2, 1, 3)
        v = self.to_v(context).view(B, M, h, -1).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(B, N, -1)
        return self.to_out(out)


class CrossAttentionDecoderLayer(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 4, dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = CrossAttention(d_model, d_model, n_heads, d_model // n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, audio_context: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        x = x + self.dropout(self.self_attn(x_norm, x_norm, x_norm)[0])
        x = x + self.cross_attn(self.norm2(x), audio_context)
        x = x + self.ffn(self.norm3(x))
        return x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


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


class TimestepEmbedder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(t.float())


class ConditionEncoder(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.audio_proj = nn.Linear(d_model, d_model)
        self.text_proj = nn.Linear(d_model, d_model)
        self.mesh_proj = nn.Linear(d_model, d_model)
        self.null_audio = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.null_text = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.null_mesh = nn.Parameter(torch.randn(1, d_model) * 0.02)

    def forward(
        self,
        audio_feat: torch.Tensor,
        text_feat: torch.Tensor,
        mesh_feat: torch.Tensor,
        n_frames: int,
        uncond: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = audio_feat.shape[0]
        if uncond:
            return (
                self.null_audio.expand(B, 1, -1),
                self.null_text.expand(B, -1),
                self.null_mesh.expand(B, -1),
            )
        return (
            self.audio_proj(audio_feat),
            self.text_proj(text_feat),
            self.mesh_proj(mesh_feat),
        )


class TrajectoryDiffusion(nn.Module):
    def __init__(
        self,
        traj_dim: int = 11,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 8,
        n_frames: int = 360,
        dropout: float = 0.1,
        n_cond_tokens: int = 3,
    ):
        super().__init__()
        self.traj_dim = traj_dim
        self.d_model = d_model
        self.n_frames = n_frames
        self.n_cond_tokens = n_cond_tokens

        self.traj_proj = nn.Linear(traj_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=n_frames + n_cond_tokens + 10)
        self.time_embed = TimestepEmbedder(d_model)

        self.layers = nn.ModuleList([
            CrossAttentionDecoderLayer(d_model, n_heads, d_model * 4, dropout)
            for _ in range(n_layers)
        ])

        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, traj_dim)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        audio_feat: torch.Tensor,
        text_feat: torch.Tensor,
        mesh_feat: torch.Tensor,
    ) -> torch.Tensor:
        B, N, _ = x_t.shape
        motion_emb = self.traj_proj(x_t)

        # Interpolate audio to match trajectory frames for temporal alignment
        if audio_feat.shape[1] != N:
            audio_feat = F.interpolate(
                audio_feat.transpose(1, 2), size=N, mode='linear', align_corners=False
            ).transpose(1, 2)

        time_token = self.time_embed(t).unsqueeze(1)
        text_token = text_feat.unsqueeze(1)
        mesh_token = mesh_feat.unsqueeze(1)

        xseq = torch.cat([time_token, text_token, mesh_token, motion_emb], dim=1)
        xseq = self.pos_encoder(xseq)

        for layer in self.layers:
            xseq = layer(xseq, audio_feat)

        output = xseq[:, self.n_cond_tokens:]
        return self.output_proj(self.output_norm(output))
