"""Text Encoder - CLIP frozen"""
import torch
import torch.nn as nn
import clip


class TextEncoder(nn.Module):
    def __init__(self, d_model: int = 256, device: str = "cuda"):
        super().__init__()
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        self.clip_model.eval()
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.proj = nn.Linear(512, d_model)

    def forward(self, texts: list[str]) -> torch.Tensor:
        """
        Args:
            texts: list of B text descriptions
        Returns:
            [B, d_model] text features
        """
        device = self.proj.weight.device
        tokens = clip.tokenize(texts, truncate=True).to(device)
        with torch.no_grad():
            features = self.clip_model.encode_text(tokens).float()
        return self.proj(features)  # [B, d_model]
