import torch
import torch.nn as nn
from torchvision import models


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=1024):
        super(ImageEncoder, self).__init__()

        # Load pretrained ResNet50
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Remove the classification head
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])
        # Output shape: (B, 2048, 1, 1)

        # Project down to desired embedding dimension
        self.proj = nn.Linear(2048, embed_dim)

        # Normalize embeddings for stability
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):  # x: (B, 3, 224, 224)
        feats = self.feature_extractor(x)             # -> (B, 2048, 1, 1)
        feats = feats.view(feats.size(0), -1)        # -> (B, 2048)
        emb = self.proj(feats)                       # -> (B, embed_dim)
        emb = self.norm(emb)
        return emb
