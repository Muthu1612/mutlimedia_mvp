import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionModel(nn.Module):
    def __init__(self, 
                 video_dim=1024,
                 image_dim=1024,
                 audio_dim=768,
                 proj_dim=512,
                 hidden_dim=512,
                 num_classes=2):
        super().__init__()

        # 1. Projection layers (learned dimensionality reduction)
        self.video_proj = nn.Linear(video_dim, proj_dim)
        self.image_proj = nn.Linear(image_dim, proj_dim)
        self.audio_proj = nn.Linear(audio_dim, proj_dim)

        # 2. Fusion MLP
        fusion_input_dim = proj_dim * 3  # concat video + image + audio

        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, video_emb, image_emb, audio_emb):
        # 1. Normalize each embedding (keeps information)
        v = F.normalize(video_emb, dim=1)
        i = F.normalize(image_emb, dim=1)
        a = F.normalize(audio_emb, dim=1)

        # 2. Project to shared dim
        v = self.video_proj(v)
        i = self.image_proj(i)
        a = self.audio_proj(a)

        # 3. Fuse — simplest stable method: concatenation
        fused = torch.cat([v, i, a], dim=1)

        # 4. Classify
        logits = self.classifier(fused)

        return logits
