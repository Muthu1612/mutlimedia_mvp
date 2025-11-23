# image_encoder.py
import torch
import torch.nn as nn
import timm

class ImageEncoder(nn.Module):
    """
    Encoder for single face images.
    Outputs a fixed-length embedding vector.
    """

    def __init__(
        self,
        backbone="efficientnet_b3",
        pretrained=True,
        out_dim=512
    ):
        super().__init__()

        # Load backbone
        self.model = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,   # no classifier
            global_pool=""   # we will pool manually
        )

        self.frame_feat_dim = self.model.num_features

        # global pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # projection head (optional)
        self.proj = nn.Sequential(
            nn.Linear(self.frame_feat_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(out_dim, out_dim)
        )

        self.out_dim = out_dim

    def forward(self, x):
        """
        x: (B, 3, 224, 224)
        returns: (B, out_dim)
        """
        feats = self.model.forward_features(x)   # (B, C, H, W)
        feats = self.pool(feats).squeeze(-1).squeeze(-1)  # (B, C)
        feats = self.proj(feats)  # (B, out_dim)
        return feats

    def get_out_dim(self):
        return self.out_dim
