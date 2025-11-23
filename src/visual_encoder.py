# visual_encoder.py
import torch
import torch.nn as nn
import timm

class VisualEncoder(nn.Module):
    """
    CelebDF video encoder:
      - Frame feature extractor via timm (e.g., Xception or ViT)
      - Bi-LSTM for temporal modeling
      - Attention pooling to produce final video embedding
    """

    def __init__(
        self,
        backbone_name="xception41",
        pretrained=True,
        lstm_hidden=512,
        lstm_layers=1,
        bidirectional=True,
        dropout=0.3,
        frame_chunk_size=32
    ):
        super().__init__()

        # 1) Backbone feature extractor
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=""   # disable default pooling
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.frame_feat_dim = self.backbone.num_features

        self.frame_chunk_size = frame_chunk_size

        # 2) Temporal encoder (Bi-LSTM)
        self.use_lstm = lstm_hidden > 0
        if self.use_lstm:
            lstm_input_dim = self.frame_feat_dim
            self.lstm = nn.LSTM(
                input_size=lstm_input_dim,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if lstm_layers > 1 else 0.0
            )

            lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

            # 3) Attention pooling
            self.attn_fc = nn.Sequential(
                nn.Linear(lstm_out_dim, lstm_out_dim // 2),
                nn.Tanh(),
                nn.Linear(lstm_out_dim // 2, 1)
            )

            self.out_dim = lstm_out_dim

        else:
            # fallback: average all frame features
            self.attn_fc = None
            self.out_dim = self.frame_feat_dim

    def forward(self, x):
        """
        Input shapes:
            (B, T, 3, 224, 224)
        Output:
            (B, out_dim)
        """

        B, T, C, H, W = x.shape
        frames = x.reshape(B*T, C, H, W)

        # chunk processing to avoid OOM
        feats = []
        for i in range(0, frames.size(0), self.frame_chunk_size):
            chunk = frames[i : i + self.frame_chunk_size]
            f = self.backbone.forward_features(chunk)
            f = self.global_pool(f).squeeze(-1).squeeze(-1)  # (chunk, D)
            feats.append(f)

        feats = torch.cat(feats, dim=0)
        feats = feats.view(B, T, -1)  # (B, T, D)

        # Temporal encoder
        if self.use_lstm:
            lstm_out, _ = self.lstm(feats)  # (B, T, H*dir)

            # Attention pooling
            attn_scores = self.attn_fc(lstm_out).squeeze(-1)  # (B, T)
            attn_weights = torch.softmax(attn_scores, dim=1)
            pooled = (lstm_out * attn_weights.unsqueeze(-1)).sum(dim=1)
            return pooled  # (B, out_dim)

        else:
            return feats.mean(dim=1)  # simple average

    def get_out_dim(self):
        return self.out_dim
