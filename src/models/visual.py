# src/models/visual.py

import torch
import torch.nn as nn
import timm


class VisualEncoder(nn.Module):
    """
    Frame-level backbone (Xception via timm) + optional temporal LSTM + attention.
    
    Input:
        x: (B,T,C,H,W) video OR (B,C,H,W) image
    Output:
        (B, out_dim)
    """

    def __init__(
        self,
        backbone_name="xception",
        pretrained=True,
        lstm_hidden=512,
        lstm_layers=1,
        bidirectional=True,
        dropout=0.3,
        frame_chunk_size=32,
    ):
        super().__init__()

        # Create backbone — no classifier head
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0
        )
        self.frame_feat_dim = self.backbone.num_features
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.frame_chunk_size = frame_chunk_size

        # LSTM setup
        self.use_lstm = lstm_hidden > 0
        if self.use_lstm:
            self.lstm = nn.LSTM(
                input_size=self.frame_feat_dim,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if lstm_layers > 1 else 0.0,
            )
            lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

            # Attention pooling
            self.attn_fc = nn.Sequential(
                nn.Linear(lstm_out_dim, lstm_out_dim // 2),
                nn.Tanh(),
                nn.Linear(lstm_out_dim // 2, 1),
            )

            self.out_dim = lstm_out_dim

        else:
            # No LSTM → just return pooled frame features
            self.out_dim = self.frame_feat_dim
            self.attn_fc = None

    def _encode_frames(self, frames):
        """
        frames: (N, C, H, W)
        return: (N, D) feature vectors
        """

        feats = self.backbone.forward_features(frames)

        # If backbone returns spatial map: GAP it
        if feats.dim() == 4:  # (N, C, H, W)
            feats = self.global_pool(feats)  # (N, C, 1, 1)
            feats = feats.flatten(1)         # (N, C)

        return feats  # (N, D)

    def forward(self, x):
        # Single image mode
        if x.dim() == 4:  # (B,C,H,W)
            feats = self._encode_frames(x)
            return feats

        # Video mode
        if x.dim() != 5:
            raise ValueError("VisualEncoder expects (B," \
            "" \
            "T,C,H,W) or (B,C,H,W)")

        B, T, C, H, W = x.shape
        frames = x.view(B * T, C, H, W)

        # Chunk processing to avoid OOM
        feats_list = []
        for i in range(0, frames.size(0), self.frame_chunk_size):
            chunk = frames[i:i + self.frame_chunk_size]
            feats_list.append(self._encode_frames(chunk))

        frame_feats = torch.cat(feats_list, dim=0)  # (B*T, D)
        frame_feats = frame_feats.view(B, T, -1)    # (B, T, D)

        # LSTM branch
        if self.use_lstm:
            lstm_out, _ = self.lstm(frame_feats)
            attn_scores = self.attn_fc(lstm_out)        # (B, T, 1)
            attn_weights = torch.softmax(attn_scores, dim=1)
            pooled = (lstm_out * attn_weights).sum(dim=1)  # (B, out_dim)
            return pooled

        # Mean over time
        return frame_feats.mean(dim=1)

    def get_out_dim(self):
        return self.out_dim

    def __repr__(self):
        return f"VisualEncoderr(backbone={self.backbone.__class__.__name__}, frame_feat_dim={self.frame_feat_dim}, out_dim={self.out_dim})"
