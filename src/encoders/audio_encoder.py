"""
AudioEncoder: Encode preprocessed audio to fixed-size embeddings for multimodal fusion
Uses Wav2Vec2 with optional projection layer
"""
import torch
import torch.nn as nn
from transformers import AutoModel


class AudioEncoder(nn.Module):
    """
    Encodes preprocessed audio input_values (B, T) into fixed-size embeddings (B, D).
    
    Designed for multimodal fusion:
    - Takes preprocessed audio features as input
    - Outputs fixed-size embeddings that can be concatenated/fused with image/video
    - Supports freezing feature extractor to save memory (8GB GPU friendly)
    - Optional projection layer to match other modality dimensions
    """
    
    def __init__(
        self,
        model_checkpoint="facebook/wav2vec2-base",
        projection_dim=512,
        freeze_feature_extractor=True,
        freeze_encoder=False
    ):
        """
        Args:
            model_checkpoint (str): HuggingFace Wav2Vec2 model name
            projection_dim (int): Output embedding dimension for fusion (None = use hidden_size)
            freeze_feature_extractor (bool): Freeze CNN feature extractor (saves memory)
            freeze_encoder (bool): Freeze entire encoder (only train projection)
        """
        super().__init__()
        
        # Load pretrained Wav2Vec2
        self.model = AutoModel.from_pretrained(model_checkpoint)
        self.hidden_size = self.model.config.hidden_size  # 768 for base
        self.out_dim = projection_dim if projection_dim else self.hidden_size
        
        # Optional projection for fusion (match image/video embedding dims)
        if projection_dim:
            self.projector = nn.Sequential(
                nn.Linear(self.hidden_size, projection_dim),
                nn.LayerNorm(projection_dim),
                nn.ReLU()
            )
        else:
            self.projector = None
        
        # Memory-saving: freeze feature extractor (CNN layers)
        if freeze_feature_extractor and hasattr(self.model, "feature_extractor"):
            for param in self.model.feature_extractor.parameters():
                param.requires_grad = False
        
        # Optional: freeze entire encoder (only train projection layer)
        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def forward(self, input_values, attention_mask=None):
        """
        Forward pass: audio features -> embeddings
        
        Args:
            input_values: torch.FloatTensor (B, T) - preprocessed audio waveform
            attention_mask: torch.LongTensor (B, T) - attention mask (optional)
            
        Returns:
            embeddings: torch.FloatTensor (B, out_dim) - fixed-size audio embeddings
        """
        # Pass through Wav2Vec2
        outputs = self.model(
            input_values,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get last hidden state: (B, time_steps, hidden_size)
        hidden = outputs.last_hidden_state
        
        # Pooling: masked mean if attention_mask provided, else simple mean
        if attention_mask is not None:
            # Wav2Vec2 downsamples the input, so we need to get the attention mask
            # that matches the output sequence length from the model
            # Use the model's built-in method to compute it
            sub_attention_mask = self.model._get_feature_vector_attention_mask(
                hidden.shape[1], attention_mask
            )
            
            # Expand mask to match hidden dimensions
            mask = sub_attention_mask.unsqueeze(-1).to(hidden.dtype)  # (B, T_hidden, 1)
            
            # Masked sum and normalize by actual lengths
            masked_hidden = hidden * mask
            summed = masked_hidden.sum(dim=1)  # (B, hidden_size)
            lengths = mask.sum(dim=1).clamp(min=1e-9)  # (B, 1)
            pooled = summed / lengths
        else:
            # Simple mean pooling over time
            pooled = hidden.mean(dim=1)  # (B, hidden_size)
        
        # Optional projection for fusion
        if self.projector is not None:
            pooled = self.projector(pooled)  # (B, projection_dim)
        
        return pooled
    
    def get_embedding_dim(self):
        """Return output embedding dimension"""
        return self.out_dim
    
    def __repr__(self):
        frozen_status = []
        if hasattr(self.model, "feature_extractor"):
            fe_frozen = not next(self.model.feature_extractor.parameters()).requires_grad
            frozen_status.append(f"feature_extractor_frozen={fe_frozen}")
        
        proj_status = f"projection={self.out_dim}" if self.projector else "no_projection"
        
        return (f"AudioEncoder(hidden={self.hidden_size}, out_dim={self.out_dim}, "
                f"{proj_status}, {', '.join(frozen_status)})")


# Example usage for multimodal fusion:
# 
# # 1. Create encoder
# audio_encoder = AudioEncoder(
#     projection_dim=512,  # Match image encoder dim
#     freeze_feature_extractor=True  # Save memory
# )
# 
# # 2. Forward pass
# audio_embeddings = audio_encoder(input_values, attention_mask)  # (B, 512)
# 
# # 3. Fuse with other modalities
# # fused = torch.cat([audio_embeddings, image_embeddings, video_embeddings], dim=-1)
