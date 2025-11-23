import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor, Wav2Vec2Model

class AudioEncoder(nn.Module):
    """
    Audio encoder using pretrained Wav2Vec2.
    Input:  wav (B, L) float32 normalized [-1, 1]
    Output: (B, D) embedding
    """

    def __init__(self, model_name="facebook/wav2vec2-base-960h"):
        super().__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.out_dim = self.model.config.hidden_size

    def forward(self, wav):
        B = wav.size(0)

        # Convert each audio sample to numpy for processor
        wav_list = [wav[i].detach().cpu().numpy() for i in range(B)]

        # Processor: handles padding
        inputs = self.processor(
            wav_list,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        input_values = inputs["input_values"].to(wav.device)

        # Optional attention mask (may not exist)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(wav.device)

        # Forward pass
        outputs = self.model(
            input_values=input_values,
            attention_mask=attention_mask
        )

        # Mean pooling over time
        embeddings = outputs.last_hidden_state.mean(dim=1)

        return embeddings
