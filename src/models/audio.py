import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from transformers import Wav2Vec2Config, Wav2Vec2Model


class AudioEncoder(nn.Module):
    """
    Audio encoder using Wav2Vec2.
    Expects raw audio waveform at 16kHz: (B, L)
    Output: (B, D)
    """

    def __init__(self, model_name="facebook/wav2vec2-base"):
        super().__init__()
        self.configuration = Wav2Vec2Config()
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model(self.configuration)
        self.out_dim = self.model.config.hidden_size

    def forward(self, wav):
        """
        wav: float tensor (B, L) at 16kHz
        Returns: (B, hidden_dim) embeddings
        """
        # Wav2Vec2 expects normalized audio in range [-1, 1]
        # Input should already be in this range for raw waveforms
        
        # Process each sample in batch separately to avoid dimension issues
        batch_size = wav.shape[0]
        wav_cpu = wav.cpu().numpy()
        
        # Process batch - convert to list of arrays for processor
        inputs = self.processor(
            [wav_cpu[i] for i in range(batch_size)],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        # Move to same device as input
        input_values = inputs["input_values"].to(wav.device)
        
        # Create attention mask if not provided
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(wav.device)

        outputs = self.model(
            input_values=input_values,
            attention_mask=attention_mask
        )

        # Mean pooling over time dimension
        return outputs.last_hidden_state.mean(dim=1)
