"""
AudioPreprocessor: Feature extraction and preprocessing for audio data
Handles Wav2Vec2 feature extraction, padding, and attention masks
"""
import torch
import numpy as np
import librosa
from transformers import AutoFeatureExtractor
from torch.nn.utils.rnn import pad_sequence


class AudioPreprocessor:
    """
    Preprocesses raw audio for Wav2Vec2 encoder.
    Handles feature extraction, padding/truncation, and attention mask generation.
    """
    
    def __init__(self, model_checkpoint="facebook/wav2vec2-base", max_duration=5.0):
        """
        Args:
            model_checkpoint (str): HuggingFace model name for feature extractor
            max_duration (float): Maximum audio duration in seconds (default: 5.0)
        """
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)
        self.max_duration = max_duration
        self.max_length = int(self.feature_extractor.sampling_rate * max_duration)
        
    def preprocess_batch(self, examples):
        """
        Preprocess a batch of audio examples for HuggingFace dataset.map()
        Loads audio from paths using librosa on-demand.
        
        Args:
            examples: Dict with 'audio_path' key containing list of file paths
            
        Returns:
            Dict with 'input_values' and 'attention_mask'
        """
        # Load audio from paths using librosa
        audio_arrays = []
        for path in examples["audio_path"]:
            try:
                audio, sr = librosa.load(
                    path, 
                    sr=self.feature_extractor.sampling_rate, 
                    mono=True
                )
                audio_arrays.append(audio.astype(np.float32))
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")
                # Use silent audio as fallback
                audio_arrays.append(
                    np.zeros(self.feature_extractor.sampling_rate, dtype=np.float32)
                )
        
        inputs = self.feature_extractor(
            audio_arrays,
            sampling_rate=self.feature_extractor.sampling_rate,
            max_length=self.max_length,
            truncation=True,
            padding=False  # We'll pad dynamically in collate
        )
        
        return inputs
    
    def collate_fn(self, batch):
        """
        Custom collate function for DataLoader.
        Pads variable-length audio to same length within batch.
        
        Args:
            batch: List of samples, each with 'input_values' and 'label'
            
        Returns:
            Dict with padded tensors: {
                'input_values': (B, T_max),
                'attention_mask': (B, T_max),
                'labels': (B,)
            }
        """
        # Label mapping for string to int conversion
        label_to_id = {'fake': 0, 'real': 1}
        
        # Extract input_values and labels
        input_values = []
        labels = []
        
        for item in batch:
            # Handle both dict and direct access
            if isinstance(item, dict):
                input_val = item['input_values']
                label = item['label']
            else:
                input_val = item[0]
                label = item[1]
            
            # Convert to tensor if needed
            if not isinstance(input_val, torch.Tensor):
                input_val = torch.tensor(input_val, dtype=torch.float32)
            
            # Convert label to integer if it's a string
            if isinstance(label, str):
                label = label_to_id[label]
            
            input_values.append(input_val)
            labels.append(label)
        
        # Pad sequences to max length in batch
        padded = pad_sequence(input_values, batch_first=True, padding_value=0.0)
        
        # Create attention mask (1 for real audio, 0 for padding)
        lengths = torch.tensor([len(x) for x in input_values], dtype=torch.long)
        max_len = padded.size(1)
        attention_mask = (
            torch.arange(max_len).unsqueeze(0) < lengths.unsqueeze(1)
        ).long()
        
        # Convert labels to tensor
        labels = torch.tensor(labels, dtype=torch.long)
        
        return {
            'input_values': padded,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def __call__(self, examples):
        """Allow using preprocessor as a function"""
        return self.preprocess_batch(examples)
    
    def __repr__(self):
        return (f"AudioPreprocessor(sampling_rate={self.feature_extractor.sampling_rate}, "
                f"max_duration={self.max_duration}s, max_length={self.max_length})")
