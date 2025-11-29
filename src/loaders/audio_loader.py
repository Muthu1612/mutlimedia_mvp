"""
AudioLoader: Load raw audio data from folder structure (lazy loading)
Supports train/val/test splits with fake/real labels
"""
import os
from pathlib import Path
from datasets import Dataset, DatasetDict


class AudioLoader:
    """
    Loads audio file paths from organized folder structure.
    Audio is loaded lazily during preprocessing to avoid loading 69k files upfront.
    Expected structure: data_dir/{train,validation,test}/{fake,real}/*.wav
    
    Returns HuggingFace Dataset with 'audio_path' and 'label' columns.
    """
    
    def __init__(self, data_dir, target_sr=16000):
        """
        Args:
            data_dir (str): Path to audio data directory
            target_sr (int): Target sampling rate (default: 16000 for Wav2Vec2)
        """
        self.data_dir = Path(data_dir)
        self.target_sr = target_sr
        self.dataset = None
        
    def load(self):
        """
        Load audio dataset by scanning directory structure (paths only - lazy loading).
        Automatically handles train/val/test splits if folders exist.
        
        Returns:
            DatasetDict with train/validation/test splits
        """
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {self.data_dir}")
        
        # Map folder names to split names
        split_mapping = {
            'training': 'train',
            'validation': 'validation', 
            'testing': 'test'
        }
        
        dataset_dict = {}
        
        for folder_name, split_name in split_mapping.items():
            split_dir = self.data_dir / folder_name
            if not split_dir.exists():
                continue
            
            print(f"Scanning {split_name} split...")
            
            # Collect audio file paths for this split
            audio_paths = []
            labels = []
            
            # Check both 'fake' and 'real' subdirectories
            for label_name in ['fake', 'real']:
                label_dir = split_dir / label_name
                if not label_dir.exists():
                    continue
                    
                # Find all .wav files (including those with complex extensions)
                wav_files = list(label_dir.glob('*.wav'))
                print(f"  Found {len(wav_files)} {label_name} files")
                
                # Store paths only - audio loaded later during preprocessing
                audio_paths.extend([str(f) for f in wav_files])
                labels.extend([label_name] * len(wav_files))
            
            if audio_paths:
                # Create dataset for this split with just paths
                dataset = Dataset.from_dict({
                    'audio_path': audio_paths,
                    'label': labels
                })
                
                dataset_dict[split_name] = dataset
                print(f"  Registered {len(dataset)} examples")
        
        if not dataset_dict:
            raise ValueError(f"No audio files found in {self.data_dir}")
        
        self.dataset = DatasetDict(dataset_dict)
        return self.dataset
    
    def get_splits(self):
        """Get available split names (e.g., ['train', 'validation', 'test'])"""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        return list(self.dataset.keys())
    
    def get_label_mapping(self):
        """
        Get label to id mapping from dataset.
        Returns dict like {'fake': 0, 'real': 1}
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        
        # Get labels from any split (they should be consistent)
        split = list(self.dataset.keys())[0]
        label_names = self.dataset[split].features["label"].names
        return {name: idx for idx, name in enumerate(label_names)}
    
    def __repr__(self):
        if self.dataset:
            splits_info = {k: len(v) for k, v in self.dataset.items()}
            return f"AudioLoader(splits={splits_info}, target_sr={self.target_sr})"
        return f"AudioLoader(data_dir={self.data_dir}, target_sr={self.target_sr})"
