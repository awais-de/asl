from pathlib import Path
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)

class ASLGPC12Dataset(Dataset):
    def __init__(self, data_path: Path = None, tokenizer_name='t5-small', 
                 tokenized_path: Path = None, max_length=128):
        """
        Initialize dataset.

        Args:
            data_path (Path): Path to raw JSONL dataset (used only for on-the-fly tokenization).
            tokenizer_name (str): HuggingFace tokenizer name.
            tokenized_path (Path): If provided, load tokenized data from this .pt file instead of raw.
            max_length (int): Max sequence length for tokenization.
        """
        self.max_length = max_length
        self.tokenized_path = tokenized_path

        if tokenized_path and tokenized_path.exists():
            # Load pre-tokenized data (fast loading)
            logging.info(f"Loading pre-tokenized dataset from {tokenized_path}")
            self.data = torch.load(tokenized_path)  # Expecting a list of dicts
            self.use_tokenized = True
            logging.info(f"Loaded {len(self.data)} pre-tokenized samples")
        else:
            # On-the-fly tokenization mode
            logging.info(f"Loading raw dataset from {data_path} with on-the-fly tokenization")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.data_path = data_path
            self.samples = self._load_raw_data()
            self.use_tokenized = False

    def _load_raw_data(self):
        with self.data_path.open('r', encoding='utf-8') as f:
            samples = [json.loads(line.strip()) for line in f]
        logging.info(f"Loaded {len(samples)} raw samples")
        return samples

    def __len__(self):
        if self.use_tokenized:
            return len(self.data)
        else:
            return len(self.samples)

    def __getitem__(self, idx):
        if self.use_tokenized:
             return {
            'input_ids': self.data['input_ids'][idx],
            'attention_mask': self.data['attention_mask'][idx],
            'labels': self.data['labels'][idx]
        }
        else:
            # Tokenize on the fly
            sample = self.samples[idx]
            text = sample.get('text', "")
            gloss = sample.get('gloss', "")

            inputs = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            targets = self.tokenizer(
                gloss,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': inputs.input_ids.squeeze(),
                'attention_mask': inputs.attention_mask.squeeze(),
                'labels': targets.input_ids.squeeze()
            }
