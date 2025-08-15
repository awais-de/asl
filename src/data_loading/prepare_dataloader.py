"""
Prepare DataLoader for ASLGPC12 dataset.

Notebook-friendly version: only uses function arguments, no CLI parsing.
"""

from pathlib import Path
from torch.utils.data import DataLoader
from src.data_loading.aslg_pc12_dataset import ASLGPC12Dataset

def get_dataloader(data_path: Path = None, tokenized_path: Path = None, tokenizer_name='t5-small',
                   batch_size=32, shuffle=True, num_workers=2):
    """
    Create a DataLoader for the ASLGPC12 dataset.

    Args:
        data_path (Path): Path to raw JSONL file (for on-the-fly tokenization).
        tokenized_path (Path): Path to pre-tokenized `.pt` file.
        tokenizer_name (str): Tokenizer to use for on-the-fly mode.
        batch_size (int): Batch size.
        shuffle (bool): Shuffle dataset.
        num_workers (int): Number of data loading workers.

    Returns:
        DataLoader: PyTorch DataLoader instance.
    """
    dataset = ASLGPC12Dataset(
        data_path=data_path,
        tokenizer_name=tokenizer_name,
        tokenized_path=tokenized_path
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

def main(tokenized_path=None, data_path=None, tokenizer_name="t5-small",
         batch_size=32, shuffle=False, num_workers=2):
    """
    Prepares a DataLoader for ASLGPC12 dataset.

    All arguments are passed directly; no command-line parsing.
    """
    tokenized_path = Path(tokenized_path) if tokenized_path else None
    data_path = Path(data_path) if data_path else None

    dataloader = get_dataloader(
        data_path=data_path,
        tokenized_path=tokenized_path,
        tokenizer_name=tokenizer_name,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    # Quick check
    for batch in dataloader:
        print("Batch keys:", batch.keys())
        print("input_ids shape:", batch['input_ids'].shape)
        print("labels shape:", batch['labels'].shape)
        break

    return dataloader
