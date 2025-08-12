"""
Prepare DataLoader for ASLGPC12 dataset.

This script loads the pre-tokenized dataset saved as `.pt` file
and returns a PyTorch DataLoader suitable for training or evaluation.
"""

from pathlib import Path
import torch
from torch.utils.data import DataLoader
from aslg_pc12_dataset import ASLGPC12Dataset

def get_dataloader(tokenized_path: Path, batch_size=32, shuffle=True, num_workers=2):
    """
    Create a DataLoader for the ASLGPC12 dataset using pre-tokenized data.

    Args:
        tokenized_path (Path): Path to pre-tokenized `.pt` file.
        batch_size (int): Batch size.
        shuffle (bool): Shuffle dataset.
        num_workers (int): Number of data loading workers.

    Returns:
        DataLoader: PyTorch DataLoader instance.
    """
    dataset = ASLGPC12Dataset(data_path=None, tokenized_path=tokenized_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


if __name__ == "__main__":
    tokenized_path = Path("data/processed/aslg_pc12_tokenized.pt")
    batch_size = 32

    dataloader = get_dataloader(tokenized_path, batch_size=batch_size)

    # Quick check
    for batch in dataloader:
        print("Batch keys:", batch.keys())
        print("input_ids shape:", batch['input_ids'].shape)
        print("labels shape:", batch['labels'].shape)
        break
