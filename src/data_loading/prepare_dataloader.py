"""
Prepare DataLoader for ASLGPC12 dataset.

Supports both pre-tokenized and on-the-fly tokenization modes.
"""

from pathlib import Path
import argparse
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
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader

def main(tokenized_path=None, data_path=None, tokenizer_name="t5-small", batch_size=32, shuffle=False, num_workers=2):
    parser = argparse.ArgumentParser(description="Prepare DataLoader for ASLGPC12 dataset")
    parser.add_argument("--tokenized_path", type=str, default=None, help="Path to pre-tokenized .pt file")
    parser.add_argument("--data_path", type=str, default=None, help="Path to raw JSONL file")
    parser.add_argument("--tokenizer_name", type=str, default="t5-small", help="Tokenizer for on-the-fly mode")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    #tokenized_path = Path(args.tokenized_path) if args.tokenized_path else None
    print(tokenized_path)
    data_path = Path(args.data_path) if args.data_path else None

    dataloader = get_dataloader(
        data_path=data_path,
        tokenized_path=tokenized_path,
        tokenizer_name=args.tokenizer_name,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers
    )

    # Quick check
    for batch in dataloader:
        print("Batch keys:", batch.keys())
        print("input_ids shape:", batch['input_ids'].shape)
        print("labels shape:", batch['labels'].shape)
        break

if __name__ == "__main__":
    main()
