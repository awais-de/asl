"""
Prepare DataLoader for ASLGPC12 dataset.

Loads the pre-tokenized dataset saved as `.pt` file
and returns a PyTorch DataLoader suitable for training or evaluation.
"""

from pathlib import Path
from torch.utils.data import DataLoader
from src.utils.logging import get_logger
from src.utils.helpers import get_latest_run_id, load_run_metadata
from aslg_pc12_dataset import ASLGPC12Dataset

logger = get_logger(__name__)

def get_dataloader(tokenized_path: Path, batch_size=32, shuffle=True, num_workers=2):
    dataset = ASLGPC12Dataset(data_path=None, tokenized_path=tokenized_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def main():
    run_id = get_latest_run_id()
    run_metadata = load_run_metadata(run_id)
    tokenized_path_str = run_metadata["artifacts"].get("aslg_pc12_tokenized.pt")
    if not tokenized_path_str:
        logger.error("Tokenized dataset path not found in run metadata!")
        return

    tokenized_path = Path(tokenized_path_str)
    batch_size = 32

    dataloader = get_dataloader(tokenized_path, batch_size=batch_size)

    # Quick check on batch data shapes
    for batch in dataloader:
        logger.info(f"Batch keys: {list(batch.keys())}")
        logger.info(f"input_ids shape: {batch['input_ids'].shape}")
        logger.info(f"labels shape: {batch['labels'].shape}")
        break


if __name__ == "__main__":
    main()
