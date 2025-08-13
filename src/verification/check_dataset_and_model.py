from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.data_loading.aslg_pc12_dataset import ASLGPC12Dataset
from src.models.text_to_gloss_model import TextToGlossModel
from src.utils.logging import get_logger
from src.utils.helpers import get_latest_run_id, load_run_metadata

logger = get_logger(__name__)

def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")
    return device

def check_dataset(tokenized_path: Path, batch_size=4):
    logger.info(f"Loading dataset from {tokenized_path}")
    dataset = ASLGPC12Dataset(data_path=None, tokenized_path=tokenized_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    batch = next(iter(dataloader))
    logger.info(f"Batch keys: {list(batch.keys())}")
    for k, v in batch.items():
        logger.info(f"{k} shape: {v.shape}")
    return batch

def check_model_forward(batch):
    device = get_device()

    model = TextToGlossModel()
    model.to(device)
    model.eval()

    with torch.no_grad():
        inputs = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        logger.info("Running forward pass...")
        outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)

        if hasattr(outputs, "logits"):
            logger.info(f"Output logits shape: {outputs.logits.shape}")
        else:
            logger.info(f"Model output: {outputs}")

def main():
    run_id = get_latest_run_id()
    run_metadata = load_run_metadata(run_id)
    tokenized_path_str = run_metadata["artifacts"].get("aslg_pc12_tokenized.pt")
    if not tokenized_path_str:
        logger.error("Tokenized dataset path not found in run metadata!")
        return
    tokenized_path = Path(tokenized_path_str)

    batch = check_dataset(tokenized_path)
    check_model_forward(batch)
    logger.info("Dataset and model check completed successfully.")

if __name__ == "__main__":
    main()
