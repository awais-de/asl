from pathlib import Path
import json
from transformers import AutoTokenizer
import torch

from src.utils.logging import get_logger
from src.utils.helpers import (
    get_latest_run_id,
    load_run_metadata,
    save_run_metadata,
    Artifact,
    add_artifact_to_metadata,
)

logger = get_logger(__name__)

class Tokenizer:
    def __init__(self, model_name='t5-small'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_data(self, input_file: Path, gloss_key='gloss', text_key='text'):
        tokenized_samples = []
        logger.info(f"Tokenizing data from {input_file}")

        with input_file.open('r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                gloss = sample.get(gloss_key, "")
                text = sample.get(text_key, "")

                input_encodings = self.tokenizer(
                    text,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                target_encodings = self.tokenizer(
                    gloss,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )

                tokenized_samples.append({
                    'input_ids': input_encodings.input_ids.squeeze(0),
                    'attention_mask': input_encodings.attention_mask.squeeze(0),
                    'labels': target_encodings.input_ids.squeeze(0)
                })

        logger.info(f"Tokenized {len(tokenized_samples)} samples")
        return tokenized_samples


def main():
    run_id = get_latest_run_id()
    run_metadata = load_run_metadata(run_id)

    # Get preprocessed ASLG jsonl path dynamically from artifacts
    aslg_cleaned_path_str = run_metadata["artifacts"].get("aslg_pc12_clean.jsonl")
    if not aslg_cleaned_path_str:
        logger.error("ASLG cleaned JSONL path not found in run metadata!")
        return
    aslg_cleaned_path = Path(aslg_cleaned_path_str)

    tokenizer = Tokenizer(model_name='t5-small')
    tokenized_data = tokenizer.tokenize_data(aslg_cleaned_path)

    # Stack tensors
    input_ids = torch.stack([item['input_ids'] for item in tokenized_data])
    attention_mask = torch.stack([item['attention_mask'] for item in tokenized_data])
    labels = torch.stack([item['labels'] for item in tokenized_data])

    tokenized_dataset = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

    # Save tokenized dataset
    save_path = Path("artifacts/aslg_pc12_tokenized.pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tokenized_dataset, save_path)
    logger.info(f"Saved tokenized dataset to {save_path}")

    # Register new artifact
    tokenized_artifact = Artifact(
        name="aslg_pc12_tokenized.pt",
        type="pt",
        run_id=run_id,
        use_run_folder=False,
    )
    add_artifact_to_metadata(run_metadata, tokenized_artifact)
    save_run_metadata(run_id, run_metadata)

    # Sanity logging
    logger.info(f"Example tokenized input_ids: {tokenized_data[0]['input_ids']}")
    logger.info(f"Example tokenized labels: {tokenized_data[0]['labels']}")


if __name__ == "__main__":
    main()
