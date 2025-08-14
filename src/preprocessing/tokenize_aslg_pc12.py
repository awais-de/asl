from pathlib import Path
import json
import argparse
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
from src.utils.artifact_names import (
    ASLG_PC12_CLEAN_JSONL,
    ASLG_PC12_TOKENIZED_PT,
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

def main(args=None):
    parser = argparse.ArgumentParser(description="Tokenize ASLG-PC12 dataset")
    parser.add_argument("--input", type=str, required=True, help="Path to cleaned ASLG-PC12 JSONL")
    parser.add_argument("--output", type=str, required=True, help="Path to save tokenized .pt file")
    parser.add_argument("--model_name", type=str, default="t5-small", help="Tokenizer model name")
    parsed_args = parser.parse_args(args)

    input_path = Path(parsed_args.input)
    output_path = Path(parsed_args.output)

    tokenizer = Tokenizer(model_name=parsed_args.model_name)
    tokenized_data = tokenizer.tokenize_data(input_path)

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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tokenized_dataset, output_path)
    logger.info(f"Saved tokenized dataset to {output_path}")

    # Register new artifact in run metadata
    run_id = get_latest_run_id()
    run_metadata = load_run_metadata(run_id)
    tokenized_artifact = Artifact(
        name=ASLG_PC12_TOKENIZED_PT,
        type="pt",
        run_id=run_id,
        use_run_folder=False,
    )
    add_artifact_to_metadata(run_metadata, tokenized_artifact)
    save_run_metadata(run_id, run_metadata)

    logger.info(f"Example tokenized input_ids: {tokenized_data[0]['input_ids']}")
    logger.info(f"Example tokenized labels: {tokenized_data[0]['labels']}")

if __name__ == "__main__":
    main()
