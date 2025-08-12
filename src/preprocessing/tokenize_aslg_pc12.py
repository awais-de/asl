from pathlib import Path
import json
from transformers import AutoTokenizer
import logging
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Tokenizer:
    def __init__(self, model_name='t5-small'):
        """
        Initialize the tokenizer from a pretrained model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_data(self, input_file: Path, gloss_key='gloss', text_key='text'):
        """
        Tokenize gloss and text fields from a JSONL file.

        Args:
            input_file (Path): Path to JSONL preprocessed data file.
            gloss_key (str): Key for gloss text in JSON.
            text_key (str): Key for English text in JSON.

        Returns:
            List of dicts with tokenized inputs and targets.
        """
        tokenized_samples = []
        logging.info(f"Tokenizing data from {input_file}")

        with input_file.open('r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                gloss = sample.get(gloss_key, "")
                text = sample.get(text_key, "")

                # Tokenize input (English text)
                input_encodings = self.tokenizer(
                    text,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )

                # Tokenize target (ASL gloss)
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

        logging.info(f"Tokenized {len(tokenized_samples)} samples")
        return tokenized_samples


def main():
    data_path = Path("data/processed/aslg_pc12_clean.jsonl")  # Adjust path if needed

    tokenizer = Tokenizer(model_name='t5-small')
    tokenized_data = tokenizer.tokenize_data(data_path)

    # Stack all tensors from the list of samples into single tensors
    input_ids = torch.stack([item['input_ids'] for item in tokenized_data])
    attention_mask = torch.stack([item['attention_mask'] for item in tokenized_data])
    labels = torch.stack([item['labels'] for item in tokenized_data])

    # Prepare dictionary for saving
    tokenized_dataset = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

    # Save the tokenized dataset to disk for fast loading later
    save_path = Path("data/processed/aslg_pc12_tokenized.pt")
    torch.save(tokenized_dataset, save_path)
    logging.info(f"Saved tokenized dataset to {save_path}")

    # Log example tokens for sanity check
    logging.info(f"Example tokenized input_ids: {tokenized_data[0]['input_ids']}")
    logging.info(f"Example tokenized labels: {tokenized_data[0]['labels']}")


if __name__ == "__main__":
    main()
