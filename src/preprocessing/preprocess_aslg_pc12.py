from pathlib import Path
import pandas as pd
from src.utils.helpers import clean_text
from src.utils.logging import get_logger

logger = get_logger(__name__)

def preprocess_aslg_pc12(parquet_path: str, output_path: str):
    logger.info(f"Loading ASLG-PC12 dataset from {parquet_path}")

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        logger.error(f"Failed to load parquet file: {e}")
        return

    glosses = []
    texts = []

    for i, row in df.iterrows():
        try:
            gloss_raw = str(row['gloss'])
            text_raw = str(row['text'])

            gloss_clean = clean_text(gloss_raw)
            text_clean = clean_text(text_raw)

            glosses.append(gloss_clean)
            texts.append(text_clean)

            if i % 1000 == 0 and i > 0:
                logger.info(f"Processed {i} rows")

        except Exception as e:
            logger.warning(f"Skipping row {i} due to error: {e}")

    cleaned_df = pd.DataFrame({'gloss': glosses, 'text': texts})

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        cleaned_df.to_json(output_path, orient='records', lines=True)
        logger.info(f"Saved cleaned data to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save cleaned data: {e}")


if __name__ == "__main__":
    parquet_file = "data/raw/aslg_pc12/data/train-00000-of-00001.parquet"
    output_file = "data/processed/aslg_pc12_clean.jsonl"
    preprocess_aslg_pc12(parquet_file, output_file)
