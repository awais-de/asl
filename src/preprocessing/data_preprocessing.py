import json
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List

from src.preprocessing.data_cleaning import clean_text
from src.utils.logging import get_logger
from src.utils.helpers import (
    get_latest_run_id,
    load_run_metadata,
    save_run_metadata,
    Artifact,
    add_artifact_to_metadata,
)
from src.utils.artifact_names import (
    WLASL_JSON,
    ASLG_PC12_PARQUET,
    ASLG_PC12_CLEAN_JSONL,
    GLOSS_TO_VIDEOID_MAP_JSON,
)

logger = get_logger(__name__)

def load_wlasl_data(json_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
    if not json_path.exists():
        logger.error(f"WLASL JSON file not found at: {json_path}")
        raise FileNotFoundError(f"File not found: {json_path}")

    gloss_list = []
    gloss_to_video_list = []

    logger.info(f"Loading WLASL JSON data from: {json_path}")

    try:
        with open(json_path, 'r') as f:
            wlasl_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read JSON file {json_path}: {e}")
        raise

    for idx, entry in enumerate(wlasl_data):
        gloss = entry.get('gloss', '').strip()
        if not gloss:
            logger.warning(f"Skipping entry {idx} with empty gloss")
            continue

        gloss_list.append({'gloss_id': idx, 'gloss': gloss})
        video_ids = [inst['video_id'] for inst in entry.get('instances', []) if 'video_id' in inst]
        gloss_to_video_list.append({'gloss_id': idx, 'gloss': gloss, 'video_ids': video_ids})

    df_gloss = pd.DataFrame(gloss_list)
    df_gloss_to_video = pd.DataFrame(gloss_to_video_list)
    gloss_to_videos_dict = dict(zip(df_gloss_to_video['gloss'], df_gloss_to_video['video_ids']))

    logger.info(f"Loaded {len(df_gloss)} gloss entries and their video mappings")
    return df_gloss, df_gloss_to_video, gloss_to_videos_dict


def save_gloss_video_mapping(mapping: Dict[str, List[str]], save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(save_path, 'w') as f:
            json.dump(mapping, f, indent=4)
        logger.info(f"Saved gloss-to-video mapping to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save gloss-to-video mapping: {e}")
        raise


def preprocess_aslg_pc12(parquet_path: Path, output_path: Path):
    if not parquet_path.exists():
        logger.error(f"ASLG-PC12 parquet file not found at: {parquet_path}")
        raise FileNotFoundError(f"File not found: {parquet_path}")

    logger.info(f"Loading ASLG-PC12 dataset from {parquet_path}")

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        logger.error(f"Failed to load parquet file: {e}")
        raise

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
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        cleaned_df.to_json(output_path, orient='records', lines=True)
        logger.info(f"Saved cleaned ASLG-PC12 data to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save cleaned ASLG-PC12 data: {e}")
        raise


def main():
    # Get latest run ID and metadata
    run_id = get_latest_run_id()
    run_metadata = load_run_metadata(run_id)

    # Get dataset paths from run metadata artifacts
    wlasl_base_path = Path(run_metadata["artifacts"].get("wlasl_dataset", ""))
    wlasl_json_path = wlasl_base_path / WLASL_JSON

    aslg_base_path = Path(run_metadata["artifacts"].get("aslg_dataset", ""))
    aslg_pc12_parquet_path = aslg_base_path / "data" / ASLG_PC12_PARQUET

    # Define output artifact paths in artifacts/
    gloss_video_map_path = Path("artifacts") / GLOSS_TO_VIDEOID_MAP_JSON
    aslg_pc12_cleaned_path = Path("artifacts") / ASLG_PC12_CLEAN_JSONL

    # Process WLASL gloss-video mapping
    df_gloss, df_gloss_to_video, gloss_to_videos = load_wlasl_data(wlasl_json_path)
    save_gloss_video_mapping(gloss_to_videos, gloss_video_map_path)

    # Process ASLG-PC12 parquet -> cleaned JSONL
    preprocess_aslg_pc12(aslg_pc12_parquet_path, aslg_pc12_cleaned_path)

    # Register new artifacts
    gloss_map_artifact = Artifact(
        name=GLOSS_TO_VIDEOID_MAP_JSON,
        type="json",
        run_id=run_id,
        use_run_folder=False,
    )
    add_artifact_to_metadata(run_metadata, gloss_map_artifact)

    aslg_clean_artifact = Artifact(
        name=ASLG_PC12_CLEAN_JSONL,
        type="jsonl",
        run_id=run_id,
        use_run_folder=False,
    )
    add_artifact_to_metadata(run_metadata, aslg_clean_artifact)

    save_run_metadata(run_id, run_metadata)

    logger.info(f"Artifacts registered for run {run_id}.")

if __name__ == "__main__":
    main()
