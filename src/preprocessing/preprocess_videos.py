import json
import pandas as pd
from typing import Tuple, Dict, List
from pathlib import Path
from src.utils.logging import get_logger

logger = get_logger(__name__)

def load_wlasl_data(json_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
    """
    Load the WLASL JSON metadata and extract gloss information and corresponding video IDs.

    Args:
        json_path (Path): Path to the WLASL JSON file.

    Returns:
        Tuple containing:
            - df_gloss (pd.DataFrame): DataFrame with gloss_id and gloss.
            - df_gloss_to_video (pd.DataFrame): DataFrame with gloss_id, gloss, and list of video_ids.
            - gloss_to_videos_dict (Dict[str, List[str]]): Dictionary mapping gloss to video IDs.
    """
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

        video_ids = [instance['video_id'] for instance in entry.get('instances', []) if 'video_id' in instance]
        gloss_to_video_list.append({'gloss_id': idx, 'gloss': gloss, 'video_ids': video_ids})

    df_gloss = pd.DataFrame(gloss_list)
    df_gloss_to_video = pd.DataFrame(gloss_to_video_list)

    gloss_to_videos_dict = dict(zip(df_gloss_to_video['gloss'], df_gloss_to_video['video_ids']))

    logger.info(f"Loaded {len(df_gloss)} gloss entries and their video mappings")

    return df_gloss, df_gloss_to_video, gloss_to_videos_dict

def save_gloss_video_mapping(mapping: Dict[str, List[str]], save_path: Path):
    """
    Save gloss to video IDs mapping as a JSON file.

    Args:
        mapping (Dict[str, List[str]]): Gloss to video IDs dictionary.
        save_path (Path): Path to save the JSON file.
    """
    try:
        with open(save_path, 'w') as f:
            json.dump(mapping, f, indent=4)
        logger.info(f"Saved gloss-to-video mapping to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save gloss-to-video mapping: {e}")
        raise

if __name__ == "__main__":
    json_path = Path("data/raw/WLASL2000/wlasl-complete/WLASL_v0.3.json")
    mapping_save_path = Path("data/processed/gloss_to_videoid_map.json")

    df_gloss, df_gloss_to_video, gloss_to_videos = load_wlasl_data(json_path)

    example_gloss = 'book'
    logger.info(f"Video IDs for gloss '{example_gloss}': {gloss_to_videos.get(example_gloss, [])}")

    save_gloss_video_mapping(gloss_to_videos, mapping_save_path)
