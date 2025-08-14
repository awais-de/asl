import os
import sys
import argparse
import json
from pathlib import Path
import shutil
from datasets import load_dataset
import kagglehub
from src.utils.logging import get_logger
from src.utils.helpers import (
    get_next_run_id,
    save_run_metadata,
    add_artifact_to_metadata,
    Artifact,
)
from src.utils.artifact_names import (
    WLASL_JSON,
    ASLG_PC12_PARQUET,
    CONF_JSON,
)

logger = get_logger(__name__)

def load_conf_json(conf_path=CONF_JSON):
    conf_path = Path(conf_path)
    if not conf_path.exists():
        logger.error(f"Configuration file {conf_path} not found.")
        sys.exit(1)
    with open(conf_path, "r") as f:
        conf = json.load(f)
    return conf

def download_and_prepare_kaggle_dataset(kaggle_path: str, dest_dir: Path):
    try:
        logger.info(f"Starting Kaggle dataset download: {kaggle_path}")
        dataset_cache_path = kagglehub.dataset_download(kaggle_path)
        logger.info(f"Kaggle dataset cached at: {dataset_cache_path}")

        shutil.copytree(dataset_cache_path, dest_dir, dirs_exist_ok=True)
        logger.info(f"Kaggle dataset copied to: {dest_dir}")

        return dest_dir

    except Exception as e:
        logger.error(f"Failed to download or copy Kaggle dataset {kaggle_path}: {e}")
        raise

def download_and_prepare_hf_dataset(dataset_name: str, dest_dir: Path):
    try:
        logger.info(f"Loading Hugging Face dataset: {dataset_name}")
        dataset = load_dataset(dataset_name)

        dataset.save_to_disk(str(dest_dir))
        logger.info(f"Hugging Face dataset saved to: {dest_dir}")

        return dest_dir

    except Exception as e:
        logger.error(f"Failed to download or save HF dataset {dataset_name}: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Download and prepare video datasets")
    parser.add_argument("--conf", type=str, default=CONF_JSON, help="Path to conf.json")
    parser.add_argument("--wlasl_dest_dir", type=str, default="data/raw/WLASL2000", help="Local directory to save WLASL dataset")
    parser.add_argument("--aslg_dest_dir", type=str, default="data/raw/ASLG_PC12", help="Local directory to save ASLG dataset")
    args = parser.parse_args()

    # Load config
    conf = load_conf_json(args.conf)

    # Get dataset URLs/IDs from conf.json
    wlasl_kaggle_path = conf.get("wlasl_kaggle_path", "sttaseen/wlasl2000-resized")
    aslg_hf_name = conf.get("aslg_hf_name", "achrafothman/aslg_pc12")

    wlasl_dest_dir = args.wlasl_dest_dir
    aslg_dest_dir = args.aslg_dest_dir

    run_id = get_next_run_id()
    run_metadata = {
        "run_id": run_id,
        "datasets": {},
    }

    try:
        wlasl_path = download_and_prepare_kaggle_dataset(wlasl_kaggle_path, Path(wlasl_dest_dir))
        wlasl_artifact = Artifact(
            name=WLASL_JSON,
            type="json",
            run_id=run_id,
            use_run_folder=False,
        )
        add_artifact_to_metadata(run_metadata, wlasl_artifact)

        aslg_path = download_and_prepare_hf_dataset(aslg_hf_name, Path(aslg_dest_dir))
        aslg_artifact = Artifact(
            name=ASLG_PC12_PARQUET,
            type="parquet",
            run_id=run_id,
            use_run_folder=False,
        )
        add_artifact_to_metadata(run_metadata, aslg_artifact)

        save_run_metadata(run_id, run_metadata)

        logger.info(f"Datasets downloaded and saved. Run ID: {run_id}")

    except Exception as e:
        logger.error(f"Dataset download failed for run {run_id}: {e}", exc_info=True)
        save_run_metadata(run_id, run_metadata)
        sys.exit(1)

if __name__ == "__main__":
    main()
