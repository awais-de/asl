import os
import sys
import argparse
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
)

logger = get_logger(__name__)

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

    default_wlasl_kaggle_path = "sttaseen/wlasl2000-resized"
    default_wlasl_dest_dir = "data/raw/WLASL2000"
    default_aslg_hf_name = "achrafothman/aslg_pc12"
    default_aslg_dest_dir = "data/raw/ASLG_PC12"

    parser.add_argument("--wlasl_kaggle_path", type=str, default=None,
                        help=f"Kaggle path for WLASL2000 dataset (default: {default_wlasl_kaggle_path})")
    parser.add_argument("--wlasl_dest_dir", type=str, default=None,
                        help=f"Local directory to save WLASL dataset (default: {default_wlasl_dest_dir})")
    parser.add_argument("--aslg_hf_name", type=str, default=None,
                        help=f"Hugging Face dataset name for ASLG-PC12 (default: {default_aslg_hf_name})")
    parser.add_argument("--aslg_dest_dir", type=str, default=None,
                        help=f"Local directory to save ASLG dataset (default: {default_aslg_dest_dir})")

    args = parser.parse_args()

    # Determine WLASL kaggle path and dest dir
    wlasl_kaggle_path = args.wlasl_kaggle_path or os.getenv("WLASL_KAGGLE_PATH")
    if not wlasl_kaggle_path:
        wlasl_kaggle_path = default_wlasl_kaggle_path
        logger.info(f"No WLASL Kaggle path provided, defaulting to '{wlasl_kaggle_path}'")

    wlasl_dest_dir = args.wlasl_dest_dir or os.getenv("WLASL_DEST_DIR")
    if not wlasl_dest_dir:
        wlasl_dest_dir = default_wlasl_dest_dir
        logger.info(f"No WLASL destination directory provided, defaulting to '{wlasl_dest_dir}'")

    # Determine ASLG hf dataset name and dest dir
    aslg_hf_name = args.aslg_hf_name or os.getenv("ASLG_HF_NAME")
    if not aslg_hf_name:
        aslg_hf_name = default_aslg_hf_name
        logger.info(f"No ASLG Hugging Face dataset name provided, defaulting to '{aslg_hf_name}'")

    aslg_dest_dir = args.aslg_dest_dir or os.getenv("ASLG_DEST_DIR")
    if not aslg_dest_dir:
        aslg_dest_dir = default_aslg_dest_dir
        logger.info(f"No ASLG destination directory provided, defaulting to '{aslg_dest_dir}'")

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
