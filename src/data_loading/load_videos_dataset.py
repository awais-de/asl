import kagglehub
from pathlib import Path
import shutil
import logging
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_and_prepare_kaggle_dataset(kaggle_path: str, dest_dir: Path):
    """
    Downloads a dataset from KaggleHub and copies it to the project data folder.
    """
    try:
        logging.info(f"Starting Kaggle dataset download: {kaggle_path}")
        dataset_cache_path = kagglehub.dataset_download(kaggle_path)
        logging.info(f"Kaggle dataset cached at: {dataset_cache_path}")

        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(dataset_cache_path, dest_dir, dirs_exist_ok=True)
        logging.info(f"Kaggle dataset copied to: {dest_dir}")

    except Exception as e:
        logging.error(f"Failed to download or copy Kaggle dataset {kaggle_path}: {e}")
        raise

def download_and_prepare_hf_dataset(dataset_name: str, dest_dir: Path):
    """
    Downloads a dataset from Hugging Face datasets and saves locally.
    """
    try:
        logging.info(f"Loading Hugging Face dataset: {dataset_name}")
        dataset = load_dataset(dataset_name)

        dest_dir.mkdir(parents=True, exist_ok=True)
        # Save the dataset locally in arrow/parquet format
        dataset.save_to_disk(str(dest_dir))
        logging.info(f"Hugging Face dataset saved to: {dest_dir}")

    except Exception as e:
        logging.error(f"Failed to download or save HF dataset {dataset_name}: {e}")
        raise

def main():
    # WLASL2000 from KaggleHub
    wlasl_info = {
        "kaggle_path": "sttaseen/wlasl2000-resized",
        "dest_dir": Path("data/raw/WLASL2000")
    }

    # ASLG-PC12 from Hugging Face
    aslg_info = {
        "hf_dataset_name": "achrafothman/aslg_pc12",
        "dest_dir": Path("data/raw/ASLG_PC12")
    }

    # Download WLASL2000 via KaggleHub
    download_and_prepare_kaggle_dataset(wlasl_info["kaggle_path"], wlasl_info["dest_dir"])

    # Download ASLG-PC12 via Hugging Face
    download_and_prepare_hf_dataset(aslg_info["hf_dataset_name"], aslg_info["dest_dir"])

if __name__ == "__main__":
    main()
