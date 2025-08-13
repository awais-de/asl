import sys
import logging
from pathlib import Path
from datetime import datetime

# Import helper functions
from src.utils.helpers import (
    get_next_run_id,
    get_artifact_path,
    save_run_metadata,
    add_step_to_metadata,
    add_artifact_to_metadata,
    timestamp_str,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Paths for raw datasets (assumed fixed)
RAW_ASLG_PC12_PARQUET = Path("data/raw/aslg_pc12/data/train-00000-of-00001.parquet")
RAW_WLASL_JSON = Path("data/raw/WLASL2000/wlasl-complete/WLASL_v0.3.json")

# Scripts to run - update imports or call patterns if needed
# Here we import functions or run scripts via subprocess or as modules

def run_load_videos_dataset():
    # Assuming this is a standalone script, run it here or import and call main if available
    from src.data_loading import load_videos_dataset
    logger.info("Step 1: Loading ASLG_PC12 and WLASL2000 datasets")
    load_videos_dataset.main()
    logger.info("Step 1 completed")

def run_preprocess_aslg_pc12(run_id: int) -> Path:
    from src.preprocessing import preprocess_aslg_pc12

    logger.info("Step 2: Preprocessing ASLG_PC12 dataset")
    output_path = get_artifact_path("aslg_pc12_clean.jsonl", run_id)
    preprocess_aslg_pc12.preprocess_aslg_pc12(str(RAW_ASLG_PC12_PARQUET), str(output_path))
    logger.info(f"Step 2 completed, output saved to {output_path}")
    return output_path

def run_preprocess_wlasl(run_id: int) -> Path:
    from src.preprocessing import preprocess_videos

    logger.info("Step 3: Preprocessing WLASL dataset")
    output_path = get_artifact_path("gloss_to_videoid_map.json", run_id)
    df_gloss, df_gloss_to_video, gloss_to_videos = preprocess_videos.load_wlasl_data(RAW_WLASL_JSON)
    preprocess_videos.save_gloss_video_mapping(gloss_to_videos, output_path)
    logger.info(f"Step 3 completed, output saved to {output_path}")
    return output_path

def run_tokenization(aslg_clean_path: Path, run_id: int) -> Path:
    from src.preprocessing import tokenize_aslg_pc12

    logger.info("Step 4: Tokenizing ASLG-PC12 dataset")
    output_path = get_artifact_path("aslg_pc12_tokenized.pt", run_id)
    tokenize_aslg_pc12.tokenize_aslg_pc12(str(aslg_clean_path), str(output_path))
    logger.info(f"Step 4 completed, output saved to {output_path}")
    return output_path

def run_prepare_dataloader(tokenized_path: Path):
    from src.data_loading import prepare_dataloader
    logger.info("Step 5: Preparing DataLoader")
    prepare_dataloader.prepare_dataloaders(str(tokenized_path))
    logger.info("Step 5 completed")

def run_verification():
    from src.verification import check_dataset_and_model
    logger.info("Step 6: Verifying environment and datasets")
    check_dataset_and_model.verify()
    logger.info("Step 6 completed")

def run_training(tokenized_path: Path, run_id: int) -> Path:
    from src.models import train

    logger.info("Step 7: Training model")
    # Override train.py to save checkpoints in run folder
    checkpoint_dir = Path("models/checkpoints") / f"run{run_id}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # You can pass args as needed for train.py interface
    train.main([
        "--tokenized_path", str(tokenized_path),
        "--max_epochs", "5",
        "--gpus", "1",
        "--batch_size", "32",
        "--checkpoint_dir", str(checkpoint_dir)  # adjust train.py to accept this arg if not present
    ])
    logger.info(f"Step 7 completed, checkpoints saved to {checkpoint_dir}")
    return checkpoint_dir

def run_evaluation(tokenized_path: Path, run_id: int):
    from src.models import evaluate

    logger.info("Step 8: Evaluating model")
    evaluate.main([
        "--tokenized_path", str(tokenized_path),
        "--checkpoint_dir", str(Path("models/checkpoints") / f"run{run_id}")
    ])
    logger.info("Step 8 completed")

def main():
    logger.info("=== Starting full pipeline run ===")
    run_id = get_next_run_id()
    run_metadata = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "steps": [],
        "artifacts": {}
    }

    try:
        # Step 1: Load raw datasets (if needed)
        add_step_to_metadata(run_metadata, "load_datasets")
        # This script is assumed to download or prepare raw datasets
        run_load_videos_dataset()
        save_run_metadata(run_id, run_metadata)

        # Step 2: Preprocess ASLG_PC12 dataset
        add_step_to_metadata(run_metadata, "preprocess_aslg_pc12")
        aslg_clean_path = run_preprocess_aslg_pc12(run_id)
        add_artifact_to_metadata(run_metadata, "aslg_pc12_cleaned_jsonl", str(aslg_clean_path))
        save_run_metadata(run_id, run_metadata)

        # Step 3: Preprocess WLASL dataset
        add_step_to_metadata(run_metadata, "preprocess_wlasl")
        gloss_to_video_path = run_preprocess_wlasl(run_id)
        add_artifact_to_metadata(run_metadata, "gloss_to_videoid_json", str(gloss_to_video_path))
        save_run_metadata(run_id, run_metadata)

        # Step 4: Tokenize ASLG_PC12 cleaned jsonl
        add_step_to_metadata(run_metadata, "tokenization")
        tokenized_path = run_tokenization(aslg_clean_path, run_id)
        add_artifact_to_metadata(run_metadata, "tokenized_dataset", str(tokenized_path))
        save_run_metadata(run_id, run_metadata)

        # Step 5: Prepare DataLoader (usually a quick check or print)
        add_step_to_metadata(run_metadata, "prepare_dataloader")
        run_prepare_dataloader(tokenized_path)
        save_run_metadata(run_id, run_metadata)

        # Step 6: Verification step
        add_step_to_metadata(run_metadata, "verification")
        run_verification()
        save_run_metadata(run_id, run_metadata)

        # Step 7: Training
        add_step_to_metadata(run_metadata, "training")
        checkpoint_dir = run_training(tokenized_path, run_id)
        add_artifact_to_metadata(run_metadata, "checkpoint_dir", str(checkpoint_dir))
        save_run_metadata(run_id, run_metadata)

        # Step 8: Evaluation
        add_step_to_metadata(run_metadata, "evaluation")
        run_evaluation(tokenized_path, run_id)
        save_run_metadata(run_id, run_metadata)

        logger.info(f"Run {run_id} completed successfully")

    except Exception as e:
        logger.error(f"Run {run_id} failed: {e}", exc_info=True)
        save_run_metadata(run_id, run_metadata)
        sys.exit(1)

if __name__ == "__main__":
    main()
