import sys
import logging
from pathlib import Path
from datetime import datetime

from src.utils.helpers import (
    get_next_run_id,
    save_run_metadata,
    add_step_to_metadata,
    add_artifact_to_metadata,
    Artifact,
)
from src.utils.artifact_names import (
    ASLG_PC12_PARQUET,
    WLASL_JSON,
    ASLG_PC12_CLEAN_JSONL,
    GLOSS_TO_VIDEOID_MAP_JSON,
    ASLG_PC12_TOKENIZED_PT,
    PREDICTIONS_CSV,
    CONF_JSON,
    RUN_INFO_JSON,
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

RAW_ASLG_PC12_PARQUET = Path("data/raw/aslg_pc12/data") / ASLG_PC12_PARQUET
RAW_WLASL_JSON = Path("data/raw/WLASL2000/wlasl-complete") / WLASL_JSON

def make_artifact_filename(run_id, timestamp, artifact_name):
    return f"run{run_id}_{timestamp}_{artifact_name}"

def run_configuration_setup():
    logger.info("Step 0: Configuration Setup")
    conf_path = Path(CONF_JSON)
    if not conf_path.exists():
        logger.error(f"Configuration file {CONF_JSON} not found. Please provide it before running the pipeline.")
        sys.exit(1)
    logger.info(f"Configuration file {CONF_JSON} found.")
    logger.info("Step 0 completed")

def run_load_videos_dataset():
    from src.data_loading import load_videos_dataset
    logger.info("Step 1: Loading ASLG_PC12 and WLASL2000 datasets")
    load_videos_dataset.main()
    logger.info("Step 1 completed")

def run_preprocess_aslg_pc12(run_id: int, timestamp: str) -> Path:
    from src.preprocessing import data_preprocessing
    logger.info("Step 2.1: Preprocessing ASLG_PC12 dataset")
    filename = make_artifact_filename(run_id, timestamp, ASLG_PC12_CLEAN_JSONL)
    output_path = Path("artifacts") / filename
    data_preprocessing.preprocess_aslg_pc12(RAW_ASLG_PC12_PARQUET, output_path)
    logger.info(f"Step 2.1 completed, output saved to {output_path}")
    return output_path

def run_preprocess_wlasl(run_id: int, timestamp: str) -> Path:
    from src.preprocessing import data_preprocessing
    logger.info("Step 2.2: Preprocessing WLASL dataset")
    filename = make_artifact_filename(run_id, timestamp, GLOSS_TO_VIDEOID_MAP_JSON)
    output_path = Path("artifacts") / filename
    df_gloss, df_gloss_to_video, gloss_to_videos = data_preprocessing.load_wlasl_data(RAW_WLASL_JSON)
    data_preprocessing.save_gloss_video_mapping(gloss_to_videos, output_path)
    logger.info(f"Step 2.2 completed, output saved to {output_path}")
    return output_path

def run_tokenization(aslg_clean_path: Path, run_id: int, timestamp: str) -> Path:
    from src.preprocessing import tokenize_aslg_pc12
    logger.info("Step 3: Tokenizing ASLG-PC12 dataset")
    filename = make_artifact_filename(run_id, timestamp, ASLG_PC12_TOKENIZED_PT)
    output_path = Path("artifacts") / filename
    tokenize_aslg_pc12.main([
        "--input", str(aslg_clean_path),
        "--output", str(output_path)
    ])
    logger.info(f"Step 3 completed, output saved to {output_path}")
    return output_path

def run_prepare_dataloader(tokenized_path: Path):
    from src.data_loading import prepare_dataloader
    logger.info("Step 4: Preparing DataLoader")
    prepare_dataloader.main(tokenized_path=str(tokenized_path))
    logger.info("Step 4 completed")

def run_verification():
    from src.verification import check_dataset_and_model
    logger.info("Step 5: Verifying environment and datasets")
    check_dataset_and_model.main()
    logger.info("Step 5 completed")

def run_training(tokenized_path: Path, run_id: int) -> Path:
    from src.models import train
    logger.info("Step 6: Training model")
    checkpoint_dir = Path("artifacts")
    train.main([
        "--tokenized_path", str(tokenized_path),
        "--max_epochs", "5",
        "--gpus", "1",
        "--batch_size", "32",
        "--checkpoint_dir", str(checkpoint_dir)
    ])
    logger.info(f"Step 6 completed, checkpoints saved to {checkpoint_dir}")
    return checkpoint_dir

def run_evaluation(tokenized_path: Path, run_id: int):
    from src.models import evaluate
    logger.info("Step 7: Evaluating model")
    evaluate.main([
        "--tokenized_path", str(tokenized_path),
        "--checkpoint_dir", str(Path("artifacts"))
    ])
    logger.info("Step 7 completed")

STEP_FUNCTIONS = [
#    ("configuration_setup", run_configuration_setup),
#    ("load_datasets", run_load_videos_dataset),
    ("preprocess_aslg_pc12", run_preprocess_aslg_pc12),
    ("preprocess_wlasl", run_preprocess_wlasl),
    ("tokenization", run_tokenization),
    ("prepare_dataloader", run_prepare_dataloader),
    ("verification", run_verification),
    ("training", run_training),
    ("evaluation", run_evaluation),
]

def parse_skip_steps(argv):
    skip_steps = set()
    for arg in argv[1:]:
        if arg.startswith("skip="):
            val = arg.split("=")[1]
            if val == "all":
                skip_steps = set(range(len(STEP_FUNCTIONS)))
            else:
                skip_steps = set(int(x) for x in val.split(",") if x.isdigit())
    return skip_steps

def main():
    logger.info("=== Starting full pipeline run ===")
    run_id = get_next_run_id()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_metadata = {
        "run_id": run_id,
        "timestamp": timestamp,
        "status": "RUNNING",
        "steps": [],
        "artifacts": {}
    }

    skip_steps = parse_skip_steps(sys.argv)
    aslg_clean_path = None
    tokenized_path = None
    checkpoint_dir = None

    try:
        for idx, (step_name, step_func) in enumerate(STEP_FUNCTIONS):
            if idx in skip_steps:
                logger.info(f"Skipping step {idx}: {step_name}")
                continue

            add_step_to_metadata(run_metadata, step_name)

            # Handle steps that require arguments or return values
            if step_name == "preprocess_aslg_pc12":
                aslg_clean_path = step_func(run_id, timestamp)
                add_artifact_to_metadata(run_metadata, Artifact(
                    name=aslg_clean_path.name,
                    type="jsonl",
                    run_id=run_id,
                    use_run_folder=False,
                ))
            elif step_name == "preprocess_wlasl":
                gloss_to_video_path = step_func(run_id, timestamp)
                add_artifact_to_metadata(run_metadata, Artifact(
                    name=gloss_to_video_path.name,
                    type="json",
                    run_id=run_id,
                    use_run_folder=False,
                ))
            elif step_name == "tokenization":
                tokenized_path = step_func(aslg_clean_path, run_id, timestamp)
                add_artifact_to_metadata(run_metadata, Artifact(
                    name=tokenized_path.name,
                    type="pt",
                    run_id=run_id,
                    use_run_folder=False,
                ))
            elif step_name == "prepare_dataloader":
                step_func(tokenized_path)
            elif step_name == "training":
                checkpoint_dir = step_func(tokenized_path, run_id)
                add_artifact_to_metadata(run_metadata, Artifact(
                    name="checkpoint_dir",
                    type="dir",
                    run_id=run_id,
                    use_run_folder=False,
                ))
            elif step_name == "evaluation":
                step_func(tokenized_path, run_id)
            else:
                step_func()
            save_run_metadata(run_id, run_metadata)

        run_metadata["status"] = "SUCCESS"
        save_run_metadata(run_id, run_metadata)
        logger.info(f"Run {run_id} completed successfully")

    except Exception as e:
        run_metadata["status"] = "FAILED"
        logger.error(f"Run {run_id} failed: {e}", exc_info=True)
        save_run_metadata(run_id, run_metadata)
        sys.exit(1)

if __name__ == "__main__":
    main()
