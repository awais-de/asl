import argparse
import torch
from pathlib import Path
from transformers import T5Tokenizer
from src.data_loading.aslg_pc12_dataset import ASLGPC12Dataset
from src.models.text_to_gloss_model import TextToGlossModel
from src.utils.logging import get_logger, add_file_handler, save_predictions_csv
from src.utils.metrics import compute_bleu, compute_rouge, compute_exact_match
import json
from datetime import datetime

from src.utils.helpers import (
    get_latest_run_id,
    load_run_metadata,
    save_run_metadata,
    Artifact,
    add_artifact_to_metadata,
)


def get_best_checkpoint_from_json(json_path: Path):
    if not json_path.exists():
        raise FileNotFoundError(f"Checkpoint info file not found: {json_path}")
    with open(json_path, "r") as f:
        checkpoint_data = json.load(f)
    # find checkpoint with minimum val_loss
    best_ckpt = min(checkpoint_data, key=lambda x: x['val_loss'])
    return Path(best_ckpt['filepath'])


def main(args=None):
    parser = argparse.ArgumentParser(description="Evaluate a trained Text-to-Gloss model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use: 'cpu' or 'cuda'")
    parsed_args = parser.parse_args(args)

    logger = get_logger(__name__)
    log_file = add_file_handler(logger, log_dir="logs", prefix="evaluation")

    # Load run metadata and get artifact paths
    run_id = get_latest_run_id()
    run_metadata = load_run_metadata(run_id)

    checkpoint_info_path_str = run_metadata["artifacts"].get("checkpoints_info.json")
    if not checkpoint_info_path_str:
        # fallback: get the latest checkpoint info artifact
        checkpoint_info_path_str = None
        for k in run_metadata["artifacts"]:
            if k.startswith("checkpoints_info_") and k.endswith(".json"):
                checkpoint_info_path_str = run_metadata["artifacts"][k]
        if not checkpoint_info_path_str:
            logger.error("Checkpoint info JSON not found in run metadata!")
            return
    checkpoint_info_path = Path(checkpoint_info_path_str)

    tokenized_path_str = run_metadata["artifacts"].get("aslg_pc12_tokenized.pt")
    if not tokenized_path_str:
        logger.error("Tokenized dataset path not found in run metadata!")
        return
    tokenized_path = Path(tokenized_path_str)

    device = torch.device(parsed_args.device if torch.cuda.is_available() or parsed_args.device == "cpu" else "cpu")
    logger.info(f"Using device: {device}")

    logger.info(f"Loading checkpoint info from: {checkpoint_info_path}")
    best_checkpoint_path = get_best_checkpoint_from_json(checkpoint_info_path)
    logger.info(f"Using best checkpoint for evaluation: {best_checkpoint_path}")

    logger.info("Loading tokenizer: t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    logger.info(f"Loading model from checkpoint: {best_checkpoint_path}")
    model = TextToGlossModel.load_from_checkpoint(best_checkpoint_path, tokenizer=tokenizer)
    model.to(device)
    model.eval()

    logger.info(f"Loading evaluation dataset from tokenized file: {tokenized_path}")
    dataset = ASLGPC12Dataset(tokenized_path=tokenized_path)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=parsed_args.batch_size)

    all_predictions = []
    all_labels = []

    logger.info("Starting evaluation...")
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            outputs = model.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=50)
            decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            all_predictions.extend(decoded_preds)
            all_labels.extend(decoded_labels)

    # Compute metrics
    bleu_score = compute_bleu(all_predictions, all_labels)
    rouge_scores = compute_rouge(all_predictions, all_labels)
    exact_match = compute_exact_match(all_predictions, all_labels)

    logger.info(f"BLEU Score: {bleu_score:.2f}")
    logger.info(f"ROUGE Scores: {rouge_scores}")
    logger.info(f"Exact Match Accuracy: {exact_match:.2f}%")

    # Save predictions CSV in artifacts/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = Path("artifacts") / f"evaluation_preds_{timestamp}.csv"
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    save_predictions_csv(all_predictions, all_labels, output_path=csv_file)
    logger.info(f"Predictions saved to: {csv_file}")

    # Register predictions CSV as artifact
    predictions_artifact = Artifact(
        name=csv_file.name,
        type="csv",
        run_id=run_id,
        use_run_folder=False,
    )
    add_artifact_to_metadata(run_metadata, predictions_artifact)
    save_run_metadata(run_id, run_metadata)

    logger.info("Sample predictions vs labels:")
    for pred, label in list(zip(all_predictions, all_labels))[:5]:
        logger.info(f"Predicted: {pred} | Label: {label}")

    logger.info(f"Evaluation complete. Logs saved to: {log_file}")


if __name__ == "__main__":
    main()
