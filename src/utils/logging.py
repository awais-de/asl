import logging
import sys
from pathlib import Path
from datetime import datetime
import csv


def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    """
    Creates and returns a logger instance with stream handler.

    Args:
        name (str): Logger name, typically __name__.
        level (int): Logging level, e.g., logging.INFO.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.hasHandlers():
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)

        logger.addHandler(ch)

    return logger


def add_file_handler(logger: logging.Logger, log_dir: str = "logs", prefix: str = "log") -> Path:
    """
    Adds a file handler to an existing logger to save logs to a file.

    Args:
        logger (logging.Logger): Existing logger instance.
        log_dir (str): Directory to save logs.
        prefix (str): Prefix for the log filename.

    Returns:
        Path: Path to the created log file.
    """
    Path(log_dir).mkdir(exist_ok=True)
    log_file = Path(log_dir) / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    fh = logging.FileHandler(log_file)
    fh.setLevel(logger.level)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    logger.addHandler(fh)
    logger.info(f"File logging enabled: {log_file}")
    return log_file


def save_predictions_csv(predictions, labels, output_dir="logs", prefix="predictions") -> Path:
    """
    Saves predictions and labels to a CSV file.

    Args:
        predictions (list[str]): Model predictions.
        labels (list[str]): Corresponding ground truth labels.
        output_dir (str): Directory to save the CSV.
        prefix (str): Prefix for the CSV filename.

    Returns:
        Path: Path to the saved CSV file.
    """
    Path(output_dir).mkdir(exist_ok=True)
    csv_file = Path(output_dir) / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Prediction", "Label"])
        for pred, label in zip(predictions, labels):
            writer.writerow([pred, label])

    return csv_file
