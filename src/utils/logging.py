import logging
import sys

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
