# logger.py
import logging
import os
import warnings
import glob
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from typing import List

def _purge_old_timestamped_logs(module_name: str, log_dir: str, keep: int) -> List[str]:
    """
    Delete older timestamped log files, keeping only the newest `keep` files.
    Only affects {module_name}_*.log
    """
    pattern = os.path.join(log_dir, f"{module_name}_*.log")
    files = glob.glob(pattern)

    if not files:
        return []

    # Sort by modification time (oldest first)
    files.sort(key=lambda p: os.path.getmtime(p))

    deleted = []
    if len(files) > keep:
        to_delete = files[:-keep]
        for f in to_delete:
            try:
                os.remove(f)
                deleted.append(f)
            except Exception:
                pass
    return deleted


def setup_logger(module_name: str,
                 log_dir: str = "logs",
                 level: int = logging.DEBUG,
                 when: str = "midnight",
                 backup_count: int = 7,
                 keep: int = 10) -> logging.Logger:
    """
    Creates ONE log file per run with timestamp: {module_name}_YYYY-MM-DD_HH-MM-SS.log
    - Uses TimedRotatingFileHandler for daily rotation (keeps `backup_count` old rotations)
    - Console output enabled
    - Keeps only the `keep` most recent session log files
    - Suppresses noisy warnings
    """
    os.makedirs(log_dir, exist_ok=True)

    # One log file per run with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(log_dir, f"{module_name}_{timestamp}.log")

    logger = logging.getLogger(module_name)
    logger.setLevel(level)

    # Avoid adding handlers multiple times
    if logger.handlers:
        deleted = _purge_old_timestamped_logs(module_name, log_dir, keep)
        if deleted:
            logger.debug(f"Purged {len(deleted)} old log(s): {deleted}")
        return logger

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")

    # TimedRotatingFileHandler on the *timestamped* file
    file_handler = TimedRotatingFileHandler(
        filename=log_filename,
        when=when,
        backupCount=backup_count,
        encoding="utf-8",
        utc=False
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Reduce third-party noise
    for noisy_logger in ["urllib3", "botocore", "boto3", "requests", "streamlit"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    # Suppress Streamlit warnings in script mode
    warnings.filterwarnings(
        "ignore",
        message=".*missing ScriptRunContext.*"
    )
    warnings.filterwarnings(
        "ignore",
        message="Session state does not function when running a script without.*"
    )

    # Purge old logs (keep only `keep` most recent)
    deleted = _purge_old_timestamped_logs(module_name, log_dir, keep)
    if deleted:
        logger.debug(f"Purged {len(deleted)} old log(s): {deleted}")

    logger.debug(f"Logger initialized: {log_filename}")
    return logger
