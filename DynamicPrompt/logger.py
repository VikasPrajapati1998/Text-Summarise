# logger.py
import logging
import os
import warnings
import glob
import re
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from typing import List

_TIMESTAMP_RE = re.compile(r"_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.log$")

def _parse_timestamp_from_filename(path: str) -> datetime:
    """
    Try to extract timestamp from filename using pattern: _YYYY-MM-DD_HH-MM-SS.log
    If parsing fails, return None.
    """
    base = os.path.basename(path)
    m = _TIMESTAMP_RE.search(base)
    if not m:
        return None
    ts = m.group(1)
    try:
        return datetime.strptime(ts, "%Y-%m-%d_%H-%M-%S")
    except Exception:
        return None

def _purge_old_logs_global(log_dir: str, keep: int) -> List[str]:
    """
    Glob all .log files in log_dir, keep only the newest `keep` files.
    Sorting priority:
      1. Parsed timestamp from filename (pattern _YYYY-MM-DD_HH-MM-SS.log) if present.
      2. File modification time if timestamp not found.
    Deletes the oldest files until only `keep` remain.
    Returns list of deleted file paths.
    """
    pattern = os.path.join(log_dir, "*.log")
    files = glob.glob(pattern)

    if not files:
        return []

    file_infos = []
    for p in files:
        parsed_ts = _parse_timestamp_from_filename(p)
        if parsed_ts:
            sort_key = parsed_ts
        else:
            try:
                sort_key = datetime.fromtimestamp(os.path.getmtime(p))
            except Exception:
                sort_key = datetime.fromtimestamp(0)
        file_infos.append((p, sort_key))

    # Sort by date (oldest first)
    file_infos.sort(key=lambda x: x[1])

    deleted = []
    total = len(file_infos)
    if total > keep:
        num_to_delete = total - keep
        to_delete = file_infos[:num_to_delete]
        for path, _ in to_delete:
            try:
                os.remove(path)
                deleted.append(path)
            except Exception:
                # ignore deletion failures but continue
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
    - Keeps only the `keep` most recent log files in the entire log_dir
    - Suppresses noisy warnings
    """
    os.makedirs(log_dir, exist_ok=True)

    # One log file per run with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = os.path.join(log_dir, f"{module_name}_{timestamp}.log")

    # Ensure the file exists immediately so purge logic can consider it
    try:
        open(log_filename, "a", encoding="utf-8").close()
    except Exception:
        # If we can't touch the file, continue; purge will still operate on existing files
        pass

    logger = logging.getLogger(module_name)
    logger.setLevel(level)

    # Avoid adding handlers multiple times
    if logger.handlers:
        # still enforce global purge in case called again
        deleted = _purge_old_logs_global(log_dir, keep)
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

    # Purge old logs globally (keep only `keep` most recent files)
    deleted = _purge_old_logs_global(log_dir, keep)
    if deleted:
        logger.debug(f"Purged {len(deleted)} old log(s): {deleted}")

    logger.debug(f"Logger initialized: {log_filename}")
    return logger
