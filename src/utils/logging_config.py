"""
Logging configuration
"""
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(
    name: str = "rideshare",
    level: int = logging.INFO,
    log_dir: Path | None = None
) -> logging.Logger:
    """
    Logger with console and optional file handlers setup.

    Parameters:
    name : str
        Logger name
    level : int
        Logging level
    log_dir : Path, optional
        Directory for log files. If None, uses project logs/ dir.

    Returns:
    logging.Logger
    """
    if log_dir is None:
        log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_fmt = logging.Formatter("%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s", datefmt="%H:%M:%S")
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # File Handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(log_dir / f"{name}_{timestamp}.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter("%(asctime)s │ %(levelname)-8s │ %(name)s.%(funcName)s:%(lineno)d │ %(message)s")
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    return logger