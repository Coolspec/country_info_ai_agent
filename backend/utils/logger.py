"""
Logging utility with auto-rotating file handler.

Logs are stored in: backend/app/logs/
- app.log          — current log file
- Auto-rotated by size (5 MB per file, keeps last 5 backups)
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent          # backend/
_LOGS_DIR = _BACKEND_DIR / "app" / "logs"
_LOGS_DIR.mkdir(parents=True, exist_ok=True)

_LOG_FILE = _LOGS_DIR / "app.log"

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d\n%(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# class ColorFormatter(logging.Formatter):
#     """Custom formatter adding color to log levels."""

#     COLORS = {
#         logging.DEBUG: "\x1b[38;20m",      # Grey
#         logging.INFO: "\x1b[32;20m",       # Green
#         logging.WARNING: "\x1b[33;20m",    # Yellow
#         logging.ERROR: "\x1b[31;20m",      # Red
#         logging.CRITICAL: "\x1b[31;1m",    # Bold Red
#     }
#     RESET = "\x1b[0m"

#     def __init__(self, fmt: str, datefmt: str | None = None):
#         super().__init__(fmt, datefmt)
#         self._formatters = {}
#         for level, color in self.COLORS.items():
#             colored_fmt = fmt.replace("%(levelname)-8s", f"{color}%(levelname)-8s{self.RESET}")
#             self._formatters[level] = logging.Formatter(colored_fmt, datefmt)
#         self._default_formatter = logging.Formatter(fmt, datefmt)

#     def format(self, record: logging.LogRecord) -> str:
#         formatter = self._formatters.get(record.levelno, self._default_formatter)
#         return formatter.format(record)

_formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
# _color_formatter = ColorFormatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

_file_handler = RotatingFileHandler(
    filename=_LOG_FILE,
    maxBytes=5 * 1024 * 1024,   # 5 MB
    backupCount=5,
    encoding="utf-8",
)
_file_handler.setFormatter(_formatter)
_file_handler.setLevel(logging.DEBUG)

_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(_formatter)
_console_handler.setLevel(logging.DEBUG)

def get_logger(name: str | None = None) -> logging.Logger:
    """
    Return a configured logger instance.

    Args:
        name: Logger name — typically ``__name__`` of the calling module.
              Defaults to the root project logger ``"country_info_agent"``.

    Returns:
        A ``logging.Logger`` with rotating file and console handlers attached.
    """
    logger_name = name or "country_info_agent"
    logger = logging.getLogger(logger_name)

    # Prevent adding duplicate handlers if called multiple times
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        logger.addHandler(_file_handler)
        logger.addHandler(_console_handler)
        logger.propagate = False

    return logger
