import logging
import sys
import os


def _supports_unicode() -> bool:
    """Check if the console supports unicode output."""
    if os.name == "nt":
        # Windows: check if the console encoding supports unicode
        try:
            encoding = sys.stdout.encoding or ""
            return encoding.lower().startswith("utf")
        except Exception:
            return False
    return True


class PrefixFormatter(logging.Formatter):
    """Custom formatter with level prefixes (ASCII-safe on Windows)."""

    if _supports_unicode():
        LEVEL_PREFIX = {
            logging.DEBUG: "[DBG]",
            logging.INFO: "[-->]",
            logging.WARNING: "[WRN]",
            logging.ERROR: "[ERR]",
            logging.CRITICAL: "[!!!]",
        }
    else:
        LEVEL_PREFIX = {
            logging.DEBUG: "[DBG]",
            logging.INFO: "[-->]",
            logging.WARNING: "[WRN]",
            logging.ERROR: "[ERR]",
            logging.CRITICAL: "[!!!]",
        }

    def format(self, record):
        record.prefix = self.LEVEL_PREFIX.get(record.levelno, "[???]")
        return super().format(record)


def setup_logging(level: str = "INFO"):
    """Configure logging for the ai_intern package."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Root logger for ai_intern package
    root_logger = logging.getLogger("ai_intern")
    root_logger.setLevel(log_level)

    # Clear existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Console handler (ASCII-safe)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(log_level)
    console_fmt = PrefixFormatter(
        "%(prefix)s [%(name)s] %(message)s"
    )
    console_handler.setFormatter(console_fmt)
    root_logger.addHandler(console_handler)

    # File handler (UTF-8 safe, can use unicode)
    file_handler = logging.FileHandler("ai_intern.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)  # Always log everything to file
    file_fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_fmt)
    root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a named logger under the ai_intern namespace."""
    return logging.getLogger(f"ai_intern.{name}")
