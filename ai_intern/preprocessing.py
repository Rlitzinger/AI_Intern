"""Request preprocessing - normalizes user input before planning."""
import re
from .execution.file import FileAgent
from .logging_config import get_logger

logger = get_logger("preprocessing")

# Common abbreviation expansions
ABBREVIATIONS = {
    "calc": "calculate",
    "avg": "average",
    "fn": "function",
    "func": "function",
    "impl": "implement",
    "dev": "develop",
    "req": "request",
    "resp": "response",
    "auth": "authentication",
    "config": "configuration",
    "db": "database",
    "docs": "documentation",
    "info": "information",
    "num": "number",
    "nums": "numbers",
    "str": "string",
    "vals": "values",
    "val": "value",
    "cals": "calories",
    "mins": "minutes",
    "secs": "seconds",
    "temp": "temperature",
    "temps": "temperatures",
}


def preprocess_request(content: str) -> str:
    """
    Normalize and enrich user input before planning.

    - Expands abbreviations
    - Resolves file references against available files
    - Returns enriched content string
    """
    original = content
    content = _expand_abbreviations(content)
    content = _resolve_file_references(content)

    if content != original:
        logger.info(f"Preprocessed: '{original}' -> '{content}'")

    return content


def _expand_abbreviations(text: str) -> str:
    """Expand common abbreviations to full words."""
    words = text.split()
    expanded = []
    for word in words:
        lower = word.lower().strip(".,!?;:")
        if lower in ABBREVIATIONS:
            replacement = ABBREVIATIONS[lower]
            # Preserve punctuation
            suffix = word[len(lower):]
            expanded.append(replacement + suffix)
        else:
            expanded.append(word)
    return " ".join(expanded)


def _resolve_file_references(text: str) -> str:
    """Resolve vague file references to actual filenames."""
    available = FileAgent.list_available_files()
    if not available:
        return text

    text_lower = text.lower()

    # Check if any file base name matches a word in the text
    for filepath in available:
        from pathlib import Path
        stem = Path(filepath).stem.lower()

        # If the stem appears in the text but the full filename doesn't
        if stem in text_lower and filepath.lower() not in text_lower:
            # Replace the bare stem with the full filename
            pattern = re.compile(re.escape(stem), re.IGNORECASE)
            text = pattern.sub(filepath, text, count=1)
            logger.debug(f"Resolved '{stem}' -> '{filepath}'")
            break

    # If "csv" mentioned but no specific file, and only one CSV exists
    if "csv" in text_lower and not any(f.lower() in text_lower for f in available):
        csvs = [f for f in available if f.endswith(".csv")]
        if len(csvs) == 1:
            text = text + f" (file: {csvs[0]})"
            logger.debug(f"Auto-resolved CSV reference to {csvs[0]}")

    return text
