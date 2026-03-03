import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from ..config import settings
from ..logging_config import get_logger

logger = get_logger("environment")


class FileInfo(BaseModel):
    """Metadata about one user data file."""
    filename: str
    file_type: str                    # "csv", "json", "txt", "other"
    size_bytes: int
    columns: list[str] = []          # CSV / JSON only
    row_count: Optional[int] = None  # CSV / JSON array only
    sample_rows: list[dict] = []     # First 3 rows (CSV / JSON array only)


class EnvironmentContext(BaseModel):
    """Everything the planner needs to know about the user's environment."""
    current_date: str          # "2026-03-02"
    current_day: str           # "Monday"
    current_time: str          # "14:30"
    available_files: list[FileInfo] = []
    capabilities: list[str] = [
        "research (web search via DuckDuckGo)",
        "code_generation (Python scripts, functions, classes)",
        "file_read (CSV, JSON, TXT from user_data/)",
        "file_write (any format to outputs/)",
        "data_analysis (pandas, statistics, aggregation)",
        "app_generation (multi-file Python applications)",
        "server_testing (Flask/FastAPI validation)",
    ]
    unavailable: list[str] = [
        "email/messaging",
        "calendar access",
        "GUI rendering (can generate GUI code but cannot display it)",
        "external API keys (use free/keyless APIs only)",
        "database servers (SQLite only, no Postgres/MySQL)",
    ]

    def to_prompt_block(self) -> str:
        """Format as a text block suitable for injection into any LLM prompt."""
        lines = [
            "=== ENVIRONMENT CONTEXT ===",
            f"Date: {self.current_date} ({self.current_day})",
            f"Time: {self.current_time}",
        ]

        if self.available_files:
            lines.append("")
            lines.append("User data files available:")
            for f in self.available_files:
                line = f"  - {f.filename} ({f.file_type}, {f.size_bytes:,} bytes)"
                if f.row_count is not None:
                    line += f" — {f.row_count} rows"
                if f.columns:
                    cols = ", ".join(f.columns[:10])
                    if len(f.columns) > 10:
                        cols += f", ... (+{len(f.columns) - 10} more)"
                    line += f"\n    Columns: [{cols}]"
                lines.append(line)

            # Include sample data for the first 2 files only
            for f in self.available_files[:2]:
                if f.sample_rows:
                    lines.append(f"  Sample from {f.filename}:")
                    for row in f.sample_rows[:3]:
                        lines.append(f"    {row}")
        else:
            lines.append("")
            lines.append("No user data files found in user_data/ directory.")

        lines.append("")
        lines.append("System capabilities: " + "; ".join(self.capabilities))
        lines.append("NOT available: " + "; ".join(self.unavailable))
        lines.append("=== END ENVIRONMENT ===")

        return "\n".join(lines)


class EnvironmentScanner:
    """Scans the user's environment before planning begins."""

    @staticmethod
    def scan() -> EnvironmentContext:
        """Scan user_data/ and build an EnvironmentContext."""
        now = datetime.now()

        ctx = EnvironmentContext(
            current_date=now.strftime("%Y-%m-%d"),
            current_day=now.strftime("%A"),
            current_time=now.strftime("%H:%M"),
        )

        user_data_dir = settings.USER_DATA_DIR
        if not user_data_dir.exists():
            logger.info("user_data/ directory does not exist — no files to scan")
            return ctx

        for file_path in sorted(user_data_dir.iterdir()):
            if file_path.is_file() and not file_path.name.startswith("."):
                try:
                    info = EnvironmentScanner._scan_file(file_path)
                    if info is not None:
                        ctx.available_files.append(info)
                except Exception as e:
                    logger.warning(f"Failed to scan {file_path.name}: {e}")

        logger.info(
            f"Environment scan complete: {len(ctx.available_files)} file(s), "
            f"date={ctx.current_date} ({ctx.current_day})"
        )
        return ctx

    @staticmethod
    def _scan_file(path: Path) -> Optional[FileInfo]:
        """Dispatch scanning by file extension."""
        suffix = path.suffix.lower()
        size = path.stat().st_size

        if suffix == ".csv":
            return EnvironmentScanner._scan_csv(path, size)
        elif suffix == ".json":
            return EnvironmentScanner._scan_json(path, size)
        elif suffix in (".txt", ".md", ".log"):
            return FileInfo(
                filename=path.name,
                file_type="txt",
                size_bytes=size,
            )
        else:
            return FileInfo(
                filename=path.name,
                file_type="other",
                size_bytes=size,
            )

    @staticmethod
    def _scan_csv(path: Path, size: int) -> FileInfo:
        """Read CSV headers, row count, and first 3 rows."""
        columns: list[str] = []
        sample_rows: list[dict] = []
        row_count = 0

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            columns = list(reader.fieldnames or [])
            for i, row in enumerate(reader):
                if i < 3:
                    sample_rows.append(dict(row))
                row_count = i + 1

        return FileInfo(
            filename=path.name,
            file_type="csv",
            size_bytes=size,
            columns=columns,
            row_count=row_count,
            sample_rows=sample_rows,
        )

    @staticmethod
    def _scan_json(path: Path, size: int) -> FileInfo:
        """Read JSON structure — extract keys if object, count items if array."""
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)

        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            columns = list(data[0].keys())
            sample_rows = [dict(row) for row in data[:3]]
            return FileInfo(
                filename=path.name,
                file_type="json",
                size_bytes=size,
                columns=columns,
                row_count=len(data),
                sample_rows=sample_rows,
            )
        elif isinstance(data, dict):
            return FileInfo(
                filename=path.name,
                file_type="json",
                size_bytes=size,
                columns=list(data.keys()),
            )
        else:
            return FileInfo(
                filename=path.name,
                file_type="json",
                size_bytes=size,
            )
