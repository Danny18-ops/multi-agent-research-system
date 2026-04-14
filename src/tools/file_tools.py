"""File I/O tools for persisting and reading research outputs."""

from __future__ import annotations

import logging
from pathlib import Path

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WriteInput(BaseModel):
    """Input schema for the file writer tool."""

    filename: str = Field(..., description="Filename (relative to output_dir)")
    content: str = Field(..., description="Text content to write")
    output_dir: str = Field("outputs", description="Directory to write the file into")


class ReadInput(BaseModel):
    """Input schema for the file reader tool."""

    filepath: str = Field(..., description="Absolute or relative path to the file")


class FileWriterTool(BaseTool):
    """Write text content to a file in the output directory.

    Creates any missing parent directories automatically.
    Existing files are overwritten.
    """

    name: str = "file_writer"
    description: str = (
        "Write text content to a file. "
        "Use this to save research notes, summaries, or final reports to disk."
    )
    args_schema: type[BaseModel] = WriteInput

    def _run(self, filename: str, content: str, output_dir: str = "outputs") -> str:
        """Write *content* to *output_dir*/*filename*."""
        dest = Path(output_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)

        try:
            dest.write_text(content, encoding="utf-8")
            logger.info("Wrote %d chars to %s", len(content), dest)
            return f"Successfully wrote {len(content)} characters to {dest}"
        except OSError as exc:
            logger.error("File write failed: %s", exc)
            return f"Failed to write file: {exc}"


class FileReaderTool(BaseTool):
    """Read text content from a file on disk.

    Returns the raw text content or an error message if the file cannot
    be found or read.
    """

    name: str = "file_reader"
    description: str = (
        "Read text content from a file. "
        "Use this to load previously saved research notes or reports."
    )
    args_schema: type[BaseModel] = ReadInput

    def _run(self, filepath: str) -> str:
        """Return the contents of *filepath*."""
        path = Path(filepath)
        if not path.exists():
            return f"File not found: {filepath}"

        try:
            content = path.read_text(encoding="utf-8")
            logger.info("Read %d chars from %s", len(content), path)
            return content
        except OSError as exc:
            logger.error("File read failed: %s", exc)
            return f"Failed to read file: {exc}"
