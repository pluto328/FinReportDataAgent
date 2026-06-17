"""Text document parsing."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


TEXT_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


class DocumentParser:
    """Extract plain text from supported text formats."""

    async def parse(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            return path.read_text(encoding="utf-8", errors="ignore")
        if suffix == ".pdf":
            return self._parse_pdf(path)
        if suffix == ".docx":
            return self._parse_docx(path)
        return path.read_text(encoding="utf-8", errors="ignore")

    @staticmethod
    def _parse_pdf(path: Path) -> str:
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    @staticmethod
    def _parse_docx(path: Path) -> str:
        from docx import Document

        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
