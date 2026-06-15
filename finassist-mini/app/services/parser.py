"""PDF parsing service based on PyMuPDF (fitz)."""

from pathlib import Path

from app.models.document import ParsedPage


class PDFParserService:
    """Reads PDF files and extracts page-level text."""

    def parse_pdf(self, file_path: Path) -> list[ParsedPage]:
        """Extract text from a PDF file while preserving page numbers.

        Args:
            file_path: Absolute path to the PDF file.

        Returns:
            List of ParsedPage values, one per non-empty page.

        Raises:
            RuntimeError: If the PDF cannot be opened or read.
        """
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("PyMuPDF is not installed. Install requirements.txt dependencies.") from exc

        pages: list[ParsedPage] = []
        try:
            with fitz.open(file_path) as pdf_document:
                for page_index, page in enumerate(pdf_document, start=1):
                    text = page.get_text("text").strip()
                    if text:
                        pages.append(ParsedPage(page_number=page_index, text=text))
        except Exception as exc:
            raise RuntimeError(f"Failed to parse PDF '{file_path.name}'.") from exc

        return pages
