"""Focused unit tests for chunking behavior."""

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.models.document import ParsedPage
from app.services.chunker import ChunkerService


class ChunkerServiceTests(unittest.TestCase):
    """Validate chunk size and overlap mechanics."""

    def test_chunker_creates_overlapping_chunks(self):
        service = ChunkerService(chunk_size_words=5, chunk_overlap_words=2)
        text = "one two three four five six seven eight"
        pages = [ParsedPage(page_number=1, text=text)]

        chunks = service.chunk_pages("doc-1", "sample.pdf", pages)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].text, "one two three four five")
        self.assertEqual(chunks[1].text, "four five six seven eight")
        self.assertEqual(chunks[2].text, "seven eight")


if __name__ == "__main__":
    unittest.main()
