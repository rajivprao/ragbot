
# -*- coding: utf-8 -*-
"""
extractors/pdf_extractor.py

PDF extractor with:
- Text extraction (PyMuPDF)
- Optional OCR for image/scanned pages (pytesseract)
- Optional vector parsing for ERD/diagrams (via DiagramExtractor)
- Cleaned, metadata-rich LangChain Document outputs

Outputs one or more Documents per page with metadata:
    file_name, source, page_number, doc_type, chunk_kind, created_at

chunk_kind ∈ {"text", "ocr", "erd"}

Usage:
    from extractors.pdf_extractor import PdfExtractor
    extractor = PdfExtractor(enable_ocr=True, ocr_dpi=220, enable_vector_parse=True)
    docs = extractor.load_pdf("/path/to/file.pdf")

    # or process an entire folder (recursively)
    docs = extractor.load_path("/path/to/folder")

Notes:
- OCR requires: `pytesseract` and Tesseract engine installed on system.
- Vector parsing relies on `extractors.diagram_extractor.DiagramExtractor`.
  If not available, ERD extraction gracefully skips.

"""

from __future__ import annotations

import io
import os
import re
import sys
import glob
import fitz  # PyMuPDF
from typing import List, Optional, Tuple


# LangChain Document
try:
    from langchain_core.documents import Document
except Exception:
    # Fallback minimal shim if langchain_core isn't available during static checks
    class Document:  # type: ignore
        def __init__(self, page_content: str, metadata: dict):
            self.page_content = page_content
            self.metadata = metadata

# Shared helpers
try:
    from vector_store.base import normalize_text, now_iso
except Exception:
    # Local fallbacks if not imported from vector_store.base
    import datetime

    def normalize_text(txt: Optional[str]) -> str:
        return " ".join((txt or "").split())

    def now_iso() -> str:
        return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

# Optional OCR
try:
    import pytesseract  # type: ignore
    from PIL import Image  # type: ignore
    _OCR_AVAILABLE = True
except Exception:
    _OCR_AVAILABLE = False

# Optional diagram/ERD vector parsing
try:
    # Expected interface:
    # class DiagramExtractor:
    #   def __init__(self, enable_ocr: bool=True, ocr_dpi: int=200): ...
    #   def parse_vector_erd(self, page: fitz.Page) -> Optional[dict]: ...
    #   def diagram_json_to_text(self, diag: dict, file_name: str, page_no: int) -> str: ...
    from extractors.diagram_extractor import DiagramExtractor  # type: ignore
    _DIAGRAM_AVAILABLE = True
except Exception:
    _DIAGRAM_AVAILABLE = False
    DiagramExtractor = None  # type: ignore


class PdfExtractor:
    """
    Extracts text, OCR text (optional), and ERD summaries (optional) from PDFs into LangChain Documents.

    Parameters
    ----------
    enable_ocr : bool
        If True and pytesseract is available, perform OCR on image-heavy/scan pages.
    ocr_dpi : int
        DPI used to rasterize pages before OCR (higher = slower but more accurate).
    enable_vector_parse : bool
        If True and DiagramExtractor is available, perform vector-based ERD extraction.
    min_text_len : int
        Minimum text length to consider a block non-empty (after normalization).
    doc_type_infer : bool
        If True, attempt to infer doc_type from file name.
    """

    def __init__(
        self,
        enable_ocr: bool = True,
        ocr_dpi: int = 200,
        enable_vector_parse: bool = True,
        min_text_len: int = 20,
        doc_type_infer: bool = True,
    ) -> None:
        self.enable_ocr = bool(enable_ocr and _OCR_AVAILABLE)
        self.ocr_dpi = int(ocr_dpi)
        self.enable_vector_parse = bool(enable_vector_parse and _DIAGRAM_AVAILABLE)
        self.min_text_len = int(min_text_len)
        self.doc_type_infer = bool(doc_type_infer)

        # Instantiate a diagram extractor if available
        self._diagram = DiagramExtractor(enable_ocr=False) if self.enable_vector_parse and DiagramExtractor else None

    # ----------------------------
    # Public API
    # ----------------------------

    def load_pdf(self, path: str, *, max_pages: Optional[int] = None) -> List[Document]:
        """
        Extract Documents from a single PDF.

        Returns a list of Documents with chunk_kind in {"text", "ocr", "erd"}.
        """
        docs: List[Document] = []
        file_name = os.path.basename(path)
        doc_type = self._infer_doc_type(file_name) if self.doc_type_infer else "unknown"

        try:
            pdf = fitz.open(path)
        except Exception as e:
            print(f"[WARN] Cannot open PDF {file_name}: {e}")
            return docs

        total_pages = len(pdf)
        page_limit = min(total_pages, max_pages) if max_pages else total_pages

        for i in range(page_limit):
            page = pdf.load_page(i)
            page_no = i + 1
            base_meta = self._base_meta(path, file_name, page_no, doc_type)

            # 1) Normal text extraction
            page_text = self._extract_text(page)
            text_ok = len(page_text) >= self.min_text_len
            if text_ok:
                docs.append(Document(page_content=page_text, metadata={**base_meta, "chunk_kind": "text"}))

            # 2) Diagram/ERD (vector parsing)
            if self._diagram is not None:
                try:
                    diag = self._diagram.parse_vector_erd(page)
                except Exception as e:
                    diag = None
                    print(f"[WARN] ERD parse failed {file_name} p{page_no}: {e}")

                if diag:
                    try:
                        erd_summary = self._diagram.diagram_json_to_text(diag, file_name, page_no)
                        if len(normalize_text(erd_summary)) >= self.min_text_len:
                            docs.append(Document(page_content=erd_summary, metadata={**base_meta, "chunk_kind": "erd"}))
                    except Exception as e:
                        print(f"[WARN] ERD summarize failed {file_name} p{page_no}: {e}")

            # 3) OCR (only if needed: image-heavy pages or text is too short)
            if self.enable_ocr:
                try:
                    has_images = bool(page.get_images())
                except Exception:
                    has_images = False
                need_ocr = has_images and not text_ok

                if need_ocr:
                    try:
                        ocr_text = self._ocr_page(page, dpi=self.ocr_dpi)
                        if len(ocr_text) >= self.min_text_len:
                            docs.append(Document(page_content=ocr_text, metadata={**base_meta, "chunk_kind": "ocr"}))
                    except Exception as e:
                        print(f"[WARN] OCR failed {file_name} p{page_no}: {e}")

        pdf.close()
        return docs

    def load_path(self, path: str, *, recursive: bool = True, max_pages_per_file: Optional[int] = None) -> List[Document]:
        """
        Extract Documents from all PDFs found under a folder (optionally recursive).
        """
        pattern = "**/*.pdf" if recursive else "*.pdf"
        files = sorted(glob.glob(os.path.join(path, pattern), recursive=recursive))
        all_docs: List[Document] = []
        for fp in files:
            all_docs.extend(self.load_pdf(fp, max_pages=max_pages_per_file))
        if not all_docs:
            print(f"[INFO] No PDFs processed under: {path}")
        return all_docs

    # ----------------------------
    # Internals
    # ----------------------------

    def _extract_text(self, page: fitz.Page) -> str:
        """
        Extracts normalized text from page text layer.
        """
        try:
            txt = page.get_text("text")
        except Exception:
            txt = ""
        return normalize_text(txt)

    def _ocr_page(self, page: fitz.Page, dpi: Optional[int] = None) -> str:
        """
        Rasterizes a page and runs Tesseract OCR (if available).
        """
        if not self.enable_ocr or not _OCR_AVAILABLE:
            return ""
        dpi = dpi or self.ocr_dpi
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        try:
            from PIL import Image  # local import (already attempted globally)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            txt = pytesseract.image_to_string(img)  # type: ignore
        except Exception:
            txt = ""
        return normalize_text(txt)

    def _base_meta(self, source_path: str, file_name: str, page_no: int, doc_type: str) -> dict:
        """
        Standard metadata applied to every output Document.
        """
        return {
            "file_name": file_name,
            "source": source_path,
            "page_number": page_no,
            "doc_type": doc_type,
            "created_at": now_iso(),
        }

    def _infer_doc_type(self, filename: str) -> str:
        """
        Heuristic doc_type inference from file name.
        """
        name = filename.lower()
        patterns = {
            "technical": ["tech", "technical", "architecture", "design"],
            "schema": ["schema", "tables", "columns", "ddl", "database"],
            "erd": ["erd", "entity", "diagram", "model"],
            "functional": ["functional", "business", "requirements", "brd"],
            "product": ["product", "features", "manual", "documentation", "guide"],
            "analytics": ["analytics", "kpi", "metric", "dashboard"],
            "reporting": ["report", "reporting"],
            "onboarding": ["onboarding", "getting_started", "introduction"],
            "user_guide": ["user_guide", "help", "how_to", "tutorial"],
            "api": ["api", "interface", "integration"],
        }
        for dt, keys in patterns.items():
            if any(k in name for k in keys):
                return dt
        return "unknown"
