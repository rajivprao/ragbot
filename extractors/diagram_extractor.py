
# -*- coding: utf-8 -*-
"""
extractors/diagram_extractor.py

Vector-based ERD/diagram extractor using PyMuPDF geometry, with optional OCR helpers.

What it does
------------
- Parses PDF pages (fitz.Page) for vector drawings (rectangles/rounded-rects/lines/curves).
- Associates nearby/inside text blocks with boxes -> creates "nodes".
- Snaps line/curve endpoints to nearest nodes -> creates "edges".
- Optionally looks for a page/title from top-most text lines.
- Converts the detected graph to a compact JSON and/or a RAG-friendly text summary.

Optional:
- OCR helpers (pytesseract) for image-only pages or to augment label detection.
  (The PdfExtractor performs OCR on demand; this class keeps a simple method too.)

Returned JSON (example)
-----------------------
{
  "title": "Calendars",
  "nodes": [{"id":"calm_mstr"}, {"id":"calh_det"}, {"id":"cald_det"}, {"id":"cals_det"}],
  "edges": [
    {"from":"calh_det","to":"calm_mstr","label":"Date"},
    {"from":"cald_det","to":"calm_mstr","label":"Reference"},
    {"from":"cals_det","to":"calm_mstr","label":"Shift Date"}
  ]
}

Notes
-----
- This is a heuristic extractor intended to work across common ERD-like diagrams exported
  from tools (Visio, Draw.io, Lucidchart, PowerPoint, etc.). You can tune thresholds below.
- It degrades gracefully when a page has no vector drawings: returns None.
- It avoids throwing if optional dependencies are missing.

Dependencies
------------
- PyMuPDF (fitz) is required.
- pytesseract & Pillow are optional (only if you use OCR helpers here).

Usage
-----
from extractors.diagram_extractor import DiagramExtractor

dex = DiagramExtractor(enable_ocr=False)
diag = dex.parse_vector_erd(page)           # -> Optional[dict]
txt  = dex.diagram_json_to_text(diag, "file.pdf", 12) if diag else ""
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import fitz  # PyMuPDF


# Optional OCR support
try:
    import pytesseract  # type: ignore
    from PIL import Image  # type: ignore
    _OCR_AVAILABLE = True
except Exception:
    _OCR_AVAILABLE = False


# ---------------------------------------------------------------------------
# Local helpers (kept self-contained so module can be used stand-alone)
# ---------------------------------------------------------------------------

def _normalize_text(txt: Optional[str]) -> str:
    if not txt:
        return ""
    return " ".join(txt.split())


def _rect_tuple(obj: Any) -> Optional[Tuple[float, float, float, float]]:
    """
    Normalize fitz.Rect or dict-like with 'rect'/'bbox' to (x0,y0,x1,y1).
    """
    if obj is None:
        return None
    if isinstance(obj, fitz.Rect):
        r = obj
        return (float(r.x0), float(r.y0), float(r.x1), float(r.y1))
    # dict-like
    rect = None
    if isinstance(obj, dict):
        if obj.get("rect") is not None:
            rect = obj.get("rect")
        elif obj.get("bbox") is not None:
            rect = obj.get("bbox")
    if isinstance(rect, fitz.Rect):
        r = rect
        return (float(r.x0), float(r.y0), float(r.x1), float(r.y1))
    if isinstance(rect, (tuple, list)) and len(rect) == 4:
        x0, y0, x1, y1 = rect
        return (float(x0), float(y0), float(x1), float(y1))
    return None


def _center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return (0.5 * (x0 + x1), 0.5 * (y0 + y1))


def _pt_dist2(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _bbox_contains(outer: Tuple[float, float, float, float],
                   inner: Tuple[float, float, float, float],
                   pad: float = 1.0) -> bool:
    x0, y0, x1, y1 = outer
    a0, b0, a1, b1 = inner
    return (a0 >= x0 - pad) and (b0 >= y0 - pad) and (a1 <= x1 + pad) and (b1 <= y1 + pad)


def _midpoint(p0: Tuple[float, float], p1: Tuple[float, float]) -> Tuple[float, float]:
    return (0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1]))


# ---------------------------------------------------------------------------
# Data classes for internal reasoning (optional but handy)
# ---------------------------------------------------------------------------

@dataclass
class DiagramNode:
    id: str
    bbox: Tuple[float, float, float, float]


@dataclass
class DiagramEdge:
    source: str
    target: str
    label: Optional[str] = None


# ---------------------------------------------------------------------------
# DiagramExtractor
# ---------------------------------------------------------------------------

class DiagramExtractor:
    """
    Vector-based diagram (ERD-like) extractor with optional OCR helpers.

    Parameters
    ----------
    enable_ocr : bool
        Enable OCR helpers (not used by parse_vector_erd itself, but available via ocr_page_text()).
    ocr_dpi : int
        DPI for rasterization before OCR.
    max_node_label_dist : float
        Max center-to-center distance (in page units) to associate a nearby text block with a node box.
    max_edge_label_dist : float
        Max distance to associate text as an edge label near the midpoint of a line.
    max_endpoint_snap_dist : float
        Max distance to snap a line endpoint to the nearest node center.
    top_title_scan : int
        How many top-most text blocks to scan when inferring a title.
    min_label_len : int
        Minimum label length to consider it meaningful.
    """

    def __init__(
        self,
        enable_ocr: bool = True,
        ocr_dpi: int = 200,
        max_node_label_dist: float = 36.0,
        max_edge_label_dist: float = 48.0,
        max_endpoint_snap_dist: float = 64.0,
        top_title_scan: int = 6,
        min_label_len: int = 2,
    ) -> None:
        self.enable_ocr = bool(enable_ocr and _OCR_AVAILABLE)
        self.ocr_dpi = int(ocr_dpi)
        self.max_node_label_dist = float(max_node_label_dist)
        self.max_edge_label_dist = float(max_edge_label_dist)
        self.max_endpoint_snap_dist = float(max_endpoint_snap_dist)
        self.top_title_scan = int(top_title_scan)
        self.min_label_len = int(min_label_len)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def parse_vector_erd(self, page: fitz.Page) -> Optional[Dict[str, Any]]:
        """
        Parse a single PDF page (fitz.Page) for a diagram structure.

        Returns:
            dict with "title", "nodes", "edges" OR None if no vector diagram found.
        """
        # 1) Gather vector drawings
        try:
            drawings = page.get_drawings() or []
        except Exception:
            drawings = []

        if not drawings:
            return None  # no vector content, likely not a diagram

        # 2) Gather text blocks with coordinates (used for labels)
        blocks = self._page_text_blocks(page)

        # 3) Collect rectangles (node candidates) and lines/curves (edge candidates)
        rects: List[Tuple[float, float, float, float]] = []
        lines: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []

        for d in drawings:
            dtype = d.get("type")
            if dtype in ("rect",) or d.get("rect") is not None or d.get("bbox") is not None:
                r = _rect_tuple(d)
                if r:
                    rects.append(r)
            elif dtype in ("line", "curve"):
                pts = d.get("points") or []
                if len(pts) >= 2:
                    # Use first and last as endpoints
                    p0 = (float(pts[0][0]), float(pts[0][1]))
                    p1 = (float(pts[-1][0]), float(pts[-1][1]))
                    lines.append((p0, p1))

        if not rects:
            # Some diagrams use rounded rects drawn as curves; fall back to None if no boxes found
            # (enhancement: detect polygons approximating rectangles)
            return None

        # 4) Build nodes: label rectangles by inside text; fallback to nearest text
        nodes: List[DiagramNode] = []
        for r in rects:
            label = self._label_inside(blocks, r)
            if not label:
                label = self._nearest_text(blocks, r, max_dist=self.max_node_label_dist)
            if label:
                node_id = _normalize_text(label).split()[0]  # ERDs often use short identifiers (e.g., so_mstr)
                if len(node_id) >= self.min_label_len:
                    nodes.append(DiagramNode(id=node_id, bbox=r))

        if not nodes:
            return None

        # Precompute node centers for snapping
        centers = [(n.id, _center(n.bbox)) for n in nodes]

        def _snap_endpoint(pt: Tuple[float, float]) -> Optional[str]:
            best = (1e18, None)
            for nid, c in centers:
                d2 = _pt_dist2(pt, c)
                if d2 < best[0]:
                    best = (d2, nid)
            # compare sqrt distance to threshold (work in squared space)
            if best[0] <= self.max_endpoint_snap_dist ** 2:
                return best[1]
            return None

        # 5) Build edges by snapping line endpoints to nearest nodes
        edges: List[DiagramEdge] = []
        for p0, p1 in lines:
            src = _snap_endpoint(p0)
            dst = _snap_endpoint(p1)
            if src and dst and src != dst:
                # Find an edge label near the line midpoint
                mid = _midpoint(p0, p1)
                elabel = self._nearest_text_point(blocks, mid, max_dist=self.max_edge_label_dist)
                if elabel and len(elabel) < 200:
                    elabel = elabel  # already normalized in helper
                else:
                    elabel = None
                edges.append(DiagramEdge(source=src, target=dst, label=elabel))

        # 6) Title inference: pick top-most plausible text
        title = self._infer_title(blocks, top_n=self.top_title_scan)

        return {
            "title": title or "",
            "nodes": [{"id": n.id} for n in nodes],
            "edges": [{"from": e.source, "to": e.target, "label": e.label} for e in edges],
        }

    def diagram_json_to_text(self, diag: Dict[str, Any], file_name: str, page_no: int) -> str:
        """
        Convert a diagram JSON into a compact RAG-friendly text summary.
        """
        nodes = ", ".join(n.get("id", "") for n in diag.get("nodes", []) if n.get("id"))
        edges = "; ".join(
            f"{e.get('from')} -> {e.get('to')}" + (f" ({e.get('label')})" if e.get("label") else "")
            for e in diag.get("edges", [])
            if e.get("from") and e.get("to")
        )
        lines = [
            f"--- DOCUMENT CHUNK (ERD) ---",
            f"SOURCE: {file_name} (Page {page_no})",
            f"CATEGORY: erd",
            f"TITLE: {_normalize_text(diag.get('title',''))}",
            f"NODES: {nodes}",
            f"EDGES: {edges}",
        ]
        return "\n".join(lines)

    # Optional OCR helper (not used by parse_vector_erd, but available if you need it)
    def ocr_page_text(self, page: fitz.Page, dpi: Optional[int] = None) -> str:
        if not (self.enable_ocr and _OCR_AVAILABLE):
            return ""
        dpi = dpi or self.ocr_dpi
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        try:
            img = Image.open(io.BytesIO(pix.tobytes("png")))  # type: ignore
            txt = pytesseract.image_to_string(img)  # type: ignore
        except Exception:
            txt = ""
        return _normalize_text(txt)

    # -----------------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------------

    def _page_text_blocks(self, page: fitz.Page) -> List[Tuple[float, float, float, float, str]]:
        """
        Return page text blocks as a list of (x0,y0,x1,y1,text), normalized.
        """
        out: List[Tuple[float, float, float, float, str]] = []
        try:
            blocks = page.get_text("blocks") or []
        except Exception:
            blocks = []
        for b in blocks:
            if len(b) >= 5:
                x0, y0, x1, y1, txt = b[0], b[1], b[2], b[3], b[4]
                t = _normalize_text(txt)
                if t:
                    out.append((float(x0), float(y0), float(x1), float(y1), t))
        return out

    def _label_inside(self,
                      blocks: Sequence[Tuple[float, float, float, float, str]],
                      box: Tuple[float, float, float, float]) -> Optional[str]:
        """
        Choose the longest text fully inside the given box.
        """
        best = ("", 0)
        for x0, y0, x1, y1, t in blocks:
            if _bbox_contains(box, (x0, y0, x1, y1), pad=1.5):
                n = len(t)
                if n > best[1]:
                    best = (t, n)
        return best[0] if best[1] > 0 else None

    def _nearest_text(self,
                      blocks: Sequence[Tuple[float, float, float, float, str]],
                      box: Tuple[float, float, float, float],
                      max_dist: float) -> Optional[str]:
        """
        Find the text whose center is nearest to the box center, within max_dist.
        """
        if not blocks:
            return None
        cx, cy = _center(box)
        best = (1e18, None)
        for x0, y0, x1, y1, t in blocks:
            tc = ((x0 + x1) * 0.5, (y0 + y1) * 0.5)
            d2 = _pt_dist2((cx, cy), tc)
            if d2 < best[0]:
                best = (d2, t)
        if best[0] <= max_dist ** 2:
            return best[1]
        return None

    def _nearest_text_point(self,
                            blocks: Sequence[Tuple[float, float, float, float, str]],
                            pt: Tuple[float, float],
                            max_dist: float) -> Optional[str]:
        """
        Find the nearest text block center to a point, within max_dist.
        """
        if not blocks:
            return None
        best = (1e18, None)
        for x0, y0, x1, y1, t in blocks:
            tc = ((x0 + x1) * 0.5, (y0 + y1) * 0.5)
            d2 = _pt_dist2(pt, tc)
            if d2 < best[0]:
                best = (d2, t)
        if best[0] <= max_dist ** 2:
            return best[1]
        return None

    def _infer_title(self,
                     blocks: Sequence[Tuple[float, float, float, float, str]],
                     top_n: int = 6) -> Optional[str]:
        """
        Pick a plausible title from the top-most blocks (short-ish text).
        """
        if not blocks:
            return None
        top_blocks = sorted(blocks, key=lambda b: b[1])[:max(1, top_n)]
        for _, _, _, _, t in top_blocks:
            tt = _normalize_text(t)
            if 3 <= len(tt) <= 80:
                return tt
        return None
