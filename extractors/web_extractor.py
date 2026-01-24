
# -*- coding: utf-8 -*-
"""
extractors/web_extractor.py

Web extractor with:
- Single-page fetch (URL → cleaned text)
- Optional lightweight site crawl (BFS) with same-domain restriction
- Boilerplate removal (BeautifulSoup) and optional high-quality extraction via trafilatura (if installed)
- Optional robots.txt compliance, polite delay, and custom User-Agent
- Emits LangChain Document objects with consistent metadata

Outputs one Document per URL with metadata:
    file_name (url), source (url), page_number=1, doc_type="web" (or inferred), chunk_kind="text", created_at

Usage:
    from extractors.web_extractor import WebExtractor

    wx = WebExtractor(
        user_agent="EnterpriseRAGBot/1.0",
        timeout=30,
        obey_robots=True,
        sleep_seconds=0.5,
        use_trafilatura=True,         # if installed
        min_text_len=100,
        doc_type_infer=True
    )

    # Single page
    docs = wx.load_url("https://example.com/docs/calendar")

    # Multiple pages
    docs = wx.load_urls(["https://example.com/a", "https://example.com/b"])

    # Small crawl (same domain)
    docs = wx.crawl_site("https://example.com/docs/", max_pages=20, same_domain=True)

Notes:
- This extractor intentionally returns *one* Document per URL. Let your downstream chunker split for retrieval.
- If `trafilatura` is installed (pip install trafilatura lxml), extraction quality improves; otherwise BeautifulSoup fallback is used.
- Respecting robots.txt is enabled by default. Disable it for controlled/internal sites if needed.
"""

from __future__ import annotations

import re
import time
import queue
import logging
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup

try:
    # High-quality text extraction (optional)
    import trafilatura  # type: ignore
    _TRAF_AVAILABLE = True
except Exception:
    _TRAF_AVAILABLE = False

# LangChain Document
try:
    from langchain_core.documents import Document
except Exception:
    class Document:  # minimal shim if langchain is not present at import-time
        def __init__(self, page_content: str, metadata: dict):
            self.page_content = page_content
            self.metadata = metadata

# robots.txt
try:
    import urllib.robotparser as robotparser
    _ROBOTS_AVAILABLE = True
except Exception:
    _ROBOTS_AVAILABLE = False

# Shared helpers
try:
    from vector_store.base import normalize_text, now_iso
except Exception:
    import datetime
    def normalize_text(txt: Optional[str]) -> str:
        return " ".join((txt or "").split())
    def now_iso() -> str:
        return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


# -----------------------------------------------------------------------------
# WebExtractor
# -----------------------------------------------------------------------------

DEFAULT_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en;q=0.9",
    "Cache-Control": "no-cache",
}

ABSOLUTE_SCHEMES = {"http", "https"}


class WebExtractor:
    """
    Fetch and clean web pages into standardized `Document`s.

    Parameters
    ----------
    user_agent : str
        Custom UA string for polite crawling.
    timeout : int
        Requests timeout (seconds).
    obey_robots : bool
        Respect robots.txt (recommended for public sites).
    sleep_seconds : float
        Polite delay between requests.
    use_trafilatura : bool
        Use trafilatura (if installed) for better text extraction.
    min_text_len : int
        Minimum normalized text length to accept a page.
    doc_type_infer : bool
        Infer doc_type from URL tokens.
    strip_tables : bool
        If True, remove <table> content in BS fallback (trafilatura keeps tables as text).
    """

    def __init__(
        self,
        user_agent: str = "EnterpriseRAGBot/1.0 (+https://example.com/bot)",
        timeout: int = 30,
        obey_robots: bool = True,
        sleep_seconds: float = 0.5,
        use_trafilatura: bool = True,
        min_text_len: int = 100,
        doc_type_infer: bool = True,
        strip_tables: bool = False,
    ) -> None:
        self.user_agent = user_agent
        self.timeout = int(timeout)
        self.obey_robots = bool(obey_robots and _ROBOTS_AVAILABLE)
        self.sleep_seconds = float(sleep_seconds)
        self.use_trafilatura = bool(use_trafilatura and _TRAF_AVAILABLE)
        self.min_text_len = int(min_text_len)
        self.doc_type_infer = bool(doc_type_infer)
        self.strip_tables = bool(strip_tables)

        self._robots_cache: Dict[str, robotparser.RobotFileParser] = {}
        self._session = requests.Session()
        self._session.headers.update({**DEFAULT_HEADERS, "User-Agent": self.user_agent})

        self._log = logging.getLogger(self.__class__.__name__)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def load_url(self, url: str) -> List[Document]:
        """
        Fetch a single URL and return a list with one Document (if accepted).
        """
        doc = self._process_url(url)
        return [doc] if doc else []

    def load_urls(self, urls: Iterable[str]) -> List[Document]:
        """
        Fetch multiple URLs (no crawling), return documents for successfully processed pages.
        """
        out: List[Document] = []
        for u in urls:
            d = self._process_url(u)
            if d:
                out.append(d)
            if self.sleep_seconds:
                time.sleep(self.sleep_seconds)
        return out

    def crawl_site(
        self,
        start_url: str,
        *,
        max_pages: int = 30,
        same_domain: bool = True,
        allow_patterns: Optional[List[str]] = None,
        deny_patterns: Optional[List[str]] = None,
    ) -> List[Document]:
        """
        Lightweight BFS crawl starting from start_url.

        Args:
            start_url: Seed URL.
            max_pages: Maximum pages to fetch (safeguard).
            same_domain: Restrict to the same registrable domain.
            allow_patterns: List of regex patterns; only URLs matching ANY are allowed (if provided).
            deny_patterns: List of regex patterns; URLs matching ANY are skipped.

        Returns:
            List[Document] for pages that yielded acceptable text.
        """
        start_url = self._normalize_url(start_url)
        start_host = urlparse(start_url).netloc

        out: List[Document] = []
        seen: Set[str] = set()
        q: "queue.Queue[str]" = queue.Queue()
        q.put(start_url)
        seen.add(start_url)

        while not q.empty() and len(out) < max_pages:
            url = q.get()

            doc = self._process_url(url)
            if doc:
                out.append(doc)

            # Collect next links (only from pages we fetched)
            try:
                html, final_url = self._fetch(url)
                if not html:
                    continue
                links = self._extract_links(html, base=final_url or url)
            except Exception as e:
                self._log.debug(f"[crawl] link extraction failed for {url}: {e}")
                continue

            for link in links:
                norm = self._normalize_url(link)
                if norm in seen:
                    continue
                if not self._url_allowed_by_policy(norm, start_url, same_domain, allow_patterns, deny_patterns):
                    continue
                seen.add(norm)
                if len(seen) > max_pages * 10:
                    # safety bound for pathological pages
                    continue
                q.put(norm)

            if self.sleep_seconds:
                time.sleep(self.sleep_seconds)

        return out

    # -------------------------------------------------------------------------
    # Internals — fetching & processing
    # -------------------------------------------------------------------------

    def _process_url(self, url: str) -> Optional[Document]:
        """
        Fetch, clean, and convert to Document if length is sufficient.
        """
        url = self._normalize_url(url)
        if self.obey_robots and not self._allowed_by_robots(url):
            self._log.info(f"[robots] Disallowed by robots.txt: {url}")
            return None

        try:
            html, final_url = self._fetch(url)
        except Exception as e:
            self._log.warning(f"[fetch] Failed for {url}: {e}")
            return None

        if not html:
            return None

        # Prefer trafilatura when available, else BeautifulSoup fallback
        text, title = self._extract_text_and_title(html, base_url=final_url or url)
        text = normalize_text(text)
        if len(text) < self.min_text_len:
            self._log.debug(f"[filter] Too little text at {final_url or url} (len={len(text)})")
            return None

        # Build metadata
        doc_type = self._infer_doc_type(final_url or url) if self.doc_type_infer else "web"
        meta = {
            "file_name": final_url or url,
            "source": final_url or url,
            "page_number": 1,
            "doc_type": doc_type or "web",
            "chunk_kind": "text",
            "created_at": now_iso(),
            "title": title or "",
        }
        return Document(page_content=text, metadata=meta)

    def _fetch(self, url: str) -> Tuple[str, str]:
        """
        Execute HTTP GET with standard headers and timeout; return (html, final_url).
        """
        resp = self._session.get(url, timeout=self.timeout, allow_redirects=True)
        resp.raise_for_status()
        ct = (resp.headers.get("Content-Type") or "").lower()
        if "text/html" not in ct and "application/xhtml+xml" not in ct and "text/plain" not in ct:
            # not an HTML-ish content; we only index textual pages
            return "", resp.url
        html = resp.text or ""
        return html, resp.url

    # -------------------------------------------------------------------------
    # Internals — extraction & cleaning
    # -------------------------------------------------------------------------

    def _extract_text_and_title(self, html: str, *, base_url: str) -> Tuple[str, str]:
        """
        Extract main text and title from HTML.
        Use trafilatura if available (and enabled), otherwise use BeautifulSoup fallback.
        """
        if self.use_trafilatura and _TRAF_AVAILABLE:
            try:
                # full HTML string; let trafilatura handle cleaning
                extracted = trafilatura.extract(
                    html,
                    include_comments=False,
                    include_tables=True,
                    url=base_url,
                    favor_recall=True,
                    include_links=False,
                    with_metadata=True,
                )
                if isinstance(extracted, str) and extracted.strip():
                    # when with_metadata=False, extract returns a string
                    text = extracted
                    title = self._bs_title(html)  # fallback title
                elif isinstance(extracted, dict) and extracted.get("text"):
                    text = extracted["text"]
                    title = extracted.get("title") or self._bs_title(html)
                else:
                    # fallback to BS if trafilatura returns nothing
                    text, title = self._bs_text_and_title(html)
            except Exception:
                text, title = self._bs_text_and_title(html)
        else:
            text, title = self._bs_text_and_title(html)

        return text, title or ""

    def _bs_text_and_title(self, html: str) -> Tuple[str, str]:
        """
        BeautifulSoup-based boilerplate removal and text extraction.
        """
        soup = BeautifulSoup(html, "html.parser")

        # Remove script/style/nav/footer/header/aside
        for bad in soup(["script", "style", "noscript", "nav", "footer", "header", "aside"]):
            bad.decompose()
        # Optionally strip tables if they tend to be noisy
        if self.strip_tables:
            for t in soup.find_all("table"):
                t.decompose()

        # Remove obvious duplicate nav elements by class/id patterns
        for tag in soup.find_all(True, {"class": re.compile(r"nav|menu|footer|header|cookie", re.I)}):
            tag.decompose()
        for tag in soup.find_all(True, {"id": re.compile(r"nav|menu|footer|header|cookie", re.I)}):
            tag.decompose()

        title = self._bs_title_from_soup(soup)
        text = soup.get_text(separator=" ")
        return normalize_text(text), title

    def _bs_title(self, html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        return self._bs_title_from_soup(soup)

    def _bs_title_from_soup(self, soup: BeautifulSoup) -> str:
        t = ""
        if soup.title and soup.title.string:
            t = soup.title.string.strip()
        if not t:
            # as a fallback, look for <h1> first non-empty
            h1 = soup.find("h1")
            if h1:
                t = normalize_text(h1.get_text())
        return t

    # -------------------------------------------------------------------------
    # Internals — link handling & policies
    # -------------------------------------------------------------------------

    def _extract_links(self, html: str, *, base: str) -> List[str]:
        soup = BeautifulSoup(html, "html.parser")
        out: List[str] = []
        for a in soup.find_all("a", href=True):
            href = a.get("href")
            if not href:
                continue
            abs_url = urljoin(base, href)
            abs_url, _ = urldefrag(abs_url)  # remove fragment
            p = urlparse(abs_url)
            if p.scheme.lower() not in ABSOLUTE_SCHEMES:
                continue
            out.append(abs_url)
        return out

    def _normalize_url(self, url: str) -> str:
        url = url.strip()
        if not url:
            return url
        # Ensure scheme
        p = urlparse(url)
        if not p.scheme:
            url = "https://" + url
            p = urlparse(url)
        # Normalize: lower scheme/host, strip fragment, keep path/query
        url, _ = urldefrag(url)
        p = urlparse(url)
        normalized = f"{p.scheme.lower()}://{p.netloc.lower()}{p.path or '/'}"
        if p.query:
            normalized += f"?{p.query}"
        return normalized

    def _same_registrable_domain(self, url_a: str, url_b: str) -> bool:
        """
        Conservative same-domain check using netloc string compare.
        (For full PSL-based check, integrate 'tldextract' if needed.)
        """
        return urlparse(url_a).netloc.lower().split(":")[0].endswith(
            urlparse(url_b).netloc.lower().split(":")[0]
        )

    def _url_allowed_by_policy(
        self,
        url: str,
        seed_url: str,
        same_domain: bool,
        allow_patterns: Optional[List[str]],
        deny_patterns: Optional[List[str]],
    ) -> bool:
        if same_domain and not self._same_registrable_domain(url, seed_url):
            return False
        if allow_patterns:
            if not any(re.search(pat, url) for pat in allow_patterns):
                return False
        if deny_patterns:
            if any(re.search(pat, url) for pat in deny_patterns):
                return False
        if self.obey_robots and not self._allowed_by_robots(url):
            return False
        return True

    # -------------------------------------------------------------------------
    # Internals — robots.txt
    # -------------------------------------------------------------------------

    def _allowed_by_robots(self, url: str) -> bool:
        if not _ROBOTS_AVAILABLE:
            return True
        p = urlparse(url)
        robots_url = f"{p.scheme}://{p.netloc}/robots.txt"
        rp = self._robots_cache.get(robots_url)
        if rp is None:
            rp = robotparser.RobotFileParser()
            try:
                rp.set_url(robots_url)
                rp.read()
            except Exception:
                # If robots cannot be fetched, default to allow
                rp = robotparser.RobotFileParser()
                rp.parse("User-agent: *\nAllow: /".splitlines())
            self._robots_cache[robots_url] = rp
        return rp.can_fetch(self.user_agent, url)

    # -------------------------------------------------------------------------
    # Internals — doc_type inference
    # -------------------------------------------------------------------------

    def _infer_doc_type(self, url: str) -> str:
        """
        Heuristic doc_type inference from URL path tokens.
        """
        path = urlparse(url).path.lower()
        tokens = re.findall(r"[a-z0-9_]+", path)
        patterns = {
            "technical": {"tech", "technical", "architecture", "design"},
            "schema": {"schema", "tables", "columns", "ddl", "database"},
            "erd": {"erd", "entity", "diagram", "model"},
            "functional": {"functional", "business", "requirements", "brd"},
            "product": {"product", "features", "manual", "documentation", "guide", "docs"},
            "analytics": {"analytics", "kpi", "metric", "dashboard"},
            "reporting": {"report", "reporting"},
            "onboarding": {"onboarding", "getting_started", "introduction"},
            "user_guide": {"user_guide", "help", "how_to", "tutorial"},
            "api": {"api", "interface", "integration", "swagger", "openapi", "reference"},
        }
        for dt, keys in patterns.items():
            if any(k in tokens for k in keys):
                return dt
        return "web"
