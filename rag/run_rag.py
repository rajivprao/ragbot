
# -*- coding: utf-8 -*-
"""
app/run_rag.py

Minimal runner for EnterpriseRAG:
- Loads settings
- Builds/attaches Qdrant embedded index
- Ingests PDFs/DOCX from a folder
- Optionally ingests a URL
- Performs a sample query

Run:
    python -m app.run_rag --docs_dir DATA/DOCS --url https://example.com/docs/calendar --question "What are the join keys?"
"""

from __future__ import annotations

import argparse
import os
import sys


# Local imports
try:
    from config.settings import (
        QDRANT_PATH,
        COLLECTION_ALIAS,
        EMBEDDING_MODEL,
        EMBEDDING_DIM,
        DISTANCE,
        OPENROUTER_MODEL,
        OPENROUTER_BASE_URL,
        load_openrouter_key,
    )
    from rag.enterprise_rag import EnterpriseRAG
except Exception as e:
    print(f"[FATAL] Import error: {e}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run EnterpriseRAG locally with Qdrant embedded.")
    parser.add_argument("--docs_dir", type=str, default=None, help="Path with PDFs/DOCX to ingest.")
    parser.add_argument("--url", type=str, default=None, help="Optional URL to fetch and ingest.")
    parser.add_argument("--question", type=str, default="What are the join keys for Calendar Master?", help="Question to ask.")
    parser.add_argument("--role", type=str, default="developer", help="developer | ba | new_joiner")
    parser.add_argument("--ocr", action="store_true", help="Enable OCR for PDFs.")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR for PDFs.")
    args = parser.parse_args()

    enable_ocr = False if args.no_ocr else True if args.ocr else True

    api_key = load_openrouter_key()
    if not api_key:
        print("[WARN] OpenRouter API key is empty. Set OPENROUTER_KEY_FILE in config/settings.py or via env.")
        print("      You can still ingest, but ask() will fail without a key.")

    rag = EnterpriseRAG(
        collection_alias=COLLECTION_ALIAS,
        qdrant_path=QDRANT_PATH,
        embedding_model=EMBEDDING_MODEL,
        embedding_dim=EMBEDDING_DIM,
        distance=DISTANCE,
        openrouter_api_key=api_key,
        openrouter_model=OPENROUTER_MODEL,
        openrouter_base_url=OPENROUTER_BASE_URL,
        enable_ocr=enable_ocr,
    )

    if args.docs_dir:
        print(f"[RUN] Ingesting documents from: {args.docs_dir}")
        rag.ingest_path(args.docs_dir, corpus_id="enterprise_docs")

    if args.url:
        print(f"[RUN] Ingesting URL: {args.url}")
        rag.add_web_url(args.url, corpus_id="web_docs")

    # Ask a question (if API key exists)
    if api_key:
        print(f"[RUN] Asking: {args.question}")
        ans = rag.ask(args.question, role=args.role)
        print("\n=== ANSWER ===\n")
        print(ans)
    else:
        print("[INFO] Skipping ask() because API key is not set.")


if __name__ == "__main__":
    main()

