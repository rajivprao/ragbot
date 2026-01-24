
# -*- coding: utf-8 -*-
"""
config/settings.py

Central settings for EnterpriseRAG.
- Paths for Qdrant embedded storage
- Vector DB collection alias
- Embedding model/dimension
- LLM (OpenRouter) configuration
- Utility to load the API key from a file or environment
"""

from __future__ import annotations

import os
from typing import Optional

import sys, os 

sys.path.append(os.path.dirname(__file__)) # add root

# -------------------------------
# Vector DB (Qdrant embedded)
# -------------------------------
QDRANT_PATH: str = os.environ.get("QDRANT_PATH", "qdrant_data")
COLLECTION_ALIAS: str = os.environ.get("COLLECTION_ALIAS", "enterprise_corpus")

# -------------------------------
# Embeddings (HF via LangChain)
# all-MiniLM-L6-v2 → 384 dims
# -------------------------------
EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIM: int = int(os.environ.get("EMBEDDING_DIM", 384))
DISTANCE: str = os.environ.get("EMBEDDING_DISTANCE", "cosine")  # cosine | dot | euclid

# -------------------------------
# LLM (OpenRouter)
# -------------------------------
OPENROUTER_MODEL: str = os.environ.get("OPENROUTER_MODEL", "mistralai/devstral-2512:free")
OPENROUTER_BASE_URL: str = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")

# API key file (preferred) or env var
OPENROUTER_KEY_FILE: str = os.environ.get("OPENROUTER_KEY_FILE", "config/openrouter_key.txt")
OPENROUTER_API_KEY_ENV: str = "OPENROUTER_API_KEY"


def load_openrouter_key() -> Optional[str]:
    """
    Load OpenRouter key from file (OPENROUTER_KEY_FILE) or env (OPENROUTER_API_KEY).
    """
    # 1) Env var wins if set explicitly
    key_env = os.environ.get(OPENROUTER_API_KEY_ENV)
    if key_env:
        return key_env.strip()

    # 2) Try key file
    if OPENROUTER_KEY_FILE and os.path.exists(OPENROUTER_KEY_FILE):
        try:
            with open(OPENROUTER_KEY_FILE, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            return None
    return None
