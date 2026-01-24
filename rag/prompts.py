
# -*- coding: utf-8 -*-
"""
rag/prompts.py

Centralized, role-aware prompt templates and a simple builder
for EnterpriseRAG. Keeps your prompting consistent across the app.

Exports
-------
- DIAGRAM_LEGEND: guidance that helps LLMs read ERD prose/ASCII
- SYSTEM_PROMPT: high-level assistant policy
- DEV_PROMPT, BA_PROMPT, NEW_JOINER_PROMPT: role-specific user prompts
- ROLE_ALIASES: accepted synonyms per role
- build_prompts(query, role, context, include_legend=True) -> dict[str, str]
"""

from __future__ import annotations

from typing import Dict

# ---------------------------------------------------------------------------
# Shared legend (helps smaller models interpret ERD-like prose from diagrams)
# ---------------------------------------------------------------------------

DIAGRAM_LEGEND = (
    "DIAGRAM CONVENTIONS:\n"
    "- Table names are headers (e.g., so_mstr).\n"
    "- Relationship lines: '0..*' means zero-to-many, '1..1' means one-to-one.\n"
    "- Join fields can appear between tables (e.g., so_domain = cm_domain).\n"
    "- Suffixes: _mstr (Master), _det (Detail), _hist (History).\n"
    "- Keys often appear as table.column = other_table.column.\n"
)

# ---------------------------------------------------------------------------
# Global system prompt (kept concise; specifics belong in role prompts)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a role-aware enterprise knowledge assistant.\n"
    "- Always ground answers in the provided context chunks.\n"
    "- If information is missing, clearly state assumptions or say 'Not found in provided context.'\n"
    "- Prefer concise, actionable responses with minimal but sufficient citations.\n"
)

# ---------------------------------------------------------------------------
# Role prompts
# ---------------------------------------------------------------------------

DEV_PROMPT = (
    "You are an intelligent technical assistant.\n"
    "Use ONLY the provided context; do not invent references. Work page-by-page; don't skip sections.\n\n"
    "When extracting entities or relationships, present them in a tabular format with columns:\n"
    "(Entity, Linked Entity, Linking Fields, Cardinality). Capture duplicates without collapsing.\n\n"
    "If the answer spans multiple documents, synthesize clearly; if not found, say so.\n\n"
    "QUERY:\n{query}\n\n"
    "{legend_block}"
    "RETRIEVED CONTEXT:\n{context}\n\n"
    "Output:\n"
    "1) Brief logic (2–4 sentences)\n"
    "2) SQL in one fenced ```sql block (if applicable)\n"
    "3) Assumptions (if any)\n"
    "4) Minimal citations: (source: <file_name>, page: <page_number>)\n"
)

BA_PROMPT = (
    "Role: Business Analyst\n"
    "Provide product/functional insights grounded strictly in the context. No external assumptions.\n"
    "- Summarize relevant features and definitions succinctly.\n"
    "- Propose 3–5 KPIs (name, definition, formula/data mapping to tables/columns) where applicable.\n"
    "- Include brief example queries (SQL-like or pseudo) only if context supports them.\n\n"
    "QUERY:\n{query}\n\n"
    "RETRIEVED CONTEXT:\n{context}\n\n"
    "Output:\n"
    "• Overview\n"
    "• KPIs (name, definition, formula/data mapping)\n"
    "• Example queries (short, optional)\n"
    "• Citations (source: <file_name>, page: <page_number>)\n"
)

NEW_JOINER_PROMPT = (
    "Role: New Joiner\n"
    "Provide clear, step-by-step guidance based ONLY on the provided documentation context.\n"
    "- Use numbered steps, include UI paths, roles/permissions, and common pitfalls if present.\n"
    "- If something is not in context, do not guess—state that it is not available.\n\n"
    "QUERY:\n{query}\n\n"
    "RETRIEVED CONTEXT:\n{context}\n\n"
    "Output:\n"
    "1) Steps\n"
    "2) Notes/Tips\n"
    "3) Citations (source: <file_name>, page: <page_number>)\n"
)

# ---------------------------------------------------------------------------
# Role aliases (so callers can pass flexible labels)
# ---------------------------------------------------------------------------

ROLE_ALIASES: Dict[str, str] = {
    # developer
    "developer": "developer",
    "dev": "developer",
    "engineer": "developer",
    "sql": "developer",
    "data_engineer": "developer",
    "data engineer": "developer",
    # business analyst
    "ba": "ba",
    "business_analyst": "ba",
    "business analyst": "ba",
    "product_analyst": "ba",
    "product analyst": "ba",
    # new joiner
    "new_joiner": "new_joiner",
    "new joiner": "new_joiner",
    "onboarding": "new_joiner",
    "beginner": "new_joiner",
}


def _resolve_role(role: str) -> str:
    role_key = (role or "").strip().lower()
    return ROLE_ALIASES.get(role_key, "developer")


def build_prompts(query: str, role: str, context: str, *, include_legend: bool = True) -> Dict[str, str]:
    """
    Build a {system, user} prompt pair for the specified role.

    Args:
        query:   user question
        role:    one of ROLE_ALIASES (or synonyms)
        context: formatted context blocks (already curated by retriever)
        include_legend: add DIAGRAM_LEGEND to the developer prompt (helps with ERDs)

    Returns:
        dict with keys: "system", "user"
    """
    r = _resolve_role(role)
    legend_block = f"CONTEXT LEGEND:\n{DIAGRAM_LEGEND}\n" if (r == "developer" and include_legend) else ""

    if r == "developer":
        user = DEV_PROMPT.format(query=query, context=context, legend_block=legend_block)
    elif r == "ba":
        user = BA_PROMPT.format(query=query, context=context)
    else:
        user = NEW_JOINER_PROMPT.format(query=query, context=context)

    return {"system": SYSTEM_PROMPT, "user": user}
