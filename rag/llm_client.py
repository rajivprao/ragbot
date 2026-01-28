
# -*- coding: utf-8 -*-
"""
rag/llm_client.py

Lightweight LLM client for EnterpriseRAG (OpenRouter-compatible).

Features
--------
- OpenRouter / OpenAI-compatible Chat Completions API
- System + user messages (with optional extra messages)
- Configurable model, temperature, top_p, max_tokens
- Robust retry with exponential backoff (handles 429/5xx)
- Optional streaming (SSE) generator
- Minimal, dependency-light (requests only)

Usage
-----
    from rag.llm_client import LLMClient, ChatMessage

    llm = LLMClient(
        api_key=os.environ["OPENROUTER_API_KEY"],
        model="mistralai/devstral-2512:free",
        base_url="https://openrouter.ai/api/v1/chat/completions",
        app_name="EnterpriseRAG",
        referer="https://your-app.example.com",
    )

    # 1) Simple call with system + user strings
    text = llm.chat(
        system="You are helpful.",
        user="Write a 2-line SQL that selects 10 rows from table foo.",
        temperature=0.2
    )

    # 2) Advanced: pass an explicit message list, supports roles "system", "user", "assistant"
    messages = [
        ChatMessage(role="system", content="You are a SQL assistant."),
        ChatMessage(role="user", content="Create a query for top 5 products by sales.")
    ]
    text = llm.chat_messages(messages, temperature=0.2)

    # 3) Streaming (generator)
    for chunk in llm.stream_chat(system="You are helpful.", user="Stream a haiku."):
        print(chunk, end="")
"""

from __future__ import annotations

import json
import time
import typing as _t
from dataclasses import dataclass, field

import requests


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class LLMClientError(RuntimeError):
    """Base exception for LLM client errors."""


class LLMRateLimitError(LLMClientError):
    """Raised when rate limits persist after retries."""


class LLMAPIError(LLMClientError):
    """Raised on non-retriable HTTP or API errors."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

@dataclass
class LLMClient:
    """
    Minimal OpenRouter-compatible chat client.
    """
    api_key: str
    model: str = "mistralai/devstral-2512:free"
    base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    timeout: int = 60
    max_retries: int = 4
    backoff_initial: float = 0.8
    backoff_multiplier: float = 1.8
    app_name: str | None = None           # sent as X-Title (optional)
    referer: str | None = None            # sent as HTTP-Referer (optional)
    session: requests.Session = field(default_factory=requests.Session)

    # ---------------------------
    # Public API
    # ---------------------------

    def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
        top_p: float | None = None,
        max_tokens: int | None = None,
        extra_messages: list[ChatMessage] | None = None,
    ) -> str:
        """
        Convenience wrapper around chat_messages() for the common case of 1 system + 1 user.
        """
        msgs = [
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=user),
        ]
        if extra_messages:
            msgs.extend(extra_messages)
        return self.chat_messages(
            messages=msgs,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    def chat_messages(
        self,
        messages: list[ChatMessage] | list[dict],
        *,
        temperature: float = 0.2,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Send a chat completion request with an explicit message list.
        Returns the assistant's final content as a string.
        """
        payload = {
            "model": self.model,
            "messages": [m.to_dict() if isinstance(m, ChatMessage) else dict(m) for m in messages],
            "temperature": float(temperature),
        }
        if top_p is not None:
            payload["top_p"] = float(top_p)
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)

        data = self._request_json(payload, stream=False)
        return self._extract_text(data)

    def stream_chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
        top_p: float | None = None,
        max_tokens: int | None = None,
        extra_messages: list[ChatMessage] | None = None,
        chunk_timeout: int = 65,
    ) -> _t.Iterable[str]:
        """
        Stream the assistant's response as a generator yielding text chunks.

        Note:
            Streaming uses Server-Sent Events (SSE). We parse 'data:' lines that
            contain JSON deltas with 'choices[0].delta.content'.
        """
        msgs = [
            ChatMessage(role="system", content=system),
            ChatMessage(role="user", content=user),
        ]
        if extra_messages:
            msgs.extend(extra_messages)

        payload = {
            "model": self.model,
            "messages": [m.to_dict() if isinstance(m, ChatMessage) else dict(m) for m in msgs],
            "temperature": float(temperature),
            "stream": True,
        }
        if top_p is not None:
            payload["top_p"] = float(top_p)
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)

        # Stream with retry loop. On retryable errors, we resume from scratch.
        for attempt in range(1, self.max_retries + 1):
            try:
                with self._post(payload, stream=True, timeout=chunk_timeout) as resp:
                    self._raise_for_status(resp)
                    for line in resp.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        # SSE lines look like: "data: {...json...}"
                        if line.startswith("data:"):
                            data_str = line[len("data:"):].strip()
                            if data_str == "[DONE]":
                                return
                            try:
                                obj = json.loads(data_str)
                                delta = self._extract_stream_delta(obj)
                                if delta:
                                    yield delta
                            except Exception:
                                # Ignore malformed chunks to keep stream going
                                continue
                return
            except (requests.Timeout, requests.ConnectionError) as e:
                if attempt >= self.max_retries:
                    raise LLMClientError(f"Streaming failed after {attempt} attempts: {e}") from e
                self._sleep_backoff(attempt)
            except LLMRateLimitError as e:
                if attempt >= self.max_retries:
                    raise
                self._sleep_backoff(attempt, rate_limited=True)
            except LLMAPIError as e:
                # Non-retriable
                raise

    # ---------------------------
    # Internals
    # ---------------------------

    def _headers(self) -> dict:
        h = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.referer:
            h["HTTP-Referer"] = self.referer
        if self.app_name:
            h["X-Title"] = self.app_name
        return h

    def _post(self, payload: dict, *, stream: bool, timeout: int | None = None) -> requests.Response:
        return self.session.post(
            self.base_url,
            headers=self._headers(),
            json=payload,
            timeout=timeout or self.timeout,
            stream=stream,
        )

    def _request_json(self, payload: dict, *, stream: bool) -> dict:
        """
        POST with retry/backoff; return parsed JSON on success.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._post(payload, stream=False)
                self._raise_for_status(resp)
                try:
                    return resp.json()
                except Exception as e:
                    raise LLMAPIError(f"Invalid JSON response: {e}") from e
            except requests.Timeout as e:
                if attempt >= self.max_retries:
                    raise LLMClientError(f"Timeout after {attempt} attempts") from e
                self._sleep_backoff(attempt)
            except requests.ConnectionError as e:
                if attempt >= self.max_retries:
                    raise LLMClientError(f"Connection error after {attempt} attempts") from e
                self._sleep_backoff(attempt)
            except LLMRateLimitError as e:
                if attempt >= self.max_retries:
                    raise
                self._sleep_backoff(attempt, rate_limited=True)
            except LLMAPIError:
                # Non-retriable HTTP errors
                raise

        # Shouldn't reach here
        raise LLMClientError("Exhausted retries without a response.")

    def _raise_for_status(self, resp: requests.Response) -> None:
        """
        Normalize HTTP errors and detect rate limits.
        """
        if 200 <= resp.status_code < 300:
            return

        # Try to parse API error message
        msg = f"HTTP {resp.status_code}"
        try:
            j = resp.json()
            if isinstance(j, dict):
                # OpenRouter-like error format
                err = j.get("error") or j.get("message") or j
                msg = f"{msg}: {err}"
        except Exception:
            # fall back to text
            if resp.text:
                msg = f"{msg}: {resp.text[:300]}"

        if resp.status_code == 429:
            raise LLMRateLimitError(msg)
        if 500 <= resp.status_code < 600:
            # treat as retriable
            raise requests.ConnectionError(msg)
        raise LLMAPIError(msg)

    def _extract_text(self, data: dict) -> str:
        """
        Extract the final assistant message text from a non-streaming response.
        Supports OpenRouter/OpenAI-like payloads.
        """
        # Expected: {"choices": [{"message": {"role":"assistant","content":"..."}, ...}]}
        try:
            choices = data.get("choices") or []
            if not choices:
                return ""
            msg = choices[0].get("message") or {}
            content = msg.get("content") or ""
            return str(content).strip()
        except Exception:
            return ""

    def _extract_stream_delta(self, obj: dict) -> str:
        """
        Extract incremental delta text from a streaming (SSE) chunk.
        """
        try:
            choices = obj.get("choices") or []
            if not choices:
                return ""
            delta = choices[0].get("delta") or {}
            content = delta.get("content") or ""
            return str(content)
        except Exception:
            return ""

    def _sleep_backoff(self, attempt: int, *, rate_limited: bool = False) -> None:
        """
        Sleep with exponential backoff. Longer if rate-limited.
        """
        base = self.backoff_initial * (self.backoff_multiplier ** (attempt - 1))
        if rate_limited:
            base *= 1.5
        time.sleep(min(base, 10.0))


#--------------------------------------------------------------------------

    GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


    # ---------------------------
    # Gemini API
    # ---------------------------

    def gemini_chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
        top_p: float | None = None,
        max_tokens: int | None = None,
        model: str = "gemini-1.5-pro",
    ) -> str:
        """
        Simple Gemini chat: system + user → text
        """
        payload = self._gemini_payload(
            system=system,
            user=user,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        url = f"{self.GEMINI_BASE_URL}/gemini-2.5-flash:generateContent"
        data = self._gemini_request_json(url, payload)
        return self._extract_gemini_text(data)

    def _gemini_payload(
        self,
        *,
        system: str,
        user: str,
        temperature: float,
        top_p: float | None,
        max_tokens: int | None,
    ) -> dict:
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user}],
                }
            ],
            "systemInstruction": {
                "parts": [{"text": system}]
            },
            "generationConfig": {
                "temperature": float(temperature),
            },
        }

        gen_cfg = payload["generationConfig"]
        if top_p is not None:
            gen_cfg["topP"] = float(top_p)
        if max_tokens is not None:
            gen_cfg["maxOutputTokens"] = int(max_tokens)

        return payload


    def _gemini_request_json(self, url: str, payload: dict) -> dict:
        """
        Gemini POST with retry/backoff.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.post(
                    url,
                    params={"key": self.api_key},
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=self.timeout,
                )

                if resp.status_code == 429:
                    raise LLMRateLimitError("Gemini rate limit")

                if 500 <= resp.status_code < 600:
                    raise requests.ConnectionError(resp.text)

                if not (200 <= resp.status_code < 300):
                    raise LLMAPIError(f"Gemini HTTP {resp.status_code}: {resp.text[:300]}")

                return resp.json()

            except (requests.Timeout, requests.ConnectionError) as e:
                if attempt >= self.max_retries:
                    raise LLMClientError(f"Gemini failed after {attempt} attempts: {e}") from e
                self._sleep_backoff(attempt)

            except LLMRateLimitError:
                if attempt >= self.max_retries:
                    raise
                self._sleep_backoff(attempt, rate_limited=True)

        raise LLMClientError("Gemini exhausted retries")


    def _extract_gemini_text(self, data: dict) -> str:
        """
        Extract text from Gemini response.
        """
        try:
            candidates = data.get("candidates") or []
            if not candidates:
                return ""

            parts = candidates[0].get("content", {}).get("parts") or []
            texts = [p.get("text", "") for p in parts]
            return "".join(texts).strip()
        except Exception:
            return ""
