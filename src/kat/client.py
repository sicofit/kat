"""Thin client for OpenAI-compatible chat endpoints (llama.cpp, Ollama, ...)."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable

from .config import Config


class LLMError(RuntimeError):
    """The model endpoint could not be reached or returned an error."""


@dataclass(slots=True)
class ChatResult:
    text: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    duration: float = 0.0

    @property
    def tokens_per_sec(self) -> float | None:
        if self.completion_tokens and self.duration > 0:
            return self.completion_tokens / self.duration
        return None


class LLMClient:
    def __init__(self, config: Config) -> None:
        self.config = config

    def health(self) -> bool:
        """True if an OpenAI-compatible server is listening at ``base_url``.

        Probes the OpenAI-standard ``/v1/models`` first -- that is the contract
        KAT actually speaks (it POSTs to ``/v1/chat/completions``) and every
        compatible backend exposes it: llama.cpp, Ollama, and vLLM alike. Falls
        back to llama.cpp's ``/health``. A non-5xx HTTP reply (even 401 or 404)
        still means a server is answering, so a backend that gates ``/v1/models``
        behind auth is not mistaken for being down; only a refused connection or
        a 5xx on every probe path counts as unreachable.
        """
        for path in ("/v1/models", "/health"):
            try:
                with urllib.request.urlopen(self.config.endpoint + path, timeout=5) as resp:
                    if 200 <= resp.status < 300:
                        return True
            except urllib.error.HTTPError as exc:
                if exc.code < 500:
                    return True
            except (urllib.error.URLError, OSError):
                continue
        return False

    def chat(
        self,
        messages: list[dict],
        *,
        stream: bool | None = None,
        on_token: Callable[[str], None] | None = None,
        max_tokens: int | None = None,
        template_kwargs: dict | None = None,
    ) -> ChatResult:
        stream = self.config.stream if stream is None else stream
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "stream": stream,
        }
        limit = max_tokens if max_tokens is not None else self.config.max_tokens
        if limit is not None:
            payload["max_tokens"] = limit
        if template_kwargs:
            payload["chat_template_kwargs"] = template_kwargs
        request = self._request(payload)
        started = time.perf_counter()
        try:
            result = self._read_stream(request, on_token) if stream else self._read_once(request)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise LLMError(f"model endpoint returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise LLMError(
                f"cannot reach model at {self.config.base_url} ({exc}). "
                "Is the local server running?"
            ) from exc
        result.duration = time.perf_counter() - started
        return result

    def _request(self, payload: dict) -> urllib.request.Request:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return urllib.request.Request(
            self.config.endpoint + "/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

    def _read_once(self, request: urllib.request.Request) -> ChatResult:
        with urllib.request.urlopen(request, timeout=self.config.request_timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        return ChatResult(
            text=choice["message"].get("content") or "",
            finish_reason=choice.get("finish_reason"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )

    def _read_stream(
        self, request: urllib.request.Request, on_token: Callable[[str], None] | None
    ) -> ChatResult:
        chunks: list[str] = []
        finish_reason: str | None = None
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        with urllib.request.urlopen(request, timeout=self.config.request_timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choice = (event.get("choices") or [{}])[0]
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    chunks.append(piece)
                    if on_token:
                        on_token(piece)
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                usage = event.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    completion_tokens = usage.get("completion_tokens", completion_tokens)
        return ChatResult(
            text="".join(chunks),
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
