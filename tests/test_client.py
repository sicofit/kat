"""Tests for the endpoint health probe in :class:`kat.client.LLMClient`.

The probe decides whether every command may run, so it must read a reachable
OpenAI-compatible server as up regardless of backend -- llama.cpp, Ollama, or
vLLM -- and only report "down" when nothing usable is listening.
"""

from __future__ import annotations

import urllib.error

from kat.client import LLMClient
from kat.config import Config


class _FakeResp:
    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _client(base_url: str = "http://127.0.0.1:9932") -> LLMClient:
    return LLMClient(Config(base_url=base_url))


def _patch_urlopen(monkeypatch, handler):
    seen: list[str] = []

    def fake(url, timeout=None):
        seen.append(url)
        return handler(url)

    monkeypatch.setattr("kat.client.urllib.request.urlopen", fake)
    return seen


def test_health_true_when_v1_models_answers(monkeypatch):
    seen = _patch_urlopen(monkeypatch, lambda url: _FakeResp(200))
    assert _client().health() is True
    # The OpenAI-standard path is probed first and is enough on its own.
    assert seen == ["http://127.0.0.1:9932/v1/models"]


def test_health_falls_back_to_health_path(monkeypatch):
    def handler(url):
        if url.endswith("/v1/models"):
            raise urllib.error.URLError("no models route")
        return _FakeResp(200)

    seen = _patch_urlopen(monkeypatch, handler)
    assert _client().health() is True
    assert seen == [
        "http://127.0.0.1:9932/v1/models",
        "http://127.0.0.1:9932/health",
    ]


def test_health_true_when_server_requires_auth(monkeypatch):
    def handler(url):
        raise urllib.error.HTTPError(url, 401, "Unauthorized", {}, None)

    _patch_urlopen(monkeypatch, handler)
    # A 401 means a server is there and answering -- reachable, not down.
    assert _client().health() is True


def test_health_false_when_nothing_listens(monkeypatch):
    def handler(url):
        raise urllib.error.URLError("connection refused")

    _patch_urlopen(monkeypatch, handler)
    assert _client().health() is False


def test_health_false_when_every_probe_is_a_server_error(monkeypatch):
    def handler(url):
        raise urllib.error.HTTPError(url, 503, "Service Unavailable", {}, None)

    _patch_urlopen(monkeypatch, handler)
    # A 5xx on every probe path means the server is not usable.
    assert _client().health() is False
