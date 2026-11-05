"""Configuration: layered defaults, config file, environment, and overrides."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

DEFAULT_CONFIG_PATHS = (
    Path("kat.toml"),
    Path.home() / ".kat" / "config.toml",
)

_ENV_PREFIX = "KAT_"
_APPROVAL_POLICIES = ("none", "active", "destructive")


@dataclass(slots=True)
class Config:
    base_url: str = "http://127.0.0.1:9932"
    api_key: str | None = None
    model: str = "local"
    profile: str = "professional"
    profile_file: str | None = None
    approval: str = "active"
    max_steps: int = 12
    temperature: float = 0.4
    request_timeout: int = 180
    tool_timeout: int = 120
    max_tool_output: int = 20_000
    session_dir: str = "~/.kat/sessions"
    stream: bool = True

    def __post_init__(self) -> None:
        if self.approval not in _APPROVAL_POLICIES:
            raise ValueError(
                f"invalid approval policy {self.approval!r}; "
                f"choose one of {', '.join(_APPROVAL_POLICIES)}"
            )

    @property
    def sessions_path(self) -> Path:
        return Path(self.session_dir).expanduser()

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/")

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None, **overrides) -> "Config":
        """Resolve config from file, then KAT_* env vars, then explicit overrides."""
        data: dict = {}
        file = _find_config(path)
        if file is not None:
            data.update(_read_toml(file))
        data.update(_read_env())
        data.update({k: v for k, v in overrides.items() if v is not None})
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def _find_config(path: str | os.PathLike[str] | None) -> Path | None:
    if path is not None:
        candidate = Path(path).expanduser()
        return candidate if candidate.is_file() else None
    for candidate in DEFAULT_CONFIG_PATHS:
        if candidate.is_file():
            return candidate
    return None


def _read_toml(file: Path) -> dict:
    with open(file, "rb") as handle:
        raw = tomllib.load(handle)
    return raw.get("kat", raw)


def _read_env() -> dict:
    out: dict = {}
    for key, value in os.environ.items():
        if key.startswith(_ENV_PREFIX):
            out[key[len(_ENV_PREFIX):].lower()] = _coerce(key, value)
    return out


def _coerce(key: str, value: str):
    name = key[len(_ENV_PREFIX):].lower()
    if name in ("max_steps", "request_timeout", "tool_timeout", "max_tool_output"):
        return int(value)
    if name == "temperature":
        return float(value)
    if name == "stream":
        return value.strip().lower() in ("1", "true", "yes", "on")
    return value
