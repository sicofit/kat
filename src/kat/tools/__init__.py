"""Tool contract, executor, and the default catalog."""

from __future__ import annotations

from .base import (
    ArgSpec,
    Executor,
    Registry,
    ToolArgError,
    ToolResult,
    ToolSpec,
)

__all__ = [
    "ArgSpec",
    "Executor",
    "Registry",
    "ToolArgError",
    "ToolResult",
    "ToolSpec",
    "default_registry",
]


def default_registry() -> Registry:
    """Build the registry from the bundled catalog defined in ``registry.py``."""
    from .registry import build_registry

    return build_registry()
