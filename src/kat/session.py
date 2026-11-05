"""Conversation state: message history, context windowing, persistence."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class Session:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    system: str = ""
    messages: list[dict] = field(default_factory=list)
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        self.updated = time.time()

    def window(self, max_chars: int = 16_000) -> list[dict]:
        """System prompt, the opening objective, and the most recent messages.

        The budget keeps the request inside a modest local context window. The
        first user turn (the task) is pinned so the agent never loses its
        objective when older turns scroll out; the remaining budget is filled
        with the most recent messages.
        """
        head = [{"role": "system", "content": self.system}] if self.system else []
        body = self.messages
        pinned: list[dict] = []
        if body and body[0]["role"] == "user":
            pinned = [body[0]]
            body = body[1:]
        fixed = head + pinned
        budget = max_chars - sum(len(m["content"]) for m in fixed)
        kept: list[dict] = []
        for message in reversed(body):
            budget -= len(message["content"])
            if budget < 0 and kept:
                break
            kept.append(message)
        kept.reverse()
        return fixed + kept

    def reset(self) -> None:
        self.messages.clear()
        self.updated = time.time()

    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{self.id}.json"
        target.write_text(json.dumps(self._raw(), indent=2), encoding="utf-8")
        return target

    def _raw(self) -> dict:
        return {
            "id": self.id,
            "system": self.system,
            "messages": self.messages,
            "created": self.created,
            "updated": self.updated,
        }

    @classmethod
    def load(cls, file: Path) -> "Session":
        data = json.loads(Path(file).read_text(encoding="utf-8"))
        return cls(
            id=data["id"],
            system=data.get("system", ""),
            messages=data.get("messages", []),
            created=data.get("created", time.time()),
            updated=data.get("updated", time.time()),
        )

    @classmethod
    def latest(cls, directory: Path) -> "Session | None":
        directory = Path(directory)
        if not directory.is_dir():
            return None
        files = sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime)
        return cls.load(files[-1]) if files else None
