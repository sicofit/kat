"""The tool contract and execution layer.

A :class:`ToolSpec` declares a command-line tool and its arguments. The
:class:`Executor` turns a spec plus a dict of values into a validated argument
vector and runs it without a shell, capturing and clipping the output. Nothing
here ever interpolates model text into a shell string.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Callable


class ToolArgError(ValueError):
    """Arguments supplied for a tool failed validation."""


_TARGET_RE = re.compile(r"^[A-Za-z0-9._:\-/@]+$")
_SHELL_META = set(";&|`$<>(){}[]!*?\"'\\ \t\n\r")


@dataclass(slots=True)
class ArgSpec:
    name: str
    kind: str = "option"  # option | flag | positional
    token: str | None = None
    required: bool = False
    choices: tuple[str, ...] | None = None
    default: Any = None
    validate: str | None = None  # target | int | port | None
    help: str = ""

    def render(self, value: Any) -> list[str]:
        if self.kind == "flag":
            return [self.token] if value else []
        value = self._check(value)
        if self.kind == "positional":
            return [str(value)]
        return [self.token, str(value)]

    def _check(self, value: Any) -> Any:
        if self.choices is not None and str(value) not in self.choices:
            raise ToolArgError(
                f"{self.name}: {value!r} is not one of {', '.join(self.choices)}"
            )
        if self.validate == "int":
            try:
                return int(value)
            except (TypeError, ValueError):
                raise ToolArgError(f"{self.name}: expected an integer") from None
        if self.validate == "port":
            return _check_ports(self.name, value)
        if self.validate == "target":
            return _check_target(self.name, value)
        return value


def _check_target(name: str, value: Any) -> str:
    text = str(value).strip()
    if not text or any(ch in _SHELL_META for ch in text) or not _TARGET_RE.match(text):
        raise ToolArgError(f"{name}: {value!r} is not a valid host, address, or URL")
    return text


def _check_ports(name: str, value: Any) -> str:
    text = str(value).strip()
    if not re.fullmatch(r"[0-9,\-]+", text):
        raise ToolArgError(f"{name}: {value!r} is not a valid port specification")
    return text


@dataclass(slots=True)
class ToolSpec:
    name: str
    binary: str
    summary: str
    category: str = "misc"
    args: tuple[ArgSpec, ...] = ()
    active: bool = True
    destructive: bool = False
    builder: Callable[[Mapping[str, Any]], list[str]] | None = None

    def resolve(self) -> str | None:
        return shutil.which(self.binary)

    def available(self) -> bool:
        return self.resolve() is not None

    def argv(self, values: Mapping[str, Any]) -> list[str]:
        binary = self.resolve() or self.binary
        if self.builder is not None:
            tail = self.builder(dict(values))
            if not isinstance(tail, list):
                raise ToolArgError(f"{self.name}: builder must return a list of strings")
            return [binary, *(str(item) for item in tail)]
        known = {spec.name for spec in self.args}
        unknown = set(values) - known
        if unknown:
            raise ToolArgError(f"{self.name}: unknown argument(s): {', '.join(sorted(unknown))}")
        argv = [binary]
        for spec in self.args:
            present = spec.name in values and values[spec.name] not in (None, "", False)
            if present:
                argv += spec.render(values[spec.name])
            elif spec.required:
                raise ToolArgError(f"{self.name}: missing required argument {spec.name!r}")
            elif spec.kind == "flag" and spec.default:
                argv += spec.render(True)
        return argv

    def arg_help(self) -> str:
        if not self.args:
            return "none"
        return ", ".join(f"{a.name}{'*' if a.required else ''}" for a in self.args)


@dataclass(slots=True)
class ToolResult:
    tool: str
    argv: list[str]
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration: float = 0.0
    truncated: bool = False
    timed_out: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and not self.timed_out and self.returncode == 0

    def summary(self) -> str:
        if self.error:
            return f"error: {self.error}"
        if self.timed_out:
            return f"timed out after {self.duration:.0f}s"
        return f"exit {self.returncode} in {self.duration:.1f}s"


class Executor:
    """Runs a tool spec with argument validation, a timeout, and output limits."""

    def __init__(self, timeout: int = 120, max_output: int = 20_000) -> None:
        self.timeout = timeout
        self.max_output = max_output

    def run(self, spec: ToolSpec, values: Mapping[str, Any]) -> ToolResult:
        if not spec.available():
            return ToolResult(spec.name, [spec.binary], error=f"{spec.binary} is not installed")
        try:
            argv = spec.argv(values)
        except ToolArgError as exc:
            return ToolResult(spec.name, [spec.binary], error=str(exc))

        started = time.perf_counter()
        try:
            proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout, _ = _clip(exc.stdout if isinstance(exc.stdout, str) else "", self.max_output)
            return ToolResult(spec.name, argv, stdout=stdout, duration=self.timeout, timed_out=True)
        except OSError as exc:
            return ToolResult(spec.name, argv, error=f"failed to launch: {exc}")

        duration = time.perf_counter() - started
        stdout, cut_out = _clip(proc.stdout, self.max_output)
        stderr, cut_err = _clip(proc.stderr, self.max_output)
        return ToolResult(
            spec.name,
            argv,
            returncode=proc.returncode,
            stdout=stdout,
            stderr=stderr,
            duration=duration,
            truncated=cut_out or cut_err,
        )


def _clip(text: str | None, limit: int) -> tuple[str, bool]:
    text = text or ""
    if len(text) <= limit:
        return text, False
    removed = len(text) - limit
    return f"{text[: limit - 200]}\n... [truncated {removed} chars] ...\n{text[-200:]}", True


class Registry:
    def __init__(self, tools: Iterable[ToolSpec] = ()) -> None:
        self._tools: dict[str, ToolSpec] = {}
        for tool in tools:
            self.add(tool)

    def add(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def all(self) -> dict[str, ToolSpec]:
        return dict(self._tools)

    def available(self) -> dict[str, ToolSpec]:
        return {name: spec for name, spec in self._tools.items() if spec.available()}

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):
        return iter(self._tools.values())
