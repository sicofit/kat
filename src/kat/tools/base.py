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

# Keys a model commonly adds to an action's args to narrate its intent. They are
# not tool options; a strict-argument tool ignores them instead of erroring.
_META_ARG_KEYS = frozenset({
    "reason", "description", "rationale", "comment", "note", "notes",
    "why", "purpose", "thought", "explanation", "goal", "intent",
})


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
            built = self.builder(dict(values))
            # A builder may redirect to a different binary by returning
            # ``(binary, tail)`` instead of a bare ``tail`` list -- for example the
            # ssh tool runs through sshpass when it is given a password. The
            # override is resolved on PATH like any other binary.
            if isinstance(built, tuple):
                override, built = built
                binary = shutil.which(override) or override
            if not isinstance(built, list):
                raise ToolArgError(f"{self.name}: builder must return a list of strings")
            return [binary, *(str(item) for item in built)]
        known = {spec.name for spec in self.args}
        # A weak model routinely leaks its own rationale into the arg dict (a
        # ``reason``/``description`` field beside the real arguments). Those are
        # not tool options, so ignore them rather than failing the whole step --
        # but still reject any other unknown key so a genuine typo (``port`` for
        # ``ports``) is caught instead of silently dropped.
        unknown = set(values) - known - _META_ARG_KEYS
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
        raw_out = proc.stdout or ""
        if _looks_binary(raw_out):
            raw_out = (
                "[binary output - showing its printable strings, as `strings` would; "
                "the raw bytes are not reproduced]\n" + _extract_strings(raw_out)
            )
        stdout, cut_out = _clip(raw_out, self.max_output)
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


_TEXT_CTRL_OK = frozenset("\t\n\r\f\v")
_PRINTABLE_RUN = re.compile(r"[\x20-\x7e]{4,}")


def _looks_binary(text: str) -> bool:
    """A tool that dumps a binary body (a pcap, zip, image, or executable) to
    stdout is useless as raw bytes in a language model's context. Detect that so
    the caller can show printable strings instead. Captured text uses
    errors="replace", so undecodable bytes arrive as U+FFFD; NUL and a high
    density of control/replacement characters mark binary content."""
    if not text:
        return False
    sample = text[:4096]
    if "\x00" in sample:
        return True
    bad = sum(
        1 for ch in sample
        if ch == "�" or (ord(ch) < 0x20 and ch not in _TEXT_CTRL_OK)
    )
    return bad / len(sample) > 0.10


def _extract_strings(text: str) -> str:
    """Printable ASCII runs of four or more characters, like ``strings``. ASCII
    survives the errors="replace" decode intact, so cleartext inside a binary
    body (credentials in a pcap, paths in a binary) is recovered verbatim."""
    return "\n".join(_PRINTABLE_RUN.findall(text))


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
