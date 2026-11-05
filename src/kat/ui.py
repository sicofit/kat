"""Terminal presentation layer built on :mod:`rich`.

Everything the user sees on the console is rendered here, so the rest of the
package can stay free of formatting concerns. The module owns a single shared
:class:`~rich.console.Console` and a small, consistent colour vocabulary:

- cyan      - system information and structural labels
- green     - successful steps and final results
- yellow    - warnings and approval prompts
- red       - errors
- magenta   - agent status (planning, thinking)
- dim/grey  - metadata and secondary detail

:class:`RichReporter` implements the agent's ``Reporter`` protocol, turning the
loop's events and streamed tokens into tagged, colour-coded lines.
"""

from __future__ import annotations

from typing import Mapping

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

THEME = Theme(
    {
        "info": "cyan",
        "success": "green",
        "warning": "yellow",
        "error": "bold red",
        "status": "magenta",
        "meta": "dim",
        "label": "cyan",
    }
)

# A single console shared across the whole CLI. ``highlight=False`` keeps output
# predictable: rich will not re-colour numbers, paths, or URLs on its own.
console = Console(theme=THEME, highlight=False)


# --- event reporting -------------------------------------------------------

# Map each agent event kind to a short tag and a theme style. Kinds not listed
# here fall back to a neutral grey tag so unexpected events still render.
_EVENT_TAGS: dict[str, tuple[str, str]] = {
    "ai": ("AI", "status"),
    "plan": ("PLAN", "status"),
    "tool": ("TOOL", "info"),
    "approval": ("APPROVAL", "warning"),
    "run": ("RUN", "info"),
    "result": ("RESULT", "success"),
    "warn": ("WARN", "warning"),
    "error": ("ERROR", "error"),
    "done": ("DONE", "success"),
}

# Width the bracketed tag is padded to, so message text lines up in a column.
# ``[APPROVAL]`` is the longest tag at ten characters.
_TAG_WIDTH = 10


def _tag_for(kind: str) -> tuple[str, str]:
    return _EVENT_TAGS.get(kind, (kind.upper(), "meta"))


def print_event(kind: str, message: str, *, console: Console | None = None) -> None:
    """Print one tagged, colour-coded status line, for example ``[TOOL] nmap``.

    The message is rendered as literal text, never interpreted as rich markup,
    so dynamic content such as commands or URLs cannot disturb the layout.
    """
    con = console or globals()["console"]
    label, style = _tag_for(kind)
    line = Text()
    line.append(f"[{label}]".ljust(_TAG_WIDTH), style=style)
    line.append(" ")
    line.append(message or "")
    con.print(line)


class RichReporter:
    """Render agent events and streamed tokens to the console.

    Implements the agent's ``Reporter`` protocol (``event`` and ``token``).
    Streamed tokens are written inline in a dim style so they read as the live
    trace; :meth:`finish` closes a partial line cleanly before anything else is
    printed. Any pending stream is also flushed automatically whenever an event
    arrives, so tags never land in the middle of a streamed line.
    """

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or globals()["console"]
        self._streaming = False
        self._line_open = False
        self._status = None

    def thinking(self, label: str = "thinking…") -> None:
        """Show a discreet spinner while the model composes its reply.

        The spinner stops as soon as the first token arrives, or when the next
        event or the final answer is printed, so it never overlaps real output.
        """
        if self._status is not None or self._streaming:
            return
        try:
            self._status = self.console.status(
                Text(label, style="status"), spinner="dots"
            )
            self._status.start()
        except Exception:
            self._status = None

    def event(self, kind: str, message: str) -> None:
        self._stop_status()
        self._end_stream()
        print_event(kind, message, console=self.console)

    def token(self, text: str) -> None:
        if not text:
            return
        self._stop_status()
        self._streaming = True
        self.console.print(
            text, end="", style="meta", markup=False, highlight=False, soft_wrap=True
        )
        try:
            self.console.file.flush()
        except Exception:
            pass
        self._line_open = not text.endswith("\n")

    def finish(self) -> None:
        """Close any open streamed line. Safe to call more than once."""
        self._stop_status()
        self._end_stream()

    def _stop_status(self) -> None:
        if self._status is not None:
            try:
                self._status.stop()
            except Exception:
                pass
            self._status = None

    def _end_stream(self) -> None:
        if self._streaming:
            if self._line_open:
                self.console.print()
            self._streaming = False
            self._line_open = False


# --- banner ----------------------------------------------------------------

_BANNER_KEYS = (
    "MODEL",
    "ENGINE",
    "GPU",
    "STATUS",
    "TOOLS",
    "PLATFORM",
    "VERSION",
    "ENDPOINT",
)

_HEALTHY_STATUS = {"ready", "online", "ok", "healthy", "up"}


def _status_style(value: str) -> str:
    low = value.lower()
    if low in _HEALTHY_STATUS:
        return "success"
    if low in ("-", "", "unknown"):
        return "meta"
    return "error"


def banner(info: Mapping[str, object]) -> None:
    """Render a bordered panel summarising the resolved configuration.

    Values are taken verbatim from ``info``; a missing or empty value is shown
    as ``-`` rather than guessed.
    """
    grid = Table.grid(padding=(0, 2))
    grid.add_column(justify="right", style="label", no_wrap=True)
    grid.add_column(justify="left")
    for key in _BANNER_KEYS:
        raw = info.get(key)
        value = "-" if raw is None or str(raw) == "" else str(raw)
        style = _status_style(value) if key == "STATUS" else ""
        grid.add_row(key, Text(value, style=style))
    console.print(
        Panel(grid, title="KAT", border_style="info", expand=False, padding=(0, 1))
    )


# --- tables ----------------------------------------------------------------


def tool_table(tools) -> Table:
    """Build a table of the tools available on this host.

    Accepts either a ``Registry`` (its ``available()`` tools are shown) or a
    mapping of name to ``ToolSpec`` already filtered to the available ones.
    """
    specs = tools.available() if hasattr(tools, "available") else dict(tools)
    table = Table(
        title="available tools",
        box=box.SIMPLE_HEAD,
        title_style="info",
        header_style="info",
        expand=False,
    )
    table.add_column("Tool", style="info", no_wrap=True)
    table.add_column("Category", style="meta", no_wrap=True)
    table.add_column("Summary", overflow="fold")
    for name in sorted(specs):
        spec = specs[name]
        table.add_row(spec.name, spec.category, spec.summary)
    if not specs:
        table.caption = "no external tools detected on this host"
        table.caption_style = "meta"
    return table


def sessions_table(rows) -> Table:
    """Build a table of saved sessions from ``(id, updated, messages)`` rows."""
    table = Table(
        title="saved sessions",
        box=box.SIMPLE_HEAD,
        title_style="info",
        header_style="info",
        expand=False,
    )
    table.add_column("ID", style="info", no_wrap=True)
    table.add_column("Updated", style="meta", no_wrap=True)
    table.add_column("Messages", justify="right")
    for session_id, updated, count in rows:
        table.add_row(str(session_id), str(updated), str(count))
    return table


# --- misc helpers ----------------------------------------------------------


def render_markdown(text: str) -> None:
    """Print the model's final answer as rendered Markdown."""
    body = (text or "").strip()
    if not body:
        return
    console.print(Markdown(body))


def rule(title: str = "") -> None:
    """Draw a horizontal rule with an optional title."""
    console.rule(title, style="meta")


def info(message: str) -> None:
    console.print(message, style="info", markup=False, highlight=False)


def success(message: str) -> None:
    console.print(message, style="success", markup=False, highlight=False)


def warning(message: str) -> None:
    console.print(message, style="warning", markup=False, highlight=False)


def error(message: str) -> None:
    console.print(message, style="error", markup=False, highlight=False)


def note(message: str) -> None:
    console.print(message, style="meta", markup=False, highlight=False)


def detail(message: str) -> None:
    """Print a secondary, indented line, used for command previews."""
    console.print("    " + str(message), style="meta", markup=False, highlight=False)
