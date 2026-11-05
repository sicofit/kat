"""Command-line interface for KAT.

Exposes :func:`main`, the entry point named in ``pyproject.toml``. The default
invocation runs a single task to completion; subcommands cover interactive use,
session management, and inspection of the host and endpoint.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import sys
import tomllib
from datetime import datetime

from . import __version__, ui
from .agent import Action, Agent
from .client import LLMClient, LLMError
from .config import Config
from .profiles import get_profile
from .session import Session
from .tools import ToolSpec, default_registry

SUBCOMMANDS = ("chat", "explain", "tools", "status", "continue", "sessions")

# Global options that consume a following value. Used to find the first
# positional token when deciding whether an invocation names a subcommand or is
# a one-shot task.
_VALUE_OPTS = {
    "--config",
    "--profile",
    "--profile-file",
    "--approval",
    "--base-url",
    "--model",
    "--max-steps",
}


class CliError(Exception):
    """A user-facing error that should end the program with a clear message."""

    def __init__(self, message: str, code: int = 2) -> None:
        super().__init__(message)
        self.code = code


# --- entry point -----------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        return _run(argv)
    except KeyboardInterrupt:
        ui.console.print()
        ui.warning("Interrupted.")
        return 130
    except CliError as exc:
        ui.error(str(exc))
        return exc.code


def _run(argv: list[str]) -> int:
    global_parser = _build_global_parser()
    first = _first_positional(argv)
    if first is not None and first not in SUBCOMMANDS:
        parser = _build_oneshot_parser(global_parser)
    else:
        parser = _build_main_parser(global_parser)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse handled --help or reported a usage error
        return int(exc.code) if exc.code is not None else 0
    handler = getattr(args, "func", None)
    if handler is None:
        parser.print_help()
        return 0
    return handler(args)


# --- argument parsing ------------------------------------------------------


def _build_global_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", metavar="PATH", help="path to a specific config file")
    parser.add_argument("--profile", metavar="NAME", help="built-in system-prompt profile")
    parser.add_argument(
        "--profile-file", metavar="PATH", help="path to a custom system-prompt file"
    )
    parser.add_argument(
        "--approval",
        choices=("none", "active", "destructive"),
        help="when to pause for approval before running a tool",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="approve tool runs automatically (sets approval to none)",
    )
    parser.add_argument("--base-url", metavar="URL", help="base URL of the model endpoint")
    parser.add_argument("--model", metavar="NAME", help="model name to request")
    parser.add_argument(
        "--no-stream", action="store_true", help="disable streaming of tokens"
    )
    parser.add_argument(
        "--max-steps", type=int, metavar="N", help="maximum think/act iterations"
    )
    return parser


def _build_main_parser(global_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kat",
        parents=[global_parser],
        description="KAT - a local-first assistant for security engineering.",
    )
    subparsers = parser.add_subparsers(
        dest="command", metavar="{" + ",".join(SUBCOMMANDS) + "}"
    )

    chat = subparsers.add_parser(
        "chat", parents=[global_parser], help="hold an interactive session"
    )
    chat.set_defaults(func=cmd_chat)

    explain = subparsers.add_parser(
        "explain",
        parents=[global_parser],
        help="explain a command, flag, or output; runs no tools",
    )
    explain.add_argument(
        "text", nargs="*", help="what to explain (wrap code in ***...***)"
    )
    explain.set_defaults(func=cmd_explain)

    tools = subparsers.add_parser(
        "tools", parents=[global_parser], help="list the tools found on this host"
    )
    tools.set_defaults(func=cmd_tools)

    status = subparsers.add_parser(
        "status",
        parents=[global_parser],
        help="check the endpoint and show the resolved configuration",
    )
    status.set_defaults(func=cmd_status)

    cont = subparsers.add_parser(
        "continue", parents=[global_parser], help="resume the most recent session"
    )
    cont.add_argument("text", nargs="*", help="optional follow-up task")
    cont.set_defaults(func=cmd_continue)

    sessions = subparsers.add_parser(
        "sessions", parents=[global_parser], help="list saved sessions"
    )
    sessions.set_defaults(func=cmd_sessions)

    return parser


def _build_oneshot_parser(global_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kat",
        parents=[global_parser],
        description="Run a single task to completion and print the result.",
    )
    parser.add_argument("task", nargs="+", help="the task to perform")
    parser.set_defaults(func=cmd_oneshot)
    return parser


def _first_positional(argv: list[str]) -> str | None:
    """Return the first non-option token, honouring options that take a value.

    This decides routing before argparse runs: a first positional that names a
    subcommand is dispatched as such, anything else is treated as a one-shot
    task. Returns ``None`` when the arguments are options only.
    """
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--":
            return argv[index + 1] if index + 1 < len(argv) else None
        if token.startswith("-") and token != "-":
            if "=" in token:
                index += 1
            elif token in _VALUE_OPTS:
                index += 2
            else:
                index += 1
            continue
        return token
    return None


# --- configuration helpers -------------------------------------------------


def _overrides(args: argparse.Namespace) -> dict:
    return {
        "base_url": args.base_url,
        "model": args.model,
        "profile": args.profile,
        "profile_file": args.profile_file,
        "approval": "none" if args.yes else args.approval,
        "max_steps": args.max_steps,
        "stream": False if args.no_stream else None,
    }


def load_config(args: argparse.Namespace) -> Config:
    try:
        return Config.load(path=args.config, **_overrides(args))
    except (ValueError, OSError, tomllib.TOMLDecodeError) as exc:
        raise CliError(f"Configuration error: {exc}") from exc


def resolve_persona(cfg: Config) -> str:
    try:
        return get_profile(cfg.profile, cfg.profile_file)
    except (ValueError, OSError) as exc:
        raise CliError(f"Profile error: {exc}") from exc


def _build_info(cfg: Config, healthy: bool, registry) -> dict:
    gpu = "nvidia-smi detected" if shutil.which("nvidia-smi") else "-"
    return {
        "MODEL": cfg.model,
        "ENGINE": "OpenAI-compatible",
        "GPU": gpu,
        "STATUS": "ready" if healthy else "offline",
        "TOOLS": str(len(registry.available())),
        "PLATFORM": platform.system() or "-",
        "VERSION": __version__,
        "ENDPOINT": cfg.endpoint,
    }


def _unreachable(cfg: Config) -> None:
    ui.error(
        f"Cannot reach the model at {cfg.endpoint}. "
        "Start your local server and retry."
    )


def _save(session: Session, cfg: Config) -> None:
    try:
        session.save(cfg.sessions_path)
    except OSError as exc:
        ui.warning(f"Could not save session: {exc}")


def _prepare(cfg: Config, args: argparse.Namespace):
    """Build the client, registry, reporter, and agent, and probe the endpoint.

    The persona is resolved separately by each command, after the health gate,
    so a configuration problem never masks an unreachable endpoint.
    """
    client = LLMClient(cfg)
    healthy = client.health()
    registry = default_registry()
    reporter = ui.RichReporter(ui.console)
    assume_yes = bool(getattr(args, "yes", False)) or cfg.approval == "none"
    approver = make_approver(ui.console, assume_yes)
    agent = Agent(client, registry, cfg, approver=approver, reporter=reporter)
    return healthy, registry, reporter, agent


# --- approval --------------------------------------------------------------


def make_approver(console, assume_yes: bool):
    """Return an approver callable for the agent's active-operation gate.

    When the agent is about to run a tool that needs approval, it calls this
    with the tool spec and the planned action. The exact command is shown
    first. ``--yes`` approves automatically; a non-interactive session declines
    by default; otherwise the operator is prompted.
    """

    def approver(spec: ToolSpec, action: Action) -> bool:
        try:
            command = " ".join(spec.argv(action.args))
        except Exception:
            command = " ".join(
                [spec.binary, *(f"{key}={value}" for key, value in action.args.items())]
            )
        reason = action.reason or spec.summary or "no reason given"
        ui.print_event("approval", f"{spec.name}: {reason}", console=console)
        ui.detail(command)

        if assume_yes:
            return True
        if not sys.stdin.isatty():
            ui.print_event(
                "approval",
                "non-interactive session; declining this operation.",
                console=console,
            )
            return False
        try:
            answer = console.input("    [warning]Run this? \\[y/N][/warning] ")
        except EOFError:
            ui.print_event("approval", "no input available; declining.", console=console)
            return False
        approved = answer.strip().lower() in ("y", "yes")
        ui.print_event(
            "approval", "approved." if approved else "declined.", console=console
        )
        return approved

    return approver


# --- commands --------------------------------------------------------------


def cmd_oneshot(args: argparse.Namespace) -> int:
    cfg = load_config(args)
    task = " ".join(args.task).strip()
    if not task:
        raise CliError('Provide a task, for example: kat "Enumerate services on 10.10.10.10."')

    healthy, registry, reporter, agent = _prepare(cfg, args)
    if not healthy:
        _unreachable(cfg)
        return 2

    persona = resolve_persona(cfg)
    session = Session(system=agent.system_prompt(persona))
    ui.banner(_build_info(cfg, healthy, registry))
    try:
        answer = agent.run(session, task)
    except LLMError as exc:
        reporter.finish()
        ui.error(f"The model request failed: {exc}")
        return 2
    reporter.finish()
    ui.rule("answer")
    ui.render_markdown(answer)
    _save(session, cfg)
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    cfg = load_config(args)
    healthy, registry, reporter, agent = _prepare(cfg, args)

    ui.banner(_build_info(cfg, healthy, registry))
    if not healthy:
        _unreachable(cfg)
        return 2

    persona = resolve_persona(cfg)
    session = Session(system=agent.system_prompt(persona))
    ui.note("Type a task. Commands: /tools, /reset, /exit.")
    while True:
        try:
            raw = ui.console.input("[info]kat>[/info] ")
        except EOFError:
            ui.console.print()
            break
        line = raw.strip()
        if not line:
            continue
        command = line.lower()
        if command in ("/exit", "/quit"):
            break
        if command == "/reset":
            session.reset()
            ui.note("Conversation cleared.")
            continue
        if command == "/tools":
            ui.console.print(ui.tool_table(registry))
            continue
        if command == "/help":
            ui.note("Commands: /tools, /reset, /exit. Anything else is sent to the agent.")
            continue
        try:
            answer = agent.run(session, line)
        except LLMError as exc:
            reporter.finish()
            ui.error(f"The model request failed: {exc}")
            continue
        reporter.finish()
        ui.rule("answer")
        ui.render_markdown(answer)
        _save(session, cfg)

    if session.messages:
        _save(session, cfg)
        ui.note("Session saved.")
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    """Explain commands and output without running anything.

    Explain mode is a single model turn with no registry and no action
    protocol, so it cannot execute a tool. It uses the ``explain`` profile
    unless the operator selected one explicitly.
    """
    cfg = load_config(args)
    client = LLMClient(cfg)
    if not client.health():
        _unreachable(cfg)
        return 2
    name = cfg.profile if (args.profile or args.profile_file) else "explain"
    try:
        persona = get_profile(name, cfg.profile_file)
    except (ValueError, OSError) as exc:
        raise CliError(f"Profile error: {exc}") from exc

    reporter = ui.RichReporter(ui.console)
    session = Session(system=persona)
    text = " ".join(args.text).strip() if args.text else ""

    if text:
        ok = _explain_once(client, session, reporter, text)
        _save(session, cfg)
        return 0 if ok else 2

    ui.note(
        "Explain mode - ask about a command, flag, or output. "
        "Wrap code in ***...***. Commands: /reset, /exit."
    )
    while True:
        try:
            raw = ui.console.input("[info]ai>[/info] ")
        except EOFError:
            ui.console.print()
            break
        line = raw.strip()
        if not line:
            continue
        command = line.lower()
        if command in ("/exit", "/quit"):
            break
        if command == "/reset":
            session.reset()
            ui.note("Conversation cleared.")
            continue
        _explain_once(client, session, reporter, line)

    if session.messages:
        _save(session, cfg)
    return 0


def _explain_once(
    client: LLMClient, session: Session, reporter: "ui.RichReporter", text: str
) -> bool:
    """Run one explain turn: spinner while the model thinks, then the answer."""
    session.add("user", text)
    reporter.thinking()
    try:
        result = client.chat(session.window(), stream=False)
    except LLMError as exc:
        reporter.finish()
        ui.error(f"The model request failed: {exc}")
        return False
    reporter.finish()
    session.add("assistant", result.text)
    ui.rule("answer")
    ui.render_markdown(result.text)
    return True


def cmd_tools(args: argparse.Namespace) -> int:
    registry = default_registry()
    available = registry.available()
    ui.console.print(ui.tool_table(available))
    ui.note(f"{len(available)} of {len(registry)} catalog tools available on this host.")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    cfg = load_config(args)
    client = LLMClient(cfg)
    healthy = client.health()
    registry = default_registry()
    ui.banner(_build_info(cfg, healthy, registry))
    return 0 if healthy else 2


def cmd_continue(args: argparse.Namespace) -> int:
    cfg = load_config(args)
    try:
        session = Session.latest(cfg.sessions_path)
    except (OSError, ValueError) as exc:
        raise CliError(f"Could not load the latest session: {exc}") from exc
    if session is None:
        ui.warning('No saved session found. Start one with: kat "<task>" or kat chat.')
        return 1

    text = " ".join(args.text).strip() if args.text else ""
    if not text:
        text = _last_user_turn(session)
        if not text:
            ui.note('Nothing to continue. Provide a task: kat continue "<task>".')
            return 0
        ui.note(f"Re-running the last task: {text}")

    healthy, registry, reporter, agent = _prepare(cfg, args)
    ui.banner(_build_info(cfg, healthy, registry))
    if not healthy:
        _unreachable(cfg)
        return 2

    if not session.system:
        session.system = agent.system_prompt(resolve_persona(cfg))
    try:
        answer = agent.run(session, text)
    except LLMError as exc:
        reporter.finish()
        ui.error(f"The model request failed: {exc}")
        return 2
    reporter.finish()
    ui.rule("answer")
    ui.render_markdown(answer)
    _save(session, cfg)
    return 0


def cmd_sessions(args: argparse.Namespace) -> int:
    cfg = load_config(args)
    directory = cfg.sessions_path
    if not directory.is_dir():
        ui.note("No sessions directory yet.")
        return 0
    files = sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    rows = []
    for path in files:
        try:
            session = Session.load(path)
        except (OSError, ValueError):
            continue
        updated = datetime.fromtimestamp(session.updated).strftime("%Y-%m-%d %H:%M")
        rows.append((session.id, updated, len(session.messages)))
    if not rows:
        ui.note("No saved sessions.")
        return 0
    ui.console.print(ui.sessions_table(rows))
    return 0


def _last_user_turn(session: Session) -> str:
    """The most recent genuine user task, skipping observations and markers."""
    for message in reversed(session.messages):
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if content.startswith("OBSERVATION [") or content == "[kat] step limit reached":
            continue
        return content
    return ""
