"""The agent loop: plan, call a tool, read the real output, decide, repeat."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Protocol

from .client import LLMClient
from .config import Config
from .session import Session
from .tools import Executor, Registry, ToolResult, ToolSpec

_ACTION_BLOCK = re.compile(r"```(?:action|json)?\s*(\{.*?\})\s*```", re.DOTALL)


@dataclass(slots=True)
class Action:
    tool: str
    args: dict
    reason: str = ""


class Reporter(Protocol):
    def event(self, kind: str, message: str) -> None: ...
    def token(self, text: str) -> None: ...
    def thinking(self) -> None: ...


class NullReporter:
    def event(self, kind: str, message: str) -> None:
        pass

    def token(self, text: str) -> None:
        pass

    def thinking(self) -> None:
        pass


Approver = Callable[[ToolSpec, "Action"], bool]


def always_approve(spec: ToolSpec, action: "Action") -> bool:
    return True


class Agent:
    def __init__(
        self,
        client: LLMClient,
        registry: Registry,
        config: Config,
        *,
        approver: Approver = always_approve,
        reporter: Reporter | None = None,
    ) -> None:
        self.client = client
        self.registry = registry
        self.config = config
        self.approver = approver
        self.reporter = reporter or NullReporter()
        self.executor = Executor(config.tool_timeout, config.max_tool_output)

    def system_prompt(self, persona: str) -> str:
        return f"{persona.rstrip()}\n\n{self._protocol()}"

    def run(self, session: Session, user_input: str) -> str:
        self.reporter.event("ai", "request received")
        session.add("user", user_input)
        for step in range(1, self.config.max_steps + 1):
            reply = self._think(session)
            session.add("assistant", reply)
            action = parse_action(reply)
            if action is None:
                self.reporter.event("done", "analysis complete")
                return reply
            session.add("user", self._act(action))
        self.reporter.event("warn", f"stopped at step limit ({self.config.max_steps})")
        session.add("user", "[kat] step limit reached")
        return "Reached the step limit before finishing. Run `kat continue` to go further."

    def _think(self, session: Session) -> str:
        self.reporter.thinking()
        return self.client.chat(session.window(), on_token=self.reporter.token).text

    def _act(self, action: Action) -> str:
        spec = self.registry.get(action.tool)
        if spec is None:
            self.reporter.event("error", f"unknown tool: {action.tool}")
            return _observe(action.tool, f"error: unknown tool {action.tool!r}")

        self.reporter.event("tool", f"{spec.name} - {action.reason or spec.summary}")
        if self._needs_approval(spec) and not self.approver(spec, action):
            self.reporter.event("warn", f"{spec.name} declined by operator")
            return _observe(spec.name, "operator declined this operation")

        self.reporter.event("run", " ".join(_preview(spec, action)))
        result = self.executor.run(spec, action.args)
        self.reporter.event("result", result.summary())
        return _observe(spec.name, render_result(result))

    def _needs_approval(self, spec: ToolSpec) -> bool:
        policy = self.config.approval
        if policy == "none":
            return False
        if policy == "destructive":
            return spec.destructive
        return spec.active or spec.destructive

    def _protocol(self) -> str:
        tools = self.registry.available()
        if tools:
            listing = "\n".join(
                f"- {name}: {spec.summary} (args: {spec.arg_help()})"
                for name, spec in sorted(tools.items())
            )
        else:
            listing = "- (no external tools detected on this host)"
        return (
            "To run a tool, reply with a single fenced block and nothing else:\n"
            "```action\n"
            '{"tool": "<name>", "args": {"<arg>": "<value>"}, "reason": "<why>"}\n'
            "```\n"
            "Rules:\n"
            "- Emit the block alone when you want to run a tool, with no prose around it.\n"
            "- You will then receive an OBSERVATION with the real output. Wait for it; "
            "never guess what a tool returned.\n"
            "- When you have enough to answer, reply in prose with no action block.\n\n"
            f"Tools available on this host:\n{listing}"
        )


def parse_action(text: str) -> Action | None:
    match = _ACTION_BLOCK.search(text)
    raw = match.group(1) if match else _bare_json(text)
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or "tool" not in data:
        return None
    args = data.get("args") or {}
    if not isinstance(args, dict):
        return None
    return Action(tool=str(data["tool"]), args=args, reason=str(data.get("reason", "")))


def render_result(result: ToolResult) -> str:
    if result.error:
        return f"error: {result.error}"
    parts = [f"exit code: {result.returncode}" + (" (timed out)" if result.timed_out else "")]
    if result.stdout.strip():
        parts.append(f"stdout:\n{result.stdout.strip()}")
    if result.stderr.strip():
        parts.append(f"stderr:\n{result.stderr.strip()}")
    if not result.stdout.strip() and not result.stderr.strip():
        parts.append("(no output)")
    return "\n".join(parts)


def _bare_json(text: str) -> str | None:
    stripped = text.strip()
    return stripped if stripped.startswith("{") and '"tool"' in stripped else None


def _preview(spec: ToolSpec, action: Action) -> list[str]:
    try:
        return spec.argv(action.args)
    except Exception:
        return [spec.binary, *(f"{k}={v}" for k, v in action.args.items())]


def _observe(tool: str, body: str) -> str:
    return f"OBSERVATION [{tool}]\n{body}"
