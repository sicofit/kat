"""The agent loop: plan, call a tool, read the real output, decide, repeat."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Protocol

from .client import LLMClient, LLMError
from .config import Config
from .session import Session
from .tools import Executor, Registry, ToolResult, ToolSpec

# Capture everything between a fence and its close (or the end of the reply, if
# the model forgot to close the fence). The body may hold one action object,
# several separated by commas or newlines, or a JSON array of them -
# _iter_json_values sorts that out. A greedy-brace regex here would mis-split on
# the braces nested inside "args", so the real JSON parser does the splitting.
_ACTION_BLOCK = re.compile(r"```(?:action|json)?\s*(.*?)(?:```|$)", re.DOTALL)
_DECODER = json.JSONDecoder()

# A reply may carry a few independent probes (several paths or indices). They
# run in order, but the count per turn is capped so one over-eager reply cannot
# flood the context window or the approval prompt before the model re-plans.
MAX_ACTIONS_PER_TURN = 4


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
        self._seen: set[str] = set()

    def system_prompt(self, persona: str) -> str:
        return f"{persona.rstrip()}\n\n{self._protocol()}"

    def run(self, session: Session, user_input: str) -> str:
        self.reporter.event("ai", "request received")
        self._seen = set()
        session.add("user", user_input)
        for step in range(1, self.config.max_steps + 1):
            reply = self._think(session)
            session.add("assistant", reply)
            actions = parse_actions(reply)
            if not actions:
                if _looks_like_action_attempt(reply):
                    self.reporter.event("warn", "malformed action - asking for valid JSON")
                    session.add("user", _observe(
                        "kat",
                        "That looked like a tool call but its JSON could not be parsed "
                        "(watch for a stray or missing brace). Re-send a valid fenced "
                        "action block:\n"
                        '```action\n{"tool": "<name>", "args": {...}, "reason": "<why>"}\n```\n'
                        "If you are finished instead, reply in prose with no braces.",
                    ))
                    continue
                self.reporter.event("done", "analysis complete")
                return reply
            for action in actions[:MAX_ACTIONS_PER_TURN]:
                session.add("user", self._act(action))
            if len(actions) > MAX_ACTIONS_PER_TURN:
                dropped = len(actions) - MAX_ACTIONS_PER_TURN
                self.reporter.event("warn", f"{dropped} extra action(s) deferred")
                session.add("user", _observe(
                    "kat",
                    f"Only the first {MAX_ACTIONS_PER_TURN} of your {len(actions)} action "
                    f"blocks ran this turn; the other {dropped} were not executed. Read the "
                    "observations above, then send the remaining steps - a few at a time.",
                ))
        self.reporter.event("warn", f"stopped at step limit ({self.config.max_steps})")
        session.add("user", "[kat] step limit reached")
        return "Reached the step limit before finishing. Run `kat continue` to go further."

    def _think(self, session: Session) -> str:
        self.reporter.thinking()
        budget = self.config.context_chars
        for attempt in range(2):
            window = session.window(budget)
            try:
                return self.client.chat(window, on_token=self.reporter.token).text
            except LLMError as exc:
                if attempt == 0 and _is_context_overflow(exc):
                    self.reporter.event("warn", "context full - trimming and retrying")
                    budget = max(4_000, budget // 2)
                    continue
                raise
        # Unreachable: the loop either returns a reply or re-raises.
        raise AssertionError("unreachable")

    def _act(self, action: Action) -> str:
        spec = self.registry.get(action.tool)
        if spec is None:
            self.reporter.event("error", f"unknown tool: {action.tool}")
            return _observe(action.tool, f"error: unknown tool {action.tool!r}")

        signature = _signature(action)
        if signature in self._seen:
            self.reporter.event("warn", f"{spec.name} - repeat skipped")
            return _observe(
                spec.name,
                "This exact call already ran earlier; its result is above in the "
                "transcript. Running it again will not reveal anything new. Do not "
                "repeat it. Take the next step instead: a different path or index, a "
                "new parameter, another tool, or your final answer if you already "
                "have what you need.",
            )

        self.reporter.event("tool", f"{spec.name} - {action.reason or spec.summary}")
        if self._needs_approval(spec) and not self.approver(spec, action):
            self.reporter.event("warn", f"{spec.name} declined by operator")
            return _observe(spec.name, "operator declined this operation")

        self.reporter.event("run", " ".join(_preview(spec, action)))
        result = self.executor.run(spec, action.args)
        self.reporter.event("result", result.summary())
        if result.ok:
            self._seen.add(signature)
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
            "- Reply with action blocks only, no prose around them. You may send a few in "
            "one reply when the probes are independent (several paths or indices); they run "
            "in order, each returning its own OBSERVATION.\n"
            "- Wait for the OBSERVATION with the real output; never guess what a tool "
            "returned.\n"
            "- When you have enough to answer, reply in prose with no action block.\n\n"
            "Method (work the target like a methodical pentester, one step at a time):\n"
            "- Enumerate before you attack: one full port/service scan, then for a web "
            "service one directory brute-force. Do not re-scan what you already mapped.\n"
            "- Read what comes back. To READ a page - HTML, JSON, JavaScript, any text - "
            "curl it WITHOUT output= so the body lands in the OBSERVATION, and actually use "
            "it: follow every link, path, parameter, username, and version string it reveals. "
            "Your next step must build on that real output, not on a guess. Never run strings "
            "on an HTML/JS/text page: it deletes the short tokens that matter (a link like "
            "/download/0 is only 11 characters and vanishes), so you go blind on the exact "
            "step that held the answer - just read the body.\n"
            "- A number in a URL or path (/data/3, ?id=7, /download/2) is almost always an "
            "insecure direct object reference, and the resource you want usually belongs to "
            "someone else at a LOW index. When you see a view route like /<name>/<id>, read "
            "that page AND immediately request its raw/download SIBLING at the lowest indices "
            "- /download/0 and /download/1 first, then /raw/0, /export/0, /file/0, /pcap/0 - "
            "even though nothing links to them (a bare /download with no id often 404s, so "
            "build the id yourself). Also fetch any .js the app loads and read it for endpoint "
            "paths.\n"
            "- Only a BINARY artefact gets saved with output=<name> - a packet capture, zip, "
            "image, or any download whose OBSERVATION reports a non-text content-type. Open it "
            "with the right reader (tcpdump -r -A for a .pcap, unzip, a strings pass); secrets "
            "live in the raw artefact, not the page. If output= returns a text/* content-type, "
            "you saved a page by mistake: re-fetch it without output= and read it.\n"
            "- The moment you recover a credential, token, key, or hash, USE it: ssh "
            "user@host, ftp://user:pass@host/path, netexec, or the matching web form. "
            "Recovering it is not the goal; what it unlocks is.\n"
            "- You can only run commands ON the target after you have a foothold: a real "
            "credential or key. Do NOT guess SSH/FTP logins or brute-force them - recover a "
            "genuine credential first (from the web app, a packet capture, a config, a "
            "readable file), then log in. For a password use the ssh tool with password=<pw> "
            "(it authenticates via sshpass) or the sshpass tool; for a key use ssh with "
            "key=<path>. Then run your checks in one command: \"id; sudo -l; getcap -r / "
            "2>/dev/null; find / -perm -4000 2>/dev/null\". Tools you run locally inspect THIS "
            "attacker machine, not the target, so they never reveal the target's own "
            "capabilities or SUID binaries.\n"
            "- The flag is a short text token, usually in user.txt or root.txt. Once you "
            "can read files on the target, read it and report it verbatim - never invent "
            "or guess a flag; only report one that appeared in a real OBSERVATION.\n\n"
            f"Tools available on this host:\n{listing}"
        )


def parse_actions(text: str) -> list[Action]:
    """Every well-formed action in a reply, in order.

    A weak model batches independent probes several ways: separate fenced
    blocks, several comma- or newline-separated objects inside one block, or a
    JSON array. All are flattened to a list of actions. Objects that do not
    parse are skipped (the loop nudges separately when a reply clearly tried to
    call a tool but nothing parsed). Falls back to bare, unfenced JSON when
    there is no fence."""
    actions: list[Action] = []
    for block in _ACTION_BLOCK.findall(text):
        actions.extend(_actions_from_values(_iter_json_values(block)))
    if actions:
        return actions
    return _actions_from_values(_iter_json_values(text.strip()))


def parse_action(text: str) -> Action | None:
    """The first action in a reply, or None. Kept for callers that want one."""
    actions = parse_actions(text)
    return actions[0] if actions else None


def _iter_json_values(text: str) -> list[object]:
    """Decode the run of JSON values at the start of ``text``.

    ``raw_decode`` consumes one value at a time, so braces nested inside
    ``args`` never split a value in the wrong place - the bug that made a single
    fenced block of comma-separated actions parse as nothing and sent the loop
    into a malformed-JSON spin. Commas and whitespace between values are
    skipped; parsing stops at the first byte that is not the start of a JSON
    value, leaving any trailing prose or stray tags alone."""
    values: list[object] = []
    index, end = 0, len(text)
    while index < end:
        while index < end and text[index] in " \t\r\n,":
            index += 1
        if index >= end or text[index] not in "{[":
            break
        try:
            value, index = _DECODER.raw_decode(text, index)
        except json.JSONDecodeError:
            break
        values.append(value)
    return values


def _actions_from_values(values: list[object]) -> list[Action]:
    """Flatten decoded JSON values (objects, or arrays of objects) to actions."""
    actions: list[Action] = []
    for value in values:
        items = value if isinstance(value, list) else [value]
        for item in items:
            action = _action_from_obj(item)
            if action is not None:
                actions.append(action)
    return actions


def _action_from_obj(data: object) -> Action | None:
    if not isinstance(data, dict) or "tool" not in data:
        return None
    args = data.get("args") or {}
    if not isinstance(args, dict):
        return None
    return Action(tool=str(data["tool"]), args=args, reason=str(data.get("reason", "")))


def _is_context_overflow(error: Exception) -> bool:
    """True when the endpoint rejected the prompt for exceeding its context.

    Lets the loop shrink the window and retry instead of aborting the mission
    on a recoverable sizing error."""
    text = str(error).lower()
    return "exceed_context_size" in text or ("context" in text and "exceed" in text)


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


def _signature(action: Action) -> str:
    return action.tool + "\x1f" + json.dumps(action.args, sort_keys=True, default=str)


def _looks_like_action_attempt(text: str) -> bool:
    """True when a reply that failed to parse was clearly trying to call a tool
    (a fenced action block, or JSON with a "tool" key) rather than a final prose
    answer - so the loop can ask for valid JSON instead of ending on a typo."""
    return "```action" in text or ("{" in text and '"tool"' in text)


def _preview(spec: ToolSpec, action: Action) -> list[str]:
    try:
        return spec.argv(action.args)
    except Exception:
        return [spec.binary, *(f"{k}={v}" for k, v in action.args.items())]


def _observe(tool: str, body: str) -> str:
    return f"OBSERVATION [{tool}]\n{body}"
