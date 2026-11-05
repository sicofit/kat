# Architecture

KAT is a small, deliberately boring agent. A local language model proposes
actions; a validated execution layer runs real tools and returns their real
output; a loop ties the two together. The design goals, in order, are honesty
(the model cannot fabricate results), safety (dangerous steps are gated), and
portability (any OpenAI-compatible model will do).

## Module map

| Module | Responsibility |
| --- | --- |
| `config` | The `Config` dataclass and its layered resolution: defaults, then a TOML file, then `KAT_*` environment variables, then explicit overrides. |
| `client` | `LLMClient`, a thin wrapper over the OpenAI-compatible chat endpoint. Handles streaming and non-streaming responses and a `/health` probe. |
| `session` | `Session`: message history, character-budget context windowing, and JSON persistence under `session_dir`. |
| `tools/base` | The tool contract (`ArgSpec`, `ToolSpec`), the `Executor` that runs a spec, the `ToolResult` it returns, and the `Registry`. |
| `tools/registry` | The bundled catalog: concrete `ToolSpec` definitions and `build_registry()`, assembled into the default `Registry`. |
| `agent` | The `Agent` loop: prompt assembly, action parsing, approval gating, execution, and observation formatting. |
| `cli` | Argument parsing and command dispatch (`kat.cli:main`). Builds the config, client, registry, and agent, then runs the chosen command. The `explain` command is a single model turn with no registry and no action protocol, so it reads and explains but never executes a tool. |
| `ui` | Console rendering built on `rich`: streams tokens, prints loop events, and presents the approval prompt. Supplies the `Reporter` and the `Approver` the agent calls. |

## The execution contract

Everything a tool can do is declared as data, not code paths.

A `ToolSpec` names a `binary`, a one-line `summary`, a `category`, and a tuple
of `ArgSpec`. Two booleans classify it for the approval gate: `active` (reaches
out to a target) and `destructive` (changes state). An `ArgSpec` declares a
single argument: its `kind` (`option`, `flag`, or `positional`), the `token`
it renders to (for example `-p`), whether it is `required`, an optional set of
`choices`, a `default`, and a `validate` rule.

`ToolSpec.argv(values)` turns a dict of supplied values into an argument
vector:

- The binary is resolved with `shutil.which`; an unresolved binary is reported
  as "not installed" rather than executed.
- Unknown argument names are rejected. Missing required arguments are rejected.
- Each value is validated before it is rendered. `target` must match a strict
  character class (letters, digits, and `. _ : - / @`) and must contain **no**
  shell metacharacters; `port` must be digits, commas, and hyphens; `int` is
  coerced to an integer. A failure raises `ToolArgError`.
- A tool with irregular argument shapes can supply a `builder` callable that
  returns the argument list directly; its output is still prefixed with the
  resolved binary and never reaches a shell.

The `Executor` runs the vector with `subprocess.run(..., shell=False)`. It
captures stdout and stderr as UTF-8 (decoding errors replaced, never fatal),
enforces `tool_timeout`, and clips each stream to `max_tool_output`, keeping the
head and tail with a marker in between. Failure modes are returned as a
`ToolResult` with an `error` or `timed_out` flag set, not raised; a tool that
cannot run must not abort the task.

**Why no model text reaches a shell.** There is no shell. Values travel from
the model as JSON, are validated per argument, and are passed to the operating
system as a list. `shell=False` means the metacharacters that matter to a shell
have no interpreter to act on them, and `target` validation rejects them before
execution anyway. Command injection has no surface here because KAT never builds
a command string.

## The agent loop and the action protocol

`Agent.run` appends the user's input to the session and then iterates up to
`max_steps` times:

1. **Think.** The current context window is sent to the model; tokens stream to
   the UI as they arrive.
2. **Parse.** The reply is scanned for a single fenced ```` ```action ```` block
   (a bare top-level JSON object is also accepted). The block carries
   `{"tool", "args", "reason"}`.
3. **Act or finish.** If there is no valid action block, the reply is the final
   answer and the loop ends. Otherwise the named tool is looked up, gated,
   executed, and its output is appended to the history as an `OBSERVATION`
   before the next step.

The structured JSON block is a deliberate trade-off against native
function-calling / tool APIs. Those offer schema validation and cleaner
multi-tool semantics, but they depend on server- and model-specific support
that local runtimes implement unevenly or not at all. A parsed JSON block
depends only on the model's ability to follow a formatting instruction, so the
same loop runs unchanged across llama.cpp, Ollama, vLLM, and whatever comes
next. The cost is that a malformed block is not a tool call: KAT treats "no
valid action" as "the model is done," so the protocol instruction tells the
model to emit the block alone and to wait for the observation. This keeps the
rule simple and the failure mode legible rather than papering over it with
heuristics.

## Approval policies

The approval gate decides which tools require a human yes before they run:

- `none` — nothing is gated. Suitable for fully unattended, pre-scoped runs.
- `active` — any tool marked `active` or `destructive` is gated. This is the
  default.
- `destructive` — only `destructive` tools are gated; read-only enumeration
  runs freely.

The gate is an `Approver` callback the operator provides (the `ui` module binds
it to a prompt; `--yes` binds it to automatic approval). The agent calls it
*before* a gated tool runs and, on refusal, records an observation that the
operator declined so the model can choose another route. This is the important
part: the gate lives outside the model. It is the operator withholding consent,
not the model declining a task, and the model cannot approve its own action.

## Context windowing

`Session.window` returns the system prompt plus as many of the most recent
messages as fit a character budget (48,000 by default), walking newest-first and
always keeping at least one message even if it alone exceeds the budget. The
budget is counted in characters, not tokens: it is cheap, deterministic, and
model-agnostic, which matters when the point is to run against arbitrary local
models. Tool output is the thing that grows a transcript fastest, so trimming
the oldest turns while preserving the system prompt and recent state keeps long
tasks inside a local model's finite context without a tokenizer dependency.

## The honesty invariant

Observations are written by exactly one place: the `Executor`, via
`render_result`, labelled `OBSERVATION [tool]`. The model's own turns never
carry tool output. The `professional` profile reinforces this from the other
side, instructing the model to base every conclusion on real output, to say so
when it has not seen output yet, and never to claim it ran something it did not.
Together these mean a sentence like "I scanned the host and found port 22 open"
can only be true if the executor actually produced that line; the model has no
channel through which to forge it.

## Extension points

- **Add a tool.** Write a `ToolSpec` (with `ArgSpec`s, or a `builder` for
  awkward argument shapes) and register it in `tools/registry`. The loop, the
  gate, and the UI need no changes; discovery and validation apply automatically.
- **Swap the endpoint.** Point `base_url` at any OpenAI-compatible server, set
  `model`, and supply `api_key` if the server requires one. Nothing else in the
  stack knows or cares which backend is behind the URL.
- **Customize the voice.** Pass `--profile-file` a plain-text prompt or a TOML
  file with a `system` key. Profiles change tone and framing only; the engine
  and its capabilities stay the same.
