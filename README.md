# KAT

[![CI](https://github.com/sicofit/kat/actions/workflows/ci.yml/badge.svg)](https://github.com/sicofit/kat/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A local-first agent for security operations: it runs real command-line tools and reasons over their real output.

KAT (short for Kali Assistant Tool) turns a locally hosted language model into a working operator. It plans a
task, runs real command-line tools through a validated execution layer, reads
the actual output, and decides what to do next. The model runs on hardware you
control: KAT speaks to any OpenAI-compatible endpoint (llama.cpp, Ollama,
vLLM), so prompts, targets, and findings stay on the host.

## Why

- **Local and private.** The model runs behind an endpoint you own. There is no
  hosted service in the loop and nothing to opt out of.
- **Real execution, not a chat transcript.** Tools run as actual subprocesses.
  KAT reads their real stdout, stderr, and exit codes, and feeds that back into
  the next decision.
- **Honest by construction.** Observations come only from the executor. The
  model is never the source of a tool's output, so it cannot quietly invent a
  scan result or a captured credential.
- **Model-agnostic.** Point KAT at a different server by changing one URL.
  There is no function-calling dialect to match and no vendor to depend on.
- **Safe by default.** Active operations pause for explicit approval and show
  the exact command first. The gate is enforced by KAT, not left to the model.

## Install

```
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/macOS
pip install -e .
```

Python 3.11 or newer.

## Run a model

KAT does not ship a model; it talks to an OpenAI-compatible server you run. Any
of these works -- pick one, then KAT is ready to answer.

**Ollama** (simplest to start):

```
ollama pull llama3.1                          # or any model you prefer
# Ollama serves an OpenAI-compatible API on http://localhost:11434
export KAT_BASE_URL=http://localhost:11434    # Windows PowerShell: $env:KAT_BASE_URL="http://localhost:11434"
export KAT_MODEL=llama3.1                      # Windows PowerShell: $env:KAT_MODEL="llama3.1"
```

**llama.cpp** (matches KAT's defaults, no config needed):

```
llama-server -m your-model.gguf --port 9932
```

Then confirm KAT can reach it -- `kat status` prints `STATUS: ready` once the
server answers:

```
kat status
```

## Quickstart

```
# Check the endpoint and show the resolved configuration.
kat status

# A question that needs no tools -- answered directly.
kat "Explain the difference between TCP and UDP."

# A task that enumerates a host -- active steps pause for approval.
kat "Enumerate the services on 10.10.10.10."
```

Before an active operation runs, KAT prints the exact command and waits. Approve
it to continue, or decline and let the agent adapt. For unattended runs, pass
`--yes` to approve automatically, or set the policy with
`--approval none|active|destructive`.

## Commands

| Command | What it does |
| --- | --- |
| `kat "task"` | Run a single task to completion and exit. |
| `kat chat` | Hold an interactive session across multiple turns. |
| `kat explain "snippet"` | Explain a command, flag, or output without running anything. |
| `kat continue` | Resume the most recent session and keep going. |
| `kat sessions` | List saved sessions under `session_dir`. |
| `kat tools` | List the tools discovered on this host. |
| `kat status` | Show the resolved configuration and whether the endpoint is reachable. |

Global flags apply to any command and override configured values:
`--profile`, `--profile-file`, `--approval`, `--yes`, `--base-url`, `--model`,
`--no-stream`, `--max-steps`, and `--config` to point at a specific config file.

## Explain mode

`kat explain` answers questions and breaks commands, flags, and output down into
plain language without running anything. It is the mode to reach for when you
have a line you do not recognise and want to understand it before you act.

```
kat explain "what does nmap -sV -p- do?"
kat explain "I ran ***ping -c 4 192.168.1.100*** -- what should I expect?"
```

Text wrapped in triple asterisks is treated as a snippet to explain, never as a
command to run. A short shell alias makes it quick to reach:

```
# Linux / Kali (~/.bashrc)
alias ai='kat explain'

# Windows PowerShell ($PROFILE)
function ai { kat explain @args }
```

Then `ai "what does nmap -sV do?"` is all it takes. Quote the question so the
shell leaves characters like `?` and `*` alone.

## How it works

```
Config  ->  LLMClient  ->  Agent loop  ->  Session
```

The agent loop is the core:

```
   task
    |
    v
  think ---> action (JSON) ---> validated execution
    ^                                   |
    |                                   v
    +------------- observation <--- real tool output
    |
    v
  prose answer (no action block) ---> done
```

On each step the model either emits a single structured JSON action block or
answers in prose. An action names a tool and its arguments; KAT validates it,
runs it as a subprocess, and feeds the real output back as an observation. The
loop repeats until the model replies with no action, at which point that reply
is the answer (or until `max_steps` is reached).

The action protocol is a parsed JSON block rather than a native function call.
That choice keeps KAT portable: it works with any model that can follow a format
instruction, regardless of whether the server supports tool-calling APIs.
Observations are always real tool output, never anything the model supplied.

## Configuration

| Key | Default | Meaning |
| --- | --- | --- |
| `base_url` | `http://127.0.0.1:9932` | OpenAI-compatible endpoint of the local model server |
| `model` | `local` | Model name sent to the endpoint |
| `profile` | `professional` | Built-in system-prompt profile |
| `approval` | `active` | Approval gate: `none`, `active`, or `destructive` |
| `max_steps` | `30` | Maximum think/act iterations before the loop stops |
| `temperature` | `0.4` | Sampling temperature sent to the model |
| `tool_timeout` | `120` | Seconds a single tool may run before it is killed |
| `max_tool_output` | `6000` | Characters of a single tool's output retained; the rest is clipped |
| `context_chars` | `16000` | Character budget for the prompt window sent to the model |
| `session_dir` | `~/.kat/sessions` | Directory for saved transcripts |
| `stream` | `true` | Stream tokens as they are produced |

Settings are resolved from `kat.toml` (or `~/.kat/config.toml`), then `KAT_*`
environment variables (for example `KAT_BASE_URL`, `KAT_MODEL`), then
command-line flags, each layer overriding the one before. See
[`config.example.toml`](config.example.toml) for a complete, commented file.

## Tools

Tools are discovered at runtime with `shutil.which`. Only tools actually
installed on the host are offered to the model, so the catalog always reflects
the machine KAT is running on rather than a fixed wishlist. List what was found:

```
kat tools
```

Each tool declares its arguments; values are validated and assembled into an
argument vector that is executed without a shell. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the execution contract.

## Development

```
pip install -e ".[dev]"
pytest -q
python evals/run_eval.py
```

The eval harness drives the full agent loop deterministically with no network:
a stand-in model and recorded tool output, so decision behavior can be checked
offline and in CI.

## Authorized use

KAT executes real commands against real targets. Use it only on systems you own
or are explicitly authorized to test: your own machines, lab networks, CTF
challenges, and penetration-test engagements within an agreed scope. Keeping it
pointed at authorized targets is your responsibility.

## License

MIT. See [LICENSE](LICENSE).
