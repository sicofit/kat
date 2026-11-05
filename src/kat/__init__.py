"""KAT - a local-first LLM agent for security operations.

KAT turns a locally hosted language model into a working assistant: it plans,
calls real command-line tools through a validated execution layer, reads the
actual output, and decides what to do next. It talks to any OpenAI-compatible
endpoint (llama.cpp, Ollama, vLLM, ...), so the model stays on your machine.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
