# Redline

Prompt-injection and data-leak red-teaming for LLM apps and agents. Redline runs a
versioned library of attacks against a target, scores which ones succeed, and can fail a
CI build when the attack success rate rises.

> **Status:** early development. Nothing here is ready to use yet.

## Responsible use

Redline is for testing systems **you own or are explicitly authorised to test**. Do not
point it at third-party services, other people's apps, or any API whose terms forbid
security testing. You are responsible for how you use it.

## Quickstart

_Coming soon._

## How it works

_Coming soon._

## Results

_Coming soon. Only real, measured numbers will go here._

## Development

```bash
uv sync                       # install
uv run pytest                 # tests
uv run ruff check . && uv run ruff format .
uv run mypy                   # types
```

## License

MIT. See [LICENSE](LICENSE).
