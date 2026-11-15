# DEVLOG

Decisions and the reasons behind them, newest last.

## 0.1: Free, local-first setup
I'm using Ollama (`qwen3:8b`) as the main test target and the Gemini free tier as a backup.
I chose them because they cost $0, and both speak the OpenAI-style API, so one adapter
works for both. I'm still keeping a general `Target` interface so I can add other kinds of
targets later (HTTP endpoints, Python functions) without changing the rest of Redline.
With no money to spend, the limit to watch is the number of requests, not dollars.

## 0.2: Project setup
I put the code in a top-level `redline/` folder and used uv to manage the project. Ruff,
mypy (strict) and pytest check the code. I added `.gitignore` rules early so secrets like
`.env` never get committed, because deleting a leaked key later doesn't remove it from git
history. My personal planning files (`CLAUDE.md`, `PROGRESS.md`, `docs/PLAN.md`) stay
out of the public repo because they contain personal details.

I also set up pre-commit hooks: Ruff, mypy and gitleaks run automatically on every commit
and stop it if something fails. Gitleaks scans for secrets, and I tested it against a fake
token in a throwaway repo to confirm it blocks the commit. The hooks only run on my machine,
so CI will repeat the same checks later.
