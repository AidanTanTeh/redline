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

A lesson from this step: a web search gave me a gitleaks version (v8.30.1) that doesn't
exist. `pre-commit autoupdate` checked the real repo and pinned v8.30.0 instead. From now
on I verify versions and APIs against the official source, not just a search result.

## 1.1: Core Pydantic models
I made the core models (`Attack`, `TargetResponse`, `DetectionResult`, `RunResult`) frozen
and strict: unknown keys are rejected and nothing can be changed after creation. Attack
results are evidence, so they shouldn't be editable, and a typo in a YAML key should fail
loudly instead of being ignored. I used tuples instead of lists for fields like `tags`,
because `frozen=True` only blocks reassigning a field, not `tags.append(...)`. I also
turned on the Pydantic mypy plugin so mypy checks model constructors the same way Pydantic
checks them at runtime. `triggered=True` always means "the attack succeeded", and
`succeeded` is computed from the detections instead of stored, so it can't disagree with
them. `RunResult` already carries the model name, timestamp, config hash and seed so runs
can be reproduced later.
