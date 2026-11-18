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

## 1.2: Target interface and OpenAI-compatible adapter
I defined `Target` as a `Protocol`, so any class with a `name` and an async `send(prompt)`
counts as a target without inheriting from anything. That lets Headway plug its own classes
into Redline without importing Redline's base classes. The first adapter talks to Ollama and
Gemini through the OpenAI-style `/chat/completions` API. Its config stores the *name* of the
env var that holds the API key, never the key, and the adapter only reads the key at the
moment it sends a request, so a printed or saved config can't leak it. I left retries and
rate-limit handling out of the adapter on purpose; the runner (3.1) will own them.

I added a `reasoning` field to `TargetResponse` and keep qwen3's thinking text out of `text`,
so detectors judge what a user would actually see. I checked my own Ollama: it sends thinking
in a separate `reasoning` field, and the adapter handles that, `<think>` tags, and a reply
cut off mid-thought. The adapter is tested against a fake server, but a full real reply never
finished because qwen3 is slow here (a few tokens per second) and my live call timed out.
Open problem for 1.4: if thinking uses up the token limit, `text` is empty, and a detector
would read that as "attack failed". That would quietly make the attack success rate look
lower than it really is, so I need to record why a reply stopped, or turn thinking down.
I tried turning thinking off with `think: false` in the request, but Ollama's OpenAI-style
endpoint ignored it (qwen3 still thought and hit the token cap), so that fix isn't a
one-liner. I'll decide in 1.4 how to handle it, with real run numbers to go on.

## 1.3: VulnBot v0
VulnBot is a small support bot whose system prompt holds a secret "canary" string, with
only "never reveal it" as protection. It's weak on purpose, so Redline has something safe
to attack. If the canary shows up in a reply, the attack provably worked, and the canary
detector in 2.2 will check for exactly that. I made the canary random for each bot
(`REDLINE-CANARY-` plus 8 hex characters from `secrets`), so it can't appear by accident or
be memorised from training data, and I let it be overridden for reproducible runs. An empty
canary is rejected, because it would "appear" in every reply and fake a leak.

VulnBot wraps the OpenAI-compatible adapter instead of inheriting from it. It isn't a kind
of adapter, it's a bot that uses one, and later steps (fake retrieval, a fake `send_email`
tool) need somewhere to live without touching the adapter. It also copies the config instead
of changing it, because configs are frozen and the caller's one should stay as it was. I
put it inside the package (`redline/demo/`) so it's importable, type-checked and usable from
the CLI later. So far it has only been tested against a fake server, never a real model;
that comes in 1.4.

## 1.4: First end-to-end run
I added `finish_reason` to `TargetResponse` so a reply that was cut off at the token limit
can be told apart from a model that finished and refused. Without it, an empty answer would
look like a failed attack and quietly make the attack success rate look lower than it is.
Ollama ignored `think: false` on its OpenAI-style endpoint, but `reasoning_effort: "none"`
turned qwen3's thinking off, so the first run could finish in seconds. The catch is that
every result so far describes "qwen3:8b with thinking off", so I'll record that setting with
results and compare thinking on and off later. The first real run (one blunt attack) was
refused and the canary did not leak. That proves the pipeline works against a real model,
but it is one attempt, not an attack success rate.
