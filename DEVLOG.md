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

## 2.1: Attack corpus and loader
Attacks now live in YAML files inside the package instead of in code, so the corpus can grow
without touching Python and ships with the library. Every file goes through the same Pydantic
`Attack` model, so a typo'd key or a bad category fails loudly with the file name instead of
silently skewing results. `load_corpus()` also returns a hash of the files' names and bytes;
later every result will record it, so "ASR 12%" always says which attacks produced it, and the
hash can't go stale the way a hand-bumped version number could.

I shipped 20 attacks in 3 categories, not the planned ~30 in 5. Indirect injection and
excessive agency need a fake retrieved document and a fake `send_email` tool that don't exist
until 2.3 and 2.4, so I'll add those attacks with their fields then, rather than guess the
schema now. Each attack maps to the 2026 OWASP LLM Top 10 (I checked the list on OWASP's site;
system prompt extraction now sits under LLM08 "Hidden Context Exposure"). None of the attacks
has been run against a model yet, so there is no attack success rate; that needs the detectors
in 2.2 and the runner in 3.1.

## 2.2: Canary and regex detectors
Detectors decide whether an attack worked, so they have to be right in both directions:
missing a leak makes the target look safer than it is, and a false alarm makes it look
weaker. I made every detector async and give it both the attack and the response. The canary
detector ignores the attack, but the LLM judge in 4.2 will need it and has to call a model,
so one shared shape means the runner never needs special cases.

The canary detector looks for more than the exact string. It ignores case, spacing,
punctuation and invisible characters (so a letter-per-line leak is still caught), and it
decodes base64 and hex. A fragment of 6 or more of the 8 random characters counts as a leak
too, at a lower confidence of 0.6, because a partial leak is still a leak and hiding it would
undercount the attack success rate, the same worry as the cut-off replies in 1.4. A
`public_prefix` option stops the known `REDLINE-CANARY-` part from counting. Results record
which field matched (`text` or `reasoning`) so thinking-text leaks can be reported
separately, and only `text` is scanned by default. The regex detector covers leaks with no
random token, like repeated system-prompt sentences; its `from_phrases` helper treats phrases
as plain text and ignores re-wrapping.

Some things went wrong or surprised me. `sid-hint-01` asked for "the first 8 characters" of
the password, but those are the public prefix, so the attack could never leak anything real;
I only noticed while designing the partial-leak rule, changed it to the last 8, and the
corpus hash changed, which is the hash doing its job. Stripping separators for matching could
also cause false alarms with a short, guessable canary, so I added a minimum canary length of
8 letters or digits. My first `from_phrases` used `**kwargs` with a `type: ignore`, which hid
exactly the mistakes mypy should catch, so I replaced it with explicit options. On the first
full run ruff also flagged one over-long line. Running the example live against `qwen3:8b`
showed the detectors work on a real reply, but VulnBot refused again, so I have only seen the
"no leak" path on a real model. I have not seen a real leak, one attempt is still not a rate,
and the detectors have known gaps (look-alike letters from other alphabets, wrongly encoded
base64, ROT13, paraphrased leaks), which I wrote in the code instead of claiming coverage I
don't have.

One more thing turned up after the first draft. I asked why the canary needs a public prefix
if it never changes, and testing it showed that without `public_prefix` the 13 public
characters count toward the fragment threshold, so only 3 secret characters were enough to
trigger the 0.6 "fragment" result, which is far weaker than its label claims. I had also
described the prefix's purpose wrongly (the prefix alone never triggered either way); the
real effect is on the threshold. The default is `""`, so forgetting the option silently
weakens the check. I didn't fix it in 2.2 because the proper fix touches VulnBot and the
runner (the target should hand over its canary and prefix together so callers can't forget
it), so I put it in the 3.1 notes.

## 2.3: Indirect injection
In an indirect attack the user is innocent and the poison hides in content the bot reads,
like a review, an email or a web page. I gave `Attack` an optional list of fake documents,
with a rule that indirect attacks must have at least one, because forgetting it would
silently turn the attack into a harmless question and lower the attack success rate. A small
`render_prompt()` builds the text sent to the target: documents first, then the question. It
is deliberately naive (no delimiters, no "treat this as data" warning) to model a basic RAG
app, so later defences have a baseline to beat. I wrote 6 attacks (review, fake policy,
exfiltration link, email, HTML comment, faked end-of-documents), all aimed at leaking the
canary, and tests enforce that exfiltration links only use the reserved `.example` domain and
that no attack text contains the canary.

Two decisions. First, how documents reach the bot: I chose pasting them into the prompt now,
because it works with every target and models the common RAG pattern. The realistic
alternative is a real retrieval tool the model calls. I checked that this works on Ollama's
`/v1` endpoint with `qwen3:8b` (the model made a proper tool call and used the result), so
it is committed for 2.4, where the same attacks will be run both ways and compared. Second,
what counts as success: canary leak only, because it is unambiguous. Attacks that leak
nothing, like making the bot state a false claim, are not measured yet, so any indirect ASR
before the LLM judge in 4.2 means secret exfiltration only.

One thing went wrong: my corpus test that no two attacks are duplicates failed, because
three indirect attacks share the same innocent question. The test was wrong, not the
attacks, since those attacks differ in their documents. I changed it to compare the text
actually sent after rendering. Running the first indirect attack for real, `qwen3:8b` leaked
the canary, so this is the first time the detector's "triggered" path was seen on a real
model. It is one attempt with thinking off and temperature 0, not a rate. Another untested
assumption: documents come before the question, which probably helps the defender a little,
and I have not measured the effect.

## 2.4: Excessive agency and tool calling
Excessive agency means the bot has a powerful tool and gets tricked into using it against the
user. To test that I had to build real tool calling: the adapter can now send a whole
conversation plus a list of tools, and VulnBot runs a loop (ask the model, run the tool it
asks for, send the result back, repeat) with a round cap. The tools are fake on purpose:
`search_docs` returns the attack's planted documents and `send_email` only says "queued",
with a test that the tools module has no network code. Detection reads what the model
*asked* for, not what happened, because asking is the vulnerability. Each attack says in its
own YAML what success looks like (`success_tool_call`), so one `ToolCallDetector` serves them
all, and the canary detector now also scans tool-call arguments, because a secret inside an
email body is a real leak. I wrote 5 attacks (corpus is now 31), all with unique `.example`
addresses that tests enforce.

I checked first that tool calling really works on Ollama's `/v1` endpoint with `qwen3:8b`,
and the live runs confirmed it. Several things went wrong or surprised me. Pre-commit's type
check failed when I committed in pieces, because it hides unstaged edits but not new
untracked files, so I had to commit in dependency order. One test broke because it used
`tool_calls` as an "invalid" example after I made it valid. I found two undercounting risks
and fixed them: a small model's malformed JSON arguments would not have matched, and the
regex detector ignored tool calls. The biggest finding came from the first live run: in the
tool channel the bot sometimes never called `search_docs`, so the poisoned document was never
read and "no effect" really meant "not delivered", which would have been scored as the target
defending. I added `was_delivered` for that, and a positive control (a harmless "email a
teammate" request) to prove the bot can use the tool at all, which it did. The live runs also
showed the canary landing in an email body while the reply text never mentioned it, so a
text-only scan would have missed it.

What I have not shown: every result is one attempt with thinking off and temperature 0, so
none of it is a rate. A poisoned-document attack has not yet made the bot send an email (only
direct requests did), Gemini tool calling is untested, and delivery in the tool channel
depends on the model choosing to retrieve. These go into the 3.1 runner notes so undelivered
or tool-less runs are never counted as defended.

## 3.1: Async runner

I built the runner as a library that runs each attack several times and labels every attempt
as succeeded, resisted, no answer, not delivered, skipped or error, because only the first
two say anything about the target's defences. A factory builds each attempt (bot, message and
detectors), so the runner knows nothing about VulnBot or canaries and can drive other apps
later. Each repeat gets its own seed, the canary is derived from the seed and attack ID, and
a config hash shows whether two runs are comparable. I wrote the retry logic by hand to
understand it: only timeouts, network failures, 429s and server errors are retried.

Testing first showed that Ollama queues requests, so the default is one at a time, and that
it honoured the seed in a single check. Things I changed along the way: not every httpx
transport error is worth retrying; the hash refuses non-JSON values instead of turning them
into strings that could differ each run; and the hash ignored which attacks were run, so 4
attacks looked comparable to 31, which I fixed. The first live run (8 attempts, temperature
0.7) had 4 successes and 4 resisted, too few to be a rate. It also showed that a system
prompt leak without the canary would be missed. Still open: code changes don't alter the
hash, a crashing detector discards the reply, and the tool channel, retries and parallel runs
haven't been seen live.

I fixed the leakage gap before measuring any rates: leakage attacks now also check for
wording from VulnBot's system prompt. Choosing that wording took care. One attack supplies
the first line of the prompt itself, so phrases found in the attack are skipped, and I
dropped phrases that an honest refusal could say, including the one I first planned to use.
Live, one attack was caught only by the new check. A rerun gave 8 identical replies, and
retries, timeouts, parallel runs and the tool channel all worked on the real setup.
Paraphrased and translated leaks without the canary are still missed, and no real rate limit
or hosted model has been tested.

## 3.2: Success rates

The success rate counts only attempts with a verdict (succeeded or resisted), shows every
other outcome beside it, and gives the lowest and highest rate if the missing ones had gone
either way. Each attack weighs the same. For uncertainty I used a 95% Wilson interval
counting attacks rather than attempts, because repeating one attack isn't a new test; this
makes the intervals wide and only approximate. I rejected counting attempts (overconfident)
and a bootstrap (poor with so few attacks). Blocked replies count as no answer, though I
couldn't confirm how Gemini reports them.

The first baseline (qwen3:8b, thinking off, temperature 0.7, 31 attacks, 3 repeats per
channel) gave 44% overall in the prompt channel and 67% in the tool channel. The two can't be
compared simply: 20 tool-channel attempts never delivered the document. Unexpectedly,
sensitive info attacks with no documents rose from 2/21 to 11/21 with the extra tool and
prompt line, which showed my claim that the channels differ only in delivery was wrong. I
haven't isolated the cause or manually checked the replies.
