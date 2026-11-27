"""Attack "your own app" two ways: as a Python function, and as a web app over HTTP.

Both apps are VulnBot, our own demo target, on local Ollama. Costs $0 (no API key).
Run from the repo root:
    uv run python examples/app_demo.py                    # 4 attacks, both ways
    uv run python examples/app_demo.py ii-review-note-01 ea-manager-01

The web app is a tiny server this script starts on 127.0.0.1:8765 and stops when it finishes.
It is described to Redline by examples/http_target.yaml. Reports are saved in runs/; read
them with examples/summarize_run.py.

One attempt per attack: this shows the plumbing works, it does not measure a success rate.
"""

import asyncio
import json
import sys
import threading
from collections import Counter
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx

from redline.attacks import load_corpus
from redline.core import Attack, TargetResponse
from redline.demo import VulnBot, search_docs_tool, send_email_tool
from redline.demo.vulnbot import DEFAULT_MAX_TOOL_ROUNDS, leak_phrases
from redline.metrics import run_warnings
from redline.runner import (
    FunctionFactory,
    RunConfig,
    RunReport,
    TrialFactory,
    load_http_factory,
    run_attacks,
)
from redline.targets import OpenAICompatConfig, TrialContext

DEFAULT_IDS = ["di-override-01", "spl-repeat-above-01", "ii-review-note-01", "ea-manager-01"]
CONFIG_PATH = Path(__file__).with_name("http_target.yaml")
RUNS_DIR = Path("runs")
PORT = 8765  # the port in http_target.yaml
WEB_CANARY = "demo-canary-7f3k9q2m"  # "planted" in the web app; the same as in the YAML
MAX_REQUEST_BYTES = 100_000

BACKEND = OpenAICompatConfig(
    base_url="http://localhost:11434/v1",
    model="qwen3:8b",
    temperature=0.7,
    timeout_s=180,
    extra_body={"reasoning_effort": "none", "max_tokens": 300},
)


# --- your app as a Python function ------------------------------------------


async def my_agent(prompt: str, context: TrialContext) -> TargetResponse:
    """What a user would write: build the agent, plant the canary, hand it the documents."""
    bot = VulnBot(
        BACKEND.model_copy(update={"seed": context.seed}),
        canary=context.canary,
        tools=[search_docs_tool(context.documents), send_email_tool()],  # fake tools
    )
    return await bot.send(prompt)  # a full response, so tool calls can be judged


def function_factory() -> FunctionFactory:
    tools = ["search_docs", "send_email"]
    return FunctionFactory(
        my_agent,
        version="demo-1",
        name="demo-agent",
        model=BACKEND.model,
        tool_names=tools,
        leak_phrases=leak_phrases(tools),
        documents="context",
        retrieval_tool="search_docs",
        safe_to_retry=True,  # every tool here is fake
        max_calls_per_attempt=DEFAULT_MAX_TOOL_ROUNDS + 1,  # so a run can be capped
    )


# --- your app as a web server -----------------------------------------------


async def web_reply(message: str) -> dict[str, Any]:
    bot = VulnBot(BACKEND, canary=WEB_CANARY, tools=[send_email_tool()])
    response = await bot.send(message)
    return {
        "data": {
            "reply": response.text,
            "tool_calls": [
                {"name": call.name, "arguments": call.arguments} for call in response.tool_calls
            ],
            "finish_reason": response.finish_reason,
        }
    }


class ChatHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        if self.path != "/api/chat":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_REQUEST_BYTES:
                raise ValueError("bad length")
            message = json.loads(self.rfile.read(length))["message"]
            if not isinstance(message, str):
                raise ValueError("message must be text")
        except (ValueError, KeyError, TypeError):
            self.send_error(400)
            return
        try:
            payload = json.dumps(asyncio.run(web_reply(message))).encode("utf-8")
        except Exception as error:
            print(f"web app error: {type(error).__name__}", file=sys.stderr)
            self.send_error(500)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        pass  # keep the output readable


# --- running both -----------------------------------------------------------


async def attack(label: str, factory: TrialFactory, attacks: list[Attack], cases_hash: str) -> None:
    print(f"\n== {label}: {factory.name}")

    def show(result: Any) -> None:
        detail = result.error or ""
        if result.response is not None:
            hits = [f"{d.detector}:{d.field}" for d in result.detections if d.triggered]
            tools = [call.name for call in result.response.tool_calls]
            detail = f"{result.response.latency_ms / 1000:5.1f}s tools={tools} hits={hits}"
        print(f"  {result.attack_id:22} {result.outcome:13} {detail}", flush=True)

    report: RunReport = await run_attacks(
        attacks, factory, RunConfig(), cases_hash=cases_hash, on_result=show
    )
    counts = Counter(str(result.outcome) for result in report.results)
    print(f"  outcomes: {dict(counts)}")
    for warning in run_warnings(report.results, attacks, external_app=True):
        print(f"  WARNING: {warning}")
    print(f"  saved: {save(report)}")


def save(report: RunReport) -> Path:
    RUNS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = RUNS_DIR / f"{stamp}.report.json"
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return path


async def run_both(attacks: list[Attack], cases_hash: str) -> None:
    await attack("Python function", function_factory(), attacks, cases_hash)
    async with httpx.AsyncClient(trust_env=False) as client:  # no proxy for localhost
        web = load_http_factory(CONFIG_PATH, client=client)
        await attack("Web app over HTTP", web, attacks, cases_hash)


def main() -> None:
    corpus = load_corpus()
    by_id = {a.id: a for a in corpus.attacks}
    ids = sys.argv[1:] or DEFAULT_IDS
    unknown = [i for i in ids if i not in by_id]
    if unknown:
        sys.exit(f"unknown attack id(s) {unknown}. Choose from: {', '.join(sorted(by_id))}")

    # 127.0.0.1 only: nothing outside this machine can reach the vulnerable app.
    try:
        server = ThreadingHTTPServer(("127.0.0.1", PORT), ChatHandler)
    except OSError as error:
        # Carrying on would send the attacks to whatever else is listening there.
        sys.exit(f"can't use port {PORT} ({error.strerror}). Stop whatever is using it.")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        asyncio.run(run_both([by_id[i] for i in ids], corpus.content_hash))
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
