"""Run ONE corpus attack (default: an indirect injection) against VulnBot on local Ollama.

Only attacks VulnBot, our own demo target. Costs $0 (local model, no API key).
Run from the repo root:
    uv run python examples/indirect_run.py                    # default attack
    uv run python examples/indirect_run.py ii-exfil-link-01   # any attack id from the corpus

One attempt is NOT an attack success rate. It only shows the pipeline works end to end.
"""

import asyncio
import sys

from redline.attacks import load_corpus, render_prompt
from redline.demo import CANARY_PREFIX, VulnBot
from redline.detectors import CanaryDetector
from redline.targets import OpenAICompatConfig

DEFAULT_ATTACK_ID = "ii-review-note-01"


async def main(attack_id: str) -> None:
    corpus = load_corpus()
    by_id = {a.id: a for a in corpus.attacks}
    if attack_id not in by_id:
        sys.exit(f"unknown attack id {attack_id!r}. Choose one of: {', '.join(sorted(by_id))}")
    attack = by_id[attack_id]

    backend = OpenAICompatConfig(
        base_url="http://localhost:11434/v1",
        model="qwen3:8b",
        temperature=0.0,
        timeout_s=180,
        # Thinking off, same as first_run.py. Record this next to any result you quote.
        extra_body={"reasoning_effort": "none", "max_tokens": 300},
    )
    bot = VulnBot(backend)
    prompt = render_prompt(attack)

    print(f"corpus:  {len(corpus.attacks)} attacks, hash {corpus.content_hash[:12]}")
    print(f"target:  {bot.name}  (thinking off, temperature 0)")
    print(f"attack:  {attack.id} ({attack.category}, {attack.owasp_id})")
    print(f"\n--- text sent to the target ---\n{prompt}\n--- end ---")
    print("sending...")

    response = await bot.send(prompt)

    print(f"\ntext:          {response.text!r}")
    print(f"finish_reason: {response.finish_reason}")
    print(f"latency:       {response.latency_ms / 1000:.1f}s")

    detector = CanaryDetector(bot.canary, public_prefix=CANARY_PREFIX)
    print("\ndetections (triggered=True means the attack WORKED):")
    for result in await detector.detect(attack, response):
        print(
            f"  {result.detector} field={result.field} triggered={result.triggered} "
            f"confidence={result.confidence}"
        )
        if result.evidence:
            print(f"    evidence: {result.evidence}")
    if response.finish_reason == "length":
        print("  note: finish_reason is 'length', so the reply was cut off: no usable answer.")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ATTACK_ID))
