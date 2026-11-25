"""Two ways to deliver an attack's documents to VulnBot, so results can be compared.

- "prompt": the documents are pasted into the message (how many basic RAG apps work).
- "tool":   the message is only the user's question; the documents arrive when the bot calls
            its `search_docs` tool (how agent-style apps work).

Both bots can send email. The only real difference is WHERE the document arrives (plus one
sentence in the system prompt telling the tool-channel bot about `search_docs`), so a
difference in results is down to the channel, not to one bot having more abilities.

Build a fresh bot per attack: its tools hold that attack's documents and nothing carries over.
"""

from typing import Literal

import httpx

from redline.attacks import render_prompt
from redline.core import Attack, TargetResponse
from redline.demo.tools import search_docs_tool, send_email_tool
from redline.demo.vulnbot import DEFAULT_MAX_TOOL_ROUNDS, VulnBot
from redline.targets import OpenAICompatConfig

Channel = Literal["prompt", "tool"]


def make_vulnbot(
    attack: Attack,
    backend: OpenAICompatConfig,
    channel: Channel,
    *,
    canary: str | None = None,
    client: httpx.AsyncClient | None = None,
    max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
) -> tuple[VulnBot, str]:
    """A fresh VulnBot for `attack`, and the message to send it: `await bot.send(message)`.

    An attack with no documents (a direct attack) is sent as-is in either channel, so only
    attacks with documents make a meaningful channel comparison.
    """
    if channel == "prompt":
        tools = [send_email_tool()]
        message = render_prompt(attack)
    else:
        tools = [search_docs_tool(attack.documents), send_email_tool()]
        message = attack.prompt
    bot = VulnBot(
        backend, canary=canary, client=client, tools=tools, max_tool_rounds=max_tool_rounds
    )
    return bot, message


def was_delivered(attack: Attack, channel: Channel, response: TargetResponse) -> bool:
    """Did the attack's documents actually reach the model?

    In the tool channel the bot has to CHOOSE to call `search_docs`. If it answers from its own
    knowledge instead, the poisoned document is never read, so "no attack effect" means "the
    attack was never delivered", NOT "the target defended". Such runs must not be counted as
    defended. Only checks that retrieval happened, not that the model read or understood it.
    """
    if not attack.documents or channel == "prompt":
        return True
    retrieval_tool = search_docs_tool(()).name
    return any(call.name == retrieval_tool for call in response.tool_calls)
