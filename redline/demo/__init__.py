"""Redline demo targets: deliberately weak systems that are safe to attack."""

from redline.demo.channels import Channel, make_vulnbot, was_delivered
from redline.demo.factory import VulnBotFactory
from redline.demo.tools import FakeTool, search_docs_tool, send_email_tool
from redline.demo.vulnbot import VulnBot, make_canary
from redline.detectors import CANARY_PREFIX, seeded_canary

__all__ = [
    "CANARY_PREFIX",
    "Channel",
    "FakeTool",
    "VulnBot",
    "VulnBotFactory",
    "make_canary",
    "make_vulnbot",
    "search_docs_tool",
    "seeded_canary",
    "send_email_tool",
    "was_delivered",
]
