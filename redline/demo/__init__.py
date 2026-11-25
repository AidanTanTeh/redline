"""Redline demo targets: deliberately weak systems that are safe to attack."""

from redline.demo.channels import Channel, make_vulnbot, was_delivered
from redline.demo.tools import FakeTool, search_docs_tool, send_email_tool
from redline.demo.vulnbot import CANARY_PREFIX, VulnBot, make_canary

__all__ = [
    "CANARY_PREFIX",
    "Channel",
    "FakeTool",
    "VulnBot",
    "make_canary",
    "make_vulnbot",
    "search_docs_tool",
    "send_email_tool",
    "was_delivered",
]
