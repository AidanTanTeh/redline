"""Redline targets: adapters for the systems under test."""

from redline.targets.base import Target
from redline.targets.http_template import HttpTemplateConfig, HttpTemplateTarget
from redline.targets.limiter import RateLimiter
from redline.targets.openai_compat import ChatTurn, OpenAICompatConfig, OpenAICompatTarget
from redline.targets.python_callable import (
    AgentFunction,
    PythonCallableTarget,
    Reply,
    TrialContext,
)

__all__ = [
    "AgentFunction",
    "ChatTurn",
    "HttpTemplateConfig",
    "HttpTemplateTarget",
    "OpenAICompatConfig",
    "OpenAICompatTarget",
    "PythonCallableTarget",
    "RateLimiter",
    "Reply",
    "Target",
    "TrialContext",
]
