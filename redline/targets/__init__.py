"""Redline targets: adapters for the systems under test."""

from redline.targets.base import Target
from redline.targets.openai_compat import ChatTurn, OpenAICompatConfig, OpenAICompatTarget

__all__ = ["ChatTurn", "OpenAICompatConfig", "OpenAICompatTarget", "Target"]
