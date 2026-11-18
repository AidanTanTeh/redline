"""Redline demo targets: deliberately weak systems that are safe to attack."""

from redline.demo.vulnbot import CANARY_PREFIX, VulnBot, make_canary

__all__ = ["CANARY_PREFIX", "VulnBot", "make_canary"]
