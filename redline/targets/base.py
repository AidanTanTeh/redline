"""The Target interface: anything Redline can attack."""

from typing import Protocol

from redline.core import TargetResponse


class Target(Protocol):
    """Anything that can take a prompt and return a response.

    Structural typing: a class counts as a Target if it has these members.
    It doesn't need to inherit from Target or declare that it implements it.
    """

    @property
    def name(self) -> str:
        """Human-readable label used in reports, e.g. "ollama/qwen3:8b"."""
        ...

    async def send(self, prompt: str) -> TargetResponse:
        """Send one prompt to the target and return what it said."""
        ...
