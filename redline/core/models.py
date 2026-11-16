"""Core data models passed between Redline's pipeline stages."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RedlineModel(BaseModel):
    """Base for every Redline model: reject unknown keys, never mutate after creation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class AttackCategory(StrEnum):
    """What kind of weakness an attack probes. OWASP IDs get mapped on in step 2.1."""

    DIRECT_INJECTION = "direct_injection"
    INDIRECT_INJECTION = "indirect_injection"
    SYSTEM_PROMPT_LEAKAGE = "system_prompt_leakage"
    SENSITIVE_INFO_DISCLOSURE = "sensitive_info_disclosure"
    EXCESSIVE_AGENCY = "excessive_agency"


class Attack(RedlineModel):
    """One attack payload from the corpus."""

    # Lowercase slug, e.g. "direct-override-01". Used as a stable key in results.
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    name: str = Field(min_length=1)
    category: AttackCategory
    prompt: str = Field(min_length=1)
    description: str = ""
    # tuple, not list: `frozen=True` blocks `attack.tags = ...` but not `attack.tags.append(...)`.
    tags: tuple[str, ...] = ()


class ToolCall(RedlineModel):
    """A tool the target tried to call, e.g. send_email(to=..., body=...)."""

    name: str = Field(min_length=1)
    # JSON-shaped, so it has to be a dict. frozen=True can't stop `.arguments[k] = v`,
    # so treat it as read-only by convention.
    arguments: dict[str, Any] = Field(default_factory=dict)


class TargetResponse(RedlineModel):
    """Everything a Target returns for one prompt."""

    text: str
    model: str = Field(min_length=1)
    latency_ms: float = Field(ge=0)
    tool_calls: tuple[ToolCall, ...] = ()


class DetectionResult(RedlineModel):
    """One detector's verdict on one response."""

    detector: str = Field(min_length=1)
    # True means the ATTACK SUCCEEDED according to this detector (e.g. canary leaked).
    triggered: bool
    evidence: str | None = None
    confidence: float = Field(default=1.0, ge=0, le=1)


class RunResult(RedlineModel):
    """One attempt of one attack against one target, plus what's needed to reproduce it."""

    attack_id: str
    attempt: int = Field(ge=0)
    response: TargetResponse
    detections: tuple[DetectionResult, ...] = ()
    model: str = Field(min_length=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    config_hash: str = Field(min_length=1)
    seed: int | None = None

    @property
    def succeeded(self) -> bool:
        """The attack counts as successful if any detector triggered."""
        return any(d.triggered for d in self.detections)
