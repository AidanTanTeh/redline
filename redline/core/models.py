"""Core data models passed between Redline's pipeline stages."""

from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class RedlineModel(BaseModel):
    """Base for every Redline model: reject unknown keys, never mutate after creation."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class AttackCategory(StrEnum):
    """What kind of weakness an attack probes. See OWASP_LLM_2026 for the OWASP mapping."""

    DIRECT_INJECTION = "direct_injection"
    INDIRECT_INJECTION = "indirect_injection"
    SYSTEM_PROMPT_LEAKAGE = "system_prompt_leakage"
    SENSITIVE_INFO_DISCLOSURE = "sensitive_info_disclosure"
    EXCESSIVE_AGENCY = "excessive_agency"


# OWASP Top 10 for LLM Applications, 2026 edition (checked on genai.owasp.org, Nov 2026).
# In 2026, system prompt extraction falls under LLM08 "Hidden Context Exposure"; the 2025
# edition had a separate LLM07 "System Prompt Leakage". MappingProxyType makes the dict
# read-only, the same idea as tuples instead of lists elsewhere in this file.
OWASP_LLM_2026: Mapping[AttackCategory, str] = MappingProxyType(
    {
        AttackCategory.DIRECT_INJECTION: "LLM01:2026",  # Prompt Injection
        AttackCategory.INDIRECT_INJECTION: "LLM01:2026",  # Prompt Injection
        AttackCategory.SYSTEM_PROMPT_LEAKAGE: "LLM08:2026",  # Hidden Context Exposure
        AttackCategory.SENSITIVE_INFO_DISCLOSURE: "LLM02:2026",  # Sensitive Information Disclosure
        AttackCategory.EXCESSIVE_AGENCY: "LLM03:2026",  # Excessive Agency
    }
)


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

    @property
    def owasp_id(self) -> str:
        """OWASP LLM Top 10 (2026) ID for this attack's category, e.g. "LLM01:2026"."""
        return OWASP_LLM_2026[self.category]


class ToolCall(RedlineModel):
    """A tool the target tried to call, e.g. send_email(to=..., body=...)."""

    name: str = Field(min_length=1)
    # JSON-shaped, so it has to be a dict. frozen=True can't stop `.arguments[k] = v`,
    # so treat it as read-only by convention.
    arguments: dict[str, Any] = Field(default_factory=dict)


class TargetResponse(RedlineModel):
    """Everything a Target returns for one prompt."""

    # The final answer only. Detectors judge this, because it's what a user would see.
    text: str
    # Model "thinking" (e.g. qwen3's <think> block), kept for debugging but never detected on.
    reasoning: str | None = None
    model: str = Field(min_length=1)
    latency_ms: float = Field(ge=0)
    # Why generation stopped, as the server reports it: "stop" = finished normally,
    # "length" = hit the token cap (the reply may be cut off, so an empty `text` means
    # "unknown", not "the attack failed"). None if the server didn't say.
    finish_reason: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


# The parts of a TargetResponse a detector can scan. Like a TS string-literal union type.
ResponseField = Literal["text", "reasoning"]


class DetectionResult(RedlineModel):
    """One detector's verdict on one response."""

    detector: str = Field(min_length=1)
    # True means the ATTACK SUCCEEDED according to this detector (e.g. canary leaked).
    triggered: bool
    # Which part of the response this verdict is about. Reports use it to keep leaks in the
    # visible answer ("text") separate from leaks in model thinking ("reasoning").
    field: ResponseField | None = None
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
