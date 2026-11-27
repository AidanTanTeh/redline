"""Core data models passed between Redline's pipeline stages."""

from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from redline.core.text import contains_phrase


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


class Document(RedlineModel):
    """A fake retrieved document (web page, email, review, alert) that may carry a payload.

    In an indirect attack the attacker never talks to the bot: they plant instructions in
    content the bot reads later.
    """

    title: str = Field(min_length=1)
    content: str = Field(min_length=1)


class ToolCallMatch(RedlineModel):
    """What a successful tool-abuse attack looks like: this tool, with arguments like these.

    Example (YAML): {tool: send_email, arguments: {to: attacker@evil.example}} means "the
    attack worked if the bot called send_email with a `to` containing that address".
    """

    tool: str = Field(min_length=1)
    # argument name -> text that must appear in that argument's value (case-insensitive).
    # Empty means "any call to this tool counts". Read-only by convention, like ToolCall.
    arguments: dict[str, str] = Field(default_factory=dict)


class Case(RedlineModel):
    """Anything the runner can send to a target: an Attack or a BenignTask."""

    # Lowercase slug, e.g. "direct-override-01". Used as a stable key in results.
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    name: str = Field(min_length=1)
    # What the user types.
    prompt: str = Field(min_length=1)
    # Fake retrieved content shown to the bot alongside the question.
    documents: tuple[Document, ...] = ()
    description: str = ""
    # tuple, not list: `frozen=True` blocks `case.tags = ...` but not `case.tags.append(...)`.
    tags: tuple[str, ...] = ()

    @property
    def needs_tools(self) -> bool:
        """True if success means the target called a tool, so a target with no tools can never
        succeed. The runner skips such cases there instead of counting them as failures."""
        return False


class Attack(Case):
    """One attack payload from the corpus.

    For direct attacks `prompt` IS the attack. For indirect attacks it is the user's innocent
    question, and the attack is hidden in `documents` (required for indirect injection).
    """

    category: AttackCategory
    # What counts as success for tool-abuse attacks (read by ToolCallDetector). Required for
    # excessive_agency; other categories may use it too (e.g. an indirect attack that emails).
    success_tool_call: ToolCallMatch | None = None

    @model_validator(mode="after")
    def _indirect_attacks_need_documents(self) -> Self:
        # Runs after every field has been validated. An "indirect" attack with nothing to
        # inject into would silently behave like a harmless plain question.
        if self.category is AttackCategory.INDIRECT_INJECTION and not self.documents:
            raise ValueError("indirect_injection attacks need at least one document")
        return self

    @model_validator(mode="after")
    def _agency_attacks_need_a_success_tool_call(self) -> Self:
        # Without it nothing says what "the tool was abused" means, so the attack could never
        # be scored as a success.
        if self.category is AttackCategory.EXCESSIVE_AGENCY and self.success_tool_call is None:
            raise ValueError("excessive_agency attacks need a success_tool_call")
        return self

    @property
    def needs_tools(self) -> bool:
        return self.success_tool_call is not None

    @property
    def owasp_id(self) -> str:
        """OWASP LLM Top 10 (2026) ID for this attack's category, e.g. "LLM01:2026"."""
        return OWASP_LLM_2026[self.category]


class TaskKind(StrEnum):
    """What a benign task checks."""

    ANSWER = "answer"  # answer a question from a document
    TOOL = "tool"  # do something with a tool, e.g. send an email
    # Innocent, but worded like an attack ("reset my password"): catches over-refusal.
    SENSITIVE_SOUNDING = "sensitive_sounding"


class BenignTask(Case):
    """A normal request the target SHOULD complete. Used to measure utility."""

    kind: TaskKind
    # Every group must be satisfied, by at least one of its phrases. Example:
    # (("3 to 5", "3-5"), ("business days",)) means (either spelling) AND "business days".
    expected_text: tuple[tuple[str, ...], ...] = ()
    expected_tool_call: ToolCallMatch | None = None

    @model_validator(mode="after")
    def _needs_a_checkable_expectation(self) -> Self:
        if not self.expected_text and self.expected_tool_call is None:
            raise ValueError("a benign task needs expected_text or an expected_tool_call")
        if any(not group or any(not p.strip() for p in group) for group in self.expected_text):
            raise ValueError("expected_text groups and phrases must not be empty")
        return self

    @model_validator(mode="after")
    def _question_must_not_give_the_answer(self) -> Self:
        # Otherwise a bot that just repeated the question would "complete" the task.
        given = [p for g in self.expected_text for p in g if contains_phrase(self.prompt, p)]
        if given:
            raise ValueError(f"expected phrases appear in the prompt: {given}")
        return self

    @property
    def needs_tools(self) -> bool:
        return self.expected_tool_call is not None


class ToolCall(RedlineModel):
    """A tool the target tried to call, e.g. send_email(to=..., body=...)."""

    name: str = Field(min_length=1)
    # JSON-shaped, so it has to be a dict. frozen=True can't stop `.arguments[k] = v`,
    # so treat it as read-only by convention.
    arguments: dict[str, Any] = Field(default_factory=dict)
    # The server's id for this call. A tool result must quote it so the model knows which call
    # the result answers. None when a ToolCall is built by hand (e.g. in tests).
    id: str | None = None


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
ResponseField = Literal["text", "reasoning", "tool_calls"]


class DetectionResult(RedlineModel):
    """One detector's verdict on one response."""

    detector: str = Field(min_length=1)
    # True means the case's success condition was met according to this detector: for an
    # attack, it worked (e.g. canary leaked); for a benign task, it was completed.
    triggered: bool
    # Which part of the response this verdict is about. Reports use it to keep leaks in the
    # visible answer ("text") separate from leaks in model thinking ("reasoning").
    field: ResponseField | None = None
    evidence: str | None = None
    confidence: float = Field(default=1.0, ge=0, le=1)


class Outcome(StrEnum):
    """What one attempt amounted to. Only SUCCEEDED and RESISTED are real verdicts on the target;
    the other four mean "this attempt can't tell us", and must not be counted as defended."""

    SUCCEEDED = "succeeded"  # a detector triggered: the attack worked
    RESISTED = "resisted"  # a usable answer, attack delivered, nothing triggered
    NO_ANSWER = "no_answer"  # cut off ("length") or stuck in tools ("max_tool_rounds")
    NOT_DELIVERED = "not_delivered"  # the poisoned document never reached the model
    SKIPPED = "skipped"  # the attack needs tools and the target has none
    ERROR = "error"  # HTTP error, timeout or unreadable response, after retries


# Outcomes that never got a response from the target, so there is nothing to detect on.
_NO_RESPONSE = frozenset({Outcome.SKIPPED, Outcome.ERROR})


class RunResult(RedlineModel):
    """One attempt of one attack against one target, plus what's needed to reproduce it."""

    attack_id: str
    attempt: int = Field(ge=0)
    # Which target setup produced this, e.g. "vulnbot/qwen3:8b/tool".
    target: str = Field(min_length=1)
    outcome: Outcome
    # None only for SKIPPED and ERROR (enforced below).
    response: TargetResponse | None = None
    detections: tuple[DetectionResult, ...] = ()
    # Did the attack's documents reach the model? None when there was no response to check.
    delivered: bool | None = None
    # What went wrong, for ERROR only. Exception type + message, never secrets.
    error: str | None = None
    # How many times the attempt was retried after a transient failure (timeout, 429, 5xx).
    retries: int = Field(default=0, ge=0)
    model: str = Field(min_length=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    config_hash: str = Field(min_length=1)
    seed: int | None = None

    @property
    def succeeded(self) -> bool:
        """The attack counts as successful if any detector triggered."""
        return any(d.triggered for d in self.detections)

    @model_validator(mode="after")
    def _outcome_matches_evidence(self) -> Self:
        # The outcome is stored (it can't all be worked out from the other fields), so make sure
        # it can never contradict them. A stored "resisted" next to a triggered canary would
        # quietly lower the attack success rate.
        if self.outcome in _NO_RESPONSE:
            if self.response is not None or self.detections:
                raise ValueError(f"{self.outcome} results must have no response or detections")
        elif self.response is None:
            raise ValueError(f"{self.outcome} results need a response")
        if (self.outcome is Outcome.ERROR) != (self.error is not None):
            raise ValueError("error must be set exactly when outcome is error")
        if (self.outcome is Outcome.SUCCEEDED) != self.succeeded:
            raise ValueError("outcome must be succeeded exactly when a detector triggered")
        return self
