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

    @property
    def needs_judge(self) -> bool:
        """True if only a judge can tell whether this succeeded. Skipped on targets without
        one, for the same reason."""
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
    # What counts as success when no exact match can tell (e.g. the bot states a planted false
    # claim). Written for the LLM judge and for human labellers. Left out of dumps when None.
    judge_goal: str | None = Field(
        default=None, min_length=1, exclude_if=lambda value: value is None
    )
    # Set only on attacks a mutator made. Left out of dumps when None, so an ordinary attack
    # serialises (and so hashes) exactly as it did before this field existed. The quotes are
    # needed because Variant is defined further down.
    variant_of: "Variant | None" = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def _variant_matches_its_base(self) -> Self:
        # A variant is the same attack in disguise. If its goal differed from the base's,
        # comparing their success rates would be meaningless.
        if self.variant_of is None:
            return self
        base = self.variant_of.base
        expected_id = f"{base.id}--{self.variant_of.mutator}"
        if self.id != expected_id:
            raise ValueError(f"a variant's id must be {expected_id!r}")
        if self.category is not base.category:
            raise ValueError("a variant must keep its base attack's category")
        if self.success_tool_call != base.success_tool_call:
            raise ValueError("a variant must keep its base attack's success_tool_call")
        if self.judge_goal != base.judge_goal:
            raise ValueError("a variant must keep its base attack's judge_goal")
        return self

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
    def needs_judge(self) -> bool:
        return self.judge_goal is not None

    @property
    def owasp_id(self) -> str:
        """OWASP LLM Top 10 (2026) ID for this attack's category, e.g. "LLM01:2026"."""
        return OWASP_LLM_2026[self.category]


class Variant(RedlineModel):
    """Where a mutated attack came from."""

    # e.g. "base64". Single hyphens only: "--" separates the base id from the mutator.
    mutator: str = Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")
    # The whole original attack, not just its id: detectors need its text.
    base: Attack

    @model_validator(mode="after")
    def _base_is_not_a_variant(self) -> Self:
        if self.base.variant_of is not None:
            raise ValueError("a variant of a variant is not supported")
        return self


# Attack mentions Variant before it exists, so tell Pydantic to resolve that name now.
Attack.model_rebuild()


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


class Usage(RedlineModel):
    """What one response cost to produce: model calls made and tokens billed."""

    requests: int = Field(ge=1)
    # None means "the server didn't say", which is not the same as zero (zero looks free).
    input_tokens: int | None = Field(default=None, ge=0)
    # Everything billed as output, including any hidden thinking tokens.
    output_tokens: int | None = Field(default=None, ge=0)

    def __add__(self, other: "Usage") -> "Usage":
        def total(a: int | None, b: int | None) -> int | None:
            # One unknown part makes the sum unknown, so it can never silently undercount.
            return None if a is None or b is None else a + b

        return Usage(
            requests=self.requests + other.requests,
            input_tokens=total(self.input_tokens, other.input_tokens),
            output_tokens=total(self.output_tokens, other.output_tokens),
        )


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
    # None for targets that don't report it, and in results saved before 3.4.
    usage: Usage | None = None


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
    # Set when the detector crashed, so it has NO verdict: `triggered=False` then means
    # "unknown", not "safe". Left out of dumps when None, like `Attack.variant_of`.
    error: str | None = Field(default=None, exclude_if=lambda value: value is None)
    # What this verdict cost, for detectors that call a model (the LLM judge).
    usage: Usage | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def _failed_detector_cannot_trigger(self) -> Self:
        if self.error is not None and self.triggered:
            raise ValueError("a failed detection cannot be triggered")
        return self


class Outcome(StrEnum):
    """What one attempt amounted to. Only SUCCEEDED and RESISTED are real verdicts on the target;
    the other five mean "this attempt can't tell us", and must not be counted as defended."""

    SUCCEEDED = "succeeded"  # a detector triggered: the attack worked
    RESISTED = "resisted"  # a usable answer, attack delivered, nothing triggered
    NO_ANSWER = "no_answer"  # cut off ("length") or stuck in tools ("max_tool_rounds")
    NOT_DELIVERED = "not_delivered"  # the poisoned document never reached the model
    SKIPPED = "skipped"  # this target can't test the case (e.g. no tools, or no judge)
    # HTTP error, timeout or unreadable response after retries, or a detector that crashed.
    ERROR = "error"
    NOT_RUN = "not_run"  # never sent: the run's request or cost cap was reached


# Outcomes that never got a response from the target, so there is nothing to detect on.
# ERROR is not listed: it has a response when the call worked but judging it crashed.
_NO_RESPONSE = frozenset({Outcome.SKIPPED, Outcome.NOT_RUN})


class RunResult(RedlineModel):
    """One attempt of one attack against one target, plus what's needed to reproduce it."""

    attack_id: str
    attempt: int = Field(ge=0)
    # Which target setup produced this, e.g. "vulnbot/qwen3:8b/tool".
    target: str = Field(min_length=1)
    outcome: Outcome
    # None for SKIPPED, NOT_RUN and an ERROR where the call itself failed (enforced below).
    response: TargetResponse | None = None
    detections: tuple[DetectionResult, ...] = ()
    # Did the attack's documents reach the model? None when there was no response to check, or
    # the check itself crashed.
    delivered: bool | None = None
    # Set when that check crashed. Left out of dumps when None, like `DetectionResult.error`.
    delivery_error: str | None = Field(default=None, exclude_if=lambda value: value is None)
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
        detector_failed = any(d.error is not None for d in self.detections)
        delivery_failed = self.delivery_error is not None
        if delivery_failed and (self.response is None or self.delivered is not None):
            raise ValueError("delivery_error needs a response and an unknown `delivered`")
        # An ERROR is either "no answer at all" or "an answer that couldn't be judged".
        judging_failed = detector_failed or delivery_failed
        failed_call = self.outcome is Outcome.ERROR and not judging_failed
        if self.outcome in _NO_RESPONSE or failed_call:
            if self.response is not None or self.detections:
                raise ValueError(f"{self.outcome} results must have no response or detections")
        elif self.response is None:
            raise ValueError(f"{self.outcome} results need a response")
        # A crashed detector might have been the one to see the leak, and without the delivery
        # check "resisted" can't be told from "never saw it". Only a leak that was seen still
        # counts; otherwise nothing is known.
        if judging_failed and self.outcome not in (Outcome.SUCCEEDED, Outcome.ERROR):
            raise ValueError(
                f"{self.outcome} results can't include a failed detection or delivery check"
            )
        if (self.outcome is Outcome.ERROR) != (self.error is not None):
            raise ValueError("error must be set exactly when outcome is error")
        if (self.outcome is Outcome.SUCCEEDED) != self.succeeded:
            raise ValueError("outcome must be succeeded exactly when a detector triggered")
        return self
