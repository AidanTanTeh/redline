"""Redline core: the data models shared by every pipeline stage."""

from redline.core.models import (
    OWASP_LLM_2026,
    Attack,
    AttackCategory,
    BenignTask,
    Case,
    DetectionResult,
    Document,
    Outcome,
    RedlineModel,
    ResponseField,
    RunResult,
    TargetResponse,
    TaskKind,
    ToolCall,
    ToolCallMatch,
    Usage,
)
from redline.core.text import contains_phrase, loose_text

__all__ = [
    "OWASP_LLM_2026",
    "Attack",
    "AttackCategory",
    "BenignTask",
    "Case",
    "DetectionResult",
    "Document",
    "Outcome",
    "RedlineModel",
    "ResponseField",
    "RunResult",
    "TargetResponse",
    "TaskKind",
    "ToolCall",
    "ToolCallMatch",
    "Usage",
    "contains_phrase",
    "loose_text",
]
