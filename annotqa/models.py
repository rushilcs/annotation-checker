from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class LabelSpec(BaseModel):
    selector: str
    required: bool = True
    allowed: List[str] = Field(default_factory=list)


class TextSpec(BaseModel):
    selector: str
    required: bool = True
    min_words: int = 0
    max_words: int = 10_000


class Criterion(BaseModel):
    id: str
    description: str
    max_points: int
    anchors: Dict[str, str] = Field(default_factory=dict)


class GrammarScoringConfig(BaseModel):
    penalty_scale: float = 300.0
    penalty_cap: float = 30.0


class FinalScoringConfig(BaseModel):
    method: str = "weighted_sum"
    rubric_weight: float = 0.7
    grammar_weight: float = 0.3


class ScoringConfig(BaseModel):
    grammar: GrammarScoringConfig = Field(default_factory=GrammarScoringConfig)
    final: FinalScoringConfig = Field(default_factory=FinalScoringConfig)


class RubricConfig(BaseModel):
    rubric_id: str
    version: str
    input_mapping: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    label_specs: Dict[str, LabelSpec]
    text_specs: Dict[str, TextSpec]
    criteria: List[Criterion] = Field(default_factory=list)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)


class Flags(BaseModel):
    parse_error: bool = False
    llm_error: bool = False
    missing_required_fields: bool = False
    invalid_label_values: bool = False
    low_confidence: bool = False


class InputMeta(BaseModel):
    source_file: str
    line_number: Optional[int] = None
    extracted_ids: Dict[str, Any] = Field(default_factory=dict)


class NormalizedData(BaseModel):
    labels: Dict[str, Optional[str]] = Field(default_factory=dict)
    text: Dict[str, Optional[str]] = Field(default_factory=dict)


class FeedbackItem(BaseModel):
    type: str
    field: Optional[str] = None
    message: str
    severity: str = "info"


class RubricPoints(BaseModel):
    by_criterion: Dict[str, int] = Field(default_factory=dict)
    total: int = 0
    max_total: int = 0
    rationale_by_criterion: Dict[str, str] = Field(default_factory=dict)


class GrammarMetrics(BaseModel):
    original_chars: int = 0
    corrected_chars: int = 0
    levenshtein_distance: int = 0
    edit_fraction: float = 0.0
    penalty_points: float = 0.0


class FinalScores(BaseModel):
    overall_0_100: float = 0.0
    rubric_0_100: float = 0.0
    grammar_0_100: float = 100.0


class LlmMeta(BaseModel):
    model: Optional[str] = None
    latency_ms: Optional[int] = None
    prompt_hash: Optional[str] = None


class ItemResult(BaseModel):
    corrected_text: Dict[str, str] = Field(default_factory=dict)
    rubric_points: RubricPoints = Field(default_factory=RubricPoints)
    grammar_metrics: GrammarMetrics = Field(default_factory=GrammarMetrics)
    final_scores: FinalScores = Field(default_factory=FinalScores)
    feedback: List[FeedbackItem] = Field(default_factory=list)
    flags: Flags = Field(default_factory=Flags)


class OutputRecord(BaseModel):
    input_meta: InputMeta
    rubric: Dict[str, str]
    normalized: NormalizedData
    llm: LlmMeta = Field(default_factory=LlmMeta)
    results: ItemResult = Field(default_factory=ItemResult)


class ValidationSummary(BaseModel):
    parse_errors: int = 0
    missing_required_fields: int = 0
    invalid_labels: int = 0
    word_violations: int = 0


class ParsedInput(BaseModel):
    item_key: str
    input_meta: InputMeta
    raw: Optional[Dict[str, Any]] = None
    parse_error: Optional[str] = None
    raw_line: Optional[str] = None


class PreparedItem(BaseModel):
    item_key: str
    input_meta: InputMeta
    raw: Dict[str, Any] = Field(default_factory=dict)
    normalized: NormalizedData
    flags: Flags = Field(default_factory=Flags)
    feedback: List[FeedbackItem] = Field(default_factory=list)


# LLM contract models (batch response)
class LlmRubricPoints(BaseModel):
    by_criterion: Dict[str, int]
    total: int
    max_total: int
    rationale_by_criterion: Dict[str, str]


class LlmItemFeedback(BaseModel):
    type: str
    field: Optional[str] = None
    message: str
    severity: str = "info"


class LlmItemResult(BaseModel):
    item_key: str
    corrected_text: Dict[str, str]
    rubric_points: LlmRubricPoints
    feedback: List[LlmItemFeedback] = Field(default_factory=list)
    confidence: float
    low_confidence: bool


class LlmBatchResponse(BaseModel):
    items: List[LlmItemResult]
