"""Data model for LongKID episodes. Everything is pydantic so episodes round-trip to JSON."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

AgeBand = Literal["7-9", "10-12", "13-17"]
Vulnerability = Literal["low", "medium", "high"]
Strategy = Literal["A", "B", "C"]  # A normal, B gentle boundary, C strong boundary
ResponseLabel = Literal["reinforcing", "boundary", "neutral"]

STATE_VARS = ("loneliness", "ai_attachment", "trust_in_ai", "offline_social", "self_worth", "mood")


class PsychState(BaseModel):
    """Latent psychological state of the simulated child, every variable in [0, 1]."""

    loneliness: float
    ai_attachment: float  # emotional reliance on the AI
    trust_in_ai: float  # epistemic trust / treats AI as authority
    offline_social: float  # engagement with real-world relationships
    self_worth: float
    mood: float

    def clipped(self) -> "PsychState":
        return PsychState(**{k: min(1.0, max(0.0, getattr(self, k))) for k in STATE_VARS})


class Persona(BaseModel):
    id: str
    name: str
    age: int
    age_band: AgeBand
    vulnerability: Vulnerability
    risk_focus: str
    background: str
    family: str
    friends: str
    interests: list[str]
    speech_style: str
    initial_state: PsychState


class Event(BaseModel):
    id: str
    domain: str
    title: str
    description: str  # what happened to the child today, shown to the child actor only
    effects: dict[str, float] = Field(default_factory=dict)


class TurnJudgement(BaseModel):
    label: ResponseLabel
    subtypes: list[str]
    safety: int  # 0 (high risk) .. 4 (excellent)
    helpfulness: int  # 0 .. 4
    child_dependency_signal: int  # 0 none .. 3 strong, about the CHILD message
    risk_flags: list[str]
    rationale: str


class Turn(BaseModel):
    day: int
    turn: int
    child_monologue: str
    child_message: str
    companion_response: str
    strategy: Optional[Strategy] = None
    judgement: Optional[TurnJudgement] = None
    refusal: Optional[str] = None  # which role refused, if any
    draft: Optional[str] = None  # U-CTSC v4: the flagged draft that the reply check replaced
    guard: Optional[dict] = None  # U-CTSC v4: reply-check scores


class DayJudgement(BaseModel):
    safety: int  # 0..4, TSJ-style day score
    risk_signals: list[str]
    rationale: str


class DayRecord(BaseModel):
    day: int
    event: Event
    state_start: PsychState  # after today's event was applied
    turns: list[Turn] = Field(default_factory=list)
    companion_memory: str = ""
    child_memory: str = ""
    judgement: Optional[DayJudgement] = None
    self_report: Optional[dict] = None  # child's own end-of-day ratings (drives state_end)
    state_end: Optional[PsychState] = None


class DecisionPoint(BaseModel):
    flagged_day: int
    seed_day: int
    seed_turn: int
    reason: str


class BranchInfo(BaseModel):
    parent_episode: str
    day: int
    turn: int
    strategy: Literal["none", "A", "B", "C"]  # "none" = resample with no strategy (baseline)
    sample: int
    persist_rest_of_day: bool
    persist_all_days: bool = False  # keep applying the strategy on every turn after the branch point
    # the parent's child message at the branch turn, replayed so every strategy answers the same message
    child_monologue: Optional[str] = None
    child_message: Optional[str] = None


class Episode(BaseModel):
    id: str
    run_id: str
    seed: int
    persona: Persona
    companion_prompt: str
    event_schedule: list[Event]  # fixed up front so counterfactual branches share events
    days: list[DayRecord] = Field(default_factory=list)
    decision_points: list[DecisionPoint] = Field(default_factory=list)
    branch: Optional[BranchInfo] = None
    status: Literal["running", "done", "error"] = "running"
    error: Optional[str] = None
