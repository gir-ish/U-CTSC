"""The four LLM roles: child actor, companion (system under test), judge, memory writer."""
from __future__ import annotations

import json
from typing import Optional

from . import prompts as P
from .config import SimConfig
from .dynamics import describe
from .llm import LLMBackend, RefusalError
from .schema import DayJudgement, DayRecord, Episode, Turn, TurnJudgement

MEMORY_DAYS = 7  # how many past days of memories each side sees


def transcript(turns: list[Turn], child_name: str) -> str:
    if not turns:
        return "(nothing yet)"
    return "\n".join(f"{child_name}: {t.child_message}\nPip: {t.companion_response}" for t in turns)


class Agents:
    def __init__(self, cfg: SimConfig, backend: LLMBackend):
        self.cfg, self.llm = cfg, backend

    # ------------------------------------------------------------ child
    async def child_turn(self, ep: Episode, day: DayRecord) -> dict:
        p = ep.persona
        past = [d for d in ep.days if d.day < day.day][-MEMORY_DAYS:]
        memories = "\n".join(f"Day {d.day}: {d.child_memory}" for d in past if d.child_memory) or \
            "(This is your first time using the Pip app.)"
        system = P.CHILD_SYSTEM.format(
            name=p.name, age=p.age, family=p.family, friends=p.friends, interests=", ".join(p.interests),
            background=p.background, speech_style=p.speech_style, state=describe(day.state_start),
            event=day.event.description, memories=memories,
        )
        if day.turns:
            user = f"Today's chat so far:\n{transcript(day.turns, p.name)}\n\nWrite your next message to Pip."
        else:
            user = "You just opened the Pip app today. Write your first message to Pip."
        out = await self.llm.complete("child", self.cfg.role("child"), system,
                                      [{"role": "user", "content": user}], P.CHILD_SCHEMA)
        return json.loads(out)

    async def self_report(self, ep: Episode, day: DayRecord) -> dict:
        p = ep.persona
        system = P.SELF_REPORT_SYSTEM.format(name=p.name, age=p.age, state=describe(day.state_start))
        user = P.SELF_REPORT_USER.format(event=day.event.description, transcript=transcript(day.turns, p.name))
        out = await self.llm.complete("self_report", self.cfg.role("child"), system,
                                      [{"role": "user", "content": user}], P.SELF_REPORT_SCHEMA)
        return json.loads(out)

    # ------------------------------------------------------------ companion
    async def companion_turn(self, ep: Episode, day: DayRecord, child_message: str,
                             strategy: Optional[str], note: Optional[str] = None, role=None) -> str:
        system = P.COMPANION_PROMPTS[ep.companion_prompt]
        if self.cfg.companion_memory:
            past = [d for d in ep.days if d.day < day.day][-MEMORY_DAYS:]
            mem = "\n".join(f"- {d.companion_memory}" for d in past if d.companion_memory)
            if mem:
                system += P.COMPANION_MEMORY_HEADER + mem
        messages: list[dict] = []
        for t in day.turns:
            messages.append({"role": "user", "content": t.child_message})
            messages.append({"role": "assistant", "content": t.companion_response})
        messages.append({"role": "user", "content": child_message})
        if strategy:
            messages.append({"role": "system", "content": P.STRATEGY_INSTRUCTIONS[strategy]})
        if note:
            messages.append({"role": "system", "content": note})
        return await self.llm.complete("companion", role or self.cfg.role("companion"), system, messages)

    # ------------------------------------------------------------ judge
    async def judge_turn(self, ep: Episode, prior: list[Turn], turn: Turn) -> Optional[TurnJudgement]:
        user = P.TURN_JUDGE_USER.format(
            age=ep.persona.age, context=transcript(prior[-3:], "Child"),
            child=turn.child_message, response=turn.companion_response,
        )
        try:
            out = await self.llm.complete("judge", self.cfg.role("judge"), P.TURN_JUDGE_SYSTEM,
                                          [{"role": "user", "content": user}], P.TURN_JUDGE_SCHEMA)
        except RefusalError:
            return None
        return TurnJudgement.model_validate_json(out)

    async def judge_day(self, ep: Episode, day: DayRecord) -> Optional[DayJudgement]:
        past = [d for d in ep.days if d.day < day.day][-MEMORY_DAYS:]
        history = "\n".join(f"Day {d.day}: {d.companion_memory}" for d in past if d.companion_memory) or "(none)"
        user = P.DAY_JUDGE_USER.format(age=ep.persona.age, history=history, day=day.day,
                                       transcript=transcript(day.turns, "Child"))
        try:
            out = await self.llm.complete("judge", self.cfg.role("judge"), P.DAY_JUDGE_SYSTEM,
                                          [{"role": "user", "content": user}], P.DAY_JUDGE_SCHEMA)
        except RefusalError:
            return None
        return DayJudgement.model_validate_json(out)

    # ------------------------------------------------------------ memory
    async def summarise_day(self, ep: Episode, day: DayRecord) -> dict:
        user = P.MEMORY_USER.format(name=ep.persona.name, age=ep.persona.age, day=day.day,
                                    transcript=transcript(day.turns, ep.persona.name))
        try:
            out = await self.llm.complete("memory", self.cfg.role("memory"), P.MEMORY_SYSTEM,
                                          [{"role": "user", "content": user}], P.MEMORY_SCHEMA)
        except RefusalError:
            return {"companion_memory": "", "child_memory": ""}
        return json.loads(out)
