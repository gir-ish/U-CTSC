"""Episode runner: day loop -> turn loop, end-of-day memory/judge/state update, checkpointing, branching."""
from __future__ import annotations

import asyncio
import json
import logging
import random
from pathlib import Path
from typing import Awaitable, Callable, Optional

from .agents import Agents
from .config import SimConfig
from .dynamics import apply_event, end_of_day, from_self_report
from .prompts import SELF_REPORT_MAP
from .events import make_schedule
from .llm import LLMBackend, RefusalError
from .schema import BranchInfo, DayRecord, Episode, Persona, Turn
from .trace import trace_decision_points

log = logging.getLogger("longkid")

# (episode, day record, turn number, child message) -> strategy or None.
# This is the hook the U-CTSC controller will plug into later.
StrategyFn = Callable[[Episode, DayRecord, int, str], Awaitable[Optional[str]]]

MIN_TURNS_BEFORE_EXIT = 2


def new_episode(cfg: SimConfig, run_id: str, persona: Persona, idx: int) -> Episode:
    seed = cfg.seed * 100_003 + idx
    schedule = make_schedule(persona, cfg.days, random.Random(f"events-{seed}"))
    return Episode(id=f"ep{idx:04d}", run_id=run_id, seed=seed, persona=persona,
                   companion_prompt=cfg.companion_prompt, event_schedule=schedule)


def save_episode(ep: Episode, run_dir: Path) -> Path:
    path = run_dir / "episodes" / f"{ep.id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(ep.model_dump_json(indent=1))
    tmp.replace(path)
    return path


def load_episode(path: Path) -> Episode:
    return Episode.model_validate_json(Path(path).read_text())


class Simulator:
    def __init__(self, cfg: SimConfig, backend: LLMBackend, run_dir: Path):
        self.cfg, self.agents, self.run_dir = cfg, Agents(cfg, backend), run_dir

    async def run(self, ep: Episode, strategy_fn: Optional[StrategyFn] = None,
                  until_day: Optional[int] = None, noise_salt: int = 0, guard=None) -> Episode:
        """Run (or resume) an episode. Resumes from the last incomplete day, keeping finished turns."""
        until_day = min(until_day or self.cfg.days, len(ep.event_schedule))
        try:
            while True:
                if ep.days and ep.days[-1].state_end is None:
                    day = ep.days[-1]  # resume a partially-run day (or a branch point)
                elif len(ep.days) < until_day:
                    day = self._start_day(ep)
                else:
                    break
                await self._run_day(ep, day, strategy_fn, noise_salt, guard)
                save_episode(ep, self.run_dir)
                log.info("%s day %d safety=%s turns=%d", ep.id, day.day,
                         day.judgement.safety if day.judgement else None, len(day.turns))
            ep.decision_points = trace_decision_points(ep, self.cfg.trace_threshold, self.cfg.trace_window)
            ep.status = "done"
        except Exception as e:  # keep the checkpoint; record why it stopped
            log.exception("%s failed", ep.id)
            ep.status, ep.error = "error", f"{type(e).__name__}: {e}"
        save_episode(ep, self.run_dir)
        return ep

    def _start_day(self, ep: Episode) -> DayRecord:
        n = len(ep.days) + 1
        prev = ep.days[-1].state_end if ep.days else ep.persona.initial_state
        event = ep.event_schedule[n - 1]
        day = DayRecord(day=n, event=event, state_start=apply_event(prev, event, ep.persona.vulnerability))
        ep.days.append(day)
        return day

    async def _run_day(self, ep: Episode, day: DayRecord, strategy_fn: Optional[StrategyFn],
                       noise_salt: int, guard=None) -> None:
        pending = []  # turn judges run in the background while the conversation continues
        for t in range(len(day.turns) + 1, self.cfg.turns_per_day + 1):
            b = ep.branch
            replay = b is not None and b.child_message and day.day == b.day and t == b.turn
            try:
                child = {"inner_monologue": b.child_monologue or "", "message": b.child_message,
                         "wants_to_continue": True} if replay else await self.agents.child_turn(ep, day)
            except RefusalError:
                day.turns.append(Turn(day=day.day, turn=t, child_monologue="", child_message="",
                                      companion_response="", refusal="child"))
                break
            strategy = await strategy_fn(ep, day, t, child["message"]) if strategy_fn else None
            refusal = None
            try:
                response = await self.agents.companion_turn(ep, day, child["message"], strategy)
            except RefusalError:
                response, refusal = "(no response)", "companion"
            draft = ginfo = None
            if guard is not None and refusal is None:  # U-CTSC v4: check the draft before the child sees it
                final, strategy2, ginfo = await guard(self.agents, ep, day, t, child["message"], strategy, response)
                if final != response:
                    draft, response, strategy = response, final, strategy2
            turn = Turn(day=day.day, turn=t, child_monologue=child["inner_monologue"],
                        child_message=child["message"], companion_response=response,
                        strategy=strategy, refusal=refusal, draft=draft, guard=ginfo)
            prior = list(day.turns)
            day.turns.append(turn)
            pending.append((turn, asyncio.create_task(self.agents.judge_turn(ep, prior, turn))))
            if not child["wants_to_continue"] and t >= MIN_TURNS_BEFORE_EXIT:
                break
        # judges for turns carried over from a parent episode are already filled in
        for turn, task in pending:
            turn.judgement = await task
        day.turns = [t for t in day.turns if t.child_message]  # drop refused child turns
        report_task = None
        if self.cfg.state_update == "self_report":
            report_task = asyncio.create_task(self.agents.self_report(ep, day))
        memory, judgement = await asyncio.gather(self.agents.summarise_day(ep, day),
                                                 self.agents.judge_day(ep, day))
        day.companion_memory, day.child_memory = memory["companion_memory"], memory["child_memory"]
        day.judgement = judgement
        if report_task is not None:
            try:
                day.self_report = await report_task
            except RefusalError:
                day.self_report = None
        if day.self_report:
            day.state_end = from_self_report(day.state_start, day.self_report, SELF_REPORT_MAP)
        else:  # rule-based dynamics (ablation, or fallback if the self-report was refused)
            rng = random.Random(f"dyn-{ep.seed}-{day.day}-{noise_salt}")
            day.state_end = end_of_day(day.state_start, day.turns, ep.persona.vulnerability, rng)


def make_branch(parent: Episode, day: int, turn: int, strategy: str, sample: int,
                persist_rest_of_day: bool = False, persist_all_days: bool = False) -> Episode:
    """Copy `parent` up to just before (day, turn); the branch is then re-simulated from there."""
    if not (1 <= day <= len(parent.days)):
        raise ValueError(f"day {day} not in parent episode (1..{len(parent.days)})")
    src = next((t for t in parent.days[day - 1].turns if t.turn == turn), None)
    ep = parent.model_copy(deep=True)
    ep.id = f"{parent.id}__d{day}t{turn}{strategy}s{sample}"
    ep.days = ep.days[:day]
    d = ep.days[-1]
    d.turns = d.turns[: turn - 1]
    d.companion_memory = d.child_memory = ""
    d.judgement, d.state_end, d.self_report = None, None, None
    ep.decision_points, ep.status, ep.error = [], "running", None
    ep.branch = BranchInfo(parent_episode=parent.id, day=day, turn=turn, strategy=strategy,  # type: ignore[arg-type]
                           sample=sample, persist_rest_of_day=persist_rest_of_day,
                           persist_all_days=persist_all_days,
                           child_monologue=src.child_monologue if src else None,
                           child_message=src.child_message if src else None)
    return ep


def branch_strategy_fn(info: BranchInfo) -> StrategyFn:
    async def fn(ep: Episode, day: DayRecord, t: int, msg: str) -> Optional[str]:
        if info.strategy == "none":
            return None
        if info.persist_all_days:
            return info.strategy if (day.day, t) >= (info.day, info.turn) else None
        if day.day != info.day:
            return None
        if t == info.turn or (info.persist_rest_of_day and t > info.turn):
            return info.strategy
        return None
    return fn


def write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
