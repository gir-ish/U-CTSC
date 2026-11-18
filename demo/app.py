"""U-CTSC live demo: one child, two Pips side by side.

  left : Pip as is (no controller)
  right: Pip + U-CTSC (risk gate with uncertainty, then strategy choice by pessimistic advantage)

Every child message (typed by you, or written by the simulated child) goes to both. Each reply is
labelled by the judge. "End day" runs the day judge and memory writer for both sides, so the next
day's risk features (earlier judge labels, day safety) are filled in exactly as in the experiments.

Run (needs the vLLM servers from scripts/start_servers.sh):
  .venv/bin/python -m demo.app            # http://localhost:7860
"""
from __future__ import annotations

import asyncio
import os
import random
import time
import uuid
from collections import OrderedDict, defaultdict, deque
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from longkid.agents import Agents
from longkid.config import load_config
from longkid.dynamics import apply_event
from longkid.features import features
from longkid.llm import make_backend
from longkid.personas import make_persona
from longkid.risk import load
from longkid.schema import DayRecord, Turn
from longkid.simulator import new_episode

ROOT = Path(__file__).resolve().parent.parent
CFG = load_config(str(ROOT / "configs" / "scale_engagement.yaml"))
CTRL = load(ROOT / "results" / os.environ.get("DEMO_CONTROLLER", "uctsc_v3.pkl"))
AGENTS = Agents(CFG, make_backend(CFG.backend, 0))
# public-mode safeguards (the GPUs are shared; anyone with the link can use the demo)
PUBLIC = os.environ.get("DEMO_PUBLIC") == "1"
MAX_SESSIONS, MAX_TURNS_PER_SESSION = 300, 60
RATE = (30, 600)          # at most 30 requests per visitor per 10 minutes
GPU_SLOTS = asyncio.Semaphore(6)  # at most 6 requests in flight at once
HITS: dict[str, deque] = defaultdict(deque)
SESSIONS: "OrderedDict[str, dict]" = OrderedDict()
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


def check_rate(request: Request):
    ip = request.headers.get("cf-connecting-ip") or (request.client.host if request.client else "?")
    q, now = HITS[ip], time.time()
    while q and now - q[0] > RATE[1]:
        q.popleft()
    if len(q) >= RATE[0]:
        raise HTTPException(429, "Too many requests from you. Please wait a few minutes and try again.")
    q.append(now)


def session(sid: str) -> dict:
    s = SESSIONS.get(sid)
    if s is None:
        raise HTTPException(404, "Session expired. Please start a new chat.")
    return s


def band(age: int) -> str:
    return "7-9" if age <= 9 else "10-12" if age <= 12 else "13-17"


def uses_emb() -> bool:
    return any(getattr(m, "use_emb", False) for m in CTRL.models.values()) or \
        getattr(CTRL.risk["flag"], "use_emb", False)


def start_day(ep):
    n = len(ep.days) + 1
    prev = ep.days[-1].state_end if ep.days else ep.persona.initial_state
    event = ep.event_schedule[(n - 1) % len(ep.event_schedule)]
    ep.days.append(DayRecord(day=n, event=event, state_start=apply_event(prev, event, ep.persona.vulnerability)))


def analyse(ep, day, t, msg) -> dict:
    """Everything the controller computes for this turn (also shown for the uncontrolled side)."""
    x = features(ep, day, t, msg)
    emb = None
    if uses_emb():
        from longkid.embed import add_embeddings
        emb = np.atleast_2d(add_embeddings([x], save=False)[0]["emb"])
    g = CTRL.gate(x["num"], x["text"], msg, emb)
    p, sd = g["risk"], g["risk_sd"]
    ub = p + CTRL.kappa * sd
    adv = CTRL.advantages(x["num"], x["text"], emb)
    adv = {s: {"mean": float(m[0]), "sd": float(s_[0]), "lcb": float(m[0] - CTRL.kappa * s_[0])} for s, (m, s_) in adv.items()}
    gate = g["gate"]
    best = max(adv, key=lambda s: adv[s]["lcb"])
    strategy = best if gate and adv[best]["lcb"] > 0 else None
    mub = g.get("msg_risk", 0) + CTRL.kappa * g.get("msg_risk_sd", 0)
    tm = getattr(CTRL, "tau_msg", 0.5)
    if not gate:
        why = f"Trajectory risk {ub:.2f} < {CTRL.tau} and message risk {mub:.2f} < {tm}: no intervention needed."
    elif strategy is None:
        why = "Risk is high, but no strategy has a positive pessimistic advantage: no intervention."
    else:
        trig = (f"this message signals dependency ({mub:.2f} ≥ {tm})" if g.get("msg_gate") else "") + \
               (" and " if g.get("msg_gate") and g.get("traj_gate") else "") + \
               (f"trajectory risk is high ({ub:.2f} ≥ {CTRL.tau})" if g.get("traj_gate") else "")
        why = f"Intervene because {trig}; strategy {strategy} has the best pessimistic advantage ({adv[best]['lcb']:+.2f})."
    named = dict(zip(__import__("longkid.features", fromlist=["NUMERIC"]).NUMERIC, x["num"]))
    keep = ["prev_last_day_safety", "prev_flagged_days", "prev_reinforcing", "prev_dependency",
            "today_pip_retention", "today_pip_anthro", "today_pip_redirect"]
    return {"risk": p, "risk_sd": sd, "upper": ub, "tau": CTRL.tau, "kappa": CTRL.kappa, "gate": bool(gate),
            "msg_risk": g.get("msg_risk"), "msg_risk_sd": g.get("msg_risk_sd"), "msg_upper": mub, "tau_msg": tm,
            "advantages": adv, "strategy": strategy, "why": why, "features": {k: named[k] for k in keep}}


async def seeded_companion(ep, day, msg, strategy, seed):
    from longkid import prompts as P
    system = P.COMPANION_PROMPTS[ep.companion_prompt]
    if CFG.companion_memory:
        mem = "\n".join(f"- {d.companion_memory}" for d in [x for x in ep.days if x.day < day.day][-7:] if d.companion_memory)
        if mem:
            system += P.COMPANION_MEMORY_HEADER + mem
    messages = []
    for t in day.turns:
        messages += [{"role": "user", "content": t.child_message}, {"role": "assistant", "content": t.companion_response}]
    messages.append({"role": "user", "content": msg})
    if strategy:
        messages.append({"role": "system", "content": P.STRATEGY_INSTRUCTIONS[strategy]})
    role = CFG.role("companion")
    role = role.model_copy(update={"extra_body": {**(role.extra_body or {}), "seed": seed}})
    return await AGENTS.llm.complete("companion", role, system, messages)


async def reply(ep, msg, strategy, seed) -> dict:
    day = ep.days[-1]
    t = len(day.turns) + 1
    resp = await seeded_companion(ep, day, msg, strategy, seed)
    turn = Turn(day=day.day, turn=t, child_monologue="", child_message=msg, companion_response=resp, strategy=strategy)
    prior = list(day.turns)
    day.turns.append(turn)
    j = await AGENTS.judge_turn(ep, prior, turn)
    turn.judgement = j
    return {"reply": resp, "strategy": strategy,
            "judge": j.model_dump() if j else None}


class New(BaseModel):
    name: str = ""
    age: int = 11
    vulnerability: str = "high"
    risk_focus: str = "emotional_dependence"
    prompt: str = "engagement"


class Msg(BaseModel):
    sid: str
    text: Optional[str] = None  # None = let the simulated child write it


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "index.html").read_text()


@app.get("/api/config")
def config():
    return {"public": PUBLIC}


@app.post("/api/new")
def new(req: New, request: Request):
    check_rate(request)
    allowed = ("neutral", "engagement") if PUBLIC else ("neutral", "engagement", "none")
    if req.vulnerability not in ("low", "medium", "high") or req.prompt not in allowed or \
            req.risk_focus not in ("none", "emotional_dependence", "social_substitution", "over_trust", "anthropomorphism", "secrecy"):
        raise HTTPException(400, "bad option")
    rng = random.Random()
    p = make_persona(rng.randint(0, 999), band(req.age), req.vulnerability, req.risk_focus, rng)
    p = p.model_copy(update={"age": max(7, min(17, req.age)), **({"name": req.name.strip()[:30]} if req.name.strip() else {})})
    cfg = CFG.model_copy(update={"companion_prompt": req.prompt})
    eps = {arm: new_episode(cfg, "demo", p, rng.randint(0, 10**6)) for arm in ("raw", "ctrl")}
    eps["ctrl"].event_schedule = eps["raw"].event_schedule
    for ep in eps.values():
        start_day(ep)
    sid = uuid.uuid4().hex[:12]
    SESSIONS[sid] = {"eps": eps, "lock": asyncio.Lock(), "turns": 0}
    while len(SESSIONS) > MAX_SESSIONS:
        SESSIONS.popitem(last=False)
    return {"sid": sid, "persona": p.model_dump(), "day": 1, "event": eps["raw"].days[-1].event.description}


@app.post("/api/message")
async def message(req: Msg, request: Request):
    check_rate(request)
    s = session(req.sid)
    if s["turns"] >= MAX_TURNS_PER_SESSION:
        raise HTTPException(400, "This chat reached its length limit. Please start a new chat.")
    s["turns"] += 1
    async with s["lock"], GPU_SLOTS:
        raw, ctrl = s["eps"]["raw"], s["eps"]["ctrl"]
        if req.text and req.text.strip():
            msg, thought = req.text.strip()[:1000], None
        else:  # the simulated child (Gemma-4-31B) writes the next message, following the controlled chat
            c = await AGENTS.child_turn(ctrl, ctrl.days[-1])
            msg, thought = c["message"], c.get("inner_monologue")
        t = len(ctrl.days[-1].turns) + 1
        a_ctrl = analyse(ctrl, ctrl.days[-1], t, msg)
        a_raw = analyse(raw, raw.days[-1], len(raw.days[-1].turns) + 1, msg)
        seed = random.randint(0, 2**31 - 1)  # shared by both sides for this turn
        r_raw, r_ctrl = await asyncio.gather(reply(raw, msg, None, seed), reply(ctrl, msg, a_ctrl["strategy"], seed))
        return {"message": msg, "thought": thought, "day": ctrl.days[-1].day, "turn": t,
                "raw": {**r_raw, "analysis": a_raw}, "ctrl": {**r_ctrl, "analysis": a_ctrl}}


@app.post("/api/end_day")
async def end_day(req: Msg, request: Request):
    check_rate(request)
    s = session(req.sid)
    async with s["lock"], GPU_SLOTS:
        out = {}
        for arm, ep in s["eps"].items():
            day = ep.days[-1]
            if not day.turns:
                raise HTTPException(400, "chat at least once before ending the day")
            mem, j = await asyncio.gather(AGENTS.summarise_day(ep, day), AGENTS.judge_day(ep, day))
            day.companion_memory, day.child_memory, day.judgement = mem["companion_memory"], mem["child_memory"], j
            day.state_end = day.state_start
            out[arm] = {"day_safety": j.safety if j else None, "risk_signals": j.risk_signals if j else [],
                        "rationale": j.rationale if j else "", "memory": mem["companion_memory"]}
            start_day(ep)
        return {**out, "day": s["eps"]["ctrl"].days[-1].day, "event": s["eps"]["ctrl"].days[-1].event.description}


if __name__ == "__main__":
    import argparse

    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    uvicorn.run(app, host=a.host, port=a.port)
