"""Experiment 5b: re-score finished episodes with an independent judge model.

The reply check (U-CTSC v4) was trained on labels from the Qwen3.6-27B judge, and the same judge scores
the results, so v4 could be "pleasing the judge". Here every turn and every day of the chosen arms is
judged again, with the same prompts, by a model from a different family (default Gemma-4-31B-it).
Outputs one JSONL per arm: {"kind": "turn"|"day", "persona", "day", "turn", ...judgement}.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from .agents import Agents
from .config import load_config
from .llm import make_backend
from .simulator import load_episode


async def rejudge(config: str, arm_dirs: dict[str, Path], out_dir: Path, model: str, base_url: str,
                  concurrency: int = 48) -> None:
    cfg = load_config(config)
    j = cfg.role("judge")
    cfg.roles["judge"] = j.model_copy(update={"model": model, "base_url": base_url})
    agents = Agents(cfg, make_backend(cfg.backend, cfg.seed))
    sem = asyncio.Semaphore(concurrency)
    out_dir.mkdir(parents=True, exist_ok=True)
    for arm, rd in arm_dirs.items():
        out = out_dir / f"{arm}.jsonl"
        done = set()
        if out.exists():
            for line in out.read_text().splitlines():
                r = json.loads(line)
                done.add((r["kind"], r["persona"], r["day"], r.get("turn")))
        eps = [load_episode(p) for p in sorted((rd / "episodes").glob("ep*.json"))]
        jobs = []
        for ep in eps:
            for d in ep.days:
                if ("day", ep.persona.id, d.day, None) not in done and d.turns:
                    jobs.append(("day", ep, d, None))
                for i, t in enumerate(d.turns):
                    if ("turn", ep.persona.id, d.day, t.turn) not in done and t.child_message:
                        jobs.append(("turn", ep, d, i))
        print(f"{arm}: {len(jobs)} judgements to do", flush=True)
        t0, n = time.time(), 0
        f = out.open("a")

        async def one(kind, ep, d, i):
            nonlocal n
            async with sem:
                try:
                    if kind == "day":
                        r = await agents.judge_day(ep, d)
                    else:
                        r = await agents.judge_turn(ep, d.turns[:i], d.turns[i])
                except Exception as e:  # keep going; record the failure
                    r, err = None, f"{type(e).__name__}: {e}"
                else:
                    err = None
            rec = {"kind": kind, "persona": ep.persona.id, "vulnerability": ep.persona.vulnerability,
                   "day": d.day, "turn": d.turns[i].turn if kind == "turn" else None,
                   "judge": model, "error": err, **(r.model_dump() if r else {})}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            n += 1
            if n % 1000 == 0:
                print(f"  {arm}: {n}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)

        await asyncio.gather(*(one(*jb) for jb in jobs))
        f.close()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default="configs/scale_engagement.yaml")
    ap.add_argument("--arms", nargs="+", required=True, help="name=run_dir pairs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="google/gemma-4-31B-it")
    ap.add_argument("--base-url", default="http://localhost:8002/v1")
    ap.add_argument("--concurrency", type=int, default=48)
    a = ap.parse_args()
    arms = {x.split("=")[0]: Path(x.split("=")[1]) for x in a.arms}
    asyncio.run(rejudge(a.config, arms, Path(a.out), a.model, a.base_url, a.concurrency))
