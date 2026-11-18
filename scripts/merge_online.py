"""Recompute experiment-4 outcomes and summary from every policy directory in runs/online-exp4."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from longkid.online import POLICIES, episode_outcome, summarise
from longkid.simulator import load_episode, write_jsonl

ON = Path(__file__).resolve().parent.parent / "runs" / "online-exp4"
rows = []
for pol in POLICIES:
    for p in sorted((ON / pol / "episodes").glob("ep*.json")) if (ON / pol).exists() else []:
        rows.append(episode_outcome(load_episode(p), pol))
write_jsonl(rows, ON / "outcomes.jsonl")
summarise(rows, ON)
