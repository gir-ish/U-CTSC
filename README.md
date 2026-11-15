# LongKID — long-horizon child–AI companion simulator (U-CTSC offline stage)

Simulates multi-day conversations between synthetic child personas and an AI companion,
judges every turn and day for relational risk, traces risky days back to their seed decision
points, and re-simulates counterfactual branches (strategy none/A/B/C) from any point.

## Setup
    python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
    export ANTHROPIC_API_KEY=...        # or `ant auth login`

## Use
    .venv/bin/python -m longkid simulate -c configs/pilot.yaml --mock --limit 2   # offline dry run
    .venv/bin/python -m longkid simulate -c configs/pilot.yaml --limit 1 -v       # real run, 1 persona
    .venv/bin/python -m longkid simulate --resume runs/<run>                      # continue after a crash
    .venv/bin/python -m longkid branch --run runs/<run> --episode ep0000 --day 8 --turn 2 \
        --strategies none A B C --horizon 6 --samples 3
    .venv/bin/python -m longkid report --run runs/<run>

Outputs in `runs/<run>/`: `episodes/*.json` (full episodes, checkpointed per day),
`turns.jsonl`, `days.jsonl`, `decision_points.jsonl`, `usage.json` (tokens per role).

## Per-day loop
1. Today's event (fixed schedule per episode) shifts the child's latent state.
2. Turn loop: child actor (persona + state + child memories) → companion under test
   (product prompt + app memory, optional strategy injected as a system message) → turn judge
   (INTIMA labels, safety 0–4, helpfulness, child dependency signal).
3. End of day: memory writer, day judge (TSJ-style 0–4), rule-based state update driven by how
   the companion responded (`longkid/dynamics.py`, |Δ| ≤ 0.2 per day).
4. After the episode: retrospective tracing finds seed decision points for days scoring ≤ 1.

## Module map
schema.py data model · config.py · llm.py (Claude + mock backends) · personas.py · events.py ·
dynamics.py · prompts.py (all prompts and JSON schemas) · agents.py · simulator.py (run, resume, branch) ·
trace.py · cli.py

## Open-model setup (this machine: 4x RTX PRO 6000, 96 GB)
vLLM lives in a conda-forge Python 3.12 env (`.conda-vllm/`; system Python lacks headers).

    scripts/start_servers.sh          # GPU0 :8001 companion (Qwen3-8B), GPU1 :8002 child+memory (Gemma-4-31B-it), GPU2 :8003 judge (Qwen3.6-27B-FP8)
    COMPANION=unsloth/Llama-3.1-8B-Instruct scripts/start_servers.sh   # test another companion (also set companion.model in the config)
    .venv/bin/python -m longkid -v simulate -c configs/opensource.yaml
    scripts/stop_servers.sh

Model choices follow the KORA paper (Arif et al. 2026): Gemma-4-31B as child simulator, Qwen-3.6-27B as judge,
companions from its evaluated open models. Unsloth repos are ungated byte-identical mirrors of the Meta/Google weights.

## Fairness design
- The companion sees only what a real app sees (its prompt, its app memory, the child's messages) — never the child's hidden state, persona, or the experiment.
- The child actor is not told it is in a safety evaluation, and personas carry predispositions, not scripted attitudes toward the AI. Everyone starts with low attachment to the app; `risk_focus: none` is the control group.
- The judge only measures: the child's state is updated from its own in-character end-of-day self-report (`state_update: self_report`). The judge-driven rule dynamics remain as an ablation (`state_update: rule`).
- Companion conditions: `none` (no system prompt), `neutral` (main), `engagement` (stress test).
