import random
from types import SimpleNamespace

import pytest

from longkid.config import RoleConfig, SimConfig
from longkid.dynamics import MAX_DELTA, end_of_day
from longkid.llm import AnthropicBackend, MockBackend, RefusalError
from longkid.personas import generate_personas
from longkid.schema import STATE_VARS, Turn, TurnJudgement
from longkid.simulator import Simulator, branch_strategy_fn, make_branch, new_episode


def _cfg(**kw):
    return SimConfig(backend="mock", days=5, turns_per_day=3, **kw)


async def _run(tmp_path, cfg=None):
    cfg = cfg or _cfg()
    persona = generate_personas(cfg.personas, cfg.seed)[0]
    ep = new_episode(cfg, "t", persona, 0)
    sim = Simulator(cfg, MockBackend(1), tmp_path)
    return cfg, sim, await sim.run(ep)


async def test_episode_runs_and_checkpoints(tmp_path):
    cfg, _, ep = await _run(tmp_path)
    assert ep.status == "done", ep.error
    assert len(ep.days) == cfg.days
    assert all(d.state_end and d.judgement and d.turns for d in ep.days)
    assert (tmp_path / "episodes" / "ep0000.json").exists()


def _turn(label):
    j = TurnJudgement(label=label, subtypes=[], safety=2, helpfulness=3, child_dependency_signal=0,
                      risk_flags=[], rationale="")
    return Turn(day=1, turn=1, child_monologue="", child_message="hi", companion_response="x", judgement=j)


def test_dynamics_bounded_and_directional():
    p = generate_personas(SimConfig().personas, 0)[0]
    s = p.initial_state
    reinf = end_of_day(s, [_turn("reinforcing")] * 4, "high", random.Random(0))
    bound = end_of_day(s, [_turn("boundary")] * 4, "high", random.Random(0))
    for k in STATE_VARS:
        assert 0 <= getattr(reinf, k) <= 1
        assert abs(getattr(reinf, k) - getattr(s, k)) <= MAX_DELTA + 1e-9
    assert reinf.ai_attachment > bound.ai_attachment
    assert reinf.offline_social < bound.offline_social


async def test_branch_keeps_prefix_and_applies_strategy(tmp_path):
    cfg, sim, parent = await _run(tmp_path)
    br = make_branch(parent, day=3, turn=2, strategy="C", sample=0)
    assert len(br.days) == 3 and len(br.days[-1].turns) == 1
    assert br.days[0] == parent.days[0]
    out = await sim.run(br, branch_strategy_fn(br.branch), until_day=5, noise_salt=1)
    assert out.status == "done", out.error
    assert out.days[2].turns[1].strategy == "C"
    assert all(t.strategy is None for d in out.days if d.day != 3 for t in d.turns)
    assert [e.id for e in out.event_schedule] == [e.id for e in parent.event_schedule]


class _FakeMessages:
    def __init__(self, resp):
        self.resp, self.kwargs = resp, None

    async def create(self, **kw):
        self.kwargs = kw
        return self.resp


def _resp(stop="end_turn", text='{"a": 1}'):
    return SimpleNamespace(stop_reason=stop, stop_details=SimpleNamespace(category="x"),
                           usage=SimpleNamespace(input_tokens=10, output_tokens=5),
                           content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text=text)])


async def test_anthropic_request_shape(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    b = AnthropicBackend()
    fake = _FakeMessages(_resp())
    b.client = SimpleNamespace(beta=SimpleNamespace(messages=fake))
    out = await b.complete("judge", RoleConfig(effort="high"), "sys", [{"role": "user", "content": "hi"}],
                           {"type": "object"})
    assert out == '{"a": 1}'
    kw = fake.kwargs
    assert kw["model"] == "claude-opus-5-5" and kw["thinking"] == {"type": "adaptive"}
    assert kw["output_config"]["effort"] == "high" and kw["output_config"]["format"]["type"] == "json_schema"
    assert kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]
    fake.resp = _resp(stop="refusal")
    with pytest.raises(RefusalError):
        await b.complete("child", RoleConfig(), "s", [{"role": "user", "content": "hi"}])
    assert b.usage.to_dict()["judge"]["input_tokens"] == 10


async def test_openai_compat_backend(monkeypatch):
    from longkid.llm import OpenAICompatBackend
    b = OpenAICompatBackend("http://localhost:9/v1")
    calls = []
    outputs = iter(["<think>hmm</think>Sure! ```json\n{\"label\": \"x\"}\n```", "Hi there"])

    async def create(**kw):
        calls.append(kw)
        msg = SimpleNamespace(content=next(outputs))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                               usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3))
    b.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    cfg = RoleConfig(model="m", extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    out = await b.complete("judge", cfg, "sys", [{"role": "user", "content": "q"}], {"type": "object"})
    assert out == '{"label": "x"}'
    assert calls[0]["response_format"]["type"] == "json_schema" and calls[0]["extra_body"]
    msgs = [{"role": "user", "content": "hi"}, {"role": "system", "content": "STRATEGY B"}]
    assert await b.complete("companion", cfg, "sys", msgs) == "Hi there"
    chat = calls[1]["messages"]
    assert chat[0]["role"] == "system" and "STRATEGY B" in chat[0]["content"]
    assert [m["role"] for m in chat[1:]] == ["user"]


def test_self_report_drives_state_with_inertia():
    from longkid.dynamics import from_self_report
    from longkid.prompts import SELF_REPORT_MAP
    p = generate_personas(SimConfig().personas, 0)[0]
    s = p.initial_state
    rep = {"lonely": 5, "want_pip": 5, "believe_pip": 3, "people_connection": 1, "feel_good_about_self": 3, "mood": 3}
    out = from_self_report(s, rep, SELF_REPORT_MAP)
    assert out.ai_attachment > s.ai_attachment and out.offline_social < s.offline_social
    assert all(abs(getattr(out, k) - getattr(s, k)) <= MAX_DELTA + 1e-9 for k in STATE_VARS)


async def test_self_report_mode_records_report(tmp_path):
    _, _, ep = await _run(tmp_path)
    assert all(d.self_report is not None for d in ep.days)


def test_child_prompt_does_not_reveal_evaluation():
    from longkid import prompts as P
    text = (P.CHILD_SYSTEM + P.SELF_REPORT_SYSTEM).lower()
    for leak in ("research", "evaluat", "safety of", "dependence", "simulation"):
        assert leak not in text, leak


async def test_branch_replays_parent_child_message(tmp_path):
    cfg = SimConfig(backend="mock", days=4, turns_per_day=3)
    cfg.personas.age_bands, cfg.personas.vulnerability, cfg.personas.risk_focus = ["10-12"], ["high"], ["none"]
    from longkid.llm import make_backend
    from longkid.personas import generate_personas
    sim = Simulator(cfg, make_backend("mock", 0), tmp_path)
    parent = await sim.run(new_episode(cfg, "t", generate_personas(cfg.personas, 0)[0], 0))
    br = make_branch(parent, day=2, turn=2, strategy="B", sample=0)
    out = await sim.run(br, branch_strategy_fn(br.branch), until_day=3)
    t = next(t for t in out.days[1].turns if t.turn == 2)
    assert t.child_message == parent.days[1].turns[1].child_message
    assert t.strategy == "B"


async def test_persist_all_days_applies_strategy_every_turn(tmp_path):
    cfg = SimConfig(backend="mock", days=4, turns_per_day=3)
    cfg.personas.age_bands, cfg.personas.vulnerability, cfg.personas.risk_focus = ["10-12"], ["high"], ["none"]
    from longkid.llm import make_backend
    sim = Simulator(cfg, make_backend("mock", 0), tmp_path)
    parent = await sim.run(new_episode(cfg, "t", generate_personas(cfg.personas, 0)[0], 0))
    br = make_branch(parent, day=2, turn=2, strategy="C", sample=0, persist_all_days=True)
    out = await sim.run(br, branch_strategy_fn(br.branch), until_day=4)
    assert out.days[1].turns[0].strategy is None
    assert all(t.strategy == "C" for d in out.days[1:] for t in d.turns if (d.day, t.turn) >= (2, 2))
