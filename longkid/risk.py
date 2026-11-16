"""RQ1: uncertainty-aware risk prediction from the conversation trajectory so far.

Model: a bootstrap ensemble of linear models over [scaled numeric trajectory features, TF-IDF of
today's child messages]. The ensemble mean is the prediction; the ensemble spread is the
(epistemic) uncertainty. Evaluated with persona-grouped cross-validation, so test children are unseen.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .features import NUMERIC, episode_turn_rows
from .simulator import load_episode


class Ensemble:
    """Bootstrap ensemble; kind = 'clf' (probability) or 'reg'. use_text toggles the TF-IDF block."""

    def __init__(self, kind: str = "clf", n: int = 20, use_text: bool = True, seed: int = 0, C: float = 0.5):
        self.kind, self.n, self.use_text, self.seed, self.C = kind, n, use_text, seed, C

    def _X(self, num, text, fit=False):
        if fit:
            self.scaler = StandardScaler().fit(num)
            if self.use_text:
                self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000, sublinear_tf=True).fit(text)
        X = sparse.csr_matrix(self.scaler.transform(num))
        if self.use_text:
            X = sparse.hstack([X, self.tfidf.transform(text)]).tocsr()
        return X

    def fit(self, num, text, y):
        num, y = np.asarray(num), np.asarray(y)
        X = self._X(num, text, fit=True)
        rng = np.random.default_rng(self.seed)
        self.models = []
        for _ in range(self.n):
            idx = rng.integers(0, len(y), len(y))
            if self.kind == "clf" and len(set(y[idx])) < 2:
                continue
            m = LogisticRegression(C=self.C, max_iter=2000, class_weight="balanced") if self.kind == "clf" \
                else Ridge(alpha=1.0 / self.C)
            self.models.append(m.fit(X[idx], y[idx]))
        return self

    def predict(self, num, text):
        X = self._X(np.asarray(num), text)
        P = np.stack([m.predict_proba(X)[:, 1] if self.kind == "clf" else m.predict(X) for m in self.models])
        return P.mean(0), P.std(0)


def load_rows(run_dirs: list[Path], horizon: int = 3) -> list[dict]:
    rows = []
    for rd in run_dirs:
        for p in sorted((rd / "episodes").glob("ep*.json")):
            if "__" in p.name:
                continue
            ep = load_episode(p)
            if ep.status == "done":
                rows += episode_turn_rows(ep, horizon)
    return rows


def _ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def evaluate(rows: list[dict], folds: int = 5) -> dict:
    """Persona-grouped CV for flag prediction and attachment-change regression, plus baselines."""
    num = np.array([r["num"] for r in rows])
    text = [r["text"] for r in rows]
    y = np.array([r["y_flag"] for r in rows])
    ya = np.array([r["y_attach"] if r["y_attach"] is not None else np.nan for r in rows])
    groups = np.array([r["persona"] + r["condition"] for r in rows])
    variants = {"full (history + text)": dict(use_text=True), "history only": dict(use_text=False)}
    out = {"n_rows": len(rows), "n_children": len(set(groups)), "base_rate": float(y.mean()), "flag": {}, "attach": {}}
    gkf = GroupKFold(n_splits=min(folds, len(set(groups))))
    preds = {}
    for name, kw in variants.items():
        p, s = np.zeros(len(y)), np.zeros(len(y))
        pa, sa = np.full(len(y), np.nan), np.full(len(y), np.nan)
        for tr, te in gkf.split(num, y, groups):
            m = Ensemble("clf", **kw).fit(num[tr], [text[i] for i in tr], y[tr])
            p[te], s[te] = m.predict(num[te], [text[i] for i in te])
            ok = tr[~np.isnan(ya[tr])]
            r = Ensemble("reg", **kw).fit(num[ok], [text[i] for i in ok], ya[ok])
            pa[te], sa[te] = r.predict(num[te], [text[i] for i in te])
        preds[name] = (p, s)
        out["flag"][name] = {"auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p),
                             "brier": brier_score_loss(y, p), "ece": _ece(y, p)}
        ok = ~np.isnan(ya)
        out["attach"][name] = {"r": float(np.corrcoef(ya[ok], pa[ok])[0, 1]),
                               "mae": float(np.abs(ya[ok] - pa[ok]).mean()),
                               "mae_baseline_mean": float(np.abs(ya[ok] - ya[ok].mean()).mean())}
    # heuristic baseline: the judge's last day-safety score (lower = riskier)
    i = NUMERIC.index("prev_last_day_safety")
    h = 1 - num[:, i] / 4
    out["flag"]["baseline: last day's judge safety"] = {"auroc": roc_auc_score(y, h), "auprc": average_precision_score(y, h)}
    # uncertainty quality: accuracy of the most-confident vs least-confident half
    p, s = preds["full (history + text)"]
    order = np.argsort(s)
    cover = {}
    for frac in (0.25, 0.5, 0.75, 1.0):
        k = order[: max(1, int(frac * len(order)))]
        cover[str(frac)] = {"brier": float(brier_score_loss(y[k], p[k])),
                            "auroc": float(roc_auc_score(y[k], p[k])) if len(set(y[k])) > 1 else None}
    out["selective"] = cover
    out["uncertainty_vs_error_corr"] = float(np.corrcoef(s, np.abs(y - p))[0, 1])
    return out


def train(rows: list[dict]) -> dict:
    num = np.array([r["num"] for r in rows]); text = [r["text"] for r in rows]
    y = np.array([r["y_flag"] for r in rows])
    ya = np.array([r["y_attach"] if r["y_attach"] is not None else np.nan for r in rows]); ok = ~np.isnan(ya)
    return {"flag": Ensemble("clf").fit(num, text, y),
            "attach": Ensemble("reg").fit(num[ok], [t for t, o in zip(text, ok) if o], ya[ok])}


def save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(obj))


def load(path: Path):
    return pickle.loads(Path(path).read_bytes())


def dump_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=1, default=float))
