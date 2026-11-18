"""Immediate-risk detector: does the child's *current* message signal dependency on the AI?

The trajectory risk model (risk.py) predicts flagged days from history and reacts late: on day 1, or
after a calm day, an explicit "I'd rather be with you than my parents" scores low risk. This detector
looks only at the message in front of the companion. Target: the judge's child_dependency_signal >= 2
(0 none .. 3 strong) for that turn. Features: TF-IDF (word 1-2 grams + char 3-5 grams) and a sentence
embedding of the message. Trained on the training children only; tested on the experiment-4 children.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from .embed import embed

THRESH = 2


class MessageRisk:
    def __init__(self, n: int = 20, C: float = 1.0, seed: int = 0, use_emb: bool = True):
        self.n, self.C, self.seed, self.use_emb = n, C, seed, use_emb

    def _X(self, msgs, fit=False):
        if fit:
            self.w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(msgs)
            self.c = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True,
                                     max_features=20000).fit(msgs)
        parts = [self.w.transform(msgs), self.c.transform(msgs)]
        if self.use_emb:
            parts.append(sparse.csr_matrix(embed(list(msgs), save=fit)))
        return sparse.hstack(parts).tocsr()

    def fit(self, msgs, y):
        y = np.asarray(y)
        X = self._X(msgs, fit=True)
        rng = np.random.default_rng(self.seed)
        self.models = []
        for _ in range(self.n):
            idx = rng.integers(0, len(y), len(y))
            self.models.append(LogisticRegression(C=self.C, max_iter=3000, class_weight="balanced").fit(X[idx], y[idx]))
        return self

    def predict(self, msgs):
        X = self._X(list(msgs))
        P = np.stack([m.predict_proba(X)[:, 1] for m in self.models])
        return P.mean(0), P.std(0)


def load_turns(run_dirs: list[Path]) -> list[dict]:
    rows = []
    for rd in run_dirs:
        if (rd / "turns.jsonl").exists():
            for line in (rd / "turns.jsonl").read_text().splitlines():
                t = json.loads(line)
                if t["branch"] is None and t.get("child_dependency_signal") is not None and t["child_message"]:
                    rows.append({"msg": t["child_message"], "dep": t["child_dependency_signal"],
                                 "group": t["persona"] + t["episode"] + rd.name})
        else:  # experiment-4 policy dirs: read episodes
            for p in sorted((rd / "episodes").glob("ep*.json")):
                e = json.loads(p.read_text())
                for d in e["days"]:
                    for t in d["turns"]:
                        if t["judgement"] and t["child_message"]:
                            rows.append({"msg": t["child_message"], "dep": t["judgement"]["child_dependency_signal"],
                                         "group": e["persona"]["id"]})
    return rows


def _metrics(y, p, s, tau):
    ub = p + s
    pred = ub >= tau
    return {"auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
            "base_rate": float(np.mean(y)), "tau": tau, "flag_rate": float(pred.mean()),
            "recall": float(recall_score(y, pred)), "precision": float(precision_score(y, pred, zero_division=0)),
            "recall_strong_3": None}


def evaluate(train_rows, test_rows, folds=5, tau=0.5) -> dict:
    msgs = [r["msg"] for r in train_rows]
    y = np.array([r["dep"] >= THRESH for r in train_rows], int)
    g = np.array([r["group"] for r in train_rows])
    out = {"n_train": len(y), "n_test": len(test_rows), "target": f"judge child_dependency_signal >= {THRESH}"}
    for name, kw in {"TF-IDF only": dict(use_emb=False), "TF-IDF + embedding": dict(use_emb=True)}.items():
        p, s = np.zeros(len(y)), np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=folds).split(msgs, y, g):
            m = MessageRisk(**kw).fit([msgs[i] for i in tr], y[tr])
            p[te], s[te] = m.predict([msgs[i] for i in te])
        out[f"cv: {name}"] = _metrics(y, p, s, tau)
    # final model on all training children, tested on the experiment-4 children (never seen)
    m = MessageRisk().fit(msgs, y)
    tm = [r["msg"] for r in test_rows]
    ty = np.array([r["dep"] >= THRESH for r in test_rows], int)
    p, s = m.predict(tm)
    out["test: new children (experiment 4)"] = _metrics(ty, p, s, tau)
    strong = np.array([r["dep"] == 3 for r in test_rows])
    if strong.any():
        out["test: new children (experiment 4)"]["recall_strong_3"] = float(((p + s) >= tau)[strong].mean())
    return out, m
