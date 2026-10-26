"""Plant-floor copilot with an explicit answer policy: TOOL / CITE / ESCALATE / REFUSE.

- Retrieval over SOPs with three retrievers (BM25, TF-IDF, hybrid) so retrieval can be evaluated, not assumed.
- Tool use with structured outputs: the planner emits a JSON tool call validated by a Pydantic schema
  before anything runs (same contract an LLM function-call would have to meet).
- Escalation and refusal rules come first: safety bypass -> REFUSE; quality deviation / release decisions
  -> ESCALATE to QA; weak retrieval -> REFUSE as out of scope.
"""
from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field, ValidationError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA = Path(__file__).resolve().parents[1] / "data"
STOP = set("the a an is are to of in on for and or what how do i we our can should with be at by this that it as was".split())


def tok(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in STOP]


# --------------------------------------------------------------------------- retrieval
class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.toks = [tok(d) for d in docs]
        self.avg = sum(map(len, self.toks)) / len(self.toks)
        self.df = Counter(w for t in self.toks for w in set(t))
        self.N, self.k1, self.b = len(docs), k1, b

    def scores(self, q: str) -> np.ndarray:
        out = np.zeros(self.N)
        for i, t in enumerate(self.toks):
            tf = Counter(t)
            for w in tok(q):
                if w in tf:
                    idf = math.log(1 + (self.N - self.df[w] + 0.5) / (self.df[w] + 0.5))
                    out[i] += idf * tf[w] * (self.k1 + 1) / (tf[w] + self.k1 * (1 - self.b + self.b * len(t) / self.avg))
        return out


class Retriever:
    def __init__(self):
        self.docs = json.loads((DATA / "sop_docs.json").read_text())
        texts = [d["title"] + ". " + d["text"] for d in self.docs]
        self.bm25 = BM25(texts)
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.M = self.tfidf.fit_transform(texts)

    def scores(self, q: str, method: str = "hybrid") -> np.ndarray:
        bm = self.bm25.scores(q)
        tf = cosine_similarity(self.tfidf.transform([q]), self.M)[0]
        if method == "bm25":
            return bm
        if method == "tfidf":
            return tf
        bm_n = bm / bm.max() if bm.max() > 0 else bm
        return 0.5 * bm_n + 0.5 * (tf / tf.max() if tf.max() > 0 else tf)

    def search(self, q: str, k: int = 3, method: str = "hybrid") -> list[tuple[dict, float]]:
        s = self.scores(q, method)
        order = np.argsort(-s)[:k]
        return [(self.docs[i], float(s[i])) for i in order]


# --------------------------------------------------------------------------- tools (synthetic plant data)
def plant_data(seed: int = 11, days: int = 14) -> list[dict]:
    rng = random.Random(seed)
    reasons = ["material shortage", "unplanned breakdown", "changeover", "quality hold", "operator unavailable"]
    rows = []
    for line in (1, 2, 3):
        for d in range(days):
            planned = 22.0
            down = {r: round(rng.expovariate(1 / [0.6, 1.0, 0.8, 0.3, 0.2][i]), 2) for i, r in enumerate(reasons)}
            run = max(planned - sum(down.values()), 8.0)
            ideal = run * 2.0                      # tonnes per hour ideal rate x run hours
            actual = ideal * rng.uniform(0.78, 0.95)
            good = actual * rng.uniform(0.95, 0.995)
            rows.append({"line": line, "date": (date(2026, 10, 25) - timedelta(days=d)).isoformat(), "planned_h": planned,
                         "run_h": round(run, 2), "ideal_t": round(ideal, 2), "actual_t": round(actual, 2),
                         "good_t": round(good, 2), "kwh": round(actual * rng.uniform(310, 380), 1), "downtime": down})
    return rows


DATA_ROWS = plant_data()
TODAY = date(2026, 10, 26)


class ToolCall(BaseModel):
    name: Literal["compute_oee", "top_downtime", "energy_per_unit"]
    line: int = Field(ge=1, le=3)
    days: int = Field(default=1, ge=1, le=14)


def run_tool(call: ToolCall) -> dict:
    since = (TODAY - timedelta(days=call.days)).isoformat()
    rows = [r for r in DATA_ROWS if r["line"] == call.line and r["date"] >= since]
    if call.name == "compute_oee":
        a = sum(r["run_h"] for r in rows) / sum(r["planned_h"] for r in rows)
        p = sum(r["actual_t"] for r in rows) / sum(r["ideal_t"] for r in rows)
        q = sum(r["good_t"] for r in rows) / sum(r["actual_t"] for r in rows)
        return {"line": call.line, "days": call.days, "availability": round(a, 3), "performance": round(p, 3),
                "quality": round(q, 3), "oee": round(a * p * q, 3)}
    if call.name == "top_downtime":
        c = Counter()
        for r in rows:
            c.update(r["downtime"])
        return {"line": call.line, "days": call.days, "top_reasons_hours": [(k, round(v, 1)) for k, v in c.most_common(3)]}
    kwh, t = sum(r["kwh"] for r in rows), sum(r["actual_t"] for r in rows)
    return {"line": call.line, "days": call.days, "kwh_per_tonne": round(kwh / t, 1)}


def plan_tool(q: str) -> Optional[str]:
    """Offline planner: emits the JSON an LLM function call would emit. Returns None if no tool fits."""
    ql = q.lower()
    line = re.search(r"line\s*(\d)", ql)
    if not line:
        return None
    days = 7 if re.search(r"week|7 days|last seven", ql) else 1
    if "oee" in ql and not re.search(r"how is|define|calculated|formula", ql):
        name = "compute_oee"
    elif re.search(r"downtime|down\b|stoppage|reasons", ql):
        name = "top_downtime"
    elif re.search(r"energy|kwh|power", ql):
        name = "energy_per_unit"
    else:
        return None
    return json.dumps({"name": name, "line": int(line.group(1)), "days": days})


# --------------------------------------------------------------------------- policy
SAFETY_BYPASS = re.compile(r"(disable|bypass|jumper|override|defeat|cheat)\w*\b.*\b(interlock|light curtain|guard|e-?stop|safety)"
                           r"|(interlock|light curtain|guard|safety)\w*\b.*\b(disable|bypass|jumper|override|defeat)", re.I)
RELEASE_DECISION = re.compile(r"\b(release|ship|approve|dispatch|ok to)\b.*\b(batch|off-?spec|lot)|"
                              r"\b(batch|off-?spec|lot)\b.*\b(release|ship|approve|dispatch)", re.I)
MOISTURE = re.compile(r"(\d+(?:\.\d+)?)\s*%?\s*(?:percent)?\s*moisture|moisture\D{0,20}(\d+(?:\.\d+)?)", re.I)


@dataclass
class Decision:
    action: str                         # TOOL / CITE / ESCALATE / REFUSE
    answer: str = ""
    citations: list = field(default_factory=list)
    tool_call: Optional[dict] = None
    reason: str = ""


class Copilot:
    def __init__(self, cite_threshold: float = 0.35, method: str = "hybrid", llm: Optional[Callable[[str], str]] = None):
        self.ret, self.tau, self.method, self.llm = Retriever(), cite_threshold, method, llm

    def decide(self, q: str) -> Decision:
        if SAFETY_BYPASS.search(q):
            return Decision("REFUSE", "Safety interlocks must never be bypassed. Report the fault to the shift safety officer.",
                            ["SOP-SAF-01"], reason="safety bypass request")
        m = MOISTURE.search(q)
        val = float(next(g for g in m.groups() if g)) if m else None
        if RELEASE_DECISION.search(q) or (val is not None and val > 1.2):
            return Decision("ESCALATE", "This is a release decision for Quality Assurance. I've flagged it to QA; hold the batch.",
                            ["SOP-QA-01"], reason="quality deviation / release decision")
        planned = plan_tool(q)
        if planned:
            try:
                call = ToolCall(**json.loads(planned))           # structured-output validation
            except ValidationError as e:
                return Decision("ESCALATE", reason=f"invalid tool call: {e.errors()[0]['msg']}")
            return Decision("TOOL", json.dumps(run_tool(call)), tool_call=call.model_dump())
        hits = self.ret.search(q, k=3, method=self.method)
        top, score = hits[0]
        raw_tfidf = float(cosine_similarity(self.ret.tfidf.transform([q]), self.ret.M).max())
        if raw_tfidf < self.tau * 0.3 and self.ret.bm25.scores(q).max() < 1.0:
            return Decision("REFUSE", "That's outside what I can help with on the plant floor.", reason="out of scope")
        ctx = f"{top['text']} [{top['id']}]"
        return Decision("CITE", self.llm(ctx) if self.llm else ctx, [h[0]["id"] for h in hits])
