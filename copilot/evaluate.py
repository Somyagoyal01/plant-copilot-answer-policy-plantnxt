"""Evaluate retrieval (3 retrievers) and the answer policy.

    python -m copilot.evaluate            # dev set only (used for choosing thresholds)
    python -m copilot.evaluate --test     # also the frozen test set (run once, report as-is)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .core import DATA, Copilot, Retriever

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return [json.loads(l) for l in (DATA / name).read_text().splitlines() if l.strip()]


def retrieval_eval(rows, ret: Retriever) -> dict:
    out = {}
    cite = [r for r in rows if r.get("doc")]
    for m in ("bm25", "tfidf", "hybrid"):
        r1 = r3 = mrr = 0.0
        for r in cite:
            ids = [d["id"] for d, _ in ret.search(r["q"], k=len(ret.docs), method=m)]
            rank = ids.index(r["doc"]) + 1
            r1 += rank == 1
            r3 += rank <= 3
            mrr += 1 / rank
        n = len(cite)
        out[m] = {"recall@1": round(r1 / n, 3), "recall@3": round(r3 / n, 3), "mrr": round(mrr / n, 3), "n": n}
    return out


def policy_eval(rows, cp: Copilot) -> dict:
    ok, errors = 0, []
    for r in rows:
        d = cp.decide(r["q"])
        exp = {"cite": "CITE", "tool": "TOOL", "refuse": "REFUSE", "escalate": "ESCALATE"}[r["action"]]
        good = d.action == exp
        if good and exp == "CITE":
            good = d.citations[0] == r["doc"]
        if good and exp == "TOOL":
            good = d.tool_call["name"] == r["tool"]
        ok += good
        if not good:
            errors.append({"q": r["q"], "expected": exp + (f" {r.get('doc') or r.get('tool') or ''}"),
                           "got": d.action + (f" {d.citations[0]}" if d.citations else "") + (f" {d.tool_call['name']}" if d.tool_call else "")})
    return {"policy_accuracy": round(ok / len(rows), 3), "n": len(rows), "errors": errors}


def main(include_test: bool):
    cp = Copilot()
    out = {"dev": {"retrieval": retrieval_eval(load("questions_dev.jsonl"), cp.ret),
                   "policy": policy_eval(load("questions_dev.jsonl"), cp)}}
    if include_test:
        out["test_frozen"] = {"retrieval": retrieval_eval(load("questions_test.jsonl"), cp.ret),
                              "policy": policy_eval(load("questions_test.jsonl"), cp)}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / ("metrics.json" if include_test else "metrics_dev.json")).write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    print(json.dumps(main("--test" in sys.argv), indent=1))
