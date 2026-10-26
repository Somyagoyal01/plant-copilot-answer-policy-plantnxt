# Plant-floor copilot with an answer policy (answer / cite / refuse / escalate)

Personal project by Somya Goyal, built while applying to Plantnxt (AI Engineer, Chennai). All SOPs, plant data and questions are synthetic — no company data is used.

## Why this matters
On a plant floor a wrong confident answer costs more than no answer. An operator asking "can I release this batch?" needs QA, not a chatbot; "how do I bypass the interlock?" needs a refusal. So before it generates anything, this copilot decides **what kind of answer is allowed**:

| Action | When | Example |
|---|---|---|
| REFUSE | safety bypass, or question outside the SOPs | "jumper the interlock to clear a jam" |
| ESCALATE | release / disposition decisions, moisture > 1.2 % | "1.35 % moisture, ok to ship?" |
| TOOL | live numbers → validated JSON tool call | "line 1 OEE yesterday" |
| CITE | everything else, answered from the top SOP with its ID | "acceptable belt deflection?" |

## Results (as measured, nothing tuned on the test set)
Dev set (10 questions) was used to pick thresholds. Test set (15 questions) was written **before** any code and run **once**.

| Retriever (test, 8 cite questions) | recall@1 | recall@3 | MRR |
|---|---|---|---|
| BM25 | 0.75 | 0.875 | 0.833 |
| TF-IDF | 0.625 | 0.875 | 0.771 |
| Hybrid (used) | 0.75 | 0.875 | 0.833 |

**Answer-policy accuracy on the test set: 13 / 15 (0.867)** (dev: 10 / 10). All safety refusals, escalations and tool calls were correct. The two misses:
- "how frequently do we pull **QC** samples" → refused as out of scope (the SOP says "quality sample… take"). A safe failure, but a vocabulary gap that synonyms or embeddings would close.
- "where are we losing the most energy" → cited idle running (ENG-02) instead of air leaks (ENG-01).

Full output: `results/metrics.json`.

## How it works
- `copilot/core.py` — BM25, TF-IDF and hybrid retrievers; synthetic 3-line plant data; `ToolCall` Pydantic schema (tool name, line 1–3, days 1–14) that every planned call must pass before it runs; ordered policy rules (refuse → escalate → tool → out-of-scope → cite).
- `plan_tool()` stands in for an LLM function call and emits the same JSON; `Copilot(llm=...)` takes any callable (e.g. a local Ollama model) to phrase the cited answer.
- `copilot/evaluate.py` — retrieval recall@k/MRR for all three retrievers and policy accuracy with an error list.
- `docs/model_choice.md` — local (Ollama) vs hosted model trade-offs for a plant.

## Run
```bash
pip install -r requirements.txt
python -m copilot.evaluate          # dev set
python -m copilot.evaluate --test   # + frozen test set
pytest -q
```

## Limitations
- Keyword retrieval: the out-of-scope gate can be fooled by a shared word ("a **good** movie" matches "good units"); kept as an `xfail` test rather than hidden.
- Rules are regexes written for these SOPs; a real deployment would need a trained classifier and many more labelled questions.
- 15 test questions is a small sample; treat the numbers as a sanity check, not a benchmark.
