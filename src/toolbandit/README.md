# toolbandit

A clean, readable rewrite of the ToolBandit experiment runtime. Replaces the two
large files in `src/training/` with small single-purpose modules.

## The idea in one paragraph

Can a system learn which tools actually work by watching them succeed or fail,
instead of trusting their descriptions? The marketplace (`data/pool/listings.jsonl`)
has good tools, reworded-but-working copies, and secretly-broken copies that have
good-looking descriptions. Plain text search can't tell them apart. The **bandit**
learns to, from feedback.

## The loop (this is the whole experiment)

For each task: **retrieve** candidates by text → **rerank** with the bandit →
**pick** one → **execute** it (faking failures for broken tools) → score the
**reward** → **update** the bandit. See `loop.py`.

## Files, in reading order

| File | What it is |
|------|-----------|
| `loop.py` | The training loop. Start here. |
| `bandit.py` | The learner: neural success model + exploration bonus + online update. The core. |
| `retriever.py` | Stage 1: TF-IDF text search to get candidates. |
| `marketplace.py` | Stage 4: run a chosen tool; inject synthetic failures for broken ones. |
| `reward.py` | Stage 5: score the result (cheap metadata, or an LLM judge). |
| `caller.py` | Stage 3: pick a tool. `policy_pick` (no LLM) and `LLMPicker` (real model). Also the anti-cheating sanitizer. |
| `data.py` | Load the data; tiny shared helpers; the list of caller models. |
| `run.py` | Command-line entry point. |

## Run it

```bash
# Free, no API keys — good for reading and learning:
PYTHONPATH=src python -m toolbandit.run sim

# Real models choose tools (needs OPENAI_API_KEY / OPENROUTER_API_KEY):
PYTHONPATH=src python -m toolbandit.run live --reward metadata
PYTHONPATH=src python -m toolbandit.run live --reward judge
```

Outputs go to `data/runs/toolbandit/`: `events.jsonl` (per round),
`learning_curve.csv`, `summary.json`.

## Notes

- The bandit's inputs are only `[query, model, tool]`. Retrieval similarity is used
  only to pick candidates, not as a ranking feature (matches `Toolbandit.pdf`).
- `exploration_weight` (in `bandit.py`) trades off trying new tools vs. exploiting
  known-good ones. It is high by default, which makes the cold start noisy over a
  single pass — lower it for steadier early behavior.
- This package reuses the small, already-clean helpers in `src/tool_pool/`
  (adapters, faults, models). The pool itself was built by the `tool_pool` code and
  is read from `data/pool/`.
