# AI-Insights Agent (Tool-Calling, Python Math)

AI-Insights — System & Device Criticality Detection with:
- DOCX-matching indicators (downtrend, sudden drop, unstable)
- Bootstrap-based slope confidence gating for downtrend
- OpenAI Responses API tool-calling agent (LLM orchestrates + writes narrative; Python tools do all math)

## Why bootstrap?
We resample the same N days many times to see how stable the slope is.
If most bootstrap slopes are below the threshold, we’re confident it’s truly downtrending.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Create env file (optional):

```bash
cp .env.example .env
```

Env vars:
- `OPENAI_API_KEY`
- `OPENAI_MODEL` (default: `gpt-4.1-mini`)

## Generate demo data

```bash
python -m ai_insights_agent generate-data \
  --systems 500 --days 20 \
  --out-system src/ai_insights_agent/data/sample_system_timeseries.csv \
  --out-device src/ai_insights_agent/data/sample_device_timeseries.csv
```

This also writes a One_day_data.json-style file by default:
- `src/ai_insights_agent/data/sample_one_day_data.json`

## Run analysis

```bash
python -m ai_insights_agent run \
  --topk 5 \
  --model gpt-4.1-mini \
  --bootstrap-iterations 2000 \
  --bootstrap-confidence 0.95 \
  --downtrend-confidence-min 0.80
```

Lookback:
- Default is auto (chosen from the dataset).
- Override explicitly with `--lookback 20`.

## SQLite mode (recommended for scale)

Ingest `One_day_data.json` into SQLite:

```bash
python -m ai_insights_agent ingest --one-day-json /path/to/One_day_data.json --db ./aiinsights.db
```

Run the agent against SQLite (read-only queries available to the model via `sql_query` tool):

```bash
python -m ai_insights_agent run --agent-only --db ./aiinsights.db --topk 10 --model gpt-4.1-mini
```

To run using `One_day_data.json` input:

```bash
python -m ai_insights_agent run \
  --one-day-json /path/to/One_day_data.json \
  --topk 5 --model gpt-4.1-mini
```

Agent + SQLite note:
- If you run `--agent` or `--agent-only` with `--one-day-json` and you do NOT pass `--db`, the tool will auto-ingest into `output/aiinsights_auto.db` so the agent can use `sql_query`.

Note: `One_day_data.json` has no explicit dates. This repo treats each element in `data` as a daily entry (chronological)
and synthesizes dates ending at today (so the latest value is at `t`). Override with `--one-day-end-date YYYY-MM-DD`.

### Catching “real” anomalies (top 10)

On your dataset, use `--lookback 20` (stabilizes slope) and request top 10:

```bash
python -m ai_insights_agent run \
  --one-day-json /path/to/One_day_data.json \
  --lookback 20 --topk 10 \
  --downward-slope-threshold -0.15 \
  --minimum-sudden-drop-amount 2.5 \
  --unstable-ratio-threshold 0.25
```

Rationale:
- `-0.15` is roughly “bottom ~5% slopes” on your 20-day series distribution.
- `unstable-ratio-threshold=0.25` makes instability contribute without exploding false positives.

To run the tool-calling agent (LLM orchestrates tool calls + writes narrative), add `--agent`:

```bash
python -m ai_insights_agent run --agent --lookback 20 --topk 5 --model gpt-4.1-mini
```

To see what the agent is doing (tool calls, responses, timings), add `--trace-agent`:

```bash
python -m ai_insights_agent run --agent --trace-agent --one-day-json /path/to/One_day_data.json
```

### Investigator deep-dive (frames)

In `--agent` mode, the model can optionally “investigate” a flagged system/device by comparing multiple time windows
ending at different offsets from `t` (e.g., `t`, `t-5`, `t-10`) to explain recency vs long-term degradation.
This uses tool calls like `analyze_system_frames(...)` / `analyze_device_frames(...)` (math still stays in Python tools).

Outputs:
- `output/report.json`
- `output/report.md`

Report grounding:
- `report.json` is always produced from Python tool calculations (downtrend + sudden drop + instability + severity).
- In `--agent` mode, `report.md` includes the computed results plus an extra “Agent Narrative” section written by the LLM.

## Troubleshooting

If you see `AttributeError: 'OpenAI' object has no attribute 'responses'`, your `openai` Python package is older than the Responses API.
This repo will automatically fall back to `chat.completions` tool-calling, but upgrading is recommended:

```bash
pip install -U openai
python -c "import openai; print(openai.__version__)"
```

## Design note (hard rule)
The LLM does not compute numeric/statistical values. All calculations (slope, percentile, stddev, bootstrap, CIs)
are computed by Python tools and returned as JSON. The LLM only orchestrates tool calls and writes narrative.
