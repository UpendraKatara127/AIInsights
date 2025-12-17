from __future__ import annotations
try:
    import truststore as _truststore

    _truststore.inject_into_ssl()
except Exception:
    pass
import json
import os
import sys
import time
import inspect
from collections import defaultdict, deque
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .config import Config
from .tools.fetch import DataStore
from .tools.metrics import bootstrap_slope, flag_series, linear_regression_slope
from .tools.rank import rank_devices, rank_systems
from .tools.sqlite_store import sql_query_readonly
from .tools.profile import recommend_lookback


def _load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class ToolRegistry:
    def __init__(self, tools: Dict[str, Callable[..., Any]]):
        self.tools = tools

    def call(self, name: str, arguments: Dict[str, Any]) -> Any:
        if name not in self.tools:
            raise KeyError(f"Unknown tool: {name}")
        return self.tools[name](**arguments)


class AIInsightsAgent:
    def __init__(
        self,
        *,
        datastore: DataStore,
        db_path: Path | None = None,
        config: Config,
        model: str,
        prompt_path: Path,
        trace: bool = False,
        max_log_chars: int = 2000,
    ) -> None:
        self.datastore = datastore
        self.db_path = db_path
        self.config = config
        self.model = model
        self.system_prompt = _load_prompt(prompt_path)
        self.trace = trace
        self.max_log_chars = max_log_chars
        self._runtime_thresholds: Dict[str, Any] = {}
        self._pending_system_fetches = deque()
        self._pending_device_fetches = deque()
        self._system_results: List[Dict[str, Any]] = []
        self._device_results_by_system: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._last_devices_system_id: str | None = None
        self._last_top_systems: List[Dict[str, Any]] = []
        self._last_top_devices_by_system: Dict[str, List[Dict[str, Any]]] = {}
        self._system_frame_analyses: Dict[str, Dict[str, Any]] = {}
        self._device_frame_analyses: Dict[tuple[str, str], Dict[str, Any]] = {}
        self._population_baselines: List[Dict[str, Any]] = []
        self._system_contexts: Dict[str, Dict[str, Any]] = {}
        self._executed_tool_calls: List[Dict[str, Any]] = []

        self.registry = ToolRegistry(
            {
                "list_system_ids": self._tool_list_system_ids,
                "fetch_system_timeseries": self._tool_fetch_system_timeseries,
                "fetch_system_timeseries_frame": self._tool_fetch_system_timeseries_frame,
                "analyze_system_frames": self._tool_analyze_system_frames,
                "flag_series": self._tool_flag_series,
                "rank_systems": self._tool_rank_systems,
                "list_devices": self._tool_list_devices,
                "fetch_device_timeseries": self._tool_fetch_device_timeseries,
                "fetch_device_timeseries_frame": self._tool_fetch_device_timeseries_frame,
                "analyze_device_frames": self._tool_analyze_device_frames,
                "rank_devices": self._tool_rank_devices,
                "screen_systems": self._tool_screen_systems,
                "screen_devices_for_system": self._tool_screen_devices_for_system,
                "population_baseline": self._tool_population_baseline,
                "system_context": self._tool_system_context,
                "sql_query": self._tool_sql_query,
                "series_summary": self._tool_series_summary,
                "calc_slope": self._tool_calc_slope,
                "calc_bootstrap_slope": self._tool_calc_bootstrap_slope,
                "recommend_lookback": self._tool_recommend_lookback,
            }
        )

    def _log(self, msg: str) -> None:
        if not self.trace:
            return
        print(msg, file=sys.stderr, flush=True)

    def _clip(self, s: str) -> str:
        if len(s) <= self.max_log_chars:
            return s
        return s[: self.max_log_chars] + f"\n…(truncated, {len(s)} chars total)…"

    def _tools_schema(self) -> List[Dict[str, Any]]:
        series_values_schema: Dict[str, Any] = {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "date": {"type": "string", "description": "YYYY-MM-DD"},
                    "v": {"type": "number"},
                },
                "required": ["date", "v"],
            },
        }
        frames_schema: Dict[str, Any] = {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "lookback_days": {"type": "integer"},
                    "end_offset_days": {"type": "integer", "description": "0 ends at t; 10 ends at t-10"},
                },
                "required": ["lookback_days", "end_offset_days"],
            },
        }
        return [
            {
                "type": "function",
                "name": "list_system_ids",
                "description": "List all known system_ids",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "type": "function",
                "name": "sql_query",
                "description": "Run a read-only SQL query against the SQLite DB. Only SELECT/WITH are allowed. Results are capped.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql": {"type": "string"},
                        "params": {"type": "array", "items": {}, "description": "Positional parameters"},
                        "named_params": {"type": "object", "additionalProperties": True},
                        "max_rows": {"type": "integer"},
                    },
                    "required": ["sql"],
                },
            },
            {
                "type": "function",
                "name": "recommend_lookback",
                "description": "Profile candidate lookback windows and recommend a good default for this dataset and top_k.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "top_k": {"type": "integer"},
                        "thresholds": {"type": "object", "additionalProperties": True},
                        "candidate_days": {"type": "array", "items": {"type": "integer"}},
                    },
                    "required": ["top_k"],
                },
            },
            {
                "type": "function",
                "name": "series_summary",
                "description": "Compute basic summary stats for a series window (n, first/last, delta, pct_change, min/max, mean/std).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "values": series_values_schema,
                    },
                    "required": ["values"],
                },
            },
            {
                "type": "function",
                "name": "calc_slope",
                "description": "Compute linear regression slope over y with x=1..n (per-step slope).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "values": series_values_schema,
                    },
                    "required": ["values"],
                },
            },
            {
                "type": "function",
                "name": "calc_bootstrap_slope",
                "description": "Compute bootstrap slope distribution summary (mean/CI/p_below_threshold/p_negative) for a given threshold.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "values": series_values_schema,
                        "threshold": {"type": "number"},
                        "iterations": {"type": "integer"},
                        "confidence": {"type": "number"},
                        "seed": {"type": "integer"},
                    },
                    "required": ["values", "threshold"],
                },
            },
            {
                "type": "function",
                "name": "screen_systems",
                "description": "Compute flags for ALL systems in Python, then return ONLY the ranked top_k systems (token-efficient).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "lookback_days": {"type": "integer"},
                        "top_k": {"type": "integer"},
                        "thresholds": {"type": "object", "additionalProperties": True},
                    },
                    "required": ["lookback_days", "top_k"],
                },
            },
            {
                "type": "function",
                "name": "population_baseline",
                "description": "Compute small population baselines (quantiles, flag counts) across all systems for context.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "lookback_days": {"type": "integer"},
                        "thresholds": {"type": "object", "additionalProperties": True},
                    },
                    "required": ["lookback_days"],
                },
            },
            {
                "type": "function",
                "name": "fetch_system_timeseries",
                "description": "Fetch last N days of system time series ending at t (latest date)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "system_id": {"type": "string"},
                        "lookback_days": {"type": "integer"},
                    },
                    "required": ["system_id", "lookback_days"],
                },
            },
            {
                "type": "function",
                "name": "fetch_system_timeseries_frame",
                "description": "Fetch a window ending at t-end_offset_days (frame analysis).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "system_id": {"type": "string"},
                        "lookback_days": {"type": "integer"},
                        "end_offset_days": {"type": "integer"},
                    },
                    "required": ["system_id", "lookback_days", "end_offset_days"],
                },
            },
            {
                "type": "function",
                "name": "flag_series",
                "description": "Compute DOCX indicators + bootstrap downtrend confidence; returns JSON flags",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "values": series_values_schema,
                        "thresholds": {"type": "object", "additionalProperties": True},
                    },
                    "required": ["values"],
                },
            },
            {
                "type": "function",
                "name": "analyze_system_frames",
                "description": "Compute flags across multiple frames for a system (investigation).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "system_id": {"type": "string"},
                        "frames": frames_schema,
                        "thresholds": {"type": "object", "additionalProperties": True},
                    },
                    "required": ["system_id", "frames", "thresholds"],
                },
            },
            {
                "type": "function",
                "name": "system_context",
                "description": "Compute a system's flags plus percentile ranks vs the population for key metrics (slope, drop_amount, unstable_ratio, severity).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "system_id": {"type": "string"},
                        "lookback_days": {"type": "integer"},
                        "thresholds": {"type": "object", "additionalProperties": True},
                    },
                    "required": ["system_id", "lookback_days"],
                },
            },
            {
                "type": "function",
                "name": "rank_systems",
                "description": "Rank system results by severity + tie-break rules",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "system_results": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
                        "top_k": {"type": "integer"},
                    },
                    "required": ["top_k"],
                },
            },
            {
                "type": "function",
                "name": "list_devices",
                "description": "List device_ids for a system",
                "parameters": {
                    "type": "object",
                    "properties": {"system_id": {"type": "string"}},
                    "required": ["system_id"],
                },
            },
            {
                "type": "function",
                "name": "screen_devices_for_system",
                "description": "Compute flags for ALL devices in a system in Python, then return ONLY the ranked top_k devices (token-efficient).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "system_id": {"type": "string"},
                        "lookback_days": {"type": "integer"},
                        "top_k": {"type": "integer"},
                        "thresholds": {"type": "object", "additionalProperties": True},
                    },
                    "required": ["system_id", "lookback_days", "top_k"],
                },
            },
            {
                "type": "function",
                "name": "fetch_device_timeseries",
                "description": "Fetch last N days of device time series ending at t (latest date)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "system_id": {"type": "string"},
                        "device_id": {"type": "string"},
                        "lookback_days": {"type": "integer"},
                    },
                    "required": ["system_id", "device_id", "lookback_days"],
                },
            },
            {
                "type": "function",
                "name": "fetch_device_timeseries_frame",
                "description": "Fetch a device window ending at t-end_offset_days (frame analysis).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "system_id": {"type": "string"},
                        "device_id": {"type": "string"},
                        "lookback_days": {"type": "integer"},
                        "end_offset_days": {"type": "integer"},
                    },
                    "required": ["system_id", "device_id", "lookback_days", "end_offset_days"],
                },
            },
            {
                "type": "function",
                "name": "rank_devices",
                "description": "Rank device results by severity + tie-break rules",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "device_results": {"type": "array", "items": {"type": "object", "additionalProperties": True}},
                        "top_k": {"type": "integer"},
                    },
                    "required": ["top_k"],
                },
            },
            {
                "type": "function",
                "name": "analyze_device_frames",
                "description": "Compute flags across multiple frames for a device (investigation).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "system_id": {"type": "string"},
                        "device_id": {"type": "string"},
                        "frames": frames_schema,
                        "thresholds": {"type": "object", "additionalProperties": True},
                    },
                    "required": ["system_id", "device_id", "frames", "thresholds"],
                },
            },
        ]

    def _tools_schema_for_chat_completions(self) -> List[Dict[str, Any]]:
        tools = []
        for t in self._tools_schema():
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("parameters", {}),
                    },
                }
            )
        return tools

    def _summarize_response(self, response: Any) -> None:
        rid = getattr(response, "id", None)
        status = getattr(response, "status", None)
        self._log(f"[agent] response_id={rid} status={status}")
        usage = getattr(response, "usage", None)
        if usage is not None:
            self._log(f"[agent] usage={usage}")
        output = getattr(response, "output", []) or []
        for item in output:
            t = getattr(item, "type", None)
            if t == "function_call":
                self._log(f"[agent] tool_call name={item.name} call_id={item.call_id}")
                self._log(f"[agent] tool_args={self._clip(item.arguments or '')}")
            elif t == "message":
                content = ""
                try:
                    content = getattr(item, "content", "") or ""
                except Exception:
                    content = ""
                if content:
                    self._log(f"[agent] assistant_message={self._clip(str(content))}")

    def run(self) -> Dict[str, Any]:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required to run the tool-calling agent")

        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        use_responses_api = bool(getattr(client, "responses", None)) and hasattr(client.responses, "create")

        thresholds = {
            **asdict(self.config.thresholds),
            "bootstrap_enabled": self.config.bootstrap.enabled,
            "bootstrap_iterations": self.config.bootstrap.iterations,
            "bootstrap_confidence": self.config.bootstrap.confidence,
            "bootstrap_seed": self.config.bootstrap.seed,
            "downtrend_confidence_min": self.config.bootstrap.downtrend_confidence_min,
        }
        self._runtime_thresholds = thresholds

        user_task = {
            "lookback_days": self.config.ranking.lookback_days,
            "max_systems_to_flag": self.config.ranking.max_systems_to_flag,
            "top_devices_per_system": self.config.ranking.top_devices_per_system,
            "thresholds": thresholds,
        }
        if self.db_path is not None:
            user_task["db"] = {
                "path": str(self.db_path),
                "read_only": True,
                "tables": [
                    {"name": "systems", "columns": ["system_id"]},
                    {"name": "system_points", "columns": ["system_id", "date", "value"]},
                    {"name": "meta", "columns": ["key", "value"]},
                ],
            }

        self._log("[agent] starting agent run")
        self._log(f"[agent] model={self.model}")
        self._log(f"[agent] lookback_days={self.config.ranking.lookback_days}")
        self._log(f"[agent] max_systems_to_flag={self.config.ranking.max_systems_to_flag}")
        self._log(f"[agent] top_devices_per_system={self.config.ranking.top_devices_per_system}")
        self._log(f"[agent] transport={'responses' if use_responses_api else 'chat.completions'}")

        output_schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "report_summary": {"type": "string"},
                "report_md": {"type": "string"},
            },
            "required": ["report_summary", "report_md"],
        }

        def openai_create(**kwargs: Any) -> Any:
            from openai import RateLimitError

            attempts = 0
            while True:
                attempts += 1
                try:
                    return client.responses.create(**kwargs)
                except RateLimitError:
                    if attempts >= 6:
                        raise
                    time.sleep(min(2.0 ** (attempts - 1), 8.0))

        t0 = time.time()
        if use_responses_api:
            create_sig = inspect.signature(client.responses.create)
            supports_text = "text" in create_sig.parameters
            create_kwargs: Dict[str, Any] = {
                "model": self.model,
                "input": [
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": "Run AI-Insights end-to-end. Input:\n" + json.dumps(user_task, indent=2),
                    },
                ],
                "tools": self._tools_schema(),
            }
            if supports_text:
                create_kwargs["text"] = {
                    "format": {"type": "json_schema", "name": "report", "schema": output_schema}
                }

            response = openai_create(**create_kwargs)
            self._log(f"[agent] openai_call_ms={int((time.time()-t0)*1000)}")
            self._summarize_response(response)

            while True:
                tool_calls = [o for o in response.output if o.type == "function_call"]
                if not tool_calls:
                    break
                tool_outputs = []
                for call in tool_calls:
                    args = json.loads(call.arguments or "{}")
                    self._log(f"[agent] executing_tool name={call.name} call_id={call.call_id}")
                    self._executed_tool_calls.append({"name": call.name, "args": args})
                    result = self.registry.call(call.name, args)
                    self._log(f"[agent] tool_result name={call.name} bytes={len(json.dumps(result))}")
                    tool_outputs.append(
                        {"type": "function_call_output", "call_id": call.call_id, "output": json.dumps(result)}
                    )
                t1 = time.time()
                follow_kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "previous_response_id": response.id,
                    "input": tool_outputs,
                    "tools": self._tools_schema(),
                }
                if supports_text:
                    follow_kwargs["text"] = {
                        "format": {"type": "json_schema", "name": "report", "schema": output_schema}
                    }

                response = openai_create(**follow_kwargs)
                self._log(f"[agent] openai_call_ms={int((time.time()-t1)*1000)}")
                self._summarize_response(response)

            text = response.output_text
            self._log(f"[agent] final_output_text={self._clip(text)}")
            try:
                return json.loads(text)
            except Exception as e:
                raise RuntimeError(f"Agent returned non-JSON output: {text[:500]}") from e

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": "Run AI-Insights end-to-end. Input:\n" + json.dumps(user_task, indent=2)},
        ]
        tools = self._tools_schema_for_chat_completions()

        while True:
            t1 = time.time()
            completion = client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                response_format={"type": "json_object"},
            )
            self._log(f"[agent] openai_call_ms={int((time.time()-t1)*1000)}")
            msg = completion.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls:
                self._log(f"[agent] tool_calls={len(tool_calls)}")
                messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                            }
                            for tc in tool_calls
                        ],
                    }
                )
                for tc in tool_calls:
                    args = json.loads(tc.function.arguments or "{}")
                    self._log(f"[agent] executing_tool name={tc.function.name} call_id={tc.id}")
                    self._executed_tool_calls.append({"name": tc.function.name, "args": args})
                    result = self.registry.call(tc.function.name, args)
                    self._log(f"[agent] tool_result name={tc.function.name} bytes={len(json.dumps(result))}")
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(result),
                        }
                    )
                continue

            text = msg.content or ""
            self._log(f"[agent] final_output_text={self._clip(text)}")
            try:
                return json.loads(text)
            except Exception as e:
                raise RuntimeError(f"Agent returned non-JSON output: {text[:500]}") from e

    def _tool_list_system_ids(self) -> Dict[str, Any]:
        ids = self.datastore.list_system_ids()
        self._log(f"[tool] list_system_ids -> {len(ids)}")
        return {"system_ids": ids}

    def _tool_fetch_system_timeseries(self, system_id: str, lookback_days: int) -> Dict[str, Any]:
        ts = self.datastore.fetch_system_timeseries(system_id, int(lookback_days))
        self._pending_system_fetches.append(ts)
        self._log(f"[tool] fetch_system_timeseries system_id={system_id} n={len(ts.get('values', []))}")
        return ts

    def _tool_fetch_system_timeseries_frame(
        self, system_id: str, lookback_days: int, end_offset_days: int
    ) -> Dict[str, Any]:
        ts = self.datastore.fetch_system_timeseries_frame(system_id, int(lookback_days), int(end_offset_days))
        self._log(
            f"[tool] fetch_system_timeseries_frame system_id={system_id} lookback_days={lookback_days} end_offset_days={end_offset_days} n={len(ts.get('values', []))}"
        )
        return ts

    def _tool_fetch_device_timeseries(self, system_id: str, device_id: str, lookback_days: int) -> Dict[str, Any]:
        ts = self.datastore.fetch_device_timeseries(system_id, device_id, int(lookback_days))
        self._pending_device_fetches.append(ts)
        self._log(
            f"[tool] fetch_device_timeseries system_id={system_id} device_id={device_id} n={len(ts.get('values', []))}"
        )
        return ts

    def _tool_fetch_device_timeseries_frame(
        self, system_id: str, device_id: str, lookback_days: int, end_offset_days: int
    ) -> Dict[str, Any]:
        ts = self.datastore.fetch_device_timeseries_frame(
            system_id, device_id, int(lookback_days), int(end_offset_days)
        )
        self._log(
            f"[tool] fetch_device_timeseries_frame system_id={system_id} device_id={device_id} lookback_days={lookback_days} end_offset_days={end_offset_days} n={len(ts.get('values', []))}"
        )
        return ts

    def _tool_analyze_system_frames(
        self, system_id: str, frames: List[Dict[str, Any]], thresholds: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        thr = thresholds if thresholds is not None else self._runtime_thresholds

        def summarize(values: List[Dict[str, Any]]) -> Dict[str, Any]:
            if not values:
                return {"n": 0}
            vs = [float(p["v"]) for p in values]
            n = len(vs)
            mean = sum(vs) / n
            var = sum((x - mean) ** 2 for x in vs) / n
            std = var ** 0.5
            first = vs[0]
            last = vs[-1]
            delta = last - first
            pct = (delta / first * 100.0) if first not in (0.0, -0.0) else None
            return {
                "n": n,
                "first": first,
                "last": last,
                "delta": delta,
                "pct_change": pct,
                "min": min(vs),
                "max": max(vs),
                "mean": mean,
                "std": std,
            }

        out_frames: List[Dict[str, Any]] = []
        for f in frames:
            lb = int(f["lookback_days"])
            off = int(f["end_offset_days"])
            ts = self.datastore.fetch_system_timeseries_frame(system_id, lb, off)
            values = ts.get("values", []) or []
            flags = flag_series(values, thr)
            out_frames.append(
                {
                    "frame": {"lookback_days": lb, "end_offset_days": off},
                    "t": ts.get("t"),
                    "range": {
                        "start": values[0]["date"] if values else None,
                        "end": values[-1]["date"] if values else None,
                    },
                    "summary": summarize(values),
                    "flags": flags,
                }
            )

        self._log(f"[tool] analyze_system_frames system_id={system_id} frames={len(out_frames)}")
        result = {"system_id": system_id, "frames": out_frames}
        self._system_frame_analyses[str(system_id)] = result
        return result

    def _tool_analyze_device_frames(
        self,
        system_id: str,
        device_id: str,
        frames: List[Dict[str, Any]],
        thresholds: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        thr = thresholds if thresholds is not None else self._runtime_thresholds

        def summarize(values: List[Dict[str, Any]]) -> Dict[str, Any]:
            if not values:
                return {"n": 0}
            vs = [float(p["v"]) for p in values]
            n = len(vs)
            mean = sum(vs) / n
            var = sum((x - mean) ** 2 for x in vs) / n
            std = var ** 0.5
            first = vs[0]
            last = vs[-1]
            delta = last - first
            pct = (delta / first * 100.0) if first not in (0.0, -0.0) else None
            return {
                "n": n,
                "first": first,
                "last": last,
                "delta": delta,
                "pct_change": pct,
                "min": min(vs),
                "max": max(vs),
                "mean": mean,
                "std": std,
            }

        out_frames: List[Dict[str, Any]] = []
        for f in frames:
            lb = int(f["lookback_days"])
            off = int(f["end_offset_days"])
            ts = self.datastore.fetch_device_timeseries_frame(system_id, device_id, lb, off)
            values = ts.get("values", []) or []
            flags = flag_series(values, thr)
            out_frames.append(
                {
                    "frame": {"lookback_days": lb, "end_offset_days": off},
                    "t": ts.get("t"),
                    "range": {
                        "start": values[0]["date"] if values else None,
                        "end": values[-1]["date"] if values else None,
                    },
                    "summary": summarize(values),
                    "flags": flags,
                }
            )

        self._log(f"[tool] analyze_device_frames system_id={system_id} device_id={device_id} frames={len(out_frames)}")
        result = {"system_id": system_id, "device_id": device_id, "frames": out_frames}
        self._device_frame_analyses[(str(system_id), str(device_id))] = result
        return result

    def collected_evidence(self) -> Dict[str, Any]:
        devices_by_system: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for (sid, _did), payload in self._device_frame_analyses.items():
            devices_by_system[sid].append(payload)
        return {
            "population_baselines": self._population_baselines,
            "system_contexts": self._system_contexts,
            "system_frames": self._system_frame_analyses,
            "device_frames": dict(devices_by_system),
        }

    def collected_tool_calls(self) -> List[Dict[str, Any]]:
        return list(self._executed_tool_calls)

    def _tool_flag_series(
        self,
        values: List[Dict[str, Any]],
        thresholds: Dict[str, Any] | None = None,
        system_id: str | None = None,
        device_id: str | None = None,
    ) -> Dict[str, Any]:
        thr = thresholds if thresholds is not None else self._runtime_thresholds
        flags = flag_series(values, thr)

        context: Dict[str, Any] | None = None
        if system_id or device_id:
            context = {"system_id": system_id, "device_id": device_id, "t": None}
        elif self._pending_device_fetches:
            fetched = self._pending_device_fetches.popleft()
            context = {
                "system_id": fetched.get("system_id"),
                "device_id": fetched.get("device_id"),
                "t": fetched.get("t"),
            }
        elif self._pending_system_fetches:
            fetched = self._pending_system_fetches.popleft()
            context = {"system_id": fetched.get("system_id"), "device_id": None, "t": fetched.get("t")}

        if context and context.get("system_id") and context.get("device_id"):
            sid = str(context["system_id"])
            did = str(context["device_id"])
            self._device_results_by_system[sid].append({"device_id": did, "t": context.get("t"), "flags": flags})
            self._log(f"[tool] flag_series stored device_result system_id={sid} device_id={did}")
        elif context and context.get("system_id"):
            sid = str(context["system_id"])
            self._system_results.append({"system_id": sid, "t": context.get("t"), "flags": flags})
            self._log(f"[tool] flag_series stored system_result system_id={sid}")
        else:
            self._log("[tool] flag_series no context; returning flags only")

        return flags

    def _tool_rank_systems(self, top_k: int, system_results: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
        results = system_results if system_results is not None else (self._system_results or self._last_top_systems)
        self._log(f"[tool] rank_systems n={len(results)} top_k={top_k}")
        return rank_systems(results, int(top_k))

    def _tool_list_devices(self, system_id: str) -> Dict[str, Any]:
        self._last_devices_system_id = system_id
        ids = self.datastore.list_devices(system_id)
        self._log(f"[tool] list_devices system_id={system_id} -> {len(ids)}")
        return {"device_ids": ids}

    def _tool_rank_devices(
        self,
        top_k: int,
        device_results: List[Dict[str, Any]] | None = None,
        system_id: str | None = None,
    ) -> Dict[str, Any]:
        sid = system_id or self._last_devices_system_id
        if device_results is None:
            device_results = self._device_results_by_system.get(str(sid), []) if sid else []
            if (not device_results) and sid and (str(sid) in self._last_top_devices_by_system):
                device_results = self._last_top_devices_by_system[str(sid)]
        self._log(f"[tool] rank_devices system_id={sid} n={len(device_results)} top_k={top_k}")
        return rank_devices(device_results, int(top_k))

    def _tool_screen_systems(
        self, lookback_days: int, top_k: int, thresholds: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        thr = thresholds if thresholds is not None else self._runtime_thresholds
        all_results: List[Dict[str, Any]] = []
        for system_id in self.datastore.list_system_ids():
            ts = self.datastore.fetch_system_timeseries(system_id, int(lookback_days))
            flags = flag_series(ts["values"], thr)
            all_results.append({"system_id": system_id, "t": ts["t"], "flags": flags})
        top = rank_systems(all_results, int(top_k))["top"]
        self._last_top_systems = top
        self._system_results = []
        self._log(f"[tool] screen_systems analyzed={len(all_results)} -> top_k={top_k}")
        return {"analyzed_systems": len(all_results), "top_systems": top}

    def _tool_screen_devices_for_system(
        self, system_id: str, lookback_days: int, top_k: int, thresholds: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        thr = thresholds if thresholds is not None else self._runtime_thresholds
        device_ids = self.datastore.list_devices(system_id)
        all_results: List[Dict[str, Any]] = []
        for device_id in device_ids:
            ts = self.datastore.fetch_device_timeseries(system_id, device_id, int(lookback_days))
            flags = flag_series(ts["values"], thr)
            all_results.append({"device_id": device_id, "t": ts["t"], "flags": flags})
        top = rank_devices(all_results, int(top_k))["top"]
        self._last_top_devices_by_system[str(system_id)] = top
        self._log(f"[tool] screen_devices_for_system system_id={system_id} analyzed={len(all_results)} -> top_k={top_k}")
        return {"analyzed_devices": len(all_results), "top_devices": top}

    def _tool_sql_query(
        self, sql: str, params: List[Any] | None = None, named_params: Dict[str, Any] | None = None, max_rows: int = 200
    ) -> Dict[str, Any]:
        if not self.db_path:
            return {
                "error": "db_not_configured",
                "message": "No SQLite DB is configured for this run; rerun with --db (or use --agent/--agent-only with --one-day-json to auto-ingest).",
            }
        bound: Any = ()
        if named_params:
            bound = dict(named_params)
        elif params:
            bound = list(params)
        return sql_query_readonly(self.db_path, sql, bound, max_rows=int(max_rows))

    def _tool_recommend_lookback(
        self,
        top_k: int,
        thresholds: Dict[str, Any] | None = None,
        candidate_days: List[int] | None = None,
    ) -> Dict[str, Any]:
        thr = thresholds if thresholds is not None else self._runtime_thresholds
        return recommend_lookback(
            self.datastore,
            thresholds=thr,
            top_k=int(top_k),
            candidate_days=[int(x) for x in candidate_days] if candidate_days else None,
        )

    def _tool_series_summary(self, values: List[Dict[str, Any]]) -> Dict[str, Any]:
        y = [float(v["v"]) for v in (values or [])]
        if not y:
            return {"n": 0}
        n = len(y)
        mean = sum(y) / n
        var = sum((yi - mean) ** 2 for yi in y) / n
        std = var ** 0.5
        first = float(y[0])
        last = float(y[-1])
        delta = last - first
        pct = (delta / first * 100.0) if first not in (0.0, -0.0) else None
        return {
            "n": int(n),
            "first": first,
            "last": last,
            "delta": float(delta),
            "pct_change": float(pct) if pct is not None else None,
            "min": float(min(y)),
            "max": float(max(y)),
            "mean": float(mean),
            "std": float(std),
        }

    def _tool_calc_slope(self, values: List[Dict[str, Any]]) -> Dict[str, Any]:
        y = [float(v["v"]) for v in (values or [])]
        return {"slope": float(linear_regression_slope(y))}

    def _tool_calc_bootstrap_slope(
        self,
        values: List[Dict[str, Any]],
        threshold: float,
        iterations: int | None = None,
        confidence: float | None = None,
        seed: int | None = None,
    ) -> Dict[str, Any]:
        y = [float(v["v"]) for v in (values or [])]
        it = int(iterations) if iterations is not None else int(self.config.bootstrap.iterations)
        conf = float(confidence) if confidence is not None else float(self.config.bootstrap.confidence)
        sd = int(seed) if seed is not None else int(self.config.bootstrap.seed)
        bcfg = type(self.config.bootstrap)(
            enabled=True,
            iterations=it,
            confidence=conf,
            seed=sd,
            downtrend_confidence_min=float(self.config.bootstrap.downtrend_confidence_min),
        )
        return {"bootstrap": bootstrap_slope(y, threshold=float(threshold), bootstrap=bcfg)}

    def _tool_population_baseline(self, lookback_days: int, thresholds: Dict[str, Any] | None = None) -> Dict[str, Any]:
        thr = thresholds if thresholds is not None else self._runtime_thresholds
        slopes: List[float] = []
        drops: List[float] = []
        unstable_ratios: List[float] = []
        severities: List[int] = []
        counts = {"downtrend": 0, "sudden": 0, "unstable": 0, "any": 0}

        for system_id in self.datastore.list_system_ids():
            ts = self.datastore.fetch_system_timeseries(system_id, int(lookback_days))
            flags = flag_series(ts["values"], thr)
            d = flags.get("downward", {}) or {}
            sd = flags.get("sudden", {}) or {}
            u = flags.get("unstable", {}) or {}
            sev = (flags.get("severity", {}) or {}).get("severity", 0)

            slope_point = float(d.get("slope_point", 0.0) or 0.0)
            drop_amount = float(sd.get("drop_amount", 0.0) or 0.0)
            unstable_ratio = float(u.get("unstable_ratio", 0.0) or 0.0)

            slopes.append(slope_point)
            drops.append(drop_amount)
            unstable_ratios.append(unstable_ratio)
            severities.append(int(sev or 0))

            down_f = bool(d.get("flag"))
            sud_f = bool(sd.get("flag"))
            un_f = bool(u.get("flag"))
            if down_f:
                counts["downtrend"] += 1
            if sud_f:
                counts["sudden"] += 1
            if un_f:
                counts["unstable"] += 1
            if down_f or sud_f or un_f:
                counts["any"] += 1

        def q(values: List[float], p: float) -> float:
            if not values:
                return 0.0
            s = sorted(values)
            idx = int((len(s) - 1) * p)
            idx = max(0, min(idx, len(s) - 1))
            return float(s[idx])

        def qi(values: List[int], p: float) -> int:
            if not values:
                return 0
            s = sorted(values)
            idx = int((len(s) - 1) * p)
            idx = max(0, min(idx, len(s) - 1))
            return int(s[idx])

        baseline = {
            "lookback_days": int(lookback_days),
            "n_systems": len(slopes),
            "flag_counts": counts,
            "quantiles": {
                "slope_point": {"p05": q(slopes, 0.05), "p50": q(slopes, 0.50), "p95": q(slopes, 0.95)},
                "drop_amount": {"p50": q(drops, 0.50), "p95": q(drops, 0.95)},
                "unstable_ratio": {"p50": q(unstable_ratios, 0.50), "p95": q(unstable_ratios, 0.95)},
                "severity": {"p50": qi(severities, 0.50), "p95": qi(severities, 0.95)},
            },
        }
        self._population_baselines.append(baseline)
        self._log(f"[tool] population_baseline lookback_days={lookback_days} n={baseline['n_systems']}")
        return baseline

    def _tool_system_context(
        self, system_id: str, lookback_days: int, thresholds: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        thr = thresholds if thresholds is not None else self._runtime_thresholds
        ts = self.datastore.fetch_system_timeseries(system_id, int(lookback_days))
        flags = flag_series(ts["values"], thr)

        slopes: List[float] = []
        drops: List[float] = []
        unstable_ratios: List[float] = []
        severities: List[int] = []
        for sid in self.datastore.list_system_ids():
            ts2 = self.datastore.fetch_system_timeseries(sid, int(lookback_days))
            f2 = flag_series(ts2["values"], thr)
            d2 = f2.get("downward", {}) or {}
            sd2 = f2.get("sudden", {}) or {}
            u2 = f2.get("unstable", {}) or {}
            sev2 = (f2.get("severity", {}) or {}).get("severity", 0)
            slopes.append(float(d2.get("slope_point", 0.0) or 0.0))
            drops.append(float(sd2.get("drop_amount", 0.0) or 0.0))
            unstable_ratios.append(float(u2.get("unstable_ratio", 0.0) or 0.0))
            severities.append(int(sev2 or 0))

        def pct_rank(values: List[float], x: float) -> float:
            if not values:
                return 0.0
            s = sorted(values)
            lo = 0
            hi = len(s)
            while lo < hi:
                mid = (lo + hi) // 2
                if s[mid] <= x:
                    lo = mid + 1
                else:
                    hi = mid
            return float(lo / len(s))

        def pct_rank_i(values: List[int], x: int) -> float:
            if not values:
                return 0.0
            s = sorted(values)
            lo = 0
            hi = len(s)
            while lo < hi:
                mid = (lo + hi) // 2
                if s[mid] <= x:
                    lo = mid + 1
                else:
                    hi = mid
            return float(lo / len(s))

        d = flags.get("downward", {}) or {}
        sd = flags.get("sudden", {}) or {}
        u = flags.get("unstable", {}) or {}
        sev = int((flags.get("severity", {}) or {}).get("severity", 0) or 0)

        context = {
            "system_id": system_id,
            "lookback_days": int(lookback_days),
            "t": ts.get("t"),
            "flags": flags,
            "percentiles": {
                "slope_point": pct_rank(slopes, float(d.get("slope_point", 0.0) or 0.0)),
                "drop_amount": pct_rank(drops, float(sd.get("drop_amount", 0.0) or 0.0)),
                "unstable_ratio": pct_rank(unstable_ratios, float(u.get("unstable_ratio", 0.0) or 0.0)),
                "severity": pct_rank_i(severities, sev),
            },
        }
        self._system_contexts[str(system_id)] = context
        self._log(f"[tool] system_context system_id={system_id} lookback_days={lookback_days}")
        return context
