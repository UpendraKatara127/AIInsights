from __future__ import annotations

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
from .tools.metrics import flag_series
from .tools.rank import rank_devices, rank_systems


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
        config: Config,
        model: str,
        prompt_path: Path,
        trace: bool = False,
        max_log_chars: int = 2000,
    ) -> None:
        self.datastore = datastore
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

        self.registry = ToolRegistry(
            {
                "list_system_ids": self._tool_list_system_ids,
                "fetch_system_timeseries": self._tool_fetch_system_timeseries,
                "flag_series": self._tool_flag_series,
                "rank_systems": self._tool_rank_systems,
                "list_devices": self._tool_list_devices,
                "fetch_device_timeseries": self._tool_fetch_device_timeseries,
                "rank_devices": self._tool_rank_devices,
                # High-level helpers to keep the agent fast/reliable and token-efficient.
                "screen_systems": self._tool_screen_systems,
                "screen_devices_for_system": self._tool_screen_devices_for_system,
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
        return [
            {
                "type": "function",
                "name": "list_system_ids",
                "description": "List all known system_ids",
                "parameters": {"type": "object", "properties": {}, "required": []},
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
                    # response.output_text is often the aggregated content; for trace we also print per message.
                    content = getattr(item, "content", "") or ""
                except Exception:
                    content = ""
                if content:
                    self._log(f"[agent] assistant_message={self._clip(str(content))}")

    def run(self) -> Dict[str, Any]:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required to run the tool-calling agent")

        # Lazy import so tests don't require the dependency at import time.
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
        # Used as default tool argument if the model omits thresholds.
        self._runtime_thresholds = thresholds

        user_task = {
            "lookback_days": self.config.ranking.lookback_days,
            "max_systems_to_flag": self.config.ranking.max_systems_to_flag,
            "top_devices_per_system": self.config.ranking.top_devices_per_system,
            "thresholds": thresholds,
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
            # Simple retry loop for transient RateLimit errors.
            from openai import RateLimitError  # type: ignore

            attempts = 0
            while True:
                attempts += 1
                try:
                    return client.responses.create(**kwargs)
                except RateLimitError as e:
                    if attempts >= 6:
                        raise
                    # Backoff: sleep a bit and retry.
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
                    # Tools must be provided on follow-up calls, otherwise the model can't emit function calls.
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
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(f"Agent returned non-JSON output: {text[:500]}") from e

        # Fallback for older OpenAI Python SDKs that don't expose Responses API on OpenAI().
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
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(f"Agent returned non-JSON output: {text[:500]}") from e

    # -----------------------
    # Tool implementations
    # -----------------------

    def _tool_list_system_ids(self) -> Dict[str, Any]:
        ids = self.datastore.list_system_ids()
        self._log(f"[tool] list_system_ids -> {len(ids)}")
        return {"system_ids": ids}

    def _tool_fetch_system_timeseries(self, system_id: str, lookback_days: int) -> Dict[str, Any]:
        ts = self.datastore.fetch_system_timeseries(system_id, int(lookback_days))
        self._pending_system_fetches.append(ts)
        self._log(f"[tool] fetch_system_timeseries system_id={system_id} n={len(ts.get('values', []))}")
        return ts

    def _tool_fetch_device_timeseries(self, system_id: str, device_id: str, lookback_days: int) -> Dict[str, Any]:
        ts = self.datastore.fetch_device_timeseries(system_id, device_id, int(lookback_days))
        self._pending_device_fetches.append(ts)
        self._log(
            f"[tool] fetch_device_timeseries system_id={system_id} device_id={device_id} n={len(ts.get('values', []))}"
        )
        return ts

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
        self._system_results = []  # avoid retaining 500 results and accidentally re-sending
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
