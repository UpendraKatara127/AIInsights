from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

from .agent import AIInsightsAgent
from .config import BootstrapConfig, Config, RankingConfig, Thresholds
from .tools.fetch import DataStore, generate_one_day_json, generate_synthetic_data
from .tools.metrics import flag_series
from .tools.rank import rank_devices, rank_systems
from .tools.sqlite_store import SqliteDataStore, ingest_one_day_json
from .tools.profile import recommend_lookback


def _ensure_output_dir() -> Path:
    out = Path("output")
    out.mkdir(exist_ok=True)
    return out


def _write_outputs(report_json: dict, report_md: str) -> tuple[Path, Path]:
    out = _ensure_output_dir()
    json_path = out / "report.json"
    md_path = out / "report.md"
    json_path.write_text(json.dumps(report_json, indent=2), encoding="utf-8")
    md_path.write_text(report_md, encoding="utf-8")
    return json_path, md_path


def _run_python_pipeline(datastore: DataStore, cfg: Config) -> dict:
    thresholds = {
        **asdict(cfg.thresholds),
        "bootstrap_enabled": cfg.bootstrap.enabled,
        "bootstrap_iterations": cfg.bootstrap.iterations,
        "bootstrap_confidence": cfg.bootstrap.confidence,
        "bootstrap_seed": cfg.bootstrap.seed,
        "downtrend_confidence_min": cfg.bootstrap.downtrend_confidence_min,
    }

    system_results = []
    for system_id in datastore.list_system_ids():
        ts = datastore.fetch_system_timeseries(system_id, cfg.ranking.lookback_days)
        flags = flag_series(ts["values"], thresholds)
        system_results.append({"system_id": system_id, "t": ts["t"], "flags": flags})

    top_systems = rank_systems(system_results, cfg.ranking.max_systems_to_flag)["top"]

    systems_out = []
    for s in top_systems:
        system_id = s["system_id"]
        devices = datastore.list_devices(system_id)
        device_results = []
        for device_id in devices:
            ts = datastore.fetch_device_timeseries(system_id, device_id, cfg.ranking.lookback_days)
            flags = flag_series(ts["values"], thresholds)
            device_results.append({"device_id": device_id, "t": ts["t"], "flags": flags})
        top_devices = rank_devices(device_results, cfg.ranking.top_devices_per_system)["top"]
        systems_out.append({**s, "top_devices": top_devices})

    return {
        "config": {
            "thresholds": thresholds,
            "lookback_days": cfg.ranking.lookback_days,
            "max_systems_to_flag": cfg.ranking.max_systems_to_flag,
            "top_devices_per_system": cfg.ranking.top_devices_per_system,
        },
        "systems": systems_out,
    }


def _render_markdown(report: dict) -> str:
    lines = []
    lines.append("# AI-Insights Report")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append(f"- Flagged systems: {len(report['systems'])}")
    cfg = report.get("config", {}) if isinstance(report, dict) else {}
    lb = (cfg.get("lookback_days") if isinstance(cfg, dict) else None) or "—"
    auto = (cfg.get("auto", {}) if isinstance(cfg, dict) else {}) or {}
    if isinstance(auto, dict) and auto.get("lookback_profile"):
        lines.append(f"- Lookback: {lb} (auto)")
    else:
        lines.append(f"- Lookback: {lb}")
    lines.append("")
    lines.append("## Top Systems")
    for s in report["systems"]:
        flags = s["flags"]
        sev = flags["severity"]["severity"]
        lines.append(f"### System `{s['system_id']}` (severity {sev})")
        d = flags["downward"]
        if d.get("bootstrap", {}).get("enabled"):
            b = d["bootstrap"]
            lines.append(
                f"- Downtrend: `{d['flag']}` slope={d['slope_point']:.4f}, "
                f"CI[{b['ci_low']:.4f},{b['ci_high']:.4f}], p_below_threshold={b['p_below_threshold']:.2f}"
            )
        else:
            lines.append(f"- Downtrend: `{d['flag']}` slope={d['slope_point']:.4f}")
        sd = flags["sudden"]
        lines.append(
            f"- Sudden drop: `{sd['flag']}` drop_amount={sd.get('drop_amount')}"
        )
        u = flags["unstable"]
        lines.append(
            f"- Unstable: `{u['flag']}` unstable_ratio={u.get('unstable_ratio')}"
        )
        if s.get("top_devices"):
            lines.append("")
            lines.append("Top devices:")
            for dev in s["top_devices"]:
                lines.append(
                    f"- `{dev['device_id']}` severity={dev['flags']['severity']['severity']}"
                )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _render_investigation_log(evidence: dict) -> str:
    ev = evidence or {}
    system_frames = ev.get("system_frames", {}) or {}
    population_baselines = ev.get("population_baselines", []) or []
    system_contexts = ev.get("system_contexts", {}) or {}
    has_any = bool(system_frames) or bool(population_baselines) or bool(system_contexts)
    if not has_any:
        return ""

    lines: list[str] = []
    lines.append("## Investigation Log (Tool Evidence)")
    lines.append("")
    lines.append("Each row below is produced by Python tools (no LLM math).")
    lines.append("")

    def fmt_num(x: object, nd: int = 4) -> str:
        if x is None:
            return "—"
        try:
            return f"{float(x):.{nd}f}"
        except Exception:
            return "—"

    def fmt_pct(x: object) -> str:
        if x is None:
            return "—"
        try:
            return f"{float(x):.2f}"
        except Exception:
            return "—"

    if population_baselines:
        last = population_baselines[-1]
        q = (last.get("quantiles") or {}) if isinstance(last, dict) else {}
        fc = (last.get("flag_counts") or {}) if isinstance(last, dict) else {}
        lines.append("### Population baseline")
        lines.append("")
        lines.append(f"- lookback_days={last.get('lookback_days')} n_systems={last.get('n_systems')}")
        lines.append(
            f"- flag_counts: any={fc.get('any')} downtrend={fc.get('downtrend')} sudden={fc.get('sudden')} unstable={fc.get('unstable')}"
        )
        sp = (q.get("slope_point") or {}) if isinstance(q, dict) else {}
        da = (q.get("drop_amount") or {}) if isinstance(q, dict) else {}
        ur = (q.get("unstable_ratio") or {}) if isinstance(q, dict) else {}
        lines.append(
            f"- slope_point quantiles: p05={fmt_num(sp.get('p05'))} p50={fmt_num(sp.get('p50'))} p95={fmt_num(sp.get('p95'))}"
        )
        lines.append(
            f"- drop_amount quantiles: p50={fmt_num(da.get('p50'))} p95={fmt_num(da.get('p95'))}"
        )
        lines.append(
            f"- unstable_ratio quantiles: p50={fmt_num(ur.get('p50'))} p95={fmt_num(ur.get('p95'))}"
        )
        lines.append("")

    for system_id in sorted(system_frames.keys() or system_contexts.keys()):
        ctx = system_contexts.get(system_id) or {}
        p = (ctx.get("percentiles") or {}) if isinstance(ctx, dict) else {}
        if ctx:
            lines.append(f"### System `{system_id}` — population context")
            lines.append("")
            lines.append(
                f"- percentiles: severity={fmt_pct(p.get('severity'))} slope_point={fmt_pct(p.get('slope_point'))} drop_amount={fmt_pct(p.get('drop_amount'))} unstable_ratio={fmt_pct(p.get('unstable_ratio'))}"
            )
            lines.append("")

        payload = system_frames.get(system_id) or {}
        frames = payload.get("frames", []) or []
        if not frames:
            continue
        payload = system_frames.get(system_id) or {}
        frames = payload.get("frames", []) or []
        if not frames:
            continue
        lines.append(f"### System `{system_id}` — frame comparisons")
        lines.append("")
        lines.append("| frame | range | slope | CI_low | CI_high | p_below | downtrend | sudden | unstable | severity |")
        lines.append("|---:|:---|---:|---:|---:|---:|:---:|:---:|:---:|---:|")
        for fr in frames:
            f = fr.get("frame", {}) or {}
            lb = f.get("lookback_days")
            off = f.get("end_offset_days")
            r = fr.get("range", {}) or {}
            start = r.get("start")
            end = r.get("end")
            flags = fr.get("flags", {}) or {}
            d = (flags.get("downward") or {}) if isinstance(flags, dict) else {}
            b = (d.get("bootstrap") or {}) if isinstance(d, dict) else {}
            sd = (flags.get("sudden") or {}) if isinstance(flags, dict) else {}
            u = (flags.get("unstable") or {}) if isinstance(flags, dict) else {}
            sev = (flags.get("severity") or {}) if isinstance(flags, dict) else {}
            frame_label = f"{lb}@t-{off}"
            date_range = f"{start}→{end}"
            lines.append(
                f"| {frame_label} | {date_range} | {fmt_num(d.get('slope_point'), 4)}"
                f" | {fmt_num(b.get('ci_low'), 4)} | {fmt_num(b.get('ci_high'), 4)}"
                f" | {fmt_pct(b.get('p_below_threshold'))}"
                f" | {'T' if d.get('flag') else 'F'}"
                f" | {'T' if sd.get('flag') else 'F'}"
                f" | {'T' if u.get('flag') else 'F'}"
                f" | {sev.get('severity', '—')} |"
            )
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def _render_tool_trace(tool_calls: list[dict]) -> str:
    if not tool_calls:
        return ""
    lines: list[str] = []
    lines.append("## Tool Trace (Executed)")
    lines.append("")
    for i, tc in enumerate(tool_calls, start=1):
        name = tc.get("name")
        args = tc.get("args", {})
        lines.append(f"{i}. {name} {json.dumps(args, sort_keys=True)}")
    lines.append("")
    return "\n".join(lines).strip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ai_insights_agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="Ingest One_day_data.json into SQLite")
    p_ing.add_argument("--one-day-json", type=Path, required=True)
    p_ing.add_argument("--db", type=Path, required=True)
    p_ing.add_argument("--one-day-end-date")
    p_ing.set_defaults(
        func=lambda a: (print(json.dumps(ingest_one_day_json(a.one_day_json, a.db, end_date=a.one_day_end_date), indent=2)) or 0)
    )

    p_gen = sub.add_parser("generate-data", help="Generate synthetic system/device CSVs")
    p_gen.add_argument("--systems", type=int, default=500)
    p_gen.add_argument("--days", type=int, default=20)
    p_gen.add_argument(
        "--out-system",
        type=Path,
        default=Path("src/ai_insights_agent/data/sample_system_timeseries.csv"),
    )
    p_gen.add_argument(
        "--out-device",
        type=Path,
        default=Path("src/ai_insights_agent/data/sample_device_timeseries.csv"),
    )
    p_gen.add_argument(
        "--out-one-day-json",
        type=Path,
        default=Path("src/ai_insights_agent/data/sample_one_day_data.json"),
        help="Optional One_day_data.json-style output",
    )
    p_gen.set_defaults(
        func=lambda a: (
            generate_synthetic_data(
                systems=a.systems,
                days=a.days,
                out_system=a.out_system,
                out_device=a.out_device,
            )
            or generate_one_day_json(
                systems=a.systems,
                days=a.days,
                out_json=a.out_one_day_json,
            )
            or 0
        )
    )

    p_run = sub.add_parser("run", help="Run AI-Insights analysis and write reports")
    p_run.add_argument(
        "--one-day-json",
        type=Path,
        help="Input in One_day_data.json format (list of {systemName, data:[...]})",
    )
    p_run.add_argument(
        "--db",
        type=Path,
        help="SQLite DB path (if provided, data is read from SQLite; use `ingest` first).",
    )
    p_run.add_argument(
        "--one-day-end-date",
        help="When using --one-day-json, synthesize dates ending at this YYYY-MM-DD (default: today).",
    )
    p_run.add_argument(
        "--system-csv",
        type=Path,
        default=Path("src/ai_insights_agent/data/sample_system_timeseries.csv"),
        help="System CSV input (ignored if --one-day-json is provided)",
    )
    p_run.add_argument(
        "--device-csv",
        type=Path,
        default=Path("src/ai_insights_agent/data/sample_device_timeseries.csv"),
        help="Optional device CSV input (ignored if --one-day-json is provided)",
    )
    p_run.add_argument("--lookback", type=int, default=None, help="Lookback days (default: auto)")
    p_run.add_argument("--topk", type=int, default=5)
    p_run.add_argument("--top-devices", type=int, default=3)
    p_run.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"))
    p_run.add_argument("--downward-slope-threshold", type=float, default=-0.15)
    p_run.add_argument("--sudden-drop-percentile", type=int, default=10)
    p_run.add_argument("--minimum-sudden-drop-amount", type=float, default=2.5)
    p_run.add_argument("--unstable-z-threshold", type=float, default=1.5)
    p_run.add_argument("--unstable-ratio-threshold", type=float, default=0.3)
    p_run.add_argument("--bootstrap-iterations", type=int, default=2000)
    p_run.add_argument("--bootstrap-confidence", type=float, default=0.95)
    p_run.add_argument("--downtrend-confidence-min", type=float, default=0.80)
    p_run.add_argument(
        "--agent",
        action="store_true",
        help="Use the tool-calling LLM agent to orchestrate; otherwise run a fast Python pipeline and write reports locally.",
    )
    p_run.add_argument(
        "--agent-only",
        action="store_true",
        help="Produce the report purely from the agent run (no fixed Python report); writes tool evidence + agent narrative.",
    )
    p_run.add_argument(
        "--trace-agent",
        action="store_true",
        help="Print LLM request/response + tool-call trace to stderr (agent mode only).",
    )
    p_run.add_argument(
        "--agent-max-log-chars",
        type=int,
        default=2000,
        help="Max characters to print per logged blob (agent mode only).",
    )
    p_run.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print output paths to stdout.",
    )

    def _run(a: argparse.Namespace) -> int:
        thresholds_obj = Thresholds(
            downward_slope_threshold=a.downward_slope_threshold,
            sudden_drop_percentile=a.sudden_drop_percentile,
            minimum_sudden_drop_amount=a.minimum_sudden_drop_amount,
            unstable_z_threshold=a.unstable_z_threshold,
            unstable_ratio_threshold=a.unstable_ratio_threshold,
        )
        bootstrap_obj = BootstrapConfig(
            enabled=True,
            iterations=a.bootstrap_iterations,
            confidence=a.bootstrap_confidence,
            seed=42,
            downtrend_confidence_min=a.downtrend_confidence_min,
        )
        db_path = a.db
        agent_mode = bool(a.agent) or bool(a.agent_only)
        if db_path:
            datastore = SqliteDataStore.from_db(db_path)
        elif agent_mode and a.one_day_json:
            out = _ensure_output_dir()
            db_path = out / "aiinsights_auto.db"
            ingest_one_day_json(a.one_day_json, db_path, end_date=a.one_day_end_date)
            if not a.quiet:
                print(f"Auto-ingested into SQLite: {db_path}")
            datastore = SqliteDataStore.from_db(db_path)
        elif a.one_day_json:
            datastore = DataStore.from_one_day_json(a.one_day_json, end_date=a.one_day_end_date)
        else:
            datastore = DataStore.from_csv(a.system_csv, a.device_csv if a.device_csv.exists() else None)

        thresholds_dict = {
            **asdict(thresholds_obj),
            "bootstrap_enabled": bootstrap_obj.enabled,
            "bootstrap_iterations": bootstrap_obj.iterations,
            "bootstrap_confidence": bootstrap_obj.confidence,
            "bootstrap_seed": bootstrap_obj.seed,
            "downtrend_confidence_min": bootstrap_obj.downtrend_confidence_min,
        }

        lookback_profile = None
        if a.lookback is None:
            lookback_profile = recommend_lookback(
                datastore,
                thresholds=thresholds_dict,
                top_k=int(a.topk),
            )
            lookback_days = int(lookback_profile["recommended_lookback_days"])
            if not a.quiet:
                print(f"Auto-selected lookback_days={lookback_days}")
        else:
            lookback_days = int(a.lookback)

        cfg = Config(
            thresholds=thresholds_obj,
            bootstrap=bootstrap_obj,
            ranking=RankingConfig(
                max_systems_to_flag=a.topk,
                top_devices_per_system=a.top_devices,
                lookback_days=lookback_days,
            ),
        )

        report = None if a.agent_only else _run_python_pipeline(datastore, cfg)
        if (report is not None) and lookback_profile:
            report.setdefault("config", {}).setdefault("auto", {})["lookback_profile"] = lookback_profile

        if agent_mode:
            agent = AIInsightsAgent(
                datastore=datastore,
                db_path=db_path,
                config=cfg,
                model=a.model,
                prompt_path=Path(__file__).with_name("prompts") / "system_prompt.txt",
                trace=bool(a.trace_agent),
                max_log_chars=int(a.agent_max_log_chars),
            )
            agent_out = agent.run()
            evidence = agent.collected_evidence()
            tool_trace = agent.collected_tool_calls()
            agent_md = str(agent_out.get("report_md", "")).strip()
            investigation_md = _render_investigation_log(evidence).strip()
            tool_trace_md = _render_tool_trace(tool_trace).strip()
            if report is None:
                report = {
                    "config": {
                        "thresholds": {
                            **asdict(cfg.thresholds),
                            "bootstrap_enabled": cfg.bootstrap.enabled,
                            "bootstrap_iterations": cfg.bootstrap.iterations,
                            "bootstrap_confidence": cfg.bootstrap.confidence,
                            "bootstrap_seed": cfg.bootstrap.seed,
                            "downtrend_confidence_min": cfg.bootstrap.downtrend_confidence_min,
                        },
                        "lookback_days": cfg.ranking.lookback_days,
                        "max_systems_to_flag": cfg.ranking.max_systems_to_flag,
                        "top_devices_per_system": cfg.ranking.top_devices_per_system,
                    }
                }
                if lookback_profile:
                    report.setdefault("config", {}).setdefault("auto", {})["lookback_profile"] = lookback_profile
                report_md = "# AI-Insights Report (Agent-Only)\n"
            else:
                report_md = _render_markdown(report)

            report["investigation_evidence"] = evidence
            report["executed_tool_calls"] = tool_trace
            report["agent_output"] = agent_out
            if tool_trace_md:
                report_md += "\n" + tool_trace_md + "\n"
            if investigation_md:
                report_md += "\n" + investigation_md + "\n"
            if agent_md:
                report_md += "\n## Agent Narrative\n\n" + agent_md + "\n"
            json_path, md_path = _write_outputs(report, report_md)
            if not a.quiet:
                print(f"Wrote {json_path}")
                print(f"Wrote {md_path}")
            return 0

        report_md = _render_markdown(report)
        json_path, md_path = _write_outputs(report, report_md)
        if not a.quiet:
            print(f"Wrote {json_path}")
            print(f"Wrote {md_path}")
        return 0

    p_run.set_defaults(func=_run)

    args = parser.parse_args(argv)
    return int(args.func(args))
