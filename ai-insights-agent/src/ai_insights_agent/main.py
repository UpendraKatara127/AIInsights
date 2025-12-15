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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ai_insights_agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

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
    p_run.add_argument("--lookback", type=int, default=20)
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
        cfg = Config(
            thresholds=Thresholds(
                downward_slope_threshold=a.downward_slope_threshold,
                sudden_drop_percentile=a.sudden_drop_percentile,
                minimum_sudden_drop_amount=a.minimum_sudden_drop_amount,
                unstable_z_threshold=a.unstable_z_threshold,
                unstable_ratio_threshold=a.unstable_ratio_threshold,
            ),
            bootstrap=BootstrapConfig(
                enabled=True,
                iterations=a.bootstrap_iterations,
                confidence=a.bootstrap_confidence,
                seed=42,
                downtrend_confidence_min=a.downtrend_confidence_min,
            ),
            ranking=RankingConfig(
                max_systems_to_flag=a.topk,
                top_devices_per_system=a.top_devices,
                lookback_days=a.lookback,
            ),
        )
        if a.one_day_json:
            datastore = DataStore.from_one_day_json(a.one_day_json, end_date=a.one_day_end_date)
        else:
            datastore = DataStore.from_csv(a.system_csv, a.device_csv if a.device_csv.exists() else None)

        report = _run_python_pipeline(datastore, cfg)

        if a.agent:
            agent = AIInsightsAgent(
                datastore=datastore,
                config=cfg,
                model=a.model,
                prompt_path=Path(__file__).with_name("prompts") / "system_prompt.txt",
                trace=bool(a.trace_agent),
                max_log_chars=int(a.agent_max_log_chars),
            )
            agent_out = agent.run()
            base_md = _render_markdown(report)
            agent_md = str(agent_out.get("report_md", "")).strip()
            report_md = base_md
            if agent_md:
                report_md += "\n## Agent Narrative\n\n" + agent_md + "\n"
            report["agent_output"] = agent_out
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
