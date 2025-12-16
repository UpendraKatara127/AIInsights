from __future__ import annotations

import csv
import datetime as dt
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class Point:
    date: str  # YYYY-MM-DD
    v: float


def _parse_date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def _format_date(d: dt.date) -> str:
    return d.isoformat()


def _slice_frame(points: List[Point], *, lookback_days: int, end_offset_days: int) -> tuple[str | None, List[Point]]:
    if lookback_days <= 0:
        raise ValueError("lookback_days must be > 0")
    if end_offset_days < 0:
        raise ValueError("end_offset_days must be >= 0")
    if not points:
        return None, []

    t0 = _parse_date(max(p.date for p in points))
    t = t0 - dt.timedelta(days=end_offset_days)
    window = [p for p in points if _parse_date(p.date) <= t][-lookback_days:]
    return _format_date(t), window


@dataclass
class DataStore:
    system_series: Dict[str, List[Point]]
    device_series: Dict[Tuple[str, str], List[Point]]

    @staticmethod
    def from_csv(system_csv: Path, device_csv: Optional[Path] = None) -> "DataStore":
        system_series: Dict[str, List[Point]] = {}
        with system_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = str(row["system_id"])
                system_series.setdefault(sid, []).append(Point(date=str(row["date"]), v=float(row["value"])))
        for sid in list(system_series.keys()):
            system_series[sid].sort(key=lambda p: p.date)

        device_series: Dict[Tuple[str, str], List[Point]] = {}
        if device_csv and device_csv.exists():
            with device_csv.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    sid = str(row["system_id"])
                    did = str(row["device_id"])
                    device_series.setdefault((sid, did), []).append(
                        Point(date=str(row["date"]), v=float(row["value"]))
                    )
            for key in list(device_series.keys()):
                device_series[key].sort(key=lambda p: p.date)

        return DataStore(system_series=system_series, device_series=device_series)

    @staticmethod
    def from_one_day_json(one_day_json: Path, *, end_date: str | None = None) -> "DataStore":
        """
        Load One_day_data.json-style payload:
          [
            {"systemName": "System 165", "data": [..float..]},
            ...
          ]

        Since no dates exist, synthesize YYYY-MM-DD dates ending at `end_date` (defaults to today).
        Each index in `data` is treated as a daily entry in chronological order.
        """
        obj = json.loads(one_day_json.read_text(encoding="utf-8"))
        if not isinstance(obj, list):
            raise ValueError("One_day_data.json must be a list of objects")
        t = _parse_date(end_date) if end_date else dt.date.today()

        system_series: Dict[str, List[Point]] = {}
        for row in obj:
            if not isinstance(row, dict) or set(row.keys()) != {"systemName", "data"}:
                raise ValueError("Invalid One_day_data.json row; expected keys: systemName, data")
            system_id = str(row["systemName"])
            data = row["data"]
            if not isinstance(data, list):
                raise ValueError("Invalid One_day_data.json row; data must be a list")
            points: List[Point] = []
            base = t - dt.timedelta(days=max(0, len(data) - 1))
            for i, v in enumerate(data):
                d = base + dt.timedelta(days=i)
                points.append(Point(date=_format_date(d), v=float(v)))
            system_series[system_id] = points

        return DataStore(system_series=system_series, device_series={})

    def list_system_ids(self) -> List[str]:
        return sorted(self.system_series.keys())

    def list_devices(self, system_id: str) -> List[str]:
        out = {did for (sid, did) in self.device_series.keys() if sid == system_id}
        return sorted(out)

    def fetch_system_timeseries(self, system_id: str, lookback_days: int) -> Dict:
        points = self.system_series.get(system_id, [])
        if not points:
            return {"system_id": system_id, "t": None, "values": []}
        t, window = _slice_frame(points, lookback_days=lookback_days, end_offset_days=0)
        return {
            "system_id": system_id,
            "t": t,
            "values": [{"date": p.date, "v": p.v} for p in window],
        }

    def fetch_device_timeseries(self, system_id: str, device_id: str, lookback_days: int) -> Dict:
        points = self.device_series.get((system_id, device_id), [])
        if not points:
            return {"system_id": system_id, "device_id": device_id, "t": None, "values": []}
        t, window = _slice_frame(points, lookback_days=lookback_days, end_offset_days=0)
        return {
            "system_id": system_id,
            "device_id": device_id,
            "t": t,
            "values": [{"date": p.date, "v": p.v} for p in window],
        }

    def fetch_system_timeseries_frame(self, system_id: str, lookback_days: int, end_offset_days: int) -> Dict:
        points = self.system_series.get(system_id, [])
        t, window = _slice_frame(points, lookback_days=int(lookback_days), end_offset_days=int(end_offset_days))
        return {
            "system_id": system_id,
            "t": t,
            "values": [{"date": p.date, "v": p.v} for p in window],
        }

    def fetch_device_timeseries_frame(
        self, system_id: str, device_id: str, lookback_days: int, end_offset_days: int
    ) -> Dict:
        points = self.device_series.get((system_id, device_id), [])
        t, window = _slice_frame(points, lookback_days=int(lookback_days), end_offset_days=int(end_offset_days))
        return {
            "system_id": system_id,
            "device_id": device_id,
            "t": t,
            "values": [{"date": p.date, "v": p.v} for p in window],
        }


def generate_synthetic_data(
    *,
    systems: int,
    days: int,
    out_system: Path,
    out_device: Optional[Path],
    seed: int = 42,
    devices_per_system: int = 5,
    anomaly_rate: float = 0.08,
) -> None:
    rng = random.Random(seed)
    start = dt.date.today() - dt.timedelta(days=days - 1)

    out_system.parent.mkdir(parents=True, exist_ok=True)
    with out_system.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["system_id", "date", "value"])
        w.writeheader()
        for s in range(1, systems + 1):
            sid = f"S{s}"
            base = rng.uniform(10.0, 20.0)
            drift = rng.uniform(-0.05, 0.05)
            inject = rng.random() < anomaly_rate
            for i in range(days):
                d = start + dt.timedelta(days=i)
                noise = rng.gauss(0, 0.5)
                v = base + drift * i + noise
                if inject and i == days - 1 and rng.random() < 0.5:
                    v -= rng.uniform(3.0, 7.0)  # sudden drop
                if inject and rng.random() < 0.2:
                    v += rng.gauss(0, 2.0)  # instability
                w.writerow({"system_id": sid, "date": _format_date(d), "value": f"{v:.6f}"})

    if out_device:
        out_device.parent.mkdir(parents=True, exist_ok=True)
        with out_device.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["system_id", "device_id", "date", "value"])
            w.writeheader()
            for s in range(1, systems + 1):
                sid = f"S{s}"
                for dev in range(1, devices_per_system + 1):
                    did = f"D{dev}"
                    base = rng.uniform(10.0, 20.0)
                    drift = rng.uniform(-0.08, 0.03)
                    inject = rng.random() < anomaly_rate
                    for i in range(days):
                        d = start + dt.timedelta(days=i)
                        noise = rng.gauss(0, 0.7)
                        v = base + drift * i + noise
                        if inject and i == days - 1 and rng.random() < 0.4:
                            v -= rng.uniform(3.0, 8.0)
                        if inject and rng.random() < 0.25:
                            v += rng.gauss(0, 2.5)
                        w.writerow(
                            {
                                "system_id": sid,
                                "device_id": did,
                                "date": _format_date(d),
                                "value": f"{v:.6f}",
                            }
                        )


def generate_one_day_json(
    *,
    systems: int,
    days: int,
    out_json: Path,
    seed: int = 42,
) -> None:
    rng = random.Random(seed)
    payload = []
    for s in range(1, systems + 1):
        system_name = f"System {s}"
        base = rng.uniform(10.0, 20.0)
        drift = rng.uniform(-0.08, 0.05)
        series = []
        for i in range(days):
            v = base + drift * i + rng.gauss(0, 0.6)
            series.append(float(f"{v:.6f}"))
        payload.append({"systemName": system_name, "data": series})
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
