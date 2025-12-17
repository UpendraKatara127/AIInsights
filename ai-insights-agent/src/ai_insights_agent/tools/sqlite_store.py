from __future__ import annotations

import json
import sqlite3
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _parse_date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def _format_date(d: dt.date) -> str:
    return d.isoformat()


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def _connect_ro(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS systems (system_id TEXT PRIMARY KEY)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS system_points ("
        "  system_id TEXT NOT NULL,"
        "  date TEXT NOT NULL,"
        "  value REAL NOT NULL,"
        "  PRIMARY KEY (system_id, date),"
        "  FOREIGN KEY (system_id) REFERENCES systems(system_id)"
        ")"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_system_points_sid_date ON system_points(system_id, date)")
    conn.commit()


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))


def get_meta(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return str(row["value"]) if row else None


def ingest_one_day_json(one_day_json: Path, db_path: Path, *, end_date: str | None = None) -> Dict[str, Any]:
    obj = json.loads(one_day_json.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError("one_day_json must be a list")
    t = _parse_date(end_date) if end_date else dt.date.today()

    conn = _connect(db_path)
    try:
        init_db(conn)
        conn.execute("BEGIN")
        inserted_points = 0
        inserted_systems = 0
        for row in obj:
            if not isinstance(row, dict):
                continue
            system_id = str(row.get("systemName", ""))
            data = row.get("data", [])
            if not system_id or not isinstance(data, list):
                continue
            cur = conn.execute("INSERT OR IGNORE INTO systems(system_id) VALUES (?)", (system_id,))
            if cur.rowcount:
                inserted_systems += 1
            base = t - dt.timedelta(days=max(0, len(data) - 1))
            for i, v in enumerate(data):
                d = _format_date(base + dt.timedelta(days=i))
                cur2 = conn.execute(
                    "INSERT OR REPLACE INTO system_points(system_id, date, value) VALUES (?, ?, ?)",
                    (system_id, d, float(v)),
                )
                inserted_points += int(cur2.rowcount or 0)
        set_meta(conn, "ready", "1")
        set_meta(conn, "source", str(one_day_json))
        set_meta(conn, "end_date", _format_date(t))
        set_meta(conn, "updated_at", _format_date(dt.date.today()))
        conn.commit()
    finally:
        conn.close()

    return {
        "db_path": str(db_path),
        "ready": True,
        "inserted_systems": int(inserted_systems),
        "inserted_points": int(inserted_points),
        "end_date": _format_date(t),
    }


@dataclass
class SqliteDataStore:
    db_path: Path

    @staticmethod
    def from_db(db_path: Path) -> "SqliteDataStore":
        return SqliteDataStore(db_path=db_path)

    def _conn_ro(self) -> sqlite3.Connection:
        return _connect_ro(self.db_path)

    def list_system_ids(self) -> List[str]:
        conn = self._conn_ro()
        try:
            rows = conn.execute("SELECT system_id FROM systems ORDER BY system_id").fetchall()
            return [str(r["system_id"]) for r in rows]
        finally:
            conn.close()

    def list_devices(self, system_id: str) -> List[str]:
        return []

    def fetch_system_timeseries(self, system_id: str, lookback_days: int) -> Dict[str, Any]:
        if lookback_days <= 0:
            raise ValueError("lookback_days must be > 0")
        conn = self._conn_ro()
        try:
            rows = conn.execute(
                "SELECT date, value FROM system_points WHERE system_id=? ORDER BY date DESC LIMIT ?",
                (system_id, int(lookback_days)),
            ).fetchall()
            if not rows:
                return {"system_id": system_id, "t": None, "values": []}
            values = [{"date": str(r["date"]), "v": float(r["value"])} for r in reversed(rows)]
            t = values[-1]["date"]
            return {"system_id": system_id, "t": t, "values": values}
        finally:
            conn.close()

    def fetch_system_timeseries_frame(self, system_id: str, lookback_days: int, end_offset_days: int) -> Dict[str, Any]:
        if lookback_days <= 0:
            raise ValueError("lookback_days must be > 0")
        if end_offset_days < 0:
            raise ValueError("end_offset_days must be >= 0")
        conn = self._conn_ro()
        try:
            t0_row = conn.execute(
                "SELECT MAX(date) AS t0 FROM system_points WHERE system_id=?",
                (system_id,),
            ).fetchone()
            if not t0_row or not t0_row["t0"]:
                return {"system_id": system_id, "t": None, "values": []}
            t0 = _parse_date(str(t0_row["t0"]))
            t = _format_date(t0 - dt.timedelta(days=int(end_offset_days)))
            rows = conn.execute(
                "SELECT date, value FROM system_points WHERE system_id=? AND date<=? ORDER BY date DESC LIMIT ?",
                (system_id, t, int(lookback_days)),
            ).fetchall()
            values = [{"date": str(r["date"]), "v": float(r["value"])} for r in reversed(rows)]
            return {"system_id": system_id, "t": t if values else None, "values": values}
        finally:
            conn.close()

    def fetch_device_timeseries(self, system_id: str, device_id: str, lookback_days: int) -> Dict[str, Any]:
        return {"system_id": system_id, "device_id": device_id, "t": None, "values": []}

    def fetch_device_timeseries_frame(
        self, system_id: str, device_id: str, lookback_days: int, end_offset_days: int
    ) -> Dict[str, Any]:
        return {"system_id": system_id, "device_id": device_id, "t": None, "values": []}


def sql_query_readonly(
    db_path: Path,
    sql: str,
    params: Any = None,
    *,
    max_rows: int = 200,
) -> Dict[str, Any]:
    s = (sql or "").strip()
    if not s:
        raise ValueError("sql is required")
    parts = [p.strip() for p in s.split(";") if p.strip()]
    if len(parts) != 1:
        raise ValueError("only a single statement is allowed")
    s = parts[0]
    head = s.lstrip().lower()
    if not (head.startswith("select") or head.startswith("with")):
        raise ValueError("only SELECT/WITH queries are allowed")

    max_rows_i = int(max_rows)
    if max_rows_i <= 0:
        max_rows_i = 200
    if max_rows_i > 1000:
        max_rows_i = 1000
    wrapped = f"SELECT * FROM ({s}) LIMIT {max_rows_i}"

    conn = _connect_ro(db_path)
    try:
        cur = conn.execute(wrapped, params or ())
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description] if cur.description else []
        out_rows: List[List[Any]] = []
        for r in rows:
            out_rows.append([r[c] for c in cols])
        return {"columns": cols, "rows": out_rows, "row_count": len(out_rows), "max_rows": max_rows_i}
    finally:
        conn.close()

