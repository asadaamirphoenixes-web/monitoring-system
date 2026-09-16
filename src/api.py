"""
KT-Radar :: control-room API
----------------------------
SQLite by default so it runs on a laptop for a demo; swap the DSN for
TimescaleDB/Postgres when you go multi-site. Endpoints:

  POST /ingest/event         edge node pushes count/state/violation events
  GET  /api/sites            live state of every camera
  GET  /api/flow             flow + speed timeseries for a site
  GET  /api/violations       filtered violation list (hashed plates)
  GET  /api/hotspots         ranked intersections by congestion-hours
  GET  /api/journey          corridor travel time from re-identified tokens
  WS   /ws/live              push feed for the dashboard

Run:  uvicorn src.api:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, model_validator

DB = Path("data/ktradar.db")
DB.parent.mkdir(parents=True, exist_ok=True)

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL NOT NULL,
  site TEXT NOT NULL,
  type TEXT NOT NULL,
  class TEXT,
  line TEXT,
  rule TEXT,
  speed_kmh REAL,
  plate_token TEXT,
  confidence REAL,
  enforceable INTEGER DEFAULT 0,
  payload TEXT
);
CREATE INDEX IF NOT EXISTS ix_events_site_ts ON events(site, ts);
CREATE INDEX IF NOT EXISTS ix_events_type_ts ON events(type, ts);
CREATE INDEX IF NOT EXISTS ix_events_token   ON events(plate_token);

CREATE TABLE IF NOT EXISTS site_state (
  site TEXT PRIMARY KEY,
  ts REAL, lat REAL, lon REAL, name TEXT,
  vehicles INTEGER, mean_speed REAL, density REAL, los TEXT, congested INTEGER
);
"""


def db():
    c = sqlite3.connect(DB, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


with closing(db()) as c:
    c.executescript(SCHEMA)
    c.commit()

app = FastAPI(title="KT-Radar Control Room")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

_clients: List[WebSocket] = []


_EVENT_ALIASES = {"class": "cls_name", "plate": "plate_token"}


class Event(BaseModel):
    """Edge nodes (src/pipeline.py) emit flat dicts - "class" (not
    cls_name, a reserved word) and "plate" (not plate_token) at the top
    level, plus rule/state-specific fields like vehicles_in_zone,
    density_veh_per_m2, los, congested, track_id, detail. Normalise all
    of that here rather than force every producer to match this shape:
    known aliases get renamed, everything else not already a field on
    this model is folded into `payload`, where every consumer below
    (site_state upsert, hotspots, journey) expects to find it.
    """

    ts: float
    site: str
    type: str
    cls_name: Optional[str] = None
    line: Optional[str] = None
    rule: Optional[str] = None
    speed_kmh: Optional[float] = None
    plate_token: Optional[str] = None
    confidence: Optional[float] = None
    enforceable: bool = False
    payload: Dict[str, Any] = {}

    @model_validator(mode="before")
    @classmethod
    def _normalise(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        for src, dst in _EVENT_ALIASES.items():
            if src in data and dst not in data:
                data[dst] = data.pop(src)
        known = set(cls.model_fields)
        payload = dict(data.get("payload") or {})
        for k in list(data.keys()):
            if k not in known:
                payload[k] = data.pop(k)
        data["payload"] = payload
        return data


async def _broadcast(msg: dict):
    dead = []
    for ws in _clients:
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)
    for d in dead:
        _clients.remove(d)


@app.post("/ingest/event")
async def ingest(e: Event):
    with closing(db()) as c:
        c.execute(
            """INSERT INTO events
               (ts,site,type,class,line,rule,speed_kmh,plate_token,confidence,
                enforceable,payload)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (e.ts, e.site, e.type, e.cls_name, e.line, e.rule, e.speed_kmh,
             e.plate_token, e.confidence, int(e.enforceable),
             json.dumps(e.payload)),
        )
        if e.type == "state":
            p = e.payload
            c.execute(
                """INSERT INTO site_state(site,ts,vehicles,mean_speed,density,los,congested)
                   VALUES(?,?,?,?,?,?,?)
                   ON CONFLICT(site) DO UPDATE SET
                     ts=excluded.ts, vehicles=excluded.vehicles,
                     mean_speed=excluded.mean_speed, density=excluded.density,
                     los=excluded.los, congested=excluded.congested""",
                (e.site, e.ts, p.get("vehicles_in_zone"), p.get("mean_speed_kmh"),
                 p.get("density_veh_per_m2"), p.get("los"),
                 int(bool(p.get("congested")))),
            )
        c.commit()
    await _broadcast(e.model_dump())
    return {"ok": True}


@app.get("/api/sites")
def sites():
    with closing(db()) as c:
        return [dict(r) for r in c.execute("SELECT * FROM site_state")]


@app.get("/api/flow")
def flow(site: str, hours: int = 6, bucket_min: int = 5):
    since = time.time() - hours * 3600
    b = bucket_min * 60
    with closing(db()) as c:
        rows = c.execute(
            """SELECT CAST(ts/? AS INT)*? AS t, class,
                      COUNT(*) AS n, AVG(speed_kmh) AS v
               FROM events
               WHERE site=? AND type='count' AND ts>=?
               GROUP BY t, class ORDER BY t""",
            (b, b, site, since)).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/violations")
def violations(site: Optional[str] = None, rule: Optional[str] = None,
               enforceable_only: bool = False, limit: int = 200):
    q = "SELECT * FROM events WHERE type='violation'"
    args: list = []
    if site:
        q += " AND site=?"
        args.append(site)
    if rule:
        q += " AND rule=?"
        args.append(rule)
    if enforceable_only:
        q += " AND enforceable=1"
    q += " ORDER BY ts DESC LIMIT ?"
    args.append(limit)
    with closing(db()) as c:
        return [dict(r) for r in c.execute(q, args)]


@app.get("/api/hotspots")
def hotspots(hours: int = 168):
    """Congestion-hours per site: the metric a city planner can actually act on."""
    since = time.time() - hours * 3600
    with closing(db()) as c:
        rows = c.execute(
            """SELECT site,
                      SUM(CASE WHEN json_extract(payload,'$.congested')=1 THEN 1 ELSE 0 END)
                        * 5.0/3600 AS congestion_hours,
                      AVG(json_extract(payload,'$.mean_speed_kmh')) AS avg_speed
               FROM events WHERE type='state' AND ts>=?
               GROUP BY site ORDER BY congestion_hours DESC""", (since,)).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/journey")
def journey(origin: str, destination: str, hours: int = 3):
    """Corridor travel time by matching hashed plate tokens across two cameras.
    No identity is involved - the token is an HMAC, unresolvable without the key."""
    since = time.time() - hours * 3600
    with closing(db()) as c:
        rows = c.execute(
            """SELECT a.plate_token, MIN(a.ts) o_ts, MIN(b.ts) d_ts
               FROM events a JOIN events b ON a.plate_token=b.plate_token
               WHERE a.site=? AND b.site=? AND a.ts>=? AND b.ts>a.ts
                 AND a.plate_token IS NOT NULL
               GROUP BY a.plate_token""", (origin, destination, since)).fetchall()
    times = [r["d_ts"] - r["o_ts"] for r in rows if 30 < (r["d_ts"] - r["o_ts"]) < 5400]
    if not times:
        return {"n": 0}
    times.sort()
    return {
        "n": len(times),
        "median_s": times[len(times)//2],
        "p85_s": times[int(len(times)*0.85)],
        "free_flow_s": times[int(len(times)*0.05)],
        "delay_index": round(times[len(times)//2] / max(times[int(len(times)*0.05)], 1), 2),
    }


@app.websocket("/ws/live")
async def live(ws: WebSocket):
    await ws.accept()
    _clients.append(ws)
    try:
        while True:
            await asyncio.sleep(30)
            await ws.send_json({"type": "ping", "ts": time.time()})
    except WebSocketDisconnect:
        if ws in _clients:
            _clients.remove(ws)
