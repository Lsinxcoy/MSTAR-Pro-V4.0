"""
MSTAR Pro v4.0 - 完整可观测性仪表盘
"""

from __future__ import annotations
import logging
import json
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DashboardEvent:
    event_id: str
    event_type: str
    program_id: str
    timestamp: str
    data: Dict


@dataclass
class FitnessSnapshot:
    snapshot_id: str
    program_id: str
    program_name: str
    timestamp: str
    fitness_score: float
    dimensions: Dict[str, float]
    explanation: str = ""


_dashboard_instance = None
_dashboard_lock = threading.RLock()


def get_dashboard(hermes_home: Optional[str] = None):
    global _dashboard_instance
    with _dashboard_lock:
        if _dashboard_instance is None:
            _dashboard_instance = ObservabilityDashboard(hermes_home)
        return _dashboard_instance


class ObservabilityDashboard:
    def __init__(self, hermes_home: Optional[str] = None):
        import os
        self.hermes_home = hermes_home or os.path.expanduser("~/.hermes")
        self.data_dir = Path(self.hermes_home) / "mstar_observability"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._events: List[DashboardEvent] = []
        self._snapshots: List[FitnessSnapshot] = []
        self._lineage_graph: Dict[str, List[str]] = {}
        self._stats = {
            'total_evolutions': 0, 'total_mutations': 0,
            'total_archives': 0, 'total_deletions': 0,
            'avg_fitness_improvement': 0.0,
        }
        self._lock = threading.RLock()
        self._save_interval = 5  # flush to disk every 5s (background), critical writes sync immediately
        self._last_save_time = datetime.now()
        self._server_port = 18792

    def record_evolution_event(self, event):
        with self._lock:
            dashboard_event = DashboardEvent(
                event_id=getattr(event, 'event_id', f"evo_{len(self._events)}"),
                event_type=getattr(event, 'event_type', 'unknown'),
                program_id=getattr(event, 'program_id', ''),
                timestamp=getattr(event, 'timestamp', datetime.now().isoformat()),
                data={
                    'fitness_before': getattr(event, 'fitness_before', 0),
                    'fitness_after': getattr(event, 'fitness_after', 0),
                    'fitness_delta': getattr(event, 'fitness_delta', 0),
                    'reason': getattr(event, 'reason', ''),
                    'decision_explanation': getattr(event, 'decision_explanation', ''),
                    'details': getattr(event, 'details', {}),
                },
            )
            self._events.append(dashboard_event)
            self._stats['total_evolutions'] += 1
            if dashboard_event.event_type == 'mutation':
                self._stats['total_mutations'] += 1
            elif dashboard_event.event_type == 'archive':
                self._stats['total_archives'] += 1
            elif dashboard_event.event_type == 'delete':
                self._stats['total_deletions'] += 1
            # IMMEDIATE sync write — don't wait for the 5s background interval
            self.save()

    def record_fitness_snapshot(self, snapshot: Dict):
        with self._lock:
            fitness_snapshot = FitnessSnapshot(
                snapshot_id=f"snap_{len(self._snapshots)}",
                program_id=snapshot.get('program_id', ''),
                program_name=snapshot.get('program_name', ''),
                timestamp=snapshot.get('timestamp', datetime.now().isoformat()),
                fitness_score=snapshot.get('fitness_score', 0.5),
                dimensions=snapshot.get('dimensions', {}),
                explanation=snapshot.get('explanation', ''),
            )
            self._snapshots.append(fitness_snapshot)
            if len(self._snapshots) > 10000:
                self._snapshots = self._snapshots[-5000:]

    def _auto_save_if_needed(self):
        now = datetime.now()
        if (now - self._last_save_time).total_seconds() > self._save_interval:
            self.save()
            self._last_save_time = now

    def save(self):
        with self._lock:
            # Persist evolution events to SQLite so the HTTP server sees them immediately
            db_path = Path(self.hermes_home) / "mstar_fitness.db"
            try:
                import sqlite3
                conn = sqlite3.connect(str(db_path), timeout=30)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")

                # Ensure fitness_snapshots table exists
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS fitness_snapshots (
                        snapshot_id TEXT PRIMARY KEY,
                        program_id TEXT NOT NULL,
                        program_name TEXT,
                        timestamp TEXT NOT NULL,
                        fitness_score REAL,
                        success_rate REAL,
                        quality_score REAL,
                        ema_10 REAL,
                        ema_50 REAL,
                        trend_slope REAL,
                        decision_explanation TEXT
                    )
                """)

                for event in self._events[-1000:]:
                    # Upsert into fitness_snapshots using event_id as snapshot_id
                    delta = event.data.get('fitness_delta', 0)
                    fitness_before = event.data.get('fitness_before', 0)
                    fitness_after = event.data.get('fitness_after', fitness_before + delta)
                    conn.execute("""
                        INSERT OR REPLACE INTO fitness_snapshots
                        (snapshot_id, program_id, program_name, timestamp,
                         fitness_score, success_rate, quality_score,
                         ema_10, ema_50, trend_slope, decision_explanation)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        event.event_id,
                        event.program_id,
                        event.program_id,  # program_name
                        event.timestamp,
                        fitness_after,
                        1.0 if delta >= 0 else 0.0,  # success_rate proxy
                        max(0.0, min(1.0, 0.5 + delta / 2)) if delta != 0 else 0.5,  # quality proxy
                        fitness_after,  # ema_10 proxy
                        fitness_before,  # ema_50 proxy
                        delta,  # trend_slope proxy
                        event.data.get('decision_explanation', ''),
                    ))
                conn.commit()
                conn.close()
            except Exception as e:
                logger.warning(f"[MSTAR Dashboard] SQLite write failed: {e}")

            # Also write JSON for human-readability
            events_path = self.data_dir / "events.json"
            with open(events_path, 'w', encoding='utf-8') as f:
                json.dump([
                    {'event_id': e.event_id, 'event_type': e.event_type,
                     'program_id': e.program_id, 'timestamp': e.timestamp, 'data': e.data}
                    for e in self._events[-1000:]
                ], f, indent=2, ensure_ascii=False)
            snapshots_path = self.data_dir / "snapshots.json"
            with open(snapshots_path, 'w', encoding='utf-8') as f:
                json.dump([
                    {'snapshot_id': s.snapshot_id, 'program_id': s.program_id,
                     'program_name': s.program_name, 'timestamp': s.timestamp,
                     'fitness_score': s.fitness_score, 'dimensions': s.dimensions,
                     'explanation': s.explanation}
                    for s in self._snapshots[-1000:]
                ], f, indent=2, ensure_ascii=False)
            stats_path = self.data_dir / "stats.json"
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(self._stats, f, indent=2, ensure_ascii=False)
            logger.info(f"[MSTAR Dashboard] Saved {len(self._events)} events")

    def generate_html_dashboard(self, output_path: Optional[str] = None) -> str:
        import os
        output_path = output_path or str(self.data_dir / "dashboard.html")
        events_data = []
        for e in self._events[-100:]:
            events_data.append({
                'event_id': e.event_id,
                'event_type': e.event_type,
                'program_id': e.program_id,
                'timestamp': e.timestamp,
                'fitness_delta': e.data.get('fitness_delta', 0),
                'decision_explanation': e.data.get('decision_explanation', ''),
            })
        events_json = json.dumps(events_data, ensure_ascii=False)
        stats_json = json.dumps(self._stats, ensure_ascii=False)
        html = '<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><title>MSTAR Pro v4.0 Dashboard</title>'
        html += '<style>body{font-family:Consolas,monospace;background:#0a0a14;color:#e0e0e0;padding:20px}'
        html += '.header{text-align:center;color:#00d4ff;font-size:2em;margin-bottom:30px}'
        html += '.stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:15px;margin-bottom:30px}'
        html += '.stat-card{background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:8px;padding:15px;text-align:center;border:1px solid #334}'
        html += '.stat-value{font-size:2em;color:#00d4ff;font-weight:bold}'
        html += '.stat-label{color:#888;font-size:0.85em;margin-top:5px}'
        html += '.timeline{background:#12121f;border-radius:8px;padding:20px}'
        html += '.event{background:#1f2a40;border-radius:6px;padding:12px;margin:10px 0;border-left:3px solid #00d4ff}'
        html += '.event-type{color:#ff6b9d;font-weight:bold}'
        html += '.event-time{color:#888;font-size:0.85em}'
        html += '.event-explanation{color:#aaa;margin-top:8px;font-size:0.9em}'
        html += '.fitness-delta{color:#4ade80}'
        html += '.fitness-delta.negative{color:#f87171}</style></head>'
        html += '<body><div class="header">MSTAR Pro v4.0 Dashboard</div>'
        html += '<div class="stats-grid">'
        html += '<div class="stat-card"><div class="stat-value">' + str(self._stats.get('total_evolutions', 0)) + '</div><div class="stat-label">Total Evolutions</div></div>'
        html += '<div class="stat-card"><div class="stat-value">' + str(self._stats.get('total_mutations', 0)) + '</div><div class="stat-label">Mutations</div></div>'
        html += '<div class="stat-card"><div class="stat-value">' + str(self._stats.get('total_archives', 0)) + '</div><div class="stat-label">Archives</div></div>'
        html += '<div class="stat-card"><div class="stat-value">' + str(self._stats.get('total_deletions', 0)) + '</div><div class="stat-label">Deletions</div></div>'
        html += '</div><div class="timeline"><h2 style="color:#ff6b9d;margin-bottom:15px">Evolution Timeline</h2><div id="events"></div></div>'
        html += '<script>var events=' + events_json + ';var stats=' + stats_json + ';'
        html += 'var eventsDiv=document.getElementById("events");'
        html += 'for(var i=0;i<events.length;i++){var e=events[i];var delta=e.fitness_delta||0;'
        html += 'var deltaClass=delta>=0?"fitness-delta":"fitness-delta negative";'
        html += 'var div=document.createElement("div");div.className="event";'
        html += "div.innerHTML='<span class=\"event-type\">'+e.event_type+'</span>'"
        html += "+'<span class=\"event-time\">'+new Date(e.timestamp).toLocaleString()+'</span>'"
        html += "+'<div>Program: '+e.program_id+'</div>'"
        html += "+'<div class=\"'+deltaClass+'\">Fitness Delta: '+(delta>0?'+':'')+delta.toFixed(3)+'</div>'"
        html += "+'<div class=\"event-explanation\">'+(e.decision_explanation||'No explanation')+'</div>';"
        html += 'eventsDiv.appendChild(div);}</script></body></html>'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        logger.info(f"[MSTAR Dashboard] Generated HTML at {output_path}")
        return output_path

    def get_lineage_graph(self) -> Dict[str, List[str]]:
        return self._lineage_graph

    def get_statistics(self) -> Dict:
        # Read live counters from SQLite so external callers (e.g. mstar_* tools via Hermes)
        # always see up-to-date values even if this Dashboard instance is a different process
        try:
            import sqlite3
            db_path = Path(self.hermes_home) / "mstar_fitness.db"
            conn = sqlite3.connect(str(db_path), timeout=30)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM programs")
            programs_tracked = cur.fetchone()[0] if cur.fetchone()[0] else 0
            cur.execute("SELECT AVG(fitness_score) FROM programs")
            avg = cur.fetchone()[0] or 0.0
            cur.execute("SELECT COUNT(*) FROM fitness_snapshots")
            snapshots = cur.fetchone()[0] if cur.fetchone()[0] else 0
            # Derive evolutions_triggered from fitness_snapshots (one snapshot per evolution event)
            evolutions_triggered = snapshots
            cur.execute("SELECT COUNT(DISTINCT program_id) FROM fitness_snapshots")
            mutations_applied = cur.fetchone()[0] if cur.fetchone()[0] else 0
            conn.close()
            result = dict(self._stats)
            result['programs_tracked'] = programs_tracked
            result['avg_fitness'] = avg
            result['snapshots_total'] = snapshots
            result['evolutions_triggered'] = evolutions_triggered
            result['mutations_applied'] = mutations_applied
            return result
        except Exception:
            return dict(self._stats)
