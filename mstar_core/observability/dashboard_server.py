"""
MSTAR Pro v4.0 - Dashboard HTTP Server
stdlib http.server + JSON API, no external dependencies
"""

from __future__ import annotations
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from functools import partial

from mstar_core.memory.forgetting import evaluate_all_forgetting

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HERMES_HOME = os.environ.get(
    "HERMES_HOME",
    os.path.join(os.path.expanduser("~"), "AppData", "Local", "hermes"),
)
AGENT_DIR = "C:/Users/41228/AppData/Local/hermes/hermes-agent"
PORT = 18792

# Global MSTARCore reference (settable for in-process operation)
_mstar_core_ref: "MSTARCore|None" = None

def set_mstar_core(core):
    """Set the global MSTARCore reference for in-process API calls."""
    global _mstar_core_ref
    _mstar_core_ref = core

def get_mstar_core():
    """Get MSTARCore instance, loading lazily if needed."""
    global _mstar_core_ref
    if _mstar_core_ref is not None:
        return _mstar_core_ref
    # Lazy load: try to create from same hermes_home as the running agent
    try:
        import sys
        agent_dir = Path(__file__).parent.parent
        if agent_dir not in sys.path:
            sys.path.insert(0, str(agent_dir))
        from mstar_core import MSTARCore
        _mstar_core_ref = MSTARCore(
            hermes_home=str(Path(HERMES_HOME).expanduser()),
            mode="balanced",
            fitness_dimensions=20,
            dashboard_enabled=False,
        )
        return _mstar_core_ref
    except Exception as e:
        return None

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def get_db_path() -> str:
    # Single source of truth: HERMES_HOME/mstar_fitness.db
    return os.path.join(HERMES_HOME, "mstar_fitness.db")


def get_obs_dir() -> Path:
    candidates = [
        Path(AGENT_DIR) / "mstar_observability",
        Path(HERMES_HOME) / "mstar_observability",
    ]
    for d in candidates:
        if d.exists():
            return d
    return candidates[0]


def row_to_dict(row) -> dict:
    if row is None:
        return None
    return dict(zip(row.keys(), row))


def get_db():
    conn = sqlite3.connect(get_db_path(), timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# API handlers (return dict or list, not Response)
# ---------------------------------------------------------------------------

def handle_statistics() -> dict:
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) as cnt FROM programs")
        programs_tracked = cur.fetchone()["cnt"]

        cur.execute("SELECT COUNT(*) as cnt FROM fitness_snapshots")
        snapshots = cur.fetchone()["cnt"]

        cur.execute("SELECT AVG(fitness_score) as avg_fs FROM programs WHERE fitness_score > 0")
        row = cur.fetchone()
        avg_fitness = float(row["avg_fs"]) if row and row["avg_fs"] else 0.0

        # Derive evolution counters from fitness_snapshots (the authoritative store)
        cur.execute("SELECT COUNT(*) as cnt FROM fitness_snapshots")
        snapshots = cur.fetchone()["cnt"]
        cur.execute("SELECT COUNT(DISTINCT program_id) as cnt FROM fitness_snapshots")
        mutations = cur.fetchone()["cnt"]

        # Count archived programs (lifecycle_status = 'archived')
        cur.execute("SELECT COUNT(*) as cnt FROM programs WHERE lifecycle_status = 'archived'")
        total_archives = cur.fetchone()["cnt"]

        # Count deleted programs
        cur.execute("SELECT COUNT(*) as cnt FROM programs WHERE lifecycle_status = 'deleted'")
        total_deletions = cur.fetchone()["cnt"]

        # Read authoritative counters from evolution_state table (single source of truth)
        cur.execute("SELECT value FROM evolution_state WHERE key = 'sessions_processed'")
        row = cur.fetchone()
        sessions_processed = int(row["value"]) if row else 0

        cur.execute("SELECT value FROM evolution_state WHERE key = 'evolutions_completed'")
        row = cur.fetchone()
        evolutions_completed = int(row["value"]) if row else 0

        return {
            "sessions_processed": sessions_processed,
            "evolutions_triggered": evolutions_completed,
            "mutations_applied": mutations,
            "programs_tracked": programs_tracked,
            "total_archives": total_archives,
            "total_deletions": total_deletions,
            "avg_fitness": round(avg_fitness, 4),
            "snapshots_total": snapshots,
        }
    finally:
        conn.close()


def handle_programs() -> dict:
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT program_id, name, lineage_depth, parent_id,
                   created_at, last_evolution_at, fitness_score
            FROM programs ORDER BY fitness_score DESC
        """)
        rows = cur.fetchall()
        result = []
        for row in rows:
            d = row_to_dict(row)
            d["fitness_history"] = []
            d["explanation_cache"] = {}
            result.append(d)
        return {"programs": result, "total": len(result)}
    finally:
        conn.close()


def handle_program(program_id: str) -> dict:
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM programs WHERE program_id = ?", (program_id,))
        row = cur.fetchone()
        if not row:
            return {"error": "Program not found"}
        d = row_to_dict(row)
        try:
            d["fitness_history"] = json.loads(row["fitness_history"]) if row["fitness_history"] else []
        except Exception:
            d["fitness_history"] = []
        try:
            d["explanation_cache"] = json.loads(row["explanation_cache"]) if row["explanation_cache"] else {}
        except Exception:
            d["explanation_cache"] = {}
        return d
    finally:
        conn.close()


def handle_fitness(program_id: str = None) -> dict:
    conn = get_db()
    try:
        cur = conn.cursor()
        if program_id:
            cur.execute("""
                SELECT snapshot_id, program_id, program_name, timestamp,
                       fitness_score, success_rate, quality_score,
                       ema_10, ema_50, trend_slope, decision_explanation
                FROM fitness_snapshots
                WHERE program_id = ? ORDER BY timestamp DESC LIMIT 200
            """, (program_id,))
            rows = cur.fetchall()
            return {"snapshots": [row_to_dict(r) for r in rows], "total": len(rows)}
        else:
            cur.execute("""
                SELECT p.program_id, p.name, p.fitness_score,
                       p.lineage_depth, p.created_at,
                       (SELECT COUNT(*) FROM fitness_snapshots s
                        WHERE s.program_id = p.program_id) as snapshot_count
                FROM programs p ORDER BY p.fitness_score DESC
            """)
            rows = cur.fetchall()
            return {"fitness": [row_to_dict(r) for r in rows]}
    finally:
        conn.close()


def handle_lineage(program_id: str) -> dict:
    lineage = []
    current_id = program_id
    conn = get_db()
    try:
        cur = conn.cursor()
        for _ in range(20):
            cur.execute("SELECT * FROM programs WHERE program_id = ?", (current_id,))
            row = cur.fetchone()
            if not row:
                break
            d = row_to_dict(row)
            lineage.append(d)
            current_id = d.get("parent_id")
            if not current_id:
                break
        return {"lineage": lineage, "depth": len(lineage)}
    finally:
        conn.close()


def handle_evolutions() -> dict:
    obs_dir = get_obs_dir()
    events = []
    if obs_dir.exists():
        # Match both "events.json" (single, written by dashboard.py)
        # and "events_*.json" (legacy/multiple files)
        for f in sorted(obs_dir.glob("events.json"), key=lambda x: x.name, reverse=True):
            try:
                data = json.loads(f.read_text())
                events.extend(data if isinstance(data, list) else [data])
            except Exception:
                pass
        for f in sorted(obs_dir.glob("events_*.json"), key=lambda x: x.name, reverse=True)[:10]:
            try:
                data = json.loads(f.read_text())
                events.extend(data if isinstance(data, list) else [data])
            except Exception:
                pass

    conn = get_db()
    db_events = []
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT snapshot_id, program_id, program_name, timestamp,
                   fitness_score, trend_slope, decision_explanation
            FROM fitness_snapshots ORDER BY timestamp DESC LIMIT 50
        """)
        db_events = [row_to_dict(r) for r in cur.fetchall()]
    finally:
        conn.close()

    return {"events": events[:50], "snapshots": db_events, "total": min(len(events), 50) + len(db_events)}


def handle_health() -> dict:
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


def handle_correction(program_id: str = None, correction_type: str = None,
                       adjustment: float = None, reason: str = None,
                       mutation_data: dict = None, strategy: str = None) -> dict:
    """
    Process an external correction signal via SelfImprovingBridge.
    Supports three correction types:
    - fitness_adjustment: direct fitness score adjustment
    - mutation_suggestion: suggest a mutation for a program
    - strategy_change: change evolution strategy for a program
    """
    mc = get_mstar_core()
    if not mc or not hasattr(mc, '_self_improver') or not mc._self_improver:
        return {"error": "SelfImprovingBridge not available"}

    from mstar_core.bridge.self_improving import CorrectionSignal
    signal = CorrectionSignal(
        source="dashboard_api",
        correction_type=correction_type or "fitness_adjustment",
        target_program_id=program_id or "",
        data={
            "adjustment": adjustment or 0,
            "reason": reason or "Dashboard correction",
            "mutation_data": mutation_data or {},
            "strategy": strategy or "",
        },
        timestamp=datetime.now().isoformat(),
    )

    try:
        mc._self_improver.receive_correction(signal)
        return {
            "success": True,
            "message": f"Correction '{correction_type}' processed for '{program_id}'",
            "correction_type": correction_type,
            "program_id": program_id,
        }
    except Exception as e:
        return {"error": str(e)}


def handle_forgetting() -> dict:
    """Trigger the forgetting mechanism and return decisions for all programs."""
    mc = get_mstar_core()
    if not mc or not hasattr(mc, 'forgetting_mechanism'):
        return {"error": "MSTARCore or ForgettingMechanism not available"}

    # Use the real ForgettingMechanism to evaluate all decisions
    decisions = evaluate_all_forgetting(mc)
    return {"decisions": decisions, "total": len(decisions)}


def handle_bridge_status() -> dict:
    """Get SelfImprovingBridge status."""
    mc = get_mstar_core()
    if not mc or not hasattr(mc, '_self_improver') or not mc._self_improver:
        return {"error": "SelfImprovingBridge not available"}
    return mc._self_improver.get_bridge_status()


def handle_trigger_evolution() -> dict:
    """Manually trigger an evolution cycle."""
    mc = get_mstar_core()
    if not mc:
        return {"error": "MSTARCore not available"}
    result = mc.evolution_engine.evaluate_session(
        session_id=f"dashboard_{datetime.now().timestamp()}",
        stats={}
    )
    return {"success": True, "result": result}


# ---------------------------------------------------------------------------
# HTML Dashboard (embedded, no file read needed)
# ---------------------------------------------------------------------------

HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>MSTAR Pro v4.0 Dashboard</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:Consolas,"Courier New",monospace;background:#0a0a14;color:#e0e0e0;padding:20px;min-height:100vh}
.header{text-align:center;color:#00d4ff;font-size:2.2em;margin-bottom:30px;text-shadow:0 0 20px rgba(0,212,255,.4)}
.header span{color:#ff6b9d}
.stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:15px;margin-bottom:30px}
.stat-card{background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:10px;padding:20px;text-align:center;border:1px solid #2a3a5e;transition:transform .2s}
.stat-card:hover{transform:translateY(-3px);border-color:#00d4ff}
.stat-value{font-size:2.5em;color:#00d4ff;font-weight:bold;text-shadow:0 0 10px rgba(0,212,255,.3)}
.stat-label{color:#7a8ba8;font-size:.85em;margin-top:8px}
.section{background:#12121f;border-radius:10px;padding:20px;margin-bottom:20px;border:1px solid #1e2a40}
.section h2{color:#ff6b9d;margin-bottom:15px;font-size:1.1em}
.prog-row{display:flex;align-items:center;padding:12px;border-bottom:1px solid #1e2a40;gap:15px}
.prog-row:last-child{border-bottom:none}
.prog-name{flex:1;color:#e0e0e0;font-weight:bold}
.prog-fitness{font-size:1.2em;color:#00d4ff;min-width:80px;text-align:right}
.prog-depth{color:#7a8ba8;font-size:.8em}
.fitness-bar{height:6px;background:#1e2a40;border-radius:3px;overflow:hidden;margin-top:4px}
.fitness-fill{height:100%;background:linear-gradient(90deg,#00d4ff,#4ade80);border-radius:3px;transition:width .5s}
.loading{color:#7a8ba8;text-align:center;padding:40px}
.error-msg{color:#f87171;background:#1f1a2e;padding:15px;border-radius:8px;border:1px solid #f87171;margin:10px 0}
#programs-list{max-height:400px;overflow-y:auto}
#programs-list::-webkit-scrollbar{width:6px}
#programs-list::-webkit-scrollbar-thumb{background:#2a3a5e;border-radius:3px}
@media(max-width:600px){.header{font-size:1.5em}.stats-grid{grid-template-columns:repeat(2,1fr)}}
</style>
</head>
<body>
<div class="header">MSTAR <span>Pro</span> v4.0 Dashboard</div>

<div class="stats-grid">
  <div class="stat-card"><div class="stat-value" id="s-evolutions">-</div><div class="stat-label">Evolutions</div></div>
  <div class="stat-card"><div class="stat-value" id="s-mutations">-</div><div class="stat-label">Mutations</div></div>
  <div class="stat-card"><div class="stat-value" id="s-programs">-</div><div class="stat-label">Programs</div></div>
  <div class="stat-card"><div class="stat-value" id="s-avg-fitness">-</div><div class="stat-label">Avg Fitness</div></div>
  <div class="stat-card"><div class="stat-value" id="s-archives">-</div><div class="stat-label">Archives</div></div>
  <div class="stat-card"><div class="stat-value" id="s-snapshots">-</div><div class="stat-label">Snapshots</div></div>
</div>

<div class="section">
  <h2>Programs</h2>
  <div id="programs-list"><div class="loading">Loading...</div></div>
</div>

<script>
const API = '/api';
async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(r.status + ' ' + r.statusText);
  return r.json();
}
async function load() {
  try {
    const [stats, progs] = await Promise.all([
      fetchJSON(API + '/statistics'),
      fetchJSON(API + '/programs')
    ]);
    document.getElementById('s-evolutions').textContent = stats.evolutions_triggered;
    document.getElementById('s-mutations').textContent = stats.mutations_applied;
    document.getElementById('s-programs').textContent = stats.programs_tracked;
    document.getElementById('s-avg-fitness').textContent = stats.avg_fitness.toFixed(3);
    document.getElementById('s-archives').textContent = stats.total_archives;
    document.getElementById('s-snapshots').textContent = stats.snapshots_total;

    const list = document.getElementById('programs-list');
    if (!progs.programs || progs.programs.length === 0) {
      list.innerHTML = '<div class="error-msg">No programs tracked yet</div>';
      return;
    }
    list.innerHTML = progs.programs.map(p => {
      const pct = Math.min(100, Math.max(0, p.fitness_score * 80)).toFixed(1);
      const depth = p.lineage_depth > 0 ? `gen ${p.lineage_depth}` : '';
      return `<div class="prog-row">
        <div class="prog-name">${p.name || p.program_id}</div>
        <div>
          <div class="prog-depth">${depth}</div>
          <div class="fitness-bar"><div class="fitness-fill" style="width:${pct}%"></div></div>
        </div>
        <div class="prog-fitness">${p.fitness_score.toFixed(3)}</div>
      </div>`;
    }).join('');
  } catch(e) {
    document.getElementById('programs-list').innerHTML =
      `<div class="error-msg">Failed to load: ${e.message}</div>`;
  }
}
load();
setInterval(load, 10000); // refresh every 10s
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------

class DashboardHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        pass  # suppress request logging

    def send_json(self, data, status=200):
        body = json.dumps(data, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, html, status=200):
        if isinstance(html, str):
            html = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(html))
        self.end_headers()
        self.wfile.write(html)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        # Route dispatch
        if path in ("/", "/dashboard.html"):
            self.send_html(HTML_DASHBOARD)

        elif path == "/api" or path == "/api/statistics":
            self.send_json(handle_statistics())

        elif path == "/api/programs":
            self.send_json(handle_programs())

        elif path.startswith("/api/programs/"):
            program_id = path[len("/api/programs/"):]
            result = handle_program(program_id)
            if "error" in result and result["error"] == "Program not found":
                self.send_json(result, 404)
            else:
                self.send_json(result)

        elif path == "/api/fitness":
            self.send_json(handle_fitness())

        elif path.startswith("/api/fitness/"):
            program_id = path[len("/api/fitness/"):]
            self.send_json(handle_fitness(program_id))

        elif path.startswith("/api/lineage/"):
            program_id = path[len("/api/lineage/"):]
            self.send_json(handle_lineage(program_id))

        elif path == "/api/evolutions":
            self.send_json(handle_evolutions())

        elif path == "/health":
            self.send_json(handle_health())

        elif path == "/api/correction":
            self.send_json({"error": "POST required"}, 405)

        elif path == "/api/bridge_status":
            self.send_json(handle_bridge_status())

        elif path == "/api/forgetting":
            self.send_json(handle_forgetting())

        elif path == "/api/evolution/trigger":
            self.send_json(handle_trigger_evolution())

        else:
            self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""

        import json as _json
        try:
            data = _json.loads(body) if body else {}
        except Exception:
            data = {}

        if path == "/api/correction":
            result = handle_correction(
                program_id=data.get("program_id"),
                correction_type=data.get("correction_type"),
                adjustment=data.get("adjustment"),
                reason=data.get("reason"),
                mutation_data=data.get("mutation_data"),
                strategy=data.get("strategy"),
            )
            self.send_json(result)
        elif path == "/api/evolution/trigger":
            self.send_json(handle_trigger_evolution())
        else:
            self.send_json({"error": "Not found"}, 404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), DashboardHandler)
    print(f"[MSTAR Dashboard] http://localhost:{PORT}/dashboard.html")
    print(f"[MSTAR Dashboard] hermes_home={HERMES_HOME}")
    print(f"[MSTAR Dashboard] agent_dir={AGENT_DIR}")
    print(f"[MSTAR Dashboard] db_path={get_db_path()}")
    server.serve_forever()
