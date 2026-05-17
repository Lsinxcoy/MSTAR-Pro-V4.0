"""
MSTAR Pro 自进化引擎工具
将 mstar_core 的所有功能通过 Hermes Agent 工具系统暴露
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

HERMES_HOME = os.getenv("HERMES_HOME", os.path.join(os.path.expanduser("~"), ".hermes"))

# Module-level singleton: reused across all tool calls in the same process
_mstar_core_cache: "MSTARCore|None" = None


def _get_mstar_core():
    """延迟导入并返回 MSTARCore 单例（进程内共享）"""
    global _mstar_core_cache
    if _mstar_core_cache is None:
        from mstar_core import MSTARCore
        # Read mode/dimensions from config.yaml so tools match agent's actual config
        _cfg = {}
        try:
            import yaml
            cfg_path = Path(HERMES_HOME) / "config.yaml"
            if cfg_path.exists():
                _cfg = (yaml.safe_load(cfg_path.read_text()) or {}).get("mstar", {})
        except Exception:
            pass
        _mode = _cfg.get("mode", "balanced")
        _dims = _cfg.get("fitness_dimensions", 20)
        _mstar_core_cache = MSTARCore(
            hermes_home=HERMES_HOME,
            mode=_mode,
            fitness_dimensions=_dims,
            dashboard_enabled=True,
        )
    return _mstar_core_cache


def mstar(action: str = "", program_id: str = "", detailed: bool = False,
          mode: str = "", min_score: float = 0.0, limit: int = 20,
          generations: int = 10, **kwargs) -> str:
    """
    MSTAR Pro 自进化引擎统一入口
    
    Actions:
      - trigger_evolution: 手动触发一次进化周期
      - get_all_fitness: 获取所有追踪程序的适应度排名
      - get_fitness: 获取指定程序的详细适应度
      - get_lineage: 获取程序的进化血缘链
      - get_statistics: 获取 MSTAR Pro 运行统计
      - get_dashboard_url: 获取 Dashboard URL
      - set_mode: 设置运行模式 (beginner/balanced/advanced)
      - get_config: 获取当前配置
      - forgetting_decision: 获取程序的遗忘决策
      - record_session: 记录一个 session 结果
      - run_ablation: 运行消融实验
    """
    try:
        mc = _get_mstar_core()
        normalized = action.lower().strip()

        # ---- trigger_evolution ----
        if normalized == "trigger_evolution":
            result = mc.evolution_engine.evaluate_session(
                session_id=f"manual_{datetime.now().timestamp()}",
                stats={}
            )
            # Bug-3 fix: convert MemoryProgram objects to serializable dicts
            if result.get('triggered') and 'program_id' not in result:
                result['program_id'] = result.get('program', {}).get('program_id') if isinstance(result.get('program'), dict) else str(result.get('program'))
            if 'program' in result:
                prog = result.pop('program')
                if hasattr(prog, 'program_id'):
                    result['program_id'] = prog.program_id
                    result['program_name'] = getattr(prog, 'name', '')
                    result['fitness_before'] = getattr(prog, 'fitness_score', None)
            if 'events' in result:
                for ev in result['events']:
                    if 'program' in ev:
                        prog = ev.pop('program')
                        if hasattr(prog, 'program_id'):
                            ev['program_id'] = prog.program_id
            return json.dumps({
                "success": True,
                "action": "trigger_evolution",
                "result": result,
            }, indent=2, ensure_ascii=False)

        # ---- get_all_fitness ----
        if normalized == "get_all_fitness":
            conn = mc.fitness_tracker.db_path
            import sqlite3
            db = sqlite3.connect(conn)
            cur = db.execute(
                "SELECT program_id, name, fitness_score, lineage_depth FROM programs ORDER BY fitness_score DESC LIMIT ?",
                (limit,)
            )
            rows = cur.fetchall()
            db.close()

            programs = []
            for row in rows:
                if row[2] < min_score:
                    continue
                p = {
                    "program_id": row[0],
                    "name": row[1],
                    "fitness_score": round(row[2], 4),
                    "lineage_depth": row[3],
                }
                if detailed:
                    p["details"] = "set()"
                programs.append(p)

            return json.dumps({
                "success": True,
                "action": "get_all_fitness",
                "programs": programs,
                "total": len(programs),
            }, indent=2, ensure_ascii=False)

        # ---- get_fitness ----
        if normalized == "get_fitness":
            if not program_id:
                return json.dumps({"success": False, "error": "program_id required"}, indent=2)

            conn = mc.fitness_tracker.db_path
            import sqlite3
            db = sqlite3.connect(conn)
            cur = db.execute(
                "SELECT program_id, name, fitness_score, lineage_depth, parent_id, created_at, last_evolution_at, fitness_history FROM programs WHERE program_id = ?",
                (program_id,)
            )
            row = cur.fetchone()
            db.close()

            if not row:
                return json.dumps({"success": False, "error": f"Program {program_id} not found"}, indent=2)

            result = {
                "program_id": row[0],
                "name": row[1],
                "fitness_score": round(row[2], 4),
                "lineage_depth": row[3],
                "parent_id": row[4],
                "created_at": row[5],
                "last_evolution_at": row[6],
            }
            if detailed:
                result["fitness_history"] = eval(row[7]) if row[7] else []

            return json.dumps({
                "success": True,
                "action": "get_fitness",
                "program": result,
            }, indent=2, ensure_ascii=False)

        # ---- get_lineage ----
        if normalized == "get_lineage":
            if not program_id:
                return json.dumps({"success": False, "error": "program_id required"}, indent=2)

            lineage = []
            current_id = program_id
            for _ in range(generations):
                conn = mc.fitness_tracker.db_path
                import sqlite3
                db = sqlite3.connect(conn)
                cur = db.execute(
                    "SELECT program_id, name, fitness_score, parent_id FROM programs WHERE program_id = ?",
                    (current_id,)
                )
                row = cur.fetchone()
                db.close()

                if not row or row[0] is None:
                    break
                lineage.append({
                    "program_id": row[0],
                    "name": row[1],
                    "fitness_score": round(row[2], 4) if row[2] else 0.0,
                    "parent_id": row[3],
                    "mutations_applied": 0,  # resolved below
                })
                current_id = row[3]
                if not current_id:
                    break

            # Resolve mutations_applied from fitness_history for each lineage entry
            try:
                import sqlite3 as _sql
                _conn = _sql.connect(mc.fitness_tracker.db_path)
                for _entry in lineage:
                    _cur = _conn.execute(
                        "SELECT fitness_history FROM programs WHERE program_id = ?",
                        (_entry["program_id"],),
                    )
                    _row = _cur.fetchone()
                    if _row and _row[0]:
                        try:
                            _hist = json.loads(_row[0])
                            _entry["mutations_applied"] = len(_hist) if isinstance(_hist, list) else 0
                        except Exception:
                            _entry["mutations_applied"] = 0
                    else:
                        _entry["mutations_applied"] = 0
                _conn.close()
            except Exception:
                pass  # leave mutations_applied=0 on any error

            return json.dumps({
                "success": True,
                "action": "get_lineage",
                "lineage": lineage,
            }, indent=2, ensure_ascii=False)

        # ---- get_statistics ----
        if normalized == "get_statistics":
            stats = mc.get_statistics()
            # Read evolutions_triggered from Dashboard API to get real-time count
            # (mstar_tools creates its own MSTARCore which doesn't share _evolutions_triggered with run_agent.py's instance)
            # Use curl to bypass corporate proxy (urllib times out on this host)
            _ev = stats.get("evolutions_triggered", 0)
            _sp = stats.get("sessions_processed", 0)
            import subprocess as _subprocess
            try:
                _curl = _subprocess.run(
                    'curl -s --max-time 10 http://localhost:18792/api/statistics',
                    shell=True, capture_output=True, text=True,
                )
                if _curl.returncode == 0 and _curl.stdout:
                    _body = json.loads(_curl.stdout)
                    _ev = _body.get("evolutions_triggered", _ev)
                    _sp = _body.get("sessions_processed", _sp)
            except Exception:
                pass
            return json.dumps({
                "success": True,
                "action": "get_statistics",
                "sessions_processed": _sp,
                "evolutions_triggered": _ev,
                "fitness_stats": stats.get("fitness_stats", {}),
            }, indent=2, ensure_ascii=False)

        # ---- get_dashboard_url ----
        if normalized == "get_dashboard_url":
            return json.dumps({
                "success": True,
                "action": "get_dashboard_url",
                "url": "http://localhost:18792",
                "api_url": "http://localhost:18792/api",
                "html_url": "http://localhost:18792/dashboard.html",
            }, indent=2, ensure_ascii=False)

        # ---- set_mode ----
        if normalized == "set_mode":
            valid_modes = {"beginner", "balanced", "advanced"}
            if mode not in valid_modes:
                return json.dumps({
                    "success": False,
                    "error": f"Invalid mode. Must be one of: {valid_modes}"
                }, indent=2)

            dimension_map = {"beginner": 10, "balanced": 20, "advanced": 55}
            new_dims = dimension_map[mode]
            mc.fitness_dimensions = new_dims
            mc.mode = mode
            mc.fitness_tracker.mode = mode
            mc.fitness_tracker.dimensions.mode = mode

            # Persist to config.yaml so restart respects the change
            try:
                import yaml
                config_path = Path(HERMES_HOME) / "config.yaml"
                if config_path.exists():
                    cfg = yaml.safe_load(config_path.read_text()) or {}
                    if "mstar" not in cfg:
                        cfg["mstar"] = {}
                    cfg["mstar"]["mode"] = mode
                    cfg["mstar"]["fitness_dimensions"] = new_dims
                    config_path.write_text(yaml.dump(cfg, allow_unicode=True))
            except Exception as _cfg_err:
                pass  # Non-fatal: in-memory change still applies

            return json.dumps({
                "success": True,
                "action": "set_mode",
                "mode": mode,
                "fitness_dimensions": new_dims,
            }, indent=2, ensure_ascii=False)

        # ---- get_config ----
        if normalized == "get_config":
            return json.dumps({
                "success": True,
                "action": "get_config",
                "config": {
                    "mode": mc.mode,
                    "fitness_dimensions": mc.fitness_dimensions,
                    "evolution_interval": 10,
                    "dashboard_port": 18792,
                }
            }, indent=2, ensure_ascii=False)

        # ---- forgetting_decision ----
        if normalized == "forgetting_decision":
            if not program_id:
                return json.dumps({"success": False, "error": "program_id required"}, indent=2)

            conn = mc.fitness_tracker.db_path
            import sqlite3
            db = sqlite3.connect(conn)
            cur = db.execute("SELECT fitness_score, lineage_depth, fitness_history FROM programs WHERE program_id = ?", (program_id,))
            row = cur.fetchone()
            db.close()

            if not row:
                return json.dumps({"success": False, "error": f"Program {program_id} not found"}, indent=2)

            fitness = row[0] if row[0] else 0.0
            depth = row[1] if row[1] else 0

            if fitness >= 0.6 and depth <= 2:
                decision = "keep"
                reason = f"High fitness ({fitness:.3f}) and shallow lineage (depth={depth})"
            elif fitness < 0.3:
                decision = "delete"
                reason = f"Low fitness ({fitness:.3f}) below threshold"
            elif depth > 5:
                decision = "archive"
                reason = f"Deep lineage (depth={depth}) - archive for lineage tracking"
            else:
                decision = "keep"
                reason = f"Moderate fitness ({fitness:.3f}) - keep for continued evolution"

            return json.dumps({
                "success": True,
                "action": "forgetting_decision",
                "program_id": program_id,
                "decision": decision,
                "reason": reason,
                "fitness_score": round(fitness, 4),
                "lineage_depth": depth,
            }, indent=2, ensure_ascii=False)

        # ---- record_session ----
        if normalized == "record_session":
            success = kwargs.get("success", True)
            quality = float(kwargs.get("quality", 0.8))
            latency = float(kwargs.get("latency", 1.0))
            tokens = int(kwargs.get("tokens_consumed", 0))
            episodes = int(kwargs.get("episode_count", 1))

            if not program_id:
                program_id = f"prog_session_{datetime.now().timestamp()}"

            mc.fitness_tracker.update(
                program_id=program_id,
                success=success,
                quality=quality * 100,
                latency=latency,
                tokens_used=tokens,
            )

            # 获取更新后的总 session 数
            import sqlite3 as _sql_conn
            _c = _sql_conn.connect(mc.fitness_tracker.db_path)
            _cur = _c.execute(
                "SELECT COUNT(*) FROM programs WHERE program_id = ?",
                (program_id,)
            )
            _row = _cur.fetchone()
            _sessions_total = _row[0] if _row else 0
            _c.close()

            return json.dumps({
                "success": True,
                "action": "record_session",
                "program_id": program_id,
                "sessions_total": _sessions_total,
                "recorded": {"success": success, "quality": quality, "latency": latency, "tokens": tokens},
            }, indent=2, ensure_ascii=False)

        # ---- run_ablation ----
        if normalized == "run_ablation":
            configurations = kwargs.get("configurations", [])
            sessions = kwargs.get("sessions_per_config", 30)
            if not configurations:
                return tool_error("run_ablation requires 'configurations' parameter")

            from mstar_core.evaluation.ablation_engine import run_ablation_experiment, ablation_to_dict
            report = run_ablation_experiment(
                configurations=configurations,
                sessions_per_config=sessions,
            )
            return json.dumps(ablation_to_dict(report), indent=2, ensure_ascii=False)

        # ---- Phase 1c: Perturbation Robustness ----
        if normalized == "compute_fitness_ci":
            if not program_id:
                return json.dumps({"success": False, "error": "program_id required"}, indent=2)
            ci = mc.compute_fitness_confidence_interval(program_id)
            return json.dumps({
                "success": True, "action": "compute_fitness_ci",
                "program_id": program_id,
                "ci": {"mean": ci.mean, "lower": ci.lower, "upper": ci.upper, "width": ci.width, "n_samples": ci.n_samples},
            }, indent=2, ensure_ascii=False)

        if normalized == "run_perturbation_test":
            if not program_id:
                return json.dumps({"success": False, "error": "program_id required"}, indent=2)
            noise_level = float(kwargs.get("noise_level", 0.1))
            result = mc.run_perturbation_test(program_id, noise_level)
            return json.dumps({"success": True, "action": "run_perturbation_test", "program_id": program_id, "result": result}, indent=2, ensure_ascii=False)

        if normalized == "get_robustness_stats":
            stats = mc.get_robustness_stats()
            return json.dumps({"success": True, "action": "get_robustness_stats", "stats": stats}, indent=2, ensure_ascii=False)

        # ---- Phase 3: Version Control ----
        if normalized == "create_version":
            if not program_id:
                return json.dumps({"success": False, "error": "program_id required"}, indent=2)
            mutation_type = kwargs.get("mutation_type", "unknown")
            mutation_details = kwargs.get("mutation_details", {})
            version = mc.create_version(program_id, mutation_type, mutation_details)
            return json.dumps({"success": True, "action": "create_version", "version_id": version.version_id, "status": version.status}, indent=2, ensure_ascii=False)

        if normalized == "get_program_versions":
            if not program_id:
                return json.dumps({"success": False, "error": "program_id required"}, indent=2)
            limit = kwargs.get("limit", 50)
            versions = mc.get_program_versions(program_id, limit)
            return json.dumps({"success": True, "action": "get_program_versions", "program_id": program_id, "versions": [v.to_dict() for v in versions]}, indent=2, ensure_ascii=False)

        if normalized == "rollback_version":
            if not program_id:
                return json.dumps({"success": False, "error": "program_id required"}, indent=2)
            version_id = kwargs.get("version_id", "")
            reason = kwargs.get("reason", "")
            version = mc.rollback_version(program_id, version_id, reason)
            if version:
                return json.dumps({"success": True, "action": "rollback_version", "rolled_back_to": version.version_id}, indent=2, ensure_ascii=False)
            return json.dumps({"success": False, "error": f"Version {version_id} not found"}, indent=2)

        if normalized == "get_version_control_stats":
            stats = mc.get_version_control_stats()
            return json.dumps({"success": True, "action": "get_version_control_stats", "stats": stats}, indent=2, ensure_ascii=False)

        # ---- Phase 5: EvolveMem ----
        if normalized == "get_retrieval_config":
            cfg = mc.get_retrieval_config()
            return json.dumps({"success": True, "action": "get_retrieval_config", "config": cfg.to_dict()}, indent=2, ensure_ascii=False)

        if normalized == "select_retrieval_config":
            cfg = mc.select_retrieval_config()
            return json.dumps({"success": True, "action": "select_retrieval_config", "selected_config": cfg.config_id}, indent=2, ensure_ascii=False)

        if normalized == "get_evolvemem_stats":
            stats = mc.get_evolvemem_stats()
            return json.dumps({"success": True, "action": "get_evolvemem_stats", "stats": stats}, indent=2, ensure_ascii=False)

        # ---- Phase 6: Experience Recall ----
        if normalized == "recall_similar_experience":
            current_state = kwargs.get("current_state", {})
            task_type = kwargs.get("task_type")
            results = mc.recall_similar_experience(current_state, task_type)
            return json.dumps({"success": True, "action": "recall_similar_experience", "results": [(t.trajectory_id, s) for t, s in results]}, indent=2, ensure_ascii=False)

        if normalized == "get_reasoning_from_history":
            current_state = kwargs.get("current_state", {})
            task_type = kwargs.get("task_type")
            reasoning = mc.get_reasoning_from_history(current_state, task_type)
            return json.dumps({"success": True, "action": "get_reasoning_from_history", "reasoning": reasoning}, indent=2, ensure_ascii=False)

        if normalized == "get_experience_recall_stats":
            stats = mc.get_experience_recall_stats()
            return json.dumps({"success": True, "action": "get_experience_recall_stats", "stats": stats}, indent=2, ensure_ascii=False)

        # ---- Phase 7: RSPL/SEPL Protocol ----
        if normalized == "register_skill":
            skill_metadata_dict = kwargs.get("skill_metadata", {})
            from mstar_core.evolution.protocol import SkillMetadata
            metadata = SkillMetadata.from_dict(skill_metadata_dict)
            result = mc.register_skill(metadata)
            return json.dumps({"success": True, "action": "register_skill", "result": result.to_dict()}, indent=2, ensure_ascii=False)

        if normalized == "list_skills":
            category = kwargs.get("category")
            result = mc.list_skills(category)
            return json.dumps({"success": True, "action": "list_skills", "result": result.to_dict()}, indent=2, ensure_ascii=False)

        if normalized == "propose_evolution":
            if not program_id:
                return json.dumps({"success": False, "error": "program_id required"}, indent=2)
            mutation_type = kwargs.get("mutation_type", "unknown")
            mutation_details = kwargs.get("mutation_details", {})
            result = mc.propose_evolution(program_id, mutation_type, mutation_details)
            return json.dumps({"success": True, "action": "propose_evolution", "result": result.to_dict()}, indent=2, ensure_ascii=False)

        if normalized == "verify_evolution":
            proposal_id = kwargs.get("proposal_id", "")
            result = mc.verify_evolution(proposal_id)
            return json.dumps({"success": True, "action": "verify_evolution", "result": result.to_dict()}, indent=2, ensure_ascii=False)

        if normalized == "apply_evolution":
            proposal_id = kwargs.get("proposal_id", "")
            result = mc.apply_evolution(proposal_id)
            return json.dumps({"success": True, "action": "apply_evolution", "result": result.to_dict()}, indent=2, ensure_ascii=False)

        if normalized == "rollback_evolution":
            proposal_id = kwargs.get("proposal_id", "")
            reason = kwargs.get("reason", "")
            result = mc.rollback_evolution(proposal_id, reason)
            return json.dumps({"success": True, "action": "rollback_evolution", "result": result.to_dict()}, indent=2, ensure_ascii=False)

        if normalized == "get_protocol_stats":
            stats = mc.get_protocol_stats()
            return json.dumps({"success": True, "action": "get_protocol_stats", "stats": stats}, indent=2, ensure_ascii=False)

        return json.dumps({
            "success": False,
            "error": f"Unknown action '{action}'",
            "available_actions": [
                "trigger_evolution", "get_all_fitness", "get_fitness", "get_lineage",
                "get_statistics", "get_dashboard_url", "set_mode", "get_config",
                "forgetting_decision", "record_session", "run_ablation",
            ]
        }, indent=2, ensure_ascii=False)

    except Exception as e:
        import traceback
        return json.dumps({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }, indent=2, ensure_ascii=False)


MSTAR_SCHEMA = {
    "name": "mstar",
    "description": """MSTAR Pro 自进化引擎 - AI Agent 自动优化系统

提供完整的自进化闭环功能：
- trigger_evolution: 手动触发进化周期，评估低适应度程序并执行变异
- get_all_fitness: 查看所有追踪程序的适应度排名
- get_fitness: 查看指定程序的详细适应度指标
- get_lineage: 追溯程序的进化血缘链
- get_statistics: 查看 MSTAR Pro 运行统计
- get_dashboard_url: 获取 Dashboard 地址
- set_mode: 设置模式 (beginner=10维 / balanced=20维 / advanced=55维)
- get_config: 查看当前配置参数
- forgetting_decision: 查询程序的遗忘策略
- record_session: 手动记录一个 session 执行结果
- run_ablation: 运行消融实验验证假设
- compute_fitness_ci: Phase 1c 计算Fitness Bootstrap置信区间
- run_perturbation_test: Phase 1c 运行扰动测试
- get_robustness_stats: Phase 1c 获取鲁棒性统计
- create_version: Phase 3 创建mutation版本
- get_program_versions: Phase 3 获取版本历史
- rollback_version: Phase 3 回滚版本
- get_version_control_stats: Phase 3 获取版本控制统计
- get_retrieval_config: Phase 5 获取当前检索配置
- select_retrieval_config: Phase 5 选择检索配置
- get_evolvemem_stats: Phase 5 获取EvolveMem统计
- recall_similar_experience: Phase 6 召回相似经验
- get_reasoning_from_history: Phase 6 获取历史推理链
- get_experience_recall_stats: Phase 6 获取经验召回统计
- register_skill: Phase 7 注册Skill (RSPL)
- list_skills: Phase 7 列出Skills (RSPL)
- propose_evolution: Phase 7 提出进化提案 (SEPL)
- verify_evolution: Phase 7 验证进化提案 (SEPL)
- apply_evolution: Phase 7 应用进化提案 (SEPL)
- rollback_evolution: Phase 7 回滚进化 (SEPL)
- get_protocol_stats: Phase 7 获取协议统计""",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "操作类型: trigger_evolution, get_all_fitness, get_fitness, get_lineage, get_statistics, get_dashboard_url, set_mode, get_config, forgetting_decision, record_session, run_ablation"
            },
            "program_id": {
                "type": "string",
                "description": "程序ID (用于 get_fitness, get_lineage, forgetting_decision)"
            },
            "detailed": {
                "type": "boolean",
                "description": "是否返回详细信息 (用于 get_fitness, get_all_fitness)"
            },
            "mode": {
                "type": "string",
                "description": "运行模式 (用于 set_mode): beginner, balanced, advanced"
            },
            "min_score": {
                "type": "number",
                "description": "最低分数过滤 (用于 get_all_fitness)"
            },
            "limit": {
                "type": "integer",
                "description": "返回数量限制 (用于 get_all_fitness, 默认20)"
            },
            "generations": {
                "type": "integer",
                "description": "血缘追溯代数 (用于 get_lineage, 默认10)"
            },
            "configurations": {
                "type": "array",
                "description": "消融实验配置列表 (用于 run_ablation)"
            },
            "sessions_per_config": {
                "type": "integer",
                "description": "每个配置的 session 数 (用于 run_ablation, 默认30)"
            },
            "success": {
                "type": "boolean",
                "description": "session 是否成功 (用于 record_session)"
            },
            "quality": {
                "type": "number",
                "description": "输出质量 0.0-1.0 (用于 record_session)"
            },
            "latency": {
                "type": "number",
                "description": "执行延迟秒数 (用于 record_session)"
            },
            "tokens_consumed": {
                "type": "integer",
                "description": "消耗 token 数 (用于 record_session)"
            },
            "episode_count": {
                "type": "integer",
                "description": "评估 episode 数 (用于 record_session)"
            },
        },
        "required": ["action"],
    },
}


def check_mstar_requirements() -> bool:
    """MSTAR 工具在交互模式和网关模式下可用"""
    return True


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="mstar",
    toolset="mstar",
    schema=MSTAR_SCHEMA,
    handler=lambda args, **kw: mstar(
        action=args.get("action", ""),
        program_id=args.get("program_id", ""),
        detailed=args.get("detailed", False),
        mode=args.get("mode", ""),
        min_score=args.get("min_score", 0.0),
        limit=args.get("limit", 20),
        generations=args.get("generations", 10),
        configurations=args.get("configurations", []),
        sessions_per_config=args.get("sessions_per_config", 30),
        success=args.get("success", True),
        quality=args.get("quality", 0.8),
        latency=args.get("latency", 1.0),
        tokens_consumed=args.get("tokens_consumed", 0),
        episode_count=args.get("episode_count", 1),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_mstar_requirements,
    emoji="🧬",
)


# ==============================================================================
# Individual tool registrations — each action exposed as a standalone tool
# ==============================================================================

_TOOL_EMOJI = "🧬"

def _mk_tool_schema(name: str, description: str, properties: dict, required: list) -> dict:
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


# --- mstar_trigger_evolution ---
registry.register(
    name="mstar_trigger_evolution",
    toolset="mstar",
    schema=_mk_tool_schema(
        "mstar_trigger_evolution",
        "MSTAR Pro 手动触发进化周期。评估所有追踪程序，对低适应度程序执行变异操作，返回进化结果摘要。",
        {},
        [],
    ),
    handler=lambda args, **kw: mstar(action="trigger_evolution", task_id=kw.get("task_id")),
    check_fn=check_mstar_requirements,
    emoji=_TOOL_EMOJI,
)

# --- mstar_get_all_fitness ---
registry.register(
    name="mstar_get_all_fitness",
    toolset="mstar",
    schema=_mk_tool_schema(
        "mstar_get_all_fitness",
        "MSTAR Pro 获取所有追踪程序的适应度排名，按分数降序排列。可选 min_score 过滤和 limit 限制数量。",
        {
            "min_score": {"type": "number", "description": "最低分数阈值 (默认0.0)"},
            "limit": {"type": "integer", "description": "返回数量限制 (默认20)"},
            "detailed": {"type": "boolean", "description": "是否返回详细信息 (默认false)"},
        },
        [],
    ),
    handler=lambda args, **kw: mstar(
        action="get_all_fitness",
        min_score=args.get("min_score", 0.0),
        limit=args.get("limit", 20),
        detailed=args.get("detailed", False),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_mstar_requirements,
    emoji=_TOOL_EMOJI,
)

# --- mstar_get_fitness ---
registry.register(
    name="mstar_get_fitness",
    toolset="mstar",
    schema=_mk_tool_schema(
        "mstar_get_fitness",
        "MSTAR Pro 获取指定程序的详细适应度指标，包括 EMA 趋势、波动性、episode 统计等。",
        {
            "program_id": {"type": "string", "description": "程序ID"},
            "detailed": {"type": "boolean", "description": "是否返回详细指标 (默认false)"},
        },
        ["program_id"],
    ),
    handler=lambda args, **kw: mstar(
        action="get_fitness",
        program_id=args.get("program_id", ""),
        detailed=args.get("detailed", False),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_mstar_requirements,
    emoji=_TOOL_EMOJI,
)

# --- mstar_get_lineage ---
registry.register(
    name="mstar_get_lineage",
    toolset="mstar",
    schema=_mk_tool_schema(
        "mstar_get_lineage",
        "MSTAR Pro 追溯程序的进化血缘链，向上追溯多代祖先和变异历史。",
        {
            "program_id": {"type": "string", "description": "程序ID"},
            "generations": {"type": "integer", "description": "向上追溯代数 (默认10)"},
        },
        ["program_id"],
    ),
    handler=lambda args, **kw: mstar(
        action="get_lineage",
        program_id=args.get("program_id", ""),
        generations=args.get("generations", 10),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_mstar_requirements,
    emoji=_TOOL_EMOJI,
)

# --- mstar_get_statistics ---
registry.register(
    name="mstar_get_statistics",
    toolset="mstar",
    schema=_mk_tool_schema(
        "mstar_get_statistics",
        "MSTAR Pro 获取运行统计：追踪程序数、进化次数、变异数、适应度均值等。",
        {},
        [],
    ),
    handler=lambda args, **kw: mstar(action="get_statistics", task_id=kw.get("task_id")),
    check_fn=check_mstar_requirements,
    emoji=_TOOL_EMOJI,
)

# --- mstar_get_dashboard_url ---
registry.register(
    name="mstar_get_dashboard_url",
    toolset="mstar",
    schema=_mk_tool_schema(
        "mstar_get_dashboard_url",
        "MSTAR Pro 获取 Dashboard HTTP 服务地址。",
        {},
        [],
    ),
    handler=lambda args, **kw: mstar(action="get_dashboard_url", task_id=kw.get("task_id")),
    check_fn=check_mstar_requirements,
    emoji=_TOOL_EMOJI,
)

# --- mstar_set_mode ---
registry.register(
    name="mstar_set_mode",
    toolset="mstar",
    schema=_mk_tool_schema(
        "mstar_set_mode",
        "MSTAR Pro 设置运行模式，影响适应度维度 (beginner=10维 / balanced=20维 / advanced=55维)。",
        {"mode": {"type": "string", "description": "模式: beginner, balanced, advanced"}},
        ["mode"],
    ),
    handler=lambda args, **kw: mstar(
        action="set_mode",
        mode=args.get("mode", "balanced"),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_mstar_requirements,
    emoji=_TOOL_EMOJI,
)

# --- mstar_get_config ---
registry.register(
    name="mstar_get_config",
    toolset="mstar",
    schema=_mk_tool_schema(
        "mstar_get_config",
        "MSTAR Pro 查看当前配置参数，包括模式、维度、进化间隔等。",
        {},
        [],
    ),
    handler=lambda args, **kw: mstar(action="get_config", task_id=kw.get("task_id")),
    check_fn=check_mstar_requirements,
    emoji=_TOOL_EMOJI,
)

# --- mstar_forgetting_decision ---
registry.register(
    name="mstar_forgetting_decision",
    toolset="mstar",
    schema=_mk_tool_schema(
        "mstar_forgetting_decision",
        "MSTAR Pro 查询程序的遗忘决策和解释，返回策略 (keep/archive/merge/delete) 和人类可读原因。",
        {"program_id": {"type": "string", "description": "程序ID"}},
        ["program_id"],
    ),
    handler=lambda args, **kw: mstar(
        action="forgetting_decision",
        program_id=args.get("program_id", ""),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_mstar_requirements,
    emoji=_TOOL_EMOJI,
)

# --- mstar_record_session ---
registry.register(
    name="mstar_record_session",
    toolset="mstar",
    schema=_mk_tool_schema(
        "mstar_record_session",
        "MSTAR Pro 手动记录一个 session 的执行结果，用于适应度追踪。",
        {
            "program_id": {"type": "string", "description": "程序/技能ID"},
            "success": {"type": "boolean", "description": "session是否成功"},
            "quality": {"type": "number", "description": "输出质量评分 0.0-1.0"},
            "latency": {"type": "number", "description": "执行延迟(秒)"},
            "tokens_consumed": {"type": "integer", "description": "消耗token数"},
            "episode_count": {"type": "integer", "description": "评估episode数 (默认1)"},
        },
        ["program_id", "success"],
    ),
    handler=lambda args, **kw: mstar(
        action="record_session",
        program_id=args.get("program_id", ""),
        success=args.get("success", True),
        quality=args.get("quality", 0.8),
        latency=args.get("latency", 1.0),
        tokens_consumed=args.get("tokens_consumed", 0),
        episode_count=args.get("episode_count", 1),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_mstar_requirements,
    emoji=_TOOL_EMOJI,
)

# --- mstar_run_ablation ---
registry.register(
    name="mstar_run_ablation",
    toolset="mstar",
    schema=_mk_tool_schema(
        "mstar_run_ablation",
        "MSTAR Pro 运行消融实验，对比不同配置的效果差异，返回统计显著性报告。",
        {
            "configurations": {
                "type": "array",
                "description": "要对比的配置列表，每个配置包含 name 和 params",
            },
            "sessions_per_config": {"type": "integer", "description": "每个配置运行的session数 (默认30)"},
        },
        [],
    ),
    handler=lambda args, **kw: mstar(
        action="run_ablation",
        configurations=args.get("configurations", []),
        sessions_per_config=args.get("sessions_per_config", 30),
        task_id=kw.get("task_id"),
    ),
    check_fn=check_mstar_requirements,
    emoji=_TOOL_EMOJI,
)

