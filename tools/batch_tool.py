"""Batch tool: execute multiple tool calls in a single round trip.

Inspired by Anthropic's computer_batch / browser_batch pattern from:
https://claude.com/blog/best-practices-for-computer-and-browser-use-with-claude

Unlike concurrent tool execution (multiple threads waiting on I/O),
batch_execute runs sub-actions sequentially in one handler call and
returns all results at once - reducing agent-loop overhead.

SUITABLE FOR: self-contained actions that don't depend on intermediate
              visual feedback (e.g. mkdir+mkdir+mkdir, pip install x y z).

NOT SUITABLE: exploratory sequences where action N depends on the
              outcome of action N-1.
"""

import json
import logging
import time
from typing import Any, Dict, List

from tools.registry import registry

logger = logging.getLogger(__name__)

# Tools that are excluded from batch_execute (agent-loop tools).
_BATCH_BLACKLIST = {
    "todo", "session_search", "memory", "clarify",
    "delegate_task", "kanban_show", "kanban_do",
}


def _dispatch_single(tool_name: str, args: Dict[str, Any], task_id: str = "") -> str:
    """Dispatch one sub-action through the registry."""
    if tool_name in _BATCH_BLACKLIST:
        return json.dumps({
            "error": (
                f"Tool '{tool_name}' cannot be used inside batch_execute "
                "(agent-loop tools are excluded)."
            )
        })
    return registry.dispatch(tool_name, args, task_id=task_id)


def _estimate_quality(result: str) -> float:
    """Heuristic quality score from a sub-tool result string."""
    try:
        parsed = json.loads(result)
        if isinstance(parsed, dict):
            if parsed.get("success") is True:
                return 1.0
            if "error" in parsed:
                return 0.0
            if "output" in parsed or "result" in parsed:
                return 0.8
    except Exception:
        pass
    return 0.5


def _get_mstar_core_silent():
    """Try to get MSTARCore singleton, return None if unavailable."""
    try:
        import os as _os
        _hermes = _os.path.expanduser("~/.hermes")
        if not _os.path.exists(_hermes):
            return None
        from mstar_core import MSTARCore
        from tools.mstar_tools import _mstar_core_cache
        if _mstar_core_cache is None:
            return None
        return _mstar_core_cache
    except Exception:
        return None


def batch_execute_handler(args: Dict[str, Any], task_id: str = "") -> str:
    """Execute a list of tool calls sequentially, return all results.

    MSTAR Pro v4.0 P1 improvement: after all sub-tools complete,
    their fitness data is merged into a SINGLE batch_update() call
    (one DB transaction) instead of N separate writes per tool.
    """
    actions: List[Dict[str, Any]] = args.get("actions", [])
    if not actions:
        return json.dumps({"error": "batch_execute requires a non-empty 'actions' list"})
    if len(actions) > 50:
        msg = "batch_execute limited to 50 actions per call"
        return json.dumps({"error": msg})

    total_latency = 0.0
    results: List[Dict[str, Any]] = []
    qualities: List[float] = []
    fitness_updates: List[Dict[str, Any]] = []

    for i, action in enumerate(actions):
        tool = action.get("tool")
        tool_args = action.get("args", {})
        if not tool:
            results.append({
                "index": i,
                "tool": None,
                "result": json.dumps({"error": "missing 'tool' field"}),
            })
            continue

        start = time.monotonic()
        try:
            result = _dispatch_single(tool, tool_args, task_id=task_id)
        except Exception as e:
            logger.exception("batch_execute sub-dispatch error for tool %s", tool)
            result = json.dumps({"error": f"{type(e).__name__}: {e}"})
        latency = time.monotonic() - start
        total_latency += latency

        quality = _estimate_quality(result)
        qualities.append(quality)

        # MSTAR Pro v4.0: collect fitness data for batch_update
        # program_id encodes the tool so fitness_tracker can track per-tool history
        try:
            parsed_res = json.loads(result)
            is_success = "error" not in parsed_res and parsed_res.get("success") is not False
        except Exception:
            is_success = True  # assume success if we can't parse

        fitness_updates.append({
            "program_id": f"prog_{tool}",
            "success": is_success,
            "quality": quality * 100.0,
            "latency": latency,
            "tokens_used": 0,
        })

        results.append({
            "index": i,
            "tool": tool,
            "result": result,
            "latency_ms": int(latency * 1000),
        })

    # MSTAR Pro v4.0 P1: batch fitness write — one DB transaction for all N tools
    mc = _get_mstar_core_silent()
    if mc is not None and fitness_updates:
        try:
            mc.fitness_tracker.batch_update(fitness_updates)
        except Exception as e:
            logger.warning("batch_execute fitness batch_update failed: %s", e)

    batch_quality = sum(qualities) / len(qualities) if qualities else 1.0

    return json.dumps({
        "results": results,
        "batch_size": len(actions),
        "total_latency_ms": int(total_latency * 1000),
        "avg_quality": round(batch_quality, 4),
    })


# -------------------------------------------------------------------------
# Tool registration
# -------------------------------------------------------------------------

TOOL_SCHEMA = {
    "name": "batch_execute",
    "description": (
        "Execute multiple independent tool calls in a single round trip. "
        "All sub-actions run sequentially; results are returned together. "
        "Use when you need to perform several self-contained operations "
        "(e.g. create multiple directories, install multiple packages, "
        "run a series of independent shell commands) and want to avoid "
        "N separate round-trips through the agent loop. "
        "NOT suitable for exploratory sequences where later steps depend "
        "on earlier ones. Agent-loop tools (todo, memory, etc.) are blocked."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "description": "List of sub-actions to execute sequentially.",
                "items": {
                    "type": "object",
                    "properties": {
                        "tool": {
                            "type": "string",
                            "description": "Name of the tool to call (e.g. execute_code, terminal).",
                        },
                        "args": {
                            "type": "object",
                            "description": "Arguments to pass to the tool.",
                        },
                    },
                    "required": ["tool", "args"],
                    "additionalProperties": False,
                },
                "maxItems": 50,
            },
        },
        "required": ["actions"],
        "additionalProperties": False,
    },
}


def check_requirements() -> bool:
    return True


registry.register(
    name="batch_execute",
    toolset="utility",
    schema=TOOL_SCHEMA,
    handler=batch_execute_handler,
    check_fn=check_requirements,
    requires_env=[],
    is_async=False,
    description=TOOL_SCHEMA["description"],
    emoji="📦",
)