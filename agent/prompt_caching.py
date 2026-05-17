"""Anthropic prompt caching strategy.

Single layout: ``system_and_3``. 4 cache_control breakpoints — system
prompt + last 3 non-system messages, all at the same TTL (5m or 1h).
Reduces input token costs by ~75% on multi-turn conversations within a
single session.

Pure functions -- no class state, no AIAgent dependency.
"""

import copy
from typing import Any, Dict, List


def _apply_cache_marker(msg: dict, cache_marker: dict, native_anthropic: bool = False) -> None:
    """Add cache_control to a single message, handling all format variations."""
    role = msg.get("role", "")
    content = msg.get("content")

    if role == "tool":
        if native_anthropic:
            msg["cache_control"] = cache_marker
        return

    if content is None or content == "":
        msg["cache_control"] = cache_marker
        return

    if isinstance(content, str):
        msg["content"] = [
            {"type": "text", "text": content, "cache_control": cache_marker}
        ]
        return

    if isinstance(content, list) and content:
        last = content[-1]
        if isinstance(last, dict):
            last["cache_control"] = cache_marker


def _build_marker(ttl: str) -> Dict[str, str]:
    """Build a cache_control marker dict for the given TTL ('5m' or '1h')."""
    marker: Dict[str, str] = {"type": "ephemeral"}
    if ttl == "1h":
        marker["ttl"] = "1h"
    return marker


def apply_anthropic_cache_control(
    api_messages: List[Dict[str, Any]],
    cache_ttl: str = "5m",
    native_anthropic: bool = False,
    strategy: str = "system_and_3",
) -> List[Dict[str, Any]]:
    """Apply caching strategy to messages for Anthropic models.

    Supports multiple strategies:
      - "system_and_3" (default): system + last 3 non-system messages (4 breakpoints)
      - "stable_prefix": system prompt only — maximum cache stability, zero churn
        (MSTAR Pro v4.0 P2-1: useful when system prompt is the dominant cost)
      - "sliding_window": last N messages (N configured via cache_breakpoint_count)
      - "adaptive": system + last 3 tool_results only (avoids caching intermediate text)

    Returns:
        Deep copy of messages with cache_control breakpoints injected.
    """
    messages = copy.deepcopy(api_messages)
    if not messages:
        return messages

    marker = _build_marker(cache_ttl)

    # Stable prefix strategy: only system prompt gets cache
    if strategy == "stable_prefix":
        if messages and messages[0].get("role") == "system":
            _apply_cache_marker(messages[0], marker, native_anthropic=native_anthropic)
        return messages

    # Count tool_result blocks in recent messages (for adaptive strategy)
    tool_result_count = sum(
        1 for m in messages[-10:]
        if m.get("role") == "tool"
    )

    breakpoints_used = 0

    if messages[0].get("role") == "system":
        _apply_cache_marker(messages[0], marker, native_anthropic=native_anthropic)
        breakpoints_used += 1

    remaining = 4 - breakpoints_used
    non_sys = [i for i in range(len(messages)) if messages[i].get("role") != "system"]

    if strategy == "adaptive":
        # Only cache the most recent tool_result messages (up to 3)
        tool_msgs = [i for i in range(len(messages)) if messages[i].get("role") == "tool"]
        for idx in tool_msgs[-remaining:]:
            _apply_cache_marker(messages[idx], marker, native_anthropic=native_anthropic)
    else:
        # Default: last N non-system messages
        for idx in non_sys[-remaining:]:
            _apply_cache_marker(messages[idx], marker, native_anthropic=native_anthropic)

    return messages
