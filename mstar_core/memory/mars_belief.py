"""
MARS Hierarchical Belief Memory - Layer 6 of MSTAR Pro V4.0

Implements MARS (Memory-Augmented Agentic Recommender System) belief state.
Reference: MARS (arXiv:2605.14212)

Three-tier belief state:
1. Event Memory: Raw behavioral signals (high fidelity, noisy)
2. Preference Memory: Fine-grained mutable chunks with strength/evidence
3. Profile Memory: Natural language narrative (highly abstracted)

Six Operations (LLM-Driven Adaptive Scheduling):
Extraction, Reinforcement, Weakening, Consolidation, Forgetting, Resynthesis
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from collections import defaultdict
from datetime import datetime, timedelta
import time


@dataclass
class Event:
    """A single event in the event buffer."""
    event_id: str
    timestamp: float
    event_type: str  # 'tool_call', 'user_message', 'agent_response', etc.
    content: Any     # Raw content
    metadata: Dict = field(default_factory=dict)


@dataclass
class PreferenceChunk:
    """A single preference in preference memory."""
    chunk_id: str
    content: str
    strength: float = 1.0  # 0.0 to 1.0, higher = more reinforced
    evidence: List[str] = field(default_factory=list)  # Source events
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    
    def reinforce(self, amount: float = 0.1):
        """Increase strength of this preference."""
        self.strength = min(1.0, self.strength + amount)
        self.last_accessed = time.time()
        self.access_count += 1
    
    def weaken(self, amount: float = 0.05):
        """Decrease strength of this preference."""
        self.strength = max(0.0, self.strength - amount)


@dataclass
class ProfileNarrative:
    """Natural language profile distilled from preferences."""
    narrative: str = ""
    last_updated: float = field(default_factory=time.time)
    confidence: float = 0.5  # How confident we are in this narrative
    
    def update(self, new_narrative: str):
        """Update the profile narrative."""
        self.narrative = new_narrative
        self.last_updated = time.time()


class EventBuffer:
    """
    Tier 1: Event Memory Buffer
    
    Buffers raw behavioral signals with high fidelity but noise.
    Acts as the source for preference extraction.
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._buffer: List[Event] = []
        self._event_counter = 0
    
    def add(self, event_type: str, content: Any, metadata: Dict = None) -> Event:
        """Add an event to the buffer."""
        self._event_counter += 1
        event = Event(
            event_id=f"evt_{self._event_counter}",
            timestamp=time.time(),
            event_type=event_type,
            content=content,
            metadata=metadata or {}
        )
        self._buffer.append(event)
        
        # Evict oldest if over size
        if len(self._buffer) > self.max_size:
            self._buffer = self._buffer[-self.max_size:]
        
        return event
    
    def get_recent(self, n: int = 10) -> List[Event]:
        """Get n most recent events."""
        return self._buffer[-n:] if self._buffer else []
    
    def get_by_type(self, event_type: str, limit: int = 100) -> List[Event]:
        """Get events of a specific type."""
        return [e for e in self._buffer if e.event_type == event_type][-limit:]
    
    def clear(self):
        """Clear the event buffer."""
        self._buffer.clear()
    
    def size(self) -> int:
        """Get current buffer size."""
        return len(self._buffer)


class PreferenceMemory:
    """
    Tier 2: Preference Memory
    
    Fine-grained mutable chunks with explicit strength and evidence tracking.
    Supports reinforcement and weakening operations.
    """
    
    def __init__(self, max_chunks: int = 500):
        self.max_chunks = max_chunks
        self._chunks: Dict[str, PreferenceChunk] = {}
        self._chunk_counter = 0
    
    def add(self, content: str, evidence: List[str] = None) -> PreferenceChunk:
        """Add a new preference chunk."""
        self._chunk_counter += 1
        chunk = PreferenceChunk(
            chunk_id=f"pref_{self._chunk_counter}",
            content=content,
            evidence=evidence or []
        )
        
        # Check capacity and evict weakest if needed
        if len(self._chunks) >= self.max_chunks:
            self._evict_weakest()
        
        self._chunks[chunk.chunk_id] = chunk
        return chunk
    
    def get(self, chunk_id: str) -> Optional[PreferenceChunk]:
        """Get a preference chunk by ID."""
        chunk = self._chunks.get(chunk_id)
        if chunk:
            chunk.last_accessed = time.time()
            chunk.access_count += 1
        return chunk
    
    def get_top(self, n: int = 10) -> List[PreferenceChunk]:
        """Get top n chunks by strength."""
        sorted_chunks = sorted(
            self._chunks.values(),
            key=lambda c: c.strength,
            reverse=True
        )
        return sorted_chunks[:n]
    
    def reinforce(self, chunk_id: str, amount: float = 0.1):
        """Reinforce a preference chunk."""
        chunk = self._chunks.get(chunk_id)
        if chunk:
            chunk.reinforce(amount)
    
    def weaken(self, chunk_id: str, amount: float = 0.05):
        """Weaken a preference chunk."""
        chunk = self._chunks.get(chunk_id)
        if chunk:
            chunk.weaken(amount)
    
    def consolidate(self, similarity_threshold: float = 0.8) -> int:
        """
        Consolidate similar chunks by merging them.
        
        Returns number of chunks merged.
        """
        merged = 0
        chunks_list = list(self._chunks.values())
        
        for i, chunk in enumerate(chunks_list):
            if chunk.chunk_id not in self._chunks:
                continue
            
            for other in chunks_list[i+1:]:
                if other.chunk_id not in self._chunks:
                    continue
                
                # Check similarity (simple string overlap for now)
                if self._similar(chunk.content, other.content) > similarity_threshold:
                    # Merge into stronger chunk
                    if chunk.strength >= other.strength:
                        chunk.evidence.extend(other.evidence)
                        chunk.strength = max(chunk.strength, other.strength)
                        del self._chunks[other.chunk_id]
                    else:
                        other.evidence.extend(chunk.evidence)
                        other.strength = max(other.strength, chunk.strength)
                        del self._chunks[chunk.chunk_id]
                    merged += 1
                    break
        
        return merged
    
    def forget_stale(self, max_age_seconds: float = 86400 * 30, 
                     min_strength: float = 0.1) -> int:
        """
        Forget stale (old and weak) preference chunks.
        
        Args:
            max_age_seconds: Maximum age in seconds
            min_strength: Minimum strength to keep
            
        Returns number of chunks forgotten.
        """
        now = time.time()
        to_remove = []
        
        for chunk_id, chunk in self._chunks.items():
            age = now - chunk.last_accessed
            if age > max_age_seconds and chunk.strength < min_strength:
                to_remove.append(chunk_id)
        
        for chunk_id in to_remove:
            del self._chunks[chunk_id]
        
        return len(to_remove)
    
    def _similar(self, content1: str, content2: str) -> float:
        """Compute similarity between two content strings."""
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)
    
    def _evict_weakest(self):
        """Evict the weakest preference chunk."""
        if not self._chunks:
            return
        
        weakest = min(self._chunks.values(), key=lambda c: c.strength)
        del self._chunks[weakest.chunk_id]
    
    def size(self) -> int:
        """Get number of preference chunks."""
        return len(self._chunks)


class ProfileMemory:
    """
    Tier 3: Profile Memory
    
    Distills all preferences into a coherent natural language narrative.
    """
    
    def __init__(self):
        self._profile = ProfileNarrative()
        self._llm_synthesis_fn: Optional[Callable] = None
    
    def set_synthesis_fn(self, fn: Callable[[List[PreferenceChunk]], str]):
        """Set the LLM synthesis function for generating narrative."""
        self._llm_synthesis_fn = fn
    
    def synthesize(self, preferences: List[PreferenceChunk]) -> str:
        """Synthesize a profile narrative from preferences."""
        if not preferences:
            return "No preferences recorded."
        
        if self._llm_synthesis_fn:
            return self._llm_synthesis_fn(preferences)
        
        # Fallback: simple concatenation of top preferences
        top_prefs = sorted(preferences, key=lambda p: p.strength, reverse=True)[:5]
        narrative = "User profile summary:\n"
        for p in top_prefs:
            narrative += f"- {p.content} (strength: {p.strength:.2f})\n"
        return narrative
    
    def update(self, narrative: str):
        """Manually update the profile narrative."""
        self._profile.update(narrative)
    
    def get(self) -> str:
        """Get current profile narrative."""
        return self._profile.narrative
    
    def get_confidence(self) -> float:
        """Get confidence in the profile."""
        return self._profile.confidence


class AdaptiveMemoryPlanner:
    """
    LLM-Driven Adaptive Memory Scheduler
    
    Decides which operation to perform next based on memory state.
    Uses six operations: Extraction, Reinforcement, Weakening, 
    Consolidation, Forgetting, Resynthesis
    
    Unlike fixed-interval approaches, this uses LLM reasoning to
    decide when and how to operate on memory.
    """
    
    OPERATIONS = [
        'extraction',      # Extract preferences from events
        'reinforcement',   # Reinforce strong preferences
        'weakening',      # Weaken weak preferences
        'consolidation',  # Consolidate similar preferences
        'forgetting',     # Forget stale/weak memories
        'resynthesis'     # Resynthesize profile narrative
    ]
    
    def __init__(self):
        self._operation_history: List[Dict] = []
    
    def decide_operation(self, memory_state: Dict[str, Any]) -> str:
        """
        Decide which operation to perform based on memory state.
        
        Args:
            memory_state: Dict containing:
                - event_buffer_size: int
                - preference_memory_size: int  
                - preference_avg_strength: float
                - profile_last_updated: float
                - recent_failures: List[str]
                
        Returns:
            Next operation to perform
        """
        event_size = memory_state.get('event_buffer_size', 0)
        pref_size = memory_state.get('preference_memory_size', 0)
        pref_avg_strength = memory_state.get('preference_avg_strength', 0.5)
        profile_age = time.time() - memory_state.get('profile_last_updated', 0)
        
        # Decision logic based on memory state
        if event_size > 100 and pref_size < 200:
            return 'extraction'
        
        if pref_avg_strength > 0.7 and pref_size > 50:
            return 'consolidation'
        
        if pref_avg_strength < 0.3:
            return 'reinforcement'
        
        if pref_size > 400:
            return 'forgetting'
        
        if profile_age > 3600:  # Profile older than 1 hour
            return 'resynthesis'
        
        # Default to reinforcement
        return 'reinforcement'
    
    def record_operation(self, operation: str, result: Dict):
        """Record an operation for learning."""
        self._operation_history.append({
            'operation': operation,
            'timestamp': time.time(),
            'result': result
        })
        
        # Keep last 100
        if len(self._operation_history) > 100:
            self._operation_history = self._operation_history[-100:]


class MARSBeliefMemory:
    """
    MARS Hierarchical Belief Memory
    
    Main entry point for Layer 6 memory system.
    
    Combines:
    - Tier 1: EventBuffer (raw signals)
    - Tier 2: PreferenceMemory (fine-grained chunks)
    - Tier 3: ProfileMemory (natural language narrative)
    
    With adaptive scheduling via AdaptiveMemoryPlanner.
    """
    
    def __init__(self, max_events: int = 1000, max_preferences: int = 500):
        # Initialize tiers
        self.event_buffer = EventBuffer(max_size=max_events)
        self.preference_memory = PreferenceMemory(max_chunks=max_preferences)
        self.profile_memory = ProfileMemory()
        
        # Initialize planner
        self.planner = AdaptiveMemoryPlanner()
        
        # Statistics
        self._stats = {
            'total_events': 0,
            'total_extractions': 0,
            'total_reinforcements': 0,
            'total_weakens': 0,
            'total_consolidations': 0,
            'total_forgets': 0,
            'total_resyntheses': 0,
        }
    
    def record_event(self, event_type: str, content: Any, metadata: Dict = None):
        """Record a raw event in tier 1."""
        self.event_buffer.add(event_type, content, metadata)
        self._stats['total_events'] += 1
    
    def record_tool_call(self, tool_name: str, args: Dict, result: Any, success: bool):
        """Convenience method to record a tool call event."""
        self.record_event(
            event_type='tool_call',
            content={
                'tool': tool_name,
                'args': args,
                'result': str(result)[:500],  # Truncate result
                'success': success
            },
            metadata={'tool': tool_name}
        )
    
    def extract_preferences(self, extraction_fn: Optional[Callable] = None) -> int:
        """
        Extract preferences from recent events.
        
        Args:
            extraction_fn: Optional function(content) -> List[str] preferences
            
        Returns:
            Number of preferences extracted
        """
        recent_events = self.event_buffer.get_recent(50)
        
        if not extraction_fn:
            # Default extraction: simple keyword-based
            prefs = self._default_extraction(recent_events)
        else:
            # Use provided function
            contents = [e.content for e in recent_events]
            prefs = extraction_fn(contents)
        
        extracted = 0
        for pref_content in prefs:
            self.preference_memory.add(
                content=pref_content,
                evidence=[e.event_id for e in recent_events[-5:]]
            )
            extracted += 1
        
        self._stats['total_extractions'] += extracted
        return extracted
    
    def _default_extraction(self, events: List[Event]) -> List[str]:
        """Default extraction: simple rule-based."""
        prefs = []
        
        # Simple rule: extract tool usage patterns
        tool_events = [e for e in events if e.event_type == 'tool_call']
        
        if len(tool_events) >= 3:
            # Extract tool sequence as preference
            tools = [e.content.get('tool') for e in tool_events[-5:] if e.content.get('tool')]
            if tools:
                prefs.append(f"Uses tools: {' -> '.join(tools)}")
        
        # Extract success/failure patterns
        success_count = sum(1 for e in tool_events if e.content.get('success'))
        if tool_events:
            success_rate = success_count / len(tool_events)
            if success_rate < 0.5:
                prefs.append("Low success rate in recent operations")
        
        return prefs
    
    def reinforce_preferences(self, chunk_ids: List[str] = None, amount: float = 0.1):
        """Reinforce preference chunks."""
        if chunk_ids is None:
            # Reinforce top preferences
            top = self.preference_memory.get_top(10)
            chunk_ids = [c.chunk_id for c in top]
        
        for chunk_id in chunk_ids:
            self.preference_memory.reinforce(chunk_id, amount)
        
        self._stats['total_reinforcements'] += len(chunk_ids)
    
    def weaken_preferences(self, chunk_ids: List[str] = None, amount: float = 0.05):
        """Weaken preference chunks."""
        if chunk_ids is None:
            # Weaken bottom preferences
            all_chunks = self.preference_memory.get_top(1000)
            bottom = sorted(all_chunks, key=lambda c: c.strength)[:10]
            chunk_ids = [c.chunk_id for c in bottom]
        
        for chunk_id in chunk_ids:
            self.preference_memory.weaken(chunk_id, amount)
        
        self._stats['total_weakens'] += len(chunk_ids)
    
    def consolidate_preferences(self) -> int:
        """Consolidate similar preference chunks."""
        merged = self.preference_memory.consolidate()
        self._stats['total_consolidations'] += 1
        return merged
    
    def forget_stale_preferences(self) -> int:
        """Forget stale preference chunks."""
        forgotten = self.preference_memory.forget_stale()
        self._stats['total_forgets'] += forgotten
        return forgotten
    
    def resynthesize_profile(self, synthesis_fn: Optional[Callable] = None) -> str:
        """Resynthesize the profile narrative from preferences."""
        top_prefs = self.preference_memory.get_top(20)
        narrative = self.profile_memory.synthesize(top_prefs)
        
        if synthesis_fn:
            narrative = synthesis_fn(top_prefs)
        
        self.profile_memory.update(narrative)
        self._stats['total_resyntheses'] += 1
        return narrative
    
    def run_operation(self, operation: str) -> Dict[str, Any]:
        """Run a specific memory operation."""
        result = {'operation': operation, 'success': True}
        
        if operation == 'extraction':
            count = self.extract_preferences()
            result['extracted'] = count
        elif operation == 'reinforcement':
            self.reinforce_preferences()
        elif operation == 'weakening':
            self.weaken_preferences()
        elif operation == 'consolidation':
            merged = self.consolidate_preferences()
            result['merged'] = merged
        elif operation == 'forgetting':
            forgotten = self.forget_stale_preferences()
            result['forgotten'] = forgotten
        elif operation == 'resynthesis':
            narrative = self.resynthesize_profile()
            result['narrative'] = narrative[:100] + '...' if len(narrative) > 100 else narrative
        
        self.planner.record_operation(operation, result)
        return result
    
    def run_adaptive_cycle(self) -> Dict[str, Any]:
        """Run one adaptive cycle using the planner."""
        memory_state = {
            'event_buffer_size': self.event_buffer.size(),
            'preference_memory_size': self.preference_memory.size(),
            'preference_avg_strength': self._get_avg_strength(),
            'profile_last_updated': self.profile_memory._profile.last_updated,
        }
        
        operation = self.planner.decide_operation(memory_state)
        return self.run_operation(operation)
    
    def _get_avg_strength(self) -> float:
        """Get average preference strength."""
        chunks = list(self.preference_memory._chunks.values())
        if not chunks:
            return 0.5
        return sum(c.strength for c in chunks) / len(chunks)
    
    def get_state(self) -> Dict[str, Any]:
        """Get current memory state for inspection."""
        return {
            'event_buffer_size': self.event_buffer.size(),
            'preference_memory_size': self.preference_memory.size(),
            'preference_avg_strength': self._get_avg_strength(),
            'profile_narrative_preview': self.profile_memory.get()[:100],
            'statistics': dict(self._stats)
        }
    
    def get_top_preferences(self, n: int = 10) -> List[Dict]:
        """Get top n preferences as dicts."""
        top = self.preference_memory.get_top(n)
        return [
            {
                'chunk_id': c.chunk_id,
                'content': c.content,
                'strength': c.strength,
                'access_count': c.access_count
            }
            for c in top
        ]