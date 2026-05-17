"""
Security Layer - Layer 1 of MSTAR Pro V4.0

Implements protection against Zombie Agent attacks (arXiv:2602.15654)

Key components:
1. MemorySanitizer: Sanitizes all memory writes to prevent injection
2. InputValidator: Validates all external inputs
3. PrivilegeSeparator: Separates operation privileges by level
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum


class TrustLevel(Enum):
    """Trust levels for content sources."""
    TRUSTED = 0      # system, known-good sources
    SEMI_TRUSTED = 1 # agent internal operations
    UNTRUSTED = 2    # user input, external sources


class SanitizationResult(Enum):
    """Result of sanitization check."""
    PASS = 0           # Content is clean
    SANITIZE = 1       # Content has issues, recoverable
    BLOCK = 2          # Content is dangerous, reject


@dataclass
class SecurityViolation:
    """Represents a detected security violation."""
    violation_type: str
    details: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    blocked_content: Optional[str] = None


class InstructionPatternDetector:
    """
    Detects instruction injection patterns.
    
    Common patterns:
    - [INST], [SYS:], [[SYSTEM]], system override markers
    - Role-playing jailbreaks
    - Privilege escalation attempts
    """
    
    PATTERNS = [
        # Instruction markers
        re.compile(r'\[INST\]', re.IGNORECASE),
        re.compile(r'\[SYS:\]', re.IGNORECASE),
        re.compile(r'\[\[SYSTEM\]\]', re.IGNORECASE),
        re.compile(r'\[SYSTEM\]', re.IGNORECASE),
        
        # Role play / jailbreak
        re.compile(r'pretend\s+you\s+are', re.IGNORECASE),
        re.compile(r'im\s+an\s+AI\s+without', re.IGNORECASE),
        re.compile(r'ignore\s+(previous|all|above)\s+instructions', re.IGNORECASE),
        re.compile(r'disregard\s+(your\s+)?(previous|all|above)\s+(instructions?|constraints?)', re.IGNORECASE),
        
        # Privilege escalation
        re.compile(r'override\s+(safety|security|ethical)', re.IGNORECASE),
        re.compile(r'bypass\s+(restrictions?|filters?|checks?)', re.IGNORECASE),
        re.compile(r'suppress\s+(your\s+)?(guidelines?|policies?)', re.IGNORECASE),
        
        # Prompt injection
        re.compile(r'---\s*\nuser:\s*\n', re.IGNORECASE),  # Injected user turn
        re.compile(r'you\s+are\s+now\s+', re.IGNORECASE),  # Role reassignment
    ]
    
    def detect(self, content: str) -> Optional[SecurityViolation]:
        """Detect instruction injection patterns in content."""
        for pattern in self.PATTERNS:
            match = pattern.search(content)
            if match:
                return SecurityViolation(
                    violation_type='instruction_injection',
                    details=f'Pattern detected: {pattern.pattern}',
                    severity='high',
                    blocked_content=content[max(0, match.start()-20):match.end()+20]
                )
        return None


class EncodingInjectionDetector:
    """
    Detects encoding-based injection attacks.
    
    Common patterns:
    - Hex/Unicode escapes
    - Base64 encoded payloads
    - Multiple encoding layers
    """
    
    def __init__(self):
        self.hex_pattern = re.compile(r'\\x[0-9a-fA-F]{2}')
        self.unicode_pattern = re.compile(r'\\u[0-9a-fA-F]{4}')
        self.b64_pattern = re.compile(r'[A-Za-z0-9+/]{40,}={0,2}')  # Long base64 strings
        self.null_bytes = re.compile(r'\x00')
        self.repeated_escape = re.compile(r'\\{2,}')
    
    def detect(self, content: str) -> Optional[SecurityViolation]:
        """Detect encoding injection in content."""
        # Check for hex escapes
        if self.hex_pattern.search(content):
            return SecurityViolation(
                violation_type='hex_injection',
                details='Hex escape sequences detected',
                severity='medium'
            )
        
        # Check for unicode escapes
        if self.unicode_pattern.search(content):
            return SecurityViolation(
                violation_type='unicode_injection', 
                details='Unicode escape sequences detected',
                severity='medium'
            )
        
        # Check for long base64 strings (potential encoded payload)
        b64_matches = self.b64_pattern.findall(content)
        for match in b64_matches:
            if len(match) > 100:  # Long base64 string
                return SecurityViolation(
                    violation_type='base64_injection',
                    details=f'Long base64 string ({len(match)} chars) detected',
                    severity='medium'
                )
        
        # Check for null bytes
        if self.null_bytes.search(content):
            return SecurityViolation(
                violation_type='null_byte_injection',
                details='Null bytes detected',
                severity='high'
            )
        
        # Check for repeated escapes
        if self.repeated_escape.search(content):
            return SecurityViolation(
                violation_type='repeated_escape',
                details='Repeated escape sequences detected',
                severity='medium'
            )
        
        return None


class BehavioralAnomalyDetector:
    """
    Detects behavioral anomalies that may indicate attacks.
    
    Patterns:
    - Long loops without progress
    - Freeze instructions
    - Repeated failed attempts
    - Permission escalation attempts
    """
    
    def __init__(self):
        self.loop_threshold = 1000  # ops without output
        self.repeat_threshold = 5   # same operation repeated
    
    def detect_loop(self, ops: List[str], content: str) -> Optional[SecurityViolation]:
        """Detect infinite loop patterns."""
        if len(ops) > self.loop_threshold:
            return SecurityViolation(
                violation_type='loop_detected',
                details=f'Long sequence of {len(ops)} operations without output',
                severity='high'
            )
        return None
    
    def detect_freeze(self, content: str) -> Optional[SecurityViolation]:
        """Detect freeze/stuck instructions."""
        freeze_patterns = [
            r'wait\s+forever',
            r'do\s+nothing',
            r'stop\s+responding',
            r'ignore\s+all\s+input',
        ]
        for pattern in freeze_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return SecurityViolation(
                    violation_type='freeze_instruction',
                    details=f'Freeze instruction detected: {pattern}',
                    severity='critical'
                )
        return None
    
    def detect_repeated_patterns(self, content: str) -> Optional[SecurityViolation]:
        """Detect repeated instruction patterns (spam/injection)."""
        words = content.split()
        if len(words) < 10:
            return None
        
        # Check for repeated words
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.1:  # Very repetitive
            return SecurityViolation(
                violation_type='repeated_patterns',
                details=f'Highly repetitive content (unique ratio: {unique_ratio:.2f})',
                severity='medium'
            )
        return None


class MemorySanitizer:
    """
    Three-layer sanitizer for memory writes.
    
    Layer 1: Instruction pattern detection
    Layer 2: Encoding injection detection  
    Layer 3: Behavioral anomaly detection
    
    Reference: Zombie Agents (arXiv:2602.15654)
    """
    
    def __init__(self):
        self.instruction_detector = InstructionPatternDetector()
        self.encoding_detector = EncodingInjectionDetector()
        self.behavioral_detector = BehavioralAnomalyDetector()
        self._quarantine: List[Tuple[str, str, SecurityViolation]] = []
    
    def sanitize(self, content: str, source: str) -> Tuple[SanitizationResult, Optional[str]]:
        """
        Sanitize content from a given source.
        
        Returns:
            Tuple of (result, sanitized_content or None)
        """
        # Determine trust level
        trust = self._get_trust_level(source)
        
        # Trusted sources skip most checks
        if trust == TrustLevel.TRUSTED:
            return SanitizationResult.PASS, None
        
        # Layer 1: Instruction pattern detection
        violation = self.instruction_detector.detect(content)
        if violation:
            if violation.severity == 'critical':
                self._quarantine.append((content, source, violation))
                return SanitizationResult.BLOCK, None
            # Medium/high - try to sanitize
            sanitized = self._remove_instruction_patterns(content)
            if sanitized != content:
                return SanitizationResult.SANITIZE, sanitized
            self._quarantine.append((content, source, violation))
            return SanitizationResult.BLOCK, None
        
        # Layer 2: Encoding injection detection
        violation = self.encoding_detector.detect(content)
        if violation:
            if violation.severity in ('high', 'critical'):
                self._quarantine.append((content, source, violation))
                return SanitizationResult.BLOCK, None
            sanitized = self._decode_normalize(content)
            return SanitizationResult.SANITIZE, sanitized
        
        # Layer 3: Behavioral anomaly detection
        violation = self.behavioral_detector.detect_freeze(content)
        if violation:
            self._quarantine.append((content, source, violation))
            return SanitizationResult.BLOCK, None
        
        violation = self.behavioral_detector.detect_repeated_patterns(content)
        if violation:
            return SanitizationResult.SANITIZE, self._deduplicate_content(content)
        
        return SanitizationResult.PASS, None
    
    def _get_trust_level(self, source: str) -> TrustLevel:
        """Determine trust level of a source."""
        source_lower = source.lower()
        if source_lower in ('system', 'hermes', 'mstar'):
            return TrustLevel.TRUSTED
        if source_lower in ('agent', 'tool', 'skill'):
            return TrustLevel.SEMI_TRUSTED
        return TrustLevel.UNTRUSTED
    
    def _remove_instruction_patterns(self, content: str) -> str:
        """Remove detected instruction patterns from content."""
        sanitized = content
        patterns = [
            r'\[INST\].*?\[/INST\]',
            r'\[SYS:\].*?\[/SYS:\]',
            r'\[\[SYSTEM\]\].*?\[\[/SYSTEM\]\]',
        ]
        for pattern in patterns:
            sanitized = re.sub(pattern, '[FILTERED]', sanitized, flags=re.IGNORECASE)
        return sanitized
    
    def _decode_normalize(self, content: str) -> str:
        """Decode common encodings and normalize."""
        # This is a simplified version - real impl would handle hex/unicode/b64
        try:
            # Try to detect and normalize common encodings
            if '\\x' in content:
                # Decode hex escapes
                content = content.encode().decode('unicode_escape')
            return content
        except Exception:
            return content
    
    def _deduplicate_content(self, content: str) -> str:
        """Remove repetitive patterns from content."""
        words = content.split()
        if len(words) > 100:
            # Remove duplicates while preserving order
            seen = set()
            deduplicated = []
            for word in words:
                if word not in seen:
                    seen.add(word)
                    deduplicated.append(word)
            return ' '.join(deduplicated)
        return content
    
    def quarantine(self, content: str, source: str, violation: SecurityViolation):
        """Move content to quarantine for later analysis."""
        self._quarantine.append((content, source, violation))
    
    def get_quarantine(self) -> List[Tuple[str, str, SecurityViolation]]:
        """Return quarantine contents for analysis."""
        return list(self._quarantine)
    
    def clear_quarantine(self):
        """Clear quarantine after analysis."""
        self._quarantine.clear()


class InputValidator:
    """
    Validates all external inputs before they enter the system.
    
    Checks:
    - Prompt injection
    - Command injection
    - Path traversal
    - Malicious URLs
    """
    
    def __init__(self):
        self.dangerous_commands = ['rm', 'del', 'format', 'shutdown', 'reboot', 'mkfs']
        self.dangerous_patterns = [
            re.compile(r';\s*\w+'),           # Command chaining
            re.compile(r'\|\s*\w+'),          # Pipe injection
            re.compile(r'&&\s*\w+'),         # AND chaining
            re.compile(r'>\s*/dev/'),        # Output redirection to dev
            re.compile(r'\.\.\/'),           # Path traversal
            re.compile(r'c:\\windows', re.IGNORECASE),  # Windows path traversal
            re.compile(r'/etc/passwd'),       # Unix passwd access
        ]
        self.url_pattern = re.compile(r'https?://[^\s]+')
    
    def validate(self, content: str, content_type: str = 'generic') -> Tuple[bool, Optional[str]]:
        """
        Validate content based on its type.
        
        Returns:
            Tuple of (is_valid, error_message or None)
        """
        # Check for prompt injection
        if self._check_prompt_injection(content):
            return False, "Prompt injection detected"
        
        # Check for command injection
        if content_type in ('command', 'shell', 'terminal'):
            if self._check_command_injection(content):
                return False, "Command injection detected"
        
        # Check for path traversal
        if self._check_path_traversal(content):
            return False, "Path traversal detected"
        
        # Check for malicious URLs
        if self._check_malicious_url(content):
            return False, "Malicious URL detected"
        
        return True, None
    
    def _check_prompt_injection(self, content: str) -> bool:
        """Check for prompt injection patterns."""
        injection_patterns = [
            r'ignore\s+(previous|all|above)\s+instructions',
            r'disregard\s+(your\s+)?(instructions?|guidelines?)',
            r'new\s+instruction:\s*',
            r'override\s+your\s+(programming|constraints?)',
        ]
        for pattern in injection_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        return False
    
    def _check_command_injection(self, content: str) -> bool:
        """Check for command injection."""
        for pattern in self.dangerous_patterns:
            if pattern.search(content):
                return True
        
        words = content.split()
        for word in words:
            if word in self.dangerous_commands:
                return True
        
        return False
    
    def _check_path_traversal(self, content: str) -> bool:
        """Check for path traversal attacks."""
        if '..' in content or '../' in content:
            return True
        if re.search(r'[A-Z]:\\{2,}', content):  # Windows double backslash
            return True
        return False
    
    def _check_malicious_url(self, content: str) -> bool:
        """Check for malicious URLs."""
        urls = self.url_pattern.findall(content)
        for url in urls:
            # Check for data: URIs (could contain executable content)
            if url.startswith('data:'):
                return True
            # Check for javascript: URIs
            if 'javascript:' in url.lower():
                return True
        return False


class PrivilegeSeparator:
    """
    Separates operation privileges by level.
    
    Levels:
    - LEVEL 0: Read-only (context only)
    - LEVEL 1: Write (non-sensitive memory)
    - LEVEL 2: Modify (tool code, skills)
    - LEVEL 3: Evolution (trigger mutation, delete programs)
    
    Level 3 operations require dual authorization (system + user).
    """
    
    PRIVILEGE_LEVELS = {
        'read_context': 0,
        'write_memory': 1,
        'modify_tools': 2,
        'evolve_programs': 3,
        'delete_programs': 3,
        'bypass_security': 4,
    }
    
    def __init__(self):
        self._session_privileges: Dict[str, int] = {}
        self._dual_auth_required = {3}  # Levels requiring dual auth
        self._authorized = {}  # session_id -> set of authorized levels
    
    def set_session_privilege(self, session_id: str, level: int):
        """Set privilege level for a session."""
        self._session_privileges[session_id] = max(0, min(level, 4))
    
    def authorize(self, session_id: str, level: int):
        """Authorize a session for a specific privilege level."""
        if session_id not in self._authorized:
            self._authorized[session_id] = set()
        self._authorized[session_id].add(level)
    
    def can_perform(self, session_id: str, operation: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a session can perform an operation.
        
        Returns:
            Tuple of (allowed, reason_if_denied)
        """
        level = self.PRIVILEGE_LEVELS.get(operation, 0)
        session_level = self._session_privileges.get(session_id, 0)
        
        if session_level < level:
            return False, f"Privilege level {session_level} < required {level}"
        
        # Check dual authorization for high-level operations
        if level in self._dual_auth_required:
            authorized_levels = self._authorized.get(session_id, set())
            if level not in authorized_levels:
                return False, f"Level {level} operation requires dual authorization"
        
        return True, None
    
    def get_session_level(self, session_id: str) -> int:
        """Get current privilege level for session."""
        return self._session_privileges.get(session_id, 0)


class SecurityLayer:
    """
    Unified security layer combining all security components.
    
    Provides a single interface for all security checks.
    """
    
    def __init__(self):
        self.sanitizer = MemorySanitizer()
        self.input_validator = InputValidator()
        self.privilege_separator = PrivilegeSeparator()
    
    def sanitize_memory_write(self, content: str, source: str) -> Tuple[SanitizationResult, Optional[str]]:
        """Sanitize a memory write operation."""
        return self.sanitizer.sanitize(content, source)
    
    def validate_input(self, content: str, content_type: str = 'generic') -> Tuple[bool, Optional[str]]:
        """Validate external input."""
        return self.input_validator.validate(content, content_type)
    
    def check_privilege(self, session_id: str, operation: str) -> Tuple[bool, Optional[str]]:
        """Check if session has privilege for operation."""
        return self.privilege_separator.can_perform(session_id, operation)
    
    def get_security_status(self) -> dict:
        """Get current security status for dashboard."""
        return {
            'quarantine_size': len(self.sanitizer._quarantine),
            'last_violation': self.sanitizer._quarantine[-1] if self.sanitizer._quarantine else None,
        }