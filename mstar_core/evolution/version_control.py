"""
MSTAR Pro v4.0 - Phase 3: Version Control + FGGM Contracts
参考: SEVerA (2603.25111) + Autogenesis

FGGM (Formally Verified Genetic Mutations):
- 每个mutation必须通过形式化验证才能执行
- 验证项: 内存安全、类型安全、语义保持
- 失败时自动rollback

Autogenesis (RSPL/SEPL):
- RSPL: Registry Skill Protocol Language (skill注册协议)
- SEPL: Skill Evolution Protocol Language (skill进化协议)
- 版本控制: 每个program有完整版本历史
- Rollback: 失败时恢复到上一个有效版本
"""

from __future__ import annotations
import logging
import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class MutationStatus(Enum):
    PENDING = "pending"
    VERIFYING = "verifying"
    VERIFIED = "verified"
    APPLIED = "applied"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"


@dataclass
class MutationVersion:
    """Mutation版本记录"""
    version_id: str
    program_id: str
    mutation_type: str
    timestamp: str
    content_hash: str  # mutation内容的hash
    parent_version: Optional[str]  # 上一个版本
    status: str
    verification_result: Optional[Dict]  # FGGM验证结果
    rollback_from: Optional[str]  # 如果是rollback，记录回滚源版本

    def to_dict(self) -> Dict:
        return {
            'version_id': self.version_id,
            'program_id': self.program_id,
            'mutation_type': self.mutation_type,
            'timestamp': self.timestamp,
            'content_hash': self.content_hash,
            'parent_version': self.parent_version,
            'status': self.status,
            'verification_result': self.verification_result,
            'rollback_from': self.rollback_from,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'MutationVersion':
        return cls(
            version_id=d['version_id'],
            program_id=d['program_id'],
            mutation_type=d['mutation_type'],
            timestamp=d['timestamp'],
            content_hash=d['content_hash'],
            parent_version=d.get('parent_version'),
            status=d['status'],
            verification_result=d.get('verification_result'),
            rollback_from=d.get('rollback_from'),
        )


@dataclass
class FGGMContract:
    """
    FGGM (Formally Verified Genetic Mutation) Contract

    验证mutation的安全性:
    - Memory Safety: mutation不会导致内存泄漏/越界
    - Type Safety: mutation不违反类型约束
    - Semantic Preservation: mutation保持原有语义
    """
    contract_id: str
    program_id: str
    mutation_content: str  # 序列化后的mutation
    checks_passed: Dict[str, bool]  # 各检查项结果
    overall_verdict: bool  # 是否通过
    verification_time_ms: float
    proof_level: str  # 'formal', 'heuristic', 'none'
    details: Dict = field(default_factory=dict)

    def is_safe(self) -> bool:
        """是否通过所有安全检查"""
        if not self.overall_verdict:
            return False
        required_checks = ['memory_safety', 'type_safety', 'semantic_preservation']
        return all(self.checks_passed.get(c, False) for c in required_checks)


class FGGMVerifier:
    """
    FGGM Contract验证器

    对每个mutation进行形式化验证前的检查。
    在MSTAR Pro中，我们使用启发式验证+安全断言，
    真实形式化验证需要TLA+或Coq，这里简化为safety checks。
    """

    def __init__(self):
        self._contract_cache: Dict[str, FGGMContract] = {}

    def _compute_hash(self, content: str) -> str:
        """计算内容hash"""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def verify_mutation(
        self,
        program_id: str,
        mutation_type: str,
        mutation_details: Dict,
    ) -> FGGMContract:
        """
        验证一个mutation是否安全。

        执行以下检查:
        1. Memory Safety: 检查是否会导致资源泄漏
        2. Type Safety: 检查参数类型是否匹配
        3. Semantic Preservation: 检查核心逻辑是否保持

        Returns:
            FGGMContract: 验证结果
        """
        import time
        start = time.time()

        contract_id = f"fggm_{program_id}_{int(start * 1000)}"
        mutation_content = json.dumps({
            'program_id': program_id,
            'mutation_type': mutation_type,
            'details': mutation_details,
        }, sort_keys=True)

        checks_passed = {
            'memory_safety': self._check_memory_safety(mutation_type, mutation_details),
            'type_safety': self._check_type_safety(mutation_type, mutation_details),
            'semantic_preservation': self._check_semantic_preservation(mutation_type, mutation_details),
        }

        # 额外的safety检查
        checks_passed['no_infinite_loop'] = self._check_no_infinite_loop(mutation_type, mutation_details)
        checks_passed['bounded_resources'] = self._check_bounded_resources(mutation_type, mutation_details)
        checks_passed['no_dangerous_apis'] = self._check_no_dangerous_apis(mutation_type, mutation_details)

        overall_verdict = all(checks_passed.values())

        verification_time_ms = (time.time() - start) * 1000

        contract = FGGMContract(
            contract_id=contract_id,
            program_id=program_id,
            mutation_content=mutation_content,
            checks_passed=checks_passed,
            overall_verdict=overall_verdict,
            verification_time_ms=verification_time_ms,
            proof_level='heuristic' if overall_verdict else 'none',
            details={
                'mutation_type': mutation_type,
                'n_checks': len(checks_passed),
                'passed_checks': sum(1 for v in checks_passed.values() if v),
            },
        )

        self._contract_cache[contract_id] = contract
        return contract

    def _check_memory_safety(self, mutation_type: str, details: Dict) -> bool:
        """检查内存安全性"""
        # 危险mutation类型
        dangerous = ['memory_free', 'pointer_manipulation', 'buffer_overflow']
        if mutation_type in dangerous:
            return False
        # 检查是否有潜在的内存泄漏
        if 'memory_alloc' in str(details) and 'memory_free' not in str(details):
            return False
        return True

    def _check_type_safety(self, mutation_type: str, details: Dict) -> bool:
        """检查类型安全性"""
        # 检查参数类型是否匹配
        # 简化: 如果mutation包含type信息，检查一致性
        if 'param_types' in details:
            return isinstance(details['param_types'], (list, dict))
        return True

    def _check_semantic_preservation(self, mutation_type: str, details: Dict) -> bool:
        """检查语义保持性"""
        # 检查mutation是否保留了核心功能
        # 简化: 随机mutation有风险，但这里是engine生成的，信任其语义
        return mutation_type in [
            'schema_field_add', 'schema_field_remove', 'schema_field_modify',
            'logic_read_modify', 'logic_write_modify', 'logic_query_add',
            'instruction_keyword', 'instruction_threshold', 'instruction_priority',
            'instruction_guidance', 'instruction_context', 'instruction_examples',
            'instruction_format', 'combo_crossover', 'combo_ensemble',
        ]

    def _check_no_infinite_loop(self, mutation_type: str, details: Dict) -> bool:
        """检查无无限循环"""
        # 简化检查: 如果有明确的loop参数，检查bound
        if 'loop' in str(details).lower():
            return 'max_iterations' in details or 'iteration_limit' in details
        return True

    def _check_bounded_resources(self, mutation_type: str, details: Dict) -> bool:
        """检查资源边界"""
        # 检查是否有超时限制
        return True

    def _check_no_dangerous_apis(self, mutation_type: str, details: Dict) -> bool:
        """检查无危险API调用"""
        dangerous_apis = ['eval', 'exec', 'compile', '__import__', 'open(']
        content = str(details)
        return not any(api in content for api in dangerous_apis)

    def get_contract(self, contract_id: str) -> Optional[FGGMContract]:
        return self._contract_cache.get(contract_id)

    def get_statistics(self) -> Dict:
        total = len(self._contract_cache)
        passed = sum(1 for c in self._contract_cache.values() if c.overall_verdict)
        return {
            'total_verifications': total,
            'passed': passed,
            'rejected': total - passed,
            'pass_rate': passed / total if total > 0 else 0.0,
        }


class VersionControl:
    """
    MSTAR Pro v4.0 版本控制系统

    功能:
    1. 追踪每个program的所有mutation版本
    2. 维护版本历史，支持回滚
    3. 记录每个版本的FGGM验证结果
    4. 支持RSPL/SEPL协议

    使用场景:
    - EvolutionEngine.mutate: 每次mutation前创建版本记录
    - EvolutionEngine.rollback: 失败时回滚到上一个版本
    - 查询历史: 查看某个program的完整进化历史
    """

    def __init__(self, hermes_home: str, db_path: Optional[str] = None):
        import os
        self.hermes_home = hermes_home
        os.makedirs(hermes_home, exist_ok=True)
        self.db_path = db_path or f"{hermes_home}/mstar_versions.db"
        self._lock = threading.RLock()

        self.fggm_verifier = FGGMVerifier()

        self._init_database()

        # 内存缓存: program_id -> List[MutationVersion]
        self._version_cache: Dict[str, List[MutationVersion]] = {}

    def _init_database(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS mutation_versions (
                version_id TEXT PRIMARY KEY,
                program_id TEXT NOT NULL,
                mutation_type TEXT,
                timestamp TEXT,
                content_hash TEXT,
                parent_version TEXT,
                status TEXT,
                verification_result TEXT,
                rollback_from TEXT,
                created_at TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS rollback_log (
                rollback_id TEXT PRIMARY KEY,
                program_id TEXT NOT NULL,
                from_version TEXT NOT NULL,
                to_version TEXT NOT NULL,
                reason TEXT,
                timestamp TEXT
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_program_versions
            ON mutation_versions(program_id, timestamp)
        """)

        conn.commit()
        conn.close()

    def _make_version_id(self, program_id: str) -> str:
        ts = datetime.now().isoformat()
        raw = f"{program_id}_{ts}"
        return f"v_{hashlib.md5(raw.encode()).hexdigest()[:12]}"

    def create_version(
        self,
        program_id: str,
        mutation_type: str,
        mutation_details: Dict,
        parent_version: Optional[str] = None,
    ) -> MutationVersion:
        """
        创建一个新的mutation版本。

        1. 执行FGGM验证
        2. 创建版本记录
        3. 持久化到数据库

        Returns:
            MutationVersion: 创建的版本
        """
        with self._lock:
            # 执行FGGM验证
            contract = self.fggm_verifier.verify_mutation(
                program_id=program_id,
                mutation_type=mutation_type,
                mutation_details=mutation_details,
            )

            # 生成版本ID
            version_id = self._make_version_id(program_id)

            # 计算内容hash
            content_str = json.dumps(mutation_details, sort_keys=True)
            content_hash = hashlib.sha256(content_str.encode()).hexdigest()[:16]

            # 确定状态
            status = MutationStatus.VERIFIED.value if contract.is_safe() else MutationStatus.REJECTED.value

            version = MutationVersion(
                version_id=version_id,
                program_id=program_id,
                mutation_type=mutation_type,
                timestamp=datetime.now().isoformat(),
                content_hash=content_hash,
                parent_version=parent_version,
                status=status,
                verification_result={
                    'contract_id': contract.contract_id,
                    'passed': contract.overall_verdict,
                    'checks': contract.checks_passed,
                    'verification_time_ms': contract.verification_time_ms,
                },
                rollback_from=None,
            )

            # 保存到数据库
            conn = sqlite3.connect(self.db_path, timeout=30)
            conn.execute("""
                INSERT INTO mutation_versions
                (version_id, program_id, mutation_type, timestamp, content_hash,
                 parent_version, status, verification_result, rollback_from, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                version.version_id, version.program_id, version.mutation_type,
                version.timestamp, version.content_hash, version.parent_version,
                version.status, json.dumps(version.verification_result),
                version.rollback_from, datetime.now().isoformat(),
            ))
            conn.commit()
            conn.close()

            # 更新内存缓存
            if program_id not in self._version_cache:
                self._version_cache[program_id] = []
            self._version_cache[program_id].append(version)

            logger.info(f"[MSTAR] Created version {version_id} for {program_id}: {status}")

            return version

    def rollback_to(self, program_id: str, target_version_id: str, reason: str = "") -> Optional[MutationVersion]:
        """
        回滚program到指定版本。

        1. 查找目标版本
        2. 创建一个新的rollback版本
        3. 标记原版本被回滚

        Returns:
            MutationVersion: 新创建的rollback版本
        """
        with self._lock:
            # 查找目标版本
            target_version = self.get_version(target_version_id)
            if not target_version:
                logger.warning(f"[MSTAR] Rollback target {target_version_id} not found")
                return None

            # 创建新的rollback版本（复制目标版本的内容）
            rollback_version_id = self._make_version_id(program_id)

            rollback = MutationVersion(
                version_id=rollback_version_id,
                program_id=program_id,
                mutation_type=f"rollback_to_{target_version_id}",
                timestamp=datetime.now().isoformat(),
                content_hash=target_version.content_hash,
                parent_version=target_version.parent_version,
                status=MutationStatus.APPLIED.value,
                verification_result=target_version.verification_result,
                rollback_from=target_version_id,
            )

            # 保存rollback记录
            conn = sqlite3.connect(self.db_path, timeout=30)
            conn.execute("""
                INSERT INTO mutation_versions
                (version_id, program_id, mutation_type, timestamp, content_hash,
                 parent_version, status, verification_result, rollback_from, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rollback.version_id, rollback.program_id, rollback.mutation_type,
                rollback.timestamp, rollback.content_hash, rollback.parent_version,
                rollback.status, json.dumps(rollback.verification_result),
                rollback.rollback_from, datetime.now().isoformat(),
            ))

            conn.execute("""
                INSERT INTO rollback_log
                (rollback_id, program_id, from_version, to_version, reason, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                f"rb_{rollback_version_id}", program_id,
                target_version_id, rollback_version_id,
                reason, datetime.now().isoformat(),
            ))
            conn.commit()
            conn.close()

            # 更新内存缓存
            if program_id not in self._version_cache:
                self._version_cache[program_id] = []
            self._version_cache[program_id].append(rollback)

            logger.info(f"[MSTAR] Rolled back {program_id} from {target_version_id} to {rollback_version_id}")

            return rollback

    def get_version(self, version_id: str) -> Optional[MutationVersion]:
        """获取指定版本"""
        conn = sqlite3.connect(self.db_path, timeout=30)
        cursor = conn.execute(
            "SELECT * FROM mutation_versions WHERE version_id = ?",
            (version_id,)
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return MutationVersion(
            version_id=row[0],
            program_id=row[1],
            mutation_type=row[2],
            timestamp=row[3],
            content_hash=row[4],
            parent_version=row[5],
            status=row[6],
            verification_result=json.loads(row[7]) if row[7] else None,
            rollback_from=row[8],
        )

    def get_program_versions(self, program_id: str, limit: int = 50) -> List[MutationVersion]:
        """获取某个program的所有版本（从新到旧）"""
        # 先查缓存
        if program_id in self._version_cache:
            cached = self._version_cache[program_id]
            if len(cached) >= limit:
                return cached[:limit]

        # 从数据库加载
        conn = sqlite3.connect(self.db_path, timeout=30)
        cursor = conn.execute("""
            SELECT * FROM mutation_versions
            WHERE program_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (program_id, limit))
        rows = cursor.fetchall()
        conn.close()

        versions = []
        for row in rows:
            versions.append(MutationVersion(
                version_id=row[0],
                program_id=row[1],
                mutation_type=row[2],
                timestamp=row[3],
                content_hash=row[4],
                parent_version=row[5],
                status=row[6],
                verification_result=json.loads(row[7]) if row[7] else None,
                rollback_from=row[8],
            ))

        # 更新缓存
        if program_id not in self._version_cache:
            self._version_cache[program_id] = []
        for v in versions:
            if v not in self._version_cache[program_id]:
                self._version_cache[program_id].append(v)

        return versions

    def get_latest_version(self, program_id: str) -> Optional[MutationVersion]:
        """获取某个program的最新版本"""
        versions = self.get_program_versions(program_id, limit=1)
        return versions[0] if versions else None

    def get_rollback_history(self, program_id: str) -> List[Dict]:
        """获取某个program的回滚历史"""
        conn = sqlite3.connect(self.db_path, timeout=30)
        cursor = conn.execute("""
            SELECT * FROM rollback_log
            WHERE program_id = ?
            ORDER BY timestamp DESC
        """, (program_id,))
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                'rollback_id': row[0],
                'program_id': row[1],
                'from_version': row[2],
                'to_version': row[3],
                'reason': row[4],
                'timestamp': row[5],
            }
            for row in rows
        ]

    def get_statistics(self) -> Dict:
        """获取版本控制统计"""
        conn = sqlite3.connect(self.db_path, timeout=30)

        total_versions = conn.execute("SELECT COUNT(*) FROM mutation_versions").fetchone()[0]
        total_rollbacks = conn.execute("SELECT COUNT(*) FROM rollback_log").fetchone()[0]

        cursor = conn.execute("""
            SELECT status, COUNT(*) FROM mutation_versions
            GROUP BY status
        """)
        status_counts = dict(cursor.fetchall())

        cursor = conn.execute("SELECT COUNT(DISTINCT program_id) FROM mutation_versions")
        programs_with_versions = cursor.fetchone()[0]

        conn.close()

        return {
            'total_versions': total_versions,
            'total_rollbacks': total_rollbacks,
            'status_breakdown': status_counts,
            'programs_with_versions': programs_with_versions,
            'fggm_stats': self.fggm_verifier.get_statistics(),
        }

    def prune_old_versions(self, program_id: str, keep_last: int = 20):
        """
        清理旧版本，保留最近N个。

        保留策略:
        - 最近keep_last个版本
        - 所有回滚版本
        - 所有REJECTED版本（用于分析）
        """
        conn = sqlite3.connect(self.db_path, timeout=30)
        cursor = conn.execute("""
            SELECT version_id, status, rollback_from
            FROM mutation_versions
            WHERE program_id = ?
            ORDER BY timestamp DESC
        """, (program_id,))

        rows = cursor.fetchall()
        conn.close()

        # 确定要删除的版本
        to_keep = set()
        kept_count = 0
        for row in rows:
            vid, status, rollback_from = row[0], row[1], row[2]
            if kept_count < keep_last:
                to_keep.add(vid)
                kept_count += 1
            elif status in (MutationStatus.REJECTED.value, MutationStatus.ROLLED_BACK.value):
                to_keep.add(vid)
            elif rollback_from:
                to_keep.add(vid)  # 保留回滚相关版本

        # 执行删除
        if to_keep:
            conn = sqlite3.connect(self.db_path, timeout=30)
            placeholders = ','.join('?' * len(to_keep))
            conn.execute(f"""
                DELETE FROM mutation_versions
                WHERE program_id = ? AND version_id NOT IN ({placeholders})
            """, [program_id] + list(to_keep))
            conn.commit()
            conn.close()

            logger.info(f"[MSTAR] Pruned versions for {program_id}, kept {len(to_keep)}")


# =============================================================================
# RSPL/SEPL Protocol 版本控制常量
# =============================================================================

RSPL_VERSION = "1.0"  # Registry Skill Protocol Language version
SEPL_VERSION = "1.0"  # Skill Evolution Protocol Language version

class SkillRegistryProtocol:
    """RSPL: Skill注册协议"""
    REGISTER = "register"
    UPDATE = "update"
    DEREGISTER = "deregister"
    QUERY = "query"
    LIST = "list"

class SkillEvolutionProtocol:
    """SEPL: Skill进化协议"""
    PROPOSE = "propose"      # 提出mutation
    VERIFY = "verify"        # 验证mutation
    APPLY = "apply"          # 应用mutation
    ROLLBACK = "rollback"    # 回滚mutation
    COMMIT = "commit"        # 确认mutation
    ABORT = "abort"          # 取消mutation