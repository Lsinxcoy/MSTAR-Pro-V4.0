"""
MSTAR Pro v4.0 - Phase 7: RSPL/SEPL Protocol Layer
参考: Autogenesis

RSPL (Registry Skill Protocol Language):
- Skill注册、注销、查询协议
- 定义skill的元数据格式
- 版本控制和依赖管理

SEPL (Skill Evolution Protocol Language):
- Skill进化协议
- 定义mutation的提交流程
- 版本控制和回滚协议
"""

from __future__ import annotations
import logging
import json
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# RSPL: Registry Skill Protocol Language
# =============================================================================

class RSPLVersion:
    CURRENT = "1.0"
    MIN_COMPATIBLE = "1.0"


class RSPLMessageType(Enum):
    """RSPL消息类型"""
    REGISTER = "rspl.register"
    UPDATE = "rspl.update"
    DEREGISTER = "rspl.deregister"
    QUERY = "rspl.query"
    LIST = "rspl.list"
    HEARTBEAT = "rspl.heartbeat"
    ERROR = "rspl.error"


@dataclass
class SkillMetadata:
    """Skill元数据"""
    skill_id: str
    name: str
    version: str
    category: str
    author: str
    description: str
    parameters: Dict[str, Any]  # 输入参数定义
    output_schema: Dict[str, Any]  # 输出格式定义
    dependencies: List[str]  # 依赖的其他skills
    created_at: str
    updated_at: str
    status: str = "active"  # active, deprecated, experimental
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'skill_id': self.skill_id,
            'name': self.name,
            'version': self.version,
            'category': self.category,
            'author': self.author,
            'description': self.description,
            'parameters': self.parameters,
            'output_schema': self.output_schema,
            'dependencies': self.dependencies,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'status': self.status,
            'tags': self.tags,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'SkillMetadata':
        return cls(**d)


@dataclass
class RSPLMessage:
    """RSPL协议消息"""
    msg_type: RSPLMessageType
    version: str = RSPLVersion.CURRENT
    message_id: str = ""
    timestamp: str = ""
    source: str = "mstar"
    payload: Dict = field(default_factory=dict)
    error: Optional[str] = None

    def __post_init__(self):
        if not self.message_id:
            import hashlib
            self.message_id = hashlib.md5(
                f"{self.msg_type.value}{datetime.now().isoformat()}".encode()
            ).hexdigest()[:12]
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        result = {
            'msg_type': self.msg_type.value if isinstance(self.msg_type, RSPLMessageType) else self.msg_type,
            'version': self.version,
            'message_id': self.message_id,
            'timestamp': self.timestamp,
            'source': self.source,
            'payload': self.payload,
        }
        if self.error:
            result['error'] = self.error
        return result

    @classmethod
    def from_dict(cls, d: Dict) -> 'RSPLMessage':
        msg_type = RSPLMessageType(d['msg_type'])
        return cls(
            msg_type=msg_type,
            version=d.get('version', RSPLVersion.CURRENT),
            message_id=d.get('message_id', ''),
            timestamp=d.get('timestamp', ''),
            source=d.get('source', 'mstar'),
            payload=d.get('payload', {}),
            error=d.get('error'),
        )


class RSPLHandler:
    """
    RSPL协议处理器

    处理Skill的注册、查询、更新、注销等操作。
    """

    def __init__(self, skill_registry: Optional['SkillRegistry'] = None):
        self.skill_registry = skill_registry or SkillRegistry()

    def handle_register(self, metadata: SkillMetadata) -> RSPLMessage:
        """处理Register请求"""
        try:
            self.skill_registry.register(metadata)
            return RSPLMessage(
                msg_type=RSPLMessageType.REGISTER,
                payload={'skill_id': metadata.skill_id, 'status': 'registered'},
            )
        except Exception as e:
            logger.error(f"[MSTAR] RSPL register failed: {e}")
            return RSPLMessage(
                msg_type=RSPLMessageType.ERROR,
                error=str(e),
                payload={'skill_id': metadata.skill_id},
            )

    def handle_update(self, skill_id: str, updates: Dict) -> RSPLMessage:
        """处理Update请求"""
        try:
            self.skill_registry.update(skill_id, updates)
            return RSPLMessage(
                msg_type=RSPLMessageType.UPDATE,
                payload={'skill_id': skill_id, 'status': 'updated'},
            )
        except Exception as e:
            logger.error(f"[MSTAR] RSPL update failed: {e}")
            return RSPLMessage(
                msg_type=RSPLMessageType.ERROR,
                error=str(e),
                payload={'skill_id': skill_id},
            )

    def handle_deregister(self, skill_id: str) -> RSPLMessage:
        """处理Deregister请求"""
        try:
            self.skill_registry.deregister(skill_id)
            return RSPLMessage(
                msg_type=RSPLMessageType.DEREGISTER,
                payload={'skill_id': skill_id, 'status': 'deregistered'},
            )
        except Exception as e:
            logger.error(f"[MSTAR] RSPL deregister failed: {e}")
            return RSPLMessage(
                msg_type=RSPLMessageType.ERROR,
                error=str(e),
                payload={'skill_id': skill_id},
            )

    def handle_query(self, skill_id: str) -> RSPLMessage:
        """处理Query请求"""
        try:
            metadata = self.skill_registry.get(skill_id)
            if metadata:
                return RSPLMessage(
                    msg_type=RSPLMessageType.QUERY,
                    payload={'skill_id': skill_id, 'metadata': metadata.to_dict()},
                )
            else:
                return RSPLMessage(
                    msg_type=RSPLMessageType.ERROR,
                    error=f"Skill not found: {skill_id}",
                    payload={'skill_id': skill_id},
                )
        except Exception as e:
            logger.error(f"[MSTAR] RSPL query failed: {e}")
            return RSPLMessage(
                msg_type=RSPLMessageType.ERROR,
                error=str(e),
                payload={'skill_id': skill_id},
            )

    def handle_list(self, category: Optional[str] = None) -> RSPLMessage:
        """处理List请求"""
        try:
            skills = self.skill_registry.list(category=category)
            return RSPLMessage(
                msg_type=RSPLMessageType.LIST,
                payload={'skills': [s.to_dict() for s in skills], 'count': len(skills)},
            )
        except Exception as e:
            logger.error(f"[MSTAR] RSPL list failed: {e}")
            return RSPLMessage(
                msg_type=RSPLMessageType.ERROR,
                error=str(e),
            )

    def parse_message(self, raw: Dict) -> RSPLMessage:
        """解析RSPL消息"""
        return RSPLMessage.from_dict(raw)

    def build_response(self, request: RSPLMessage, result: Any) -> RSPLMessage:
        """构建响应消息"""
        return RSPLMessage(
            msg_type=request.msg_type,
            payload={'result': result, 'original_message_id': request.message_id},
        )


# =============================================================================
# Skill Registry (RSPL的后端存储)
# =============================================================================

class SkillRegistry:
    """
    Skill注册表

    维护所有技能的元数据。
    """

    def __init__(self):
        self._skills: Dict[str, SkillMetadata] = {}
        self._lock = threading.RLock()

    def register(self, metadata: SkillMetadata) -> bool:
        """注册一个新Skill"""
        with self._lock:
            if metadata.skill_id in self._skills:
                logger.warning(f"[MSTAR] Skill {metadata.skill_id} already registered, updating")
            metadata.updated_at = datetime.now().isoformat()
            self._skills[metadata.skill_id] = metadata
            logger.info(f"[MSTAR] Registered skill: {metadata.skill_id}")
            return True

    def update(self, skill_id: str, updates: Dict) -> bool:
        """更新Skill元数据"""
        with self._lock:
            if skill_id not in self._skills:
                raise KeyError(f"Skill not found: {skill_id}")
            metadata = self._skills[skill_id]
            for key, value in updates.items():
                if hasattr(metadata, key):
                    setattr(metadata, key, value)
            metadata.updated_at = datetime.now().isoformat()
            return True

    def deregister(self, skill_id: str) -> bool:
        """注销一个Skill"""
        with self._lock:
            if skill_id in self._skills:
                del self._skills[skill_id]
                logger.info(f"[MSTAR] Deregistered skill: {skill_id}")
                return True
            return False

    def get(self, skill_id: str) -> Optional[SkillMetadata]:
        """获取Skill元数据"""
        return self._skills.get(skill_id)

    def list(self, category: Optional[str] = None, status: Optional[str] = None) -> List[SkillMetadata]:
        """列出Skills"""
        result = list(self._skills.values())
        if category:
            result = [s for s in result if s.category == category]
        if status:
            result = [s for s in result if s.status == status]
        return result

    def exists(self, skill_id: str) -> bool:
        """检查Skill是否存在"""
        return skill_id in self._skills


# =============================================================================
# SEPL: Skill Evolution Protocol Language
# =============================================================================

class SEPLVersion:
    CURRENT = "1.0"
    MIN_COMPATIBLE = "1.0"


class SEPLMessageType(Enum):
    """SEPL消息类型"""
    PROPOSE = "sepl.propose"
    VERIFY = "sepl.verify"
    APPLY = "sepl.apply"
    ROLLBACK = "sepl.rollback"
    COMMIT = "sepl.commit"
    ABORT = "sepl.abort"
    STATUS = "sepl.status"
    ERROR = "sepl.error"


@dataclass
class EvolutionProposal:
    """进化提案"""
    proposal_id: str
    program_id: str
    mutation_type: str
    mutation_details: Dict
    proposed_at: str
    proposer: str = "mstar"
    status: str = "proposed"  # proposed, verified, applied, rolled_back, committed, aborted
    verification_result: Optional[Dict] = None
    application_result: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return {
            'proposal_id': self.proposal_id,
            'program_id': self.program_id,
            'mutation_type': self.mutation_type,
            'mutation_details': self.mutation_details,
            'proposed_at': self.proposed_at,
            'proposer': self.proposer,
            'status': self.status,
            'verification_result': self.verification_result,
            'application_result': self.application_result,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'EvolutionProposal':
        return cls(**d)


@dataclass
class SEPLMessage:
    """SEPL协议消息"""
    msg_type: SEPLMessageType
    version: str = SEPLVersion.CURRENT
    message_id: str = ""
    timestamp: str = ""
    source: str = "mstar"
    payload: Dict = field(default_factory=dict)
    error: Optional[str] = None

    def __post_init__(self):
        if not self.message_id:
            import hashlib
            self.message_id = hashlib.md5(
                f"{self.msg_type.value}{datetime.now().isoformat()}".encode()
            ).hexdigest()[:12]
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        result = {
            'msg_type': self.msg_type.value if isinstance(self.msg_type, SEPLMessageType) else self.msg_type,
            'version': self.version,
            'message_id': self.message_id,
            'timestamp': self.timestamp,
            'source': self.source,
            'payload': self.payload,
        }
        if self.error:
            result['error'] = self.error
        return result

    @classmethod
    def from_dict(cls, d: Dict) -> 'SEPLMessage':
        msg_type = SEPLMessageType(d['msg_type'])
        return cls(
            msg_type=msg_type,
            version=d.get('version', SEPLVersion.CURRENT),
            message_id=d.get('message_id', ''),
            timestamp=d.get('timestamp', ''),
            source=d.get('source', 'mstar'),
            payload=d.get('payload', {}),
            error=d.get('error'),
        )


class SEPLHandler:
    """
    SEPL协议处理器

    处理Skill进化的完整生命周期:
    1. Propose: 提出mutation
    2. Verify: FGGM验证
    3. Apply: 应用mutation
    4. Commit/Rollback: 确认或回滚
    """

    def __init__(
        self,
        version_control: Optional['VersionControl'] = None,
        fggm_verifier: Optional['FGGMVerifier'] = None,
    ):
        self.version_control = version_control
        self.fggm_verifier = fggm_verifier
        self._proposals: Dict[str, EvolutionProposal] = {}
        self._lock = threading.RLock()

    def handle_propose(self, proposal: EvolutionProposal) -> SEPLMessage:
        """处理Propose请求"""
        try:
            with self._lock:
                proposal.status = 'proposed'
                self._proposals[proposal.proposal_id] = proposal

            logger.info(f"[MSTAR] SEPL proposal created: {proposal.proposal_id}")

            return SEPLMessage(
                msg_type=SEPLMessageType.PROPOSE,
                payload={'proposal_id': proposal.proposal_id, 'status': 'proposed'},
            )
        except Exception as e:
            logger.error(f"[MSTAR] SEPL propose failed: {e}")
            return SEPLMessage(
                msg_type=SEPLMessageType.ERROR,
                error=str(e),
                payload={'proposal_id': proposal.proposal_id},
            )

    def handle_verify(self, proposal_id: str) -> SEPLMessage:
        """处理Verify请求"""
        try:
            with self._lock:
                proposal = self._proposals.get(proposal_id)
                if not proposal:
                    raise KeyError(f"Proposal not found: {proposal_id}")

            # 执行FGGM验证
            if self.fggm_verifier:
                contract = self.fggm_verifier.verify_mutation(
                    program_id=proposal.program_id,
                    mutation_type=proposal.mutation_type,
                    mutation_details=proposal.mutation_details,
                )
                proposal.verification_result = contract.to_dict() if hasattr(contract, 'to_dict') else {
                    'passed': contract.overall_verdict if hasattr(contract, 'overall_verdict') else False,
                }
                proposal.status = 'verified' if contract.overall_verdict else 'rejected'

                logger.info(f"[MSTAR] SEPL verification {'passed' if contract.overall_verdict else 'failed'}: {proposal_id}")

                return SEPLMessage(
                    msg_type=SEPLMessageType.VERIFY,
                    payload={
                        'proposal_id': proposal_id,
                        'status': proposal.status,
                        'verification_result': proposal.verification_result,
                    },
                )
            else:
                proposal.status = 'verified'
                return SEPLMessage(
                    msg_type=SEPLMessageType.VERIFY,
                    payload={'proposal_id': proposal_id, 'status': 'verified'},
                )

        except Exception as e:
            logger.error(f"[MSTAR] SEPL verify failed: {e}")
            return SEPLMessage(
                msg_type=SEPLMessageType.ERROR,
                error=str(e),
                payload={'proposal_id': proposal_id},
            )

    def handle_apply(self, proposal_id: str) -> SEPLMessage:
        """处理Apply请求"""
        try:
            with self._lock:
                proposal = self._proposals.get(proposal_id)
                if not proposal:
                    raise KeyError(f"Proposal not found: {proposal_id}")

                if proposal.status not in ('proposed', 'verified'):
                    raise ValueError(f"Cannot apply proposal in status: {proposal.status}")

            # 使用version control应用mutation
            if self.version_control:
                version = self.version_control.create_version(
                    program_id=proposal.program_id,
                    mutation_type=proposal.mutation_type,
                    mutation_details=proposal.mutation_details,
                )
                proposal.application_result = {'version_id': version.version_id}
                proposal.status = 'applied'

                logger.info(f"[MSTAR] SEPL applied: {proposal_id} -> version {version.version_id}")
            else:
                proposal.status = 'applied'
                proposal.application_result = {'applied': True}

            return SEPLMessage(
                msg_type=SEPLMessageType.APPLY,
                payload={
                    'proposal_id': proposal_id,
                    'status': 'applied',
                    'result': proposal.application_result,
                },
            )

        except Exception as e:
            logger.error(f"[MSTAR] SEPL apply failed: {e}")
            return SEPLMessage(
                msg_type=SEPLMessageType.ERROR,
                error=str(e),
                payload={'proposal_id': proposal_id},
            )

    def handle_rollback(self, proposal_id: str, reason: str = "") -> SEPLMessage:
        """处理Rollback请求"""
        try:
            with self._lock:
                proposal = self._proposals.get(proposal_id)
                if not proposal:
                    raise KeyError(f"Proposal not found: {proposal_id}")

                app_result = proposal.application_result or {}
                version_id = app_result.get('version_id')

                if self.version_control and version_id:
                    # 回滚到上一个版本
                    target_version = self.version_control.get_version(version_id)
                    if target_version and target_version.parent_version:
                        self.version_control.rollback_to(
                            program_id=proposal.program_id,
                            target_version_id=target_version.parent_version,
                            reason=f"SEPL rollback: {reason}",
                        )

                proposal.status = 'rolled_back'

                logger.info(f"[MSTAR] SEPL rolled back: {proposal_id}")

                return SEPLMessage(
                    msg_type=SEPLMessageType.ROLLBACK,
                    payload={'proposal_id': proposal_id, 'status': 'rolled_back'},
                )

        except Exception as e:
            logger.error(f"[MSTAR] SEPL rollback failed: {e}")
            return SEPLMessage(
                msg_type=SEPLMessageType.ERROR,
                error=str(e),
                payload={'proposal_id': proposal_id},
            )

    def handle_commit(self, proposal_id: str) -> SEPLMessage:
        """处理Commit请求（确认mutation）"""
        try:
            with self._lock:
                proposal = self._proposals.get(proposal_id)
                if not proposal:
                    raise KeyError(f"Proposal not found: {proposal_id}")

                if proposal.status != 'applied':
                    raise ValueError(f"Cannot commit proposal in status: {proposal.status}")

                proposal.status = 'committed'

                logger.info(f"[MSTAR] SEPL committed: {proposal_id}")

                return SEPLMessage(
                    msg_type=SEPLMessageType.COMMIT,
                    payload={'proposal_id': proposal_id, 'status': 'committed'},
                )

        except Exception as e:
            logger.error(f"[MSTAR] SEPL commit failed: {e}")
            return SEPLMessage(
                msg_type=SEPLMessageType.ERROR,
                error=str(e),
                payload={'proposal_id': proposal_id},
            )

    def handle_abort(self, proposal_id: str, reason: str = "") -> SEPLMessage:
        """处理Abort请求（放弃proposal）"""
        try:
            with self._lock:
                proposal = self._proposals.get(proposal_id)
                if not proposal:
                    raise KeyError(f"Proposal not found: {proposal_id}")

                proposal.status = 'aborted'

                logger.info(f"[MSTAR] SEPL aborted: {proposal_id} - {reason}")

                return SEPLMessage(
                    msg_type=SEPLMessageType.ABORT,
                    payload={'proposal_id': proposal_id, 'status': 'aborted', 'reason': reason},
                )

        except Exception as e:
            logger.error(f"[MSTAR] SEPL abort failed: {e}")
            return SEPLMessage(
                msg_type=SEPLMessageType.ERROR,
                error=str(e),
                payload={'proposal_id': proposal_id},
            )

    def get_proposal_status(self, proposal_id: str) -> Optional[str]:
        """获取proposal状态"""
        proposal = self._proposals.get(proposal_id)
        return proposal.status if proposal else None

    def list_proposals(self, status: Optional[str] = None) -> List[EvolutionProposal]:
        """列出proposals"""
        if status:
            return [p for p in self._proposals.values() if p.status == status]
        return list(self._proposals.values())

    def parse_message(self, raw: Dict) -> SEPLMessage:
        """解析SEPL消息"""
        return SEPLMessage.from_dict(raw)


# =============================================================================
# Protocol Constants
# =============================================================================

PROTOCOL_VERSION = "2.0"
RSPL_VERSION_CURRENT = "1.0"
SEPL_VERSION_CURRENT = "1.0"

# Protocol超时设置（秒）
PROTOCOL_TIMEOUT = {
    'verify': 30,
    'apply': 60,
    'rollback': 30,
    'commit': 10,
}

# Protocol重试次数
PROTOCOL_MAX_RETRIES = 3