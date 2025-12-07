"""
User Authentication and Authorization System for CYRP
穿黄工程用户认证与权限管理系统

功能:
- 用户认证(用户名/密码, 证书, 双因素)
- 角色权限管理(RBAC)
- 会话管理
- 审计日志
- 密码策略
"""

import asyncio
import hashlib
import hmac
import json
import os
import secrets
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, Flag, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import base64


class Permission(Flag):
    """权限标志"""
    NONE = 0

    # 查看权限
    VIEW_DASHBOARD = auto()
    VIEW_TRENDS = auto()
    VIEW_ALARMS = auto()
    VIEW_REPORTS = auto()
    VIEW_EQUIPMENT = auto()
    VIEW_SETTINGS = auto()

    # 操作权限
    ACKNOWLEDGE_ALARM = auto()
    OPERATE_VALVE = auto()
    OPERATE_PUMP = auto()
    START_STOP_EQUIPMENT = auto()
    MODIFY_SETPOINTS = auto()

    # 管理权限
    MANAGE_USERS = auto()
    MANAGE_ROLES = auto()
    MANAGE_ALARMS = auto()
    MANAGE_REPORTS = auto()
    MANAGE_EQUIPMENT = auto()
    MANAGE_SYSTEM = auto()

    # 审计权限
    VIEW_AUDIT_LOG = auto()
    EXPORT_DATA = auto()

    # 组合权限
    VIEWER = VIEW_DASHBOARD | VIEW_TRENDS | VIEW_ALARMS | VIEW_REPORTS | VIEW_EQUIPMENT
    OPERATOR = VIEWER | ACKNOWLEDGE_ALARM | OPERATE_VALVE | OPERATE_PUMP | START_STOP_EQUIPMENT
    ENGINEER = OPERATOR | MODIFY_SETPOINTS | MANAGE_ALARMS | MANAGE_REPORTS | VIEW_SETTINGS
    ADMIN = ENGINEER | MANAGE_USERS | MANAGE_ROLES | MANAGE_EQUIPMENT | MANAGE_SYSTEM | VIEW_AUDIT_LOG | EXPORT_DATA


class AuthenticationMethod(Enum):
    """认证方式"""
    PASSWORD = auto()
    CERTIFICATE = auto()
    TWO_FACTOR = auto()
    LDAP = auto()
    SSO = auto()


class SessionState(Enum):
    """会话状态"""
    ACTIVE = auto()
    EXPIRED = auto()
    LOCKED = auto()
    TERMINATED = auto()


class AuditEventType(Enum):
    """审计事件类型"""
    LOGIN_SUCCESS = auto()
    LOGIN_FAILURE = auto()
    LOGOUT = auto()
    SESSION_EXPIRED = auto()
    PASSWORD_CHANGE = auto()
    PERMISSION_DENIED = auto()
    OPERATION_PERFORMED = auto()
    CONFIGURATION_CHANGE = auto()
    USER_CREATED = auto()
    USER_MODIFIED = auto()
    USER_DELETED = auto()
    ROLE_CREATED = auto()
    ROLE_MODIFIED = auto()
    ROLE_DELETED = auto()


@dataclass
class PasswordPolicy:
    """密码策略"""
    min_length: int = 8
    max_length: int = 128
    require_uppercase: bool = True
    require_lowercase: bool = True
    require_digit: bool = True
    require_special: bool = True
    special_chars: str = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    max_age_days: int = 90
    history_count: int = 5  # 不能重复使用最近N个密码
    max_failed_attempts: int = 5
    lockout_duration_minutes: int = 30

    def validate(self, password: str) -> Tuple[bool, List[str]]:
        """验证密码是否符合策略"""
        errors = []

        if len(password) < self.min_length:
            errors.append(f"密码长度至少{self.min_length}位")
        if len(password) > self.max_length:
            errors.append(f"密码长度不能超过{self.max_length}位")
        if self.require_uppercase and not any(c.isupper() for c in password):
            errors.append("密码必须包含大写字母")
        if self.require_lowercase and not any(c.islower() for c in password):
            errors.append("密码必须包含小写字母")
        if self.require_digit and not any(c.isdigit() for c in password):
            errors.append("密码必须包含数字")
        if self.require_special and not any(c in self.special_chars for c in password):
            errors.append(f"密码必须包含特殊字符({self.special_chars})")

        return len(errors) == 0, errors


@dataclass
class Role:
    """角色"""
    role_id: str
    name: str
    description: str
    permissions: Permission
    is_system: bool = False  # 系统内置角色不可删除
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def has_permission(self, permission: Permission) -> bool:
        """检查是否有权限"""
        return (self.permissions & permission) == permission


@dataclass
class User:
    """用户"""
    user_id: str
    username: str
    display_name: str
    email: str
    password_hash: str
    salt: str
    roles: List[str] = field(default_factory=list)
    is_active: bool = True
    is_locked: bool = False
    locked_until: Optional[datetime] = None
    failed_login_attempts: int = 0
    last_login: Optional[datetime] = None
    password_changed_at: datetime = field(default_factory=datetime.now)
    password_history: List[str] = field(default_factory=list)
    two_factor_enabled: bool = False
    two_factor_secret: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """会话"""
    session_id: str
    user_id: str
    username: str
    created_at: datetime
    expires_at: datetime
    last_activity: datetime
    ip_address: str
    user_agent: str
    state: SessionState = SessionState.ACTIVE
    permissions: Permission = Permission.NONE
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """检查会话是否有效"""
        if self.state != SessionState.ACTIVE:
            return False
        if datetime.now() > self.expires_at:
            return False
        return True


@dataclass
class AuditLogEntry:
    """审计日志条目"""
    entry_id: str
    timestamp: datetime
    event_type: AuditEventType
    user_id: Optional[str]
    username: Optional[str]
    session_id: Optional[str]
    ip_address: Optional[str]
    resource: Optional[str]
    action: Optional[str]
    result: str  # success/failure
    details: Dict[str, Any] = field(default_factory=dict)


class PasswordHasher:
    """密码哈希器"""

    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """哈希密码"""
        if salt is None:
            salt = secrets.token_hex(32)

        # 使用PBKDF2-SHA256
        key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            iterations=100000,
            dklen=32
        )

        return base64.b64encode(key).decode('utf-8'), salt

    @staticmethod
    def verify_password(password: str, password_hash: str, salt: str) -> bool:
        """验证密码"""
        computed_hash, _ = PasswordHasher.hash_password(password, salt)
        return hmac.compare_digest(computed_hash, password_hash)


class TOTPGenerator:
    """TOTP(基于时间的一次性密码)生成器"""

    @staticmethod
    def generate_secret() -> str:
        """生成TOTP密钥"""
        return base64.b32encode(secrets.token_bytes(20)).decode('utf-8')

    @staticmethod
    def generate_totp(secret: str, timestamp: Optional[int] = None) -> str:
        """生成TOTP验证码"""
        if timestamp is None:
            timestamp = int(time.time())

        # 30秒时间步长
        counter = timestamp // 30

        # 解码密钥
        key = base64.b32decode(secret.upper())

        # 计算HMAC-SHA1
        counter_bytes = counter.to_bytes(8, 'big')
        hmac_hash = hmac.new(key, counter_bytes, 'sha1').digest()

        # 动态截断
        offset = hmac_hash[-1] & 0x0F
        code = ((hmac_hash[offset] & 0x7F) << 24 |
                (hmac_hash[offset + 1] & 0xFF) << 16 |
                (hmac_hash[offset + 2] & 0xFF) << 8 |
                (hmac_hash[offset + 3] & 0xFF))

        return str(code % 1000000).zfill(6)

    @staticmethod
    def verify_totp(secret: str, code: str, window: int = 1) -> bool:
        """验证TOTP验证码"""
        current_time = int(time.time())

        for i in range(-window, window + 1):
            timestamp = current_time + (i * 30)
            expected_code = TOTPGenerator.generate_totp(secret, timestamp)
            if hmac.compare_digest(code, expected_code):
                return True

        return False


class SessionManager:
    """会话管理器"""

    def __init__(
        self,
        session_timeout_minutes: int = 30,
        max_sessions_per_user: int = 5
    ):
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        self.max_sessions_per_user = max_sessions_per_user
        self._sessions: Dict[str, Session] = {}
        self._user_sessions: Dict[str, List[str]] = {}
        self._lock = asyncio.Lock()

    async def create_session(
        self,
        user: User,
        permissions: Permission,
        ip_address: str,
        user_agent: str
    ) -> Session:
        """创建会话"""
        async with self._lock:
            # 检查用户会话数量
            user_sessions = self._user_sessions.get(user.user_id, [])
            if len(user_sessions) >= self.max_sessions_per_user:
                # 移除最旧的会话
                oldest_session_id = user_sessions[0]
                await self._terminate_session(oldest_session_id)

            # 创建新会话
            session_id = secrets.token_urlsafe(32)
            now = datetime.now()

            session = Session(
                session_id=session_id,
                user_id=user.user_id,
                username=user.username,
                created_at=now,
                expires_at=now + self.session_timeout,
                last_activity=now,
                ip_address=ip_address,
                user_agent=user_agent,
                permissions=permissions
            )

            self._sessions[session_id] = session

            if user.user_id not in self._user_sessions:
                self._user_sessions[user.user_id] = []
            self._user_sessions[user.user_id].append(session_id)

            return session

    async def get_session(self, session_id: str) -> Optional[Session]:
        """获取会话"""
        session = self._sessions.get(session_id)
        if session and session.is_valid():
            return session
        return None

    async def refresh_session(self, session_id: str) -> bool:
        """刷新会话"""
        async with self._lock:
            session = self._sessions.get(session_id)
            if session and session.is_valid():
                now = datetime.now()
                session.last_activity = now
                session.expires_at = now + self.session_timeout
                return True
            return False

    async def terminate_session(self, session_id: str):
        """终止会话"""
        async with self._lock:
            await self._terminate_session(session_id)

    async def _terminate_session(self, session_id: str):
        """内部终止会话"""
        session = self._sessions.get(session_id)
        if session:
            session.state = SessionState.TERMINATED
            del self._sessions[session_id]

            if session.user_id in self._user_sessions:
                if session_id in self._user_sessions[session.user_id]:
                    self._user_sessions[session.user_id].remove(session_id)

    async def terminate_user_sessions(self, user_id: str):
        """终止用户所有会话"""
        async with self._lock:
            session_ids = self._user_sessions.get(user_id, []).copy()
            for session_id in session_ids:
                await self._terminate_session(session_id)

    async def cleanup_expired_sessions(self):
        """清理过期会话"""
        async with self._lock:
            expired = [
                sid for sid, session in self._sessions.items()
                if not session.is_valid()
            ]
            for session_id in expired:
                await self._terminate_session(session_id)


class AuditLogger:
    """审计日志记录器"""

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self._entries: List[AuditLogEntry] = []
        self._lock = asyncio.Lock()

    async def log(
        self,
        event_type: AuditEventType,
        result: str,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """记录审计日志"""
        entry = AuditLogEntry(
            entry_id=secrets.token_hex(16),
            timestamp=datetime.now(),
            event_type=event_type,
            user_id=user_id,
            username=username,
            session_id=session_id,
            ip_address=ip_address,
            resource=resource,
            action=action,
            result=result,
            details=details or {}
        )

        async with self._lock:
            self._entries.append(entry)

            # 写入文件
            if self.log_file:
                try:
                    with open(self.log_file, 'a', encoding='utf-8') as f:
                        log_line = json.dumps({
                            "entry_id": entry.entry_id,
                            "timestamp": entry.timestamp.isoformat(),
                            "event_type": entry.event_type.name,
                            "user_id": entry.user_id,
                            "username": entry.username,
                            "session_id": entry.session_id,
                            "ip_address": entry.ip_address,
                            "resource": entry.resource,
                            "action": entry.action,
                            "result": entry.result,
                            "details": entry.details
                        }, ensure_ascii=False)
                        f.write(log_line + '\n')
                except Exception:
                    pass

    async def query(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None,
        user_id: Optional[str] = None,
        limit: int = 1000
    ) -> List[AuditLogEntry]:
        """查询审计日志"""
        async with self._lock:
            entries = self._entries

            if start_time:
                entries = [e for e in entries if e.timestamp >= start_time]
            if end_time:
                entries = [e for e in entries if e.timestamp <= end_time]
            if event_types:
                entries = [e for e in entries if e.event_type in event_types]
            if user_id:
                entries = [e for e in entries if e.user_id == user_id]

            return entries[-limit:]


class UserStore(ABC):
    """用户存储基类"""

    @abstractmethod
    async def get_user(self, user_id: str) -> Optional[User]:
        """获取用户"""
        pass

    @abstractmethod
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """通过用户名获取用户"""
        pass

    @abstractmethod
    async def save_user(self, user: User):
        """保存用户"""
        pass

    @abstractmethod
    async def delete_user(self, user_id: str):
        """删除用户"""
        pass

    @abstractmethod
    async def list_users(self) -> List[User]:
        """列出所有用户"""
        pass


class InMemoryUserStore(UserStore):
    """内存用户存储"""

    def __init__(self):
        self._users: Dict[str, User] = {}
        self._username_index: Dict[str, str] = {}

    async def get_user(self, user_id: str) -> Optional[User]:
        return self._users.get(user_id)

    async def get_user_by_username(self, username: str) -> Optional[User]:
        user_id = self._username_index.get(username.lower())
        if user_id:
            return self._users.get(user_id)
        return None

    async def save_user(self, user: User):
        self._users[user.user_id] = user
        self._username_index[user.username.lower()] = user.user_id

    async def delete_user(self, user_id: str):
        user = self._users.get(user_id)
        if user:
            del self._users[user_id]
            if user.username.lower() in self._username_index:
                del self._username_index[user.username.lower()]

    async def list_users(self) -> List[User]:
        return list(self._users.values())


class RoleStore(ABC):
    """角色存储基类"""

    @abstractmethod
    async def get_role(self, role_id: str) -> Optional[Role]:
        """获取角色"""
        pass

    @abstractmethod
    async def save_role(self, role: Role):
        """保存角色"""
        pass

    @abstractmethod
    async def delete_role(self, role_id: str):
        """删除角色"""
        pass

    @abstractmethod
    async def list_roles(self) -> List[Role]:
        """列出所有角色"""
        pass


class InMemoryRoleStore(RoleStore):
    """内存角色存储"""

    def __init__(self):
        self._roles: Dict[str, Role] = {}

    async def get_role(self, role_id: str) -> Optional[Role]:
        return self._roles.get(role_id)

    async def save_role(self, role: Role):
        self._roles[role.role_id] = role

    async def delete_role(self, role_id: str):
        role = self._roles.get(role_id)
        if role and not role.is_system:
            del self._roles[role_id]

    async def list_roles(self) -> List[Role]:
        return list(self._roles.values())


class AuthenticationManager:
    """认证管理器"""

    def __init__(
        self,
        user_store: UserStore,
        role_store: RoleStore,
        session_manager: SessionManager,
        audit_logger: AuditLogger,
        password_policy: Optional[PasswordPolicy] = None
    ):
        self.user_store = user_store
        self.role_store = role_store
        self.session_manager = session_manager
        self.audit_logger = audit_logger
        self.password_policy = password_policy or PasswordPolicy()

    async def authenticate(
        self,
        username: str,
        password: str,
        ip_address: str = "unknown",
        user_agent: str = "unknown",
        totp_code: Optional[str] = None
    ) -> Tuple[bool, Optional[Session], str]:
        """用户认证"""
        user = await self.user_store.get_user_by_username(username)

        if not user:
            await self.audit_logger.log(
                AuditEventType.LOGIN_FAILURE,
                "failure",
                username=username,
                ip_address=ip_address,
                details={"reason": "用户不存在"}
            )
            return False, None, "用户名或密码错误"

        # 检查用户状态
        if not user.is_active:
            await self.audit_logger.log(
                AuditEventType.LOGIN_FAILURE,
                "failure",
                user_id=user.user_id,
                username=username,
                ip_address=ip_address,
                details={"reason": "用户已禁用"}
            )
            return False, None, "用户已被禁用"

        # 检查锁定状态
        if user.is_locked:
            if user.locked_until and datetime.now() < user.locked_until:
                remaining = (user.locked_until - datetime.now()).seconds // 60
                return False, None, f"账户已锁定，请{remaining}分钟后重试"
            else:
                # 解锁
                user.is_locked = False
                user.locked_until = None
                user.failed_login_attempts = 0
                await self.user_store.save_user(user)

        # 验证密码
        if not PasswordHasher.verify_password(password, user.password_hash, user.salt):
            user.failed_login_attempts += 1

            if user.failed_login_attempts >= self.password_policy.max_failed_attempts:
                user.is_locked = True
                user.locked_until = datetime.now() + timedelta(
                    minutes=self.password_policy.lockout_duration_minutes
                )

            await self.user_store.save_user(user)

            await self.audit_logger.log(
                AuditEventType.LOGIN_FAILURE,
                "failure",
                user_id=user.user_id,
                username=username,
                ip_address=ip_address,
                details={"reason": "密码错误", "attempts": user.failed_login_attempts}
            )
            return False, None, "用户名或密码错误"

        # 验证双因素认证
        if user.two_factor_enabled:
            if not totp_code:
                return False, None, "请输入双因素验证码"

            if not user.two_factor_secret:
                return False, None, "双因素认证配置错误"

            if not TOTPGenerator.verify_totp(user.two_factor_secret, totp_code):
                await self.audit_logger.log(
                    AuditEventType.LOGIN_FAILURE,
                    "failure",
                    user_id=user.user_id,
                    username=username,
                    ip_address=ip_address,
                    details={"reason": "双因素验证码错误"}
                )
                return False, None, "双因素验证码错误"

        # 重置失败计数
        user.failed_login_attempts = 0
        user.last_login = datetime.now()
        await self.user_store.save_user(user)

        # 获取用户权限
        permissions = await self._get_user_permissions(user)

        # 创建会话
        session = await self.session_manager.create_session(
            user, permissions, ip_address, user_agent
        )

        await self.audit_logger.log(
            AuditEventType.LOGIN_SUCCESS,
            "success",
            user_id=user.user_id,
            username=username,
            session_id=session.session_id,
            ip_address=ip_address
        )

        return True, session, "登录成功"

    async def logout(self, session_id: str):
        """用户登出"""
        session = await self.session_manager.get_session(session_id)
        if session:
            await self.audit_logger.log(
                AuditEventType.LOGOUT,
                "success",
                user_id=session.user_id,
                username=session.username,
                session_id=session_id,
                ip_address=session.ip_address
            )
            await self.session_manager.terminate_session(session_id)

    async def change_password(
        self,
        user_id: str,
        old_password: str,
        new_password: str,
        session: Optional[Session] = None
    ) -> Tuple[bool, str]:
        """修改密码"""
        user = await self.user_store.get_user(user_id)
        if not user:
            return False, "用户不存在"

        # 验证旧密码
        if not PasswordHasher.verify_password(old_password, user.password_hash, user.salt):
            return False, "原密码错误"

        # 验证新密码策略
        valid, errors = self.password_policy.validate(new_password)
        if not valid:
            return False, "; ".join(errors)

        # 检查密码历史
        for old_hash in user.password_history[-self.password_policy.history_count:]:
            if PasswordHasher.verify_password(new_password, old_hash, user.salt):
                return False, f"不能使用最近{self.password_policy.history_count}个使用过的密码"

        # 更新密码
        new_hash, new_salt = PasswordHasher.hash_password(new_password)
        user.password_history.append(user.password_hash)
        user.password_hash = new_hash
        user.salt = new_salt
        user.password_changed_at = datetime.now()
        await self.user_store.save_user(user)

        await self.audit_logger.log(
            AuditEventType.PASSWORD_CHANGE,
            "success",
            user_id=user_id,
            username=user.username,
            session_id=session.session_id if session else None,
            ip_address=session.ip_address if session else None
        )

        return True, "密码修改成功"

    async def _get_user_permissions(self, user: User) -> Permission:
        """获取用户权限"""
        permissions = Permission.NONE

        for role_id in user.roles:
            role = await self.role_store.get_role(role_id)
            if role:
                permissions |= role.permissions

        return permissions


class AuthorizationManager:
    """授权管理器"""

    def __init__(
        self,
        session_manager: SessionManager,
        audit_logger: AuditLogger
    ):
        self.session_manager = session_manager
        self.audit_logger = audit_logger

    async def check_permission(
        self,
        session_id: str,
        required_permission: Permission,
        resource: Optional[str] = None
    ) -> bool:
        """检查权限"""
        session = await self.session_manager.get_session(session_id)
        if not session:
            return False

        has_permission = (session.permissions & required_permission) == required_permission

        if not has_permission:
            await self.audit_logger.log(
                AuditEventType.PERMISSION_DENIED,
                "failure",
                user_id=session.user_id,
                username=session.username,
                session_id=session_id,
                ip_address=session.ip_address,
                resource=resource,
                details={"required": required_permission.name}
            )

        return has_permission

    async def log_operation(
        self,
        session_id: str,
        resource: str,
        action: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """记录操作"""
        session = await self.session_manager.get_session(session_id)
        if session:
            await self.audit_logger.log(
                AuditEventType.OPERATION_PERFORMED,
                "success",
                user_id=session.user_id,
                username=session.username,
                session_id=session_id,
                ip_address=session.ip_address,
                resource=resource,
                action=action,
                details=details
            )


class UserManager:
    """用户管理器"""

    def __init__(
        self,
        user_store: UserStore,
        role_store: RoleStore,
        password_policy: PasswordPolicy,
        audit_logger: AuditLogger
    ):
        self.user_store = user_store
        self.role_store = role_store
        self.password_policy = password_policy
        self.audit_logger = audit_logger

    async def create_user(
        self,
        username: str,
        password: str,
        display_name: str,
        email: str,
        roles: List[str],
        operator_session: Optional[Session] = None
    ) -> Tuple[bool, Optional[User], str]:
        """创建用户"""
        # 检查用户名是否存在
        existing = await self.user_store.get_user_by_username(username)
        if existing:
            return False, None, "用户名已存在"

        # 验证密码策略
        valid, errors = self.password_policy.validate(password)
        if not valid:
            return False, None, "; ".join(errors)

        # 验证角色
        for role_id in roles:
            role = await self.role_store.get_role(role_id)
            if not role:
                return False, None, f"角色不存在: {role_id}"

        # 创建用户
        password_hash, salt = PasswordHasher.hash_password(password)
        user = User(
            user_id=secrets.token_hex(16),
            username=username,
            display_name=display_name,
            email=email,
            password_hash=password_hash,
            salt=salt,
            roles=roles
        )

        await self.user_store.save_user(user)

        await self.audit_logger.log(
            AuditEventType.USER_CREATED,
            "success",
            user_id=operator_session.user_id if operator_session else None,
            username=operator_session.username if operator_session else None,
            session_id=operator_session.session_id if operator_session else None,
            resource=f"user:{user.user_id}",
            details={"created_user": username}
        )

        return True, user, "用户创建成功"

    async def update_user(
        self,
        user_id: str,
        display_name: Optional[str] = None,
        email: Optional[str] = None,
        roles: Optional[List[str]] = None,
        is_active: Optional[bool] = None,
        operator_session: Optional[Session] = None
    ) -> Tuple[bool, str]:
        """更新用户"""
        user = await self.user_store.get_user(user_id)
        if not user:
            return False, "用户不存在"

        if display_name:
            user.display_name = display_name
        if email:
            user.email = email
        if roles is not None:
            # 验证角色
            for role_id in roles:
                role = await self.role_store.get_role(role_id)
                if not role:
                    return False, f"角色不存在: {role_id}"
            user.roles = roles
        if is_active is not None:
            user.is_active = is_active

        user.updated_at = datetime.now()
        await self.user_store.save_user(user)

        await self.audit_logger.log(
            AuditEventType.USER_MODIFIED,
            "success",
            user_id=operator_session.user_id if operator_session else None,
            username=operator_session.username if operator_session else None,
            session_id=operator_session.session_id if operator_session else None,
            resource=f"user:{user_id}",
            details={"modified_user": user.username}
        )

        return True, "用户更新成功"

    async def delete_user(
        self,
        user_id: str,
        operator_session: Optional[Session] = None
    ) -> Tuple[bool, str]:
        """删除用户"""
        user = await self.user_store.get_user(user_id)
        if not user:
            return False, "用户不存在"

        await self.user_store.delete_user(user_id)

        await self.audit_logger.log(
            AuditEventType.USER_DELETED,
            "success",
            user_id=operator_session.user_id if operator_session else None,
            username=operator_session.username if operator_session else None,
            session_id=operator_session.session_id if operator_session else None,
            resource=f"user:{user_id}",
            details={"deleted_user": user.username}
        )

        return True, "用户删除成功"

    async def enable_two_factor(
        self,
        user_id: str
    ) -> Tuple[bool, Optional[str], str]:
        """启用双因素认证"""
        user = await self.user_store.get_user(user_id)
        if not user:
            return False, None, "用户不存在"

        secret = TOTPGenerator.generate_secret()
        user.two_factor_secret = secret
        user.two_factor_enabled = True
        await self.user_store.save_user(user)

        return True, secret, "双因素认证已启用"

    async def disable_two_factor(
        self,
        user_id: str,
        operator_session: Optional[Session] = None
    ) -> Tuple[bool, str]:
        """禁用双因素认证"""
        user = await self.user_store.get_user(user_id)
        if not user:
            return False, "用户不存在"

        user.two_factor_enabled = False
        user.two_factor_secret = None
        await self.user_store.save_user(user)

        return True, "双因素认证已禁用"


class SecurityManager:
    """安全管理器(统一入口)"""

    def __init__(self):
        self.user_store = InMemoryUserStore()
        self.role_store = InMemoryRoleStore()
        self.password_policy = PasswordPolicy()
        self.session_manager = SessionManager()
        self.audit_logger = AuditLogger()

        self.auth_manager = AuthenticationManager(
            self.user_store,
            self.role_store,
            self.session_manager,
            self.audit_logger,
            self.password_policy
        )

        self.authz_manager = AuthorizationManager(
            self.session_manager,
            self.audit_logger
        )

        self.user_manager = UserManager(
            self.user_store,
            self.role_store,
            self.password_policy,
            self.audit_logger
        )

    async def initialize_default_roles(self):
        """初始化默认角色"""
        default_roles = [
            Role(
                role_id="viewer",
                name="查看员",
                description="只能查看系统信息",
                permissions=Permission.VIEWER,
                is_system=True
            ),
            Role(
                role_id="operator",
                name="操作员",
                description="可以进行日常操作",
                permissions=Permission.OPERATOR,
                is_system=True
            ),
            Role(
                role_id="engineer",
                name="工程师",
                description="可以进行工程配置",
                permissions=Permission.ENGINEER,
                is_system=True
            ),
            Role(
                role_id="admin",
                name="系统管理员",
                description="拥有所有权限",
                permissions=Permission.ADMIN,
                is_system=True
            ),
        ]

        for role in default_roles:
            await self.role_store.save_role(role)

    async def initialize_default_admin(self, password: str = "Admin@123"):
        """初始化默认管理员"""
        existing = await self.user_store.get_user_by_username("admin")
        if existing:
            return

        password_hash, salt = PasswordHasher.hash_password(password)
        admin = User(
            user_id="admin-001",
            username="admin",
            display_name="系统管理员",
            email="admin@cyrp.com",
            password_hash=password_hash,
            salt=salt,
            roles=["admin"]
        )

        await self.user_store.save_user(admin)


def create_cyrp_security_system() -> SecurityManager:
    """创建穿黄工程安全系统"""
    return SecurityManager()
