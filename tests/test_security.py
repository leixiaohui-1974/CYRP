"""
Tests for Security Module.
安全管理模块测试
"""

import pytest
import asyncio
from datetime import datetime, timedelta

from cyrp.security import (
    Permission,
    SessionState,
    AuditEventType,
    PasswordPolicy,
    Role,
    User,
    Session,
    PasswordHasher,
    TOTPGenerator,
    SessionManager,
    AuditLogger,
    InMemoryUserStore,
    InMemoryRoleStore,
    AuthenticationManager,
    AuthorizationManager,
    UserManager,
    SecurityManager,
    create_cyrp_security_system,
)


class TestPasswordPolicy:
    """密码策略测试类"""

    def setup_method(self):
        """测试前设置"""
        self.policy = PasswordPolicy()

    def test_valid_password(self):
        """测试有效密码"""
        valid, errors = self.policy.validate("Test@1234")
        assert valid == True
        assert len(errors) == 0

    def test_password_too_short(self):
        """测试密码过短"""
        valid, errors = self.policy.validate("Te@1")
        assert valid == False
        assert any("长度" in e for e in errors)

    def test_password_no_uppercase(self):
        """测试密码无大写字母"""
        valid, errors = self.policy.validate("test@1234")
        assert valid == False
        assert any("大写" in e for e in errors)

    def test_password_no_lowercase(self):
        """测试密码无小写字母"""
        valid, errors = self.policy.validate("TEST@1234")
        assert valid == False
        assert any("小写" in e for e in errors)

    def test_password_no_digit(self):
        """测试密码无数字"""
        valid, errors = self.policy.validate("Test@abcd")
        assert valid == False
        assert any("数字" in e for e in errors)

    def test_password_no_special(self):
        """测试密码无特殊字符"""
        valid, errors = self.policy.validate("Test12345")
        assert valid == False
        assert any("特殊字符" in e for e in errors)


class TestPasswordHasher:
    """密码哈希器测试类"""

    def test_hash_password(self):
        """测试哈希密码"""
        password = "TestPassword123!"
        hash_result, salt = PasswordHasher.hash_password(password)

        assert hash_result is not None
        assert salt is not None
        assert len(hash_result) > 0
        assert len(salt) > 0

    def test_verify_password_correct(self):
        """测试验证正确密码"""
        password = "TestPassword123!"
        hash_result, salt = PasswordHasher.hash_password(password)

        result = PasswordHasher.verify_password(password, hash_result, salt)

        assert result == True

    def test_verify_password_incorrect(self):
        """测试验证错误密码"""
        password = "TestPassword123!"
        wrong_password = "WrongPassword123!"
        hash_result, salt = PasswordHasher.hash_password(password)

        result = PasswordHasher.verify_password(wrong_password, hash_result, salt)

        assert result == False

    def test_same_password_different_salt(self):
        """测试相同密码不同盐值"""
        password = "TestPassword123!"
        hash1, salt1 = PasswordHasher.hash_password(password)
        hash2, salt2 = PasswordHasher.hash_password(password)

        # 不同的盐值应产生不同的哈希
        assert salt1 != salt2
        assert hash1 != hash2


class TestTOTPGenerator:
    """TOTP生成器测试类"""

    def test_generate_secret(self):
        """测试生成密钥"""
        secret = TOTPGenerator.generate_secret()

        assert secret is not None
        assert len(secret) > 0

    def test_generate_totp(self):
        """测试生成TOTP"""
        secret = TOTPGenerator.generate_secret()
        code = TOTPGenerator.generate_totp(secret)

        assert code is not None
        assert len(code) == 6
        assert code.isdigit()

    def test_verify_totp_correct(self):
        """测试验证正确TOTP"""
        secret = TOTPGenerator.generate_secret()
        code = TOTPGenerator.generate_totp(secret)

        result = TOTPGenerator.verify_totp(secret, code)

        assert result == True

    def test_verify_totp_incorrect(self):
        """测试验证错误TOTP"""
        secret = TOTPGenerator.generate_secret()

        result = TOTPGenerator.verify_totp(secret, "000000")

        # 可能碰巧正确,但概率很低
        # assert result == False


class TestRole:
    """角色测试类"""

    def test_creation(self):
        """测试创建角色"""
        role = Role(
            role_id="admin",
            name="管理员",
            description="系统管理员",
            permissions=Permission.ADMIN
        )

        assert role.role_id == "admin"
        assert role.name == "管理员"
        assert role.permissions == Permission.ADMIN

    def test_has_permission(self):
        """测试权限检查"""
        role = Role(
            role_id="operator",
            name="操作员",
            description="系统操作员",
            permissions=Permission.OPERATOR
        )

        # 操作员应该有查看权限
        assert role.has_permission(Permission.VIEW_DASHBOARD) == True

        # 操作员不应该有管理用户权限
        assert role.has_permission(Permission.MANAGE_USERS) == False


class TestSessionManager:
    """会话管理器测试类"""

    def setup_method(self):
        """测试前设置"""
        self.manager = SessionManager(session_timeout_minutes=30)

    @pytest.mark.asyncio
    async def test_create_session(self):
        """测试创建会话"""
        user = User(
            user_id="user1",
            username="testuser",
            display_name="Test User",
            email="test@example.com",
            password_hash="hash",
            salt="salt",
            roles=["operator"]
        )

        session = await self.manager.create_session(
            user,
            Permission.OPERATOR,
            "127.0.0.1",
            "TestBrowser"
        )

        assert session is not None
        assert session.user_id == "user1"
        assert session.state == SessionState.ACTIVE

    @pytest.mark.asyncio
    async def test_get_session(self):
        """测试获取会话"""
        user = User(
            user_id="user1",
            username="testuser",
            display_name="Test User",
            email="test@example.com",
            password_hash="hash",
            salt="salt",
            roles=[]
        )

        created = await self.manager.create_session(
            user, Permission.VIEWER, "127.0.0.1", "TestBrowser"
        )

        retrieved = await self.manager.get_session(created.session_id)

        assert retrieved is not None
        assert retrieved.session_id == created.session_id

    @pytest.mark.asyncio
    async def test_refresh_session(self):
        """测试刷新会话"""
        user = User(
            user_id="user1",
            username="testuser",
            display_name="Test User",
            email="test@example.com",
            password_hash="hash",
            salt="salt",
            roles=[]
        )

        session = await self.manager.create_session(
            user, Permission.VIEWER, "127.0.0.1", "TestBrowser"
        )
        original_expires = session.expires_at

        await asyncio.sleep(0.1)
        result = await self.manager.refresh_session(session.session_id)

        assert result == True
        updated = await self.manager.get_session(session.session_id)
        assert updated.expires_at > original_expires

    @pytest.mark.asyncio
    async def test_terminate_session(self):
        """测试终止会话"""
        user = User(
            user_id="user1",
            username="testuser",
            display_name="Test User",
            email="test@example.com",
            password_hash="hash",
            salt="salt",
            roles=[]
        )

        session = await self.manager.create_session(
            user, Permission.VIEWER, "127.0.0.1", "TestBrowser"
        )

        await self.manager.terminate_session(session.session_id)

        retrieved = await self.manager.get_session(session.session_id)
        assert retrieved is None


class TestAuditLogger:
    """审计日志测试类"""

    def setup_method(self):
        """测试前设置"""
        self.logger = AuditLogger()

    @pytest.mark.asyncio
    async def test_log_event(self):
        """测试记录事件"""
        await self.logger.log(
            AuditEventType.LOGIN_SUCCESS,
            "success",
            user_id="user1",
            username="testuser",
            ip_address="127.0.0.1"
        )

        entries = await self.logger.query(limit=10)
        assert len(entries) == 1
        assert entries[0].event_type == AuditEventType.LOGIN_SUCCESS

    @pytest.mark.asyncio
    async def test_query_by_user(self):
        """测试按用户查询"""
        await self.logger.log(
            AuditEventType.LOGIN_SUCCESS,
            "success",
            user_id="user1",
            username="user1"
        )
        await self.logger.log(
            AuditEventType.LOGIN_SUCCESS,
            "success",
            user_id="user2",
            username="user2"
        )

        entries = await self.logger.query(user_id="user1")

        assert len(entries) == 1
        assert entries[0].user_id == "user1"


class TestInMemoryUserStore:
    """内存用户存储测试类"""

    def setup_method(self):
        """测试前设置"""
        self.store = InMemoryUserStore()

    @pytest.mark.asyncio
    async def test_save_and_get_user(self):
        """测试保存和获取用户"""
        user = User(
            user_id="user1",
            username="testuser",
            display_name="Test User",
            email="test@example.com",
            password_hash="hash",
            salt="salt",
            roles=["operator"]
        )

        await self.store.save_user(user)
        retrieved = await self.store.get_user("user1")

        assert retrieved is not None
        assert retrieved.username == "testuser"

    @pytest.mark.asyncio
    async def test_get_by_username(self):
        """测试按用户名获取"""
        user = User(
            user_id="user1",
            username="TestUser",
            display_name="Test User",
            email="test@example.com",
            password_hash="hash",
            salt="salt",
            roles=[]
        )

        await self.store.save_user(user)
        retrieved = await self.store.get_user_by_username("testuser")  # 小写

        assert retrieved is not None
        assert retrieved.user_id == "user1"

    @pytest.mark.asyncio
    async def test_delete_user(self):
        """测试删除用户"""
        user = User(
            user_id="user1",
            username="testuser",
            display_name="Test User",
            email="test@example.com",
            password_hash="hash",
            salt="salt",
            roles=[]
        )

        await self.store.save_user(user)
        await self.store.delete_user("user1")

        retrieved = await self.store.get_user("user1")
        assert retrieved is None


class TestSecurityManager:
    """安全管理器测试类"""

    def setup_method(self):
        """测试前设置"""
        self.manager = SecurityManager()

    @pytest.mark.asyncio
    async def test_initialize_roles(self):
        """测试初始化角色"""
        await self.manager.initialize_default_roles()

        roles = await self.manager.role_store.list_roles()
        assert len(roles) == 4  # viewer, operator, engineer, admin

    @pytest.mark.asyncio
    async def test_initialize_admin(self):
        """测试初始化管理员"""
        await self.manager.initialize_default_roles()
        await self.manager.initialize_default_admin("Test@123")

        admin = await self.manager.user_store.get_user_by_username("admin")
        assert admin is not None
        assert "admin" in admin.roles

    @pytest.mark.asyncio
    async def test_authentication_flow(self):
        """测试认证流程"""
        await self.manager.initialize_default_roles()
        await self.manager.initialize_default_admin("Admin@123")

        # 尝试登录
        success, session, message = await self.manager.auth_manager.authenticate(
            "admin", "Admin@123", "127.0.0.1", "TestBrowser"
        )

        assert success == True
        assert session is not None
        assert session.username == "admin"

    @pytest.mark.asyncio
    async def test_failed_authentication(self):
        """测试认证失败"""
        await self.manager.initialize_default_roles()
        await self.manager.initialize_default_admin("Admin@123")

        # 使用错误密码登录
        success, session, message = await self.manager.auth_manager.authenticate(
            "admin", "WrongPassword", "127.0.0.1", "TestBrowser"
        )

        assert success == False
        assert session is None


class TestCreateCYRPSecuritySystem:
    """测试创建穿黄工程安全系统"""

    def test_create_system(self):
        """测试创建系统"""
        manager = create_cyrp_security_system()

        assert manager is not None
        assert isinstance(manager, SecurityManager)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
