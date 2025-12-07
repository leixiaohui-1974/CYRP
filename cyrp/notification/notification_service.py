"""
Notification Service Module for CYRP
穿黄工程通知服务模块

实现多渠道通知推送，支持邮件、短信、Webhook、企业微信、钉钉等
"""

import asyncio
import json
import time
import uuid
import hashlib
import hmac
import base64
import urllib.parse
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Set
from collections import defaultdict
import logging
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

logger = logging.getLogger(__name__)


# ============================================================
# 枚举定义
# ============================================================

class NotificationChannel(Enum):
    """通知渠道"""
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    WECHAT_WORK = "wechat_work"  # 企业微信
    DINGTALK = "dingtalk"        # 钉钉
    FEISHU = "feishu"            # 飞书
    PUSH = "push"                # 推送通知
    VOICE = "voice"              # 语音通知


class NotificationPriority(Enum):
    """通知优先级"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3
    CRITICAL = 4


class NotificationStatus(Enum):
    """通知状态"""
    PENDING = "pending"
    SENDING = "sending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class NotificationType(Enum):
    """通知类型"""
    ALARM = "alarm"           # 报警通知
    WARNING = "warning"       # 预警通知
    INFO = "info"             # 信息通知
    REPORT = "report"         # 报表通知
    SYSTEM = "system"         # 系统通知
    MAINTENANCE = "maintenance"  # 维护通知
    APPROVAL = "approval"     # 审批通知


# ============================================================
# 数据类定义
# ============================================================

@dataclass
class NotificationRecipient:
    """通知接收者"""
    recipient_id: str
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    wechat_id: Optional[str] = None
    dingtalk_id: Optional[str] = None
    channels: Set[NotificationChannel] = field(default_factory=set)
    preferences: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class NotificationTemplate:
    """通知模板"""
    template_id: str
    name: str
    notification_type: NotificationType
    channel: NotificationChannel
    subject_template: str
    body_template: str
    variables: List[str] = field(default_factory=list)
    enabled: bool = True

    def render(self, variables: Dict[str, Any]) -> tuple:
        """渲染模板"""
        subject = self.subject_template
        body = self.body_template

        for key, value in variables.items():
            placeholder = f"${{{key}}}"
            subject = subject.replace(placeholder, str(value))
            body = body.replace(placeholder, str(value))

        return subject, body


@dataclass
class Notification:
    """通知"""
    notification_id: str
    notification_type: NotificationType
    priority: NotificationPriority
    subject: str
    body: str
    recipients: List[str]
    channels: List[NotificationChannel]
    source: str
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    attachments: List[str] = field(default_factory=list)

    @property
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


@dataclass
class DeliveryRecord:
    """投递记录"""
    record_id: str
    notification_id: str
    recipient_id: str
    channel: NotificationChannel
    status: NotificationStatus
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    response: Optional[Dict[str, Any]] = None


@dataclass
class NotificationRule:
    """通知规则"""
    rule_id: str
    name: str
    condition: str  # 条件表达式
    notification_type: NotificationType
    channels: List[NotificationChannel]
    recipients: List[str]
    template_id: Optional[str] = None
    cooldown_seconds: int = 300  # 冷却时间
    max_per_hour: int = 10       # 每小时最大通知数
    enabled: bool = True
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0


@dataclass
class ChannelConfig:
    """渠道配置"""
    channel: NotificationChannel
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    rate_limit: int = 100  # 每分钟
    retry_count: int = 3
    retry_delay: float = 60.0


# ============================================================
# 通知发送器基类
# ============================================================

class NotificationSender(ABC):
    """通知发送器基类"""

    def __init__(self, config: ChannelConfig):
        self.config = config
        self._rate_limiter = RateLimiter(config.rate_limit, 60.0)

    @abstractmethod
    async def send(
        self,
        notification: Notification,
        recipient: NotificationRecipient
    ) -> DeliveryRecord:
        """发送通知"""
        pass

    @abstractmethod
    async def validate_config(self) -> bool:
        """验证配置"""
        pass


class RateLimiter:
    """速率限制器"""

    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: List[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """获取令牌"""
        async with self._lock:
            now = time.time()
            # 清理过期请求
            self.requests = [t for t in self.requests
                           if now - t < self.window_seconds]

            if len(self.requests) >= self.max_requests:
                return False

            self.requests.append(now)
            return True

    async def wait(self):
        """等待直到可以发送"""
        while not await self.acquire():
            await asyncio.sleep(1.0)


# ============================================================
# 邮件发送器
# ============================================================

class EmailSender(NotificationSender):
    """邮件发送器"""

    async def send(
        self,
        notification: Notification,
        recipient: NotificationRecipient
    ) -> DeliveryRecord:
        """发送邮件"""
        record_id = str(uuid.uuid4())

        if not recipient.email:
            return DeliveryRecord(
                record_id=record_id,
                notification_id=notification.notification_id,
                recipient_id=recipient.recipient_id,
                channel=NotificationChannel.EMAIL,
                status=NotificationStatus.FAILED,
                error_message="Recipient has no email address"
            )

        await self._rate_limiter.wait()

        try:
            # 创建邮件
            msg = MIMEMultipart()
            msg['From'] = self.config.config.get('from_address', 'noreply@cyrp.local')
            msg['To'] = recipient.email
            msg['Subject'] = notification.subject

            # 添加正文
            body = MIMEText(notification.body, 'html', 'utf-8')
            msg.attach(body)

            # 添加附件
            for attachment_path in notification.attachments:
                try:
                    with open(attachment_path, 'rb') as f:
                        part = MIMEBase('application', 'octet-stream')
                        part.set_payload(f.read())
                        encoders.encode_base64(part)
                        part.add_header(
                            'Content-Disposition',
                            f'attachment; filename="{attachment_path.split("/")[-1]}"'
                        )
                        msg.attach(part)
                except Exception as e:
                    logger.warning(f"Failed to attach file {attachment_path}: {e}")

            # 发送邮件
            smtp_host = self.config.config.get('smtp_host', 'localhost')
            smtp_port = self.config.config.get('smtp_port', 25)
            smtp_user = self.config.config.get('smtp_user')
            smtp_password = self.config.config.get('smtp_password')
            use_tls = self.config.config.get('use_tls', False)

            # 模拟发送（实际实现中使用真实SMTP）
            logger.info(f"Sending email to {recipient.email}: {notification.subject}")

            # 实际SMTP发送代码
            # with smtplib.SMTP(smtp_host, smtp_port) as server:
            #     if use_tls:
            #         server.starttls()
            #     if smtp_user and smtp_password:
            #         server.login(smtp_user, smtp_password)
            #     server.send_message(msg)

            return DeliveryRecord(
                record_id=record_id,
                notification_id=notification.notification_id,
                recipient_id=recipient.recipient_id,
                channel=NotificationChannel.EMAIL,
                status=NotificationStatus.SENT,
                sent_at=datetime.now()
            )

        except Exception as e:
            logger.error(f"Email sending failed: {e}")
            return DeliveryRecord(
                record_id=record_id,
                notification_id=notification.notification_id,
                recipient_id=recipient.recipient_id,
                channel=NotificationChannel.EMAIL,
                status=NotificationStatus.FAILED,
                error_message=str(e)
            )

    async def validate_config(self) -> bool:
        """验证邮件配置"""
        required = ['smtp_host']
        for key in required:
            if key not in self.config.config:
                return False
        return True


# ============================================================
# 短信发送器
# ============================================================

class SMSSender(NotificationSender):
    """短信发送器"""

    async def send(
        self,
        notification: Notification,
        recipient: NotificationRecipient
    ) -> DeliveryRecord:
        """发送短信"""
        record_id = str(uuid.uuid4())

        if not recipient.phone:
            return DeliveryRecord(
                record_id=record_id,
                notification_id=notification.notification_id,
                recipient_id=recipient.recipient_id,
                channel=NotificationChannel.SMS,
                status=NotificationStatus.FAILED,
                error_message="Recipient has no phone number"
            )

        await self._rate_limiter.wait()

        try:
            # 短信内容限制
            sms_content = notification.body[:500]  # 限制长度

            # 调用短信API（模拟）
            provider = self.config.config.get('provider', 'aliyun')
            logger.info(f"Sending SMS via {provider} to {recipient.phone}: {sms_content[:50]}...")

            # 实际实现中调用对应的短信API
            # response = await self._call_sms_api(recipient.phone, sms_content)

            return DeliveryRecord(
                record_id=record_id,
                notification_id=notification.notification_id,
                recipient_id=recipient.recipient_id,
                channel=NotificationChannel.SMS,
                status=NotificationStatus.SENT,
                sent_at=datetime.now()
            )

        except Exception as e:
            logger.error(f"SMS sending failed: {e}")
            return DeliveryRecord(
                record_id=record_id,
                notification_id=notification.notification_id,
                recipient_id=recipient.recipient_id,
                channel=NotificationChannel.SMS,
                status=NotificationStatus.FAILED,
                error_message=str(e)
            )

    async def validate_config(self) -> bool:
        """验证短信配置"""
        required = ['provider', 'api_key']
        for key in required:
            if key not in self.config.config:
                return False
        return True


# ============================================================
# Webhook发送器
# ============================================================

class WebhookSender(NotificationSender):
    """Webhook发送器"""

    async def send(
        self,
        notification: Notification,
        recipient: NotificationRecipient
    ) -> DeliveryRecord:
        """发送Webhook"""
        record_id = str(uuid.uuid4())

        webhook_url = self.config.config.get('url')
        if not webhook_url:
            return DeliveryRecord(
                record_id=record_id,
                notification_id=notification.notification_id,
                recipient_id=recipient.recipient_id,
                channel=NotificationChannel.WEBHOOK,
                status=NotificationStatus.FAILED,
                error_message="Webhook URL not configured"
            )

        await self._rate_limiter.wait()

        try:
            # 构建请求体
            payload = {
                'notification_id': notification.notification_id,
                'type': notification.notification_type.value,
                'priority': notification.priority.value,
                'subject': notification.subject,
                'body': notification.body,
                'recipient': recipient.recipient_id,
                'timestamp': datetime.now().isoformat(),
                'metadata': notification.metadata
            }

            # 添加签名
            secret = self.config.config.get('secret')
            if secret:
                payload['signature'] = self._generate_signature(payload, secret)

            # 发送HTTP请求（模拟）
            logger.info(f"Sending webhook to {webhook_url}")

            # 实际实现中使用aiohttp
            # async with aiohttp.ClientSession() as session:
            #     async with session.post(webhook_url, json=payload) as resp:
            #         response = await resp.json()

            return DeliveryRecord(
                record_id=record_id,
                notification_id=notification.notification_id,
                recipient_id=recipient.recipient_id,
                channel=NotificationChannel.WEBHOOK,
                status=NotificationStatus.SENT,
                sent_at=datetime.now()
            )

        except Exception as e:
            logger.error(f"Webhook sending failed: {e}")
            return DeliveryRecord(
                record_id=record_id,
                notification_id=notification.notification_id,
                recipient_id=recipient.recipient_id,
                channel=NotificationChannel.WEBHOOK,
                status=NotificationStatus.FAILED,
                error_message=str(e)
            )

    def _generate_signature(self, payload: Dict, secret: str) -> str:
        """生成签名"""
        message = json.dumps(payload, sort_keys=True)
        signature = hmac.new(
            secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature

    async def validate_config(self) -> bool:
        """验证Webhook配置"""
        return 'url' in self.config.config


# ============================================================
# 企业微信发送器
# ============================================================

class WeChatWorkSender(NotificationSender):
    """企业微信发送器"""

    async def send(
        self,
        notification: Notification,
        recipient: NotificationRecipient
    ) -> DeliveryRecord:
        """发送企业微信消息"""
        record_id = str(uuid.uuid4())

        webhook_key = self.config.config.get('webhook_key')
        if not webhook_key:
            return DeliveryRecord(
                record_id=record_id,
                notification_id=notification.notification_id,
                recipient_id=recipient.recipient_id,
                channel=NotificationChannel.WECHAT_WORK,
                status=NotificationStatus.FAILED,
                error_message="WeChat Work webhook key not configured"
            )

        await self._rate_limiter.wait()

        try:
            # 构建消息
            msg_type = self.config.config.get('msg_type', 'markdown')

            if msg_type == 'markdown':
                payload = {
                    'msgtype': 'markdown',
                    'markdown': {
                        'content': f"### {notification.subject}\n\n{notification.body}"
                    }
                }
            else:
                payload = {
                    'msgtype': 'text',
                    'text': {
                        'content': f"{notification.subject}\n\n{notification.body}",
                        'mentioned_list': [recipient.wechat_id] if recipient.wechat_id else []
                    }
                }

            # 发送到企业微信（模拟）
            webhook_url = f"https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key={webhook_key}"
            logger.info(f"Sending WeChat Work message: {notification.subject}")

            return DeliveryRecord(
                record_id=record_id,
                notification_id=notification.notification_id,
                recipient_id=recipient.recipient_id,
                channel=NotificationChannel.WECHAT_WORK,
                status=NotificationStatus.SENT,
                sent_at=datetime.now()
            )

        except Exception as e:
            logger.error(f"WeChat Work sending failed: {e}")
            return DeliveryRecord(
                record_id=record_id,
                notification_id=notification.notification_id,
                recipient_id=recipient.recipient_id,
                channel=NotificationChannel.WECHAT_WORK,
                status=NotificationStatus.FAILED,
                error_message=str(e)
            )

    async def validate_config(self) -> bool:
        """验证企业微信配置"""
        return 'webhook_key' in self.config.config


# ============================================================
# 钉钉发送器
# ============================================================

class DingTalkSender(NotificationSender):
    """钉钉发送器"""

    async def send(
        self,
        notification: Notification,
        recipient: NotificationRecipient
    ) -> DeliveryRecord:
        """发送钉钉消息"""
        record_id = str(uuid.uuid4())

        access_token = self.config.config.get('access_token')
        if not access_token:
            return DeliveryRecord(
                record_id=record_id,
                notification_id=notification.notification_id,
                recipient_id=recipient.recipient_id,
                channel=NotificationChannel.DINGTALK,
                status=NotificationStatus.FAILED,
                error_message="DingTalk access token not configured"
            )

        await self._rate_limiter.wait()

        try:
            # 生成签名（如果配置了secret）
            secret = self.config.config.get('secret')
            sign_params = ""
            if secret:
                timestamp = str(round(time.time() * 1000))
                string_to_sign = f"{timestamp}\n{secret}"
                hmac_code = hmac.new(
                    secret.encode(),
                    string_to_sign.encode(),
                    hashlib.sha256
                ).digest()
                sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
                sign_params = f"&timestamp={timestamp}&sign={sign}"

            # 构建消息
            msg_type = self.config.config.get('msg_type', 'markdown')

            if msg_type == 'markdown':
                payload = {
                    'msgtype': 'markdown',
                    'markdown': {
                        'title': notification.subject,
                        'text': f"### {notification.subject}\n\n{notification.body}"
                    }
                }
            else:
                payload = {
                    'msgtype': 'text',
                    'text': {
                        'content': f"{notification.subject}\n\n{notification.body}"
                    }
                }

            # 发送到钉钉（模拟）
            webhook_url = f"https://oapi.dingtalk.com/robot/send?access_token={access_token}{sign_params}"
            logger.info(f"Sending DingTalk message: {notification.subject}")

            return DeliveryRecord(
                record_id=record_id,
                notification_id=notification.notification_id,
                recipient_id=recipient.recipient_id,
                channel=NotificationChannel.DINGTALK,
                status=NotificationStatus.SENT,
                sent_at=datetime.now()
            )

        except Exception as e:
            logger.error(f"DingTalk sending failed: {e}")
            return DeliveryRecord(
                record_id=record_id,
                notification_id=notification.notification_id,
                recipient_id=recipient.recipient_id,
                channel=NotificationChannel.DINGTALK,
                status=NotificationStatus.FAILED,
                error_message=str(e)
            )

    async def validate_config(self) -> bool:
        """验证钉钉配置"""
        return 'access_token' in self.config.config


# ============================================================
# 接收者管理器
# ============================================================

class RecipientManager:
    """接收者管理器"""

    def __init__(self):
        self.recipients: Dict[str, NotificationRecipient] = {}
        self.groups: Dict[str, Set[str]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def add_recipient(self, recipient: NotificationRecipient) -> str:
        """添加接收者"""
        async with self._lock:
            self.recipients[recipient.recipient_id] = recipient
            return recipient.recipient_id

    async def remove_recipient(self, recipient_id: str) -> bool:
        """移除接收者"""
        async with self._lock:
            if recipient_id in self.recipients:
                del self.recipients[recipient_id]
                # 从所有组中移除
                for group_members in self.groups.values():
                    group_members.discard(recipient_id)
                return True
            return False

    async def get_recipient(self, recipient_id: str) -> Optional[NotificationRecipient]:
        """获取接收者"""
        return self.recipients.get(recipient_id)

    async def add_to_group(self, recipient_id: str, group_name: str):
        """添加到组"""
        async with self._lock:
            self.groups[group_name].add(recipient_id)

    async def remove_from_group(self, recipient_id: str, group_name: str):
        """从组中移除"""
        async with self._lock:
            self.groups[group_name].discard(recipient_id)

    async def get_group_members(self, group_name: str) -> List[NotificationRecipient]:
        """获取组成员"""
        member_ids = self.groups.get(group_name, set())
        return [self.recipients[mid] for mid in member_ids
                if mid in self.recipients]

    async def resolve_recipients(
        self,
        recipient_ids: List[str]
    ) -> List[NotificationRecipient]:
        """解析接收者列表（包括组）"""
        resolved = []
        seen = set()

        for rid in recipient_ids:
            # 检查是否为组
            if rid.startswith('@'):
                group_name = rid[1:]
                members = await self.get_group_members(group_name)
                for member in members:
                    if member.recipient_id not in seen and member.enabled:
                        resolved.append(member)
                        seen.add(member.recipient_id)
            else:
                recipient = await self.get_recipient(rid)
                if recipient and recipient.recipient_id not in seen and recipient.enabled:
                    resolved.append(recipient)
                    seen.add(recipient.recipient_id)

        return resolved


# ============================================================
# 模板管理器
# ============================================================

class TemplateManager:
    """模板管理器"""

    def __init__(self):
        self.templates: Dict[str, NotificationTemplate] = {}
        self._lock = asyncio.Lock()

    async def add_template(self, template: NotificationTemplate) -> str:
        """添加模板"""
        async with self._lock:
            self.templates[template.template_id] = template
            return template.template_id

    async def get_template(self, template_id: str) -> Optional[NotificationTemplate]:
        """获取模板"""
        return self.templates.get(template_id)

    async def render_template(
        self,
        template_id: str,
        variables: Dict[str, Any]
    ) -> Optional[tuple]:
        """渲染模板"""
        template = await self.get_template(template_id)
        if not template or not template.enabled:
            return None
        return template.render(variables)


# ============================================================
# 通知队列
# ============================================================

class NotificationQueue:
    """通知队列"""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_size)
        self._processing: Dict[str, Notification] = {}
        self._lock = asyncio.Lock()

    async def enqueue(self, notification: Notification):
        """入队"""
        priority = notification.priority.value
        scheduled = notification.scheduled_at or datetime.now()
        # 优先级越低值越小，越先处理
        await self._queue.put((priority, scheduled.timestamp(), notification))

    async def dequeue(self) -> Optional[Notification]:
        """出队"""
        try:
            _, _, notification = await asyncio.wait_for(
                self._queue.get(),
                timeout=1.0
            )

            # 检查是否过期
            if notification.is_expired:
                logger.warning(f"Notification {notification.notification_id} expired")
                return await self.dequeue()

            # 检查是否到达计划时间
            if notification.scheduled_at and notification.scheduled_at > datetime.now():
                await self.enqueue(notification)
                return await self.dequeue()

            async with self._lock:
                self._processing[notification.notification_id] = notification

            return notification

        except asyncio.TimeoutError:
            return None

    async def complete(self, notification_id: str):
        """完成处理"""
        async with self._lock:
            self._processing.pop(notification_id, None)

    @property
    def size(self) -> int:
        """队列大小"""
        return self._queue.qsize()

    @property
    def processing_count(self) -> int:
        """处理中的数量"""
        return len(self._processing)


# ============================================================
# 规则引擎
# ============================================================

class NotificationRuleEngine:
    """通知规则引擎"""

    def __init__(self):
        self.rules: Dict[str, NotificationRule] = {}
        self._lock = asyncio.Lock()

    async def add_rule(self, rule: NotificationRule) -> str:
        """添加规则"""
        async with self._lock:
            self.rules[rule.rule_id] = rule
            return rule.rule_id

    async def remove_rule(self, rule_id: str) -> bool:
        """移除规则"""
        async with self._lock:
            if rule_id in self.rules:
                del self.rules[rule_id]
                return True
            return False

    async def evaluate(
        self,
        context: Dict[str, Any]
    ) -> List[NotificationRule]:
        """评估上下文，返回匹配的规则"""
        matched = []

        for rule in self.rules.values():
            if not rule.enabled:
                continue

            # 检查冷却时间
            if rule.last_triggered:
                elapsed = (datetime.now() - rule.last_triggered).total_seconds()
                if elapsed < rule.cooldown_seconds:
                    continue

            # 评估条件
            try:
                if self._evaluate_condition(rule.condition, context):
                    matched.append(rule)
            except Exception as e:
                logger.error(f"Rule evaluation failed: {e}")

        return matched

    def _evaluate_condition(
        self,
        condition: str,
        context: Dict[str, Any]
    ) -> bool:
        """评估条件表达式"""
        # 简单的条件评估实现
        # 实际实现中可以使用更复杂的表达式解析器

        # 支持的操作符
        operators = {
            '==': lambda a, b: a == b,
            '!=': lambda a, b: a != b,
            '>': lambda a, b: a > b,
            '<': lambda a, b: a < b,
            '>=': lambda a, b: a >= b,
            '<=': lambda a, b: a <= b,
            'contains': lambda a, b: b in a,
            'startswith': lambda a, b: str(a).startswith(str(b)),
        }

        # 解析简单条件：field operator value
        pattern = r'(\w+)\s*(==|!=|>=|<=|>|<|contains|startswith)\s*(.+)'
        match = re.match(pattern, condition.strip())

        if not match:
            return False

        field, operator, value = match.groups()
        value = value.strip().strip('"\'')

        if field not in context:
            return False

        field_value = context[field]

        # 类型转换
        try:
            if isinstance(field_value, (int, float)):
                value = float(value)
        except ValueError:
            pass

        op_func = operators.get(operator)
        if not op_func:
            return False

        return op_func(field_value, value)

    async def mark_triggered(self, rule_id: str):
        """标记规则已触发"""
        async with self._lock:
            if rule_id in self.rules:
                self.rules[rule_id].last_triggered = datetime.now()
                self.rules[rule_id].trigger_count += 1


# ============================================================
# 通知服务
# ============================================================

class NotificationService:
    """通知服务"""

    def __init__(self):
        self.recipient_manager = RecipientManager()
        self.template_manager = TemplateManager()
        self.rule_engine = NotificationRuleEngine()
        self.notification_queue = NotificationQueue()

        # 发送器
        self.senders: Dict[NotificationChannel, NotificationSender] = {}

        # 投递记录
        self.delivery_records: Dict[str, List[DeliveryRecord]] = defaultdict(list)

        # 运行状态
        self._running = False
        self._workers: List[asyncio.Task] = []

    def register_sender(
        self,
        channel: NotificationChannel,
        config: ChannelConfig
    ):
        """注册发送器"""
        sender_classes = {
            NotificationChannel.EMAIL: EmailSender,
            NotificationChannel.SMS: SMSSender,
            NotificationChannel.WEBHOOK: WebhookSender,
            NotificationChannel.WECHAT_WORK: WeChatWorkSender,
            NotificationChannel.DINGTALK: DingTalkSender,
        }

        sender_class = sender_classes.get(channel)
        if sender_class:
            self.senders[channel] = sender_class(config)
            logger.info(f"Registered sender for channel: {channel.value}")

    async def start(self, worker_count: int = 4):
        """启动服务"""
        self._running = True

        # 启动工作线程
        for i in range(worker_count):
            worker = asyncio.create_task(self._process_queue())
            self._workers.append(worker)

        logger.info(f"Notification service started with {worker_count} workers")

    async def stop(self):
        """停止服务"""
        self._running = False

        # 等待工作线程结束
        for worker in self._workers:
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass

        self._workers.clear()
        logger.info("Notification service stopped")

    async def send(
        self,
        notification_type: NotificationType,
        subject: str,
        body: str,
        recipients: List[str],
        channels: Optional[List[NotificationChannel]] = None,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        source: str = "system",
        scheduled_at: Optional[datetime] = None,
        expires_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
        attachments: Optional[List[str]] = None
    ) -> str:
        """发送通知"""
        notification = Notification(
            notification_id=str(uuid.uuid4()),
            notification_type=notification_type,
            priority=priority,
            subject=subject,
            body=body,
            recipients=recipients,
            channels=channels or [NotificationChannel.EMAIL],
            source=source,
            scheduled_at=scheduled_at,
            expires_at=expires_at,
            metadata=metadata or {},
            attachments=attachments or []
        )

        await self.notification_queue.enqueue(notification)
        logger.info(f"Notification queued: {notification.notification_id}")

        return notification.notification_id

    async def send_with_template(
        self,
        template_id: str,
        variables: Dict[str, Any],
        recipients: List[str],
        priority: NotificationPriority = NotificationPriority.NORMAL,
        **kwargs
    ) -> Optional[str]:
        """使用模板发送通知"""
        result = await self.template_manager.render_template(template_id, variables)
        if not result:
            logger.error(f"Template not found: {template_id}")
            return None

        subject, body = result
        template = await self.template_manager.get_template(template_id)

        return await self.send(
            notification_type=template.notification_type,
            subject=subject,
            body=body,
            recipients=recipients,
            channels=[template.channel],
            priority=priority,
            **kwargs
        )

    async def trigger_by_rules(
        self,
        context: Dict[str, Any],
        source: str = "rule_engine"
    ) -> List[str]:
        """根据规则触发通知"""
        matched_rules = await self.rule_engine.evaluate(context)
        notification_ids = []

        for rule in matched_rules:
            if rule.template_id:
                nid = await self.send_with_template(
                    template_id=rule.template_id,
                    variables=context,
                    recipients=rule.recipients,
                    source=source
                )
            else:
                nid = await self.send(
                    notification_type=rule.notification_type,
                    subject=f"Alert: {rule.name}",
                    body=json.dumps(context, indent=2, default=str),
                    recipients=rule.recipients,
                    channels=rule.channels,
                    source=source
                )

            if nid:
                notification_ids.append(nid)
                await self.rule_engine.mark_triggered(rule.rule_id)

        return notification_ids

    async def _process_queue(self):
        """处理队列"""
        while self._running:
            try:
                notification = await self.notification_queue.dequeue()
                if notification is None:
                    continue

                await self._process_notification(notification)
                await self.notification_queue.complete(notification.notification_id)

            except Exception as e:
                logger.error(f"Queue processing error: {e}")
                await asyncio.sleep(1.0)

    async def _process_notification(self, notification: Notification):
        """处理单个通知"""
        # 解析接收者
        recipients = await self.recipient_manager.resolve_recipients(
            notification.recipients
        )

        if not recipients:
            logger.warning(f"No valid recipients for notification {notification.notification_id}")
            return

        # 发送到各个渠道
        for channel in notification.channels:
            sender = self.senders.get(channel)
            if not sender:
                logger.warning(f"No sender registered for channel: {channel.value}")
                continue

            # 发送给每个接收者
            for recipient in recipients:
                # 检查接收者是否支持该渠道
                if recipient.channels and channel not in recipient.channels:
                    continue

                record = await sender.send(notification, recipient)
                self.delivery_records[notification.notification_id].append(record)

                # 重试失败的投递
                if record.status == NotificationStatus.FAILED:
                    if record.retry_count < sender.config.retry_count:
                        await asyncio.sleep(sender.config.retry_delay)
                        record.retry_count += 1
                        retry_record = await sender.send(notification, recipient)
                        self.delivery_records[notification.notification_id].append(retry_record)

    async def get_delivery_status(
        self,
        notification_id: str
    ) -> List[DeliveryRecord]:
        """获取投递状态"""
        return self.delivery_records.get(notification_id, [])

    async def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_notifications = len(self.delivery_records)
        total_deliveries = sum(len(records) for records in self.delivery_records.values())
        successful = sum(
            1 for records in self.delivery_records.values()
            for r in records if r.status == NotificationStatus.SENT
        )
        failed = sum(
            1 for records in self.delivery_records.values()
            for r in records if r.status == NotificationStatus.FAILED
        )

        return {
            'total_notifications': total_notifications,
            'total_deliveries': total_deliveries,
            'successful_deliveries': successful,
            'failed_deliveries': failed,
            'queue_size': self.notification_queue.size,
            'processing_count': self.notification_queue.processing_count,
            'registered_channels': list(self.senders.keys())
        }


# ============================================================
# 工厂函数
# ============================================================

def create_cyrp_notification_service(
    email_config: Optional[Dict[str, Any]] = None,
    sms_config: Optional[Dict[str, Any]] = None,
    webhook_config: Optional[Dict[str, Any]] = None,
    wechat_config: Optional[Dict[str, Any]] = None,
    dingtalk_config: Optional[Dict[str, Any]] = None
) -> NotificationService:
    """创建CYRP通知服务实例

    Args:
        email_config: 邮件配置
        sms_config: 短信配置
        webhook_config: Webhook配置
        wechat_config: 企业微信配置
        dingtalk_config: 钉钉配置

    Returns:
        NotificationService: 配置好的通知服务实例
    """
    service = NotificationService()

    # 注册邮件发送器
    if email_config:
        service.register_sender(
            NotificationChannel.EMAIL,
            ChannelConfig(
                channel=NotificationChannel.EMAIL,
                config=email_config
            )
        )

    # 注册短信发送器
    if sms_config:
        service.register_sender(
            NotificationChannel.SMS,
            ChannelConfig(
                channel=NotificationChannel.SMS,
                config=sms_config
            )
        )

    # 注册Webhook发送器
    if webhook_config:
        service.register_sender(
            NotificationChannel.WEBHOOK,
            ChannelConfig(
                channel=NotificationChannel.WEBHOOK,
                config=webhook_config
            )
        )

    # 注册企业微信发送器
    if wechat_config:
        service.register_sender(
            NotificationChannel.WECHAT_WORK,
            ChannelConfig(
                channel=NotificationChannel.WECHAT_WORK,
                config=wechat_config
            )
        )

    # 注册钉钉发送器
    if dingtalk_config:
        service.register_sender(
            NotificationChannel.DINGTALK,
            ChannelConfig(
                channel=NotificationChannel.DINGTALK,
                config=dingtalk_config
            )
        )

    # 添加默认模板
    _add_default_templates(service)

    return service


def _add_default_templates(service: NotificationService):
    """添加默认模板"""
    import asyncio

    templates = [
        NotificationTemplate(
            template_id="alarm_critical",
            name="严重报警模板",
            notification_type=NotificationType.ALARM,
            channel=NotificationChannel.EMAIL,
            subject_template="[紧急] 穿黄工程报警: ${alarm_name}",
            body_template="""
<h2>严重报警通知</h2>
<p><strong>报警名称:</strong> ${alarm_name}</p>
<p><strong>报警时间:</strong> ${alarm_time}</p>
<p><strong>报警位置:</strong> ${location}</p>
<p><strong>当前值:</strong> ${current_value}</p>
<p><strong>限值:</strong> ${limit_value}</p>
<p><strong>详情:</strong> ${description}</p>
<p>请立即处理!</p>
""",
            variables=['alarm_name', 'alarm_time', 'location', 'current_value', 'limit_value', 'description']
        ),
        NotificationTemplate(
            template_id="maintenance_reminder",
            name="维护提醒模板",
            notification_type=NotificationType.MAINTENANCE,
            channel=NotificationChannel.EMAIL,
            subject_template="[维护提醒] ${equipment_name} 即将到期",
            body_template="""
<h2>设备维护提醒</h2>
<p><strong>设备名称:</strong> ${equipment_name}</p>
<p><strong>维护类型:</strong> ${maintenance_type}</p>
<p><strong>计划日期:</strong> ${scheduled_date}</p>
<p><strong>负责人:</strong> ${assignee}</p>
<p><strong>备注:</strong> ${notes}</p>
""",
            variables=['equipment_name', 'maintenance_type', 'scheduled_date', 'assignee', 'notes']
        ),
        NotificationTemplate(
            template_id="daily_report",
            name="日报模板",
            notification_type=NotificationType.REPORT,
            channel=NotificationChannel.EMAIL,
            subject_template="穿黄工程运行日报 - ${report_date}",
            body_template="""
<h2>穿黄工程运行日报</h2>
<p><strong>报告日期:</strong> ${report_date}</p>
<h3>运行概况</h3>
<ul>
<li>总流量: ${total_flow} m³</li>
<li>平均流速: ${avg_velocity} m/s</li>
<li>报警次数: ${alarm_count}</li>
</ul>
<h3>设备状态</h3>
<p>${equipment_status}</p>
<h3>备注</h3>
<p>${remarks}</p>
""",
            variables=['report_date', 'total_flow', 'avg_velocity', 'alarm_count', 'equipment_status', 'remarks']
        ),
    ]

    async def add_templates():
        for template in templates:
            await service.template_manager.add_template(template)

    # 在事件循环中运行
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(add_templates())
    except RuntimeError:
        asyncio.run(add_templates())
