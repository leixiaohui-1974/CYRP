"""
Alarm Management System for CYRP
穿黄工程报警管理系统

功能：
- 报警定义与配置
- 报警触发与抑制
- 报警确认与清除
- 报警升级与推送
- 报警统计与分析
"""

import threading
import queue
import time
import json
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum, auto
from collections import deque
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)


# ============================================================================
# 报警定义
# ============================================================================

class AlarmSeverity(Enum):
    """报警严重程度"""
    INFO = 0        # 信息
    WARNING = 1     # 警告
    MINOR = 2       # 次要
    MAJOR = 3       # 主要
    CRITICAL = 4    # 严重
    EMERGENCY = 5   # 紧急


class AlarmState(Enum):
    """报警状态"""
    NORMAL = auto()         # 正常
    ACTIVE = auto()         # 活动
    ACKNOWLEDGED = auto()   # 已确认
    CLEARED = auto()        # 已清除
    SHELVED = auto()        # 已搁置
    SUPPRESSED = auto()     # 已抑制


class AlarmType(Enum):
    """报警类型"""
    # 过程报警
    HIGH_HIGH = "HH"        # 高高限
    HIGH = "H"              # 高限
    LOW = "L"               # 低限
    LOW_LOW = "LL"          # 低低限
    RATE_OF_CHANGE = "ROC"  # 变化率
    DEVIATION = "DEV"       # 偏差
    # 设备报警
    EQUIPMENT_FAULT = "EQF"     # 设备故障
    COMMUNICATION_LOSS = "COM"  # 通信丢失
    SENSOR_FAULT = "SNS"        # 传感器故障
    # 系统报警
    SYSTEM_ERROR = "SYS"        # 系统错误
    SECURITY = "SEC"            # 安全报警
    # 自定义
    CUSTOM = "CUS"              # 自定义


@dataclass
class AlarmDefinition:
    """报警定义"""
    alarm_id: str
    name: str
    description: str
    alarm_type: AlarmType
    severity: AlarmSeverity
    source: str                     # 数据源/标签
    # 触发条件
    setpoint: Optional[float] = None
    deadband: float = 0.0           # 死区
    delay_on: float = 0.0           # 报警延时(秒)
    delay_off: float = 0.0          # 恢复延时(秒)
    # 限值 (用于过程报警)
    high_high_limit: Optional[float] = None
    high_limit: Optional[float] = None
    low_limit: Optional[float] = None
    low_low_limit: Optional[float] = None
    rate_limit: Optional[float] = None  # 变化率限值
    # 配置
    enabled: bool = True
    auto_acknowledge: bool = False
    requires_comment: bool = False
    escalation_time: Optional[float] = None  # 升级时间(秒)
    suppression_group: Optional[str] = None
    # 动作
    actions: List[str] = field(default_factory=list)  # 触发的动作
    # 分组
    area: str = ""
    group: str = ""
    priority: int = 1
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class AlarmInstance:
    """报警实例"""
    instance_id: str
    definition: AlarmDefinition
    state: AlarmState = AlarmState.NORMAL
    # 时间戳
    occurred_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    cleared_at: Optional[datetime] = None
    # 确认信息
    acknowledged_by: Optional[str] = None
    acknowledge_comment: Optional[str] = None
    # 值
    trigger_value: Optional[float] = None
    current_value: Optional[float] = None
    # 计数
    occurrence_count: int = 0
    # 升级
    escalated: bool = False
    escalation_level: int = 0

    def to_dict(self) -> Dict:
        return {
            'instance_id': self.instance_id,
            'alarm_id': self.definition.alarm_id,
            'name': self.definition.name,
            'description': self.definition.description,
            'severity': self.definition.severity.name,
            'state': self.state.name,
            'source': self.definition.source,
            'occurred_at': self.occurred_at.isoformat() if self.occurred_at else None,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'cleared_at': self.cleared_at.isoformat() if self.cleared_at else None,
            'acknowledged_by': self.acknowledged_by,
            'trigger_value': self.trigger_value,
            'current_value': self.current_value,
            'occurrence_count': self.occurrence_count,
        }


@dataclass
class AlarmEvent:
    """报警事件"""
    event_id: str
    instance_id: str
    alarm_id: str
    event_type: str  # RAISED, ACKNOWLEDGED, CLEARED, ESCALATED
    timestamp: datetime
    user: Optional[str] = None
    comment: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# 报警处理器
# ============================================================================

class AlarmProcessor:
    """报警处理器"""

    def __init__(self):
        self._pending_alarms: Dict[str, Tuple[datetime, float]] = {}  # alarm_id -> (start_time, value)
        self._pending_clears: Dict[str, Tuple[datetime, float]] = {}

    def process_value(self, definition: AlarmDefinition, value: float,
                     current_state: AlarmState) -> Optional[Tuple[AlarmState, float]]:
        """
        处理数值，判断是否触发/清除报警

        Returns:
            (新状态, 触发值) 或 None表示无变化
        """
        alarm_id = definition.alarm_id
        now = datetime.now()

        # 检查是否应该报警
        should_alarm = self._check_alarm_condition(definition, value)

        if should_alarm:
            # 应该报警
            if current_state in [AlarmState.NORMAL, AlarmState.CLEARED]:
                # 检查延时
                if alarm_id in self._pending_alarms:
                    start_time, trigger_val = self._pending_alarms[alarm_id]
                    if (now - start_time).total_seconds() >= definition.delay_on:
                        del self._pending_alarms[alarm_id]
                        return (AlarmState.ACTIVE, trigger_val)
                else:
                    self._pending_alarms[alarm_id] = (now, value)
                    if definition.delay_on <= 0:
                        del self._pending_alarms[alarm_id]
                        return (AlarmState.ACTIVE, value)

            # 清除pending clear
            if alarm_id in self._pending_clears:
                del self._pending_clears[alarm_id]

        else:
            # 不应该报警
            if alarm_id in self._pending_alarms:
                del self._pending_alarms[alarm_id]

            if current_state in [AlarmState.ACTIVE, AlarmState.ACKNOWLEDGED]:
                # 检查清除延时
                if alarm_id in self._pending_clears:
                    start_time, _ = self._pending_clears[alarm_id]
                    if (now - start_time).total_seconds() >= definition.delay_off:
                        del self._pending_clears[alarm_id]
                        return (AlarmState.CLEARED, value)
                else:
                    self._pending_clears[alarm_id] = (now, value)
                    if definition.delay_off <= 0:
                        del self._pending_clears[alarm_id]
                        return (AlarmState.CLEARED, value)

        return None

    def _check_alarm_condition(self, definition: AlarmDefinition, value: float) -> bool:
        """检查报警条件"""
        alarm_type = definition.alarm_type

        if alarm_type == AlarmType.HIGH_HIGH:
            if definition.high_high_limit is not None:
                return value >= definition.high_high_limit - definition.deadband

        elif alarm_type == AlarmType.HIGH:
            if definition.high_limit is not None:
                return value >= definition.high_limit - definition.deadband

        elif alarm_type == AlarmType.LOW:
            if definition.low_limit is not None:
                return value <= definition.low_limit + definition.deadband

        elif alarm_type == AlarmType.LOW_LOW:
            if definition.low_low_limit is not None:
                return value <= definition.low_low_limit + definition.deadband

        elif alarm_type == AlarmType.DEVIATION:
            if definition.setpoint is not None and definition.high_limit is not None:
                return abs(value - definition.setpoint) > definition.high_limit

        return False


# ============================================================================
# 报警通知
# ============================================================================

class NotificationChannel(Enum):
    """通知渠道"""
    EMAIL = "email"
    SMS = "sms"
    VOICE = "voice"
    WECHAT = "wechat"
    WEBHOOK = "webhook"
    CONSOLE = "console"


@dataclass
class NotificationConfig:
    """通知配置"""
    channel: NotificationChannel
    enabled: bool = True
    min_severity: AlarmSeverity = AlarmSeverity.WARNING
    recipients: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)


class AlarmNotifier:
    """报警通知器"""

    def __init__(self):
        self.channels: Dict[NotificationChannel, NotificationConfig] = {}
        self._notification_queue: queue.Queue = queue.Queue()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

    def add_channel(self, config: NotificationConfig):
        """添加通知渠道"""
        self.channels[config.channel] = config

    def start(self):
        """启动通知服务"""
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def stop(self):
        """停止通知服务"""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5.0)

    def notify(self, alarm: AlarmInstance, event_type: str):
        """发送通知"""
        self._notification_queue.put((alarm, event_type))

    def _worker_loop(self):
        """工作循环"""
        while self._running:
            try:
                alarm, event_type = self._notification_queue.get(timeout=1.0)
                self._send_notifications(alarm, event_type)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Notification error: {e}")

    def _send_notifications(self, alarm: AlarmInstance, event_type: str):
        """发送所有渠道通知"""
        for channel, config in self.channels.items():
            if not config.enabled:
                continue
            if alarm.definition.severity.value < config.min_severity.value:
                continue

            try:
                if channel == NotificationChannel.EMAIL:
                    self._send_email(alarm, event_type, config)
                elif channel == NotificationChannel.WEBHOOK:
                    self._send_webhook(alarm, event_type, config)
                elif channel == NotificationChannel.CONSOLE:
                    self._send_console(alarm, event_type, config)
            except Exception as e:
                logger.error(f"Failed to send {channel.value} notification: {e}")

    def _send_email(self, alarm: AlarmInstance, event_type: str, config: NotificationConfig):
        """发送邮件"""
        smtp_config = config.config
        if not smtp_config.get('smtp_server'):
            return

        subject = f"[{alarm.definition.severity.name}] {event_type}: {alarm.definition.name}"
        body = f"""
报警通知
========
报警名称: {alarm.definition.name}
报警描述: {alarm.definition.description}
严重程度: {alarm.definition.severity.name}
事件类型: {event_type}
发生时间: {alarm.occurred_at}
触发值: {alarm.trigger_value}
数据源: {alarm.definition.source}
        """

        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = smtp_config.get('from_addr', 'alarm@cyrp.local')
        msg['To'] = ', '.join(config.recipients)
        msg.attach(MIMEText(body, 'plain', 'utf-8'))

        try:
            with smtplib.SMTP(smtp_config['smtp_server'], smtp_config.get('smtp_port', 25)) as server:
                if smtp_config.get('use_tls'):
                    server.starttls()
                if smtp_config.get('username'):
                    server.login(smtp_config['username'], smtp_config.get('password', ''))
                server.send_message(msg)
            logger.info(f"Email notification sent for {alarm.definition.alarm_id}")
        except Exception as e:
            logger.error(f"Email send failed: {e}")

    def _send_webhook(self, alarm: AlarmInstance, event_type: str, config: NotificationConfig):
        """发送Webhook"""
        import urllib.request

        url = config.config.get('url')
        if not url:
            return

        payload = {
            'event_type': event_type,
            'alarm': alarm.to_dict(),
            'timestamp': datetime.now().isoformat()
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            urllib.request.urlopen(req, timeout=10)
            logger.info(f"Webhook notification sent for {alarm.definition.alarm_id}")
        except Exception as e:
            logger.error(f"Webhook send failed: {e}")

    def _send_console(self, alarm: AlarmInstance, event_type: str, config: NotificationConfig):
        """控制台输出"""
        severity_colors = {
            AlarmSeverity.INFO: '\033[94m',      # 蓝色
            AlarmSeverity.WARNING: '\033[93m',   # 黄色
            AlarmSeverity.MINOR: '\033[93m',
            AlarmSeverity.MAJOR: '\033[91m',     # 红色
            AlarmSeverity.CRITICAL: '\033[91m',
            AlarmSeverity.EMERGENCY: '\033[95m', # 紫色
        }
        reset = '\033[0m'
        color = severity_colors.get(alarm.definition.severity, '')

        print(f"{color}[{event_type}] {alarm.definition.severity.name}: "
              f"{alarm.definition.name} - {alarm.definition.description}{reset}")


# ============================================================================
# 报警管理器
# ============================================================================

class AlarmManager:
    """报警管理器"""

    def __init__(self):
        self.definitions: Dict[str, AlarmDefinition] = {}
        self.instances: Dict[str, AlarmInstance] = {}
        self.active_alarms: Dict[str, AlarmInstance] = {}
        self.event_history: deque = deque(maxlen=10000)

        self.processor = AlarmProcessor()
        self.notifier = AlarmNotifier()

        self._callbacks: List[Callable] = []
        self._suppression_groups: Dict[str, Set[str]] = {}  # group -> alarm_ids
        self._shelved_alarms: Dict[str, datetime] = {}  # alarm_id -> shelve_until

        self._lock = threading.Lock()
        self._running = False
        self._escalation_thread: Optional[threading.Thread] = None

    def start(self):
        """启动报警管理器"""
        self._running = True
        self.notifier.start()

        # 启动升级检查线程
        self._escalation_thread = threading.Thread(target=self._escalation_loop, daemon=True)
        self._escalation_thread.start()

        logger.info("Alarm manager started")

    def stop(self):
        """停止报警管理器"""
        self._running = False
        self.notifier.stop()
        if self._escalation_thread:
            self._escalation_thread.join(timeout=5.0)
        logger.info("Alarm manager stopped")

    def register_alarm(self, definition: AlarmDefinition):
        """注册报警定义"""
        with self._lock:
            self.definitions[definition.alarm_id] = definition

            # 创建实例
            instance = AlarmInstance(
                instance_id=str(uuid.uuid4()),
                definition=definition
            )
            self.instances[definition.alarm_id] = instance

            # 添加到抑制组
            if definition.suppression_group:
                if definition.suppression_group not in self._suppression_groups:
                    self._suppression_groups[definition.suppression_group] = set()
                self._suppression_groups[definition.suppression_group].add(definition.alarm_id)

    def process_value(self, alarm_id: str, value: float):
        """处理数值更新"""
        with self._lock:
            if alarm_id not in self.instances:
                return

            instance = self.instances[alarm_id]
            definition = instance.definition

            if not definition.enabled:
                return

            # 检查是否被搁置
            if alarm_id in self._shelved_alarms:
                if datetime.now() > self._shelved_alarms[alarm_id]:
                    del self._shelved_alarms[alarm_id]
                else:
                    return

            # 更新当前值
            instance.current_value = value

            # 处理状态变化
            result = self.processor.process_value(definition, value, instance.state)

            if result:
                new_state, trigger_value = result
                self._handle_state_change(instance, new_state, trigger_value)

    def _handle_state_change(self, instance: AlarmInstance, new_state: AlarmState,
                            trigger_value: float):
        """处理状态变化"""
        old_state = instance.state
        instance.state = new_state
        instance.trigger_value = trigger_value
        now = datetime.now()

        if new_state == AlarmState.ACTIVE:
            instance.occurred_at = now
            instance.occurrence_count += 1
            instance.cleared_at = None
            instance.acknowledged_at = None
            instance.acknowledged_by = None

            self.active_alarms[instance.definition.alarm_id] = instance

            # 记录事件
            self._record_event(instance, "RAISED")

            # 检查抑制
            if not self._check_suppression(instance):
                # 发送通知
                self.notifier.notify(instance, "RAISED")

            # 触发回调
            self._trigger_callbacks(instance, "RAISED")

        elif new_state == AlarmState.CLEARED:
            instance.cleared_at = now

            if instance.definition.alarm_id in self.active_alarms:
                del self.active_alarms[instance.definition.alarm_id]

            self._record_event(instance, "CLEARED")

            if instance.definition.auto_acknowledge:
                instance.state = AlarmState.NORMAL
                instance.acknowledged_at = now
                instance.acknowledged_by = "AUTO"

            self.notifier.notify(instance, "CLEARED")
            self._trigger_callbacks(instance, "CLEARED")

    def acknowledge(self, alarm_id: str, user: str, comment: Optional[str] = None) -> bool:
        """确认报警"""
        with self._lock:
            if alarm_id not in self.instances:
                return False

            instance = self.instances[alarm_id]

            if instance.state not in [AlarmState.ACTIVE]:
                return False

            if instance.definition.requires_comment and not comment:
                return False

            instance.state = AlarmState.ACKNOWLEDGED
            instance.acknowledged_at = datetime.now()
            instance.acknowledged_by = user
            instance.acknowledge_comment = comment

            self._record_event(instance, "ACKNOWLEDGED", user, comment)
            self._trigger_callbacks(instance, "ACKNOWLEDGED")

            return True

    def acknowledge_all(self, user: str, severity: Optional[AlarmSeverity] = None) -> int:
        """批量确认报警"""
        count = 0
        with self._lock:
            for alarm_id, instance in list(self.active_alarms.items()):
                if severity and instance.definition.severity != severity:
                    continue
                if instance.state == AlarmState.ACTIVE:
                    instance.state = AlarmState.ACKNOWLEDGED
                    instance.acknowledged_at = datetime.now()
                    instance.acknowledged_by = user
                    self._record_event(instance, "ACKNOWLEDGED", user, "Batch acknowledge")
                    count += 1
        return count

    def shelve(self, alarm_id: str, duration_minutes: int, user: str) -> bool:
        """搁置报警"""
        with self._lock:
            if alarm_id not in self.definitions:
                return False

            shelve_until = datetime.now() + timedelta(minutes=duration_minutes)
            self._shelved_alarms[alarm_id] = shelve_until

            if alarm_id in self.instances:
                instance = self.instances[alarm_id]
                instance.state = AlarmState.SHELVED
                self._record_event(instance, "SHELVED", user, f"Duration: {duration_minutes} min")

            return True

    def unshelve(self, alarm_id: str, user: str) -> bool:
        """取消搁置"""
        with self._lock:
            if alarm_id in self._shelved_alarms:
                del self._shelved_alarms[alarm_id]

                if alarm_id in self.instances:
                    instance = self.instances[alarm_id]
                    instance.state = AlarmState.NORMAL
                    self._record_event(instance, "UNSHELVED", user)

                return True
            return False

    def _check_suppression(self, instance: AlarmInstance) -> bool:
        """检查是否被抑制"""
        group = instance.definition.suppression_group
        if not group or group not in self._suppression_groups:
            return False

        # 检查组内是否有更高优先级的报警
        for alarm_id in self._suppression_groups[group]:
            if alarm_id == instance.definition.alarm_id:
                continue
            if alarm_id in self.active_alarms:
                other = self.active_alarms[alarm_id]
                if other.definition.priority < instance.definition.priority:
                    instance.state = AlarmState.SUPPRESSED
                    return True

        return False

    def _escalation_loop(self):
        """升级检查循环"""
        while self._running:
            time.sleep(10)

            with self._lock:
                now = datetime.now()
                for alarm_id, instance in self.active_alarms.items():
                    if instance.state != AlarmState.ACTIVE:
                        continue
                    if instance.escalated:
                        continue

                    definition = instance.definition
                    if definition.escalation_time and instance.occurred_at:
                        elapsed = (now - instance.occurred_at).total_seconds()
                        if elapsed >= definition.escalation_time:
                            instance.escalated = True
                            instance.escalation_level += 1
                            self._record_event(instance, "ESCALATED")
                            self.notifier.notify(instance, "ESCALATED")

    def _record_event(self, instance: AlarmInstance, event_type: str,
                     user: Optional[str] = None, comment: Optional[str] = None):
        """记录事件"""
        event = AlarmEvent(
            event_id=str(uuid.uuid4()),
            instance_id=instance.instance_id,
            alarm_id=instance.definition.alarm_id,
            event_type=event_type,
            timestamp=datetime.now(),
            user=user,
            comment=comment
        )
        self.event_history.append(event)

    def _trigger_callbacks(self, instance: AlarmInstance, event_type: str):
        """触发回调"""
        for callback in self._callbacks:
            try:
                callback(instance, event_type)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def register_callback(self, callback: Callable):
        """注册回调"""
        self._callbacks.append(callback)

    # ========== 查询接口 ==========

    def get_active_alarms(self, severity: Optional[AlarmSeverity] = None,
                         area: Optional[str] = None) -> List[AlarmInstance]:
        """获取活动报警"""
        with self._lock:
            alarms = list(self.active_alarms.values())

            if severity:
                alarms = [a for a in alarms if a.definition.severity == severity]
            if area:
                alarms = [a for a in alarms if a.definition.area == area]

            return sorted(alarms, key=lambda x: (-x.definition.severity.value, x.occurred_at))

    def get_alarm_summary(self) -> Dict[str, int]:
        """获取报警摘要"""
        with self._lock:
            summary = {s.name: 0 for s in AlarmSeverity}
            for instance in self.active_alarms.values():
                summary[instance.definition.severity.name] += 1
            summary['TOTAL'] = len(self.active_alarms)
            return summary

    def get_event_history(self, alarm_id: Optional[str] = None,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None,
                         limit: int = 100) -> List[AlarmEvent]:
        """获取事件历史"""
        events = list(self.event_history)

        if alarm_id:
            events = [e for e in events if e.alarm_id == alarm_id]
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]

        return events[-limit:]


# ============================================================================
# 便捷函数
# ============================================================================

def create_cyrp_alarm_system() -> AlarmManager:
    """创建穿黄工程报警系统"""
    manager = AlarmManager()

    # 添加控制台通知
    manager.notifier.add_channel(NotificationConfig(
        channel=NotificationChannel.CONSOLE,
        enabled=True,
        min_severity=AlarmSeverity.WARNING
    ))

    # 注册穿黄工程报警
    alarms = [
        # 流量报警
        AlarmDefinition("FLOW_HH", "进口流量高高", "进口流量超过高高限", AlarmType.HIGH_HIGH,
                       AlarmSeverity.CRITICAL, "inlet_flow", high_high_limit=320.0,
                       delay_on=5.0, area="进口", group="流量"),
        AlarmDefinition("FLOW_H", "进口流量高", "进口流量超过高限", AlarmType.HIGH,
                       AlarmSeverity.WARNING, "inlet_flow", high_limit=280.0,
                       delay_on=10.0, area="进口", group="流量"),
        AlarmDefinition("FLOW_L", "进口流量低", "进口流量低于低限", AlarmType.LOW,
                       AlarmSeverity.WARNING, "inlet_flow", low_limit=50.0,
                       delay_on=30.0, area="进口", group="流量"),

        # 压力报警
        AlarmDefinition("PRESS_HH", "进口压力高高", "进口压力超过高高限", AlarmType.HIGH_HIGH,
                       AlarmSeverity.EMERGENCY, "inlet_pressure", high_high_limit=1000.0,
                       delay_on=3.0, escalation_time=60.0, area="进口", group="压力"),
        AlarmDefinition("PRESS_H", "进口压力高", "进口压力超过高限", AlarmType.HIGH,
                       AlarmSeverity.MAJOR, "inlet_pressure", high_limit=800.0,
                       delay_on=5.0, area="进口", group="压力"),
        AlarmDefinition("PRESS_LL", "进口压力低低", "进口压力低于低低限(负压)", AlarmType.LOW_LOW,
                       AlarmSeverity.CRITICAL, "inlet_pressure", low_low_limit=-50.0,
                       delay_on=2.0, area="进口", group="压力"),

        # 渗漏报警
        AlarmDefinition("LEAK_H", "渗漏量高", "渗漏量超过警戒值", AlarmType.HIGH,
                       AlarmSeverity.MAJOR, "leakage_rate", high_limit=0.1,
                       delay_on=60.0, area="隧洞", group="渗漏"),
        AlarmDefinition("LEAK_HH", "渗漏量高高", "渗漏量严重超标", AlarmType.HIGH_HIGH,
                       AlarmSeverity.CRITICAL, "leakage_rate", high_high_limit=0.5,
                       delay_on=10.0, escalation_time=300.0, area="隧洞", group="渗漏"),

        # 振动报警
        AlarmDefinition("VIB_H", "振动高", "结构振动超过警戒值", AlarmType.HIGH,
                       AlarmSeverity.WARNING, "vibration_max", high_limit=5.0,
                       delay_on=10.0, area="隧洞", group="振动"),
        AlarmDefinition("VIB_HH", "振动高高", "结构振动严重超标", AlarmType.HIGH_HIGH,
                       AlarmSeverity.MAJOR, "vibration_max", high_high_limit=10.0,
                       delay_on=5.0, area="隧洞", group="振动"),

        # 通信报警
        AlarmDefinition("COM_LOSS", "通信中断", "与现场设备通信中断", AlarmType.COMMUNICATION_LOSS,
                       AlarmSeverity.MAJOR, "plc_heartbeat", delay_on=30.0,
                       escalation_time=300.0, area="系统", group="通信"),

        # 设备报警
        AlarmDefinition("VALVE_FAULT", "阀门故障", "阀门动作异常", AlarmType.EQUIPMENT_FAULT,
                       AlarmSeverity.MAJOR, "valve_status", delay_on=5.0,
                       area="执行器", group="设备"),
        AlarmDefinition("PUMP_FAULT", "水泵故障", "水泵运行异常", AlarmType.EQUIPMENT_FAULT,
                       AlarmSeverity.MAJOR, "pump_status", delay_on=5.0,
                       area="执行器", group="设备"),
    ]

    for alarm in alarms:
        manager.register_alarm(alarm)

    return manager
