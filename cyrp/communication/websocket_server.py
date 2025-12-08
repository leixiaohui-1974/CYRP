"""
WebSocket 实时推送服务 - WebSocket Real-time Push Service

实现基于 WebSocket 的实时数据推送，支持：
- 告警实时推送
- 监控数据订阅
- 系统状态更新
- 事件广播

Implements WebSocket-based real-time data push supporting:
- Real-time alert notifications
- Monitoring data subscriptions
- System status updates
- Event broadcasting
"""

import asyncio
import json
import time
import uuid
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Set, Callable
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """消息类型"""
    # 系统消息
    WELCOME = "welcome"
    HEARTBEAT = "heartbeat"
    ERROR = "error"

    # 订阅相关
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    SUBSCRIBED = "subscribed"
    UNSUBSCRIBED = "unsubscribed"

    # 数据推送
    ALERT = "alert"
    METRIC = "metric"
    STATUS = "status"
    EVENT = "event"

    # 控制命令
    COMMAND = "command"
    COMMAND_RESPONSE = "command_response"


class SubscriptionChannel(Enum):
    """订阅频道"""
    ALERTS = "alerts"                    # 告警频道
    METRICS = "metrics"                  # 指标频道
    STATUS = "status"                    # 系统状态频道
    EVENTS = "events"                    # 事件频道
    ALL = "all"                          # 全部频道


@dataclass
class WebSocketMessage:
    """WebSocket消息"""
    type: MessageType
    payload: Any
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_json(self) -> str:
        """转换为JSON"""
        return json.dumps({
            "type": self.type.value,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "message_id": self.message_id
        })

    @classmethod
    def from_json(cls, json_str: str) -> 'WebSocketMessage':
        """从JSON解析"""
        data = json.loads(json_str)
        return cls(
            type=MessageType(data["type"]),
            payload=data["payload"],
            timestamp=data.get("timestamp", time.time()),
            message_id=data.get("message_id", str(uuid.uuid4()))
        )


@dataclass
class WebSocketClient:
    """WebSocket客户端"""
    client_id: str
    connected_at: float
    subscriptions: Set[SubscriptionChannel] = field(default_factory=set)
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_heartbeat: float = field(default_factory=time.time)
    message_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    is_connected: bool = True

    def __hash__(self):
        return hash(self.client_id)

    def __eq__(self, other):
        if isinstance(other, WebSocketClient):
            return self.client_id == other.client_id
        return False


class WebSocketConnectionManager:
    """WebSocket连接管理器"""

    def __init__(self, heartbeat_interval: float = 30.0):
        """
        初始化连接管理器

        Args:
            heartbeat_interval: 心跳间隔（秒）
        """
        self.heartbeat_interval = heartbeat_interval
        self._clients: Dict[str, WebSocketClient] = {}
        self._channel_subscribers: Dict[SubscriptionChannel, Set[str]] = defaultdict(set)
        self._lock = threading.Lock()
        self._message_handlers: Dict[MessageType, List[Callable]] = defaultdict(list)
        self._running = False
        self._stats = {
            "total_connections": 0,
            "total_messages_sent": 0,
            "total_messages_received": 0,
            "active_connections": 0
        }

    def connect(self, client_id: str, user_id: Optional[str] = None) -> WebSocketClient:
        """
        客户端连接

        Args:
            client_id: 客户端ID
            user_id: 用户ID（可选）

        Returns:
            WebSocketClient: 客户端对象
        """
        with self._lock:
            client = WebSocketClient(
                client_id=client_id,
                connected_at=time.time(),
                user_id=user_id
            )
            self._clients[client_id] = client
            self._stats["total_connections"] += 1
            self._stats["active_connections"] = len(self._clients)

            logger.info(f"WebSocket client connected: {client_id}")

            return client

    def disconnect(self, client_id: str):
        """
        客户端断开

        Args:
            client_id: 客户端ID
        """
        with self._lock:
            if client_id in self._clients:
                client = self._clients[client_id]
                client.is_connected = False

                # 移除所有订阅
                for channel in client.subscriptions:
                    self._channel_subscribers[channel].discard(client_id)

                del self._clients[client_id]
                self._stats["active_connections"] = len(self._clients)

                logger.info(f"WebSocket client disconnected: {client_id}")

    def subscribe(self, client_id: str, channel: SubscriptionChannel) -> bool:
        """
        订阅频道

        Args:
            client_id: 客户端ID
            channel: 频道

        Returns:
            bool: 是否成功
        """
        with self._lock:
            if client_id not in self._clients:
                return False

            client = self._clients[client_id]
            client.subscriptions.add(channel)
            self._channel_subscribers[channel].add(client_id)

            logger.debug(f"Client {client_id} subscribed to {channel.value}")
            return True

    def unsubscribe(self, client_id: str, channel: SubscriptionChannel) -> bool:
        """
        取消订阅

        Args:
            client_id: 客户端ID
            channel: 频道

        Returns:
            bool: 是否成功
        """
        with self._lock:
            if client_id not in self._clients:
                return False

            client = self._clients[client_id]
            client.subscriptions.discard(channel)
            self._channel_subscribers[channel].discard(client_id)

            logger.debug(f"Client {client_id} unsubscribed from {channel.value}")
            return True

    def get_subscribers(self, channel: SubscriptionChannel) -> List[str]:
        """获取频道订阅者列表"""
        with self._lock:
            return list(self._channel_subscribers[channel])

    def get_client(self, client_id: str) -> Optional[WebSocketClient]:
        """获取客户端"""
        return self._clients.get(client_id)

    def get_all_clients(self) -> List[WebSocketClient]:
        """获取所有客户端"""
        return list(self._clients.values())

    async def send_to_client(self, client_id: str, message: WebSocketMessage) -> bool:
        """
        发送消息到指定客户端

        Args:
            client_id: 客户端ID
            message: 消息

        Returns:
            bool: 是否成功
        """
        client = self._clients.get(client_id)
        if not client or not client.is_connected:
            return False

        try:
            await client.message_queue.put(message)
            self._stats["total_messages_sent"] += 1
            return True
        except Exception as e:
            logger.error(f"Failed to send message to {client_id}: {e}")
            return False

    async def broadcast_to_channel(
        self,
        channel: SubscriptionChannel,
        message: WebSocketMessage
    ) -> int:
        """
        广播消息到频道

        Args:
            channel: 频道
            message: 消息

        Returns:
            int: 成功发送的数量
        """
        subscribers = self.get_subscribers(channel)
        sent_count = 0

        for client_id in subscribers:
            if await self.send_to_client(client_id, message):
                sent_count += 1

        # ALL频道也要广播
        if channel != SubscriptionChannel.ALL:
            for client_id in self.get_subscribers(SubscriptionChannel.ALL):
                if client_id not in subscribers:
                    if await self.send_to_client(client_id, message):
                        sent_count += 1

        return sent_count

    async def broadcast_all(self, message: WebSocketMessage) -> int:
        """
        广播消息到所有客户端

        Args:
            message: 消息

        Returns:
            int: 成功发送的数量
        """
        sent_count = 0
        for client_id in self._clients.keys():
            if await self.send_to_client(client_id, message):
                sent_count += 1
        return sent_count

    def register_handler(self, message_type: MessageType, handler: Callable):
        """注册消息处理器"""
        self._message_handlers[message_type].append(handler)

    async def handle_message(
        self,
        client_id: str,
        message: WebSocketMessage
    ) -> Optional[WebSocketMessage]:
        """
        处理收到的消息

        Args:
            client_id: 客户端ID
            message: 消息

        Returns:
            Optional[WebSocketMessage]: 响应消息
        """
        self._stats["total_messages_received"] += 1

        # 内置消息处理
        if message.type == MessageType.SUBSCRIBE:
            channel_name = message.payload.get("channel", "")
            try:
                channel = SubscriptionChannel(channel_name)
                success = self.subscribe(client_id, channel)
                return WebSocketMessage(
                    type=MessageType.SUBSCRIBED,
                    payload={"channel": channel_name, "success": success}
                )
            except ValueError:
                return WebSocketMessage(
                    type=MessageType.ERROR,
                    payload={"error": f"Unknown channel: {channel_name}"}
                )

        elif message.type == MessageType.UNSUBSCRIBE:
            channel_name = message.payload.get("channel", "")
            try:
                channel = SubscriptionChannel(channel_name)
                success = self.unsubscribe(client_id, channel)
                return WebSocketMessage(
                    type=MessageType.UNSUBSCRIBED,
                    payload={"channel": channel_name, "success": success}
                )
            except ValueError:
                return WebSocketMessage(
                    type=MessageType.ERROR,
                    payload={"error": f"Unknown channel: {channel_name}"}
                )

        elif message.type == MessageType.HEARTBEAT:
            client = self._clients.get(client_id)
            if client:
                client.last_heartbeat = time.time()
            return WebSocketMessage(
                type=MessageType.HEARTBEAT,
                payload={"server_time": time.time()}
            )

        # 调用自定义处理器
        handlers = self._message_handlers.get(message.type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(client_id, message)
                else:
                    result = handler(client_id, message)
                if result:
                    return result
            except Exception as e:
                logger.error(f"Handler error: {e}")

        return None

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            "channels": {
                channel.value: len(subscribers)
                for channel, subscribers in self._channel_subscribers.items()
            }
        }

    def update_heartbeat(self, client_id: str):
        """更新心跳时间"""
        client = self._clients.get(client_id)
        if client:
            client.last_heartbeat = time.time()

    def cleanup_stale_connections(self, timeout: float = 60.0) -> List[str]:
        """
        清理过期连接

        Args:
            timeout: 超时时间（秒）

        Returns:
            List[str]: 清理的客户端ID列表
        """
        now = time.time()
        stale_clients = []

        with self._lock:
            for client_id, client in list(self._clients.items()):
                if now - client.last_heartbeat > timeout:
                    stale_clients.append(client_id)

        for client_id in stale_clients:
            self.disconnect(client_id)
            logger.info(f"Cleaned up stale connection: {client_id}")

        return stale_clients


class RealtimePushService:
    """实时推送服务"""

    def __init__(
        self,
        connection_manager: Optional[WebSocketConnectionManager] = None
    ):
        """
        初始化实时推送服务

        Args:
            connection_manager: 连接管理器
        """
        self.connection_manager = connection_manager or WebSocketConnectionManager()
        self._alert_buffer: List[Dict] = []
        self._metric_buffer: Dict[str, Any] = {}
        self._push_interval = 1.0  # 推送间隔（秒）
        self._running = False
        self._push_task = None

    async def push_alert(self, alert: Dict) -> int:
        """
        推送告警

        Args:
            alert: 告警数据

        Returns:
            int: 推送的客户端数量
        """
        message = WebSocketMessage(
            type=MessageType.ALERT,
            payload=alert
        )
        return await self.connection_manager.broadcast_to_channel(
            SubscriptionChannel.ALERTS,
            message
        )

    async def push_metric(self, metric_name: str, value: float, labels: Dict = None) -> int:
        """
        推送指标

        Args:
            metric_name: 指标名称
            value: 值
            labels: 标签

        Returns:
            int: 推送的客户端数量
        """
        message = WebSocketMessage(
            type=MessageType.METRIC,
            payload={
                "name": metric_name,
                "value": value,
                "labels": labels or {},
                "timestamp": time.time()
            }
        )
        return await self.connection_manager.broadcast_to_channel(
            SubscriptionChannel.METRICS,
            message
        )

    async def push_metrics_batch(self, metrics: Dict[str, float]) -> int:
        """
        批量推送指标

        Args:
            metrics: 指标字典 {name: value}

        Returns:
            int: 推送的客户端数量
        """
        message = WebSocketMessage(
            type=MessageType.METRIC,
            payload={
                "batch": True,
                "metrics": metrics,
                "timestamp": time.time()
            }
        )
        return await self.connection_manager.broadcast_to_channel(
            SubscriptionChannel.METRICS,
            message
        )

    async def push_status(self, status: Dict) -> int:
        """
        推送系统状态

        Args:
            status: 状态数据

        Returns:
            int: 推送的客户端数量
        """
        message = WebSocketMessage(
            type=MessageType.STATUS,
            payload=status
        )
        return await self.connection_manager.broadcast_to_channel(
            SubscriptionChannel.STATUS,
            message
        )

    async def push_event(self, event: Dict) -> int:
        """
        推送事件

        Args:
            event: 事件数据

        Returns:
            int: 推送的客户端数量
        """
        message = WebSocketMessage(
            type=MessageType.EVENT,
            payload=event
        )
        return await self.connection_manager.broadcast_to_channel(
            SubscriptionChannel.EVENTS,
            message
        )

    def buffer_alert(self, alert: Dict):
        """缓冲告警（用于批量推送）"""
        self._alert_buffer.append(alert)

    def buffer_metric(self, name: str, value: float):
        """缓冲指标（用于批量推送）"""
        self._metric_buffer[name] = value

    async def flush_buffers(self) -> Dict[str, int]:
        """
        刷新缓冲区

        Returns:
            Dict[str, int]: 各类型推送的数量
        """
        result = {"alerts": 0, "metrics": 0}

        # 推送缓冲的告警
        for alert in self._alert_buffer:
            result["alerts"] += await self.push_alert(alert)
        self._alert_buffer.clear()

        # 推送缓冲的指标
        if self._metric_buffer:
            result["metrics"] = await self.push_metrics_batch(self._metric_buffer)
            self._metric_buffer.clear()

        return result

    async def start_periodic_push(self, interval: float = None):
        """
        启动周期性推送

        Args:
            interval: 推送间隔（秒）
        """
        if interval:
            self._push_interval = interval

        self._running = True

        while self._running:
            try:
                await self.flush_buffers()
                await asyncio.sleep(self._push_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Periodic push error: {e}")
                await asyncio.sleep(1.0)

    def stop_periodic_push(self):
        """停止周期性推送"""
        self._running = False


class EventBusWebSocketBridge:
    """事件总线与WebSocket桥接器"""

    def __init__(
        self,
        push_service: RealtimePushService,
        event_bus=None
    ):
        """
        初始化桥接器

        Args:
            push_service: 推送服务
            event_bus: 事件总线实例
        """
        self.push_service = push_service
        self.event_bus = event_bus
        self._subscriptions = []

    def setup_subscriptions(self):
        """设置事件订阅"""
        if not self.event_bus:
            return

        # 订阅告警事件
        self._subscribe("alert.*", self._on_alert)

        # 订阅指标事件
        self._subscribe("metric.*", self._on_metric)

        # 订阅状态事件
        self._subscribe("status.*", self._on_status)

        # 订阅系统事件
        self._subscribe("system.*", self._on_system_event)

    def _subscribe(self, pattern: str, handler: Callable):
        """订阅事件"""
        try:
            sub_id = self.event_bus.subscribe(
                topic_pattern=pattern,
                handler=handler,
                subscriber_id="websocket_bridge"
            )
            self._subscriptions.append(sub_id)
        except Exception as e:
            logger.error(f"Failed to subscribe to {pattern}: {e}")

    async def _on_alert(self, event):
        """告警事件处理"""
        await self.push_service.push_alert(event.payload)

    async def _on_metric(self, event):
        """指标事件处理"""
        payload = event.payload
        if isinstance(payload, dict):
            await self.push_service.push_metrics_batch(payload)
        else:
            # 从topic提取指标名
            metric_name = event.topic.replace("metric.", "")
            await self.push_service.push_metric(metric_name, payload)

    async def _on_status(self, event):
        """状态事件处理"""
        await self.push_service.push_status(event.payload)

    async def _on_system_event(self, event):
        """系统事件处理"""
        await self.push_service.push_event({
            "topic": event.topic,
            "payload": event.payload,
            "source": event.source,
            "timestamp": event.timestamp.isoformat() if hasattr(event.timestamp, 'isoformat') else str(event.timestamp)
        })

    def teardown(self):
        """清理订阅"""
        if self.event_bus:
            for sub_id in self._subscriptions:
                try:
                    self.event_bus.unsubscribe(sub_id)
                except Exception:
                    pass
        self._subscriptions.clear()


def create_realtime_push_system(
    event_bus=None,
    heartbeat_interval: float = 30.0
) -> Dict[str, Any]:
    """
    创建实时推送系统

    Args:
        event_bus: 事件总线实例（可选）
        heartbeat_interval: 心跳间隔

    Returns:
        Dict containing connection_manager, push_service, and bridge
    """
    connection_manager = WebSocketConnectionManager(heartbeat_interval)
    push_service = RealtimePushService(connection_manager)

    bridge = None
    if event_bus:
        bridge = EventBusWebSocketBridge(push_service, event_bus)
        bridge.setup_subscriptions()

    return {
        "connection_manager": connection_manager,
        "push_service": push_service,
        "bridge": bridge
    }
