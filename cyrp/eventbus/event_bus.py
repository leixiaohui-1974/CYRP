"""
Event Bus Module for CYRP
穿黄工程事件总线模块

实现模块间的松耦合通信，支持发布订阅模式、事件过滤、优先级队列等功能
"""

import asyncio
import threading
import time
import uuid
import json
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Callable, Set, Tuple,
    TypeVar, Generic, Coroutine, Union
)
from collections import defaultdict
from queue import PriorityQueue
from concurrent.futures import ThreadPoolExecutor
import weakref
import logging

logger = logging.getLogger(__name__)


# ============================================================
# 枚举定义
# ============================================================

class EventPriority(Enum):
    """事件优先级"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class EventStatus(Enum):
    """事件状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    DELIVERED = "delivered"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class DeliveryMode(Enum):
    """投递模式"""
    BROADCAST = "broadcast"      # 广播给所有订阅者
    ROUND_ROBIN = "round_robin"  # 轮询投递
    LOAD_BALANCE = "load_balance"  # 负载均衡
    FIRST_MATCH = "first_match"  # 第一个匹配的订阅者


class SubscriptionType(Enum):
    """订阅类型"""
    PERMANENT = "permanent"  # 永久订阅
    TEMPORARY = "temporary"  # 临时订阅
    ONE_TIME = "one_time"    # 一次性订阅


# ============================================================
# 数据类定义
# ============================================================

@dataclass
class Event:
    """事件"""
    event_id: str
    topic: str
    payload: Any
    source: str
    priority: EventPriority = EventPriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    ttl_seconds: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """检查事件是否过期"""
        if self.ttl_seconds is None:
            return False
        elapsed = (datetime.now() - self.timestamp).total_seconds()
        return elapsed > self.ttl_seconds

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'event_id': self.event_id,
            'topic': self.topic,
            'payload': self.payload,
            'source': self.source,
            'priority': self.priority.value,
            'timestamp': self.timestamp.isoformat(),
            'correlation_id': self.correlation_id,
            'reply_to': self.reply_to,
            'ttl_seconds': self.ttl_seconds,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """从字典创建"""
        return cls(
            event_id=data['event_id'],
            topic=data['topic'],
            payload=data['payload'],
            source=data['source'],
            priority=EventPriority(data['priority']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            correlation_id=data.get('correlation_id'),
            reply_to=data.get('reply_to'),
            ttl_seconds=data.get('ttl_seconds'),
            metadata=data.get('metadata', {})
        )


@dataclass
class Subscription:
    """订阅"""
    subscription_id: str
    topic_pattern: str
    handler: Callable
    subscriber_id: str
    subscription_type: SubscriptionType = SubscriptionType.PERMANENT
    priority_filter: Optional[Set[EventPriority]] = None
    metadata_filter: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    is_async: bool = False
    max_retries: int = 3

    @property
    def is_expired(self) -> bool:
        """检查订阅是否过期"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def matches_topic(self, topic: str) -> bool:
        """检查主题是否匹配"""
        # 支持通配符: * 匹配单级, # 匹配多级
        pattern_parts = self.topic_pattern.split('/')
        topic_parts = topic.split('/')

        return self._match_parts(pattern_parts, topic_parts)

    def _match_parts(self, pattern: List[str], topic: List[str]) -> bool:
        """递归匹配主题部分"""
        if not pattern and not topic:
            return True
        if not pattern:
            return False
        if pattern[0] == '#':
            return True
        if not topic:
            return False
        if pattern[0] == '*' or pattern[0] == topic[0]:
            return self._match_parts(pattern[1:], topic[1:])
        return False

    def matches_event(self, event: Event) -> bool:
        """检查事件是否匹配订阅条件"""
        # 主题匹配
        if not self.matches_topic(event.topic):
            return False

        # 优先级过滤
        if self.priority_filter and event.priority not in self.priority_filter:
            return False

        # 元数据过滤
        if self.metadata_filter:
            for key, value in self.metadata_filter.items():
                if event.metadata.get(key) != value:
                    return False

        return True


@dataclass
class DeliveryResult:
    """投递结果"""
    event_id: str
    subscription_id: str
    status: EventStatus
    delivered_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0


@dataclass
class EventBusStats:
    """事件总线统计"""
    total_events_published: int = 0
    total_events_delivered: int = 0
    total_events_failed: int = 0
    total_events_expired: int = 0
    total_subscriptions: int = 0
    active_subscriptions: int = 0
    events_per_topic: Dict[str, int] = field(default_factory=dict)
    avg_delivery_time_ms: float = 0.0


# ============================================================
# 事件存储
# ============================================================

class EventStore(ABC):
    """事件存储抽象基类"""

    @abstractmethod
    async def save(self, event: Event) -> bool:
        """保存事件"""
        pass

    @abstractmethod
    async def get(self, event_id: str) -> Optional[Event]:
        """获取事件"""
        pass

    @abstractmethod
    async def get_by_topic(self, topic: str, limit: int = 100) -> List[Event]:
        """按主题获取事件"""
        pass

    @abstractmethod
    async def delete(self, event_id: str) -> bool:
        """删除事件"""
        pass


class InMemoryEventStore(EventStore):
    """内存事件存储"""

    def __init__(self, max_events: int = 10000):
        self.max_events = max_events
        self.events: Dict[str, Event] = {}
        self.topic_index: Dict[str, List[str]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def save(self, event: Event) -> bool:
        """保存事件"""
        async with self._lock:
            # 检查容量
            if len(self.events) >= self.max_events:
                # 删除最旧的事件
                oldest_id = min(self.events.keys(),
                               key=lambda x: self.events[x].timestamp)
                await self.delete(oldest_id)

            self.events[event.event_id] = event
            self.topic_index[event.topic].append(event.event_id)
            return True

    async def get(self, event_id: str) -> Optional[Event]:
        """获取事件"""
        return self.events.get(event_id)

    async def get_by_topic(self, topic: str, limit: int = 100) -> List[Event]:
        """按主题获取事件"""
        event_ids = self.topic_index.get(topic, [])[-limit:]
        return [self.events[eid] for eid in event_ids if eid in self.events]

    async def delete(self, event_id: str) -> bool:
        """删除事件"""
        if event_id in self.events:
            event = self.events.pop(event_id)
            if event.topic in self.topic_index:
                try:
                    self.topic_index[event.topic].remove(event_id)
                except ValueError:
                    pass
            return True
        return False


# ============================================================
# 事件处理器
# ============================================================

class EventHandler:
    """事件处理器封装"""

    def __init__(self, handler: Callable, is_async: bool = False):
        self.handler = handler
        self.is_async = is_async
        self._executor = ThreadPoolExecutor(max_workers=4)

    async def handle(self, event: Event) -> Any:
        """处理事件"""
        if self.is_async:
            return await self.handler(event)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._executor, self.handler, event
            )


class EventInterceptor(ABC):
    """事件拦截器基类"""

    @abstractmethod
    async def before_publish(self, event: Event) -> Optional[Event]:
        """发布前拦截，返回None则取消发布"""
        pass

    @abstractmethod
    async def after_publish(self, event: Event, results: List[DeliveryResult]):
        """发布后拦截"""
        pass


class LoggingInterceptor(EventInterceptor):
    """日志拦截器"""

    async def before_publish(self, event: Event) -> Optional[Event]:
        """发布前记录日志"""
        logger.debug(f"Publishing event: {event.event_id} to topic: {event.topic}")
        return event

    async def after_publish(self, event: Event, results: List[DeliveryResult]):
        """发布后记录日志"""
        success = sum(1 for r in results if r.status == EventStatus.DELIVERED)
        logger.debug(f"Event {event.event_id} delivered to {success}/{len(results)} subscribers")


class MetricsInterceptor(EventInterceptor):
    """指标拦截器"""

    def __init__(self):
        self.publish_count = 0
        self.delivery_count = 0
        self.failure_count = 0
        self.latencies: List[float] = []

    async def before_publish(self, event: Event) -> Optional[Event]:
        """发布前记录开始时间"""
        event.metadata['_publish_start'] = time.time()
        self.publish_count += 1
        return event

    async def after_publish(self, event: Event, results: List[DeliveryResult]):
        """发布后记录指标"""
        start_time = event.metadata.pop('_publish_start', time.time())
        latency = (time.time() - start_time) * 1000
        self.latencies.append(latency)

        for result in results:
            if result.status == EventStatus.DELIVERED:
                self.delivery_count += 1
            elif result.status == EventStatus.FAILED:
                self.failure_count += 1


# ============================================================
# 订阅管理器
# ============================================================

class SubscriptionManager:
    """订阅管理器"""

    def __init__(self):
        self.subscriptions: Dict[str, Subscription] = {}
        self.topic_subscriptions: Dict[str, Set[str]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def add(self, subscription: Subscription) -> str:
        """添加订阅"""
        async with self._lock:
            self.subscriptions[subscription.subscription_id] = subscription

            # 添加到主题索引
            base_topic = subscription.topic_pattern.split('/')[0]
            self.topic_subscriptions[base_topic].add(subscription.subscription_id)

            return subscription.subscription_id

    async def remove(self, subscription_id: str) -> bool:
        """移除订阅"""
        async with self._lock:
            if subscription_id not in self.subscriptions:
                return False

            subscription = self.subscriptions.pop(subscription_id)

            # 从主题索引移除
            base_topic = subscription.topic_pattern.split('/')[0]
            self.topic_subscriptions[base_topic].discard(subscription_id)

            return True

    async def get_matching(self, event: Event) -> List[Subscription]:
        """获取匹配的订阅"""
        matching = []

        # 清理过期订阅
        expired = [sid for sid, sub in self.subscriptions.items() if sub.is_expired]
        for sid in expired:
            await self.remove(sid)

        # 查找匹配的订阅
        for subscription in self.subscriptions.values():
            if subscription.matches_event(event):
                matching.append(subscription)

        return matching

    async def get_all(self) -> List[Subscription]:
        """获取所有订阅"""
        return list(self.subscriptions.values())

    async def get_stats(self) -> Dict[str, int]:
        """获取订阅统计"""
        active = sum(1 for sub in self.subscriptions.values() if not sub.is_expired)
        return {
            'total': len(self.subscriptions),
            'active': active,
            'expired': len(self.subscriptions) - active
        }


# ============================================================
# 事件分发器
# ============================================================

class EventDispatcher:
    """事件分发器"""

    def __init__(
        self,
        delivery_mode: DeliveryMode = DeliveryMode.BROADCAST,
        max_workers: int = 10
    ):
        self.delivery_mode = delivery_mode
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._round_robin_index: Dict[str, int] = defaultdict(int)
        self._lock = asyncio.Lock()

    async def dispatch(
        self,
        event: Event,
        subscriptions: List[Subscription]
    ) -> List[DeliveryResult]:
        """分发事件到订阅者"""
        if not subscriptions:
            return []

        # 根据投递模式选择订阅者
        targets = await self._select_targets(event, subscriptions)

        # 并发投递
        tasks = [self._deliver(event, sub) for sub in targets]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理结果
        delivery_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                delivery_results.append(DeliveryResult(
                    event_id=event.event_id,
                    subscription_id=targets[i].subscription_id,
                    status=EventStatus.FAILED,
                    error_message=str(result)
                ))
            else:
                delivery_results.append(result)

        return delivery_results

    async def _select_targets(
        self,
        event: Event,
        subscriptions: List[Subscription]
    ) -> List[Subscription]:
        """根据投递模式选择目标订阅者"""
        if self.delivery_mode == DeliveryMode.BROADCAST:
            return subscriptions

        elif self.delivery_mode == DeliveryMode.ROUND_ROBIN:
            async with self._lock:
                index = self._round_robin_index[event.topic]
                self._round_robin_index[event.topic] = (index + 1) % len(subscriptions)
                return [subscriptions[index]]

        elif self.delivery_mode == DeliveryMode.FIRST_MATCH:
            return [subscriptions[0]]

        elif self.delivery_mode == DeliveryMode.LOAD_BALANCE:
            # 简单的随机负载均衡
            import random
            return [random.choice(subscriptions)]

        return subscriptions

    async def _deliver(
        self,
        event: Event,
        subscription: Subscription
    ) -> DeliveryResult:
        """投递事件到单个订阅者"""
        start_time = datetime.now()

        try:
            handler = EventHandler(subscription.handler, subscription.is_async)
            await handler.handle(event)

            # 一次性订阅处理后标记
            if subscription.subscription_type == SubscriptionType.ONE_TIME:
                subscription.expires_at = datetime.now()

            return DeliveryResult(
                event_id=event.event_id,
                subscription_id=subscription.subscription_id,
                status=EventStatus.DELIVERED,
                delivered_at=datetime.now()
            )

        except Exception as e:
            logger.error(f"Event delivery failed: {e}")
            return DeliveryResult(
                event_id=event.event_id,
                subscription_id=subscription.subscription_id,
                status=EventStatus.FAILED,
                error_message=str(e)
            )


# ============================================================
# 重试处理器
# ============================================================

class RetryHandler:
    """重试处理器"""

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_backoff: bool = True
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_backoff = exponential_backoff
        self._retry_queue: asyncio.Queue = asyncio.Queue()
        self._running = False

    async def start(self):
        """启动重试处理器"""
        self._running = True
        asyncio.create_task(self._process_retries())

    async def stop(self):
        """停止重试处理器"""
        self._running = False

    async def schedule_retry(
        self,
        event: Event,
        subscription: Subscription,
        retry_count: int
    ):
        """调度重试"""
        if retry_count >= self.max_retries:
            logger.warning(f"Max retries reached for event {event.event_id}")
            return

        delay = self._calculate_delay(retry_count)
        await self._retry_queue.put((event, subscription, retry_count, delay))

    def _calculate_delay(self, retry_count: int) -> float:
        """计算重试延迟"""
        if self.exponential_backoff:
            delay = self.base_delay * (2 ** retry_count)
        else:
            delay = self.base_delay
        return min(delay, self.max_delay)

    async def _process_retries(self):
        """处理重试队列"""
        while self._running:
            try:
                event, subscription, retry_count, delay = await asyncio.wait_for(
                    self._retry_queue.get(),
                    timeout=1.0
                )
                await asyncio.sleep(delay)

                # 重新投递
                handler = EventHandler(subscription.handler, subscription.is_async)
                try:
                    await handler.handle(event)
                    logger.info(f"Retry successful for event {event.event_id}")
                except Exception as e:
                    logger.warning(f"Retry failed: {e}")
                    await self.schedule_retry(event, subscription, retry_count + 1)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Retry processor error: {e}")


# ============================================================
# 死信队列
# ============================================================

class DeadLetterQueue:
    """死信队列"""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.queue: List[Tuple[Event, str, datetime]] = []
        self._lock = asyncio.Lock()

    async def add(self, event: Event, reason: str):
        """添加到死信队列"""
        async with self._lock:
            if len(self.queue) >= self.max_size:
                self.queue.pop(0)

            self.queue.append((event, reason, datetime.now()))
            logger.warning(f"Event {event.event_id} added to DLQ: {reason}")

    async def get_all(self) -> List[Tuple[Event, str, datetime]]:
        """获取所有死信"""
        return self.queue.copy()

    async def replay(
        self,
        event_bus: 'EventBus',
        event_id: Optional[str] = None
    ) -> int:
        """重放死信"""
        replayed = 0
        async with self._lock:
            remaining = []
            for event, reason, added_at in self.queue:
                if event_id and event.event_id != event_id:
                    remaining.append((event, reason, added_at))
                    continue

                # 重新发布
                await event_bus.publish(event)
                replayed += 1

            self.queue = remaining

        return replayed

    async def clear(self):
        """清空死信队列"""
        async with self._lock:
            self.queue.clear()


# ============================================================
# 事件总线
# ============================================================

class EventBus:
    """事件总线"""

    def __init__(
        self,
        delivery_mode: DeliveryMode = DeliveryMode.BROADCAST,
        enable_persistence: bool = False,
        enable_retry: bool = True
    ):
        self.delivery_mode = delivery_mode
        self.enable_persistence = enable_persistence
        self.enable_retry = enable_retry

        # 组件
        self.subscription_manager = SubscriptionManager()
        self.dispatcher = EventDispatcher(delivery_mode)
        self.store = InMemoryEventStore() if enable_persistence else None
        self.retry_handler = RetryHandler() if enable_retry else None
        self.dead_letter_queue = DeadLetterQueue()

        # 拦截器
        self.interceptors: List[EventInterceptor] = []

        # 统计
        self._stats = EventBusStats()
        self._stats_lock = asyncio.Lock()

        # 运行状态
        self._running = False

    async def start(self):
        """启动事件总线"""
        self._running = True

        if self.retry_handler:
            await self.retry_handler.start()

        logger.info("Event bus started")

    async def stop(self):
        """停止事件总线"""
        self._running = False

        if self.retry_handler:
            await self.retry_handler.stop()

        logger.info("Event bus stopped")

    def add_interceptor(self, interceptor: EventInterceptor):
        """添加拦截器"""
        self.interceptors.append(interceptor)

    async def publish(
        self,
        topic: str,
        payload: Any,
        source: str = "unknown",
        priority: EventPriority = EventPriority.NORMAL,
        correlation_id: Optional[str] = None,
        reply_to: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Event:
        """发布事件"""
        # 创建事件
        event = Event(
            event_id=str(uuid.uuid4()),
            topic=topic,
            payload=payload,
            source=source,
            priority=priority,
            correlation_id=correlation_id,
            reply_to=reply_to,
            ttl_seconds=ttl_seconds,
            metadata=metadata or {}
        )

        return await self._publish_event(event)

    async def _publish_event(self, event: Event) -> Event:
        """内部发布事件"""
        # 检查过期
        if event.is_expired:
            await self.dead_letter_queue.add(event, "Event expired before publish")
            async with self._stats_lock:
                self._stats.total_events_expired += 1
            return event

        # 执行前置拦截器
        for interceptor in self.interceptors:
            result = await interceptor.before_publish(event)
            if result is None:
                return event
            event = result

        # 持久化
        if self.store:
            await self.store.save(event)

        # 获取匹配的订阅
        subscriptions = await self.subscription_manager.get_matching(event)

        # 分发事件
        results = await self.dispatcher.dispatch(event, subscriptions)

        # 处理失败的投递
        for result in results:
            if result.status == EventStatus.FAILED:
                if self.retry_handler:
                    subscription = self.subscription_manager.subscriptions.get(
                        result.subscription_id
                    )
                    if subscription:
                        await self.retry_handler.schedule_retry(
                            event, subscription, result.retry_count
                        )
                else:
                    await self.dead_letter_queue.add(
                        event, f"Delivery failed: {result.error_message}"
                    )

        # 执行后置拦截器
        for interceptor in self.interceptors:
            await interceptor.after_publish(event, results)

        # 更新统计
        async with self._stats_lock:
            self._stats.total_events_published += 1
            self._stats.total_events_delivered += sum(
                1 for r in results if r.status == EventStatus.DELIVERED
            )
            self._stats.total_events_failed += sum(
                1 for r in results if r.status == EventStatus.FAILED
            )
            self._stats.events_per_topic[event.topic] = \
                self._stats.events_per_topic.get(event.topic, 0) + 1

        return event

    async def subscribe(
        self,
        topic_pattern: str,
        handler: Callable,
        subscriber_id: str = None,
        subscription_type: SubscriptionType = SubscriptionType.PERMANENT,
        priority_filter: Optional[Set[EventPriority]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        is_async: bool = False
    ) -> str:
        """订阅主题"""
        subscription = Subscription(
            subscription_id=str(uuid.uuid4()),
            topic_pattern=topic_pattern,
            handler=handler,
            subscriber_id=subscriber_id or str(uuid.uuid4()),
            subscription_type=subscription_type,
            priority_filter=priority_filter,
            metadata_filter=metadata_filter,
            is_async=is_async
        )

        sub_id = await self.subscription_manager.add(subscription)

        async with self._stats_lock:
            self._stats.total_subscriptions += 1
            self._stats.active_subscriptions += 1

        logger.info(f"New subscription: {sub_id} for pattern: {topic_pattern}")
        return sub_id

    async def unsubscribe(self, subscription_id: str) -> bool:
        """取消订阅"""
        result = await self.subscription_manager.remove(subscription_id)

        if result:
            async with self._stats_lock:
                self._stats.active_subscriptions -= 1
            logger.info(f"Subscription removed: {subscription_id}")

        return result

    async def request(
        self,
        topic: str,
        payload: Any,
        source: str = "unknown",
        timeout: float = 30.0
    ) -> Optional[Any]:
        """请求-响应模式"""
        reply_topic = f"_reply/{uuid.uuid4()}"
        response_future: asyncio.Future = asyncio.Future()

        # 订阅响应主题
        async def response_handler(event: Event):
            if not response_future.done():
                response_future.set_result(event.payload)

        sub_id = await self.subscribe(
            reply_topic,
            response_handler,
            subscription_type=SubscriptionType.ONE_TIME,
            is_async=True
        )

        try:
            # 发布请求
            await self.publish(
                topic=topic,
                payload=payload,
                source=source,
                reply_to=reply_topic
            )

            # 等待响应
            response = await asyncio.wait_for(response_future, timeout=timeout)
            return response

        except asyncio.TimeoutError:
            logger.warning(f"Request to {topic} timed out")
            return None

        finally:
            await self.unsubscribe(sub_id)

    async def reply(self, original_event: Event, payload: Any):
        """回复事件"""
        if not original_event.reply_to:
            raise ValueError("Original event has no reply_to topic")

        await self.publish(
            topic=original_event.reply_to,
            payload=payload,
            source="reply",
            correlation_id=original_event.correlation_id
        )

    async def get_stats(self) -> EventBusStats:
        """获取统计信息"""
        async with self._stats_lock:
            sub_stats = await self.subscription_manager.get_stats()
            self._stats.active_subscriptions = sub_stats['active']
            return self._stats


# ============================================================
# 通道
# ============================================================

class Channel:
    """事件通道 - 命名空间隔离"""

    def __init__(self, name: str, event_bus: EventBus):
        self.name = name
        self.event_bus = event_bus
        self._prefix = f"channel/{name}/"

    async def publish(
        self,
        topic: str,
        payload: Any,
        **kwargs
    ) -> Event:
        """在通道内发布事件"""
        full_topic = f"{self._prefix}{topic}"
        return await self.event_bus.publish(full_topic, payload, **kwargs)

    async def subscribe(
        self,
        topic_pattern: str,
        handler: Callable,
        **kwargs
    ) -> str:
        """在通道内订阅"""
        full_pattern = f"{self._prefix}{topic_pattern}"
        return await self.event_bus.subscribe(full_pattern, handler, **kwargs)


# ============================================================
# 事件聚合器
# ============================================================

class EventAggregator:
    """事件聚合器"""

    def __init__(
        self,
        event_bus: EventBus,
        aggregation_topic: str,
        window_seconds: float = 10.0,
        min_events: int = 1
    ):
        self.event_bus = event_bus
        self.aggregation_topic = aggregation_topic
        self.window_seconds = window_seconds
        self.min_events = min_events

        self._buffer: List[Event] = []
        self._last_flush = time.time()
        self._lock = asyncio.Lock()
        self._running = False

    async def start(self):
        """启动聚合器"""
        self._running = True
        asyncio.create_task(self._flush_loop())

    async def stop(self):
        """停止聚合器"""
        self._running = False
        await self._flush()

    async def add(self, event: Event):
        """添加事件到聚合缓冲"""
        async with self._lock:
            self._buffer.append(event)

    async def _flush_loop(self):
        """定时刷新循环"""
        while self._running:
            await asyncio.sleep(self.window_seconds)
            await self._flush()

    async def _flush(self):
        """刷新聚合的事件"""
        async with self._lock:
            if len(self._buffer) < self.min_events:
                return

            events = self._buffer.copy()
            self._buffer.clear()

        # 发布聚合事件
        await self.event_bus.publish(
            topic=self.aggregation_topic,
            payload={
                'events': [e.to_dict() for e in events],
                'count': len(events),
                'window_start': self._last_flush,
                'window_end': time.time()
            },
            source="aggregator"
        )

        self._last_flush = time.time()


# ============================================================
# 工厂函数
# ============================================================

def create_cyrp_event_bus(
    delivery_mode: DeliveryMode = DeliveryMode.BROADCAST,
    enable_persistence: bool = True,
    enable_retry: bool = True,
    enable_logging: bool = True,
    enable_metrics: bool = True
) -> EventBus:
    """创建CYRP事件总线实例

    Args:
        delivery_mode: 投递模式
        enable_persistence: 是否启用持久化
        enable_retry: 是否启用重试
        enable_logging: 是否启用日志
        enable_metrics: 是否启用指标

    Returns:
        EventBus: 配置好的事件总线实例
    """
    event_bus = EventBus(
        delivery_mode=delivery_mode,
        enable_persistence=enable_persistence,
        enable_retry=enable_retry
    )

    # 添加拦截器
    if enable_logging:
        event_bus.add_interceptor(LoggingInterceptor())

    if enable_metrics:
        event_bus.add_interceptor(MetricsInterceptor())

    return event_bus


# ============================================================
# 示例用法
# ============================================================

async def example_usage():
    """示例用法"""
    # 创建事件总线
    event_bus = create_cyrp_event_bus()
    await event_bus.start()

    # 定义处理器
    async def handle_sensor_data(event: Event):
        print(f"Received sensor data: {event.payload}")

    async def handle_alarm(event: Event):
        print(f"Alarm triggered: {event.payload}")
        # 如果有reply_to，发送响应
        if event.reply_to:
            await event_bus.reply(event, {'acknowledged': True})

    # 订阅
    await event_bus.subscribe(
        "sensor/#",
        handle_sensor_data,
        is_async=True
    )

    await event_bus.subscribe(
        "alarm/*",
        handle_alarm,
        priority_filter={EventPriority.HIGH, EventPriority.CRITICAL},
        is_async=True
    )

    # 发布事件
    await event_bus.publish(
        topic="sensor/temperature/zone1",
        payload={'value': 25.5, 'unit': 'celsius'},
        source="sensor-gateway"
    )

    await event_bus.publish(
        topic="alarm/high_temp",
        payload={'message': 'Temperature too high', 'zone': 'zone1'},
        source="alarm-system",
        priority=EventPriority.HIGH
    )

    # 请求-响应
    response = await event_bus.request(
        topic="alarm/confirm",
        payload={'alarm_id': 'ALM001'},
        timeout=10.0
    )
    print(f"Response: {response}")

    # 获取统计
    stats = await event_bus.get_stats()
    print(f"Stats: {stats}")

    await event_bus.stop()


if __name__ == "__main__":
    asyncio.run(example_usage())
