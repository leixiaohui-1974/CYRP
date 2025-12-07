"""
Base Agent Definition for CYRP Multi-Agent System.
穿黄工程多智能体系统基础智能体定义
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from abc import ABC, abstractmethod
from enum import Enum
import uuid
import time
from collections import deque


class AgentRole(Enum):
    """智能体角色"""
    COORDINATOR = "coordinator"  # 协调智能体 (总调度)
    PERCEPTION = "perception"  # 感知智能体
    CONTROL = "control"  # 控制智能体
    SAFETY = "safety"  # 安全智能体
    SCENARIO = "scenario"  # 场景智能体
    DIAGNOSTIC = "diagnostic"  # 诊断智能体
    OPTIMIZATION = "optimization"  # 优化智能体


class AgentState(Enum):
    """智能体状态"""
    IDLE = "idle"  # 空闲
    ACTIVE = "active"  # 活动
    PROCESSING = "processing"  # 处理中
    WAITING = "waiting"  # 等待
    ERROR = "error"  # 错误
    SHUTDOWN = "shutdown"  # 关闭


class MessageType(Enum):
    """消息类型"""
    OBSERVATION = "observation"  # 观测数据
    COMMAND = "command"  # 控制命令
    STATUS = "status"  # 状态报告
    ALERT = "alert"  # 告警
    REQUEST = "request"  # 请求
    RESPONSE = "response"  # 响应
    BROADCAST = "broadcast"  # 广播


@dataclass
class Message:
    """
    智能体间消息

    用于智能体间通信
    """
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    msg_type: MessageType = MessageType.STATUS
    sender: str = ""
    receiver: str = ""  # 空表示广播
    content: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # 1-5, 5最高
    timestamp: float = field(default_factory=time.time)
    reply_to: Optional[str] = None

    def is_broadcast(self) -> bool:
        return self.receiver == "" or self.receiver == "*"


class Agent(ABC):
    """
    智能体基类

    所有智能体的抽象基类，定义通用接口
    """

    def __init__(
        self,
        name: str,
        role: AgentRole,
        priority: int = 1
    ):
        """
        初始化智能体

        Args:
            name: 智能体名称
            role: 智能体角色
            priority: 优先级
        """
        self.name = name
        self.role = role
        self.priority = priority
        self.agent_id = str(uuid.uuid4())[:8]

        # 状态
        self.state = AgentState.IDLE
        self.last_update_time = 0.0

        # 消息队列
        self.inbox: deque = deque(maxlen=1000)
        self.outbox: deque = deque(maxlen=1000)

        # 知识库
        self.knowledge: Dict[str, Any] = {}

        # 连接的智能体
        self.connected_agents: List[str] = []

        # 回调函数
        self.on_message_received: Optional[Callable] = None
        self.on_state_changed: Optional[Callable] = None

    @abstractmethod
    def perceive(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        感知环境

        Args:
            environment: 环境信息

        Returns:
            感知结果
        """
        pass

    @abstractmethod
    def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """
        决策

        Args:
            perception: 感知结果

        Returns:
            决策结果
        """
        pass

    @abstractmethod
    def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行动作

        Args:
            decision: 决策结果

        Returns:
            执行结果
        """
        pass

    def step(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        单步执行 (感知-决策-执行循环)

        Args:
            environment: 环境信息

        Returns:
            执行结果
        """
        self.state = AgentState.PROCESSING

        # 处理收到的消息
        self._process_inbox()

        # 感知
        perception = self.perceive(environment)

        # 决策
        decision = self.decide(perception)

        # 执行
        result = self.act(decision)

        self.last_update_time = time.time()
        self.state = AgentState.ACTIVE

        return result

    def send_message(self, message: Message):
        """发送消息"""
        message.sender = self.name
        message.timestamp = time.time()
        self.outbox.append(message)

    def receive_message(self, message: Message):
        """接收消息"""
        if message.receiver == self.name or message.is_broadcast():
            self.inbox.append(message)
            if self.on_message_received:
                self.on_message_received(message)

    def _process_inbox(self):
        """处理收件箱消息"""
        while self.inbox:
            message = self.inbox.popleft()
            self._handle_message(message)

    def _handle_message(self, message: Message):
        """处理单条消息"""
        # 子类可以重写此方法
        pass

    def broadcast(self, content: Dict[str, Any], msg_type: MessageType = MessageType.BROADCAST):
        """广播消息"""
        message = Message(
            msg_type=msg_type,
            sender=self.name,
            receiver="*",
            content=content,
            priority=self.priority
        )
        self.send_message(message)

    def request(
        self,
        receiver: str,
        content: Dict[str, Any],
        priority: int = 3
    ) -> Message:
        """发送请求"""
        message = Message(
            msg_type=MessageType.REQUEST,
            sender=self.name,
            receiver=receiver,
            content=content,
            priority=priority
        )
        self.send_message(message)
        return message

    def respond(self, request: Message, content: Dict[str, Any]):
        """响应请求"""
        message = Message(
            msg_type=MessageType.RESPONSE,
            sender=self.name,
            receiver=request.sender,
            content=content,
            priority=request.priority,
            reply_to=request.msg_id
        )
        self.send_message(message)

    def alert(self, content: Dict[str, Any], priority: int = 5):
        """发送告警"""
        message = Message(
            msg_type=MessageType.ALERT,
            sender=self.name,
            receiver="*",
            content=content,
            priority=priority
        )
        self.send_message(message)

    def update_knowledge(self, key: str, value: Any):
        """更新知识库"""
        self.knowledge[key] = value

    def get_knowledge(self, key: str, default: Any = None) -> Any:
        """获取知识"""
        return self.knowledge.get(key, default)

    def connect(self, agent_name: str):
        """连接到其他智能体"""
        if agent_name not in self.connected_agents:
            self.connected_agents.append(agent_name)

    def disconnect(self, agent_name: str):
        """断开连接"""
        if agent_name in self.connected_agents:
            self.connected_agents.remove(agent_name)

    def shutdown(self):
        """关闭智能体"""
        self.state = AgentState.SHUTDOWN
        self.inbox.clear()
        self.outbox.clear()

    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        return {
            'name': self.name,
            'role': self.role.value,
            'state': self.state.value,
            'priority': self.priority,
            'inbox_size': len(self.inbox),
            'outbox_size': len(self.outbox),
            'connected_agents': self.connected_agents,
            'last_update': self.last_update_time
        }
