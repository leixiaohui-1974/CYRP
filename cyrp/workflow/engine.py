"""
Workflow Engine for CYRP
穿黄工程工作流引擎

功能:
- 可视化工作流设计
- 条件分支与循环
- 并行/串行任务执行
- 事件触发与定时触发
- 审批流程支持
- 工作流监控与日志
"""

import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import re


class NodeType(Enum):
    """节点类型"""
    START = auto()           # 开始节点
    END = auto()             # 结束节点
    TASK = auto()            # 任务节点
    DECISION = auto()        # 决策节点(条件分支)
    PARALLEL_GATEWAY = auto()  # 并行网关
    JOIN_GATEWAY = auto()    # 汇合网关
    SUBPROCESS = auto()      # 子流程
    TIMER = auto()           # 定时器节点
    EVENT = auto()           # 事件节点
    APPROVAL = auto()        # 审批节点
    SCRIPT = auto()          # 脚本节点
    SERVICE = auto()         # 服务调用节点


class WorkflowStatus(Enum):
    """工作流状态"""
    DRAFT = auto()
    ACTIVE = auto()
    SUSPENDED = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class TaskStatus(Enum):
    """任务状态"""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()
    WAITING = auto()        # 等待(如审批)
    TIMEOUT = auto()


class TriggerType(Enum):
    """触发器类型"""
    MANUAL = auto()         # 手动触发
    SCHEDULE = auto()       # 定时触发
    EVENT = auto()          # 事件触发
    API = auto()            # API触发
    CONDITION = auto()      # 条件触发


@dataclass
class WorkflowVariable:
    """工作流变量"""
    name: str
    value: Any
    var_type: str = "string"  # string, number, boolean, object, list
    scope: str = "workflow"   # workflow, task, global


@dataclass
class Connection:
    """节点连接"""
    connection_id: str
    source_node_id: str
    target_node_id: str
    condition: Optional[str] = None  # 条件表达式
    label: str = ""


@dataclass
class WorkflowNode:
    """工作流节点"""
    node_id: str
    node_type: NodeType
    name: str
    description: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    position: Tuple[int, int] = (0, 0)
    timeout_seconds: int = 0
    retry_count: int = 0
    retry_delay_seconds: int = 60
    on_error: str = "fail"  # fail, skip, retry


@dataclass
class TaskExecution:
    """任务执行记录"""
    execution_id: str
    node_id: str
    node_name: str
    status: TaskStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    retry_count: int = 0


@dataclass
class WorkflowDefinition:
    """工作流定义"""
    workflow_id: str
    name: str
    description: str = ""
    version: int = 1
    nodes: List[WorkflowNode] = field(default_factory=list)
    connections: List[Connection] = field(default_factory=list)
    variables: List[WorkflowVariable] = field(default_factory=list)
    triggers: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    is_active: bool = True
    tags: List[str] = field(default_factory=list)


@dataclass
class WorkflowInstance:
    """工作流实例"""
    instance_id: str
    workflow_id: str
    workflow_name: str
    status: WorkflowStatus
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    current_nodes: List[str] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    task_executions: List[TaskExecution] = field(default_factory=list)
    triggered_by: str = ""
    error_message: str = ""
    parent_instance_id: Optional[str] = None


class TaskHandler(ABC):
    """任务处理器基类"""

    @abstractmethod
    async def execute(
        self,
        node: WorkflowNode,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行任务"""
        pass


class ScriptTaskHandler(TaskHandler):
    """脚本任务处理器"""

    async def execute(
        self,
        node: WorkflowNode,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行Python脚本"""
        script = node.config.get("script", "")
        if not script:
            return {}

        # 创建安全的执行环境
        local_vars = {"context": context, "result": {}}

        try:
            exec(script, {"__builtins__": {}}, local_vars)
            return local_vars.get("result", {})
        except Exception as e:
            raise RuntimeError(f"脚本执行失败: {e}")


class ServiceTaskHandler(TaskHandler):
    """服务调用任务处理器"""

    def __init__(self, services: Dict[str, Callable] = None):
        self.services = services or {}

    def register_service(self, name: str, handler: Callable):
        """注册服务"""
        self.services[name] = handler

    async def execute(
        self,
        node: WorkflowNode,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """调用服务"""
        service_name = node.config.get("service")
        if not service_name or service_name not in self.services:
            raise ValueError(f"服务不存在: {service_name}")

        handler = self.services[service_name]
        params = node.config.get("params", {})

        # 替换参数中的变量引用
        resolved_params = self._resolve_params(params, context)

        if asyncio.iscoroutinefunction(handler):
            result = await handler(**resolved_params)
        else:
            result = handler(**resolved_params)

        return {"result": result}

    def _resolve_params(
        self,
        params: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """解析参数中的变量引用"""
        resolved = {}
        for key, value in params.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                var_path = value[2:-1]
                resolved[key] = self._get_nested_value(context, var_path)
            else:
                resolved[key] = value
        return resolved

    def _get_nested_value(self, data: Dict, path: str) -> Any:
        """获取嵌套值"""
        keys = path.split(".")
        value = data
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return None
        return value


class ApprovalTaskHandler(TaskHandler):
    """审批任务处理器"""

    def __init__(self):
        self._pending_approvals: Dict[str, Dict] = {}
        self._approval_results: Dict[str, Dict] = {}

    async def execute(
        self,
        node: WorkflowNode,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """创建审批任务"""
        approval_id = str(uuid.uuid4())
        approvers = node.config.get("approvers", [])
        approval_type = node.config.get("approval_type", "any")  # any, all

        self._pending_approvals[approval_id] = {
            "node_id": node.node_id,
            "approvers": approvers,
            "approval_type": approval_type,
            "decisions": {},
            "created_at": datetime.now(),
            "context": context
        }

        # 等待审批结果
        while approval_id not in self._approval_results:
            await asyncio.sleep(1)

        result = self._approval_results.pop(approval_id)
        del self._pending_approvals[approval_id]

        return result

    def submit_approval(
        self,
        approval_id: str,
        approver: str,
        approved: bool,
        comment: str = ""
    ) -> bool:
        """提交审批决定"""
        if approval_id not in self._pending_approvals:
            return False

        approval = self._pending_approvals[approval_id]
        approval["decisions"][approver] = {
            "approved": approved,
            "comment": comment,
            "time": datetime.now()
        }

        # 检查是否可以结束审批
        approval_type = approval["approval_type"]
        decisions = approval["decisions"]
        approvers = approval["approvers"]

        if approval_type == "any":
            # 任一审批人通过即可
            if any(d["approved"] for d in decisions.values()):
                self._approval_results[approval_id] = {
                    "approved": True,
                    "decisions": decisions
                }
                return True
            elif len(decisions) == len(approvers):
                # 所有人都拒绝
                self._approval_results[approval_id] = {
                    "approved": False,
                    "decisions": decisions
                }
                return True
        elif approval_type == "all":
            # 需要所有人通过
            if len(decisions) == len(approvers):
                all_approved = all(d["approved"] for d in decisions.values())
                self._approval_results[approval_id] = {
                    "approved": all_approved,
                    "decisions": decisions
                }
                return True

        return False


class ConditionEvaluator:
    """条件表达式求值器"""

    @staticmethod
    def evaluate(expression: str, context: Dict[str, Any]) -> bool:
        """评估条件表达式"""
        if not expression:
            return True

        # 替换变量引用
        def replace_var(match):
            var_name = match.group(1)
            value = context.get(var_name)
            if isinstance(value, str):
                return f'"{value}"'
            elif value is None:
                return "None"
            else:
                return str(value)

        expr = re.sub(r'\$\{(\w+)\}', replace_var, expression)

        # 安全评估
        try:
            # 只允许基本操作
            allowed_names = {
                "True": True,
                "False": False,
                "None": None,
                "and": None,
                "or": None,
                "not": None,
            }
            return eval(expr, {"__builtins__": {}}, {**allowed_names, **context})
        except Exception:
            return False


class WorkflowExecutor:
    """工作流执行器"""

    def __init__(self):
        self.task_handlers: Dict[NodeType, TaskHandler] = {}
        self._instances: Dict[str, WorkflowInstance] = {}
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._event_handlers: Dict[str, List[Callable]] = {}

        # 注册默认处理器
        self.register_handler(NodeType.SCRIPT, ScriptTaskHandler())
        self.register_handler(NodeType.SERVICE, ServiceTaskHandler())
        self.register_handler(NodeType.APPROVAL, ApprovalTaskHandler())

    def register_handler(self, node_type: NodeType, handler: TaskHandler):
        """注册任务处理器"""
        self.task_handlers[node_type] = handler

    def add_event_handler(self, event_type: str, handler: Callable):
        """添加事件处理器"""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)

    async def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """发送事件"""
        handlers = self._event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception:
                pass

    async def start_workflow(
        self,
        definition: WorkflowDefinition,
        input_data: Optional[Dict[str, Any]] = None,
        triggered_by: str = "system"
    ) -> WorkflowInstance:
        """启动工作流"""
        instance_id = str(uuid.uuid4())

        # 初始化变量
        variables = {v.name: v.value for v in definition.variables}
        if input_data:
            variables.update(input_data)

        instance = WorkflowInstance(
            instance_id=instance_id,
            workflow_id=definition.workflow_id,
            workflow_name=definition.name,
            status=WorkflowStatus.ACTIVE,
            variables=variables,
            triggered_by=triggered_by
        )

        self._instances[instance_id] = instance

        # 找到开始节点
        start_nodes = [n for n in definition.nodes if n.node_type == NodeType.START]
        if not start_nodes:
            instance.status = WorkflowStatus.FAILED
            instance.error_message = "未找到开始节点"
            return instance

        # 发送工作流启动事件
        await self._emit_event("workflow_started", {
            "instance_id": instance_id,
            "workflow_id": definition.workflow_id
        })

        # 开始执行
        asyncio.create_task(
            self._execute_workflow(instance, definition, start_nodes[0])
        )

        return instance

    async def _execute_workflow(
        self,
        instance: WorkflowInstance,
        definition: WorkflowDefinition,
        current_node: WorkflowNode
    ):
        """执行工作流"""
        try:
            await self._execute_node(instance, definition, current_node)
        except Exception as e:
            instance.status = WorkflowStatus.FAILED
            instance.error_message = str(e)
            await self._emit_event("workflow_failed", {
                "instance_id": instance.instance_id,
                "error": str(e)
            })

    async def _execute_node(
        self,
        instance: WorkflowInstance,
        definition: WorkflowDefinition,
        node: WorkflowNode
    ):
        """执行节点"""
        instance.current_nodes.append(node.node_id)

        # 创建任务执行记录
        execution = TaskExecution(
            execution_id=str(uuid.uuid4()),
            node_id=node.node_id,
            node_name=node.name,
            status=TaskStatus.RUNNING,
            started_at=datetime.now(),
            input_data=dict(instance.variables)
        )
        instance.task_executions.append(execution)

        await self._emit_event("node_started", {
            "instance_id": instance.instance_id,
            "node_id": node.node_id,
            "node_name": node.name
        })

        try:
            # 根据节点类型执行
            if node.node_type == NodeType.START:
                output = {}
            elif node.node_type == NodeType.END:
                instance.status = WorkflowStatus.COMPLETED
                instance.completed_at = datetime.now()
                execution.status = TaskStatus.COMPLETED
                execution.completed_at = datetime.now()

                await self._emit_event("workflow_completed", {
                    "instance_id": instance.instance_id
                })
                return
            elif node.node_type == NodeType.DECISION:
                output = await self._execute_decision(instance, definition, node)
            elif node.node_type == NodeType.PARALLEL_GATEWAY:
                output = await self._execute_parallel(instance, definition, node)
            elif node.node_type == NodeType.TIMER:
                output = await self._execute_timer(node)
            else:
                output = await self._execute_task(node, instance.variables)

            execution.status = TaskStatus.COMPLETED
            execution.completed_at = datetime.now()
            execution.output_data = output

            # 更新变量
            if output:
                instance.variables.update(output)

            # 找到下一个节点
            next_nodes = self._get_next_nodes(definition, node, instance.variables)

            instance.current_nodes.remove(node.node_id)

            # 执行下一个节点
            for next_node in next_nodes:
                await self._execute_node(instance, definition, next_node)

        except Exception as e:
            execution.status = TaskStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.now()

            if node.on_error == "skip":
                # 跳过错误,继续执行
                next_nodes = self._get_next_nodes(definition, node, instance.variables)
                instance.current_nodes.remove(node.node_id)
                for next_node in next_nodes:
                    await self._execute_node(instance, definition, next_node)
            elif node.on_error == "retry" and execution.retry_count < node.retry_count:
                # 重试
                execution.retry_count += 1
                await asyncio.sleep(node.retry_delay_seconds)
                await self._execute_node(instance, definition, node)
            else:
                raise

    async def _execute_task(
        self,
        node: WorkflowNode,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行任务节点"""
        handler = self.task_handlers.get(node.node_type)
        if not handler:
            return {}

        if node.timeout_seconds > 0:
            try:
                return await asyncio.wait_for(
                    handler.execute(node, context),
                    timeout=node.timeout_seconds
                )
            except asyncio.TimeoutError:
                raise TimeoutError(f"任务超时: {node.name}")
        else:
            return await handler.execute(node, context)

    async def _execute_decision(
        self,
        instance: WorkflowInstance,
        definition: WorkflowDefinition,
        node: WorkflowNode
    ) -> Dict[str, Any]:
        """执行决策节点"""
        # 决策逻辑在_get_next_nodes中处理
        return {}

    async def _execute_parallel(
        self,
        instance: WorkflowInstance,
        definition: WorkflowDefinition,
        node: WorkflowNode
    ) -> Dict[str, Any]:
        """执行并行网关"""
        # 找到所有并行分支
        connections = [c for c in definition.connections if c.source_node_id == node.node_id]
        next_node_ids = [c.target_node_id for c in connections]

        # 并行执行所有分支
        tasks = []
        for node_id in next_node_ids:
            next_node = next((n for n in definition.nodes if n.node_id == node_id), None)
            if next_node:
                tasks.append(self._execute_node(instance, definition, next_node))

        if tasks:
            await asyncio.gather(*tasks)

        return {}

    async def _execute_timer(self, node: WorkflowNode) -> Dict[str, Any]:
        """执行定时器节点"""
        delay_seconds = node.config.get("delay_seconds", 0)
        if delay_seconds > 0:
            await asyncio.sleep(delay_seconds)
        return {}

    def _get_next_nodes(
        self,
        definition: WorkflowDefinition,
        current_node: WorkflowNode,
        context: Dict[str, Any]
    ) -> List[WorkflowNode]:
        """获取下一个节点"""
        connections = [
            c for c in definition.connections
            if c.source_node_id == current_node.node_id
        ]

        next_nodes = []
        for conn in connections:
            # 评估条件
            if conn.condition:
                if not ConditionEvaluator.evaluate(conn.condition, context):
                    continue

            # 找到目标节点
            target_node = next(
                (n for n in definition.nodes if n.node_id == conn.target_node_id),
                None
            )
            if target_node:
                next_nodes.append(target_node)

        return next_nodes

    async def cancel_workflow(self, instance_id: str) -> bool:
        """取消工作流"""
        if instance_id not in self._instances:
            return False

        instance = self._instances[instance_id]
        instance.status = WorkflowStatus.CANCELLED
        instance.completed_at = datetime.now()

        # 取消正在运行的任务
        if instance_id in self._running_tasks:
            self._running_tasks[instance_id].cancel()

        await self._emit_event("workflow_cancelled", {
            "instance_id": instance_id
        })

        return True

    async def suspend_workflow(self, instance_id: str) -> bool:
        """暂停工作流"""
        if instance_id not in self._instances:
            return False

        instance = self._instances[instance_id]
        instance.status = WorkflowStatus.SUSPENDED

        await self._emit_event("workflow_suspended", {
            "instance_id": instance_id
        })

        return True

    async def resume_workflow(self, instance_id: str) -> bool:
        """恢复工作流"""
        if instance_id not in self._instances:
            return False

        instance = self._instances[instance_id]
        if instance.status != WorkflowStatus.SUSPENDED:
            return False

        instance.status = WorkflowStatus.ACTIVE

        await self._emit_event("workflow_resumed", {
            "instance_id": instance_id
        })

        return True

    def get_instance(self, instance_id: str) -> Optional[WorkflowInstance]:
        """获取工作流实例"""
        return self._instances.get(instance_id)

    def list_instances(
        self,
        workflow_id: Optional[str] = None,
        status: Optional[WorkflowStatus] = None
    ) -> List[WorkflowInstance]:
        """列出工作流实例"""
        instances = list(self._instances.values())

        if workflow_id:
            instances = [i for i in instances if i.workflow_id == workflow_id]
        if status:
            instances = [i for i in instances if i.status == status]

        return instances


class WorkflowManager:
    """工作流管理器"""

    def __init__(self):
        self.definitions: Dict[str, WorkflowDefinition] = {}
        self.executor = WorkflowExecutor()
        self._triggers: Dict[str, Dict] = {}
        self._scheduler_task: Optional[asyncio.Task] = None
        self._running = False

    def register_workflow(self, definition: WorkflowDefinition):
        """注册工作流"""
        self.definitions[definition.workflow_id] = definition

        # 注册触发器
        for trigger in definition.triggers:
            trigger_id = f"{definition.workflow_id}_{trigger.get('type', 'manual')}"
            self._triggers[trigger_id] = {
                "workflow_id": definition.workflow_id,
                **trigger
            }

    def unregister_workflow(self, workflow_id: str):
        """注销工作流"""
        if workflow_id in self.definitions:
            del self.definitions[workflow_id]

        # 移除触发器
        to_remove = [
            tid for tid, t in self._triggers.items()
            if t["workflow_id"] == workflow_id
        ]
        for tid in to_remove:
            del self._triggers[tid]

    async def start(self):
        """启动工作流管理器"""
        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())

    async def stop(self):
        """停止工作流管理器"""
        self._running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass

    async def _scheduler_loop(self):
        """调度循环(处理定时触发)"""
        while self._running:
            now = datetime.now()

            for trigger_id, trigger in self._triggers.items():
                if trigger.get("type") != "schedule":
                    continue

                # 检查是否应该执行
                # 简化实现:使用interval_seconds
                interval = trigger.get("interval_seconds", 0)
                last_run = trigger.get("last_run")

                if interval > 0:
                    if last_run is None or (now - last_run).total_seconds() >= interval:
                        workflow_id = trigger["workflow_id"]
                        if workflow_id in self.definitions:
                            await self.start_workflow(workflow_id, triggered_by="scheduler")
                            trigger["last_run"] = now

            await asyncio.sleep(10)  # 每10秒检查一次

    async def start_workflow(
        self,
        workflow_id: str,
        input_data: Optional[Dict[str, Any]] = None,
        triggered_by: str = "manual"
    ) -> Optional[WorkflowInstance]:
        """启动工作流"""
        if workflow_id not in self.definitions:
            return None

        definition = self.definitions[workflow_id]
        return await self.executor.start_workflow(definition, input_data, triggered_by)

    async def trigger_event(self, event_name: str, event_data: Dict[str, Any]):
        """触发事件"""
        for trigger_id, trigger in self._triggers.items():
            if trigger.get("type") != "event":
                continue

            if trigger.get("event_name") == event_name:
                workflow_id = trigger["workflow_id"]
                if workflow_id in self.definitions:
                    await self.start_workflow(
                        workflow_id,
                        input_data=event_data,
                        triggered_by=f"event:{event_name}"
                    )

    def register_service(self, name: str, handler: Callable):
        """注册服务(供工作流调用)"""
        service_handler = self.executor.task_handlers.get(NodeType.SERVICE)
        if isinstance(service_handler, ServiceTaskHandler):
            service_handler.register_service(name, handler)


def create_cyrp_workflow_system() -> WorkflowManager:
    """创建穿黄工程工作流系统"""
    manager = WorkflowManager()

    # 注册常用服务
    async def send_notification(message: str, recipients: List[str]):
        print(f"[通知] 发送给 {recipients}: {message}")
        return {"sent": True}

    async def update_setpoint(tag: str, value: float):
        print(f"[设定值] {tag} = {value}")
        return {"updated": True}

    async def log_event(event_type: str, message: str):
        print(f"[事件] [{event_type}] {message}")
        return {"logged": True}

    manager.register_service("send_notification", send_notification)
    manager.register_service("update_setpoint", update_setpoint)
    manager.register_service("log_event", log_event)

    # 创建示例工作流:报警处理流程
    alarm_workflow = WorkflowDefinition(
        workflow_id="alarm_handling",
        name="报警处理流程",
        description="当发生报警时的标准处理流程",
        nodes=[
            WorkflowNode(
                node_id="start",
                node_type=NodeType.START,
                name="开始"
            ),
            WorkflowNode(
                node_id="check_severity",
                node_type=NodeType.DECISION,
                name="检查报警级别",
                config={"condition_field": "severity"}
            ),
            WorkflowNode(
                node_id="notify_operator",
                node_type=NodeType.SERVICE,
                name="通知操作员",
                config={
                    "service": "send_notification",
                    "params": {
                        "message": "${alarm_message}",
                        "recipients": ["operator@cyrp.com"]
                    }
                }
            ),
            WorkflowNode(
                node_id="notify_manager",
                node_type=NodeType.SERVICE,
                name="通知管理层",
                config={
                    "service": "send_notification",
                    "params": {
                        "message": "${alarm_message}",
                        "recipients": ["manager@cyrp.com", "director@cyrp.com"]
                    }
                }
            ),
            WorkflowNode(
                node_id="log_alarm",
                node_type=NodeType.SERVICE,
                name="记录报警",
                config={
                    "service": "log_event",
                    "params": {
                        "event_type": "alarm",
                        "message": "${alarm_message}"
                    }
                }
            ),
            WorkflowNode(
                node_id="end",
                node_type=NodeType.END,
                name="结束"
            ),
        ],
        connections=[
            Connection("c1", "start", "check_severity"),
            Connection("c2", "check_severity", "notify_operator", condition="${severity} < 3"),
            Connection("c3", "check_severity", "notify_manager", condition="${severity} >= 3"),
            Connection("c4", "notify_operator", "log_alarm"),
            Connection("c5", "notify_manager", "log_alarm"),
            Connection("c6", "log_alarm", "end"),
        ],
        triggers=[
            {"type": "event", "event_name": "alarm_triggered"}
        ]
    )
    manager.register_workflow(alarm_workflow)

    return manager
