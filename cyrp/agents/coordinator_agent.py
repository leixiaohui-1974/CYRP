"""
Coordinator Agent for CYRP Multi-Agent System.
穿黄工程协调智能体 (总调度)
"""

from typing import Dict, Any, List, Optional
import time

from cyrp.agents.base_agent import Agent, AgentRole, AgentState, Message, MessageType


class CoordinatorAgent(Agent):
    """
    协调智能体 (总调度)

    负责:
    1. 多智能体协调调度
    2. 全局决策优化
    3. 资源分配
    4. 冲突解决
    5. 系统状态监控
    """

    def __init__(self, name: str = "CoordinatorAgent"):
        super().__init__(name, AgentRole.COORDINATOR, priority=5)

        # 管理的智能体
        self.managed_agents: Dict[str, Agent] = {}

        # 调度队列
        self.schedule_queue: List[str] = []

        # 系统状态
        self.system_status = {
            'mode': 'normal',
            'active_agents': 0,
            'messages_processed': 0,
            'last_cycle_time': 0.0
        }

        # 决策历史
        self.decision_history: List[Dict] = []

    def register_agent(self, agent: Agent):
        """注册智能体"""
        self.managed_agents[agent.name] = agent
        agent.connect(self.name)
        self.connect(agent.name)

    def unregister_agent(self, agent_name: str):
        """注销智能体"""
        if agent_name in self.managed_agents:
            agent = self.managed_agents.pop(agent_name)
            agent.disconnect(self.name)
            self.disconnect(agent_name)

    def perceive(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """感知系统全局状态"""
        # 收集所有智能体状态
        agent_statuses = {}
        for name, agent in self.managed_agents.items():
            agent_statuses[name] = agent.get_status()

        # 收集消息
        all_messages = []
        for agent in self.managed_agents.values():
            while agent.outbox:
                msg = agent.outbox.popleft()
                all_messages.append(msg)

        return {
            'environment': environment,
            'agent_statuses': agent_statuses,
            'messages': all_messages,
            'time': environment.get('time', 0.0)
        }

    def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """全局决策"""
        agent_statuses = perception['agent_statuses']
        messages = perception['messages']

        # 消息路由决策
        routed_messages = self._route_messages(messages)

        # 检查是否需要干预
        interventions = self._check_interventions(agent_statuses)

        # 调度决策
        schedule = self._decide_schedule(agent_statuses)

        # 冲突检测
        conflicts = self._detect_conflicts(messages)

        return {
            'routed_messages': routed_messages,
            'interventions': interventions,
            'schedule': schedule,
            'conflicts': conflicts
        }

    def _route_messages(self, messages: List[Message]) -> Dict[str, List[Message]]:
        """消息路由"""
        routed = {name: [] for name in self.managed_agents}

        for msg in messages:
            if msg.is_broadcast():
                # 广播给所有智能体
                for name in self.managed_agents:
                    if name != msg.sender:
                        routed[name].append(msg)
            elif msg.receiver in routed:
                routed[msg.receiver].append(msg)

        return routed

    def _check_interventions(self, statuses: Dict[str, Dict]) -> List[Dict]:
        """检查是否需要干预"""
        interventions = []

        for name, status in statuses.items():
            # 检查错误状态
            if status['state'] == AgentState.ERROR.value:
                interventions.append({
                    'agent': name,
                    'action': 'restart',
                    'reason': 'agent_error'
                })

            # 检查长时间未响应
            if status.get('last_update', 0) > 0:
                elapsed = time.time() - status['last_update']
                if elapsed > 60:  # 60秒未更新
                    interventions.append({
                        'agent': name,
                        'action': 'check',
                        'reason': 'no_response'
                    })

        return interventions

    def _decide_schedule(self, statuses: Dict[str, Dict]) -> List[str]:
        """决定调度顺序"""
        # 按优先级排序
        sorted_agents = sorted(
            statuses.items(),
            key=lambda x: x[1].get('priority', 1),
            reverse=True
        )

        return [name for name, _ in sorted_agents]

    def _detect_conflicts(self, messages: List[Message]) -> List[Dict]:
        """检测冲突"""
        conflicts = []

        # 检测控制指令冲突
        commands = [m for m in messages if m.msg_type == MessageType.COMMAND]
        gate_commands = {}

        for cmd in commands:
            if 'gate_1' in cmd.content:
                if 'gate_1' in gate_commands:
                    conflicts.append({
                        'type': 'control_conflict',
                        'target': 'gate_1',
                        'sources': [gate_commands['gate_1'], cmd.sender]
                    })
                gate_commands['gate_1'] = cmd.sender

        return conflicts

    def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """执行协调动作"""
        # 分发消息
        for agent_name, messages in decision['routed_messages'].items():
            if agent_name in self.managed_agents:
                for msg in messages:
                    self.managed_agents[agent_name].receive_message(msg)

        # 执行干预
        for intervention in decision['interventions']:
            self._execute_intervention(intervention)

        # 处理冲突
        for conflict in decision['conflicts']:
            self._resolve_conflict(conflict)

        # 更新系统状态
        self.system_status['active_agents'] = len([
            a for a in self.managed_agents.values()
            if a.state == AgentState.ACTIVE
        ])
        self.system_status['messages_processed'] = sum(
            len(msgs) for msgs in decision['routed_messages'].values()
        )
        self.system_status['last_cycle_time'] = time.time()

        return {
            'success': True,
            'messages_routed': self.system_status['messages_processed'],
            'interventions': len(decision['interventions']),
            'conflicts_resolved': len(decision['conflicts'])
        }

    def _execute_intervention(self, intervention: Dict):
        """执行干预"""
        agent_name = intervention['agent']
        action = intervention['action']

        if agent_name not in self.managed_agents:
            return

        agent = self.managed_agents[agent_name]

        if action == 'restart':
            agent.state = AgentState.IDLE
            # 发送重启通知
            self.broadcast({
                'event': 'agent_restarted',
                'agent': agent_name
            }, MessageType.STATUS)

        elif action == 'check':
            # 发送心跳检查
            self.request(agent_name, {'request': 'heartbeat'})

    def _resolve_conflict(self, conflict: Dict):
        """解决冲突"""
        if conflict['type'] == 'control_conflict':
            # 优先级解决：选择高优先级智能体的指令
            sources = conflict['sources']
            priorities = [
                self.managed_agents[s].priority if s in self.managed_agents else 0
                for s in sources
            ]
            winner = sources[priorities.index(max(priorities))]

            # 通知被覆盖的智能体
            for source in sources:
                if source != winner:
                    self.send_message(Message(
                        msg_type=MessageType.STATUS,
                        receiver=source,
                        content={
                            'event': 'command_overridden',
                            'target': conflict['target'],
                            'by': winner
                        }
                    ))

    def _handle_message(self, message: Message):
        """处理消息"""
        if message.msg_type == MessageType.REQUEST:
            if message.content.get('request') == 'get_system_status':
                self.respond(message, {'status': self.system_status})

            elif message.content.get('request') == 'get_agent_list':
                self.respond(message, {
                    'agents': list(self.managed_agents.keys())
                })

    def run_cycle(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行完整的协调周期

        Args:
            environment: 环境信息

        Returns:
            周期结果
        """
        cycle_start = time.time()

        # 1. 协调器感知
        perception = self.perceive(environment)

        # 2. 协调器决策
        decision = self.decide(perception)

        # 3. 协调器执行
        coord_result = self.act(decision)

        # 4. 按调度顺序运行各智能体
        agent_results = {}
        for agent_name in decision['schedule']:
            if agent_name in self.managed_agents:
                agent = self.managed_agents[agent_name]
                try:
                    result = agent.step(environment)
                    agent_results[agent_name] = result
                except Exception as e:
                    agent.state = AgentState.ERROR
                    agent_results[agent_name] = {'error': str(e)}

        cycle_time = time.time() - cycle_start

        return {
            'coordinator': coord_result,
            'agents': agent_results,
            'cycle_time': cycle_time
        }

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'coordinator': self.get_status(),
            'system': self.system_status,
            'agents': {
                name: agent.get_status()
                for name, agent in self.managed_agents.items()
            }
        }
