"""
Multi-Agent System Integration for CYRP.
穿黄工程多智能体系统集成
"""

from typing import Dict, Any, List, Optional
import time

from cyrp.agents.base_agent import Agent, AgentRole, AgentState
from cyrp.agents.coordinator_agent import CoordinatorAgent
from cyrp.agents.perception_agent import PerceptionAgent
from cyrp.agents.control_agent import ControlAgent
from cyrp.agents.safety_agent import SafetyAgent
from cyrp.agents.scenario_agent import ScenarioAgent


class MultiAgentSystem:
    """
    多智能体系统

    整合所有智能体，实现穿黄工程的全场景自主运行
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化多智能体系统

        Args:
            config: 系统配置
        """
        self.config = config or {}

        # 创建智能体
        self._create_agents()

        # 运行状态
        self.running = False
        self.cycle_count = 0
        self.start_time = 0.0

        # 性能统计
        self.stats = {
            'total_cycles': 0,
            'avg_cycle_time': 0.0,
            'max_cycle_time': 0.0,
            'messages_total': 0
        }

    def _create_agents(self):
        """创建所有智能体"""
        # 协调智能体 (总调度)
        self.coordinator = CoordinatorAgent("Coordinator")

        # 感知智能体
        self.perception_agent = PerceptionAgent("Perception")

        # 控制智能体
        self.control_agent = ControlAgent("Control")

        # 安全智能体
        self.safety_agent = SafetyAgent("Safety")

        # 场景智能体
        self.scenario_agent = ScenarioAgent("Scenario")

        # 注册到协调器
        self.coordinator.register_agent(self.perception_agent)
        self.coordinator.register_agent(self.control_agent)
        self.coordinator.register_agent(self.safety_agent)
        self.coordinator.register_agent(self.scenario_agent)

        # 智能体列表
        self.agents: Dict[str, Agent] = {
            'coordinator': self.coordinator,
            'perception': self.perception_agent,
            'control': self.control_agent,
            'safety': self.safety_agent,
            'scenario': self.scenario_agent
        }

    def step(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行单步更新

        Args:
            environment: 环境信息（物理系统状态）

        Returns:
            步进结果
        """
        cycle_start = time.time()

        # 运行协调周期
        result = self.coordinator.run_cycle(environment)

        # 更新统计
        cycle_time = time.time() - cycle_start
        self.cycle_count += 1
        self.stats['total_cycles'] = self.cycle_count
        self.stats['avg_cycle_time'] = (
            (self.stats['avg_cycle_time'] * (self.cycle_count - 1) + cycle_time)
            / self.cycle_count
        )
        self.stats['max_cycle_time'] = max(self.stats['max_cycle_time'], cycle_time)

        return result

    def run(
        self,
        environment_generator,
        duration: float,
        dt: float = 0.1,
        callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        运行仿真

        Args:
            environment_generator: 环境生成器函数
            duration: 仿真时长 (s)
            dt: 时间步长 (s)
            callback: 每步回调函数

        Returns:
            仿真结果列表
        """
        self.running = True
        self.start_time = time.time()
        results = []

        t = 0.0
        while t < duration and self.running:
            # 获取环境
            environment = environment_generator(t)
            environment['time'] = t
            environment['dt'] = dt

            # 执行步进
            result = self.step(environment)
            result['time'] = t
            results.append(result)

            # 回调
            if callback:
                callback(t, result)

            t += dt

        self.running = False
        return results

    def stop(self):
        """停止运行"""
        self.running = False

    def get_control_output(self) -> Dict[str, float]:
        """获取当前控制输出"""
        control = self.control_agent.last_control
        return {
            'gate_1': float(control[0]),
            'gate_2': float(control[1])
        }

    def get_scenario(self) -> str:
        """获取当前场景"""
        return self.scenario_agent.get_current_scenario().value

    def get_risk_level(self) -> int:
        """获取当前风险等级"""
        return self.safety_agent.risk_level

    def is_safe(self) -> bool:
        """检查系统是否安全"""
        return self.safety_agent.is_safe()

    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'running': self.running,
            'cycle_count': self.cycle_count,
            'stats': self.stats,
            'agents': self.coordinator.get_system_status(),
            'scenario': self.get_scenario(),
            'risk_level': self.get_risk_level(),
            'control': self.get_control_output()
        }

    def set_reference(self, Q_total: float):
        """设置流量参考值"""
        self.control_agent.set_reference(Q_total)

    def trigger_emergency(self, scenario_type) -> bool:
        """触发应急场景"""
        return self.scenario_agent.trigger_emergency(scenario_type)

    def reset(self):
        """重置系统"""
        for agent in self.agents.values():
            if hasattr(agent, 'reset'):
                agent.reset()
            agent.state = AgentState.IDLE

        self.cycle_count = 0
        self.stats = {
            'total_cycles': 0,
            'avg_cycle_time': 0.0,
            'max_cycle_time': 0.0,
            'messages_total': 0
        }

    def shutdown(self):
        """关闭系统"""
        self.running = False
        for agent in self.agents.values():
            agent.shutdown()
