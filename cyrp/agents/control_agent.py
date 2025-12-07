"""
Control Agent for CYRP Multi-Agent System.
穿黄工程控制智能体
"""

from typing import Dict, Any, Optional
import numpy as np

from cyrp.agents.base_agent import Agent, AgentRole, Message, MessageType
from cyrp.control import HDMPCController
from cyrp.scenarios.scenario_definitions import ScenarioType


class ControlAgent(Agent):
    """
    控制智能体

    负责:
    1. 分层分布式MPC控制
    2. 控制指令生成
    3. 场景自适应控制切换
    4. 控制性能监测
    """

    def __init__(self, name: str = "ControlAgent"):
        super().__init__(name, AgentRole.CONTROL, priority=5)

        # HD-MPC控制器
        self.controller = HDMPCController()

        # 控制目标
        self.reference = {
            'Q_total': 265.0,
            'flow_balance': True,
            'pressure_stable': True
        }

        # 控制性能指标
        self.performance_metrics = {
            'tracking_error': 0.0,
            'control_effort': 0.0,
            'oscillation_index': 0.0
        }

        # 上一时刻控制
        self.last_control = np.array([1.0, 1.0])

    def perceive(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """感知环境和来自其他智能体的消息"""
        system_state = environment.get('system_state')
        sensor_data = environment.get('sensor_data', {})
        current_time = environment.get('time', 0.0)

        # 从消息获取场景信息
        scenario = self.get_knowledge('current_scenario', ScenarioType.S1_A_DUAL_BALANCED)

        # 从消息获取参考目标
        if 'reference' in environment:
            self.reference.update(environment['reference'])

        return {
            'system_state': system_state,
            'sensor_data': sensor_data,
            'scenario': scenario,
            'reference': self.reference,
            'time': current_time
        }

    def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """决策 - 计算控制量"""
        system_state = perception['system_state']
        scenario = perception['scenario']
        reference = perception['reference']
        sensor_data = perception['sensor_data']
        current_time = perception['time']

        if system_state is None:
            return {'control': self.last_control, 'error': 'No system state'}

        # 切换场景对应的控制器
        self.controller.switch_scenario(scenario, bumpless=True)

        # 构建状态字典
        state_dict = {
            'Q1': getattr(system_state.hydraulic, 'Q1', 132.5),
            'Q2': getattr(system_state.hydraulic, 'Q2', 132.5),
            'H_inlet': getattr(system_state.hydraulic, 'H_inlet', 106.05),
            'H_outlet': getattr(system_state.hydraulic, 'H_outlet', 104.79),
            'gate_1': getattr(system_state.actuators, 'gate_inlet_1', 1.0),
            'gate_2': getattr(system_state.actuators, 'gate_inlet_2', 1.0)
        }

        # 计算控制
        dt = 0.1
        output = self.controller.compute(
            state_dict,
            reference,
            sensor_data,
            current_time,
            dt
        )

        # 更新性能指标
        self._update_performance(state_dict, reference, output)

        return {
            'control': output.u_local,
            'mpc_output': output,
            'interlock_triggered': output.interlock_triggered,
            'interlock_actions': output.interlock_actions
        }

    def _update_performance(
        self,
        state: Dict[str, float],
        reference: Dict[str, Any],
        output: Any
    ):
        """更新性能指标"""
        Q_actual = state['Q1'] + state['Q2']
        Q_ref = reference.get('Q_total', 265.0)

        self.performance_metrics['tracking_error'] = abs(Q_actual - Q_ref) / Q_ref
        self.performance_metrics['control_effort'] = np.linalg.norm(
            output.u_local - self.last_control
        )

    def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """执行 - 发送控制指令"""
        control = decision['control']
        self.last_control = control

        # 发送控制指令
        self.send_message(Message(
            msg_type=MessageType.COMMAND,
            receiver='*',
            content={
                'command_type': 'gate_control',
                'gate_1': float(control[0]),
                'gate_2': float(control[1])
            },
            priority=5
        ))

        # 如果触发联锁，发送告警
        if decision.get('interlock_triggered', False):
            for action in decision.get('interlock_actions', []):
                self.alert({
                    'type': 'interlock',
                    'action': action.action_type,
                    'target': action.target,
                    'message': action.message
                })

        # 更新知识库
        self.update_knowledge('last_control', control.tolist())
        self.update_knowledge('performance', self.performance_metrics)

        return {
            'success': True,
            'control': control.tolist(),
            'performance': self.performance_metrics
        }

    def _handle_message(self, message: Message):
        """处理消息"""
        if message.msg_type == MessageType.OBSERVATION:
            # 更新场景信息
            if 'scenario' in message.content:
                scenario_str = message.content['scenario']
                try:
                    scenario = ScenarioType(scenario_str)
                    self.update_knowledge('current_scenario', scenario)
                except ValueError:
                    pass

        elif message.msg_type == MessageType.REQUEST:
            if message.content.get('request') == 'set_reference':
                self.reference.update(message.content.get('reference', {}))
                self.respond(message, {'status': 'reference_updated'})

            elif message.content.get('request') == 'get_performance':
                self.respond(message, {'performance': self.performance_metrics})

    def set_reference(self, Q_total: float):
        """设置流量参考值"""
        self.reference['Q_total'] = Q_total
