"""
Safety Agent for CYRP Multi-Agent System.
穿黄工程安全智能体
"""

from typing import Dict, Any, List
import numpy as np

from cyrp.agents.base_agent import Agent, AgentRole, Message, MessageType
from cyrp.control.safety_interlocks import SafetyInterlockSystem, InterlockType, InterlockState


class SafetyAgent(Agent):
    """
    安全智能体

    负责:
    1. 全系统安全监测
    2. 联锁逻辑管理
    3. 应急响应协调
    4. 风险评估
    """

    def __init__(self, name: str = "SafetyAgent"):
        super().__init__(name, AgentRole.SAFETY, priority=5)  # 最高优先级

        # 安全联锁系统
        self.interlock_system = SafetyInterlockSystem()

        # 风险等级
        self.risk_level = 1  # 1-5

        # 安全边界
        self.safety_limits = {
            'P_max': 1.0e6,
            'P_min': -5e4,
            'Q_max': 305.0,
            'flow_imbalance_max': 0.1,
            'vibration_max': 0.1,
            'settlement_max': 0.05
        }

        # 报警历史
        self.alarm_history: List[Dict] = []

    def perceive(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """感知安全相关信息"""
        system_state = environment.get('system_state')
        sensor_data = environment.get('sensor_data', {})
        current_time = environment.get('time', 0.0)

        # 检查联锁
        triggered, actions = self.interlock_system.check_all(sensor_data, current_time)

        # 评估风险
        risk_assessment = self._assess_risk(system_state, sensor_data)

        return {
            'system_state': system_state,
            'sensor_data': sensor_data,
            'interlock_triggered': triggered,
            'interlock_actions': actions,
            'risk_assessment': risk_assessment,
            'time': current_time
        }

    def _assess_risk(
        self,
        system_state: Any,
        sensor_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """风险评估"""
        risk_factors = []
        risk_score = 0

        if system_state is None:
            return {'level': 1, 'factors': [], 'score': 0}

        # 压力风险
        P_max = sensor_data.get('pressure_max', 5e5)
        if P_max > self.safety_limits['P_max'] * 0.9:
            risk_factors.append('high_pressure')
            risk_score += 3
        elif P_max > self.safety_limits['P_max'] * 0.8:
            risk_factors.append('elevated_pressure')
            risk_score += 1

        P_min = sensor_data.get('pressure_min', 5e5)
        if P_min < self.safety_limits['P_min'] * 0.5:
            risk_factors.append('vacuum_risk')
            risk_score += 4

        # 流量不平衡风险
        gates = sensor_data.get('gate_positions', [1.0, 1.0])
        if len(gates) >= 2:
            imbalance = abs(gates[0] - gates[1])
            if imbalance > self.safety_limits['flow_imbalance_max']:
                risk_factors.append('flow_imbalance')
                risk_score += 2

        # 结构风险
        if hasattr(system_state, 'structural'):
            if system_state.structural.liquefaction_index > 0.5:
                risk_factors.append('liquefaction')
                risk_score += 5

            if system_state.structural.leakage_rate > 0.01:
                risk_factors.append('leakage')
                risk_score += 4

        # 计算风险等级
        if risk_score >= 10:
            risk_level = 5
        elif risk_score >= 7:
            risk_level = 4
        elif risk_score >= 4:
            risk_level = 3
        elif risk_score >= 2:
            risk_level = 2
        else:
            risk_level = 1

        self.risk_level = risk_level

        return {
            'level': risk_level,
            'factors': risk_factors,
            'score': risk_score
        }

    def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """决策 - 确定安全响应"""
        risk_assessment = perception['risk_assessment']
        interlock_triggered = perception['interlock_triggered']
        interlock_actions = perception['interlock_actions']

        response_actions = []

        # 根据风险等级决定响应
        if risk_assessment['level'] >= 4:
            # 高风险 - 紧急响应
            response_actions.append({
                'action': 'emergency_response',
                'target': 'all',
                'details': risk_assessment['factors']
            })

        if interlock_triggered:
            # 应用联锁动作
            for action in interlock_actions:
                response_actions.append({
                    'action': 'interlock',
                    'target': action.target,
                    'value': action.value,
                    'message': action.message
                })

        # 检查是否需要触发应急场景切换
        trigger_emergency_scenario = False
        emergency_scenario = None

        if 'liquefaction' in risk_assessment['factors']:
            trigger_emergency_scenario = True
            emergency_scenario = 'S6_A_LIQUEFACTION'
        elif 'leakage' in risk_assessment['factors']:
            trigger_emergency_scenario = True
            emergency_scenario = 'S5_A_INNER_LEAK'

        return {
            'risk_level': risk_assessment['level'],
            'response_actions': response_actions,
            'trigger_emergency': trigger_emergency_scenario,
            'emergency_scenario': emergency_scenario,
            'interlock_status': self.interlock_system.get_status()
        }

    def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """执行 - 发送安全响应"""
        risk_level = decision['risk_level']
        response_actions = decision['response_actions']

        # 发送风险告警
        if risk_level >= 3:
            self.alert({
                'type': 'risk_warning',
                'level': risk_level,
                'actions': response_actions
            }, priority=risk_level)

        # 触发应急场景
        if decision.get('trigger_emergency', False):
            self.broadcast({
                'request': 'switch_scenario',
                'scenario': decision['emergency_scenario'],
                'reason': 'safety_triggered'
            }, MessageType.COMMAND)

        # 发送联锁状态
        self.broadcast({
            'interlock_status': decision['interlock_status'],
            'risk_level': risk_level
        }, MessageType.STATUS)

        # 记录告警历史
        if response_actions:
            self.alarm_history.append({
                'time': self.last_update_time,
                'risk_level': risk_level,
                'actions': response_actions
            })

        # 更新知识库
        self.update_knowledge('risk_level', risk_level)
        self.update_knowledge('interlock_status', decision['interlock_status'])

        return {
            'success': True,
            'risk_level': risk_level,
            'actions_taken': len(response_actions)
        }

    def _handle_message(self, message: Message):
        """处理消息"""
        if message.msg_type == MessageType.REQUEST:
            if message.content.get('request') == 'get_risk_level':
                self.respond(message, {'risk_level': self.risk_level})

            elif message.content.get('request') == 'reset_interlocks':
                self.interlock_system.reset_all()
                self.respond(message, {'status': 'interlocks_reset'})

    def is_safe(self) -> bool:
        """检查系统是否安全"""
        return self.risk_level <= 2
