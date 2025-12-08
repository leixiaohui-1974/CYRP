"""
Integrated Safety Agent with State Evaluation
集成安全智能体 - 结合状态评价功能

将状态评价模块与安全智能体集成
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import numpy as np
import time

from cyrp.agents.base_agent import Agent, AgentRole, Message, MessageType
from cyrp.control.safety_interlocks import SafetyInterlockSystem


@dataclass
class SafetyAssessment:
    """安全评估结果"""
    timestamp: float = 0.0
    risk_level: int = 1  # 1-5
    risk_score: float = 0.0
    risk_factors: List[str] = field(default_factory=list)

    # 安全边界检查
    boundary_violations: List[str] = field(default_factory=list)
    safety_margin: float = 1.0

    # 偏差
    deviation_from_target: Dict[str, float] = field(default_factory=dict)

    # 建议
    recommendations: List[str] = field(default_factory=list)


class IntegratedSafetyAgent(Agent):
    """
    集成安全智能体

    结合状态评价模块，实现：
    1. 系统状态实时评价
    2. 与控制目标偏差分析
    3. 安全边界监测
    4. 风险预测与预警
    """

    def __init__(self, name: str = "IntegratedSafetyAgent"):
        super().__init__(name, AgentRole.SAFETY, priority=5)

        # 安全联锁系统
        self.interlock_system = SafetyInterlockSystem()

        # 安全边界配置
        self.safety_limits = {
            'P_max': 1.0e6,     # 最大压力 Pa
            'P_min': -5e4,      # 最小压力 Pa
            'Q_max': 305.0,     # 最大流量 m³/s
            'Q_min': 0.0,       # 最小流量 m³/s
        }

        # 控制目标
        self.control_targets = {
            'pressure': 5e5,    # 目标压力 500kPa
            'flow_rate': 280.0, # 目标流量 280 m³/s
        }

        # 风险等级
        self.risk_level = 1

        # 报警历史
        self.alarm_history: List[Dict] = []

        # 评估历史
        self.assessment_history: List[SafetyAssessment] = []

    def perceive(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """感知安全相关信息"""
        system_state = environment.get('system_state')
        sensor_data = environment.get('sensor_data', {})
        current_time = environment.get('time', time.time())

        # 检查联锁
        triggered, actions = self.interlock_system.check_all(sensor_data, current_time)

        # 执行综合安全评估
        assessment = self._comprehensive_assessment(
            system_state, sensor_data, current_time
        )

        return {
            'system_state': system_state,
            'sensor_data': sensor_data,
            'interlock_triggered': triggered,
            'interlock_actions': actions,
            'safety_assessment': assessment,
            'time': current_time
        }

    def _comprehensive_assessment(
        self,
        system_state: Any,
        sensor_data: Dict[str, Any],
        current_time: float
    ) -> SafetyAssessment:
        """执行综合安全评估"""
        assessment = SafetyAssessment(timestamp=current_time)

        if system_state is None:
            return assessment

        # 1. 提取当前状态值
        current_values = self._extract_state_values(system_state, sensor_data)

        # 2. 计算与控制目标的偏差
        for key, target in self.control_targets.items():
            if key in current_values:
                deviation = (current_values[key] - target) / max(abs(target), 1)
                assessment.deviation_from_target[key] = deviation

        # 3. 安全边界检查
        boundary_check = self._check_safety_boundaries(current_values)
        assessment.boundary_violations = boundary_check['violations']
        assessment.safety_margin = boundary_check['margin']

        # 4. 计算综合风险
        risk_result = self._calculate_risk(current_values, boundary_check)
        assessment.risk_level = risk_result['level']
        assessment.risk_score = risk_result['score']
        assessment.risk_factors = risk_result['factors']

        # 5. 生成建议
        assessment.recommendations = self._generate_recommendations(assessment)

        # 记录评估历史
        self.assessment_history.append(assessment)
        if len(self.assessment_history) > 1000:
            self.assessment_history = self.assessment_history[-500:]

        # 更新风险等级
        self.risk_level = assessment.risk_level

        return assessment

    def _extract_state_values(
        self,
        system_state: Any,
        sensor_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """提取状态值"""
        values = {}

        # 从系统状态提取
        if hasattr(system_state, 'hydraulic'):
            hydraulic = system_state.hydraulic
            if hasattr(hydraulic, 'pressure'):
                if isinstance(hydraulic.pressure, np.ndarray):
                    values['pressure'] = float(np.mean(hydraulic.pressure))
                    values['pressure_max'] = float(np.max(hydraulic.pressure))
                    values['pressure_min'] = float(np.min(hydraulic.pressure))
                else:
                    values['pressure'] = float(hydraulic.pressure)
                    values['pressure_max'] = float(hydraulic.pressure)
                    values['pressure_min'] = float(hydraulic.pressure)
            if hasattr(hydraulic, 'flow_rate') and hydraulic.flow_rate:
                values['flow_rate'] = float(hydraulic.flow_rate)

        # 从传感器数据补充
        if 'pressure_max' not in values:
            values['pressure_max'] = sensor_data.get('pressure_max', 5e5)
        if 'pressure_min' not in values:
            values['pressure_min'] = sensor_data.get('pressure_min', 5e5)
        if 'flow_rate' not in values:
            values['flow_rate'] = sensor_data.get('flow', 280.0)

        return values

    def _check_safety_boundaries(
        self,
        values: Dict[str, float]
    ) -> Dict[str, Any]:
        """检查安全边界"""
        violations = []
        margins = []

        # 压力上限
        P_max = values.get('pressure_max', 5e5)
        margin_P_max = (self.safety_limits['P_max'] - P_max) / self.safety_limits['P_max']
        margins.append(margin_P_max)
        if P_max > self.safety_limits['P_max']:
            violations.append(f"压力超上限: {P_max/1e6:.2f} MPa")

        # 压力下限
        P_min = values.get('pressure_min', 5e5)
        if P_min < self.safety_limits['P_min']:
            violations.append(f"压力低于下限: {P_min/1e3:.1f} kPa")
            margins.append(-0.5)
        else:
            margin_P_min = (P_min - self.safety_limits['P_min']) / abs(self.safety_limits['P_min'])
            margins.append(min(margin_P_min, 1.0))

        # 流量上限
        Q = values.get('flow_rate', 280.0)
        margin_Q = (self.safety_limits['Q_max'] - Q) / self.safety_limits['Q_max']
        margins.append(margin_Q)
        if Q > self.safety_limits['Q_max']:
            violations.append(f"流量超上限: {Q:.1f} m³/s")

        return {
            'violations': violations,
            'margin': float(np.min(margins)) if margins else 1.0
        }

    def _calculate_risk(
        self,
        values: Dict[str, float],
        boundary_check: Dict[str, Any]
    ) -> Dict[str, Any]:
        """计算综合风险"""
        risk_factors = []
        risk_score = 0.0

        # 边界违反风险
        if boundary_check['violations']:
            risk_factors.extend(boundary_check['violations'])
            risk_score += len(boundary_check['violations']) * 2.0

        # 安全裕度风险
        margin = boundary_check['margin']
        if margin < 0.1:
            risk_factors.append("安全裕度不足")
            risk_score += 3.0
        elif margin < 0.2:
            risk_factors.append("安全裕度偏低")
            risk_score += 1.0

        # 计算风险等级 (1-5)
        if risk_score >= 6:
            level = 5
        elif risk_score >= 4:
            level = 4
        elif risk_score >= 2:
            level = 3
        elif risk_score >= 1:
            level = 2
        else:
            level = 1

        return {
            'level': level,
            'score': risk_score,
            'factors': risk_factors
        }

    def _generate_recommendations(
        self,
        assessment: SafetyAssessment
    ) -> List[str]:
        """生成安全建议"""
        recommendations = []

        # 基于风险等级
        if assessment.risk_level >= 4:
            recommendations.append("建议立即启动应急响应程序")
        elif assessment.risk_level >= 3:
            recommendations.append("建议提高监测频率，准备应急措施")

        # 基于边界违反
        for violation in assessment.boundary_violations:
            if "压力超上限" in violation:
                recommendations.append("建议打开泄压阀或减少进口流量")
            elif "压力低于下限" in violation:
                recommendations.append("建议检查是否存在泄漏或气蚀风险")
            elif "流量超上限" in violation:
                recommendations.append("建议调整进口闸门开度")

        return recommendations

    def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """决策"""
        assessment = perception.get('safety_assessment', SafetyAssessment())
        interlock_actions = perception.get('interlock_actions', [])

        # 决策逻辑
        action = 'monitor'
        emergency_actions = []

        if assessment.risk_level >= 5:
            action = 'emergency_stop'
            emergency_actions.append('trigger_emergency_shutdown')
        elif assessment.risk_level >= 4:
            action = 'reduce_risk'
            if assessment.boundary_violations:
                emergency_actions.append('activate_pressure_relief')
        elif assessment.risk_level >= 3:
            action = 'alert'

        return {
            'action': action,
            'emergency_actions': emergency_actions,
            'interlock_actions': interlock_actions,
            'risk_level': assessment.risk_level,
            'recommendations': assessment.recommendations
        }

    def act(self, decision: Dict[str, Any]) -> List[Message]:
        """执行决策"""
        messages = []

        action = decision.get('action', 'monitor')

        if action == 'emergency_stop':
            messages.append(Message(
                sender=self.name,
                receiver="CoordinatorAgent",
                msg_type=MessageType.ALERT,
                content={
                    'alert_type': 'emergency',
                    'action': 'emergency_stop',
                    'risk_level': decision.get('risk_level', 5)
                }
            ))

        elif action in ['reduce_risk', 'alert']:
            messages.append(Message(
                sender=self.name,
                receiver="ControlAgent",
                msg_type=MessageType.WARNING,
                content={
                    'warning_type': action,
                    'risk_level': decision.get('risk_level', 3),
                    'recommendations': decision.get('recommendations', [])
                }
            ))

        return messages

    def get_safety_report(self) -> Dict[str, Any]:
        """获取安全报告"""
        if not self.assessment_history:
            return {'status': 'no_data'}

        recent = self.assessment_history[-1]
        return {
            'timestamp': recent.timestamp,
            'risk_level': recent.risk_level,
            'risk_score': recent.risk_score,
            'risk_factors': recent.risk_factors,
            'boundary_violations': recent.boundary_violations,
            'safety_margin': recent.safety_margin,
            'recommendations': recent.recommendations,
            'assessment_count': len(self.assessment_history),
        }
