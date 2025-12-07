"""
Perception Agent for CYRP Multi-Agent System.
穿黄工程感知智能体
"""

from typing import Dict, Any, Optional
import numpy as np

from cyrp.agents.base_agent import Agent, AgentRole, Message, MessageType
from cyrp.perception import PerceptionSystem, PerceptionOutput


class PerceptionAgent(Agent):
    """
    感知智能体

    负责:
    1. 多传感器数据采集
    2. 数据融合与滤波
    3. 异常检测
    4. 场景特征提取
    """

    def __init__(self, name: str = "PerceptionAgent"):
        super().__init__(name, AgentRole.PERCEPTION, priority=4)

        # 感知系统
        self.perception_system = PerceptionSystem()

        # 感知结果缓存
        self.last_perception: Optional[PerceptionOutput] = None

        # 异常检测阈值
        self.anomaly_thresholds = {
            'leak_confidence': 0.8,
            'vibration_level': 0.1,
            'pressure_deviation': 0.1
        }

    def perceive(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """感知环境"""
        # 从环境获取系统状态
        system_state = environment.get('system_state')

        if system_state is None:
            return {'error': 'No system state available'}

        # 更新感知系统
        dt = environment.get('dt', 0.1)
        perception_output = self.perception_system.update(system_state, dt)

        self.last_perception = perception_output

        # 提取特征
        features = self._extract_features(perception_output)

        # 检测异常
        anomalies = self._detect_anomalies(perception_output)

        return {
            'perception': perception_output,
            'features': features,
            'anomalies': anomalies,
            'scenario': perception_output.scenario,
            'alarms': perception_output.alarms
        }

    def _extract_features(self, perception: PerceptionOutput) -> Dict[str, float]:
        """提取感知特征"""
        fused = perception.fused_state
        if fused is None:
            return {}

        return {
            'total_flow': fused.Q1 + fused.Q2,
            'flow_imbalance': abs(fused.Q1 - fused.Q2) / max(fused.Q1 + fused.Q2, 1),
            'pressure_inlet': fused.P_inlet,
            'pressure_outlet': fused.P_outlet,
            'leak_detected': 1.0 if fused.leak_detected else 0.0,
            'leak_position': fused.leak_position,
            'vibration_level': fused.vibration_level,
            'settlement': fused.settlement,
            'tilt': fused.tilt
        }

    def _detect_anomalies(self, perception: PerceptionOutput) -> Dict[str, bool]:
        """检测异常"""
        anomalies = {}

        if perception.fused_state:
            fused = perception.fused_state

            # 渗漏异常
            anomalies['leakage'] = (
                fused.leak_detected and
                fused.confidence > self.anomaly_thresholds['leak_confidence']
            )

            # 振动异常
            anomalies['vibration'] = (
                fused.vibration_level > self.anomaly_thresholds['vibration_level']
            )

            # 压力异常
            expected_pressure = 5.5e5
            pressure_dev = abs(fused.P_inlet - expected_pressure) / expected_pressure
            anomalies['pressure'] = (
                pressure_dev > self.anomaly_thresholds['pressure_deviation']
            )

        return anomalies

    def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """决策"""
        anomalies = perception.get('anomalies', {})
        scenario = perception.get('scenario')

        # 根据异常情况决定是否发送告警
        alerts = []
        if anomalies.get('leakage', False):
            alerts.append({
                'type': 'leakage',
                'severity': 'critical',
                'position': perception['features'].get('leak_position', 0)
            })

        if anomalies.get('vibration', False):
            alerts.append({
                'type': 'vibration',
                'severity': 'warning',
                'level': perception['features'].get('vibration_level', 0)
            })

        return {
            'alerts': alerts,
            'scenario': scenario,
            'features': perception['features']
        }

    def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """执行"""
        # 发送告警消息
        for alert in decision.get('alerts', []):
            self.alert(alert, priority=5 if alert['severity'] == 'critical' else 3)

        # 广播场景识别结果
        if decision.get('scenario'):
            self.broadcast({
                'scenario': decision['scenario'].scenario_type.value,
                'confidence': decision['scenario'].confidence,
                'features': decision['features']
            }, MessageType.OBSERVATION)

        # 更新知识库
        self.update_knowledge('last_features', decision['features'])
        if decision.get('scenario'):
            self.update_knowledge('current_scenario', decision['scenario'].scenario_type)

        return {
            'success': True,
            'alerts_sent': len(decision.get('alerts', [])),
            'scenario': decision.get('scenario')
        }

    def _handle_message(self, message: Message):
        """处理消息"""
        if message.msg_type == MessageType.REQUEST:
            if message.content.get('request') == 'get_features':
                self.respond(message, {'features': self.get_knowledge('last_features', {})})
