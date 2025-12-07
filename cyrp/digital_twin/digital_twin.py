"""
Digital Twin for Yellow River Crossing Project.
穿黄工程数字孪生

实现物理系统的数字镜像，支持实时同步和预测分析
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np
import time
from collections import deque

from cyrp.core import PhysicalSystem, HydraulicState
from cyrp.core.physical_system import SystemState, ControlCommand


@dataclass
class TwinState:
    """数字孪生状态"""
    timestamp: float = 0.0
    sync_status: str = "idle"
    prediction_horizon: float = 0.0

    # 物理状态镜像
    hydraulic_state: Optional[HydraulicState] = None
    structural_state: Any = None

    # 预测状态
    predicted_states: List[Any] = field(default_factory=list)

    # 健康评估
    health_score: float = 1.0
    remaining_life: float = float('inf')

    # 同步误差
    sync_error: float = 0.0


class DigitalTwin:
    """
    穿黄工程数字孪生

    功能:
    1. 实时状态同步
    2. 预测性仿真
    3. 健康评估
    4. What-if分析
    """

    def __init__(self):
        """初始化数字孪生"""
        # 物理系统镜像
        self.physical_mirror = PhysicalSystem()

        # 状态历史
        self.state_history: deque = deque(maxlen=10000)

        # 当前状态
        self.twin_state = TwinState()

        # 同步参数
        self.sync_interval = 1.0  # 同步间隔 (s)
        self.last_sync_time = 0.0

        # 预测参数
        self.prediction_horizon = 3600.0  # 预测时域 (s)
        self.prediction_dt = 10.0  # 预测步长 (s)

    def sync(self, real_state: SystemState):
        """
        同步实际系统状态

        Args:
            real_state: 实际系统状态
        """
        current_time = time.time()

        # 更新镜像状态
        self.physical_mirror.state = real_state

        # 计算同步误差
        if self.twin_state.hydraulic_state:
            error = self._compute_sync_error(
                real_state.hydraulic,
                self.twin_state.hydraulic_state
            )
            self.twin_state.sync_error = error

        # 更新孪生状态
        self.twin_state.timestamp = current_time
        self.twin_state.sync_status = "synced"
        self.twin_state.hydraulic_state = real_state.hydraulic
        self.twin_state.structural_state = real_state.structural

        # 记录历史
        self.state_history.append({
            'time': current_time,
            'state': real_state,
            'sync_error': self.twin_state.sync_error
        })

        self.last_sync_time = current_time

    def _compute_sync_error(
        self,
        real: HydraulicState,
        predicted: HydraulicState
    ) -> float:
        """计算同步误差"""
        errors = [
            abs(real.Q1 - predicted.Q1) / max(real.Q1, 1),
            abs(real.Q2 - predicted.Q2) / max(real.Q2, 1),
            abs(real.H_inlet - predicted.H_inlet) / max(real.H_inlet, 1),
            abs(real.H_outlet - predicted.H_outlet) / max(real.H_outlet, 1)
        ]
        return np.mean(errors)

    def predict(
        self,
        control_sequence: List[np.ndarray],
        horizon: Optional[float] = None
    ) -> List[SystemState]:
        """
        预测未来状态

        Args:
            control_sequence: 控制序列
            horizon: 预测时域 (s)

        Returns:
            预测状态序列
        """
        horizon = horizon or self.prediction_horizon

        # 创建预测用的系统副本
        pred_system = PhysicalSystem()
        pred_system.state = self.physical_mirror.state

        predicted_states = []
        t = 0.0
        idx = 0

        while t < horizon:
            # 获取控制
            if idx < len(control_sequence):
                control = control_sequence[idx]
            else:
                control = control_sequence[-1] if control_sequence else np.array([1.0, 1.0])

            # 构建控制指令
            cmd = ControlCommand(
                gate_inlet_1_target=control[0],
                gate_inlet_2_target=control[1]
            )

            # 步进预测
            state = pred_system.step(cmd, self.prediction_dt)
            predicted_states.append(state)

            t += self.prediction_dt
            idx += 1

        self.twin_state.predicted_states = predicted_states
        self.twin_state.prediction_horizon = horizon

        return predicted_states

    def what_if(
        self,
        scenario: Dict[str, Any],
        duration: float = 3600.0
    ) -> Dict[str, Any]:
        """
        What-if分析

        Args:
            scenario: 假设场景
            duration: 分析时长

        Returns:
            分析结果
        """
        # 创建分析用的系统副本
        analysis_system = PhysicalSystem()
        analysis_system.state = self.physical_mirror.state

        # 应用场景条件
        if 'fault' in scenario:
            fault = scenario['fault']
            analysis_system.fault_injector.schedule_fault(
                fault['type'],
                0,  # 立即开始
                duration,
                fault.get('parameters', {})
            )

        # 运行分析
        results = {
            'states': [],
            'metrics': {}
        }

        control = scenario.get('control', np.array([1.0, 1.0]))
        dt = 10.0
        t = 0.0

        while t < duration:
            cmd = ControlCommand(
                gate_inlet_1_target=control[0],
                gate_inlet_2_target=control[1]
            )
            state = analysis_system.step(cmd, dt)
            results['states'].append(state)
            t += dt

        # 计算指标
        states = results['states']
        if states:
            Q_total = [s.hydraulic.Q1 + s.hydraulic.Q2 for s in states]
            results['metrics'] = {
                'avg_flow': np.mean(Q_total),
                'min_flow': np.min(Q_total),
                'max_flow': np.max(Q_total),
                'flow_stability': np.std(Q_total) / np.mean(Q_total),
                'max_risk_level': max([
                    5 if s.mode.value == 'emergency' else 1
                    for s in states
                ])
            }

        return results

    def assess_health(self) -> Dict[str, Any]:
        """
        健康评估

        Returns:
            健康评估结果
        """
        if not self.twin_state.structural_state:
            return {'health_score': 1.0, 'status': 'unknown'}

        struct = self.twin_state.structural_state

        # 安全系数
        sf = getattr(struct, 'safety_factor', 2.0)

        # 健康评分 (0-1)
        if sf >= 2.0:
            health_score = 1.0
        elif sf >= 1.5:
            health_score = 0.8
        elif sf >= 1.0:
            health_score = 0.5
        else:
            health_score = 0.2

        # 失效模式
        failure_mode = getattr(struct, 'failure_mode', None)

        # 剩余寿命估算 (简化)
        if hasattr(struct, 'leakage_rate') and struct.leakage_rate > 0:
            remaining_life = 365 * 24 * 3600 / struct.leakage_rate  # 秒
        else:
            remaining_life = float('inf')

        self.twin_state.health_score = health_score
        self.twin_state.remaining_life = remaining_life

        return {
            'health_score': health_score,
            'safety_factor': sf,
            'failure_mode': str(failure_mode) if failure_mode else 'none',
            'remaining_life_hours': remaining_life / 3600 if remaining_life < float('inf') else None,
            'recommendations': self._generate_recommendations(health_score, failure_mode)
        }

    def _generate_recommendations(
        self,
        health_score: float,
        failure_mode: Any
    ) -> List[str]:
        """生成维护建议"""
        recommendations = []

        if health_score < 0.5:
            recommendations.append("建议立即进行检修")
        elif health_score < 0.8:
            recommendations.append("建议安排检查维护")

        if failure_mode:
            mode_str = str(failure_mode)
            if 'leak' in mode_str.lower():
                recommendations.append("检查衬砌完整性")
            if 'buckling' in mode_str.lower():
                recommendations.append("监测外水压力变化")

        return recommendations

    def get_state(self) -> TwinState:
        """获取孪生状态"""
        return self.twin_state

    def get_visualization_data(self) -> Dict[str, Any]:
        """获取可视化数据"""
        if not self.twin_state.hydraulic_state:
            return {}

        hydro = self.twin_state.hydraulic_state

        return {
            'tunnel_1': {
                'flow': hydro.Q1,
                'velocity': hydro.V1,
                'pressure_profile': hydro.H_profile_1.tolist() if hasattr(hydro.H_profile_1, 'tolist') else []
            },
            'tunnel_2': {
                'flow': hydro.Q2,
                'velocity': hydro.V2,
                'pressure_profile': hydro.H_profile_2.tolist() if hasattr(hydro.H_profile_2, 'tolist') else []
            },
            'system': {
                'total_flow': hydro.Q1 + hydro.Q2,
                'head_loss': hydro.total_head_loss,
                'inlet_level': hydro.H_inlet,
                'outlet_level': hydro.H_outlet
            },
            'health': {
                'score': self.twin_state.health_score,
                'sync_error': self.twin_state.sync_error
            }
        }
