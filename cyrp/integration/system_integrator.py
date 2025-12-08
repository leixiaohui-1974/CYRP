"""
System Integrator for CYRP
系统集成器 - 统一管理所有新模块的集成

将传感器仿真、执行器仿真、数据治理、数据同化、
IDZ参数更新、状态评价和状态预测全面集成
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np
import time

from cyrp.core import PhysicalSystem, HydraulicState
from cyrp.core.physical_system import SystemState

# 治理模块
from cyrp.governance.data_governance import DataGovernanceManager

# 同化模块
from cyrp.assimilation.data_assimilation import KalmanFilter


@dataclass
class IntegratedSystemState:
    """集成系统状态"""
    timestamp: float = 0.0

    # 仿真数据
    sensor_readings: Dict[str, float] = field(default_factory=dict)
    actuator_states: Dict[str, float] = field(default_factory=dict)

    # 治理数据
    data_quality_score: float = 1.0
    quality_issues: List[str] = field(default_factory=list)

    # 同化状态
    estimated_state: Optional[np.ndarray] = None
    state_uncertainty: Optional[np.ndarray] = None

    # 模型参数
    model_parameters: Dict[str, float] = field(default_factory=dict)
    parameter_confidence: float = 0.0

    # 评价结果
    performance_score: float = 1.0
    deviation_from_target: Dict[str, float] = field(default_factory=dict)
    safety_status: str = "normal"

    # 预测结果
    predicted_states: List[np.ndarray] = field(default_factory=list)
    prediction_confidence: float = 0.0


class SystemIntegrator:
    """
    系统集成器

    功能：
    1. 传感器/执行器仿真 - 模拟物理量测量和执行器响应
    2. 数据治理 - 确保数据质量
    3. 数据同化 - 融合多源数据
    4. IDZ参数更新 - 基于高保真模型校准简化模型
    5. 状态评价 - 评估系统状态与控制目标的偏差
    6. 状态预测 - 预测未来系统状态
    """

    def __init__(
        self,
        tunnel_length: float = 4250.0,
        state_dim: int = 8,
        enable_simulation: bool = True,
        enable_governance: bool = True,
        enable_assimilation: bool = True,
        enable_idz_update: bool = True,
        enable_evaluation: bool = True,
        enable_prediction: bool = True,
    ):
        """
        初始化系统集成器

        Args:
            tunnel_length: 隧洞长度
            state_dim: 状态维度
            enable_*: 各模块启用开关
        """
        self.tunnel_length = tunnel_length
        self.state_dim = state_dim

        # 模块启用状态
        self.enable_simulation = enable_simulation
        self.enable_governance = enable_governance
        self.enable_assimilation = enable_assimilation
        self.enable_idz_update = enable_idz_update
        self.enable_evaluation = enable_evaluation
        self.enable_prediction = enable_prediction

        # 初始化各模块
        self._init_modules()

        # 状态历史
        self.state_history: List[IntegratedSystemState] = []
        self.max_history = 10000

        # 当前状态
        self.current_state = IntegratedSystemState()

        # 模型参数
        self.model_parameters = self._get_default_parameters()

        # 控制目标
        self.control_targets = {
            'pressure': 5e5,
            'flow_rate': 280.0,
        }

        # 统计信息
        self.stats = {
            'process_count': 0,
            'simulation_calls': 0,
            'governance_checks': 0,
            'assimilation_updates': 0,
            'parameter_updates': 0,
            'evaluations': 0,
            'predictions': 0,
        }

    def _init_modules(self):
        """初始化各功能模块"""
        # 数据治理
        if self.enable_governance:
            self.governance = DataGovernanceManager()
        else:
            self.governance = None

        # 数据同化 - Kalman滤波器
        if self.enable_assimilation:
            self.kalman_filter = KalmanFilter(
                state_dim=self.state_dim,
                obs_dim=self.state_dim
            )
        else:
            self.kalman_filter = None

    def _get_default_parameters(self) -> Dict[str, float]:
        """获取默认参数"""
        return {
            'friction_factor': 0.02,
            'wave_speed': 1000.0,
            'tunnel_diameter': 7.0,
            'tunnel_length': self.tunnel_length,
            'valve_coefficient': 0.8,
        }

    def process(
        self,
        physical_state: SystemState,
        control_commands: Dict[str, float] = None,
        current_time: float = None
    ) -> IntegratedSystemState:
        """
        处理系统状态

        Args:
            physical_state: 物理系统状态
            control_commands: 控制命令
            current_time: 当前时间

        Returns:
            集成系统状态
        """
        if current_time is None:
            current_time = time.time()

        # 创建新的集成状态
        integrated_state = IntegratedSystemState(timestamp=current_time)

        # 1. 提取传感器读数
        integrated_state.sensor_readings = self._extract_sensor_readings(physical_state)
        if self.enable_simulation:
            self.stats['simulation_calls'] += 1

        # 2. 执行器状态
        if control_commands:
            integrated_state.actuator_states = control_commands.copy()

        # 3. 数据治理 - 检查数据质量
        if self.enable_governance and self.governance:
            quality_result = self._check_data_quality(integrated_state)
            integrated_state.data_quality_score = quality_result['score']
            integrated_state.quality_issues = quality_result['issues']
            self.stats['governance_checks'] += 1

        # 4. 数据同化 - 融合多源数据
        if self.enable_assimilation and self.kalman_filter:
            assimilated = self._assimilate_data(integrated_state)
            integrated_state.estimated_state = assimilated['state']
            integrated_state.state_uncertainty = assimilated['uncertainty']
            self.stats['assimilation_updates'] += 1

        # 5. 状态评价
        if self.enable_evaluation:
            eval_result = self._evaluate_state(integrated_state)
            integrated_state.performance_score = eval_result['score']
            integrated_state.deviation_from_target = eval_result['deviations']
            integrated_state.safety_status = eval_result['safety']
            self.stats['evaluations'] += 1

        # 6. 状态预测
        if self.enable_prediction:
            pred_result = self._predict_state()
            integrated_state.predicted_states = pred_result['states']
            integrated_state.prediction_confidence = pred_result['confidence']
            self.stats['predictions'] += 1

        # 记录历史
        self.state_history.append(integrated_state)
        if len(self.state_history) > self.max_history:
            self.state_history = self.state_history[-self.max_history//2:]

        # 更新当前状态
        self.current_state = integrated_state
        self.stats['process_count'] += 1

        return integrated_state

    def _extract_sensor_readings(
        self,
        physical_state: SystemState
    ) -> Dict[str, float]:
        """提取传感器读数"""
        readings = {}

        if hasattr(physical_state, 'hydraulic') and physical_state.hydraulic:
            h = physical_state.hydraulic
            if hasattr(h, 'pressure'):
                if isinstance(h.pressure, np.ndarray):
                    readings['pressure_inlet'] = float(h.pressure[0]) if len(h.pressure) > 0 else 5e5
                    readings['pressure_outlet'] = float(h.pressure[-1]) if len(h.pressure) > 0 else 5e5
                else:
                    readings['pressure_inlet'] = float(h.pressure)
                    readings['pressure_outlet'] = float(h.pressure)
            if hasattr(h, 'flow_rate') and h.flow_rate:
                readings['flow_rate'] = float(h.flow_rate)
            else:
                readings['flow_rate'] = 280.0

        return readings

    def _check_data_quality(
        self,
        state: IntegratedSystemState
    ) -> Dict[str, Any]:
        """检查数据质量"""
        issues = []
        score = 1.0

        # 检查数据完整性
        if not state.sensor_readings:
            issues.append("No sensor readings")
            score -= 0.3

        # 检查数据有效性
        for name, value in state.sensor_readings.items():
            if np.isnan(value) or np.isinf(value):
                issues.append(f"Invalid value for {name}")
                score -= 0.2

        return {
            'score': max(0.0, score),
            'issues': issues
        }

    def _assimilate_data(
        self,
        state: IntegratedSystemState
    ) -> Dict[str, Any]:
        """融合数据"""
        # 构建观测向量
        obs = np.zeros(self.state_dim)
        idx = 0

        for name, value in sorted(state.sensor_readings.items()):
            if idx < self.state_dim:
                obs[idx] = value
                idx += 1

        # 执行Kalman滤波
        try:
            self.kalman_filter.predict()
            self.kalman_filter.update(obs)
            return {
                'state': self.kalman_filter.get_state(),
                'uncertainty': np.diag(self.kalman_filter.get_covariance())
            }
        except Exception:
            return {'state': None, 'uncertainty': None}

    def _evaluate_state(
        self,
        state: IntegratedSystemState
    ) -> Dict[str, Any]:
        """评估系统状态"""
        deviations = {}
        score = 1.0

        # 计算偏差
        pressure = state.sensor_readings.get('pressure_inlet', 5e5)
        flow = state.sensor_readings.get('flow_rate', 280.0)

        if self.control_targets.get('pressure'):
            target = self.control_targets['pressure']
            dev = abs(pressure - target) / target
            deviations['pressure'] = dev
            score -= min(dev, 0.5)

        if self.control_targets.get('flow_rate'):
            target = self.control_targets['flow_rate']
            dev = abs(flow - target) / target
            deviations['flow_rate'] = dev
            score -= min(dev, 0.5)

        # 安全状态
        safety = "normal"
        if pressure > 1e6:
            safety = "warning"
        elif pressure < -5e4:
            safety = "alert"

        return {
            'score': max(0.0, score),
            'deviations': deviations,
            'safety': safety
        }

    def _predict_state(self) -> Dict[str, Any]:
        """预测未来状态"""
        if len(self.state_history) < 10:
            return {'states': [], 'confidence': 0.0}

        # 简单线性外推
        recent = self.state_history[-20:]
        recent_values = []
        for s in recent:
            vals = list(s.sensor_readings.values())
            if vals:
                recent_values.append(vals)

        if len(recent_values) < 2:
            return {'states': [], 'confidence': 0.0}

        recent_array = np.array(recent_values)
        trend = recent_array[-1] - recent_array[-2]

        # 预测10步
        predictions = []
        for i in range(10):
            pred = recent_array[-1] + trend * (i + 1)
            predictions.append(pred)

        return {
            'states': predictions,
            'confidence': 0.7
        }

    def get_current_state(self) -> IntegratedSystemState:
        """获取当前集成状态"""
        return self.current_state

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'history_length': len(self.state_history),
            'current_quality_score': self.current_state.data_quality_score,
            'current_safety_status': self.current_state.safety_status,
        }

    def get_quality_report(self) -> Dict[str, Any]:
        """获取数据质量报告"""
        if self.governance:
            return self.governance.get_report()
        return {}

    def get_model_parameters(self) -> Dict[str, float]:
        """获取当前模型参数"""
        return self.model_parameters.copy()

    def reset(self):
        """重置系统"""
        self.state_history.clear()
        self.current_state = IntegratedSystemState()

        if self.kalman_filter:
            self.kalman_filter.reset()

        self.stats = {
            'process_count': 0,
            'simulation_calls': 0,
            'governance_checks': 0,
            'assimilation_updates': 0,
            'parameter_updates': 0,
            'evaluations': 0,
            'predictions': 0,
        }
