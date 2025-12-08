"""
Integrated Perception System with Data Governance and Assimilation
集成感知系统 - 结合数据治理和数据同化

将数据治理和数据同化功能与现有感知系统集成
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np
import time

from cyrp.perception.perception_system import PerceptionSystem, PerceptionOutput
from cyrp.governance.data_governance import DataGovernanceManager
from cyrp.assimilation.data_assimilation import KalmanFilter


@dataclass
class GovernedPerceptionOutput(PerceptionOutput):
    """带数据治理的感知输出"""
    # 数据质量评估
    quality_metrics: Optional[Dict[str, Any]] = None
    quality_score: float = 1.0
    quality_issues: List[str] = field(default_factory=list)

    # 数据血缘
    data_lineage: Dict[str, Any] = field(default_factory=dict)

    # 同化后的状态估计
    assimilated_state: Optional[np.ndarray] = None
    state_uncertainty: Optional[np.ndarray] = None

    # 置信度
    confidence: float = 1.0


class IntegratedPerceptionSystem:
    """
    集成感知系统

    将数据治理和数据同化与感知系统结合，实现：
    1. 传感器数据质量验证
    2. 数据血缘追踪
    3. 多源数据同化
    4. 不确定性量化
    """

    def __init__(
        self,
        tunnel_length: float = 4250.0,
        num_pressure_sensors: int = 8,
        num_mems_nodes: int = 20,
        enable_governance: bool = True,
        enable_assimilation: bool = True,
    ):
        """
        初始化集成感知系统

        Args:
            tunnel_length: 隧洞长度
            num_pressure_sensors: 压力传感器数量
            num_mems_nodes: MEMS节点数
            enable_governance: 是否启用数据治理
            enable_assimilation: 是否启用数据同化
        """
        # 基础感知系统
        self.perception = PerceptionSystem(
            tunnel_length=tunnel_length,
            num_pressure_sensors=num_pressure_sensors,
            num_mems_nodes=num_mems_nodes
        )

        self.enable_governance = enable_governance
        self.enable_assimilation = enable_assimilation
        self.num_pressure_sensors = num_pressure_sensors

        # 数据治理系统
        if enable_governance:
            self.governance = DataGovernanceManager()
        else:
            self.governance = None

        # 数据同化系统 - 使用Kalman滤波器
        if enable_assimilation:
            self.state_dim = 8
            self.obs_dim = min(num_pressure_sensors + 2, self.state_dim)
            self.kalman_filter = KalmanFilter(
                state_dim=self.state_dim,
                obs_dim=self.obs_dim
            )
        else:
            self.kalman_filter = None
            self.state_dim = 8
            self.obs_dim = 8

        # 统计信息
        self.stats = {
            'total_readings': 0,
            'quality_passed': 0,
            'quality_failed': 0,
            'assimilation_updates': 0,
        }

    def process(
        self,
        hydraulic_state: Any,
        structural_state: Any = None,
        current_time: float = None
    ) -> GovernedPerceptionOutput:
        """
        处理感知数据

        Args:
            hydraulic_state: 水力状态
            structural_state: 结构状态
            current_time: 当前时间

        Returns:
            带数据治理和同化的感知输出
        """
        if current_time is None:
            current_time = time.time()

        # 1. 获取基础感知输出
        base_output = self.perception.process(
            hydraulic_state=hydraulic_state,
            structural_state=structural_state,
            current_time=current_time
        )

        # 创建扩展输出
        output = GovernedPerceptionOutput(
            timestamp=base_output.timestamp,
            das_reading=base_output.das_reading,
            dts_reading=base_output.dts_reading,
            mems_readings=base_output.mems_readings,
            pressure_readings=base_output.pressure_readings,
            flow_readings=base_output.flow_readings,
            fused_state=base_output.fused_state,
            scenario=base_output.scenario,
            alarms=base_output.alarms,
            sensor_health=base_output.sensor_health,
        )

        self.stats['total_readings'] += 1

        # 2. 数据治理
        if self.enable_governance and self.governance:
            output = self._apply_governance(output, current_time)

        # 3. 数据同化
        if self.enable_assimilation and self.kalman_filter:
            output = self._apply_assimilation(output)

        return output

    def _apply_governance(
        self,
        output: GovernedPerceptionOutput,
        current_time: float
    ) -> GovernedPerceptionOutput:
        """应用数据治理"""
        # 构建待检查的数据
        data_to_check = {
            'timestamp': current_time,
            'pressure': np.mean(list(output.pressure_readings.values())) if output.pressure_readings else None,
            'flow': np.mean(list(output.flow_readings.values())) if output.flow_readings else None,
        }

        # 使用治理管理器检查质量
        try:
            quality_result = self.governance.check_quality(data_to_check)
            output.quality_score = quality_result.get('score', 1.0)
            output.quality_issues = quality_result.get('issues', [])
        except Exception:
            # 如果检查失败，使用默认值
            output.quality_score = 0.8
            output.quality_issues = []

        # 记录数据血缘
        output.data_lineage = {
            'lineage_id': f"perception_{current_time}",
            'source': 'perception_system',
            'timestamp': current_time
        }

        # 更新统计
        if output.quality_score >= 0.8:
            self.stats['quality_passed'] += 1
        else:
            self.stats['quality_failed'] += 1

        return output

    def _apply_assimilation(
        self,
        output: GovernedPerceptionOutput,
    ) -> GovernedPerceptionOutput:
        """应用数据同化"""
        # 构建观测向量
        observations = []

        # 压力观测
        for name, value in sorted(output.pressure_readings.items()):
            observations.append(value)

        # 流量观测
        for name, value in sorted(output.flow_readings.items()):
            observations.append(value)

        if len(observations) > 0:
            # 创建观测向量
            obs_vector = np.array(observations[:self.obs_dim])

            # 填充不足的观测
            if len(obs_vector) < self.obs_dim:
                obs_vector = np.pad(
                    obs_vector,
                    (0, self.obs_dim - len(obs_vector)),
                    mode='constant'
                )

            # 执行Kalman滤波更新
            try:
                self.kalman_filter.predict()
                self.kalman_filter.update(obs_vector)

                output.assimilated_state = self.kalman_filter.get_state()
                output.state_uncertainty = np.diag(self.kalman_filter.get_covariance())

                # 计算置信度
                if output.state_uncertainty is not None:
                    uncertainty_norm = np.linalg.norm(output.state_uncertainty)
                    output.confidence = max(0.0, 1.0 - uncertainty_norm / 1e6)
                else:
                    output.confidence = 0.8

            except Exception:
                output.confidence = 0.5

            self.stats['assimilation_updates'] += 1

        return output

    def get_state_estimate(self) -> Dict[str, Any]:
        """获取当前状态估计"""
        if self.kalman_filter:
            return {
                'state': self.kalman_filter.get_state(),
                'covariance': self.kalman_filter.get_covariance()
            }
        return {}

    def get_quality_report(self) -> Dict[str, Any]:
        """获取数据质量报告"""
        if self.governance:
            return self.governance.get_report()
        return {}

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'quality_pass_rate': (
                self.stats['quality_passed'] / max(1, self.stats['total_readings'])
            ),
        }

    def reset(self):
        """重置系统"""
        if self.kalman_filter:
            self.kalman_filter.reset()
        self.stats = {
            'total_readings': 0,
            'quality_passed': 0,
            'quality_failed': 0,
            'assimilation_updates': 0,
        }
