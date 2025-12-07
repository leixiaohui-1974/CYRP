"""
Data Fusion Engine for CYRP Perception System.
穿黄工程数据融合引擎

采用扩展卡尔曼滤波(EKF)进行多源数据同化
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from scipy.linalg import inv, cholesky
from enum import Enum


@dataclass
class FusedState:
    """融合后的状态估计"""
    timestamp: float = 0.0

    # 水力状态
    Q1: float = 132.5  # 1#洞流量
    Q2: float = 132.5  # 2#洞流量
    P_inlet: float = 6.0e5  # 进口压力
    P_outlet: float = 5.0e5  # 出口压力

    # 结构状态
    settlement: float = 0.0
    tilt: float = 0.0
    vibration_level: float = 0.0

    # 渗漏状态
    leak_detected: bool = False
    leak_position: float = 0.0
    leak_rate: float = 0.0
    leak_type: str = "none"

    # 环形空腔
    cavity_level: float = 0.0
    cavity_pressure: float = 1.0e5

    # 估计协方差
    covariance: np.ndarray = field(default_factory=lambda: np.eye(10) * 0.01)

    # 置信度
    confidence: float = 0.95


class DataFusionEngine:
    """
    多源数据融合引擎

    实现:
    1. 扩展卡尔曼滤波 (EKF)
    2. 多传感器时空对齐
    3. 异常数据剔除
    4. 状态估计与预测
    """

    def __init__(self, state_dim: int = 10, measurement_dim: int = 15):
        """
        初始化融合引擎

        Args:
            state_dim: 状态向量维度
            measurement_dim: 测量向量维度
        """
        self.n = state_dim
        self.m = measurement_dim

        # 状态向量: [Q1, Q2, P_in, P_out, Z_gap, leak_pos, leak_rate, settle, tilt, vib]
        self.x = np.zeros(state_dim)
        self.x[0] = 132.5  # Q1
        self.x[1] = 132.5  # Q2
        self.x[2] = 6.0e5  # P_in
        self.x[3] = 5.0e5  # P_out

        # 状态协方差
        self.P = np.eye(state_dim) * 0.01

        # 过程噪声协方差
        self.Q = np.diag([
            1.0,  # Q1 流量波动
            1.0,  # Q2
            1000.0,  # P_in 压力波动
            1000.0,  # P_out
            0.001,  # Z_gap
            10.0,  # leak_pos
            0.001,  # leak_rate
            0.0001,  # settlement
            0.00001,  # tilt
            0.01  # vibration
        ])

        # 测量噪声协方差
        self.R = np.diag([
            2.0,  # Q1 流量计
            2.0,  # Q2
            5000.0,  # P1 压力计
            5000.0,  # P2
            5000.0,  # P3
            5000.0,  # P4
            0.01,  # cavity level
            0.1,  # DAS leak pos
            0.01,  # DAS leak intensity
            0.5,  # DTS temperature
            0.001,  # MEMS acc
            0.001,
            0.001,
            0.0001,  # MEMS tilt
            0.0001
        ])

        # 上一时刻状态
        self.x_prev = self.x.copy()
        self.t_prev = 0.0

    def _state_transition(self, x: np.ndarray, dt: float) -> np.ndarray:
        """
        状态转移函数

        x_{k+1} = f(x_k) + w

        Args:
            x: 当前状态
            dt: 时间步长

        Returns:
            预测状态
        """
        x_pred = x.copy()

        # 流量变化 (近似稳态)
        x_pred[0] = x[0]  # Q1 保持
        x_pred[1] = x[1]  # Q2 保持

        # 压力变化
        x_pred[2] = x[2]  # P_in
        x_pred[3] = x[3]  # P_out

        # 空腔水位变化 (如果有渗漏)
        leak_rate = x[6]
        x_pred[4] = x[4] + leak_rate * dt / 100.0  # 简化的水位上升模型

        # 渗漏位置不变
        x_pred[5] = x[5]

        # 渗漏率可能增长 (裂缝扩展)
        if x[6] > 0:
            x_pred[6] = x[6] * (1 + 0.001 * dt)

        # 沉降缓慢变化
        x_pred[7] = x[7]

        # 倾斜
        x_pred[8] = x[8]

        # 振动 (衰减)
        x_pred[9] = x[9] * 0.99

        return x_pred

    def _jacobian_F(self, x: np.ndarray, dt: float) -> np.ndarray:
        """
        状态转移雅可比矩阵

        F = ∂f/∂x
        """
        F = np.eye(self.n)

        # 空腔水位对渗漏率的偏导
        F[4, 6] = dt / 100.0

        # 渗漏率自身增长
        if x[6] > 0:
            F[6, 6] = 1 + 0.001 * dt

        # 振动衰减
        F[9, 9] = 0.99

        return F

    def _measurement_function(self, x: np.ndarray) -> np.ndarray:
        """
        测量函数

        z = h(x) + v

        Args:
            x: 状态向量

        Returns:
            预测测量向量
        """
        z = np.zeros(self.m)

        # 流量测量
        z[0] = x[0]  # Q1
        z[1] = x[1]  # Q2

        # 压力测量 (4个位置)
        z[2] = x[2]  # P_in_1
        z[3] = x[2]  # P_in_2
        z[4] = x[3]  # P_out_1
        z[5] = x[3]  # P_out_2

        # 空腔水位
        z[6] = x[4]

        # DAS 测量
        z[7] = x[5]  # leak position
        z[8] = x[6]  # leak rate indicator

        # DTS 测量 (温度异常幅度)
        z[9] = x[6] * 10  # 与渗漏率相关

        # MEMS 测量
        z[10] = x[9]  # vibration x
        z[11] = x[9]  # vibration y
        z[12] = x[9]  # vibration z
        z[13] = x[8]  # tilt x
        z[14] = x[8]  # tilt y

        return z

    def _jacobian_H(self, x: np.ndarray) -> np.ndarray:
        """
        测量雅可比矩阵

        H = ∂h/∂x
        """
        H = np.zeros((self.m, self.n))

        # 流量
        H[0, 0] = 1.0
        H[1, 1] = 1.0

        # 压力
        H[2, 2] = 1.0
        H[3, 2] = 1.0
        H[4, 3] = 1.0
        H[5, 3] = 1.0

        # 空腔水位
        H[6, 4] = 1.0

        # DAS
        H[7, 5] = 1.0
        H[8, 6] = 1.0

        # DTS
        H[9, 6] = 10.0

        # MEMS
        H[10, 9] = 1.0
        H[11, 9] = 1.0
        H[12, 9] = 1.0
        H[13, 8] = 1.0
        H[14, 8] = 1.0

        return H

    def predict(self, dt: float) -> FusedState:
        """
        预测步骤

        Args:
            dt: 时间步长 (s)

        Returns:
            预测状态
        """
        # 状态预测
        self.x = self._state_transition(self.x, dt)

        # 协方差预测
        F = self._jacobian_F(self.x, dt)
        self.P = F @ self.P @ F.T + self.Q

        return self._to_fused_state()

    def update(self, measurements: Dict[str, Any]) -> FusedState:
        """
        更新步骤

        Args:
            measurements: 传感器测量数据字典

        Returns:
            更新后状态
        """
        # 构造测量向量
        z = self._parse_measurements(measurements)

        # 测量预测
        z_pred = self._measurement_function(self.x)

        # 创新向量
        y = z - z_pred

        # 测量雅可比
        H = self._jacobian_H(self.x)

        # 创新协方差
        S = H @ self.P @ H.T + self.R

        # 卡尔曼增益
        K = self.P @ H.T @ inv(S)

        # 状态更新
        self.x = self.x + K @ y

        # 协方差更新 (Joseph形式，数值稳定)
        I_KH = np.eye(self.n) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        # 确保正定性
        self.P = (self.P + self.P.T) / 2
        self.P += np.eye(self.n) * 1e-10

        return self._to_fused_state()

    def _parse_measurements(self, measurements: Dict[str, Any]) -> np.ndarray:
        """解析测量数据"""
        z = np.zeros(self.m)

        # 流量计
        if 'flow_1' in measurements:
            z[0] = measurements['flow_1']
        if 'flow_2' in measurements:
            z[1] = measurements['flow_2']

        # 压力计
        if 'pressure' in measurements:
            pressures = measurements['pressure']
            z[2:6] = pressures[:4] if len(pressures) >= 4 else [pressures[0]] * 4

        # 空腔水位
        if 'cavity_level' in measurements:
            z[6] = measurements['cavity_level']

        # DAS
        if 'das' in measurements:
            das = measurements['das']
            z[7] = das.get('leak_position', 0)
            z[8] = das.get('leak_intensity', 0)

        # DTS
        if 'dts' in measurements:
            z[9] = measurements['dts'].get('anomaly_magnitude', 0)

        # MEMS
        if 'mems' in measurements:
            mems = measurements['mems']
            z[10:13] = mems.get('acceleration', [0, 0, 0])
            z[13:15] = mems.get('tilt', [0, 0])

        return z

    def _to_fused_state(self) -> FusedState:
        """转换为融合状态对象"""
        return FusedState(
            timestamp=0.0,
            Q1=self.x[0],
            Q2=self.x[1],
            P_inlet=self.x[2],
            P_outlet=self.x[3],
            cavity_level=self.x[4],
            leak_detected=self.x[6] > 0.01,
            leak_position=self.x[5],
            leak_rate=self.x[6],
            settlement=self.x[7],
            tilt=self.x[8],
            vibration_level=self.x[9],
            covariance=self.P.copy(),
            confidence=self._compute_confidence()
        )

    def _compute_confidence(self) -> float:
        """计算估计置信度"""
        # 基于协方差迹
        trace = np.trace(self.P)
        confidence = np.exp(-trace / 10)
        return np.clip(confidence, 0.5, 0.99)

    def fuse(
        self,
        measurements: Dict[str, Any],
        dt: float
    ) -> FusedState:
        """
        执行完整的融合过程

        Args:
            measurements: 传感器测量数据
            dt: 时间步长

        Returns:
            融合后状态
        """
        # 预测
        self.predict(dt)

        # 更新
        return self.update(measurements)


class OutlierDetector:
    """
    异常数据检测器

    基于马氏距离检测离群点
    """

    def __init__(self, threshold: float = 3.0):
        """
        初始化检测器

        Args:
            threshold: 马氏距离阈值
        """
        self.threshold = threshold
        self.history_mean = None
        self.history_cov = None
        self.history_buffer: List[np.ndarray] = []
        self.buffer_size = 100

    def update_statistics(self, data: np.ndarray):
        """更新历史统计量"""
        self.history_buffer.append(data)
        if len(self.history_buffer) > self.buffer_size:
            self.history_buffer.pop(0)

        if len(self.history_buffer) >= 10:
            data_matrix = np.array(self.history_buffer)
            self.history_mean = np.mean(data_matrix, axis=0)
            self.history_cov = np.cov(data_matrix.T) + np.eye(len(data)) * 1e-6

    def is_outlier(self, data: np.ndarray) -> Tuple[bool, float]:
        """
        检测是否为异常数据

        Args:
            data: 测量数据

        Returns:
            是否异常, 马氏距离
        """
        if self.history_mean is None or self.history_cov is None:
            return False, 0.0

        # 计算马氏距离
        diff = data - self.history_mean
        try:
            cov_inv = inv(self.history_cov)
            mahal_dist = np.sqrt(diff @ cov_inv @ diff)
        except np.linalg.LinAlgError:
            mahal_dist = np.linalg.norm(diff)

        is_outlier = mahal_dist > self.threshold
        return is_outlier, mahal_dist
