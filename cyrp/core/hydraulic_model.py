"""
Hydraulic Model for Yellow River Crossing Inverted Siphon.
穿黄倒虹吸水力学模型

包含:
- 圣维南方程 (Saint-Venant Equations)
- Preissmann狭缝模型 (气液两相流)
- 水锤理论 (Water Hammer)
- 摩阻损失模型
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from enum import Enum
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


class FlowRegime(Enum):
    """流态类型"""
    PRESSURIZED = "pressurized"  # 有压流
    FREE_SURFACE = "free_surface"  # 明流
    TWO_PHASE = "two_phase"  # 气液两相流
    EMPTY = "empty"  # 空管


@dataclass
class HydraulicState:
    """
    水力状态向量

    包含隧洞系统的完整水力状态信息
    """
    # 时间戳
    time: float = 0.0

    # 流量状态 (m³/s)
    Q1: float = 132.5  # 1#洞流量
    Q2: float = 132.5  # 2#洞流量

    # 水位/压力状态 (m)
    H_inlet: float = 106.05  # 进口水位
    H_outlet: float = 104.79  # 出口水位
    H_junction: float = 105.0  # 汇流处水头

    # 沿程水头分布 (m) - 离散节点
    H_profile_1: np.ndarray = field(default_factory=lambda: np.zeros(100))
    H_profile_2: np.ndarray = field(default_factory=lambda: np.zeros(100))

    # 流速分布 (m/s)
    V1: float = 3.44  # 1#洞平均流速
    V2: float = 3.44  # 2#洞平均流速

    # 压力分布 (Pa)
    P_max: float = 6.0e5  # 最大内压
    P_min: float = 1.0e5  # 最小内压

    # 水锤压力 (Pa)
    P_hammer: float = 0.0  # 水锤超压

    # 气液状态
    flow_regime_1: FlowRegime = FlowRegime.PRESSURIZED
    flow_regime_2: FlowRegime = FlowRegime.PRESSURIZED
    air_pocket_volume: float = 0.0  # 气囊体积 (m³)
    water_level_filling: float = 7.0  # 充水水位 (m)

    # 环形空腔状态
    cavity_water_level: float = 0.0  # 空腔水位 (m)
    cavity_pressure: float = 1.0e5  # 空腔压力 (Pa)

    # 损失
    head_loss_friction: float = 0.0  # 沿程损失 (m)
    head_loss_local: float = 0.0  # 局部损失 (m)

    @property
    def total_flow(self) -> float:
        """总流量"""
        return self.Q1 + self.Q2

    @property
    def flow_imbalance(self) -> float:
        """流量不平衡度"""
        if self.total_flow == 0:
            return 0.0
        return abs(self.Q1 - self.Q2) / self.total_flow

    @property
    def total_head_loss(self) -> float:
        """总水头损失"""
        return self.head_loss_friction + self.head_loss_local

    def to_vector(self) -> np.ndarray:
        """转换为状态向量"""
        return np.array([
            self.Q1, self.Q2, self.H_inlet, self.H_outlet,
            self.V1, self.V2, self.P_max, self.P_min,
            self.cavity_water_level, self.air_pocket_volume
        ])

    @classmethod
    def from_vector(cls, x: np.ndarray, time: float = 0.0) -> 'HydraulicState':
        """从状态向量恢复"""
        return cls(
            time=time,
            Q1=x[0], Q2=x[1],
            H_inlet=x[2], H_outlet=x[3],
            V1=x[4], V2=x[5],
            P_max=x[6], P_min=x[7],
            cavity_water_level=x[8],
            air_pocket_volume=x[9]
        )


class HydraulicModel:
    """
    穿黄倒虹吸水力学模型

    实现:
    1. 有压管流动力学方程
    2. Preissmann狭缝法处理气液转换
    3. MOC特征线法计算水锤
    4. 多洞耦合水力学
    """

    # 物理常数
    G = 9.81  # 重力加速度 (m/s²)
    RHO = 1000.0  # 水密度 (kg/m³)
    WAVE_SPEED = 1000.0  # 压力波速 (m/s)

    def __init__(
        self,
        length: float = 4250.0,
        diameter: float = 7.0,
        manning_n: float = 0.014,
        num_nodes: int = 100
    ):
        """
        初始化水力模型

        Args:
            length: 隧洞长度 (m)
            diameter: 隧洞内径 (m)
            manning_n: 曼宁糙率系数
            num_nodes: 离散节点数
        """
        self.L = length
        self.D = diameter
        self.n = manning_n
        self.num_nodes = num_nodes

        # 几何计算
        self.A = np.pi * (diameter / 2) ** 2  # 断面积
        self.P = np.pi * diameter  # 湿周
        self.R = self.A / self.P  # 水力半径

        # 空间离散
        self.dx = length / (num_nodes - 1)
        self.x = np.linspace(0, length, num_nodes)

        # 时间步长 (CFL条件)
        self.dt_max = self.dx / self.WAVE_SPEED

        # Preissmann狭缝参数
        self.slot_width = self.G * self.A / (self.WAVE_SPEED ** 2)

        # 局部损失系数
        self.xi_inlet = 0.1  # 进口损失系数
        self.xi_outlet = 0.2  # 出口损失系数
        self.xi_bend = 0.05  # 弯道损失系数

    def compute_friction_loss(self, Q: float, regime: FlowRegime = FlowRegime.PRESSURIZED) -> float:
        """
        计算沿程摩擦损失 (曼宁公式)

        h_f = n² Q |Q| L / (A² R^(4/3))

        Args:
            Q: 流量 (m³/s)
            regime: 流态

        Returns:
            沿程损失水头 (m)
        """
        if regime == FlowRegime.EMPTY or abs(Q) < 1e-6:
            return 0.0

        h_f = (self.n ** 2 * Q * abs(Q) * self.L) / (self.A ** 2 * self.R ** (4 / 3))
        return h_f

    def compute_local_loss(self, Q: float) -> float:
        """
        计算局部损失

        h_j = ξ * v² / (2g) = ξ * Q² / (2g * A²)

        Args:
            Q: 流量 (m³/s)

        Returns:
            局部损失水头 (m)
        """
        xi_total = self.xi_inlet + self.xi_outlet + self.xi_bend
        h_j = xi_total * Q ** 2 / (2 * self.G * self.A ** 2)
        return h_j

    def compute_velocity(self, Q: float) -> float:
        """计算断面平均流速"""
        return Q / self.A

    def compute_pressure(self, H: float, z: float) -> float:
        """
        计算静水压力

        P = ρg(H - z)

        Args:
            H: 测压管水头 (m)
            z: 高程 (m)

        Returns:
            压力 (Pa)
        """
        return self.RHO * self.G * (H - z)

    def pressurized_flow_ode(
        self,
        t: float,
        state: np.ndarray,
        H_up: float,
        H_down: float
    ) -> np.ndarray:
        """
        有压管流常微分方程

        dQ/dt = (g*A/L) * (H_up - H_down - h_f - h_j)

        Args:
            t: 时间 (s)
            state: [Q] 状态向量
            H_up: 上游水头 (m)
            H_down: 下游水头 (m)

        Returns:
            状态导数
        """
        Q = state[0]

        # 计算损失
        h_f = self.compute_friction_loss(Q)
        h_j = self.compute_local_loss(Q)

        # 动量方程
        dQ_dt = (self.G * self.A / self.L) * (H_up - H_down - h_f - h_j)

        return np.array([dQ_dt])

    def simulate_pressurized_flow(
        self,
        initial_Q: float,
        H_up: float,
        H_down: float,
        duration: float,
        dt: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        模拟有压管流动态过程

        Args:
            initial_Q: 初始流量 (m³/s)
            H_up: 上游水头 (m)
            H_down: 下游水头 (m)
            duration: 模拟时长 (s)
            dt: 输出时间步 (s)

        Returns:
            时间序列, 流量序列
        """
        t_span = (0, duration)
        t_eval = np.arange(0, duration, dt)

        sol = solve_ivp(
            lambda t, y: self.pressurized_flow_ode(t, y, H_up, H_down),
            t_span,
            [initial_Q],
            t_eval=t_eval,
            method='RK45'
        )

        return sol.t, sol.y[0]

    def preissmann_slot_model(
        self,
        t: float,
        state: np.ndarray,
        gate_opening: float,
        downstream_H: float
    ) -> np.ndarray:
        """
        Preissmann狭缝模型 - 处理气液两相流

        用于模拟充水/排空过程中的气液界面运动

        Args:
            t: 时间 (s)
            state: [Q, h_water, V_air] - 流量、水位、气囊体积
            gate_opening: 闸门开度 (0-1)
            downstream_H: 下游水头 (m)

        Returns:
            状态导数
        """
        Q, h_water, V_air = state

        # 过水面积 (水位低于管顶时为部分充满)
        if h_water >= self.D:
            A_wet = self.A
            regime = FlowRegime.PRESSURIZED
        else:
            theta = 2 * np.arccos(1 - 2 * h_water / self.D)
            A_wet = self.D ** 2 / 8 * (theta - np.sin(theta))
            regime = FlowRegime.TWO_PHASE

        # 等效波速
        if regime == FlowRegime.PRESSURIZED:
            c = self.WAVE_SPEED
        else:
            c = np.sqrt(self.G * A_wet / self.slot_width)

        # 气囊压力 (理想气体)
        P_atm = 101325.0
        if V_air > 0:
            P_air = P_atm * (V_air + 100) / (V_air + 1e-6)  # 避免除零
        else:
            P_air = P_atm

        # 流量方程
        H_up = gate_opening * 10.0  # 简化的上游水头
        h_f = self.compute_friction_loss(Q, regime)
        h_air = (P_air - P_atm) / (self.RHO * self.G)

        dQ_dt = (self.G * A_wet / self.L) * (H_up - downstream_H - h_f - h_air)

        # 水位变化 (质量守恒)
        dh_dt = Q / (self.L * self.slot_width) if h_water < self.D else 0

        # 气囊体积变化
        dV_air_dt = -Q if V_air > 0 else 0

        return np.array([dQ_dt, dh_dt, dV_air_dt])

    def water_hammer_moc(
        self,
        initial_state: HydraulicState,
        gate_closure_time: float,
        gate_closure_rate: float,
        simulation_time: float
    ) -> List[HydraulicState]:
        """
        特征线法 (MOC) 计算水锤

        基于特征线方程:
        C+: dH/dt + (a/gA) * dQ/dt + f*Q|Q|/(2gDA²) = 0
        C-: dH/dt - (a/gA) * dQ/dt - f*Q|Q|/(2gDA²) = 0

        Args:
            initial_state: 初始水力状态
            gate_closure_time: 闸门关闭起始时间 (s)
            gate_closure_rate: 闸门关闭速率 (1/s)
            simulation_time: 模拟总时间 (s)

        Returns:
            水力状态时间序列
        """
        # 时空离散
        dt = self.dx / self.WAVE_SPEED  # CFL条件
        num_t = int(simulation_time / dt)
        a = self.WAVE_SPEED
        f = 0.02  # Darcy摩擦因子

        # 初始化场
        H = np.zeros((num_t, self.num_nodes))
        Q = np.zeros((num_t, self.num_nodes))

        # 初始条件
        H[0, :] = np.linspace(initial_state.H_inlet, initial_state.H_outlet, self.num_nodes)
        Q[0, :] = initial_state.Q1

        states = [initial_state]

        # 时间推进
        for n in range(num_t - 1):
            t = n * dt

            # 内部节点 - 特征线法
            for i in range(1, self.num_nodes - 1):
                # C+ 特征线
                Cp = H[n, i - 1] + (a / (self.G * self.A)) * Q[n, i - 1] - \
                     f * dt * Q[n, i - 1] * abs(Q[n, i - 1]) / (2 * self.G * self.D * self.A ** 2)

                # C- 特征线
                Cm = H[n, i + 1] - (a / (self.G * self.A)) * Q[n, i + 1] + \
                     f * dt * Q[n, i + 1] * abs(Q[n, i + 1]) / (2 * self.G * self.D * self.A ** 2)

                B = a / (self.G * self.A)
                H[n + 1, i] = (Cp + Cm) / 2
                Q[n + 1, i] = (Cp - Cm) / (2 * B)

            # 上游边界 (定水头)
            H[n + 1, 0] = initial_state.H_inlet

            # 下游边界 (闸门)
            if t >= gate_closure_time:
                closure_fraction = min(1.0, (t - gate_closure_time) * gate_closure_rate)
                gate_factor = 1.0 - closure_fraction
            else:
                gate_factor = 1.0

            # 闸门方程
            Q[n + 1, -1] = gate_factor * Q[0, -1]
            H[n + 1, -1] = H[n + 1, -2] + (a / (self.G * self.A)) * (Q[n + 1, -2] - Q[n + 1, -1])

            # 记录状态
            if n % 10 == 0:
                state = HydraulicState(
                    time=t,
                    Q1=Q[n + 1, self.num_nodes // 2],
                    Q2=Q[n + 1, self.num_nodes // 2],
                    H_inlet=H[n + 1, 0],
                    H_outlet=H[n + 1, -1],
                    P_max=self.compute_pressure(np.max(H[n + 1, :]), 85.0),
                    P_min=self.compute_pressure(np.min(H[n + 1, :]), 85.0),
                    P_hammer=self.compute_pressure(np.max(H[n + 1, :]) - np.max(H[0, :]), 0)
                )
                states.append(state)

        return states

    def dual_tunnel_coupled_model(
        self,
        t: float,
        state: np.ndarray,
        u: np.ndarray
    ) -> np.ndarray:
        """
        双洞耦合水力学模型

        考虑:
        1. 两条隧洞的并联水力特性
        2. 汇流处的动量交换
        3. 闸门控制输入

        Args:
            t: 时间 (s)
            state: [Q1, Q2, H_in, H_out, H_junc] 状态向量
            u: [e1, e2] 闸门开度 (0-1)

        Returns:
            状态导数
        """
        Q1, Q2, H_in, H_out, H_junc = state
        e1, e2 = u

        # 有效过水面积
        A1_eff = e1 * self.A
        A2_eff = e2 * self.A

        # 各洞水头损失
        h_f1 = self.compute_friction_loss(Q1)
        h_f2 = self.compute_friction_loss(Q2)
        h_j1 = self.compute_local_loss(Q1)
        h_j2 = self.compute_local_loss(Q2)

        # 动量方程
        if A1_eff > 1e-6:
            dQ1_dt = (self.G * A1_eff / self.L) * (H_in - H_junc - h_f1 - h_j1)
        else:
            dQ1_dt = -Q1 / 10.0  # 闸门关闭时的衰减

        if A2_eff > 1e-6:
            dQ2_dt = (self.G * A2_eff / self.L) * (H_in - H_junc - h_f2 - h_j2)
        else:
            dQ2_dt = -Q2 / 10.0

        # 汇流点质量守恒
        Q_out = Q1 + Q2
        h_f_out = 0.01 * Q_out ** 2  # 简化的出口损失
        dH_junc_dt = (Q1 + Q2 - Q_out) / (self.A * 10)  # 假设汇流池长度10m

        # 进出口水位变化 (假设由外部控制)
        dH_in_dt = 0.0
        dH_out_dt = 0.0

        return np.array([dQ1_dt, dQ2_dt, dH_in_dt, dH_out_dt, dH_junc_dt])

    def step(
        self,
        state: HydraulicState,
        control: np.ndarray,
        dt: float
    ) -> HydraulicState:
        """
        单步仿真

        Args:
            state: 当前状态
            control: [e1, e2] 闸门控制
            dt: 时间步长 (s)

        Returns:
            新状态
        """
        x = np.array([state.Q1, state.Q2, state.H_inlet, state.H_outlet, state.H_junction])

        # RK4积分
        k1 = self.dual_tunnel_coupled_model(state.time, x, control)
        k2 = self.dual_tunnel_coupled_model(state.time + dt / 2, x + dt / 2 * k1, control)
        k3 = self.dual_tunnel_coupled_model(state.time + dt / 2, x + dt / 2 * k2, control)
        k4 = self.dual_tunnel_coupled_model(state.time + dt, x + dt * k3, control)

        x_new = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # 构造新状态
        new_state = HydraulicState(
            time=state.time + dt,
            Q1=max(0, x_new[0]),
            Q2=max(0, x_new[1]),
            H_inlet=x_new[2],
            H_outlet=x_new[3],
            H_junction=x_new[4],
            V1=self.compute_velocity(x_new[0]),
            V2=self.compute_velocity(x_new[1]),
            head_loss_friction=self.compute_friction_loss(x_new[0]) + self.compute_friction_loss(x_new[1]),
            head_loss_local=self.compute_local_loss(x_new[0]) + self.compute_local_loss(x_new[1])
        )

        # 更新压力
        new_state.P_max = self.compute_pressure(max(x_new[2], x_new[4]), 85.0)
        new_state.P_min = self.compute_pressure(min(x_new[3], x_new[4]), 90.0)

        return new_state

    def get_steady_state(self, H_up: float, H_down: float) -> HydraulicState:
        """
        计算稳态工况

        Args:
            H_up: 上游水头 (m)
            H_down: 下游水头 (m)

        Returns:
            稳态水力状态
        """
        # 牛顿迭代求解稳态流量
        dH = H_up - H_down
        Q = 100.0  # 初始猜测

        for _ in range(50):
            h_f = self.compute_friction_loss(Q)
            h_j = self.compute_local_loss(Q)
            h_total = h_f + h_j

            # 残差
            residual = dH - h_total

            # 雅可比
            dh_dQ = (2 * self.n ** 2 * Q * self.L) / (self.A ** 2 * self.R ** (4 / 3)) + \
                    (self.xi_inlet + self.xi_outlet + self.xi_bend) * Q / (self.G * self.A ** 2)

            if abs(dh_dQ) < 1e-10:
                break

            Q = Q + residual / dh_dQ

            if abs(residual) < 1e-6:
                break

        Q = max(0, Q)
        V = self.compute_velocity(Q)

        return HydraulicState(
            Q1=Q / 2,
            Q2=Q / 2,
            H_inlet=H_up,
            H_outlet=H_down,
            V1=V,
            V2=V,
            head_loss_friction=self.compute_friction_loss(Q),
            head_loss_local=self.compute_local_loss(Q)
        )
