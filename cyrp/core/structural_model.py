"""
Structural Dynamics Model for Yellow River Crossing Tunnel.
穿黄隧洞结构动力学模型

包含:
- 衬砌应力应变分析
- 土-结构相互作用
- 地震响应分析
- 渗漏/失效模型
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from enum import Enum
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm


class StructuralCondition(Enum):
    """结构状态等级"""
    HEALTHY = "healthy"  # 健康
    DEGRADED = "degraded"  # 退化
    DAMAGED = "damaged"  # 损伤
    CRITICAL = "critical"  # 危急
    FAILED = "failed"  # 失效


class FailureMode(Enum):
    """失效模式"""
    NONE = "none"
    INNER_LEAKAGE = "inner_leakage"  # 内衬渗漏 (S5-A)
    OUTER_INTRUSION = "outer_intrusion"  # 外衬入侵 (S5-B)
    JOINT_DISLOCATION = "joint_dislocation"  # 接头错位 (S5-C)
    BUCKLING = "buckling"  # 屈曲失稳
    FLOTATION = "flotation"  # 上浮
    LIQUEFACTION = "liquefaction"  # 液化 (S6-A)


@dataclass
class StructuralState:
    """
    结构状态向量

    描述隧洞结构的力学状态
    """
    time: float = 0.0

    # 应力状态 (Pa)
    stress_hoop_inner: float = 0.0  # 内衬环向应力
    stress_hoop_outer: float = 0.0  # 外衬环向应力
    stress_axial: float = 0.0  # 轴向应力
    stress_radial: float = 0.0  # 径向应力

    # 变形状态 (m)
    displacement_radial: float = 0.0  # 径向位移
    displacement_axial: float = 0.0  # 轴向位移
    settlement: float = 0.0  # 沉降量
    heave: float = 0.0  # 上浮量

    # 姿态状态 (rad)
    tilt_angle_x: float = 0.0  # X轴倾斜
    tilt_angle_y: float = 0.0  # Y轴倾斜
    rotation: float = 0.0  # 扭转

    # 接头状态
    joint_opening: float = 0.0  # 接头张开量 (m)
    joint_offset: float = 0.0  # 接头错位量 (m)

    # 渗漏状态
    leakage_rate: float = 0.0  # 渗漏率 (m³/s)
    leakage_location: float = 0.0  # 渗漏位置 (m, 沿程)

    # 土体状态
    pore_pressure: float = 0.0  # 孔隙水压力 (Pa)
    effective_stress: float = 0.0  # 有效应力 (Pa)
    liquefaction_index: float = 0.0  # 液化指数 (0-1)

    # 健康评估
    condition: StructuralCondition = StructuralCondition.HEALTHY
    failure_mode: FailureMode = FailureMode.NONE
    safety_factor: float = 2.0  # 安全系数

    def to_vector(self) -> np.ndarray:
        """转换为状态向量"""
        return np.array([
            self.stress_hoop_inner,
            self.stress_hoop_outer,
            self.displacement_radial,
            self.settlement,
            self.heave,
            self.tilt_angle_x,
            self.tilt_angle_y,
            self.joint_opening,
            self.leakage_rate,
            self.liquefaction_index
        ])


class StructuralModel:
    """
    穿黄隧洞结构动力学模型

    包含:
    1. 衬砌力学分析
    2. 土-结构相互作用
    3. 地震动力响应
    4. 渗漏演化模型
    """

    # 材料常数
    E_CONCRETE = 3.0e10  # 混凝土弹性模量 (Pa)
    E_STEEL = 2.1e11  # 钢材弹性模量 (Pa)
    NU = 0.2  # 泊松比
    RHO_CONCRETE = 2500.0  # 混凝土密度 (kg/m³)
    RHO_WATER = 1000.0  # 水密度 (kg/m³)
    G = 9.81  # 重力加速度 (m/s²)

    def __init__(
        self,
        inner_diameter: float = 7.0,
        outer_diameter: float = 9.03,
        inner_thickness: float = 0.45,
        outer_thickness: float = 0.55,
        length: float = 4250.0,
        segment_length: float = 2.0,
        burial_depth: float = 70.0
    ):
        """
        初始化结构模型

        Args:
            inner_diameter: 内径 (m)
            outer_diameter: 外径 (m)
            inner_thickness: 内衬厚度 (m)
            outer_thickness: 外衬厚度 (m)
            length: 隧洞长度 (m)
            segment_length: 管节长度 (m)
            burial_depth: 埋深 (m)
        """
        self.D_in = inner_diameter
        self.D_out = outer_diameter
        self.t_in = inner_thickness
        self.t_out = outer_thickness
        self.L = length
        self.L_seg = segment_length
        self.H_burial = burial_depth

        # 计算几何参数
        self.r_in = inner_diameter / 2
        self.r_out = outer_diameter / 2
        self.num_segments = int(length / segment_length)

        # 截面特性
        self.A_inner = np.pi * ((self.r_in + self.t_in) ** 2 - self.r_in ** 2)
        self.I_inner = np.pi / 4 * ((self.r_in + self.t_in) ** 4 - self.r_in ** 4)

        # 土体参数
        self.k_soil = 5.0e7  # 地基反力系数 (N/m³)
        self.c_soil = 1.0e5  # 土体阻尼系数 (Ns/m³)

        # 初始化接头刚度矩阵
        self._init_joint_stiffness()

    def _init_joint_stiffness(self):
        """初始化接头刚度矩阵"""
        # 接头轴向刚度
        self.k_axial = 1.0e9  # (N/m)
        # 接头剪切刚度
        self.k_shear = 5.0e8  # (N/m)
        # 接头弯曲刚度
        self.k_bending = 1.0e8  # (Nm/rad)

    def compute_hoop_stress(
        self,
        P_internal: float,
        P_external: float,
        layer: str = "inner"
    ) -> float:
        """
        计算环向应力 (厚壁圆筒公式)

        σ_θ = (P_i * r_i² - P_o * r_o²) / (r_o² - r_i²) +
              (P_i - P_o) * r_i² * r_o² / (r² * (r_o² - r_i²))

        Args:
            P_internal: 内压 (Pa)
            P_external: 外压 (Pa)
            layer: "inner" 或 "outer"

        Returns:
            环向应力 (Pa)
        """
        if layer == "inner":
            r_i = self.r_in
            r_o = self.r_in + self.t_in
            r = (r_i + r_o) / 2  # 中面半径
        else:
            r_i = self.r_out - self.t_out
            r_o = self.r_out
            r = (r_i + r_o) / 2

        # Lame公式
        term1 = (P_internal * r_i ** 2 - P_external * r_o ** 2) / (r_o ** 2 - r_i ** 2)
        term2 = (P_internal - P_external) * r_i ** 2 * r_o ** 2 / (r ** 2 * (r_o ** 2 - r_i ** 2))

        sigma_theta = term1 + term2
        return sigma_theta

    def compute_radial_displacement(
        self,
        P_internal: float,
        P_external: float
    ) -> float:
        """
        计算径向位移

        u_r = r / E * [(1-ν)σ_θ - ν*σ_r]

        Args:
            P_internal: 内压 (Pa)
            P_external: 外压 (Pa)

        Returns:
            径向位移 (m)
        """
        sigma_theta = self.compute_hoop_stress(P_internal, P_external, "inner")
        sigma_r = -(P_internal + P_external) / 2  # 近似

        u_r = self.r_in / self.E_CONCRETE * ((1 - self.NU) * sigma_theta - self.NU * sigma_r)
        return u_r

    def compute_buckling_pressure(self) -> float:
        """
        计算屈曲临界外压

        P_cr = 2 * E * (t/r)³ / (1 - ν²)

        Returns:
            临界屈曲压力 (Pa)
        """
        t = self.t_in
        r = self.r_in + t / 2

        P_cr = 2 * self.E_CONCRETE * (t / r) ** 3 / (1 - self.NU ** 2)
        return P_cr

    def compute_flotation_force(
        self,
        liquefaction_ratio: float = 0.0
    ) -> float:
        """
        计算上浮力

        F_buoy = ρ_soil * g * V_tunnel * λ

        Args:
            liquefaction_ratio: 液化程度 (0-1)

        Returns:
            上浮力 (N/m)
        """
        # 隧洞截面积
        A_tunnel = np.pi * self.r_out ** 2

        # 液化土体容重增大
        rho_soil = 1800 + 400 * liquefaction_ratio  # kg/m³

        # 浮力
        F_buoy = rho_soil * self.G * A_tunnel * liquefaction_ratio

        return F_buoy

    def compute_anti_float_weight(
        self,
        water_fill_ratio: float = 1.0
    ) -> float:
        """
        计算抗浮重量

        W_anti = W_structure + W_water

        Args:
            water_fill_ratio: 充水率 (0-1)

        Returns:
            抗浮重量 (N/m)
        """
        # 结构自重
        A_struct = np.pi * (self.r_out ** 2 - self.r_in ** 2)
        W_struct = self.RHO_CONCRETE * self.G * A_struct

        # 水重
        A_water = np.pi * self.r_in ** 2 * water_fill_ratio
        W_water = self.RHO_WATER * self.G * A_water

        return W_struct + W_water

    def leakage_evolution_model(
        self,
        t: float,
        state: np.ndarray,
        P_diff: float
    ) -> np.ndarray:
        """
        渗漏演化模型

        dq/dt = α * P_diff * A_crack
        dA/dt = β * q (裂缝扩展)

        Args:
            t: 时间 (s)
            state: [q, A_crack] 渗漏量和裂缝面积
            P_diff: 压差 (Pa)

        Returns:
            状态导数
        """
        q, A_crack = state

        # 渗漏系数
        alpha = 1e-10  # m²/(Pa·s)
        # 裂缝扩展系数
        beta = 1e-8  # 1/s

        # 渗漏流量
        dq_dt = alpha * P_diff * A_crack

        # 裂缝扩展 (仅当渗漏持续时)
        if q > 0:
            dA_dt = beta * q
        else:
            dA_dt = 0

        return np.array([dq_dt, dA_dt])

    def seismic_response_model(
        self,
        t: float,
        state: np.ndarray,
        ground_acceleration: float
    ) -> np.ndarray:
        """
        地震动力响应模型 (简化单自由度)

        m * ü + c * u̇ + k * u = -m * a_g

        Args:
            t: 时间 (s)
            state: [u, v] 位移和速度
            ground_acceleration: 地面加速度 (m/s²)

        Returns:
            状态导数
        """
        u, v = state

        # 等效参数
        m = self.RHO_CONCRETE * self.A_inner  # 质量/长度
        k = self.k_soil * np.pi * self.D_out  # 刚度/长度
        c = self.c_soil * np.pi * self.D_out  # 阻尼/长度

        # 阻尼比
        zeta = c / (2 * np.sqrt(k * m))

        # 运动方程
        du_dt = v
        dv_dt = -ground_acceleration - 2 * zeta * np.sqrt(k / m) * v - (k / m) * u

        return np.array([du_dt, dv_dt])

    def liquefaction_assessment(
        self,
        pga: float,
        magnitude: float = 7.0,
        depth: float = 10.0
    ) -> float:
        """
        液化势评估 (简化方法)

        基于 CSR 和 CRR 比值

        Args:
            pga: 峰值地面加速度 (g)
            magnitude: 震级
            depth: 评估深度 (m)

        Returns:
            液化指数 (0-1)
        """
        # 循环应力比 CSR
        sigma_v = 1800 * self.G * depth  # 总应力
        sigma_v_eff = sigma_v - 1000 * self.G * depth * 0.8  # 有效应力
        rd = 1 - 0.015 * depth  # 深度折减系数
        CSR = 0.65 * pga * (sigma_v / sigma_v_eff) * rd

        # 循环阻力比 CRR (假设中密砂)
        N_60 = 15  # 标准贯入击数
        CRR = 0.05 + 0.01 * N_60

        # 震级修正
        MSF = 10 ** (2.24) / magnitude ** 2.56

        # 液化安全系数
        FS_liq = CRR * MSF / CSR

        # 转换为液化指数
        if FS_liq >= 1.5:
            liq_index = 0.0
        elif FS_liq <= 0.5:
            liq_index = 1.0
        else:
            liq_index = 1.0 - (FS_liq - 0.5)

        return liq_index

    def joint_mechanics(
        self,
        axial_force: float,
        shear_force: float,
        moment: float
    ) -> Tuple[float, float, float]:
        """
        计算接头变形

        Args:
            axial_force: 轴力 (N)
            shear_force: 剪力 (N)
            moment: 弯矩 (Nm)

        Returns:
            轴向位移, 剪切位移, 转角
        """
        delta_axial = axial_force / self.k_axial
        delta_shear = shear_force / self.k_shear
        theta = moment / self.k_bending

        return delta_axial, delta_shear, theta

    def evaluate_safety_factor(
        self,
        P_internal: float,
        P_external: float,
        seismic_load: float = 0.0
    ) -> Tuple[float, FailureMode]:
        """
        评估安全系数

        Args:
            P_internal: 内压 (Pa)
            P_external: 外压 (Pa)
            seismic_load: 地震荷载 (Pa)

        Returns:
            安全系数, 主控失效模式
        """
        failure_modes = {}

        # 内压失效
        sigma_allow = 2.0e7  # 允许应力 (Pa)
        sigma_hoop = self.compute_hoop_stress(P_internal, P_external, "inner")
        SF_tension = sigma_allow / max(abs(sigma_hoop), 1e-6)
        failure_modes[FailureMode.INNER_LEAKAGE] = SF_tension

        # 屈曲失效
        P_cr = self.compute_buckling_pressure()
        if P_external > P_internal:
            SF_buckling = P_cr / (P_external - P_internal)
        else:
            SF_buckling = 10.0
        failure_modes[FailureMode.BUCKLING] = SF_buckling

        # 上浮失效
        if seismic_load > 0:
            F_buoy = self.compute_flotation_force(0.5)
            W_anti = self.compute_anti_float_weight(1.0)
            SF_float = W_anti / max(F_buoy, 1e-6)
            failure_modes[FailureMode.FLOTATION] = SF_float
        else:
            failure_modes[FailureMode.FLOTATION] = 10.0

        # 取最小安全系数
        min_mode = min(failure_modes, key=failure_modes.get)
        min_sf = failure_modes[min_mode]

        return min_sf, min_mode

    def step(
        self,
        state: StructuralState,
        P_internal: float,
        P_external: float,
        ground_motion: float = 0.0,
        dt: float = 0.01
    ) -> StructuralState:
        """
        单步仿真

        Args:
            state: 当前结构状态
            P_internal: 内压 (Pa)
            P_external: 外压 (Pa)
            ground_motion: 地面运动加速度 (m/s²)
            dt: 时间步长 (s)

        Returns:
            新结构状态
        """
        new_state = StructuralState(time=state.time + dt)

        # 计算应力
        new_state.stress_hoop_inner = self.compute_hoop_stress(P_internal, P_external, "inner")
        new_state.stress_hoop_outer = self.compute_hoop_stress(P_internal, P_external, "outer")

        # 计算位移
        new_state.displacement_radial = self.compute_radial_displacement(P_internal, P_external)

        # 地震响应
        if abs(ground_motion) > 0.01:
            x0 = np.array([state.settlement, state.heave])
            sol = solve_ivp(
                lambda t, y: self.seismic_response_model(t, y, ground_motion),
                [0, dt],
                x0,
                method='RK45'
            )
            new_state.settlement = sol.y[0, -1]
            new_state.heave = sol.y[1, -1]

            # 液化评估
            new_state.liquefaction_index = self.liquefaction_assessment(
                abs(ground_motion) / self.G, 7.0, self.H_burial / 2
            )
        else:
            new_state.settlement = state.settlement
            new_state.heave = state.heave
            new_state.liquefaction_index = 0.0

        # 渗漏演化
        if state.leakage_rate > 0 or (P_internal - P_external) > 5e5:
            x0 = np.array([state.leakage_rate, 1e-4])  # 假设初始裂缝
            sol = solve_ivp(
                lambda t, y: self.leakage_evolution_model(t, y, P_internal - P_external),
                [0, dt],
                x0,
                method='RK45'
            )
            new_state.leakage_rate = sol.y[0, -1]

        # 安全评估
        new_state.safety_factor, new_state.failure_mode = self.evaluate_safety_factor(
            P_internal, P_external, ground_motion
        )

        # 状态分级
        if new_state.safety_factor >= 2.0:
            new_state.condition = StructuralCondition.HEALTHY
        elif new_state.safety_factor >= 1.5:
            new_state.condition = StructuralCondition.DEGRADED
        elif new_state.safety_factor >= 1.0:
            new_state.condition = StructuralCondition.DAMAGED
        elif new_state.safety_factor >= 0.8:
            new_state.condition = StructuralCondition.CRITICAL
        else:
            new_state.condition = StructuralCondition.FAILED

        return new_state
