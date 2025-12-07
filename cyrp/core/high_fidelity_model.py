"""
高精度本体仿真模型 - High-Fidelity Physical Simulation Model

实现精细化水力学、热力学、结构耦合仿真
Implements detailed hydraulics, thermodynamics, and structural coupling simulation
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import warnings


# 物理常数
GRAVITY = 9.81  # m/s²
WATER_DENSITY = 1000.0  # kg/m³
WATER_BULK_MODULUS = 2.2e9  # Pa
WATER_VISCOSITY = 1.0e-6  # m²/s (动力粘度)
WATER_SPECIFIC_HEAT = 4186.0  # J/(kg·K)
WATER_THERMAL_CONDUCTIVITY = 0.6  # W/(m·K)


@dataclass
class TunnelGeometry:
    """隧道几何参数"""
    length: float = 4250.0          # 隧道长度 (m)
    diameter: float = 7.0           # 内径 (m)
    wall_thickness: float = 0.6     # 壁厚 (m)
    slope: float = 0.0001           # 坡度
    roughness: float = 0.015        # 曼宁糙率

    # 断面特性
    @property
    def area(self) -> float:
        return np.pi * (self.diameter / 2) ** 2

    @property
    def wetted_perimeter(self) -> float:
        return np.pi * self.diameter

    @property
    def hydraulic_radius(self) -> float:
        return self.area / self.wetted_perimeter


@dataclass
class MaterialProperties:
    """材料属性"""
    # 混凝土
    concrete_E: float = 3.0e10      # 弹性模量 (Pa)
    concrete_nu: float = 0.2        # 泊松比
    concrete_density: float = 2500.0  # 密度 (kg/m³)
    concrete_alpha: float = 1.0e-5  # 热膨胀系数 (1/K)
    concrete_k: float = 1.8         # 导热系数 (W/(m·K))

    # 钢筋
    steel_E: float = 2.1e11         # 弹性模量 (Pa)
    steel_fy: float = 400e6         # 屈服强度 (Pa)
    steel_ratio: float = 0.015      # 配筋率


@dataclass
class SimulationState:
    """仿真状态"""
    time: float
    # 水力学状态
    water_level: np.ndarray         # 水位 (m)
    velocity: np.ndarray            # 流速 (m/s)
    pressure: np.ndarray            # 压力 (Pa)
    flow_rate: np.ndarray           # 流量 (m³/s)

    # 热力学状态
    water_temperature: np.ndarray   # 水温 (K)
    wall_temperature: np.ndarray    # 壁面温度 (K)

    # 结构状态
    stress: np.ndarray             # 应力 (Pa)
    strain: np.ndarray             # 应变
    displacement: np.ndarray       # 位移 (m)


class HighFidelityHydraulicModel:
    """高精度水力学模型"""

    def __init__(self, geometry: TunnelGeometry, n_nodes: int = 100):
        self.geom = geometry
        self.n_nodes = n_nodes
        self.dx = geometry.length / (n_nodes - 1)

        # 节点位置
        self.x = np.linspace(0, geometry.length, n_nodes)

        # 状态变量
        self.h = np.ones(n_nodes) * geometry.diameter * 0.9  # 水位
        self.Q = np.ones(n_nodes) * 265.0  # 流量
        self.A = np.ones(n_nodes) * geometry.area  # 断面积

        # 波速 (考虑管壁弹性)
        self.wave_speed = self._compute_wave_speed()

        # Preissmann狭槽参数
        self.slot_width = 0.01 * geometry.diameter  # 狭槽宽度
        self.transition_depth = 0.95 * geometry.diameter  # 过渡水深

    def _compute_wave_speed(self) -> float:
        """计算考虑管壁弹性的波速"""
        E = WATER_BULK_MODULUS
        rho = WATER_DENSITY
        c0 = np.sqrt(E / rho)

        # 考虑管壁弹性的修正 (Korteweg公式)
        D = self.geom.diameter
        e = self.geom.wall_thickness
        E_pipe = 3.0e10  # 混凝土弹性模量

        c = c0 / np.sqrt(1 + (E * D) / (E_pipe * e))
        return c

    def _compute_area(self, h: float) -> float:
        """计算过流面积 (圆形断面 + Preissmann狭槽)"""
        D = self.geom.diameter
        R = D / 2

        if h <= 0:
            return 0.0
        elif h <= D:
            # 圆形断面部分
            theta = 2 * np.arccos((R - h) / R)
            return R**2 * (theta - np.sin(theta)) / 2
        else:
            # 满管 + 狭槽
            A_full = np.pi * R**2
            A_slot = self.slot_width * (h - D)
            return A_full + A_slot

    def _compute_wetted_perimeter(self, h: float) -> float:
        """计算湿周"""
        D = self.geom.diameter
        R = D / 2

        if h <= 0:
            return 0.0
        elif h <= D:
            theta = 2 * np.arccos((R - h) / R)
            return R * theta
        else:
            return np.pi * D + 2 * (h - D)

    def step_saint_venant(self, dt: float, Q_upstream: float, h_downstream: float):
        """
        Saint-Venant方程显式求解

        ∂A/∂t + ∂Q/∂x = q (连续性方程)
        ∂Q/∂t + ∂(Q²/A)/∂x + gA∂h/∂x = gA(S₀ - Sf) (动量方程)
        """
        n = self.n_nodes
        h_new = self.h.copy()
        Q_new = self.Q.copy()

        # CFL条件检查
        max_velocity = np.max(np.abs(self.Q / self.A))
        cfl = (max_velocity + self.wave_speed) * dt / self.dx
        if cfl > 0.9:
            warnings.warn(f"CFL number {cfl:.2f} > 0.9, stability may be compromised")

        for i in range(1, n - 1):
            # 断面参数
            A = self.A[i]
            R = A / self._compute_wetted_perimeter(self.h[i])

            # 摩阻坡度 (曼宁公式)
            V = self.Q[i] / A if A > 0 else 0
            Sf = (self.geom.roughness ** 2 * V * abs(V)) / (R ** (4/3)) if R > 0 else 0

            # 连续性方程
            dQ_dx = (self.Q[i+1] - self.Q[i-1]) / (2 * self.dx)
            dA_dt = -dQ_dx

            # 更新水位 (假设棱柱形断面)
            B = self._compute_surface_width(self.h[i])
            if B > 0:
                h_new[i] = self.h[i] + dA_dt * dt / B

            # 动量方程
            Q2_A = self.Q ** 2 / (self.A + 1e-6)
            dQ2A_dx = (Q2_A[i+1] - Q2_A[i-1]) / (2 * self.dx)
            dh_dx = (self.h[i+1] - self.h[i-1]) / (2 * self.dx)

            dQ_dt = -dQ2A_dx - GRAVITY * A * dh_dx + GRAVITY * A * (self.geom.slope - Sf)
            Q_new[i] = self.Q[i] + dQ_dt * dt

        # 边界条件
        Q_new[0] = Q_upstream
        h_new[-1] = h_downstream

        # 上游水位 (能量关系)
        h_new[0] = h_new[1] + (Q_new[0]**2 - Q_new[1]**2) / (2 * GRAVITY * self.A[0]**2 + 1e-6)

        # 下游流量 (外推)
        Q_new[-1] = Q_new[-2]

        # 更新状态
        self.h = h_new
        self.Q = Q_new
        for i in range(n):
            self.A[i] = self._compute_area(self.h[i])

    def _compute_surface_width(self, h: float) -> float:
        """计算水面宽度"""
        D = self.geom.diameter
        R = D / 2

        if h <= 0:
            return 0.0
        elif h <= D:
            if abs(h - R) < R:
                return 2 * np.sqrt(R**2 - (R - h)**2)
            return D
        else:
            return self.slot_width

    def compute_water_hammer(self, dt: float, valve_closure_rate: float) -> np.ndarray:
        """
        水锤计算 (特征线法 MOC)

        Args:
            dt: 时间步长
            valve_closure_rate: 阀门关闭速率 (0-1)/s

        Returns:
            压力波分布
        """
        n = self.n_nodes
        c = self.wave_speed

        # 检查Courant条件
        courant = c * dt / self.dx
        if courant > 1.0:
            warnings.warn(f"Courant number {courant:.2f} > 1, reduce dt for MOC")

        # 头损和流量 (用H和V表示)
        H = self.h + self.Q**2 / (2 * GRAVITY * self.A**2 + 1e-6)  # 总水头
        V = self.Q / (self.A + 1e-6)

        H_new = H.copy()
        V_new = V.copy()

        # 特征线方程
        for i in range(1, n - 1):
            # 管道常数
            A = self.A[i]
            R = A / self._compute_wetted_perimeter(self.h[i])

            # 摩阻项
            f = 0.02  # Darcy摩阻系数
            J = f * V[i] * abs(V[i]) / (2 * GRAVITY * self.geom.diameter) if R > 0 else 0

            # C+特征线 (从上游)
            Cp = H[i-1] + c/GRAVITY * V[i-1] - J * self.dx

            # C-特征线 (从下游)
            Cm = H[i+1] - c/GRAVITY * V[i+1] - J * self.dx

            # 求解
            H_new[i] = (Cp + Cm) / 2
            V_new[i] = GRAVITY / c * (Cp - Cm) / 2

        # 边界条件处理
        # 上游: 恒定水头
        H_new[0] = H[0]
        V_new[0] = V[0] + GRAVITY/c * (H[0] - H_new[1])

        # 下游: 阀门 (时变边界)
        Cv = max(0, 1 - valve_closure_rate)  # 阀门开度系数
        V_new[-1] = Cv * V[-1]
        Cm = H[-2] - c/GRAVITY * V[-2]
        H_new[-1] = Cm + c/GRAVITY * V_new[-1]

        # 更新
        self.h = H_new - V_new**2 / (2 * GRAVITY + 1e-6)
        self.Q = V_new * self.A

        # 返回压力分布
        pressure = WATER_DENSITY * GRAVITY * H_new
        return pressure

    def get_state(self) -> Dict[str, np.ndarray]:
        """获取当前状态"""
        return {
            'x': self.x,
            'water_level': self.h,
            'flow_rate': self.Q,
            'area': self.A,
            'velocity': self.Q / (self.A + 1e-6),
            'pressure': WATER_DENSITY * GRAVITY * self.h,
        }


class ThermalModel:
    """热力学模型"""

    def __init__(self, geometry: TunnelGeometry, material: MaterialProperties, n_nodes: int = 100):
        self.geom = geometry
        self.mat = material
        self.n_nodes = n_nodes
        self.dx = geometry.length / (n_nodes - 1)

        # 温度场
        self.T_water = np.ones(n_nodes) * 288.15  # 水温 (K)
        self.T_wall = np.ones(n_nodes) * 288.15   # 壁温 (K)
        self.T_ground = 288.15                     # 地温 (K)

        # 径向网格 (用于壁面热传导)
        self.n_radial = 10
        self.r = np.linspace(
            geometry.diameter / 2,
            geometry.diameter / 2 + geometry.wall_thickness,
            self.n_radial
        )
        self.T_radial = np.ones((n_nodes, self.n_radial)) * 288.15

    def step(self, dt: float, velocity: np.ndarray, inlet_temperature: float):
        """
        热传递计算

        包括:
        1. 水体对流传热
        2. 水-壁面换热
        3. 壁面热传导
        4. 壁面-土体换热
        """
        n = self.n_nodes

        # 1. 水体对流 (上风格式)
        T_water_new = self.T_water.copy()
        for i in range(1, n):
            if velocity[i] > 0:
                dT_dx = (self.T_water[i] - self.T_water[i-1]) / self.dx
            else:
                dT_dx = (self.T_water[min(i+1, n-1)] - self.T_water[i]) / self.dx

            T_water_new[i] = self.T_water[i] - velocity[i] * dT_dx * dt

        # 入口边界
        T_water_new[0] = inlet_temperature

        # 2. 水-壁面换热
        h_conv = self._compute_heat_transfer_coefficient(velocity)
        D = self.geom.diameter

        for i in range(n):
            # 换热量
            Q_conv = h_conv[i] * np.pi * D * self.dx * (self.T_wall[i] - T_water_new[i])

            # 水温变化
            m_water = WATER_DENSITY * self.geom.area * self.dx
            dT_water = Q_conv * dt / (m_water * WATER_SPECIFIC_HEAT)
            T_water_new[i] += dT_water

            # 壁面温度变化 (简化)
            Q_to_ground = self._compute_ground_heat_flux(i)
            self.T_wall[i] += (-Q_conv + Q_to_ground) * dt / (
                self.mat.concrete_density * np.pi *
                ((D/2 + self.geom.wall_thickness)**2 - (D/2)**2) *
                self.dx * 900  # 混凝土比热
            )

        self.T_water = T_water_new

    def _compute_heat_transfer_coefficient(self, velocity: np.ndarray) -> np.ndarray:
        """计算对流换热系数 (Dittus-Boelter公式)"""
        D = self.geom.diameter
        h = np.zeros_like(velocity)

        for i, v in enumerate(velocity):
            Re = abs(v) * D / WATER_VISCOSITY
            Pr = WATER_VISCOSITY * WATER_DENSITY * WATER_SPECIFIC_HEAT / WATER_THERMAL_CONDUCTIVITY

            if Re > 10000:  # 湍流
                Nu = 0.023 * Re**0.8 * Pr**0.4
            elif Re > 2300:  # 过渡区
                Nu = 0.116 * (Re**(2/3) - 125) * Pr**(1/3)
            else:  # 层流
                Nu = 3.66

            h[i] = Nu * WATER_THERMAL_CONDUCTIVITY / D

        return h

    def _compute_ground_heat_flux(self, node_idx: int) -> float:
        """计算与地层的换热"""
        # 简化模型: 恒定地温假设
        R_ground = 2.0  # 热阻 (m²·K/W)
        return (self.T_ground - self.T_wall[node_idx]) / R_ground

    def get_state(self) -> Dict[str, np.ndarray]:
        """获取温度状态"""
        return {
            'water_temperature': self.T_water,
            'wall_temperature': self.T_wall,
            'temperature_gradient': np.gradient(self.T_water, self.dx)
        }


class StructuralModel:
    """结构力学模型"""

    def __init__(self, geometry: TunnelGeometry, material: MaterialProperties, n_nodes: int = 100):
        self.geom = geometry
        self.mat = material
        self.n_nodes = n_nodes
        self.dx = geometry.length / (n_nodes - 1)

        # 结构状态
        self.stress_hoop = np.zeros(n_nodes)      # 环向应力
        self.stress_axial = np.zeros(n_nodes)     # 轴向应力
        self.strain = np.zeros(n_nodes)           # 应变
        self.displacement = np.zeros(n_nodes)     # 径向位移

        # 外部荷载
        self.external_pressure = np.ones(n_nodes) * 0.7e6  # 外部水土压力 (Pa)
        self.seismic_load = np.zeros(n_nodes)     # 地震荷载

    def compute_stress(self, internal_pressure: np.ndarray,
                      temperature: np.ndarray,
                      reference_temperature: float = 288.15):
        """
        计算应力状态

        Args:
            internal_pressure: 内部水压 (Pa)
            temperature: 温度场 (K)
            reference_temperature: 参考温度 (K)
        """
        D = self.geom.diameter
        t = self.geom.wall_thickness
        E = self.mat.concrete_E
        nu = self.mat.concrete_nu
        alpha = self.mat.concrete_alpha

        for i in range(self.n_nodes):
            p_int = internal_pressure[i]
            p_ext = self.external_pressure[i]
            delta_p = p_int - p_ext

            # 薄壁圆筒应力 (Lame公式简化)
            # 环向应力
            self.stress_hoop[i] = delta_p * D / (2 * t)

            # 轴向应力 (考虑端部约束)
            self.stress_axial[i] = delta_p * D / (4 * t)

            # 温度应力
            delta_T = temperature[i] - reference_temperature
            stress_thermal = -E * alpha * delta_T / (1 - nu)
            self.stress_hoop[i] += stress_thermal

            # 地震附加应力
            self.stress_hoop[i] += self.seismic_load[i]

            # 应变
            self.strain[i] = (self.stress_hoop[i] - nu * self.stress_axial[i]) / E

            # 径向位移
            self.displacement[i] = self.strain[i] * D / 2

    def check_safety(self) -> Dict[str, Any]:
        """检查结构安全"""
        # 许用应力
        sigma_allow_tension = 2.0e6    # 混凝土抗拉 (Pa)
        sigma_allow_compress = 20.0e6  # 混凝土抗压 (Pa)

        # 安全系数
        max_tension = np.max(self.stress_hoop)
        max_compress = np.min(self.stress_hoop)

        tension_safety = sigma_allow_tension / max(max_tension, 1)
        compress_safety = abs(sigma_allow_compress / min(max_compress, -1))

        # 裂缝检查
        crack_width = self._estimate_crack_width()

        return {
            'max_hoop_stress': np.max(np.abs(self.stress_hoop)),
            'max_axial_stress': np.max(np.abs(self.stress_axial)),
            'max_displacement': np.max(np.abs(self.displacement)),
            'tension_safety_factor': tension_safety,
            'compression_safety_factor': compress_safety,
            'estimated_crack_width': crack_width,
            'is_safe': tension_safety > 2.0 and compress_safety > 2.5
        }

    def _estimate_crack_width(self) -> float:
        """估算裂缝宽度"""
        max_strain = np.max(self.strain)
        if max_strain > 0:
            # 简化裂缝宽度估算
            crack_spacing = 200  # mm
            return max_strain * crack_spacing
        return 0.0

    def apply_seismic_load(self, acceleration: float, response_spectrum: Optional[np.ndarray] = None):
        """
        施加地震荷载

        Args:
            acceleration: 地面加速度 (g)
            response_spectrum: 反应谱 (可选)
        """
        # 简化的静力法
        g = 9.81
        m = self.mat.concrete_density * np.pi * self.geom.diameter * self.geom.wall_thickness * self.dx

        # 惯性力产生的应力
        seismic_force = m * acceleration * g
        self.seismic_load = seismic_force / (np.pi * self.geom.diameter * self.geom.wall_thickness)

    def get_state(self) -> Dict[str, np.ndarray]:
        """获取结构状态"""
        return {
            'hoop_stress': self.stress_hoop,
            'axial_stress': self.stress_axial,
            'strain': self.strain,
            'displacement': self.displacement
        }


class CoupledPhysicalModel:
    """耦合物理模型 - 水力-热力-结构耦合"""

    def __init__(self, n_nodes: int = 100):
        self.geometry = TunnelGeometry()
        self.material = MaterialProperties()
        self.n_nodes = n_nodes

        # 子模型
        self.hydraulic = HighFidelityHydraulicModel(self.geometry, n_nodes)
        self.thermal = ThermalModel(self.geometry, self.material, n_nodes)
        self.structural = StructuralModel(self.geometry, self.material, n_nodes)

        # 仿真时间
        self.time = 0.0

        # 耦合参数
        self.coupling_interval = 10  # 耦合计算间隔 (步)
        self.step_count = 0

    def step(self, dt: float,
            Q_upstream: float,
            h_downstream: float,
            T_inlet: float = 288.15,
            valve_closure_rate: float = 0.0,
            seismic_acceleration: float = 0.0) -> SimulationState:
        """
        耦合仿真步进

        Args:
            dt: 时间步长 (s)
            Q_upstream: 上游流量 (m³/s)
            h_downstream: 下游水位 (m)
            T_inlet: 入口水温 (K)
            valve_closure_rate: 阀门关闭速率
            seismic_acceleration: 地震加速度 (g)

        Returns:
            仿真状态
        """
        # 1. 水力学计算
        if valve_closure_rate > 0:
            pressure = self.hydraulic.compute_water_hammer(dt, valve_closure_rate)
        else:
            self.hydraulic.step_saint_venant(dt, Q_upstream, h_downstream)
            pressure = WATER_DENSITY * GRAVITY * self.hydraulic.h

        # 2. 热力学计算
        velocity = self.hydraulic.Q / (self.hydraulic.A + 1e-6)
        self.thermal.step(dt, velocity, T_inlet)

        # 3. 结构计算 (间隔进行以提高效率)
        if self.step_count % self.coupling_interval == 0:
            # 施加地震荷载
            if seismic_acceleration > 0:
                self.structural.apply_seismic_load(seismic_acceleration)

            # 计算应力
            self.structural.compute_stress(pressure, self.thermal.T_wall)

        self.time += dt
        self.step_count += 1

        # 返回状态
        return SimulationState(
            time=self.time,
            water_level=self.hydraulic.h.copy(),
            velocity=velocity.copy(),
            pressure=pressure.copy(),
            flow_rate=self.hydraulic.Q.copy(),
            water_temperature=self.thermal.T_water.copy(),
            wall_temperature=self.thermal.T_wall.copy(),
            stress=self.structural.stress_hoop.copy(),
            strain=self.structural.strain.copy(),
            displacement=self.structural.displacement.copy()
        )

    def reset(self, initial_flow: float = 265.0, initial_temp: float = 288.15):
        """重置仿真"""
        self.time = 0.0
        self.step_count = 0

        # 重置水力学
        self.hydraulic.h = np.ones(self.n_nodes) * self.geometry.diameter * 0.9
        self.hydraulic.Q = np.ones(self.n_nodes) * initial_flow
        self.hydraulic.A = np.ones(self.n_nodes) * self.geometry.area

        # 重置热力学
        self.thermal.T_water = np.ones(self.n_nodes) * initial_temp
        self.thermal.T_wall = np.ones(self.n_nodes) * initial_temp

        # 重置结构
        self.structural.stress_hoop = np.zeros(self.n_nodes)
        self.structural.stress_axial = np.zeros(self.n_nodes)
        self.structural.strain = np.zeros(self.n_nodes)
        self.structural.displacement = np.zeros(self.n_nodes)
        self.structural.seismic_load = np.zeros(self.n_nodes)

    def get_full_state(self) -> Dict[str, Any]:
        """获取完整状态"""
        return {
            'time': self.time,
            'hydraulic': self.hydraulic.get_state(),
            'thermal': self.thermal.get_state(),
            'structural': self.structural.get_state(),
            'safety': self.structural.check_safety()
        }

    def inject_fault(self, fault_type: str, parameters: Dict[str, Any]):
        """
        故障注入

        Args:
            fault_type: 故障类型
            parameters: 故障参数
        """
        if fault_type == 'leakage':
            location = parameters.get('location', self.n_nodes // 2)
            rate = parameters.get('rate', 0.01)
            # 模拟泄漏: 减少局部流量
            self.hydraulic.Q[location:] *= (1 - rate)

        elif fault_type == 'blockage':
            location = parameters.get('location', self.n_nodes // 2)
            ratio = parameters.get('ratio', 0.2)
            # 模拟堵塞: 减少局部面积
            self.hydraulic.A[location] *= (1 - ratio)

        elif fault_type == 'external_pressure_increase':
            increase = parameters.get('increase', 0.1e6)
            self.structural.external_pressure += increase

        elif fault_type == 'temperature_anomaly':
            location = parameters.get('location', self.n_nodes // 2)
            delta_T = parameters.get('delta_T', 10)
            width = parameters.get('width', 10)
            # 局部温度异常
            start = max(0, location - width // 2)
            end = min(self.n_nodes, location + width // 2)
            self.thermal.T_water[start:end] += delta_T
