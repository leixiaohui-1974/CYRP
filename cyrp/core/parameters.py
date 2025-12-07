"""
System parameters for the Yellow River Crossing Project.
穿黄工程系统参数定义

Based on Report I - Physical Entity & Digital Twin Basis
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import numpy as np


class TunnelID(Enum):
    """隧洞标识"""
    TUNNEL_1 = 1  # 1# 洞
    TUNNEL_2 = 2  # 2# 洞


@dataclass
class TunnelParameters:
    """
    穿黄隧洞结构参数

    双层衬砌结构：外衬为盾构管片，内衬为预应力钢筋混凝土
    中间设有环形充水空腔（Annular Cavity）
    """
    # 基本几何参数
    length: float = 4250.0  # 单洞长度 L (m)
    inner_diameter: float = 7.0  # 内径 D (m)
    outer_diameter: float = 9.03  # 外衬外径 (m)
    wall_thickness_inner: float = 0.45  # 内衬厚度 (m)
    wall_thickness_outer: float = 0.55  # 外衬厚度 (m)

    # 水力设计参数
    design_flow_rate: float = 265.0  # 设计流量 Q_des (m³/s)
    max_head_difference: float = 60.0  # 最大承压水头 ΔH_max (m)

    # 管节参数
    segment_length: float = 2.0  # 管节长度 (m)
    num_segments: int = 2125  # 管节数量

    # 材料特性
    concrete_elastic_modulus: float = 3.0e10  # 混凝土弹性模量 E (Pa)
    steel_elastic_modulus: float = 2.1e11  # 钢材弹性模量 E (Pa)
    poisson_ratio: float = 0.2  # 泊松比

    # 摩阻参数
    manning_coefficient: float = 0.014  # 曼宁系数
    darcy_friction_factor: float = 0.02  # 达西摩擦因子

    @property
    def cross_section_area(self) -> float:
        """过水断面面积 (m²)"""
        return np.pi * (self.inner_diameter / 2) ** 2

    @property
    def wetted_perimeter(self) -> float:
        """湿周 (m)"""
        return np.pi * self.inner_diameter

    @property
    def hydraulic_radius(self) -> float:
        """水力半径 (m)"""
        return self.cross_section_area / self.wetted_perimeter

    @property
    def annular_cavity_area(self) -> float:
        """环形空腔截面积 (m²)"""
        r_outer = self.outer_diameter / 2
        r_inner = (self.inner_diameter + 2 * self.wall_thickness_inner) / 2
        return np.pi * (r_outer**2 - r_inner**2)


@dataclass
class EnvironmentParameters:
    """
    环境边界条件参数

    外水压力场：黄河河床下的孔隙水压力
    地质风险场：第四系冲积层
    """
    # 黄河水文参数
    yellow_river_water_level: float = 95.0  # 黄河水位 (m, 高程)
    yellow_river_water_temp: float = 12.0  # 黄河水温 (°C)

    # 丹江口来水参数
    source_water_temp: float = 8.0  # 丹江口水温 (°C)
    source_water_turbidity: float = 5.0  # 丹江口浊度 (NTU)

    # 地质参数
    burial_depth: float = 70.0  # 埋深 (m)
    soil_unit_weight: float = 19.0  # 土体容重 (kN/m³)
    groundwater_level: float = 90.0  # 地下水位 (m, 高程)

    # 外水压力
    pore_water_pressure: float = 0.6  # 孔隙水压力系数

    # 地震参数
    seismic_intensity: float = 7.0  # 设防烈度
    peak_ground_acceleration: float = 0.15  # PGA (g)
    liquefaction_susceptibility: float = 0.7  # 液化敏感度

    # 环境温度
    ambient_temperature: float = 15.0  # 环境温度 (°C)

    def get_external_water_pressure(self, elevation: float) -> float:
        """
        计算外水压力

        Args:
            elevation: 计算点高程 (m)

        Returns:
            外水压力 (Pa)
        """
        depth_below_gwl = max(0, self.groundwater_level - elevation)
        return 9810 * depth_below_gwl * self.pore_water_pressure


@dataclass
class GateParameters:
    """闸门参数"""
    max_opening: float = 7.0  # 最大开度 (m)
    min_opening: float = 0.0  # 最小开度 (m)
    max_velocity: float = 0.01  # 最大开启速度 (m/s)
    response_time: float = 0.5  # 响应时间 (s)

    # 水力系数
    discharge_coefficient: float = 0.75  # 流量系数


@dataclass
class PumpParameters:
    """泵站参数"""
    max_flow_rate: float = 10.0  # 最大流量 (m³/s)
    rated_head: float = 30.0  # 额定扬程 (m)
    efficiency: float = 0.85  # 效率
    power_rating: float = 500.0  # 额定功率 (kW)


@dataclass
class SystemLimits:
    """系统安全限值"""
    # 压力限值
    max_internal_pressure: float = 1.0e6  # 最大内压 (Pa)
    min_internal_pressure: float = -5.0e4  # 最小内压（真空限值）(Pa)
    max_pressure_rate: float = 5.0e4  # 最大压力变化率 (Pa/s)

    # 流量限值
    max_flow_rate: float = 300.0  # 最大流量 (m³/s)
    min_flow_rate: float = 0.0  # 最小流量 (m³/s)
    max_flow_imbalance: float = 0.1  # 最大流量不平衡度

    # 流速限值
    max_velocity: float = 5.0  # 最大流速 (m/s)
    min_velocity_for_sediment: float = 3.0  # 冲淤最小流速 (m/s)

    # 水位变化限值
    max_water_level_rate: float = 0.5  # 最大水位变化率 (m/min)

    # 结构限值
    max_settlement: float = 0.05  # 最大沉降 (m)
    max_tilt_angle: float = 0.005  # 最大倾斜角 (rad)

    # 水锤限值
    max_water_hammer_pressure: float = 2.0e6  # 最大水锤压力 (Pa)
