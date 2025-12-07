"""
穿黄工程真实参数 - Real Parameters from Engineering Documents

基于用户提供的工程参数文档定义的真实设计参数
Based on real design parameters from engineering documents
"""

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


@dataclass
class RealTunnelParameters:
    """
    穿黄隧道真实设计参数

    数据来源: 南水北调中线穿黄工程设计文档
    """
    # ===== 几何参数 =====
    # 隧道总长度 (m) - 北岸竖井至南岸竖井
    total_length: float = 4250.0

    # 隧道内径 (m)
    inner_diameter: float = 7.0

    # 隧道壁厚 (m) - 钢筋混凝土衬砌
    wall_thickness: float = 0.6

    # 隧道外径 (m)
    outer_diameter: float = 8.2

    # 隧道数量
    n_tunnels: int = 2  # 双洞布置

    # 隧道间距 (m)
    tunnel_spacing: float = 28.0

    # 纵坡 (m/m)
    longitudinal_slope: float = 1.0 / 10000  # 万分之一

    # 埋深范围 (m)
    burial_depth_min: float = 30.0
    burial_depth_max: float = 70.0

    # ===== 水力学参数 =====
    # 设计流量 (m³/s)
    design_flow: float = 265.0

    # 加大流量 (m³/s)
    max_flow: float = 320.0

    # 最小运行流量 (m³/s)
    min_flow: float = 50.0

    # 设计流速 (m/s)
    design_velocity: float = 3.44

    # 进口设计水位 (m，相对于黄海高程)
    inlet_design_level: float = 118.72

    # 出口设计水位 (m)
    outlet_design_level: float = 117.83

    # 水头差 (m)
    head_difference: float = 0.89

    # 最大内水压力 (MPa)
    max_internal_pressure: float = 1.0

    # 最大外水压力 (MPa) - 黄河水位最高时
    max_external_pressure: float = 0.7

    # 曼宁糙率
    manning_roughness: float = 0.014

    # 水锤波速 (m/s)
    water_hammer_wave_speed: float = 1000.0

    # ===== 材料参数 =====
    # 混凝土强度等级
    concrete_grade: str = "C40"

    # 混凝土弹性模量 (Pa)
    concrete_E: float = 3.25e10

    # 混凝土泊松比
    concrete_poisson: float = 0.2

    # 混凝土密度 (kg/m³)
    concrete_density: float = 2500.0

    # 钢筋屈服强度 (Pa)
    steel_yield_strength: float = 400.0e6

    # 配筋率
    reinforcement_ratio: float = 0.015

    # ===== 地质参数 =====
    # 穿越地层
    geological_layers: str = "第四系全新统黄河冲积物"

    # 地层液化等级
    liquefaction_level: str = "轻微-中等"

    # 地下水位深度 (m)
    groundwater_depth: float = 5.0

    # ===== 控制节点 =====
    # 北岸进口里程
    north_inlet_chainage: float = 0.0

    # 北岸竖井位置
    north_shaft_chainage: float = 100.0

    # 最低点位置 (黄河河床下)
    lowest_point_chainage: float = 2125.0

    # 南岸竖井位置
    south_shaft_chainage: float = 4150.0

    # 南岸出口里程
    south_outlet_chainage: float = 4250.0

    # ===== 运行限制 =====
    # 阀门全行程时间 (s)
    valve_stroke_time: float = 120.0

    # 紧急关闭时间 (s)
    emergency_close_time: float = 30.0

    # 流量变化率限制 (m³/s/min)
    flow_change_rate_limit: float = 10.0

    # 压力变化率限制 (kPa/s)
    pressure_change_rate_limit: float = 50.0

    # ===== 安全阈值 =====
    # 负压保护阈值 (Pa)
    vacuum_protection_threshold: float = -50000.0

    # 超压保护阈值 (Pa)
    overpressure_threshold: float = 1.0e6

    # 双洞不对称流量差限制
    asymmetric_flow_limit: float = 0.1  # 10%

    # 流速报警阈值 (m/s)
    velocity_alarm_threshold: float = 4.5

    # ===== 地震参数 =====
    # 设防烈度
    seismic_intensity: str = "VII度"

    # 设计基本加速度 (g)
    design_acceleration: float = 0.15

    @property
    def cross_section_area(self) -> float:
        """过流断面积 (m²)"""
        return np.pi * (self.inner_diameter / 2) ** 2

    @property
    def wetted_perimeter(self) -> float:
        """湿周 (m)"""
        return np.pi * self.inner_diameter

    @property
    def hydraulic_radius(self) -> float:
        """水力半径 (m)"""
        return self.cross_section_area / self.wetted_perimeter

    def get_chainage_depth(self, chainage: float) -> float:
        """根据里程获取埋深"""
        # 简化的埋深曲线 (实际应使用精确的纵断面数据)
        if chainage < 500:
            return self.burial_depth_min
        elif chainage > 3750:
            return self.burial_depth_min
        else:
            # 中间段埋深较大
            progress = (chainage - 500) / (3750 - 500)
            depth_range = self.burial_depth_max - self.burial_depth_min
            return self.burial_depth_min + depth_range * np.sin(np.pi * progress)


@dataclass
class RealScenarioThresholds:
    """
    场景识别阈值 - 基于报告I的ODD定义
    """
    # ===== 域1: 常规运行 =====
    nominal_flow_high: float = 280.0      # 高流量阈值
    nominal_flow_medium: float = 180.0    # 中流量阈值
    nominal_flow_low: float = 100.0       # 低流量阈值

    # ===== 域2: 过渡运行 =====
    transition_flow_rate: float = 10.0    # 切换时流量变化率 (m³/s/min)
    maintenance_flow_max: float = 150.0   # 检修模式最大流量

    # ===== 域3: 应急运行 =====
    # 渗漏等级阈值 (渗漏率 %)
    leakage_minor: float = 0.01           # 轻微渗漏
    leakage_moderate: float = 0.03        # 中度渗漏
    leakage_severe: float = 0.05          # 严重渗漏
    leakage_critical: float = 0.08        # 临界渗漏

    # 地震烈度阈值 (g)
    seismic_vi: float = 0.05              # VI度
    seismic_vii: float = 0.15             # VII度
    seismic_viii: float = 0.30            # VIII度

    # 压力异常阈值 (相对于正常值的偏差 %)
    pressure_deviation_warning: float = 0.1
    pressure_deviation_alarm: float = 0.2
    pressure_deviation_critical: float = 0.3


@dataclass
class RealControlParameters:
    """
    控制参数 - 基于报告II的HD-MPC设计
    """
    # ===== MPC参数 =====
    # 预测时域 (步数)
    prediction_horizon_nominal: int = 30
    prediction_horizon_emergency: int = 10

    # 控制时域
    control_horizon: int = 10

    # 采样周期 (s)
    sampling_period: float = 1.0

    # 状态权重矩阵对角元素
    Q_flow: float = 100.0
    Q_pressure: float = 50.0
    Q_velocity: float = 20.0

    # 控制权重
    R_valve: float = 1.0

    # 终端权重
    Qf_multiplier: float = 10.0

    # ===== PID参数 (级联控制) =====
    # 外环 (流量)
    Kp_flow_outer: float = 2.0
    Ki_flow_outer: float = 0.1
    Kd_flow_outer: float = 0.05

    # 内环 (阀门位置)
    Kp_valve_inner: float = 5.0
    Ki_valve_inner: float = 0.5
    Kd_valve_inner: float = 0.1

    # ===== 约束参数 =====
    # 阀门位置约束
    valve_min: float = 0.0
    valve_max: float = 1.0

    # 阀门速率约束 (%/s)
    valve_rate_max: float = 0.01

    # 流量约束
    flow_min: float = 0.0
    flow_max: float = 320.0


# 创建全局参数实例
REAL_TUNNEL_PARAMS = RealTunnelParameters()
REAL_SCENARIO_THRESHOLDS = RealScenarioThresholds()
REAL_CONTROL_PARAMS = RealControlParameters()


def validate_parameters() -> Dict[str, Any]:
    """验证参数一致性"""
    params = REAL_TUNNEL_PARAMS

    validation_results = {
        'passed': True,
        'checks': []
    }

    # 检查流速计算
    calc_velocity = params.design_flow / params.cross_section_area
    check_velocity = {
        'name': '设计流速验证',
        'expected': params.design_velocity,
        'calculated': calc_velocity,
        'passed': abs(calc_velocity - params.design_velocity) < 0.1
    }
    validation_results['checks'].append(check_velocity)

    # 检查水头损失 (曼宁公式)
    R = params.hydraulic_radius
    n = params.manning_roughness
    V = params.design_velocity
    L = params.total_length
    S = (n * V / R**(2/3)) ** 2
    head_loss = S * L
    check_head = {
        'name': '水头损失验证',
        'expected': params.head_difference,
        'calculated': head_loss,
        'passed': abs(head_loss - params.head_difference) < 0.5
    }
    validation_results['checks'].append(check_head)

    # 检查断面积
    check_area = {
        'name': '断面积验证',
        'expected': 38.48,  # π * 3.5²
        'calculated': params.cross_section_area,
        'passed': abs(params.cross_section_area - 38.48) < 0.1
    }
    validation_results['checks'].append(check_area)

    # 汇总
    validation_results['passed'] = all(c['passed'] for c in validation_results['checks'])

    return validation_results


def print_parameter_summary():
    """打印参数摘要"""
    params = REAL_TUNNEL_PARAMS

    print("=" * 60)
    print("穿黄工程真实设计参数")
    print("=" * 60)

    print("\n【几何参数】")
    print(f"  隧道长度: {params.total_length} m")
    print(f"  隧道内径: {params.inner_diameter} m")
    print(f"  隧道数量: {params.n_tunnels} (双洞)")
    print(f"  隧道间距: {params.tunnel_spacing} m")
    print(f"  埋深范围: {params.burial_depth_min}-{params.burial_depth_max} m")

    print("\n【水力学参数】")
    print(f"  设计流量: {params.design_flow} m³/s")
    print(f"  加大流量: {params.max_flow} m³/s")
    print(f"  设计流速: {params.design_velocity} m/s")
    print(f"  过流断面积: {params.cross_section_area:.2f} m²")
    print(f"  水力半径: {params.hydraulic_radius:.3f} m")
    print(f"  进出口水位差: {params.head_difference} m")
    print(f"  最大内水压力: {params.max_internal_pressure} MPa")

    print("\n【控制参数】")
    print(f"  阀门全行程时间: {params.valve_stroke_time} s")
    print(f"  紧急关闭时间: {params.emergency_close_time} s")
    print(f"  流量变化率限制: {params.flow_change_rate_limit} m³/s/min")

    print("\n【安全阈值】")
    print(f"  负压保护: {params.vacuum_protection_threshold/1000} kPa")
    print(f"  超压保护: {params.overpressure_threshold/1e6} MPa")
    print(f"  不对称流量限制: {params.asymmetric_flow_limit*100}%")

    # 验证
    print("\n【参数验证】")
    results = validate_parameters()
    for check in results['checks']:
        status = "✓" if check['passed'] else "✗"
        print(f"  {status} {check['name']}: "
              f"期望={check['expected']:.2f}, 计算={check['calculated']:.2f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    print_parameter_summary()
