"""
Full Scenario Matrix Generator for CYRP
穿黄工程全场景矩阵生成器

实现成千上万种场景的自动生成，包括：
1. 参数化场景变体
2. 组合场景
3. 时序场景
4. 极端场景
5. 故障注入场景
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Iterator, Any, Callable
from enum import Enum, auto
from itertools import product, combinations
import hashlib
import json
from datetime import datetime, timedelta


# ============================================================================
# 场景维度定义
# ============================================================================

class FlowLevel(Enum):
    """流量等级"""
    ZERO = 0.0              # 停水
    MINIMAL = 0.1           # 最小流量 26.5 m³/s
    LOW = 0.3               # 低流量 79.5 m³/s
    MEDIUM_LOW = 0.5        # 中低 132.5 m³/s
    MEDIUM = 0.7            # 中等 185.5 m³/s
    DESIGN = 0.8            # 设计流量 212 m³/s
    NORMAL = 1.0            # 正常 265 m³/s
    HIGH = 1.1              # 高流量 291.5 m³/s
    MAX_DESIGN = 1.2        # 加大流量 318 m³/s
    OVERLOAD = 1.3          # 超载 344.5 m³/s


class PressureLevel(Enum):
    """压力等级"""
    NEGATIVE = -0.05        # 负压 -50 kPa
    ATMOSPHERIC = 0.0       # 大气压
    LOW = 0.2               # 低压 200 kPa
    MEDIUM_LOW = 0.4        # 中低 400 kPa
    MEDIUM = 0.5            # 中等 500 kPa
    DESIGN = 0.6            # 设计压力 600 kPa
    HIGH = 0.8              # 高压 800 kPa
    MAX = 1.0               # 最大 1000 kPa
    OVER_PRESSURE = 1.1     # 超压 1100 kPa


class TemperatureLevel(Enum):
    """温度等级"""
    FREEZING = 0            # 冰点 0°C
    COLD = 5                # 寒冷 5°C
    COOL = 10               # 凉爽 10°C
    NORMAL = 15             # 正常 15°C
    WARM = 20               # 温暖 20°C
    HOT = 25                # 炎热 25°C
    EXTREME = 30            # 极端 30°C


class SeasonType(Enum):
    """季节"""
    SPRING = "spring"       # 春季 3-5月
    SUMMER = "summer"       # 夏季 6-8月
    AUTUMN = "autumn"       # 秋季 9-11月
    WINTER = "winter"       # 冬季 12-2月


class TimeOfDay(Enum):
    """时段"""
    NIGHT_EARLY = 0         # 凌晨 0-4
    DAWN = 4                # 黎明 4-6
    MORNING = 6             # 上午 6-10
    MIDDAY = 10             # 中午 10-14
    AFTERNOON = 14          # 下午 14-18
    EVENING = 18            # 傍晚 18-20
    NIGHT = 20              # 夜间 20-24


class OperationMode(Enum):
    """运行模式"""
    SHUTDOWN = "shutdown"           # 停机
    STARTUP = "startup"             # 启动
    NORMAL = "normal"               # 正常运行
    STANDBY = "standby"             # 待机
    MAINTENANCE = "maintenance"     # 检修
    EMERGENCY = "emergency"         # 应急
    TRANSITION = "transition"       # 过渡


class ValveState(Enum):
    """阀门状态"""
    FULLY_CLOSED = 0.0
    NEARLY_CLOSED = 0.1
    QUARTER_OPEN = 0.25
    HALF_OPEN = 0.5
    THREE_QUARTER_OPEN = 0.75
    NEARLY_OPEN = 0.9
    FULLY_OPEN = 1.0


class PumpState(Enum):
    """泵状态"""
    OFF = "off"
    STARTING = "starting"
    RUNNING_LOW = "running_low"
    RUNNING_NORMAL = "running_normal"
    RUNNING_HIGH = "running_high"
    STOPPING = "stopping"
    FAULT = "fault"


class FaultType(Enum):
    """故障类型"""
    NONE = "none"
    # 传感器故障
    SENSOR_DRIFT = "sensor_drift"
    SENSOR_NOISE = "sensor_noise"
    SENSOR_STUCK = "sensor_stuck"
    SENSOR_BIAS = "sensor_bias"
    SENSOR_FAILURE = "sensor_failure"
    # 执行器故障
    VALVE_STUCK = "valve_stuck"
    VALVE_LEAKAGE = "valve_leakage"
    VALVE_SLOW = "valve_slow"
    PUMP_CAVITATION = "pump_cavitation"
    PUMP_VIBRATION = "pump_vibration"
    PUMP_FAILURE = "pump_failure"
    # 结构故障
    LEAKAGE_MINOR = "leakage_minor"
    LEAKAGE_MAJOR = "leakage_major"
    CRACK_DETECTED = "crack_detected"
    SETTLEMENT = "settlement"
    CORROSION = "corrosion"
    # 系统故障
    POWER_FLUCTUATION = "power_fluctuation"
    POWER_FAILURE = "power_failure"
    COMMUNICATION_LOSS = "communication_loss"
    CONTROL_FAILURE = "control_failure"


class ExternalEvent(Enum):
    """外部事件"""
    NONE = "none"
    # 自然灾害
    FLOOD_WARNING = "flood_warning"
    FLOOD_LEVEL1 = "flood_level1"
    FLOOD_LEVEL2 = "flood_level2"
    EARTHQUAKE_III = "earthquake_III"
    EARTHQUAKE_V = "earthquake_V"
    EARTHQUAKE_VII = "earthquake_VII"
    EARTHQUAKE_VIII = "earthquake_VIII"
    STORM = "storm"
    LIGHTNING = "lightning"
    # 人为事件
    UPSTREAM_CHANGE = "upstream_change"
    DOWNSTREAM_DEMAND = "downstream_demand"
    SCHEDULED_OUTAGE = "scheduled_outage"
    EMERGENCY_SHUTOFF = "emergency_shutoff"


class WaterQuality(Enum):
    """水质状况"""
    EXCELLENT = "excellent"
    GOOD = "good"
    NORMAL = "normal"
    POOR = "poor"
    CONTAMINATED = "contaminated"
    SEDIMENT_HIGH = "sediment_high"


class TunnelSection(Enum):
    """隧道分区"""
    INLET = "inlet"                 # 进口段 0-500m
    UPSTREAM = "upstream"           # 上游段 500-1500m
    MIDDLE_UP = "middle_up"         # 中上段 1500-2125m
    MIDDLE = "middle"               # 中间段 2125m (最深处)
    MIDDLE_DOWN = "middle_down"     # 中下段 2125-2750m
    DOWNSTREAM = "downstream"       # 下游段 2750-3750m
    OUTLET = "outlet"               # 出口段 3750-4250m


# ============================================================================
# 场景参数结构
# ============================================================================

@dataclass
class ScenarioParameters:
    """场景参数集"""
    # 标识
    scenario_id: str = ""
    name: str = ""
    description: str = ""

    # 水力参数
    flow_level: FlowLevel = FlowLevel.NORMAL
    pressure_level: PressureLevel = PressureLevel.MEDIUM
    flow_rate: float = 265.0        # m³/s
    pressure_inlet: float = 500.0   # kPa
    pressure_outlet: float = 400.0  # kPa
    velocity: float = 3.44          # m/s

    # 环境参数
    temperature: TemperatureLevel = TemperatureLevel.NORMAL
    water_temperature: float = 15.0  # °C
    ambient_temperature: float = 20.0  # °C
    season: SeasonType = SeasonType.SUMMER
    time_of_day: TimeOfDay = TimeOfDay.MIDDAY

    # 运行参数
    operation_mode: OperationMode = OperationMode.NORMAL
    tunnel_id: int = 1              # 1 or 2 (双洞)

    # 设备状态
    inlet_valve_state: ValveState = ValveState.FULLY_OPEN
    outlet_valve_state: ValveState = ValveState.FULLY_OPEN
    emergency_valve_state: ValveState = ValveState.FULLY_CLOSED
    pump_state: PumpState = PumpState.OFF

    # 故障状态
    fault_types: List[FaultType] = field(default_factory=list)
    fault_locations: List[TunnelSection] = field(default_factory=list)
    fault_severity: float = 0.0     # 0-1

    # 外部事件
    external_events: List[ExternalEvent] = field(default_factory=list)

    # 水质
    water_quality: WaterQuality = WaterQuality.NORMAL
    sediment_concentration: float = 0.0  # kg/m³

    # 结构状态
    structural_health: float = 1.0  # 0-1
    leakage_rate: float = 0.0       # L/s/m

    # 元数据
    priority: str = "normal"        # low, normal, high, critical
    category: str = "normal"        # normal, transition, fault, emergency, composite
    parent_scenarios: List[str] = field(default_factory=list)  # 组合场景的父场景

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'scenario_id': self.scenario_id,
            'name': self.name,
            'description': self.description,
            'flow_level': self.flow_level.name,
            'pressure_level': self.pressure_level.name,
            'flow_rate': self.flow_rate,
            'pressure_inlet': self.pressure_inlet,
            'pressure_outlet': self.pressure_outlet,
            'velocity': self.velocity,
            'temperature': self.temperature.name,
            'water_temperature': self.water_temperature,
            'season': self.season.value,
            'time_of_day': self.time_of_day.name,
            'operation_mode': self.operation_mode.value,
            'tunnel_id': self.tunnel_id,
            'inlet_valve_state': self.inlet_valve_state.value,
            'outlet_valve_state': self.outlet_valve_state.value,
            'fault_types': [f.value for f in self.fault_types],
            'external_events': [e.value for e in self.external_events],
            'water_quality': self.water_quality.value,
            'structural_health': self.structural_health,
            'priority': self.priority,
            'category': self.category,
            'parent_scenarios': self.parent_scenarios,
        }

    def generate_id(self) -> str:
        """生成唯一ID"""
        data = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(data.encode()).hexdigest()[:12]


# ============================================================================
# 场景矩阵生成器
# ============================================================================

class ScenarioMatrixGenerator:
    """
    全场景矩阵生成器

    通过参数组合生成大规模场景集合
    """

    def __init__(self):
        # 设计参数
        self.design_flow = 265.0  # m³/s
        self.max_flow = 320.0     # m³/s
        self.design_pressure = 600.0  # kPa
        self.max_pressure = 1000.0  # kPa

        # 场景计数器
        self.scenario_counter = 0

        # 生成的场景
        self.scenarios: Dict[str, ScenarioParameters] = {}

        # 场景索引
        self.index_by_category: Dict[str, List[str]] = {}
        self.index_by_priority: Dict[str, List[str]] = {}
        self.index_by_flow: Dict[str, List[str]] = {}
        self.index_by_fault: Dict[str, List[str]] = {}

    def generate_all(self,
                     include_normal: bool = True,
                     include_transitions: bool = True,
                     include_faults: bool = True,
                     include_external: bool = True,
                     include_composite: bool = True,
                     max_scenarios: Optional[int] = None) -> int:
        """
        生成所有场景

        Args:
            include_normal: 包含正常运行场景
            include_transitions: 包含过渡场景
            include_faults: 包含故障场景
            include_external: 包含外部事件场景
            include_composite: 包含组合场景
            max_scenarios: 最大场景数限制

        Returns:
            生成的场景总数
        """
        if include_normal:
            self._generate_normal_scenarios()

        if include_transitions:
            self._generate_transition_scenarios()

        if include_faults:
            self._generate_fault_scenarios()

        if include_external:
            self._generate_external_event_scenarios()

        if include_composite:
            self._generate_composite_scenarios()

        # 如果超过限制，进行采样
        if max_scenarios and len(self.scenarios) > max_scenarios:
            self._sample_scenarios(max_scenarios)

        # 构建索引
        self._build_indices()

        return len(self.scenarios)

    def _generate_normal_scenarios(self):
        """生成正常运行场景"""

        # 流量×压力×温度×季节×时段×运行模式 组合
        flow_levels = [FlowLevel.MINIMAL, FlowLevel.LOW, FlowLevel.MEDIUM_LOW,
                      FlowLevel.MEDIUM, FlowLevel.DESIGN, FlowLevel.NORMAL,
                      FlowLevel.HIGH, FlowLevel.MAX_DESIGN]

        pressure_levels = [PressureLevel.LOW, PressureLevel.MEDIUM_LOW,
                          PressureLevel.MEDIUM, PressureLevel.DESIGN,
                          PressureLevel.HIGH]

        temperatures = [TemperatureLevel.COLD, TemperatureLevel.COOL,
                       TemperatureLevel.NORMAL, TemperatureLevel.WARM,
                       TemperatureLevel.HOT]

        seasons = list(SeasonType)
        times = list(TimeOfDay)
        modes = [OperationMode.NORMAL, OperationMode.STANDBY]
        tunnels = [1, 2]

        # 生成基础场景 (减少组合以避免过多)
        for flow in flow_levels:
            for pressure in pressure_levels:
                for temp in temperatures:
                    for season in seasons:
                        # 每个季节只取代表性时段
                        representative_times = [TimeOfDay.MORNING, TimeOfDay.AFTERNOON, TimeOfDay.NIGHT]
                        for time in representative_times:
                            for mode in modes:
                                for tunnel in tunnels:
                                    scenario = self._create_normal_scenario(
                                        flow, pressure, temp, season, time, mode, tunnel
                                    )
                                    self._add_scenario(scenario)

    def _create_normal_scenario(self, flow: FlowLevel, pressure: PressureLevel,
                                temp: TemperatureLevel, season: SeasonType,
                                time: TimeOfDay, mode: OperationMode,
                                tunnel: int) -> ScenarioParameters:
        """创建正常场景"""
        scenario = ScenarioParameters()

        # 基本参数
        scenario.flow_level = flow
        scenario.pressure_level = pressure
        scenario.temperature = temp
        scenario.season = season
        scenario.time_of_day = time
        scenario.operation_mode = mode
        scenario.tunnel_id = tunnel

        # 计算实际值
        scenario.flow_rate = self.design_flow * flow.value
        scenario.pressure_inlet = self.max_pressure * pressure.value
        scenario.pressure_outlet = scenario.pressure_inlet * 0.8
        scenario.velocity = scenario.flow_rate / 38.48  # A = πr² = 38.48 m²
        scenario.water_temperature = float(temp.value)

        # 阀门状态
        if mode == OperationMode.NORMAL:
            scenario.inlet_valve_state = ValveState.FULLY_OPEN
            scenario.outlet_valve_state = ValveState.FULLY_OPEN
        else:
            scenario.inlet_valve_state = ValveState.HALF_OPEN
            scenario.outlet_valve_state = ValveState.HALF_OPEN

        # 名称和描述
        scenario.name = f"NORM_{flow.name}_{pressure.name}_{season.value}_T{tunnel}"
        scenario.description = (f"正常运行: {flow.name}流量, {pressure.name}压力, "
                               f"{season.value}季, {tunnel}号洞")
        scenario.category = "normal"
        scenario.priority = "normal"

        return scenario

    def _generate_transition_scenarios(self):
        """生成过渡场景"""

        # 启动过渡
        startup_stages = [
            (0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
            (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)
        ]

        for i, (start, end) in enumerate(startup_stages):
            scenario = ScenarioParameters()
            scenario.name = f"STARTUP_STAGE_{i+1}"
            scenario.description = f"启动过渡阶段 {i+1}/10: {start*100:.0f}% → {end*100:.0f}%"
            scenario.operation_mode = OperationMode.STARTUP
            scenario.flow_rate = self.design_flow * end
            scenario.flow_level = FlowLevel.MEDIUM if end > 0.5 else FlowLevel.LOW
            scenario.category = "transition"
            scenario.priority = "high"
            self._add_scenario(scenario)

        # 停机过渡
        shutdown_stages = [(1.0 - s, 1.0 - e) for s, e in startup_stages]
        for i, (start, end) in enumerate(shutdown_stages):
            scenario = ScenarioParameters()
            scenario.name = f"SHUTDOWN_STAGE_{i+1}"
            scenario.description = f"停机过渡阶段 {i+1}/10: {start*100:.0f}% → {end*100:.0f}%"
            scenario.operation_mode = OperationMode.SHUTDOWN
            scenario.flow_rate = self.design_flow * end
            scenario.category = "transition"
            scenario.priority = "high"
            self._add_scenario(scenario)

        # 流量调节过渡
        flow_transitions = [
            (FlowLevel.NORMAL, FlowLevel.HIGH),
            (FlowLevel.NORMAL, FlowLevel.LOW),
            (FlowLevel.LOW, FlowLevel.HIGH),
            (FlowLevel.DESIGN, FlowLevel.MAX_DESIGN),
        ]

        for from_flow, to_flow in flow_transitions:
            # 10个过渡步骤
            for step in range(10):
                ratio = step / 9.0
                scenario = ScenarioParameters()
                scenario.name = f"TRANS_{from_flow.name}_TO_{to_flow.name}_S{step+1}"
                scenario.description = f"流量过渡: {from_flow.name} → {to_flow.name} 步骤{step+1}"
                scenario.flow_rate = self.design_flow * (
                    from_flow.value + ratio * (to_flow.value - from_flow.value)
                )
                scenario.operation_mode = OperationMode.TRANSITION
                scenario.category = "transition"
                scenario.priority = "normal"
                self._add_scenario(scenario)

    def _generate_fault_scenarios(self):
        """生成故障场景"""

        # 传感器故障
        sensor_faults = [
            FaultType.SENSOR_DRIFT, FaultType.SENSOR_NOISE,
            FaultType.SENSOR_STUCK, FaultType.SENSOR_BIAS,
            FaultType.SENSOR_FAILURE
        ]

        sensor_types = ['pressure', 'flow', 'temperature', 'vibration', 'das', 'dts']
        severity_levels = [0.1, 0.3, 0.5, 0.7, 0.9]  # 轻微到严重

        for fault in sensor_faults:
            for sensor in sensor_types:
                for severity in severity_levels:
                    for section in TunnelSection:
                        scenario = ScenarioParameters()
                        scenario.name = f"FAULT_{fault.name}_{sensor}_{section.name}_S{int(severity*100)}"
                        scenario.description = f"{fault.name}: {sensor}传感器在{section.value}段, 严重度{severity*100:.0f}%"
                        scenario.fault_types = [fault]
                        scenario.fault_locations = [section]
                        scenario.fault_severity = severity
                        scenario.category = "fault"
                        scenario.priority = "critical" if severity > 0.5 else "high"
                        self._add_scenario(scenario)

        # 执行器故障
        actuator_faults = [
            FaultType.VALVE_STUCK, FaultType.VALVE_LEAKAGE, FaultType.VALVE_SLOW,
            FaultType.PUMP_CAVITATION, FaultType.PUMP_VIBRATION, FaultType.PUMP_FAILURE
        ]

        for fault in actuator_faults:
            for severity in severity_levels:
                scenario = ScenarioParameters()
                scenario.name = f"FAULT_{fault.name}_S{int(severity*100)}"
                scenario.description = f"执行器故障 {fault.name}, 严重度{severity*100:.0f}%"
                scenario.fault_types = [fault]
                scenario.fault_severity = severity
                scenario.category = "fault"
                scenario.priority = "critical" if severity > 0.5 else "high"

                # 根据故障类型设置阀门状态
                if fault == FaultType.VALVE_STUCK:
                    stuck_positions = [0.0, 0.25, 0.5, 0.75, 1.0]
                    for pos in stuck_positions:
                        s = ScenarioParameters(**vars(scenario))
                        s.name += f"_POS{int(pos*100)}"
                        s.inlet_valve_state = ValveState(pos)
                        self._add_scenario(s)
                else:
                    self._add_scenario(scenario)

        # 结构故障
        structural_faults = [
            FaultType.LEAKAGE_MINOR, FaultType.LEAKAGE_MAJOR,
            FaultType.CRACK_DETECTED, FaultType.SETTLEMENT, FaultType.CORROSION
        ]

        leakage_rates = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]  # L/s/m

        for fault in structural_faults:
            for section in TunnelSection:
                for rate in leakage_rates:
                    scenario = ScenarioParameters()
                    scenario.name = f"STRUCT_{fault.name}_{section.name}_R{int(rate*100)}"
                    scenario.description = f"结构问题 {fault.name} 在{section.value}, 渗漏率{rate}L/s/m"
                    scenario.fault_types = [fault]
                    scenario.fault_locations = [section]
                    scenario.leakage_rate = rate
                    scenario.structural_health = 1.0 - min(0.9, rate * 0.1)
                    scenario.category = "fault"
                    scenario.priority = "critical" if rate > 0.5 else "high"
                    self._add_scenario(scenario)

        # 系统故障
        system_faults = [
            FaultType.POWER_FLUCTUATION, FaultType.POWER_FAILURE,
            FaultType.COMMUNICATION_LOSS, FaultType.CONTROL_FAILURE
        ]

        for fault in system_faults:
            for severity in severity_levels:
                scenario = ScenarioParameters()
                scenario.name = f"SYS_{fault.name}_S{int(severity*100)}"
                scenario.description = f"系统故障 {fault.name}, 严重度{severity*100:.0f}%"
                scenario.fault_types = [fault]
                scenario.fault_severity = severity
                scenario.category = "fault"
                scenario.priority = "critical"
                self._add_scenario(scenario)

    def _generate_external_event_scenarios(self):
        """生成外部事件场景"""

        # 洪水场景
        flood_levels = [
            (ExternalEvent.FLOOD_WARNING, 1.0),
            (ExternalEvent.FLOOD_LEVEL1, 1.1),
            (ExternalEvent.FLOOD_LEVEL2, 1.2),
        ]

        for event, flow_mult in flood_levels:
            for season in [SeasonType.SUMMER, SeasonType.AUTUMN]:  # 汛期
                scenario = ScenarioParameters()
                scenario.name = f"EXT_{event.name}_{season.value}"
                scenario.description = f"洪水事件 {event.value} ({season.value})"
                scenario.external_events = [event]
                scenario.flow_rate = self.max_flow * flow_mult
                scenario.flow_level = FlowLevel.OVERLOAD
                scenario.season = season
                scenario.category = "external"
                scenario.priority = "critical"
                self._add_scenario(scenario)

        # 地震场景
        earthquake_levels = [
            (ExternalEvent.EARTHQUAKE_III, 0.0),
            (ExternalEvent.EARTHQUAKE_V, 0.1),
            (ExternalEvent.EARTHQUAKE_VII, 0.3),
            (ExternalEvent.EARTHQUAKE_VIII, 0.5),
        ]

        for event, damage in earthquake_levels:
            for section in TunnelSection:
                scenario = ScenarioParameters()
                scenario.name = f"EXT_{event.name}_{section.name}"
                scenario.description = f"地震事件 {event.value} 影响{section.value}段"
                scenario.external_events = [event]
                scenario.structural_health = 1.0 - damage
                scenario.fault_locations = [section]
                scenario.category = "external"
                scenario.priority = "critical"
                self._add_scenario(scenario)

        # 其他外部事件
        other_events = [
            ExternalEvent.STORM, ExternalEvent.LIGHTNING,
            ExternalEvent.UPSTREAM_CHANGE, ExternalEvent.DOWNSTREAM_DEMAND,
            ExternalEvent.SCHEDULED_OUTAGE, ExternalEvent.EMERGENCY_SHUTOFF
        ]

        for event in other_events:
            scenario = ScenarioParameters()
            scenario.name = f"EXT_{event.name}"
            scenario.description = f"外部事件 {event.value}"
            scenario.external_events = [event]
            scenario.category = "external"
            scenario.priority = "high" if event in [ExternalEvent.EMERGENCY_SHUTOFF] else "normal"
            self._add_scenario(scenario)

        # 水质事件
        for quality in WaterQuality:
            sediment_levels = [0.0, 0.5, 1.0, 2.0, 5.0] if quality == WaterQuality.SEDIMENT_HIGH else [0.0]
            for sed in sediment_levels:
                scenario = ScenarioParameters()
                scenario.name = f"QUALITY_{quality.name}_SED{int(sed*10)}"
                scenario.description = f"水质状况 {quality.value}, 含沙量{sed}kg/m³"
                scenario.water_quality = quality
                scenario.sediment_concentration = sed
                scenario.category = "external"
                scenario.priority = "high" if quality in [WaterQuality.CONTAMINATED, WaterQuality.SEDIMENT_HIGH] else "normal"
                self._add_scenario(scenario)

    def _generate_composite_scenarios(self):
        """生成组合场景"""

        # 故障+运行状态组合
        base_faults = [FaultType.SENSOR_DRIFT, FaultType.VALVE_SLOW, FaultType.LEAKAGE_MINOR]
        flow_states = [FlowLevel.LOW, FlowLevel.NORMAL, FlowLevel.HIGH]

        for fault in base_faults:
            for flow in flow_states:
                scenario = ScenarioParameters()
                scenario.name = f"COMP_{fault.name}_{flow.name}"
                scenario.description = f"组合: {fault.name} + {flow.name}流量"
                scenario.fault_types = [fault]
                scenario.flow_level = flow
                scenario.flow_rate = self.design_flow * flow.value
                scenario.category = "composite"
                scenario.priority = "high"
                self._add_scenario(scenario)

        # 多重故障组合
        fault_combinations = list(combinations(base_faults, 2))
        for faults in fault_combinations:
            scenario = ScenarioParameters()
            scenario.name = f"MULTI_{'_'.join(f.name for f in faults)}"
            scenario.description = f"多重故障: {', '.join(f.name for f in faults)}"
            scenario.fault_types = list(faults)
            scenario.fault_severity = 0.5
            scenario.category = "composite"
            scenario.priority = "critical"
            self._add_scenario(scenario)

        # 外部事件+故障组合
        external_base = [ExternalEvent.FLOOD_WARNING, ExternalEvent.EARTHQUAKE_V]
        for event in external_base:
            for fault in base_faults:
                scenario = ScenarioParameters()
                scenario.name = f"COMP_{event.name}_{fault.name}"
                scenario.description = f"组合: {event.value} + {fault.name}"
                scenario.external_events = [event]
                scenario.fault_types = [fault]
                scenario.category = "composite"
                scenario.priority = "critical"
                self._add_scenario(scenario)

        # 季节+故障+运行模式组合
        for season in [SeasonType.SUMMER, SeasonType.WINTER]:
            for fault in base_faults:
                for mode in [OperationMode.NORMAL, OperationMode.TRANSITION]:
                    scenario = ScenarioParameters()
                    scenario.name = f"COMP_{season.value}_{fault.name}_{mode.value}"
                    scenario.description = f"组合: {season.value} + {fault.name} + {mode.value}"
                    scenario.season = season
                    scenario.fault_types = [fault]
                    scenario.operation_mode = mode
                    scenario.category = "composite"
                    scenario.priority = "high"
                    self._add_scenario(scenario)

        # 双洞不对称运行
        asymmetric_configs = [
            (FlowLevel.NORMAL, FlowLevel.LOW),
            (FlowLevel.HIGH, FlowLevel.MEDIUM),
            (FlowLevel.NORMAL, FlowLevel.ZERO),  # 单洞运行
        ]

        for flow1, flow2 in asymmetric_configs:
            scenario = ScenarioParameters()
            scenario.name = f"ASYM_T1_{flow1.name}_T2_{flow2.name}"
            scenario.description = f"不对称运行: 1号洞{flow1.name}, 2号洞{flow2.name}"
            scenario.flow_level = flow1
            scenario.flow_rate = self.design_flow * (flow1.value + flow2.value) / 2
            scenario.category = "composite"
            scenario.priority = "high"
            # 存储双洞配置
            scenario.parent_scenarios = [f"T1:{flow1.name}", f"T2:{flow2.name}"]
            self._add_scenario(scenario)

    def _add_scenario(self, scenario: ScenarioParameters):
        """添加场景到集合"""
        self.scenario_counter += 1
        scenario.scenario_id = f"SCN_{self.scenario_counter:06d}_{scenario.generate_id()}"
        self.scenarios[scenario.scenario_id] = scenario

    def _sample_scenarios(self, max_count: int):
        """采样场景（保持分布）"""
        if len(self.scenarios) <= max_count:
            return

        # 按类别分组
        by_category: Dict[str, List[str]] = {}
        for sid, s in self.scenarios.items():
            cat = s.category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(sid)

        # 按比例采样
        sampled = {}
        for cat, ids in by_category.items():
            ratio = len(ids) / len(self.scenarios)
            n_sample = max(1, int(max_count * ratio))
            selected = np.random.choice(ids, min(n_sample, len(ids)), replace=False)
            for sid in selected:
                sampled[sid] = self.scenarios[sid]

        self.scenarios = sampled

    def _build_indices(self):
        """构建索引"""
        self.index_by_category = {}
        self.index_by_priority = {}
        self.index_by_flow = {}
        self.index_by_fault = {}

        for sid, s in self.scenarios.items():
            # 按类别
            if s.category not in self.index_by_category:
                self.index_by_category[s.category] = []
            self.index_by_category[s.category].append(sid)

            # 按优先级
            if s.priority not in self.index_by_priority:
                self.index_by_priority[s.priority] = []
            self.index_by_priority[s.priority].append(sid)

            # 按流量
            flow_key = s.flow_level.name if s.flow_level else "UNKNOWN"
            if flow_key not in self.index_by_flow:
                self.index_by_flow[flow_key] = []
            self.index_by_flow[flow_key].append(sid)

            # 按故障
            for fault in s.fault_types:
                if fault.name not in self.index_by_fault:
                    self.index_by_fault[fault.name] = []
                self.index_by_fault[fault.name].append(sid)

    def get_statistics(self) -> Dict[str, Any]:
        """获取场景统计"""
        return {
            'total_scenarios': len(self.scenarios),
            'by_category': {k: len(v) for k, v in self.index_by_category.items()},
            'by_priority': {k: len(v) for k, v in self.index_by_priority.items()},
            'by_flow_level': {k: len(v) for k, v in self.index_by_flow.items()},
            'by_fault_type': {k: len(v) for k, v in self.index_by_fault.items()},
        }

    def query(self,
              category: Optional[str] = None,
              priority: Optional[str] = None,
              flow_level: Optional[str] = None,
              fault_type: Optional[str] = None,
              limit: int = 100) -> List[ScenarioParameters]:
        """查询场景"""
        candidates = set(self.scenarios.keys())

        if category and category in self.index_by_category:
            candidates &= set(self.index_by_category[category])

        if priority and priority in self.index_by_priority:
            candidates &= set(self.index_by_priority[priority])

        if flow_level and flow_level in self.index_by_flow:
            candidates &= set(self.index_by_flow[flow_level])

        if fault_type and fault_type in self.index_by_fault:
            candidates &= set(self.index_by_fault[fault_type])

        results = [self.scenarios[sid] for sid in list(candidates)[:limit]]
        return results

    def iterate_all(self) -> Iterator[ScenarioParameters]:
        """迭代所有场景"""
        for scenario in self.scenarios.values():
            yield scenario

    def export_to_json(self, filepath: str):
        """导出到JSON"""
        data = {
            'generated_at': datetime.now().isoformat(),
            'statistics': self.get_statistics(),
            'scenarios': {sid: s.to_dict() for sid, s in self.scenarios.items()}
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# ============================================================================
# 时序场景生成器
# ============================================================================

class TemporalScenarioGenerator:
    """
    时序场景生成器

    生成随时间演化的场景序列
    """

    def __init__(self, matrix_generator: ScenarioMatrixGenerator):
        self.matrix = matrix_generator

    def generate_daily_cycle(self, base_scenario: ScenarioParameters,
                             time_resolution_hours: float = 1.0) -> List[ScenarioParameters]:
        """生成日周期场景"""
        scenarios = []
        n_steps = int(24 / time_resolution_hours)

        for i in range(n_steps):
            hour = i * time_resolution_hours
            scenario = ScenarioParameters(**vars(base_scenario))

            # 确定时段
            for tod in TimeOfDay:
                if hour >= tod.value:
                    scenario.time_of_day = tod

            # 根据时段调整流量（模拟用水高峰）
            if 6 <= hour < 10 or 18 <= hour < 22:
                # 用水高峰
                scenario.flow_rate *= 1.1
            elif 0 <= hour < 6:
                # 夜间低谷
                scenario.flow_rate *= 0.8

            scenario.name = f"{base_scenario.name}_H{int(hour):02d}"
            scenario.description = f"{base_scenario.description} - {hour:.0f}时"
            scenarios.append(scenario)

        return scenarios

    def generate_seasonal_cycle(self, base_scenario: ScenarioParameters) -> List[ScenarioParameters]:
        """生成季节周期场景"""
        scenarios = []

        seasonal_factors = {
            SeasonType.SPRING: {'flow': 1.0, 'temp': 15.0},
            SeasonType.SUMMER: {'flow': 1.2, 'temp': 25.0},  # 汛期
            SeasonType.AUTUMN: {'flow': 1.1, 'temp': 18.0},
            SeasonType.WINTER: {'flow': 0.8, 'temp': 5.0},   # 枯水期
        }

        for season, factors in seasonal_factors.items():
            scenario = ScenarioParameters(**vars(base_scenario))
            scenario.season = season
            scenario.flow_rate = base_scenario.flow_rate * factors['flow']
            scenario.water_temperature = factors['temp']
            scenario.name = f"{base_scenario.name}_{season.value}"
            scenario.description = f"{base_scenario.description} - {season.value}"
            scenarios.append(scenario)

        return scenarios

    def generate_fault_evolution(self, fault_type: FaultType,
                                 initial_severity: float = 0.1,
                                 final_severity: float = 0.9,
                                 n_stages: int = 10) -> List[ScenarioParameters]:
        """生成故障演化场景序列"""
        scenarios = []

        for i in range(n_stages):
            severity = initial_severity + (final_severity - initial_severity) * i / (n_stages - 1)
            scenario = ScenarioParameters()
            scenario.fault_types = [fault_type]
            scenario.fault_severity = severity
            scenario.name = f"EVOLVE_{fault_type.name}_S{i+1}"
            scenario.description = f"{fault_type.name} 演化阶段 {i+1}/{n_stages}, 严重度{severity*100:.0f}%"
            scenario.category = "temporal"
            scenario.priority = "critical" if severity > 0.5 else "high"
            scenarios.append(scenario)

        return scenarios


# ============================================================================
# 场景覆盖分析器
# ============================================================================

class ScenarioCoverageAnalyzer:
    """
    场景覆盖分析器

    分析场景集合的覆盖完整性
    """

    def __init__(self, matrix_generator: ScenarioMatrixGenerator):
        self.matrix = matrix_generator

    def analyze_coverage(self) -> Dict[str, Any]:
        """分析覆盖率"""
        total = len(self.matrix.scenarios)

        # 维度覆盖
        flow_coverage = len(self.matrix.index_by_flow) / len(FlowLevel)
        fault_coverage = len(self.matrix.index_by_fault) / len(FaultType)

        # 类别分布
        category_distribution = {
            k: len(v) / total for k, v in self.matrix.index_by_category.items()
        }

        # 优先级分布
        priority_distribution = {
            k: len(v) / total for k, v in self.matrix.index_by_priority.items()
        }

        return {
            'total_scenarios': total,
            'flow_level_coverage': f"{flow_coverage*100:.1f}%",
            'fault_type_coverage': f"{fault_coverage*100:.1f}%",
            'category_distribution': category_distribution,
            'priority_distribution': priority_distribution,
            'dimensions': {
                'flow_levels': len(FlowLevel),
                'pressure_levels': len(PressureLevel),
                'temperature_levels': len(TemperatureLevel),
                'seasons': len(SeasonType),
                'time_periods': len(TimeOfDay),
                'operation_modes': len(OperationMode),
                'fault_types': len(FaultType),
                'external_events': len(ExternalEvent),
                'tunnel_sections': len(TunnelSection),
            },
            'theoretical_max': self._calculate_theoretical_max(),
        }

    def _calculate_theoretical_max(self) -> int:
        """计算理论最大场景数"""
        # 简化计算：主要维度的笛卡尔积
        return (
            len(FlowLevel) *
            len(PressureLevel) *
            len(TemperatureLevel) *
            len(SeasonType) *
            len(TimeOfDay) *
            len(OperationMode) *
            2  # 双洞
        )

    def find_gaps(self) -> Dict[str, List[str]]:
        """找出覆盖空白"""
        gaps = {}

        # 检查流量等级
        covered_flows = set(self.matrix.index_by_flow.keys())
        all_flows = {f.name for f in FlowLevel}
        missing_flows = all_flows - covered_flows
        if missing_flows:
            gaps['missing_flow_levels'] = list(missing_flows)

        # 检查故障类型
        covered_faults = set(self.matrix.index_by_fault.keys())
        all_faults = {f.name for f in FaultType if f != FaultType.NONE}
        missing_faults = all_faults - covered_faults
        if missing_faults:
            gaps['missing_fault_types'] = list(missing_faults)

        return gaps


# ============================================================================
# 便捷函数
# ============================================================================

def create_full_scenario_matrix(max_scenarios: Optional[int] = None) -> ScenarioMatrixGenerator:
    """创建完整场景矩阵"""
    generator = ScenarioMatrixGenerator()
    generator.generate_all(max_scenarios=max_scenarios)
    return generator


def get_scenario_summary(generator: ScenarioMatrixGenerator) -> str:
    """获取场景摘要"""
    stats = generator.get_statistics()
    lines = [
        "=" * 60,
        "穿黄工程全场景矩阵摘要",
        "=" * 60,
        f"场景总数: {stats['total_scenarios']:,}",
        "",
        "按类别分布:",
    ]
    for cat, count in stats['by_category'].items():
        lines.append(f"  {cat}: {count:,}")

    lines.extend([
        "",
        "按优先级分布:",
    ])
    for pri, count in stats['by_priority'].items():
        lines.append(f"  {pri}: {count:,}")

    lines.append("=" * 60)
    return "\n".join(lines)
