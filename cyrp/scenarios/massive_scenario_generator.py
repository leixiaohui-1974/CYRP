"""
Massive Scenario Generator for CYRP
穿黄工程超大规模场景生成器

生成数万至数十万种场景，覆盖所有可能的运行工况
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Iterator, Any, Generator
from enum import Enum, auto
from itertools import product, combinations, permutations
import hashlib
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

from cyrp.scenarios.full_scenario_matrix import (
    ScenarioParameters, FlowLevel, PressureLevel, TemperatureLevel,
    SeasonType, TimeOfDay, OperationMode, ValveState, PumpState,
    FaultType, ExternalEvent, WaterQuality, TunnelSection
)


# ============================================================================
# 参数扫描定义
# ============================================================================

@dataclass
class ParameterSweep:
    """参数扫描配置"""
    name: str
    values: List[Any]
    weight: float = 1.0  # 采样权重
    required: bool = False  # 是否必须包含


class ParameterSpace:
    """参数空间定义"""

    def __init__(self):
        # 连续参数的离散化
        self.flow_rates = np.linspace(0, 350, 36).tolist()  # 0-350 m³/s, 10 m³/s步长
        self.pressures = np.linspace(0, 1200, 25).tolist()  # 0-1200 kPa, 50 kPa步长
        self.temperatures = np.linspace(-5, 35, 21).tolist()  # -5 to 35°C, 2°C步长
        self.valve_positions = np.linspace(0, 1, 21).tolist()  # 0-100%, 5%步长
        self.fault_severities = np.linspace(0, 1, 11).tolist()  # 0-100%, 10%步长

        # 时间参数
        self.hours = list(range(24))
        self.months = list(range(1, 13))

        # 位置参数
        self.tunnel_positions = np.linspace(0, 4250, 86).tolist()  # 50m间隔

    def get_dimension_sizes(self) -> Dict[str, int]:
        """获取各维度大小"""
        return {
            'flow_rates': len(self.flow_rates),
            'pressures': len(self.pressures),
            'temperatures': len(self.temperatures),
            'valve_positions': len(self.valve_positions),
            'fault_severities': len(self.fault_severities),
            'hours': len(self.hours),
            'months': len(self.months),
            'tunnel_positions': len(self.tunnel_positions),
            'flow_levels': len(FlowLevel),
            'pressure_levels': len(PressureLevel),
            'temperature_levels': len(TemperatureLevel),
            'seasons': len(SeasonType),
            'time_periods': len(TimeOfDay),
            'operation_modes': len(OperationMode),
            'valve_states': len(ValveState),
            'pump_states': len(PumpState),
            'fault_types': len(FaultType),
            'external_events': len(ExternalEvent),
            'water_quality': len(WaterQuality),
            'tunnel_sections': len(TunnelSection),
        }


# ============================================================================
# 超大规模场景生成器
# ============================================================================

class MassiveScenarioGenerator:
    """
    超大规模场景生成器

    通过多维参数组合生成数万种场景
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.param_space = ParameterSpace()
        self.design_flow = 265.0
        self.max_flow = 320.0
        self.max_pressure = 1000.0

        # 场景存储
        self.scenarios: Dict[str, ScenarioParameters] = {}
        self.scenario_index = 0

        # 生成配置
        self.generation_config = {
            'normal_full_sweep': True,
            'transition_detailed': True,
            'fault_comprehensive': True,
            'composite_exhaustive': True,
            'temporal_fine': True,
            'extreme_cases': True,
        }

    def generate_massive(self,
                        target_count: int = 50000,
                        balance_categories: bool = True) -> int:
        """
        生成大规模场景集

        Args:
            target_count: 目标场景数量
            balance_categories: 是否平衡各类别

        Returns:
            实际生成数量
        """
        category_targets = self._calculate_category_targets(target_count, balance_categories)

        # 1. 正常运行场景 (参数全扫描)
        self._generate_normal_full_sweep(category_targets.get('normal', 20000))

        # 2. 过渡场景 (细粒度)
        self._generate_transition_detailed(category_targets.get('transition', 5000))

        # 3. 故障场景 (全面覆盖)
        self._generate_fault_comprehensive(category_targets.get('fault', 10000))

        # 4. 外部事件场景
        self._generate_external_comprehensive(category_targets.get('external', 3000))

        # 5. 组合场景 (穷举)
        self._generate_composite_exhaustive(category_targets.get('composite', 10000))

        # 6. 时序场景
        self._generate_temporal_fine(category_targets.get('temporal', 2000))

        # 7. 极端场景
        self._generate_extreme_cases()

        return len(self.scenarios)

    def _calculate_category_targets(self, total: int, balance: bool) -> Dict[str, int]:
        """计算各类别目标数量"""
        if balance:
            return {
                'normal': int(total * 0.4),
                'transition': int(total * 0.1),
                'fault': int(total * 0.2),
                'external': int(total * 0.05),
                'composite': int(total * 0.2),
                'temporal': int(total * 0.05),
            }
        else:
            # 不平衡时正常场景占多数
            return {
                'normal': int(total * 0.6),
                'transition': int(total * 0.05),
                'fault': int(total * 0.15),
                'external': int(total * 0.05),
                'composite': int(total * 0.1),
                'temporal': int(total * 0.05),
            }

    def _generate_normal_full_sweep(self, target: int):
        """全参数扫描正常场景"""
        count = 0

        # 主要维度组合
        for flow_rate in self.param_space.flow_rates:
            if flow_rate <= 0:
                continue
            for pressure in self.param_space.pressures[5:20]:  # 合理压力范围
                for temp in self.param_space.temperatures[5:15]:  # 常见温度范围
                    for month in [1, 4, 7, 10]:  # 季节代表月份
                        for hour in [0, 6, 12, 18]:  # 代表时刻
                            for tunnel in [1, 2]:
                                if count >= target:
                                    return

                                scenario = self._create_scenario(
                                    category='normal',
                                    flow_rate=flow_rate,
                                    pressure_inlet=pressure,
                                    water_temperature=temp,
                                    month=month,
                                    hour=hour,
                                    tunnel_id=tunnel
                                )
                                self._add_scenario(scenario)
                                count += 1

    def _generate_transition_detailed(self, target: int):
        """详细过渡场景"""
        count = 0

        # 启动过程 - 100个阶段
        for stage in range(100):
            if count >= target // 3:
                break
            ratio = stage / 99.0
            scenario = self._create_scenario(
                category='transition',
                flow_rate=self.design_flow * ratio,
                operation_mode=OperationMode.STARTUP,
                name=f"STARTUP_FINE_{stage:03d}",
                description=f"精细启动过程 {stage+1}/100, 流量{ratio*100:.1f}%"
            )
            self._add_scenario(scenario)
            count += 1

        # 停机过程 - 100个阶段
        for stage in range(100):
            if count >= target * 2 // 3:
                break
            ratio = 1.0 - stage / 99.0
            scenario = self._create_scenario(
                category='transition',
                flow_rate=self.design_flow * ratio,
                operation_mode=OperationMode.SHUTDOWN,
                name=f"SHUTDOWN_FINE_{stage:03d}",
                description=f"精细停机过程 {stage+1}/100, 流量{ratio*100:.1f}%"
            )
            self._add_scenario(scenario)
            count += 1

        # 流量调节过渡
        flow_transitions = []
        for f1 in self.param_space.flow_rates[::5]:
            for f2 in self.param_space.flow_rates[::5]:
                if f1 != f2:
                    flow_transitions.append((f1, f2))

        for from_flow, to_flow in flow_transitions:
            if count >= target:
                break
            # 每个过渡20个阶段
            for stage in range(20):
                ratio = stage / 19.0
                current_flow = from_flow + ratio * (to_flow - from_flow)
                scenario = self._create_scenario(
                    category='transition',
                    flow_rate=current_flow,
                    operation_mode=OperationMode.TRANSITION,
                    name=f"FLOW_TRANS_{int(from_flow)}_{int(to_flow)}_S{stage:02d}",
                    description=f"流量过渡 {from_flow:.0f}→{to_flow:.0f} m³/s, 阶段{stage+1}"
                )
                self._add_scenario(scenario)
                count += 1

        # 阀门调节过渡
        for valve_from in self.param_space.valve_positions[::4]:
            for valve_to in self.param_space.valve_positions[::4]:
                if valve_from != valve_to and count < target:
                    for stage in range(10):
                        ratio = stage / 9.0
                        current_pos = valve_from + ratio * (valve_to - valve_from)
                        scenario = self._create_scenario(
                            category='transition',
                            inlet_valve=current_pos,
                            name=f"VALVE_TRANS_{int(valve_from*100)}_{int(valve_to*100)}_S{stage}",
                            description=f"阀门调节 {valve_from*100:.0f}%→{valve_to*100:.0f}%"
                        )
                        self._add_scenario(scenario)
                        count += 1

    def _generate_fault_comprehensive(self, target: int):
        """全面故障场景"""
        count = 0

        # 所有故障类型
        sensor_faults = [
            FaultType.SENSOR_DRIFT, FaultType.SENSOR_NOISE,
            FaultType.SENSOR_STUCK, FaultType.SENSOR_BIAS, FaultType.SENSOR_FAILURE
        ]
        actuator_faults = [
            FaultType.VALVE_STUCK, FaultType.VALVE_LEAKAGE, FaultType.VALVE_SLOW,
            FaultType.PUMP_CAVITATION, FaultType.PUMP_VIBRATION, FaultType.PUMP_FAILURE
        ]
        structural_faults = [
            FaultType.LEAKAGE_MINOR, FaultType.LEAKAGE_MAJOR,
            FaultType.CRACK_DETECTED, FaultType.SETTLEMENT, FaultType.CORROSION
        ]
        system_faults = [
            FaultType.POWER_FLUCTUATION, FaultType.POWER_FAILURE,
            FaultType.COMMUNICATION_LOSS, FaultType.CONTROL_FAILURE
        ]

        # 传感器故障 - 每种类型×位置×严重度×传感器类型
        sensor_types = ['pressure', 'flow', 'temperature', 'vibration', 'strain', 'das', 'dts']
        for fault in sensor_faults:
            for sensor in sensor_types:
                for section in TunnelSection:
                    for severity in self.param_space.fault_severities:
                        if count >= target * 0.3:
                            break
                        scenario = self._create_scenario(
                            category='fault',
                            fault_types=[fault],
                            fault_severity=severity,
                            fault_section=section,
                            name=f"SENSOR_{fault.name}_{sensor}_{section.name}_S{int(severity*100)}",
                            description=f"传感器故障: {sensor}在{section.value}, {fault.name}, 严重度{severity*100:.0f}%",
                            priority='critical' if severity > 0.5 else 'high'
                        )
                        self._add_scenario(scenario)
                        count += 1

        # 执行器故障
        for fault in actuator_faults:
            for severity in self.param_space.fault_severities:
                # 阀门卡死在不同位置
                if fault == FaultType.VALVE_STUCK:
                    for stuck_pos in self.param_space.valve_positions:
                        if count >= target * 0.5:
                            break
                        scenario = self._create_scenario(
                            category='fault',
                            fault_types=[fault],
                            fault_severity=severity,
                            inlet_valve=stuck_pos,
                            name=f"VALVE_STUCK_POS{int(stuck_pos*100)}_S{int(severity*100)}",
                            description=f"阀门卡死在{stuck_pos*100:.0f}%, 严重度{severity*100:.0f}%",
                            priority='critical'
                        )
                        self._add_scenario(scenario)
                        count += 1
                else:
                    if count < target * 0.5:
                        scenario = self._create_scenario(
                            category='fault',
                            fault_types=[fault],
                            fault_severity=severity,
                            name=f"ACTUATOR_{fault.name}_S{int(severity*100)}",
                            description=f"执行器故障: {fault.name}, 严重度{severity*100:.0f}%",
                            priority='critical' if severity > 0.5 else 'high'
                        )
                        self._add_scenario(scenario)
                        count += 1

        # 结构故障 - 每个位置的不同程度
        leakage_rates = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
        for fault in structural_faults:
            for section in TunnelSection:
                for position in self.param_space.tunnel_positions[::10]:  # 每500m
                    for rate in leakage_rates:
                        if count >= target * 0.8:
                            break
                        scenario = self._create_scenario(
                            category='fault',
                            fault_types=[fault],
                            fault_section=section,
                            leakage_rate=rate,
                            name=f"STRUCT_{fault.name}_{section.name}_P{int(position)}_R{int(rate*1000)}",
                            description=f"结构故障: {fault.name}在{section.value} {position:.0f}m处, 渗漏{rate}L/s/m",
                            priority='critical' if rate > 0.1 else 'high'
                        )
                        self._add_scenario(scenario)
                        count += 1

        # 系统故障
        for fault in system_faults:
            for severity in self.param_space.fault_severities:
                if count < target:
                    scenario = self._create_scenario(
                        category='fault',
                        fault_types=[fault],
                        fault_severity=severity,
                        name=f"SYSTEM_{fault.name}_S{int(severity*100)}",
                        description=f"系统故障: {fault.name}, 严重度{severity*100:.0f}%",
                        priority='critical'
                    )
                    self._add_scenario(scenario)
                    count += 1

    def _generate_external_comprehensive(self, target: int):
        """全面外部事件场景"""
        count = 0

        # 洪水场景 - 不同水位、流量组合
        flood_flows = np.linspace(300, 400, 11).tolist()  # 超设计流量
        flood_levels = [1, 2, 3]  # 洪水等级

        for level in flood_levels:
            for flow in flood_flows:
                for section in TunnelSection:
                    if count >= target * 0.3:
                        break
                    event = [ExternalEvent.FLOOD_WARNING, ExternalEvent.FLOOD_LEVEL1,
                            ExternalEvent.FLOOD_LEVEL2][level-1]
                    scenario = self._create_scenario(
                        category='external',
                        external_events=[event],
                        flow_rate=flow,
                        fault_section=section,
                        name=f"FLOOD_L{level}_F{int(flow)}_{section.name}",
                        description=f"{level}级洪水, 流量{flow:.0f}m³/s, 影响{section.value}",
                        priority='critical'
                    )
                    self._add_scenario(scenario)
                    count += 1

        # 地震场景 - 不同烈度×位置×结构损伤
        earthquake_levels = [
            (ExternalEvent.EARTHQUAKE_III, 0.0, 0.05),
            (ExternalEvent.EARTHQUAKE_V, 0.05, 0.15),
            (ExternalEvent.EARTHQUAKE_VII, 0.15, 0.35),
            (ExternalEvent.EARTHQUAKE_VIII, 0.35, 0.6),
        ]

        for event, min_damage, max_damage in earthquake_levels:
            damages = np.linspace(min_damage, max_damage, 5).tolist()
            for damage in damages:
                for section in TunnelSection:
                    if count >= target * 0.6:
                        break
                    scenario = self._create_scenario(
                        category='external',
                        external_events=[event],
                        structural_health=1.0 - damage,
                        fault_section=section,
                        name=f"QUAKE_{event.name}_{section.name}_D{int(damage*100)}",
                        description=f"地震{event.value}, {section.value}段损伤{damage*100:.0f}%",
                        priority='critical'
                    )
                    self._add_scenario(scenario)
                    count += 1

        # 其他外部事件
        other_events = [
            ExternalEvent.STORM, ExternalEvent.LIGHTNING,
            ExternalEvent.UPSTREAM_CHANGE, ExternalEvent.DOWNSTREAM_DEMAND,
            ExternalEvent.SCHEDULED_OUTAGE, ExternalEvent.EMERGENCY_SHUTOFF
        ]

        for event in other_events:
            # 与不同运行状态组合
            for flow in self.param_space.flow_rates[::5]:
                if count >= target:
                    break
                scenario = self._create_scenario(
                    category='external',
                    external_events=[event],
                    flow_rate=flow,
                    name=f"EXT_{event.name}_F{int(flow)}",
                    description=f"外部事件: {event.value}, 流量{flow:.0f}m³/s",
                    priority='high'
                )
                self._add_scenario(scenario)
                count += 1

        # 水质事件
        for quality in WaterQuality:
            for sediment in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
                for flow in self.param_space.flow_rates[::10]:
                    if count >= target:
                        break
                    scenario = self._create_scenario(
                        category='external',
                        water_quality=quality,
                        sediment=sediment,
                        flow_rate=flow,
                        name=f"QUALITY_{quality.name}_SED{int(sediment*10)}_F{int(flow)}",
                        description=f"水质{quality.value}, 含沙{sediment}kg/m³, 流量{flow:.0f}m³/s",
                        priority='high' if quality in [WaterQuality.CONTAMINATED] else 'normal'
                    )
                    self._add_scenario(scenario)
                    count += 1

    def _generate_composite_exhaustive(self, target: int):
        """穷举组合场景"""
        count = 0

        # 1. 双重故障组合
        all_faults = list(FaultType)
        fault_pairs = list(combinations([f for f in all_faults if f != FaultType.NONE], 2))

        for fault1, fault2 in fault_pairs:
            for severity in [0.3, 0.5, 0.7]:
                if count >= target * 0.3:
                    break
                scenario = self._create_scenario(
                    category='composite',
                    fault_types=[fault1, fault2],
                    fault_severity=severity,
                    name=f"DUAL_{fault1.name}_{fault2.name}_S{int(severity*100)}",
                    description=f"双重故障: {fault1.name}+{fault2.name}, 严重度{severity*100:.0f}%",
                    priority='critical'
                )
                self._add_scenario(scenario)
                count += 1

        # 2. 三重故障组合 (采样)
        fault_triples = list(combinations([f for f in all_faults if f != FaultType.NONE], 3))
        sampled_triples = self.rng.choice(len(fault_triples), min(500, len(fault_triples)), replace=False)

        for idx in sampled_triples:
            if count >= target * 0.4:
                break
            faults = fault_triples[idx]
            scenario = self._create_scenario(
                category='composite',
                fault_types=list(faults),
                fault_severity=0.5,
                name=f"TRIPLE_{'_'.join(f.name for f in faults)}",
                description=f"三重故障: {'+'.join(f.name for f in faults)}",
                priority='critical'
            )
            self._add_scenario(scenario)
            count += 1

        # 3. 故障+外部事件组合
        external_events = [e for e in ExternalEvent if e != ExternalEvent.NONE]
        faults_subset = [FaultType.SENSOR_DRIFT, FaultType.VALVE_SLOW,
                        FaultType.LEAKAGE_MINOR, FaultType.POWER_FLUCTUATION]

        for event in external_events:
            for fault in faults_subset:
                for severity in [0.3, 0.5, 0.7]:
                    if count >= target * 0.55:
                        break
                    scenario = self._create_scenario(
                        category='composite',
                        external_events=[event],
                        fault_types=[fault],
                        fault_severity=severity,
                        name=f"COMP_{event.name}_{fault.name}_S{int(severity*100)}",
                        description=f"组合: {event.value}+{fault.name}, 严重度{severity*100:.0f}%",
                        priority='critical'
                    )
                    self._add_scenario(scenario)
                    count += 1

        # 4. 故障+运行状态组合
        for fault in faults_subset:
            for flow in self.param_space.flow_rates[::7]:
                for pressure in self.param_space.pressures[5::5]:
                    if count >= target * 0.7:
                        break
                    scenario = self._create_scenario(
                        category='composite',
                        fault_types=[fault],
                        flow_rate=flow,
                        pressure_inlet=pressure,
                        name=f"FAULT_STATE_{fault.name}_F{int(flow)}_P{int(pressure)}",
                        description=f"故障运行: {fault.name}, 流量{flow:.0f}m³/s, 压力{pressure:.0f}kPa",
                        priority='high'
                    )
                    self._add_scenario(scenario)
                    count += 1

        # 5. 双洞不对称运行
        for flow1 in self.param_space.flow_rates[::5]:
            for flow2 in self.param_space.flow_rates[::5]:
                if abs(flow1 - flow2) > 20:  # 明显不对称
                    if count >= target * 0.8:
                        break
                    asymmetry = abs(flow1 - flow2) / max(flow1, flow2, 1) * 100
                    scenario = self._create_scenario(
                        category='composite',
                        flow_rate=(flow1 + flow2) / 2,
                        name=f"ASYM_T1_{int(flow1)}_T2_{int(flow2)}",
                        description=f"不对称运行: 1号洞{flow1:.0f}m³/s, 2号洞{flow2:.0f}m³/s, 不对称度{asymmetry:.0f}%",
                        priority='high' if asymmetry > 30 else 'normal'
                    )
                    self._add_scenario(scenario)
                    count += 1

        # 6. 季节+故障+时段组合
        for season in SeasonType:
            for fault in faults_subset:
                for hour in [0, 6, 12, 18]:
                    for mode in [OperationMode.NORMAL, OperationMode.TRANSITION]:
                        if count >= target:
                            break
                        scenario = self._create_scenario(
                            category='composite',
                            season=season,
                            fault_types=[fault],
                            hour=hour,
                            operation_mode=mode,
                            name=f"SEASONAL_{season.value}_{fault.name}_H{hour}_{mode.value}",
                            description=f"季节组合: {season.value}+{fault.name}+{hour}时+{mode.value}",
                            priority='high'
                        )
                        self._add_scenario(scenario)
                        count += 1

    def _generate_temporal_fine(self, target: int):
        """细粒度时序场景"""
        count = 0

        # 1. 24小时×365天 场景 (采样)
        base_flows = [200, 265, 300]  # 代表性流量

        for base_flow in base_flows:
            for month in range(1, 13):
                for hour in range(24):
                    if count >= target * 0.5:
                        break
                    # 根据月份和时段调整流量
                    seasonal_factor = self._get_seasonal_factor(month)
                    daily_factor = self._get_daily_factor(hour)
                    flow = base_flow * seasonal_factor * daily_factor

                    scenario = self._create_scenario(
                        category='temporal',
                        flow_rate=flow,
                        month=month,
                        hour=hour,
                        name=f"TEMPORAL_M{month:02d}_H{hour:02d}_F{int(base_flow)}",
                        description=f"时序: {month}月{hour}时, 基准流量{base_flow}→实际{flow:.0f}m³/s"
                    )
                    self._add_scenario(scenario)
                    count += 1

        # 2. 故障演化序列
        evolving_faults = [FaultType.LEAKAGE_MINOR, FaultType.SENSOR_DRIFT, FaultType.CORROSION]

        for fault in evolving_faults:
            for n_stages in [20, 50, 100]:
                for stage in range(n_stages):
                    if count >= target:
                        break
                    severity = stage / (n_stages - 1)
                    scenario = self._create_scenario(
                        category='temporal',
                        fault_types=[fault],
                        fault_severity=severity,
                        name=f"EVOLVE_{fault.name}_N{n_stages}_S{stage:03d}",
                        description=f"{fault.name}演化 {stage+1}/{n_stages}, 严重度{severity*100:.1f}%",
                        priority='critical' if severity > 0.7 else 'high' if severity > 0.3 else 'normal'
                    )
                    self._add_scenario(scenario)
                    count += 1

    def _generate_extreme_cases(self):
        """生成极端场景"""

        extreme_scenarios = [
            # 极端流量
            {'flow_rate': 0.0, 'name': 'EXTREME_ZERO_FLOW', 'desc': '零流量'},
            {'flow_rate': 400.0, 'name': 'EXTREME_MAX_FLOW', 'desc': '极限流量400m³/s'},

            # 极端压力
            {'pressure_inlet': -50.0, 'name': 'EXTREME_NEGATIVE_PRESSURE', 'desc': '负压-50kPa'},
            {'pressure_inlet': 1200.0, 'name': 'EXTREME_MAX_PRESSURE', 'desc': '极限压力1200kPa'},

            # 极端温度
            {'water_temperature': -2.0, 'name': 'EXTREME_FREEZING', 'desc': '冰点以下'},
            {'water_temperature': 35.0, 'name': 'EXTREME_HIGH_TEMP', 'desc': '高温35°C'},

            # 极端阀门状态
            {'inlet_valve': 0.0, 'outlet_valve': 1.0, 'name': 'EXTREME_VALVE_MISMATCH', 'desc': '阀门状态不匹配'},

            # 全故障
            {'fault_types': list(FaultType)[1:6], 'fault_severity': 0.9, 'name': 'EXTREME_MULTI_FAULT', 'desc': '多重严重故障'},

            # 最恶劣组合
            {'flow_rate': 350.0, 'fault_types': [FaultType.LEAKAGE_MAJOR], 'external_events': [ExternalEvent.EARTHQUAKE_VIII],
             'name': 'EXTREME_WORST_CASE', 'desc': '最恶劣组合: 高流量+严重渗漏+强震'},
        ]

        for config in extreme_scenarios:
            scenario = self._create_scenario(
                category='extreme',
                priority='critical',
                **config
            )
            self._add_scenario(scenario)

    def _create_scenario(self, category: str, **kwargs) -> ScenarioParameters:
        """创建场景"""
        scenario = ScenarioParameters()
        scenario.category = category

        # 设置参数
        if 'flow_rate' in kwargs:
            scenario.flow_rate = kwargs['flow_rate']
            # 确定流量等级
            ratio = kwargs['flow_rate'] / self.design_flow
            if ratio <= 0:
                scenario.flow_level = FlowLevel.ZERO
            elif ratio <= 0.2:
                scenario.flow_level = FlowLevel.MINIMAL
            elif ratio <= 0.5:
                scenario.flow_level = FlowLevel.LOW
            elif ratio <= 0.8:
                scenario.flow_level = FlowLevel.MEDIUM
            elif ratio <= 1.0:
                scenario.flow_level = FlowLevel.NORMAL
            elif ratio <= 1.2:
                scenario.flow_level = FlowLevel.HIGH
            else:
                scenario.flow_level = FlowLevel.OVERLOAD

        if 'pressure_inlet' in kwargs:
            scenario.pressure_inlet = kwargs['pressure_inlet']
            scenario.pressure_outlet = kwargs['pressure_inlet'] * 0.8

        if 'water_temperature' in kwargs:
            scenario.water_temperature = kwargs['water_temperature']

        if 'inlet_valve' in kwargs:
            scenario.inlet_valve_state = self._nearest_valve_state(kwargs['inlet_valve'])
        if 'outlet_valve' in kwargs:
            scenario.outlet_valve_state = self._nearest_valve_state(kwargs['outlet_valve'])

        if 'fault_types' in kwargs:
            scenario.fault_types = kwargs['fault_types']
        if 'fault_severity' in kwargs:
            scenario.fault_severity = kwargs['fault_severity']
        if 'fault_section' in kwargs:
            scenario.fault_locations = [kwargs['fault_section']]

        if 'external_events' in kwargs:
            scenario.external_events = kwargs['external_events']

        if 'season' in kwargs:
            scenario.season = kwargs['season']
        if 'month' in kwargs:
            month = kwargs['month']
            if month in [3, 4, 5]:
                scenario.season = SeasonType.SPRING
            elif month in [6, 7, 8]:
                scenario.season = SeasonType.SUMMER
            elif month in [9, 10, 11]:
                scenario.season = SeasonType.AUTUMN
            else:
                scenario.season = SeasonType.WINTER

        if 'hour' in kwargs:
            hour = kwargs['hour']
            for tod in TimeOfDay:
                if hour >= tod.value:
                    scenario.time_of_day = tod

        if 'operation_mode' in kwargs:
            scenario.operation_mode = kwargs['operation_mode']

        if 'tunnel_id' in kwargs:
            scenario.tunnel_id = kwargs['tunnel_id']

        if 'structural_health' in kwargs:
            scenario.structural_health = kwargs['structural_health']

        if 'leakage_rate' in kwargs:
            scenario.leakage_rate = kwargs['leakage_rate']

        if 'water_quality' in kwargs:
            scenario.water_quality = kwargs['water_quality']
        if 'sediment' in kwargs:
            scenario.sediment_concentration = kwargs['sediment']

        if 'name' in kwargs:
            scenario.name = kwargs['name']
        if 'description' in kwargs or 'desc' in kwargs:
            scenario.description = kwargs.get('description', kwargs.get('desc', ''))

        if 'priority' in kwargs:
            scenario.priority = kwargs['priority']
        else:
            scenario.priority = 'normal'

        return scenario

    def _nearest_valve_state(self, value: float) -> ValveState:
        """找到最接近的阀门状态"""
        valve_values = [
            (0.0, ValveState.FULLY_CLOSED),
            (0.1, ValveState.NEARLY_CLOSED),
            (0.25, ValveState.QUARTER_OPEN),
            (0.5, ValveState.HALF_OPEN),
            (0.75, ValveState.THREE_QUARTER_OPEN),
            (0.9, ValveState.NEARLY_OPEN),
            (1.0, ValveState.FULLY_OPEN),
        ]
        # 找最近的
        best = valve_values[0][1]
        best_dist = abs(value - valve_values[0][0])
        for v, state in valve_values[1:]:
            dist = abs(value - v)
            if dist < best_dist:
                best_dist = dist
                best = state
        return best

    def _add_scenario(self, scenario: ScenarioParameters):
        """添加场景"""
        self.scenario_index += 1
        scenario.scenario_id = f"MSC_{self.scenario_index:07d}_{scenario.generate_id()}"
        if not scenario.name:
            scenario.name = f"SCENARIO_{self.scenario_index:07d}"
        self.scenarios[scenario.scenario_id] = scenario

    def _get_seasonal_factor(self, month: int) -> float:
        """获取季节流量系数"""
        # 夏季汛期流量大，冬季枯水期流量小
        factors = {
            1: 0.7, 2: 0.7, 3: 0.8, 4: 0.9,
            5: 1.0, 6: 1.1, 7: 1.2, 8: 1.2,
            9: 1.1, 10: 1.0, 11: 0.9, 12: 0.8
        }
        return factors.get(month, 1.0)

    def _get_daily_factor(self, hour: int) -> float:
        """获取日内流量系数"""
        # 用水高峰在早晚
        if 6 <= hour < 10 or 18 <= hour < 22:
            return 1.1
        elif 0 <= hour < 6:
            return 0.8
        else:
            return 1.0

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        by_category = {}
        by_priority = {}

        for s in self.scenarios.values():
            by_category[s.category] = by_category.get(s.category, 0) + 1
            by_priority[s.priority] = by_priority.get(s.priority, 0) + 1

        return {
            'total': len(self.scenarios),
            'by_category': by_category,
            'by_priority': by_priority,
        }

    def iterate(self, category: Optional[str] = None) -> Iterator[ScenarioParameters]:
        """迭代场景"""
        for s in self.scenarios.values():
            if category is None or s.category == category:
                yield s

    def export_summary(self) -> str:
        """导出摘要"""
        stats = self.get_statistics()
        lines = [
            "=" * 70,
            "穿黄工程超大规模场景集摘要",
            "=" * 70,
            f"场景总数: {stats['total']:,}",
            "",
            "按类别分布:",
        ]
        for cat, count in sorted(stats['by_category'].items(), key=lambda x: -x[1]):
            pct = count / stats['total'] * 100
            lines.append(f"  {cat}: {count:,} ({pct:.1f}%)")

        lines.extend(["", "按优先级分布:"])
        for pri, count in sorted(stats['by_priority'].items()):
            pct = count / stats['total'] * 100
            lines.append(f"  {pri}: {count:,} ({pct:.1f}%)")

        lines.append("=" * 70)
        return "\n".join(lines)


# ============================================================================
# 便捷函数
# ============================================================================

def generate_massive_scenarios(target: int = 50000) -> MassiveScenarioGenerator:
    """生成大规模场景集"""
    generator = MassiveScenarioGenerator()
    generator.generate_massive(target_count=target)
    return generator


def get_scenario_count_estimate() -> Dict[str, int]:
    """估算各配置下的场景数量"""
    space = ParameterSpace()
    dims = space.get_dimension_sizes()

    estimates = {
        'minimal': 5000,
        'standard': 20000,
        'comprehensive': 50000,
        'exhaustive': 100000,
        'theoretical_max': (
            dims['flow_rates'] *
            dims['pressures'] *
            dims['temperatures'] *
            dims['hours'] *
            dims['months'] *
            2  # 双洞
        ),
    }
    return estimates
