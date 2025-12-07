"""
Planning and Forecast Integration for Scenario System
计划信息与预测信息集成模块

将调度计划、检修计划、气象预报、水情预报等信息
融入场景生成和场景识别系统
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Any, Callable
from enum import Enum, auto
from datetime import datetime, timedelta
import json
from abc import ABC, abstractmethod


# ============================================================================
# 计划信息定义
# ============================================================================

class PlanType(Enum):
    """计划类型"""
    DISPATCH = "dispatch"           # 调度计划
    MAINTENANCE = "maintenance"     # 检修计划
    EMERGENCY_DRILL = "drill"       # 应急演练
    INSPECTION = "inspection"       # 巡检计划
    COORDINATION = "coordination"   # 协调计划
    STAFFING = "staffing"          # 人员排班


class MaintenanceLevel(Enum):
    """检修等级"""
    ROUTINE = "routine"             # 日常检修
    MINOR = "minor"                 # 小修
    MAJOR = "major"                 # 大修
    OVERHAUL = "overhaul"           # 大修
    EMERGENCY = "emergency"         # 抢修


class CoordinationType(Enum):
    """协调类型"""
    UPSTREAM_RESERVOIR = "upstream"     # 上游水库
    DOWNSTREAM_CANAL = "downstream"     # 下游干渠
    PARALLEL_TUNNEL = "parallel"        # 并行隧洞
    GRID_POWER = "grid"                 # 电网
    EMERGENCY_TEAM = "emergency"        # 应急队伍


@dataclass
class DispatchPlan:
    """调度计划"""
    plan_id: str
    start_time: datetime
    end_time: datetime
    target_flow: float              # 目标流量 m³/s
    flow_profile: List[Tuple[datetime, float]]  # 时间-流量序列
    priority: str = "normal"
    constraints: Dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def get_flow_at(self, time: datetime) -> float:
        """获取指定时刻的目标流量"""
        if not self.flow_profile:
            return self.target_flow

        for i, (t, f) in enumerate(self.flow_profile):
            if time < t:
                if i == 0:
                    return self.flow_profile[0][1]
                # 线性插值
                t0, f0 = self.flow_profile[i-1]
                t1, f1 = self.flow_profile[i]
                ratio = (time - t0).total_seconds() / (t1 - t0).total_seconds()
                return f0 + ratio * (f1 - f0)
        return self.flow_profile[-1][1]


@dataclass
class MaintenancePlan:
    """检修计划"""
    plan_id: str
    equipment_id: str
    equipment_type: str             # valve, pump, sensor, structure
    location: float                 # 位置 (m)
    start_time: datetime
    end_time: datetime
    level: MaintenanceLevel
    requires_shutdown: bool = False
    flow_restriction: Optional[float] = None  # 流量限制
    personnel: int = 1
    risk_level: str = "low"
    description: str = ""


@dataclass
class CoordinationPlan:
    """协调计划"""
    plan_id: str
    coord_type: CoordinationType
    partner: str                    # 协调方
    start_time: datetime
    end_time: datetime
    expected_impact: Dict[str, float]  # 预期影响
    communication_protocol: str = "standard"
    backup_plan: Optional[str] = None


@dataclass
class EmergencyDrillPlan:
    """应急演练计划"""
    plan_id: str
    drill_type: str                 # 演练类型
    scenario_id: str                # 模拟场景
    start_time: datetime
    duration: timedelta
    participants: List[str]
    objectives: List[str]
    safety_measures: List[str]


# ============================================================================
# 预测信息定义
# ============================================================================

class ForecastType(Enum):
    """预测类型"""
    WEATHER = "weather"             # 气象预报
    FLOOD = "flood"                 # 洪水预报
    DEMAND = "demand"               # 需求预测
    EQUIPMENT = "equipment"         # 设备状态预测
    WATER_QUALITY = "water_quality" # 水质预测
    SEISMIC = "seismic"            # 地震预警


class WeatherSeverity(Enum):
    """天气严重程度"""
    NORMAL = 0
    ADVISORY = 1        # 关注
    WATCH = 2           # 注意
    WARNING = 3         # 警告
    EMERGENCY = 4       # 紧急


class FloodLevel(Enum):
    """洪水等级"""
    NORMAL = 0          # 正常
    ATTENTION = 1       # 关注
    WARNING = 2         # 警戒
    DANGER = 3          # 危险
    EXTREME = 4         # 极端


@dataclass
class WeatherForecast:
    """气象预报"""
    forecast_id: str
    issue_time: datetime
    valid_from: datetime
    valid_to: datetime
    temperature_min: float
    temperature_max: float
    precipitation: float            # 降水量 mm
    precipitation_probability: float  # 降水概率
    wind_speed: float               # 风速 m/s
    humidity: float                 # 湿度 %
    severity: WeatherSeverity
    warnings: List[str] = field(default_factory=list)

    def get_scenario_impact(self) -> Dict[str, float]:
        """获取对场景的影响"""
        impact = {}
        # 温度影响
        if self.temperature_min < 0:
            impact['freezing_risk'] = min(1.0, abs(self.temperature_min) / 10)
        if self.temperature_max > 30:
            impact['high_temp_risk'] = min(1.0, (self.temperature_max - 30) / 10)
        # 降水影响
        if self.precipitation > 50:
            impact['flood_risk'] = min(1.0, self.precipitation / 200)
        # 风速影响
        if self.wind_speed > 15:
            impact['wind_risk'] = min(1.0, self.wind_speed / 30)
        return impact


@dataclass
class FloodForecast:
    """洪水预报"""
    forecast_id: str
    issue_time: datetime
    valid_time: datetime
    river_section: str
    water_level: float              # 水位 m
    flow_rate: float                # 流量 m³/s
    level: FloodLevel
    peak_time: Optional[datetime] = None
    peak_level: Optional[float] = None
    confidence: float = 0.8

    def get_scenario_parameters(self) -> Dict[str, Any]:
        """获取场景参数"""
        params = {
            'flow_rate': self.flow_rate,
            'water_level': self.water_level,
            'flood_level': self.level.value,
            'external_event': f"FLOOD_L{self.level.value}" if self.level.value > 0 else None
        }
        if self.peak_time:
            params['peak_eta'] = (self.peak_time - datetime.now()).total_seconds() / 3600
        return params


@dataclass
class DemandForecast:
    """需求预测"""
    forecast_id: str
    issue_time: datetime
    forecast_horizon: timedelta
    hourly_demand: List[Tuple[datetime, float]]  # 小时需求序列
    daily_total: float
    peak_demand: float
    peak_time: datetime
    confidence: float = 0.85
    model_type: str = "ensemble"

    def get_demand_at(self, time: datetime) -> float:
        """获取指定时刻的需求"""
        for t, d in self.hourly_demand:
            if t <= time < t + timedelta(hours=1):
                return d
        return self.daily_total / 24  # 默认平均值


@dataclass
class EquipmentHealthForecast:
    """设备健康预测"""
    equipment_id: str
    equipment_type: str
    current_health: float           # 当前健康度 0-1
    predicted_health: Dict[int, float]  # 未来N天的健康度
    failure_probability: Dict[int, float]  # 未来N天的故障概率
    recommended_maintenance: Optional[datetime] = None
    remaining_useful_life: Optional[int] = None  # 剩余寿命(天)
    degradation_rate: float = 0.0   # 退化速率

    def get_health_at(self, days_ahead: int) -> float:
        """获取未来某天的健康度"""
        if days_ahead in self.predicted_health:
            return self.predicted_health[days_ahead]
        # 线性外推
        return max(0, self.current_health - self.degradation_rate * days_ahead)


@dataclass
class WaterQualityForecast:
    """水质预测"""
    forecast_id: str
    issue_time: datetime
    location: str
    turbidity: float                # 浊度 NTU
    sediment: float                 # 含沙量 kg/m³
    temperature: float              # 水温 °C
    ph: float
    dissolved_oxygen: float         # 溶解氧 mg/L
    quality_grade: str              # I-V类
    trend: str = "stable"           # rising, stable, falling


@dataclass
class SeismicAlert:
    """地震预警"""
    alert_id: str
    issue_time: datetime
    epicenter_lat: float
    epicenter_lon: float
    depth: float                    # km
    magnitude: float
    estimated_intensity: int        # 烈度
    eta_seconds: float              # 到达时间(秒)
    confidence: float = 0.9

    def get_impact_level(self, distance_km: float) -> int:
        """根据距离估算影响烈度"""
        # 简化的烈度衰减模型
        base_intensity = self.estimated_intensity
        attenuation = np.log10(distance_km / 10 + 1) * 1.5
        return max(1, int(base_intensity - attenuation))


# ============================================================================
# 计划管理器
# ============================================================================

class PlanManager:
    """计划管理器"""

    def __init__(self):
        self.dispatch_plans: Dict[str, DispatchPlan] = {}
        self.maintenance_plans: Dict[str, MaintenancePlan] = {}
        self.coordination_plans: Dict[str, CoordinationPlan] = {}
        self.drill_plans: Dict[str, EmergencyDrillPlan] = {}

    def add_dispatch_plan(self, plan: DispatchPlan):
        """添加调度计划"""
        self.dispatch_plans[plan.plan_id] = plan

    def add_maintenance_plan(self, plan: MaintenancePlan):
        """添加检修计划"""
        self.maintenance_plans[plan.plan_id] = plan

    def add_coordination_plan(self, plan: CoordinationPlan):
        """添加协调计划"""
        self.coordination_plans[plan.plan_id] = plan

    def add_drill_plan(self, plan: EmergencyDrillPlan):
        """添加演练计划"""
        self.drill_plans[plan.plan_id] = plan

    def get_active_plans(self, time: datetime) -> Dict[str, List]:
        """获取指定时刻的活动计划"""
        active = {
            'dispatch': [],
            'maintenance': [],
            'coordination': [],
            'drill': []
        }

        for plan in self.dispatch_plans.values():
            if plan.start_time <= time <= plan.end_time:
                active['dispatch'].append(plan)

        for plan in self.maintenance_plans.values():
            if plan.start_time <= time <= plan.end_time:
                active['maintenance'].append(plan)

        for plan in self.coordination_plans.values():
            if plan.start_time <= time <= plan.end_time:
                active['coordination'].append(plan)

        for plan in self.drill_plans.values():
            if plan.start_time <= time <= plan.start_time + plan.duration:
                active['drill'].append(plan)

        return active

    def get_constraints_at(self, time: datetime) -> Dict[str, Any]:
        """获取指定时刻的约束条件"""
        constraints = {
            'flow_restrictions': [],
            'equipment_unavailable': [],
            'personnel_requirements': 0,
            'shutdown_required': False
        }

        active = self.get_active_plans(time)

        # 检修计划的约束
        for plan in active['maintenance']:
            if plan.flow_restriction:
                constraints['flow_restrictions'].append({
                    'equipment': plan.equipment_id,
                    'max_flow': plan.flow_restriction
                })
            if plan.requires_shutdown:
                constraints['shutdown_required'] = True
            constraints['equipment_unavailable'].append(plan.equipment_id)
            constraints['personnel_requirements'] += plan.personnel

        return constraints


# ============================================================================
# 预测管理器
# ============================================================================

class ForecastManager:
    """预测管理器"""

    def __init__(self):
        self.weather_forecasts: Dict[str, WeatherForecast] = {}
        self.flood_forecasts: Dict[str, FloodForecast] = {}
        self.demand_forecasts: Dict[str, DemandForecast] = {}
        self.equipment_forecasts: Dict[str, EquipmentHealthForecast] = {}
        self.quality_forecasts: Dict[str, WaterQualityForecast] = {}
        self.seismic_alerts: Dict[str, SeismicAlert] = {}

    def add_weather_forecast(self, forecast: WeatherForecast):
        self.weather_forecasts[forecast.forecast_id] = forecast

    def add_flood_forecast(self, forecast: FloodForecast):
        self.flood_forecasts[forecast.forecast_id] = forecast

    def add_demand_forecast(self, forecast: DemandForecast):
        self.demand_forecasts[forecast.forecast_id] = forecast

    def add_equipment_forecast(self, forecast: EquipmentHealthForecast):
        self.equipment_forecasts[forecast.equipment_id] = forecast

    def add_quality_forecast(self, forecast: WaterQualityForecast):
        self.quality_forecasts[forecast.forecast_id] = forecast

    def add_seismic_alert(self, alert: SeismicAlert):
        self.seismic_alerts[alert.alert_id] = alert

    def get_valid_forecasts(self, time: datetime) -> Dict[str, Any]:
        """获取指定时刻有效的预测"""
        valid = {
            'weather': None,
            'flood': None,
            'demand': None,
            'equipment': {},
            'quality': None,
            'seismic': None
        }

        # 气象预报
        for f in self.weather_forecasts.values():
            if f.valid_from <= time <= f.valid_to:
                valid['weather'] = f
                break

        # 洪水预报
        for f in self.flood_forecasts.values():
            if abs((f.valid_time - time).total_seconds()) < 3600:
                valid['flood'] = f
                break

        # 需求预测
        for f in self.demand_forecasts.values():
            if f.issue_time <= time <= f.issue_time + f.forecast_horizon:
                valid['demand'] = f
                break

        # 设备预测
        valid['equipment'] = dict(self.equipment_forecasts)

        # 水质预测
        for f in sorted(self.quality_forecasts.values(),
                       key=lambda x: x.issue_time, reverse=True):
            valid['quality'] = f
            break

        # 地震预警
        for a in self.seismic_alerts.values():
            if (time - a.issue_time).total_seconds() < 300:  # 5分钟内
                valid['seismic'] = a
                break

        return valid

    def get_risk_assessment(self, time: datetime) -> Dict[str, float]:
        """获取综合风险评估"""
        risks = {
            'weather_risk': 0.0,
            'flood_risk': 0.0,
            'demand_risk': 0.0,
            'equipment_risk': 0.0,
            'quality_risk': 0.0,
            'seismic_risk': 0.0,
            'overall_risk': 0.0
        }

        forecasts = self.get_valid_forecasts(time)

        if forecasts['weather']:
            impact = forecasts['weather'].get_scenario_impact()
            risks['weather_risk'] = max(impact.values()) if impact else 0.0

        if forecasts['flood']:
            risks['flood_risk'] = forecasts['flood'].level.value / 4.0

        if forecasts['demand']:
            # 需求超过设计值的风险
            peak = forecasts['demand'].peak_demand
            risks['demand_risk'] = max(0, (peak - 265) / 55)

        if forecasts['equipment']:
            # 最差设备的健康度
            min_health = min(
                (f.current_health for f in forecasts['equipment'].values()),
                default=1.0
            )
            risks['equipment_risk'] = 1.0 - min_health

        if forecasts['quality']:
            q = forecasts['quality']
            if q.quality_grade in ['IV', 'V']:
                risks['quality_risk'] = 0.5 if q.quality_grade == 'IV' else 0.8

        if forecasts['seismic']:
            risks['seismic_risk'] = min(1.0, forecasts['seismic'].magnitude / 8.0)

        # 综合风险
        weights = [0.15, 0.25, 0.1, 0.2, 0.1, 0.2]
        risk_values = [risks[k] for k in ['weather_risk', 'flood_risk', 'demand_risk',
                                          'equipment_risk', 'quality_risk', 'seismic_risk']]
        risks['overall_risk'] = sum(w * r for w, r in zip(weights, risk_values))

        return risks


# ============================================================================
# 场景集成器
# ============================================================================

class PlanningForecastScenarioIntegrator:
    """
    计划-预测-场景集成器

    将计划信息和预测信息融入场景生成和识别
    """

    def __init__(self, plan_manager: PlanManager, forecast_manager: ForecastManager):
        self.plans = plan_manager
        self.forecasts = forecast_manager

    def get_scenario_context(self, time: datetime) -> Dict[str, Any]:
        """
        获取场景上下文

        整合计划和预测信息，为场景生成和识别提供上下文
        """
        context = {
            'time': time.isoformat(),
            'active_plans': self.plans.get_active_plans(time),
            'constraints': self.plans.get_constraints_at(time),
            'forecasts': self.forecasts.get_valid_forecasts(time),
            'risks': self.forecasts.get_risk_assessment(time),
        }

        # 推断场景类型
        context['inferred_scenario_type'] = self._infer_scenario_type(context)

        # 推荐的运行参数
        context['recommended_params'] = self._get_recommended_params(context)

        return context

    def _infer_scenario_type(self, context: Dict) -> str:
        """推断场景类型"""
        risks = context['risks']
        plans = context['active_plans']
        constraints = context['constraints']

        # 优先级判断
        if risks['seismic_risk'] > 0.3:
            return 'S6-C'  # 地震响应
        if risks['flood_risk'] > 0.5:
            return 'S4-A'  # 洪水响应
        if constraints['shutdown_required']:
            return 'S1-B'  # 检修停机
        if plans['maintenance']:
            return 'S3-B'  # 检修运行
        if risks['equipment_risk'] > 0.5:
            return 'S5-A'  # 设备故障
        if risks['demand_risk'] > 0.3:
            return 'S4-B'  # 高需求
        if plans['drill']:
            return 'S7'    # 演练

        return 'S2-A'  # 正常运行

    def _get_recommended_params(self, context: Dict) -> Dict[str, Any]:
        """获取推荐的运行参数"""
        params = {
            'flow_rate': 265.0,  # 默认设计流量
            'pressure_setpoint': 500.0,
            'valve_position': 1.0,
            'control_mode': 'auto'
        }

        # 根据调度计划调整
        dispatch_plans = context['active_plans']['dispatch']
        if dispatch_plans:
            plan = dispatch_plans[0]
            params['flow_rate'] = plan.get_flow_at(datetime.now())

        # 根据需求预测调整
        demand = context['forecasts'].get('demand')
        if demand:
            current_demand = demand.get_demand_at(datetime.now())
            params['flow_rate'] = min(params['flow_rate'], current_demand * 1.1)

        # 根据约束调整
        constraints = context['constraints']
        for restriction in constraints['flow_restrictions']:
            params['flow_rate'] = min(params['flow_rate'], restriction['max_flow'])

        # 根据风险调整
        risks = context['risks']
        if risks['overall_risk'] > 0.5:
            params['control_mode'] = 'conservative'
            params['flow_rate'] *= 0.9

        return params

    def generate_scenarios_from_plans(self, start_time: datetime,
                                      horizon_hours: int = 24) -> List[Dict]:
        """
        根据计划生成场景序列

        Args:
            start_time: 开始时间
            horizon_hours: 预测时长(小时)

        Returns:
            场景序列
        """
        scenarios = []
        current = start_time

        while current < start_time + timedelta(hours=horizon_hours):
            context = self.get_scenario_context(current)
            scenario = {
                'time': current.isoformat(),
                'scenario_type': context['inferred_scenario_type'],
                'params': context['recommended_params'],
                'risks': context['risks'],
                'active_plans': len(context['active_plans']['maintenance']) +
                               len(context['active_plans']['dispatch'])
            }
            scenarios.append(scenario)
            current += timedelta(hours=1)

        return scenarios

    def classify_with_context(self, sensor_data: Dict[str, float],
                             time: datetime) -> Dict[str, Any]:
        """
        带上下文的场景分类

        结合传感器数据和计划/预测信息进行分类
        """
        context = self.get_scenario_context(time)

        # 基础分类（从传感器数据）
        base_classification = self._classify_from_sensors(sensor_data)

        # 上下文调整
        final_scenario = self._adjust_classification(
            base_classification,
            context
        )

        return {
            'scenario_id': final_scenario,
            'base_classification': base_classification,
            'context_adjusted': final_scenario != base_classification,
            'context': context,
            'confidence': self._calculate_confidence(sensor_data, context)
        }

    def _classify_from_sensors(self, data: Dict[str, float]) -> str:
        """基于传感器的基础分类"""
        flow = data.get('flow_rate', 265)
        pressure = data.get('pressure', 500)
        leakage = data.get('leakage_rate', 0)

        if leakage > 0.1:
            return 'S5-C'
        if flow > 300:
            return 'S4-A'
        if flow < 50:
            return 'S1-A'
        if pressure > 800:
            return 'S5-A'

        return 'S2-A'

    def _adjust_classification(self, base: str, context: Dict) -> str:
        """根据上下文调整分类"""
        inferred = context['inferred_scenario_type']
        risks = context['risks']

        # 如果推断场景是紧急类型，优先使用
        emergency_scenarios = ['S4-A', 'S5-C', 'S5-D', 'S6-A', 'S6-B', 'S6-C', 'S7']
        if inferred in emergency_scenarios:
            return inferred

        # 如果风险很高，使用推断场景
        if risks['overall_risk'] > 0.5:
            return inferred

        # 否则使用基础分类
        return base

    def _calculate_confidence(self, sensor_data: Dict, context: Dict) -> float:
        """计算分类置信度"""
        base_confidence = 0.8

        # 预测信息增强置信度
        if context['forecasts'].get('weather'):
            base_confidence += 0.05
        if context['forecasts'].get('flood'):
            base_confidence += 0.05
        if context['forecasts'].get('demand'):
            base_confidence += 0.05

        # 计划信息增强
        if context['active_plans']['dispatch']:
            base_confidence += 0.03

        return min(0.99, base_confidence)


# ============================================================================
# 示例数据生成
# ============================================================================

def create_sample_plans() -> PlanManager:
    """创建示例计划"""
    manager = PlanManager()

    now = datetime.now()

    # 调度计划
    manager.add_dispatch_plan(DispatchPlan(
        plan_id="DISP-001",
        start_time=now,
        end_time=now + timedelta(days=1),
        target_flow=265.0,
        flow_profile=[
            (now, 200.0),
            (now + timedelta(hours=6), 265.0),
            (now + timedelta(hours=12), 280.0),
            (now + timedelta(hours=18), 265.0),
            (now + timedelta(hours=24), 200.0),
        ],
        priority="normal"
    ))

    # 检修计划
    manager.add_maintenance_plan(MaintenancePlan(
        plan_id="MAINT-001",
        equipment_id="GV-001",
        equipment_type="valve",
        location=100.0,
        start_time=now + timedelta(days=2),
        end_time=now + timedelta(days=2, hours=8),
        level=MaintenanceLevel.ROUTINE,
        requires_shutdown=False,
        flow_restriction=200.0,
        personnel=3,
        description="入口闸阀例行检修"
    ))

    # 协调计划
    manager.add_coordination_plan(CoordinationPlan(
        plan_id="COORD-001",
        coord_type=CoordinationType.UPSTREAM_RESERVOIR,
        partner="丹江口水库",
        start_time=now,
        end_time=now + timedelta(days=7),
        expected_impact={'flow_change': 10.0}
    ))

    return manager


def create_sample_forecasts() -> ForecastManager:
    """创建示例预测"""
    manager = ForecastManager()

    now = datetime.now()

    # 气象预报
    manager.add_weather_forecast(WeatherForecast(
        forecast_id="WX-001",
        issue_time=now,
        valid_from=now,
        valid_to=now + timedelta(hours=24),
        temperature_min=15.0,
        temperature_max=25.0,
        precipitation=5.0,
        precipitation_probability=0.3,
        wind_speed=8.0,
        humidity=65.0,
        severity=WeatherSeverity.NORMAL
    ))

    # 洪水预报
    manager.add_flood_forecast(FloodForecast(
        forecast_id="FLOOD-001",
        issue_time=now,
        valid_time=now + timedelta(hours=12),
        river_section="黄河穿黄段",
        water_level=95.5,
        flow_rate=280.0,
        level=FloodLevel.ATTENTION,
        confidence=0.85
    ))

    # 需求预测
    hourly = [(now + timedelta(hours=h), 250 + 30 * np.sin(h * np.pi / 12))
              for h in range(24)]
    manager.add_demand_forecast(DemandForecast(
        forecast_id="DEM-001",
        issue_time=now,
        forecast_horizon=timedelta(hours=24),
        hourly_demand=hourly,
        daily_total=6000.0,
        peak_demand=290.0,
        peak_time=now + timedelta(hours=18),
        confidence=0.9
    ))

    # 设备健康预测
    manager.add_equipment_forecast(EquipmentHealthForecast(
        equipment_id="GV-001",
        equipment_type="gate_valve",
        current_health=0.92,
        predicted_health={7: 0.90, 14: 0.88, 30: 0.85},
        failure_probability={7: 0.01, 14: 0.02, 30: 0.05},
        recommended_maintenance=now + timedelta(days=20),
        remaining_useful_life=90,
        degradation_rate=0.001
    ))

    return manager


# ============================================================================
# 便捷函数
# ============================================================================

def create_integrated_scenario_system():
    """创建集成场景系统"""
    plan_manager = create_sample_plans()
    forecast_manager = create_sample_forecasts()
    integrator = PlanningForecastScenarioIntegrator(plan_manager, forecast_manager)
    return integrator
