"""
Predictive Maintenance Module for CYRP
穿黄工程预测性维护模块

功能：
- 设备健康评估
- 剩余寿命预测 (RUL)
- 维护计划优化
- 备件需求预测
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum, auto
from collections import deque
import logging
import json

logger = logging.getLogger(__name__)


# ============================================================================
# 设备模型
# ============================================================================

class EquipmentType(Enum):
    """设备类型"""
    GATE_VALVE = "gate_valve"           # 闸阀
    BUTTERFLY_VALVE = "butterfly_valve" # 蝶阀
    PUMP = "pump"                       # 水泵
    MOTOR = "motor"                     # 电机
    SENSOR_PRESSURE = "sensor_pressure" # 压力传感器
    SENSOR_FLOW = "sensor_flow"         # 流量传感器
    SENSOR_TEMPERATURE = "sensor_temp"  # 温度传感器
    DAS = "das"                         # 分布式声学传感
    DTS = "dts"                         # 分布式温度传感
    PLC = "plc"                         # 可编程控制器
    LINING = "lining"                   # 隧洞衬砌


class HealthStatus(Enum):
    """健康状态"""
    EXCELLENT = 5   # 优秀 (健康度 > 90%)
    GOOD = 4        # 良好 (健康度 70-90%)
    FAIR = 3        # 一般 (健康度 50-70%)
    POOR = 2        # 较差 (健康度 30-50%)
    CRITICAL = 1    # 危急 (健康度 < 30%)


class MaintenanceType(Enum):
    """维护类型"""
    INSPECTION = "inspection"       # 巡检
    PREVENTIVE = "preventive"       # 预防性维护
    CORRECTIVE = "corrective"       # 纠正性维护
    PREDICTIVE = "predictive"       # 预测性维护
    OVERHAUL = "overhaul"          # 大修
    REPLACEMENT = "replacement"     # 更换


@dataclass
class EquipmentSpec:
    """设备规格"""
    equipment_id: str
    name: str
    equipment_type: EquipmentType
    location: str
    # 设计参数
    design_life_years: float = 20.0
    rated_capacity: float = 100.0
    # 维护参数
    mtbf: float = 8760.0            # 平均故障间隔时间(小时)
    mttr: float = 4.0               # 平均修复时间(小时)
    inspection_interval_days: int = 30
    preventive_interval_days: int = 180
    # 成本
    replacement_cost: float = 100000.0
    maintenance_cost: float = 5000.0
    downtime_cost_per_hour: float = 10000.0
    # 元数据
    installation_date: Optional[datetime] = None
    last_maintenance: Optional[datetime] = None
    manufacturer: str = ""
    model: str = ""


@dataclass
class HealthIndicator:
    """健康指标"""
    name: str
    value: float
    unit: str
    min_threshold: float
    max_threshold: float
    weight: float = 1.0
    trend: float = 0.0  # 趋势 (正=恶化, 负=改善)

    @property
    def normalized_value(self) -> float:
        """归一化值 (0=最差, 1=最好)"""
        if self.max_threshold == self.min_threshold:
            return 1.0 if self.value <= self.min_threshold else 0.0

        # 线性归一化
        norm = 1.0 - (self.value - self.min_threshold) / (self.max_threshold - self.min_threshold)
        return max(0.0, min(1.0, norm))


@dataclass
class EquipmentHealth:
    """设备健康状态"""
    equipment_id: str
    timestamp: datetime
    health_score: float             # 综合健康度 (0-1)
    status: HealthStatus
    indicators: List[HealthIndicator]
    remaining_useful_life_days: Optional[int] = None
    failure_probability_30d: float = 0.0
    recommended_action: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'equipment_id': self.equipment_id,
            'timestamp': self.timestamp.isoformat(),
            'health_score': self.health_score,
            'status': self.status.name,
            'rul_days': self.remaining_useful_life_days,
            'failure_probability_30d': self.failure_probability_30d,
            'recommended_action': self.recommended_action,
            'indicators': [
                {'name': i.name, 'value': i.value, 'normalized': i.normalized_value}
                for i in self.indicators
            ]
        }


# ============================================================================
# 健康评估
# ============================================================================

class HealthAssessor:
    """健康评估器"""

    def __init__(self, spec: EquipmentSpec):
        self.spec = spec
        self.indicator_configs: Dict[str, Dict] = {}
        self._history: deque = deque(maxlen=1000)

    def configure_indicator(self, name: str, min_threshold: float,
                           max_threshold: float, weight: float = 1.0):
        """配置健康指标"""
        self.indicator_configs[name] = {
            'min_threshold': min_threshold,
            'max_threshold': max_threshold,
            'weight': weight
        }

    def assess(self, measurements: Dict[str, float]) -> EquipmentHealth:
        """评估健康状态"""
        now = datetime.now()
        indicators = []
        total_weight = 0.0
        weighted_score = 0.0

        for name, config in self.indicator_configs.items():
            if name not in measurements:
                continue

            value = measurements[name]

            # 计算趋势
            trend = self._calculate_trend(name, value)

            indicator = HealthIndicator(
                name=name,
                value=value,
                unit="",
                min_threshold=config['min_threshold'],
                max_threshold=config['max_threshold'],
                weight=config['weight'],
                trend=trend
            )
            indicators.append(indicator)

            weighted_score += indicator.normalized_value * config['weight']
            total_weight += config['weight']

        # 综合健康度
        health_score = weighted_score / total_weight if total_weight > 0 else 1.0

        # 确定状态
        status = self._score_to_status(health_score)

        # 记录历史
        self._history.append((now, health_score, measurements.copy()))

        return EquipmentHealth(
            equipment_id=self.spec.equipment_id,
            timestamp=now,
            health_score=health_score,
            status=status,
            indicators=indicators
        )

    def _calculate_trend(self, indicator_name: str, current_value: float) -> float:
        """计算趋势"""
        if len(self._history) < 10:
            return 0.0

        # 获取历史值
        historical = []
        for ts, score, measurements in list(self._history)[-100:]:
            if indicator_name in measurements:
                historical.append(measurements[indicator_name])

        if len(historical) < 2:
            return 0.0

        # 简单线性趋势
        x = np.arange(len(historical))
        coeffs = np.polyfit(x, historical, 1)
        return coeffs[0]  # 斜率

    def _score_to_status(self, score: float) -> HealthStatus:
        """健康度转状态"""
        if score >= 0.9:
            return HealthStatus.EXCELLENT
        elif score >= 0.7:
            return HealthStatus.GOOD
        elif score >= 0.5:
            return HealthStatus.FAIR
        elif score >= 0.3:
            return HealthStatus.POOR
        else:
            return HealthStatus.CRITICAL


# ============================================================================
# 剩余寿命预测
# ============================================================================

class RULPredictor:
    """剩余使用寿命预测器"""

    def __init__(self, spec: EquipmentSpec):
        self.spec = spec
        self._health_history: List[Tuple[datetime, float]] = []
        self._degradation_model: Optional[Callable] = None

    def update_health(self, timestamp: datetime, health_score: float):
        """更新健康度历史"""
        self._health_history.append((timestamp, health_score))

        # 保留最近的数据
        if len(self._health_history) > 10000:
            self._health_history = self._health_history[-5000:]

    def predict_rul(self, failure_threshold: float = 0.3) -> Optional[int]:
        """
        预测剩余使用寿命

        Args:
            failure_threshold: 失效阈值 (健康度低于此值视为失效)

        Returns:
            预测的剩余天数, 或None如果无法预测
        """
        if len(self._health_history) < 30:
            return None

        # 提取时间序列
        timestamps = [h[0] for h in self._health_history]
        health_values = [h[1] for h in self._health_history]

        # 计算时间（天）
        t0 = timestamps[0]
        days = [(t - t0).total_seconds() / 86400 for t in timestamps]

        # 线性退化模型
        coeffs = np.polyfit(days, health_values, 1)
        slope = coeffs[0]
        intercept = coeffs[1]

        if slope >= 0:
            # 健康度没有下降或在改善
            return None

        # 预测到达失效阈值的时间
        current_day = days[-1]
        current_health = health_values[-1]

        # h(t) = slope * t + intercept = failure_threshold
        # t = (failure_threshold - intercept) / slope
        failure_day = (failure_threshold - intercept) / slope
        rul_days = failure_day - current_day

        return max(0, int(rul_days))

    def predict_failure_probability(self, days_ahead: int = 30) -> float:
        """
        预测指定时间内的故障概率

        Args:
            days_ahead: 预测天数

        Returns:
            故障概率 (0-1)
        """
        rul = self.predict_rul()

        if rul is None:
            # 无法预测，使用基于MTBF的概率
            mtbf_days = self.spec.mtbf / 24
            return 1 - np.exp(-days_ahead / mtbf_days)

        if rul <= 0:
            return 1.0

        if rul >= days_ahead * 3:
            return 0.01

        # 指数分布
        lambda_param = 1.0 / rul
        return 1 - np.exp(-lambda_param * days_ahead)

    def fit_degradation_model(self, model_type: str = 'linear'):
        """拟合退化模型"""
        if len(self._health_history) < 30:
            return

        timestamps = [h[0] for h in self._health_history]
        health_values = [h[1] for h in self._health_history]
        t0 = timestamps[0]
        days = np.array([(t - t0).total_seconds() / 86400 for t in timestamps])
        health = np.array(health_values)

        if model_type == 'linear':
            coeffs = np.polyfit(days, health, 1)
            self._degradation_model = lambda t: coeffs[0] * t + coeffs[1]

        elif model_type == 'exponential':
            # h(t) = a * exp(-b*t)
            # log(h) = log(a) - b*t
            log_health = np.log(np.maximum(health, 0.01))
            coeffs = np.polyfit(days, log_health, 1)
            a = np.exp(coeffs[1])
            b = -coeffs[0]
            self._degradation_model = lambda t: a * np.exp(-b * t)


# ============================================================================
# 维护优化
# ============================================================================

@dataclass
class MaintenanceTask:
    """维护任务"""
    task_id: str
    equipment_id: str
    maintenance_type: MaintenanceType
    description: str
    priority: int               # 1=最高
    estimated_duration_hours: float
    estimated_cost: float
    due_date: Optional[datetime] = None
    scheduled_date: Optional[datetime] = None
    completed_date: Optional[datetime] = None
    status: str = "pending"     # pending, scheduled, in_progress, completed
    spare_parts: List[str] = field(default_factory=list)


@dataclass
class MaintenanceRecommendation:
    """维护建议"""
    equipment_id: str
    equipment_name: str
    current_health: float
    predicted_rul_days: Optional[int]
    failure_probability: float
    recommended_type: MaintenanceType
    recommended_date: datetime
    urgency: str                # immediate, urgent, planned, routine
    estimated_cost: float
    risk_if_delayed: str
    actions: List[str]


class MaintenanceOptimizer:
    """维护优化器"""

    def __init__(self):
        self.equipment_specs: Dict[str, EquipmentSpec] = {}
        self.health_assessors: Dict[str, HealthAssessor] = {}
        self.rul_predictors: Dict[str, RULPredictor] = {}
        self.maintenance_history: List[MaintenanceTask] = []

    def register_equipment(self, spec: EquipmentSpec):
        """注册设备"""
        self.equipment_specs[spec.equipment_id] = spec
        self.health_assessors[spec.equipment_id] = HealthAssessor(spec)
        self.rul_predictors[spec.equipment_id] = RULPredictor(spec)

    def configure_health_indicators(self, equipment_id: str,
                                    indicators: Dict[str, Dict]):
        """配置健康指标"""
        if equipment_id in self.health_assessors:
            assessor = self.health_assessors[equipment_id]
            for name, config in indicators.items():
                assessor.configure_indicator(
                    name,
                    config.get('min', 0),
                    config.get('max', 100),
                    config.get('weight', 1.0)
                )

    def update_measurements(self, equipment_id: str, measurements: Dict[str, float]):
        """更新测量值"""
        if equipment_id not in self.health_assessors:
            return None

        assessor = self.health_assessors[equipment_id]
        health = assessor.assess(measurements)

        # 更新RUL预测器
        predictor = self.rul_predictors[equipment_id]
        predictor.update_health(health.timestamp, health.health_score)

        # 补充预测信息
        health.remaining_useful_life_days = predictor.predict_rul()
        health.failure_probability_30d = predictor.predict_failure_probability(30)

        # 生成建议
        health.recommended_action = self._generate_action_recommendation(
            equipment_id, health
        )

        return health

    def _generate_action_recommendation(self, equipment_id: str,
                                        health: EquipmentHealth) -> str:
        """生成行动建议"""
        spec = self.equipment_specs[equipment_id]

        if health.health_score < 0.3:
            return "立即进行维修或更换，设备处于危急状态"
        elif health.health_score < 0.5:
            return "建议尽快安排预防性维护，防止故障发生"
        elif health.health_score < 0.7:
            return "计划下次维护周期进行检查"
        elif health.failure_probability_30d > 0.2:
            return "虽然当前健康度良好，但故障风险较高，建议加强监测"
        else:
            return "设备状态良好，按计划维护即可"

    def generate_recommendations(self) -> List[MaintenanceRecommendation]:
        """生成维护建议"""
        recommendations = []
        now = datetime.now()

        for equip_id, spec in self.equipment_specs.items():
            assessor = self.health_assessors[equip_id]
            predictor = self.rul_predictors[equip_id]

            # 获取最新健康状态
            if not assessor._history:
                continue

            last_ts, last_health, _ = assessor._history[-1]

            rul_days = predictor.predict_rul()
            failure_prob = predictor.predict_failure_probability(30)

            # 确定维护类型和紧急程度
            if last_health < 0.3 or (rul_days and rul_days < 7):
                maint_type = MaintenanceType.CORRECTIVE
                urgency = "immediate"
                rec_date = now
            elif last_health < 0.5 or (rul_days and rul_days < 30):
                maint_type = MaintenanceType.PREDICTIVE
                urgency = "urgent"
                rec_date = now + timedelta(days=7)
            elif failure_prob > 0.3:
                maint_type = MaintenanceType.PREVENTIVE
                urgency = "planned"
                rec_date = now + timedelta(days=14)
            elif spec.last_maintenance:
                days_since = (now - spec.last_maintenance).days
                if days_since >= spec.preventive_interval_days:
                    maint_type = MaintenanceType.PREVENTIVE
                    urgency = "routine"
                    rec_date = now + timedelta(days=30)
                else:
                    continue
            else:
                continue

            # 风险评估
            if urgency == "immediate":
                risk = "高风险：可能导致非计划停机和设备损坏"
            elif urgency == "urgent":
                risk = "中风险：延迟维护可能导致故障"
            else:
                risk = "低风险：按计划维护可保持设备可靠性"

            recommendation = MaintenanceRecommendation(
                equipment_id=equip_id,
                equipment_name=spec.name,
                current_health=last_health,
                predicted_rul_days=rul_days,
                failure_probability=failure_prob,
                recommended_type=maint_type,
                recommended_date=rec_date,
                urgency=urgency,
                estimated_cost=spec.maintenance_cost,
                risk_if_delayed=risk,
                actions=self._get_maintenance_actions(spec, maint_type)
            )
            recommendations.append(recommendation)

        # 按紧急程度排序
        urgency_order = {'immediate': 0, 'urgent': 1, 'planned': 2, 'routine': 3}
        recommendations.sort(key=lambda x: urgency_order.get(x.urgency, 4))

        return recommendations

    def _get_maintenance_actions(self, spec: EquipmentSpec,
                                 maint_type: MaintenanceType) -> List[str]:
        """获取维护动作"""
        actions = []

        if spec.equipment_type == EquipmentType.GATE_VALVE:
            if maint_type == MaintenanceType.INSPECTION:
                actions = ["目视检查阀门外观", "检查密封状态", "测试开关动作"]
            elif maint_type == MaintenanceType.PREVENTIVE:
                actions = ["润滑阀杆", "更换密封件", "校准行程开关", "清洗阀体"]
            elif maint_type == MaintenanceType.CORRECTIVE:
                actions = ["拆解检查", "更换磨损部件", "重新组装测试"]

        elif spec.equipment_type == EquipmentType.PUMP:
            if maint_type == MaintenanceType.INSPECTION:
                actions = ["检查振动和噪声", "测量轴承温度", "检查密封泄漏"]
            elif maint_type == MaintenanceType.PREVENTIVE:
                actions = ["更换润滑油", "检查联轴器对中", "清洗过滤器"]
            elif maint_type == MaintenanceType.CORRECTIVE:
                actions = ["更换轴承", "更换机械密封", "动平衡校正"]

        elif spec.equipment_type in [EquipmentType.SENSOR_PRESSURE,
                                    EquipmentType.SENSOR_FLOW]:
            if maint_type == MaintenanceType.INSPECTION:
                actions = ["校准检查", "检查接线", "清洁传感器"]
            elif maint_type == MaintenanceType.PREVENTIVE:
                actions = ["重新校准", "更换电缆", "检查安装"]
            elif maint_type == MaintenanceType.CORRECTIVE:
                actions = ["更换传感器", "更新配置"]

        return actions or ["执行标准维护程序"]

    def optimize_schedule(self, planning_horizon_days: int = 90,
                         max_concurrent_tasks: int = 2) -> List[MaintenanceTask]:
        """优化维护计划"""
        recommendations = self.generate_recommendations()
        scheduled_tasks = []
        now = datetime.now()

        # 按日期和紧急程度排序
        recommendations.sort(key=lambda x: (x.recommended_date, x.urgency))

        # 简单调度：依次安排任务
        current_date = now
        concurrent_count = 0

        for rec in recommendations:
            if rec.recommended_date > now + timedelta(days=planning_horizon_days):
                continue

            # 确定调度日期
            schedule_date = max(rec.recommended_date, current_date)

            task = MaintenanceTask(
                task_id=f"MT_{len(scheduled_tasks)+1:04d}",
                equipment_id=rec.equipment_id,
                maintenance_type=rec.recommended_type,
                description=f"{rec.equipment_name} - {rec.recommended_type.value}",
                priority=1 if rec.urgency == 'immediate' else
                        2 if rec.urgency == 'urgent' else
                        3 if rec.urgency == 'planned' else 4,
                estimated_duration_hours=4.0,
                estimated_cost=rec.estimated_cost,
                due_date=rec.recommended_date,
                scheduled_date=schedule_date,
                status="scheduled"
            )
            scheduled_tasks.append(task)

            concurrent_count += 1
            if concurrent_count >= max_concurrent_tasks:
                current_date = schedule_date + timedelta(days=1)
                concurrent_count = 0

        return scheduled_tasks


# ============================================================================
# 备件需求预测
# ============================================================================

@dataclass
class SparePart:
    """备件"""
    part_id: str
    name: str
    equipment_types: List[EquipmentType]
    unit_cost: float
    lead_time_days: int
    min_stock: int
    current_stock: int
    reorder_point: int
    consumption_rate: float  # 每月消耗量


class SparePartPredictor:
    """备件需求预测器"""

    def __init__(self):
        self.parts: Dict[str, SparePart] = {}
        self.consumption_history: Dict[str, List[Tuple[datetime, int]]] = {}

    def register_part(self, part: SparePart):
        """注册备件"""
        self.parts[part.part_id] = part
        self.consumption_history[part.part_id] = []

    def record_consumption(self, part_id: str, quantity: int):
        """记录消耗"""
        if part_id in self.consumption_history:
            self.consumption_history[part_id].append((datetime.now(), quantity))

    def predict_demand(self, part_id: str, months_ahead: int = 3) -> Dict[str, Any]:
        """预测需求"""
        if part_id not in self.parts:
            return {}

        part = self.parts[part_id]
        history = self.consumption_history.get(part_id, [])

        # 计算历史平均消耗
        if history:
            recent = [h for h in history
                     if h[0] > datetime.now() - timedelta(days=180)]
            avg_monthly = sum(h[1] for h in recent) / max(1, len(recent)) * 30 / 180
        else:
            avg_monthly = part.consumption_rate

        predicted_demand = avg_monthly * months_ahead
        safety_stock = part.min_stock
        reorder_needed = part.current_stock - predicted_demand < part.reorder_point

        return {
            'part_id': part_id,
            'part_name': part.name,
            'current_stock': part.current_stock,
            'predicted_demand': predicted_demand,
            'avg_monthly_consumption': avg_monthly,
            'safety_stock': safety_stock,
            'reorder_needed': reorder_needed,
            'suggested_order_qty': max(0, predicted_demand + safety_stock - part.current_stock)
        }


# ============================================================================
# 便捷函数
# ============================================================================

def create_cyrp_maintenance_system() -> MaintenanceOptimizer:
    """创建穿黄工程维护管理系统"""
    optimizer = MaintenanceOptimizer()

    # 注册设备
    equipment_list = [
        EquipmentSpec("GV-001", "进口闸阀", EquipmentType.GATE_VALVE, "进口段",
                     design_life_years=30, mtbf=50000,
                     installation_date=datetime(2014, 1, 1)),
        EquipmentSpec("GV-002", "出口闸阀", EquipmentType.GATE_VALVE, "出口段",
                     design_life_years=30, mtbf=50000,
                     installation_date=datetime(2014, 1, 1)),
        EquipmentSpec("BV-001", "紧急蝶阀", EquipmentType.BUTTERFLY_VALVE, "中段",
                     design_life_years=25, mtbf=40000,
                     installation_date=datetime(2014, 6, 1)),
        EquipmentSpec("PT-001", "进口压力传感器", EquipmentType.SENSOR_PRESSURE, "进口段",
                     design_life_years=10, mtbf=20000,
                     installation_date=datetime(2018, 1, 1)),
        EquipmentSpec("PT-002", "出口压力传感器", EquipmentType.SENSOR_PRESSURE, "出口段",
                     design_life_years=10, mtbf=20000,
                     installation_date=datetime(2018, 1, 1)),
        EquipmentSpec("FT-001", "进口流量计", EquipmentType.SENSOR_FLOW, "进口段",
                     design_life_years=15, mtbf=30000,
                     installation_date=datetime(2016, 1, 1)),
        EquipmentSpec("DAS-001", "分布式声学传感系统", EquipmentType.DAS, "全段",
                     design_life_years=15, mtbf=25000,
                     installation_date=datetime(2019, 1, 1)),
    ]

    for spec in equipment_list:
        optimizer.register_equipment(spec)

    # 配置健康指标
    valve_indicators = {
        'response_time': {'min': 0, 'max': 30, 'weight': 2.0},
        'leak_rate': {'min': 0, 'max': 0.1, 'weight': 1.5},
        'motor_current': {'min': 0, 'max': 150, 'weight': 1.0},
        'vibration': {'min': 0, 'max': 10, 'weight': 1.0},
    }

    sensor_indicators = {
        'drift': {'min': 0, 'max': 5, 'weight': 2.0},
        'noise': {'min': 0, 'max': 3, 'weight': 1.5},
        'zero_offset': {'min': 0, 'max': 2, 'weight': 1.0},
    }

    for equip_id, spec in optimizer.equipment_specs.items():
        if spec.equipment_type in [EquipmentType.GATE_VALVE, EquipmentType.BUTTERFLY_VALVE]:
            optimizer.configure_health_indicators(equip_id, valve_indicators)
        elif spec.equipment_type in [EquipmentType.SENSOR_PRESSURE, EquipmentType.SENSOR_FLOW]:
            optimizer.configure_health_indicators(equip_id, sensor_indicators)

    return optimizer
