"""
Energy Analysis Module for CYRP
穿黄工程能耗分析模块

实现能耗监测、分析、优化建议、成本核算等功能
"""

import asyncio
import uuid
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Callable, Tuple
)
from collections import defaultdict
import logging
import statistics

logger = logging.getLogger(__name__)


# ============================================================
# 枚举定义
# ============================================================

class EnergyType(Enum):
    """能源类型"""
    ELECTRICITY = "electricity"    # 电力
    DIESEL = "diesel"              # 柴油
    WATER = "water"                # 水
    GAS = "gas"                    # 天然气
    SOLAR = "solar"                # 太阳能
    OTHER = "other"                # 其他


class MeterType(Enum):
    """计量表类型"""
    MAIN = "main"                  # 主表
    SUB = "sub"                    # 分表
    CHECK = "check"                # 考核表


class TariffType(Enum):
    """电价类型"""
    PEAK = "peak"                  # 尖峰
    HIGH = "high"                  # 高峰
    NORMAL = "normal"              # 平段
    VALLEY = "valley"              # 低谷


class EquipmentCategory(Enum):
    """设备类别"""
    PUMP = "pump"                  # 水泵
    MOTOR = "motor"                # 电机
    LIGHTING = "lighting"          # 照明
    HVAC = "hvac"                  # 暖通空调
    CONTROL = "control"            # 控制系统
    AUXILIARY = "auxiliary"        # 辅助设备


# ============================================================
# 数据类定义
# ============================================================

@dataclass
class EnergyMeter:
    """能源计量表"""
    meter_id: str
    name: str
    energy_type: EnergyType
    meter_type: MeterType
    location: str
    rated_capacity: float = 0.0
    unit: str = "kWh"
    multiplier: float = 1.0
    parent_meter_id: Optional[str] = None
    equipment_ids: List[str] = field(default_factory=list)
    installed_at: Optional[datetime] = None
    last_reading: float = 0.0
    last_reading_time: Optional[datetime] = None


@dataclass
class MeterReading:
    """计量表读数"""
    reading_id: str
    meter_id: str
    timestamp: datetime
    reading_value: float
    consumption: float = 0.0
    tariff_type: Optional[TariffType] = None
    power_factor: float = 1.0
    max_demand: float = 0.0
    quality: int = 100


@dataclass
class TariffRate:
    """电价费率"""
    rate_id: str
    name: str
    tariff_type: TariffType
    start_time: str  # HH:MM
    end_time: str    # HH:MM
    rate: float      # 元/kWh
    effective_from: date
    effective_to: Optional[date] = None
    weekday_only: bool = False


@dataclass
class EnergyEquipment:
    """用能设备"""
    equipment_id: str
    name: str
    category: EquipmentCategory
    rated_power: float  # kW
    efficiency: float = 0.85
    operating_hours_per_day: float = 24.0
    location: str = ""
    meter_id: Optional[str] = None
    specifications: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnergyConsumption:
    """能耗数据"""
    period_start: datetime
    period_end: datetime
    meter_id: str
    total_consumption: float
    peak_consumption: float = 0.0
    valley_consumption: float = 0.0
    normal_consumption: float = 0.0
    max_demand: float = 0.0
    avg_power_factor: float = 1.0
    cost: float = 0.0


@dataclass
class EnergyBenchmark:
    """能耗基准"""
    benchmark_id: str
    name: str
    category: str
    unit: str
    target_value: float
    warning_threshold: float
    alarm_threshold: float
    description: str = ""


@dataclass
class EnergyReport:
    """能耗报告"""
    report_id: str
    report_type: str
    period_start: date
    period_end: date
    generated_at: datetime
    summary: Dict[str, Any] = field(default_factory=dict)
    details: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


# ============================================================
# 计量表管理器
# ============================================================

class MeterManager:
    """计量表管理器"""

    def __init__(self):
        self.meters: Dict[str, EnergyMeter] = {}
        self.readings: Dict[str, List[MeterReading]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def add_meter(self, meter: EnergyMeter) -> str:
        """添加计量表"""
        async with self._lock:
            self.meters[meter.meter_id] = meter
            return meter.meter_id

    async def get_meter(self, meter_id: str) -> Optional[EnergyMeter]:
        """获取计量表"""
        return self.meters.get(meter_id)

    async def record_reading(self, reading: MeterReading) -> str:
        """记录读数"""
        async with self._lock:
            meter = self.meters.get(reading.meter_id)
            if meter:
                # 计算消耗量
                if meter.last_reading_time:
                    reading.consumption = (reading.reading_value - meter.last_reading) * meter.multiplier

                meter.last_reading = reading.reading_value
                meter.last_reading_time = reading.timestamp

            self.readings[reading.meter_id].append(reading)
            return reading.reading_id

    async def get_readings(
        self,
        meter_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[MeterReading]:
        """获取读数"""
        readings = self.readings.get(meter_id, [])
        return [
            r for r in readings
            if start_time <= r.timestamp <= end_time
        ]

    async def get_consumption(
        self,
        meter_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> float:
        """获取消耗量"""
        readings = await self.get_readings(meter_id, start_time, end_time)
        return sum(r.consumption for r in readings)


# ============================================================
# 费率管理器
# ============================================================

class TariffManager:
    """费率管理器"""

    def __init__(self):
        self.rates: Dict[str, TariffRate] = {}
        self._lock = asyncio.Lock()

    async def add_rate(self, rate: TariffRate) -> str:
        """添加费率"""
        async with self._lock:
            self.rates[rate.rate_id] = rate
            return rate.rate_id

    async def get_rate(self, timestamp: datetime) -> Optional[TariffRate]:
        """获取指定时间的费率"""
        time_str = timestamp.strftime("%H:%M")
        current_date = timestamp.date()

        for rate in self.rates.values():
            # 检查有效期
            if rate.effective_from > current_date:
                continue
            if rate.effective_to and rate.effective_to < current_date:
                continue

            # 检查工作日
            if rate.weekday_only and timestamp.weekday() >= 5:
                continue

            # 检查时间范围
            if rate.start_time <= time_str <= rate.end_time:
                return rate

        return None

    async def calculate_cost(
        self,
        consumption: float,
        timestamp: datetime
    ) -> Tuple[float, TariffType]:
        """计算费用"""
        rate = await self.get_rate(timestamp)
        if rate:
            return consumption * rate.rate, rate.tariff_type
        return consumption * 0.5, TariffType.NORMAL  # 默认费率

    async def get_all_rates(self) -> List[TariffRate]:
        """获取所有费率"""
        return list(self.rates.values())


# ============================================================
# 能耗分析器
# ============================================================

class EnergyAnalyzer:
    """能耗分析器"""

    def __init__(
        self,
        meter_manager: MeterManager,
        tariff_manager: TariffManager
    ):
        self.meter_manager = meter_manager
        self.tariff_manager = tariff_manager

    async def analyze_consumption(
        self,
        meter_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> EnergyConsumption:
        """分析能耗"""
        readings = await self.meter_manager.get_readings(
            meter_id, start_time, end_time
        )

        if not readings:
            return EnergyConsumption(
                period_start=start_time,
                period_end=end_time,
                meter_id=meter_id,
                total_consumption=0.0
            )

        total = sum(r.consumption for r in readings)
        peak = sum(r.consumption for r in readings
                   if r.tariff_type == TariffType.PEAK)
        high = sum(r.consumption for r in readings
                   if r.tariff_type == TariffType.HIGH)
        valley = sum(r.consumption for r in readings
                     if r.tariff_type == TariffType.VALLEY)
        normal = total - peak - high - valley

        max_demand = max(r.max_demand for r in readings) if readings else 0
        power_factors = [r.power_factor for r in readings if r.power_factor > 0]
        avg_pf = statistics.mean(power_factors) if power_factors else 1.0

        # 计算费用
        cost = 0.0
        for reading in readings:
            c, _ = await self.tariff_manager.calculate_cost(
                reading.consumption, reading.timestamp
            )
            cost += c

        return EnergyConsumption(
            period_start=start_time,
            period_end=end_time,
            meter_id=meter_id,
            total_consumption=total,
            peak_consumption=peak + high,
            valley_consumption=valley,
            normal_consumption=normal,
            max_demand=max_demand,
            avg_power_factor=avg_pf,
            cost=cost
        )

    async def compare_periods(
        self,
        meter_id: str,
        current_start: datetime,
        current_end: datetime,
        previous_start: datetime,
        previous_end: datetime
    ) -> Dict[str, Any]:
        """对比分析"""
        current = await self.analyze_consumption(
            meter_id, current_start, current_end
        )
        previous = await self.analyze_consumption(
            meter_id, previous_start, previous_end
        )

        def calc_change(curr: float, prev: float) -> float:
            if prev == 0:
                return 0.0
            return (curr - prev) / prev * 100

        return {
            'current_period': {
                'start': current_start.isoformat(),
                'end': current_end.isoformat(),
                'consumption': current.total_consumption,
                'cost': current.cost
            },
            'previous_period': {
                'start': previous_start.isoformat(),
                'end': previous_end.isoformat(),
                'consumption': previous.total_consumption,
                'cost': previous.cost
            },
            'changes': {
                'consumption_change': calc_change(
                    current.total_consumption,
                    previous.total_consumption
                ),
                'cost_change': calc_change(current.cost, previous.cost),
                'peak_change': calc_change(
                    current.peak_consumption,
                    previous.peak_consumption
                )
            }
        }

    async def analyze_trends(
        self,
        meter_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """趋势分析"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        daily_data = []
        current = start_time.replace(hour=0, minute=0, second=0, microsecond=0)

        while current < end_time:
            next_day = current + timedelta(days=1)
            consumption = await self.meter_manager.get_consumption(
                meter_id, current, next_day
            )
            daily_data.append({
                'date': current.date().isoformat(),
                'consumption': consumption
            })
            current = next_day

        # 计算统计量
        values = [d['consumption'] for d in daily_data]
        if not values:
            return {'error': 'No data available'}

        mean = statistics.mean(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0
        trend = self._calculate_trend(values)

        return {
            'period_days': days,
            'daily_data': daily_data,
            'statistics': {
                'mean': mean,
                'std_dev': std_dev,
                'min': min(values),
                'max': max(values),
                'total': sum(values)
            },
            'trend': trend
        }

    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """计算趋势"""
        n = len(values)
        if n < 2:
            return {'direction': 'stable', 'slope': 0}

        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(values) / n

        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, values))
        denominator = sum((xi - x_mean) ** 2 for xi in x)

        slope = numerator / denominator if denominator != 0 else 0

        if abs(slope) < 0.01 * y_mean:
            direction = 'stable'
        elif slope > 0:
            direction = 'increasing'
        else:
            direction = 'decreasing'

        return {
            'direction': direction,
            'slope': slope,
            'daily_change': slope
        }


# ============================================================
# 效率分析器
# ============================================================

class EfficiencyAnalyzer:
    """效率分析器"""

    def __init__(self):
        self.equipment: Dict[str, EnergyEquipment] = {}
        self.benchmarks: Dict[str, EnergyBenchmark] = {}

    async def add_equipment(self, equipment: EnergyEquipment) -> str:
        """添加设备"""
        self.equipment[equipment.equipment_id] = equipment
        return equipment.equipment_id

    async def add_benchmark(self, benchmark: EnergyBenchmark) -> str:
        """添加基准"""
        self.benchmarks[benchmark.benchmark_id] = benchmark
        return benchmark.benchmark_id

    async def calculate_efficiency(
        self,
        equipment_id: str,
        actual_consumption: float,
        output_value: float
    ) -> Dict[str, Any]:
        """计算效率"""
        equipment = self.equipment.get(equipment_id)
        if not equipment:
            return {'error': 'Equipment not found'}

        # 理论能耗
        theoretical = (
            equipment.rated_power *
            equipment.operating_hours_per_day /
            equipment.efficiency
        )

        # 效率指标
        efficiency_ratio = theoretical / actual_consumption if actual_consumption > 0 else 0
        specific_consumption = actual_consumption / output_value if output_value > 0 else 0

        return {
            'equipment_id': equipment_id,
            'name': equipment.name,
            'rated_power': equipment.rated_power,
            'actual_consumption': actual_consumption,
            'theoretical_consumption': theoretical,
            'efficiency_ratio': efficiency_ratio,
            'specific_consumption': specific_consumption,
            'rating': self._rate_efficiency(efficiency_ratio)
        }

    def _rate_efficiency(self, ratio: float) -> str:
        """评级效率"""
        if ratio >= 0.95:
            return 'excellent'
        elif ratio >= 0.85:
            return 'good'
        elif ratio >= 0.70:
            return 'fair'
        else:
            return 'poor'

    async def benchmark_analysis(
        self,
        category: str,
        actual_value: float
    ) -> Dict[str, Any]:
        """基准分析"""
        applicable = [
            b for b in self.benchmarks.values()
            if b.category == category
        ]

        if not applicable:
            return {'error': 'No benchmark found'}

        results = []
        for benchmark in applicable:
            deviation = (actual_value - benchmark.target_value) / benchmark.target_value * 100

            if actual_value <= benchmark.target_value:
                status = 'normal'
            elif actual_value <= benchmark.warning_threshold:
                status = 'warning'
            else:
                status = 'alarm'

            results.append({
                'benchmark': benchmark.name,
                'target': benchmark.target_value,
                'actual': actual_value,
                'deviation': deviation,
                'status': status
            })

        return {'benchmarks': results}


# ============================================================
# 优化建议生成器
# ============================================================

class OptimizationAdvisor:
    """优化建议生成器"""

    def __init__(self):
        self.rules: List[Dict[str, Any]] = []
        self._init_rules()

    def _init_rules(self):
        """初始化规则"""
        self.rules = [
            {
                'id': 'peak_shift',
                'condition': lambda d: d.get('peak_ratio', 0) > 0.4,
                'advice': '尖峰时段用电比例较高，建议将部分负荷转移至低谷时段',
                'category': 'load_shifting',
                'potential_saving': 0.15
            },
            {
                'id': 'low_power_factor',
                'condition': lambda d: d.get('power_factor', 1) < 0.9,
                'advice': '功率因数偏低，建议增加无功补偿装置',
                'category': 'power_quality',
                'potential_saving': 0.05
            },
            {
                'id': 'high_demand',
                'condition': lambda d: d.get('load_factor', 1) < 0.6,
                'advice': '负荷率偏低，最大需量较高，建议优化设备启动时序',
                'category': 'demand_management',
                'potential_saving': 0.10
            },
            {
                'id': 'efficiency_decline',
                'condition': lambda d: d.get('efficiency_trend') == 'declining',
                'advice': '设备效率呈下降趋势，建议检查设备运行状态或安排维护',
                'category': 'maintenance',
                'potential_saving': 0.08
            },
            {
                'id': 'standby_power',
                'condition': lambda d: d.get('standby_ratio', 0) > 0.1,
                'advice': '待机能耗比例较高，建议优化设备运行策略或使用智能控制',
                'category': 'operational',
                'potential_saving': 0.05
            },
        ]

    async def generate_recommendations(
        self,
        analysis_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """生成优化建议"""
        recommendations = []

        for rule in self.rules:
            try:
                if rule['condition'](analysis_data):
                    recommendations.append({
                        'rule_id': rule['id'],
                        'advice': rule['advice'],
                        'category': rule['category'],
                        'potential_saving': rule['potential_saving'],
                        'priority': self._calculate_priority(rule, analysis_data)
                    })
            except Exception as e:
                logger.warning(f"Rule evaluation failed: {e}")

        # 按优先级排序
        recommendations.sort(key=lambda x: x['priority'], reverse=True)
        return recommendations

    def _calculate_priority(
        self,
        rule: Dict,
        data: Dict
    ) -> int:
        """计算优先级"""
        base_priority = int(rule['potential_saving'] * 100)
        return base_priority


# ============================================================
# 报告生成器
# ============================================================

class ReportGenerator:
    """报告生成器"""

    def __init__(
        self,
        analyzer: EnergyAnalyzer,
        efficiency_analyzer: EfficiencyAnalyzer,
        advisor: OptimizationAdvisor
    ):
        self.analyzer = analyzer
        self.efficiency_analyzer = efficiency_analyzer
        self.advisor = advisor

    async def generate_daily_report(
        self,
        meter_ids: List[str],
        report_date: date
    ) -> EnergyReport:
        """生成日报"""
        start = datetime.combine(report_date, datetime.min.time())
        end = start + timedelta(days=1)

        details = []
        total_consumption = 0.0
        total_cost = 0.0

        for meter_id in meter_ids:
            consumption = await self.analyzer.analyze_consumption(
                meter_id, start, end
            )
            details.append({
                'meter_id': meter_id,
                'consumption': consumption.total_consumption,
                'cost': consumption.cost,
                'peak': consumption.peak_consumption,
                'valley': consumption.valley_consumption,
                'power_factor': consumption.avg_power_factor
            })
            total_consumption += consumption.total_consumption
            total_cost += consumption.cost

        # 生成建议
        analysis_data = {
            'peak_ratio': sum(d['peak'] for d in details) / total_consumption if total_consumption > 0 else 0,
            'power_factor': statistics.mean([d['power_factor'] for d in details]) if details else 1
        }
        recommendations = await self.advisor.generate_recommendations(analysis_data)

        return EnergyReport(
            report_id=str(uuid.uuid4()),
            report_type='daily',
            period_start=report_date,
            period_end=report_date,
            generated_at=datetime.now(),
            summary={
                'total_consumption': total_consumption,
                'total_cost': total_cost,
                'meter_count': len(meter_ids)
            },
            details=details,
            recommendations=[r['advice'] for r in recommendations[:5]]
        )

    async def generate_monthly_report(
        self,
        meter_ids: List[str],
        year: int,
        month: int
    ) -> EnergyReport:
        """生成月报"""
        start = datetime(year, month, 1)
        if month == 12:
            end = datetime(year + 1, 1, 1)
        else:
            end = datetime(year, month + 1, 1)

        details = []
        total_consumption = 0.0
        total_cost = 0.0

        for meter_id in meter_ids:
            consumption = await self.analyzer.analyze_consumption(
                meter_id, start, end
            )

            # 同比分析
            prev_year_start = datetime(year - 1, month, 1)
            if month == 12:
                prev_year_end = datetime(year, 1, 1)
            else:
                prev_year_end = datetime(year - 1, month + 1, 1)

            prev_consumption = await self.analyzer.analyze_consumption(
                meter_id, prev_year_start, prev_year_end
            )

            yoy_change = 0.0
            if prev_consumption.total_consumption > 0:
                yoy_change = (
                    (consumption.total_consumption - prev_consumption.total_consumption) /
                    prev_consumption.total_consumption * 100
                )

            details.append({
                'meter_id': meter_id,
                'consumption': consumption.total_consumption,
                'cost': consumption.cost,
                'max_demand': consumption.max_demand,
                'power_factor': consumption.avg_power_factor,
                'yoy_change': yoy_change
            })
            total_consumption += consumption.total_consumption
            total_cost += consumption.cost

        return EnergyReport(
            report_id=str(uuid.uuid4()),
            report_type='monthly',
            period_start=date(year, month, 1),
            period_end=(end - timedelta(days=1)).date(),
            generated_at=datetime.now(),
            summary={
                'total_consumption': total_consumption,
                'total_cost': total_cost,
                'daily_average': total_consumption / (end - start).days,
                'meter_count': len(meter_ids)
            },
            details=details,
            recommendations=[]
        )


# ============================================================
# 能耗管理服务
# ============================================================

class EnergyManagementService:
    """能耗管理服务"""

    def __init__(self):
        self.meter_manager = MeterManager()
        self.tariff_manager = TariffManager()
        self.analyzer = EnergyAnalyzer(self.meter_manager, self.tariff_manager)
        self.efficiency_analyzer = EfficiencyAnalyzer()
        self.advisor = OptimizationAdvisor()
        self.report_generator = ReportGenerator(
            self.analyzer, self.efficiency_analyzer, self.advisor
        )

    async def initialize(self):
        """初始化服务"""
        await self._add_default_tariffs()
        await self._add_default_benchmarks()
        logger.info("Energy management service initialized")

    async def _add_default_tariffs(self):
        """添加默认费率"""
        tariffs = [
            TariffRate(
                rate_id="peak",
                name="尖峰电价",
                tariff_type=TariffType.PEAK,
                start_time="10:00",
                end_time="12:00",
                rate=1.2,
                effective_from=date(2024, 1, 1)
            ),
            TariffRate(
                rate_id="high",
                name="高峰电价",
                tariff_type=TariffType.HIGH,
                start_time="08:00",
                end_time="22:00",
                rate=0.85,
                effective_from=date(2024, 1, 1)
            ),
            TariffRate(
                rate_id="valley",
                name="低谷电价",
                tariff_type=TariffType.VALLEY,
                start_time="22:00",
                end_time="08:00",
                rate=0.35,
                effective_from=date(2024, 1, 1)
            ),
        ]

        for tariff in tariffs:
            await self.tariff_manager.add_rate(tariff)

    async def _add_default_benchmarks(self):
        """添加默认基准"""
        benchmarks = [
            EnergyBenchmark(
                benchmark_id="pump_efficiency",
                name="水泵单位能耗",
                category="pump",
                unit="kWh/万m³",
                target_value=100,
                warning_threshold=120,
                alarm_threshold=150,
                description="每万立方米输水能耗"
            ),
            EnergyBenchmark(
                benchmark_id="control_power",
                name="控制系统能耗",
                category="control",
                unit="kWh/日",
                target_value=50,
                warning_threshold=70,
                alarm_threshold=100,
                description="控制系统日均能耗"
            ),
        ]

        for benchmark in benchmarks:
            await self.efficiency_analyzer.add_benchmark(benchmark)

    async def get_dashboard(self) -> Dict[str, Any]:
        """获取仪表板数据"""
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        # 汇总所有计量表
        total_today = 0.0
        total_cost = 0.0
        meter_count = len(self.meter_manager.meters)

        for meter_id in self.meter_manager.meters:
            consumption = await self.analyzer.analyze_consumption(
                meter_id, today_start, now
            )
            total_today += consumption.total_consumption
            total_cost += consumption.cost

        # 获取费率信息
        current_rate = await self.tariff_manager.get_rate(now)

        return {
            'current_time': now.isoformat(),
            'current_tariff': current_rate.tariff_type.value if current_rate else 'unknown',
            'current_rate': current_rate.rate if current_rate else 0,
            'today': {
                'consumption': total_today,
                'cost': total_cost,
                'meter_count': meter_count
            },
            'alerts': [],
            'generated_at': datetime.now().isoformat()
        }


# ============================================================
# 工厂函数
# ============================================================

def create_cyrp_energy_service() -> EnergyManagementService:
    """创建CYRP能耗管理服务实例

    Returns:
        EnergyManagementService: 能耗管理服务实例
    """
    return EnergyManagementService()
