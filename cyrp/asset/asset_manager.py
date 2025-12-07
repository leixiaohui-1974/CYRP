"""
Asset Management Module for CYRP
穿黄工程资产管理模块

实现设备资产全生命周期管理，包括资产登记、状态跟踪、折旧计算、备件管理等
"""

import asyncio
import uuid
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Callable, Set, Tuple,
    TypeVar, Generic
)
from collections import defaultdict
import logging
import math

logger = logging.getLogger(__name__)


# ============================================================
# 枚举定义
# ============================================================

class AssetType(Enum):
    """资产类型"""
    PUMP = "pump"                    # 泵
    VALVE = "valve"                  # 阀门
    SENSOR = "sensor"                # 传感器
    PLC = "plc"                      # PLC控制器
    HMI = "hmi"                      # 人机界面
    MOTOR = "motor"                  # 电机
    TRANSFORMER = "transformer"      # 变压器
    CABLE = "cable"                  # 电缆
    PIPE = "pipe"                    # 管道
    INSTRUMENT = "instrument"        # 仪表
    VEHICLE = "vehicle"              # 车辆
    BUILDING = "building"            # 建筑
    SOFTWARE = "software"            # 软件
    OTHER = "other"                  # 其他


class AssetStatus(Enum):
    """资产状态"""
    PLANNED = "planned"              # 计划采购
    ORDERED = "ordered"              # 已下单
    IN_TRANSIT = "in_transit"        # 运输中
    IN_STOCK = "in_stock"            # 库存
    INSTALLED = "installed"          # 已安装
    IN_SERVICE = "in_service"        # 服役中
    MAINTENANCE = "maintenance"      # 维护中
    FAULT = "fault"                  # 故障
    DECOMMISSIONED = "decommissioned"  # 退役
    DISPOSED = "disposed"            # 已处置


class DepreciationMethod(Enum):
    """折旧方法"""
    STRAIGHT_LINE = "straight_line"       # 直线法
    DECLINING_BALANCE = "declining_balance"  # 余额递减法
    DOUBLE_DECLINING = "double_declining"    # 双倍余额递减法
    SUM_OF_YEARS = "sum_of_years"            # 年数总和法
    UNITS_OF_PRODUCTION = "units_of_production"  # 工作量法


class MaintenanceType(Enum):
    """维护类型"""
    PREVENTIVE = "preventive"        # 预防性维护
    CORRECTIVE = "corrective"        # 纠正性维护
    PREDICTIVE = "predictive"        # 预测性维护
    CONDITION_BASED = "condition_based"  # 状态维护
    EMERGENCY = "emergency"          # 紧急维护


class SparePartStatus(Enum):
    """备件状态"""
    AVAILABLE = "available"          # 可用
    RESERVED = "reserved"            # 已预留
    USED = "used"                    # 已使用
    EXPIRED = "expired"              # 已过期
    DEFECTIVE = "defective"          # 缺陷


# ============================================================
# 数据类定义
# ============================================================

@dataclass
class Location:
    """位置"""
    location_id: str
    name: str
    building: str = ""
    floor: str = ""
    room: str = ""
    area: str = ""
    coordinates: Optional[Tuple[float, float, float]] = None  # x, y, z
    parent_location_id: Optional[str] = None


@dataclass
class Manufacturer:
    """制造商"""
    manufacturer_id: str
    name: str
    country: str = ""
    contact: str = ""
    email: str = ""
    phone: str = ""
    website: str = ""


@dataclass
class AssetCategory:
    """资产类别"""
    category_id: str
    name: str
    description: str = ""
    parent_category_id: Optional[str] = None
    depreciation_method: DepreciationMethod = DepreciationMethod.STRAIGHT_LINE
    default_useful_life_years: int = 10
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Asset:
    """资产"""
    asset_id: str
    name: str
    asset_type: AssetType
    category_id: str
    status: AssetStatus = AssetStatus.IN_SERVICE

    # 基本信息
    description: str = ""
    serial_number: str = ""
    model: str = ""
    manufacturer_id: Optional[str] = None
    barcode: str = ""
    rfid_tag: str = ""

    # 位置
    location_id: Optional[str] = None
    parent_asset_id: Optional[str] = None  # 父资产（组件关系）

    # 财务信息
    purchase_date: Optional[date] = None
    purchase_price: float = 0.0
    currency: str = "CNY"
    salvage_value: float = 0.0
    useful_life_years: int = 10
    depreciation_method: DepreciationMethod = DepreciationMethod.STRAIGHT_LINE

    # 运维信息
    installation_date: Optional[date] = None
    warranty_expiry: Optional[date] = None
    last_maintenance_date: Optional[date] = None
    next_maintenance_date: Optional[date] = None
    maintenance_cycle_days: int = 365

    # 技术规格
    specifications: Dict[str, Any] = field(default_factory=dict)
    documents: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)

    # 元数据
    tags: List[str] = field(default_factory=list)
    custom_attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def calculate_depreciation(self, as_of_date: date = None) -> Dict[str, float]:
        """计算折旧"""
        if as_of_date is None:
            as_of_date = date.today()

        if self.purchase_date is None:
            return {
                'accumulated_depreciation': 0.0,
                'book_value': self.purchase_price,
                'annual_depreciation': 0.0
            }

        # 计算已使用年数
        days_used = (as_of_date - self.purchase_date).days
        years_used = days_used / 365.0

        depreciable_amount = self.purchase_price - self.salvage_value

        if self.depreciation_method == DepreciationMethod.STRAIGHT_LINE:
            annual = depreciable_amount / self.useful_life_years
            accumulated = min(annual * years_used, depreciable_amount)

        elif self.depreciation_method == DepreciationMethod.DECLINING_BALANCE:
            rate = 1 / self.useful_life_years
            accumulated = self.purchase_price * (1 - (1 - rate) ** years_used)
            accumulated = min(accumulated, depreciable_amount)
            annual = (self.purchase_price - accumulated) * rate

        elif self.depreciation_method == DepreciationMethod.DOUBLE_DECLINING:
            rate = 2 / self.useful_life_years
            accumulated = self.purchase_price * (1 - (1 - rate) ** years_used)
            accumulated = min(accumulated, depreciable_amount)
            annual = (self.purchase_price - accumulated) * rate

        elif self.depreciation_method == DepreciationMethod.SUM_OF_YEARS:
            sum_years = self.useful_life_years * (self.useful_life_years + 1) / 2
            year = int(years_used) + 1
            if year <= self.useful_life_years:
                factor = (self.useful_life_years - year + 1) / sum_years
                annual = depreciable_amount * factor
            else:
                annual = 0
            accumulated = 0
            for y in range(1, min(int(years_used) + 1, self.useful_life_years + 1)):
                factor = (self.useful_life_years - y + 1) / sum_years
                accumulated += depreciable_amount * factor
            accumulated = min(accumulated, depreciable_amount)

        else:
            annual = depreciable_amount / self.useful_life_years
            accumulated = min(annual * years_used, depreciable_amount)

        book_value = self.purchase_price - accumulated

        return {
            'accumulated_depreciation': round(accumulated, 2),
            'book_value': round(book_value, 2),
            'annual_depreciation': round(annual, 2),
            'years_used': round(years_used, 2),
            'remaining_years': max(0, self.useful_life_years - years_used)
        }


@dataclass
class AssetTransaction:
    """资产变动"""
    transaction_id: str
    asset_id: str
    transaction_type: str  # purchase, transfer, disposal, write_off, revaluation
    from_location_id: Optional[str] = None
    to_location_id: Optional[str] = None
    from_status: Optional[AssetStatus] = None
    to_status: Optional[AssetStatus] = None
    amount: float = 0.0
    description: str = ""
    performed_by: str = ""
    performed_at: datetime = field(default_factory=datetime.now)
    documents: List[str] = field(default_factory=list)


@dataclass
class SparePart:
    """备件"""
    part_id: str
    name: str
    part_number: str
    description: str = ""
    category: str = ""
    manufacturer_id: Optional[str] = None
    unit_price: float = 0.0
    quantity: int = 0
    min_quantity: int = 0
    max_quantity: int = 0
    reorder_point: int = 0
    location: str = ""
    status: SparePartStatus = SparePartStatus.AVAILABLE
    compatible_assets: List[str] = field(default_factory=list)
    expiry_date: Optional[date] = None
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def needs_reorder(self) -> bool:
        """是否需要补货"""
        return self.quantity <= self.reorder_point


@dataclass
class MaintenanceRecord:
    """维护记录"""
    record_id: str
    asset_id: str
    maintenance_type: MaintenanceType
    description: str
    performed_by: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    cost: float = 0.0
    parts_used: List[Dict[str, Any]] = field(default_factory=list)
    labor_hours: float = 0.0
    findings: str = ""
    recommendations: str = ""
    next_maintenance_date: Optional[date] = None
    documents: List[str] = field(default_factory=list)


@dataclass
class AssetHealthScore:
    """资产健康评分"""
    asset_id: str
    score: float  # 0-100
    calculated_at: datetime
    components: Dict[str, float] = field(default_factory=dict)
    factors: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


# ============================================================
# 资产仓库
# ============================================================

class AssetRepository:
    """资产仓库"""

    def __init__(self):
        self.assets: Dict[str, Asset] = {}
        self.categories: Dict[str, AssetCategory] = {}
        self.locations: Dict[str, Location] = {}
        self.manufacturers: Dict[str, Manufacturer] = {}
        self.transactions: Dict[str, AssetTransaction] = {}
        self._lock = asyncio.Lock()

    async def add_asset(self, asset: Asset) -> str:
        """添加资产"""
        async with self._lock:
            self.assets[asset.asset_id] = asset
            logger.info(f"Asset added: {asset.asset_id} - {asset.name}")
            return asset.asset_id

    async def update_asset(self, asset: Asset) -> bool:
        """更新资产"""
        async with self._lock:
            if asset.asset_id not in self.assets:
                return False
            asset.updated_at = datetime.now()
            self.assets[asset.asset_id] = asset
            return True

    async def get_asset(self, asset_id: str) -> Optional[Asset]:
        """获取资产"""
        return self.assets.get(asset_id)

    async def delete_asset(self, asset_id: str) -> bool:
        """删除资产"""
        async with self._lock:
            if asset_id in self.assets:
                del self.assets[asset_id]
                return True
            return False

    async def search_assets(
        self,
        asset_type: Optional[AssetType] = None,
        status: Optional[AssetStatus] = None,
        location_id: Optional[str] = None,
        category_id: Optional[str] = None,
        keyword: Optional[str] = None
    ) -> List[Asset]:
        """搜索资产"""
        results = []

        for asset in self.assets.values():
            if asset_type and asset.asset_type != asset_type:
                continue
            if status and asset.status != status:
                continue
            if location_id and asset.location_id != location_id:
                continue
            if category_id and asset.category_id != category_id:
                continue
            if keyword:
                keyword_lower = keyword.lower()
                if keyword_lower not in asset.name.lower() and \
                   keyword_lower not in asset.description.lower() and \
                   keyword_lower not in asset.serial_number.lower():
                    continue
            results.append(asset)

        return results

    async def add_transaction(self, transaction: AssetTransaction) -> str:
        """添加变动记录"""
        async with self._lock:
            self.transactions[transaction.transaction_id] = transaction
            return transaction.transaction_id

    async def get_asset_transactions(self, asset_id: str) -> List[AssetTransaction]:
        """获取资产变动历史"""
        return [t for t in self.transactions.values() if t.asset_id == asset_id]

    async def add_category(self, category: AssetCategory) -> str:
        """添加类别"""
        async with self._lock:
            self.categories[category.category_id] = category
            return category.category_id

    async def add_location(self, location: Location) -> str:
        """添加位置"""
        async with self._lock:
            self.locations[location.location_id] = location
            return location.location_id

    async def add_manufacturer(self, manufacturer: Manufacturer) -> str:
        """添加制造商"""
        async with self._lock:
            self.manufacturers[manufacturer.manufacturer_id] = manufacturer
            return manufacturer.manufacturer_id


# ============================================================
# 备件管理器
# ============================================================

class SparePartManager:
    """备件管理器"""

    def __init__(self):
        self.parts: Dict[str, SparePart] = {}
        self.usage_history: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()

    async def add_part(self, part: SparePart) -> str:
        """添加备件"""
        async with self._lock:
            self.parts[part.part_id] = part
            return part.part_id

    async def get_part(self, part_id: str) -> Optional[SparePart]:
        """获取备件"""
        return self.parts.get(part_id)

    async def update_quantity(
        self,
        part_id: str,
        delta: int,
        reason: str = "",
        reference_id: str = ""
    ) -> bool:
        """更新库存数量"""
        async with self._lock:
            if part_id not in self.parts:
                return False

            part = self.parts[part_id]
            new_quantity = part.quantity + delta

            if new_quantity < 0:
                return False

            part.quantity = new_quantity

            # 记录使用历史
            self.usage_history.append({
                'part_id': part_id,
                'delta': delta,
                'new_quantity': new_quantity,
                'reason': reason,
                'reference_id': reference_id,
                'timestamp': datetime.now()
            })

            return True

    async def reserve_part(
        self,
        part_id: str,
        quantity: int,
        reference_id: str
    ) -> bool:
        """预留备件"""
        async with self._lock:
            part = self.parts.get(part_id)
            if not part or part.quantity < quantity:
                return False

            part.quantity -= quantity
            self.usage_history.append({
                'part_id': part_id,
                'action': 'reserve',
                'quantity': quantity,
                'reference_id': reference_id,
                'timestamp': datetime.now()
            })

            return True

    async def get_low_stock_parts(self) -> List[SparePart]:
        """获取低库存备件"""
        return [p for p in self.parts.values() if p.needs_reorder]

    async def get_compatible_parts(self, asset_id: str) -> List[SparePart]:
        """获取兼容备件"""
        return [
            p for p in self.parts.values()
            if asset_id in p.compatible_assets
        ]

    async def get_expiring_parts(self, days: int = 30) -> List[SparePart]:
        """获取即将过期的备件"""
        threshold = date.today() + timedelta(days=days)
        return [
            p for p in self.parts.values()
            if p.expiry_date and p.expiry_date <= threshold
        ]


# ============================================================
# 维护管理器
# ============================================================

class MaintenanceManager:
    """维护管理器"""

    def __init__(self, repository: AssetRepository):
        self.repository = repository
        self.records: Dict[str, MaintenanceRecord] = {}
        self._lock = asyncio.Lock()

    async def create_record(self, record: MaintenanceRecord) -> str:
        """创建维护记录"""
        async with self._lock:
            self.records[record.record_id] = record

            # 更新资产维护日期
            asset = await self.repository.get_asset(record.asset_id)
            if asset:
                asset.last_maintenance_date = record.started_at.date()
                if record.next_maintenance_date:
                    asset.next_maintenance_date = record.next_maintenance_date
                elif asset.maintenance_cycle_days:
                    asset.next_maintenance_date = \
                        record.started_at.date() + timedelta(days=asset.maintenance_cycle_days)
                await self.repository.update_asset(asset)

            return record.record_id

    async def complete_record(
        self,
        record_id: str,
        completed_at: datetime,
        findings: str = "",
        recommendations: str = ""
    ) -> bool:
        """完成维护记录"""
        async with self._lock:
            if record_id not in self.records:
                return False

            record = self.records[record_id]
            record.completed_at = completed_at
            record.findings = findings
            record.recommendations = recommendations

            return True

    async def get_asset_records(self, asset_id: str) -> List[MaintenanceRecord]:
        """获取资产维护历史"""
        return [r for r in self.records.values() if r.asset_id == asset_id]

    async def get_overdue_maintenance(self) -> List[Asset]:
        """获取过期未维护的资产"""
        today = date.today()
        overdue = []

        for asset in self.repository.assets.values():
            if asset.next_maintenance_date and asset.next_maintenance_date < today:
                if asset.status == AssetStatus.IN_SERVICE:
                    overdue.append(asset)

        return overdue

    async def get_upcoming_maintenance(self, days: int = 30) -> List[Asset]:
        """获取即将维护的资产"""
        today = date.today()
        threshold = today + timedelta(days=days)
        upcoming = []

        for asset in self.repository.assets.values():
            if asset.next_maintenance_date:
                if today <= asset.next_maintenance_date <= threshold:
                    if asset.status == AssetStatus.IN_SERVICE:
                        upcoming.append(asset)

        return upcoming


# ============================================================
# 健康评估器
# ============================================================

class AssetHealthAssessor:
    """资产健康评估器"""

    def __init__(
        self,
        repository: AssetRepository,
        maintenance_manager: MaintenanceManager
    ):
        self.repository = repository
        self.maintenance_manager = maintenance_manager

    async def assess(self, asset_id: str) -> AssetHealthScore:
        """评估资产健康状况"""
        asset = await self.repository.get_asset(asset_id)
        if not asset:
            raise ValueError(f"Asset not found: {asset_id}")

        components = {}
        factors = {}
        recommendations = []

        # 1. 使用年限评分 (0-25分)
        if asset.purchase_date:
            years_used = (date.today() - asset.purchase_date).days / 365.0
            life_ratio = years_used / asset.useful_life_years
            age_score = max(0, 25 * (1 - life_ratio))

            if life_ratio > 0.8:
                recommendations.append("资产即将达到使用年限，建议评估更换计划")
            elif life_ratio > 1.0:
                recommendations.append("资产已超出设计使用年限，建议加强监测或更换")

        else:
            age_score = 25
        components['age'] = age_score
        factors['years_used'] = years_used if asset.purchase_date else 0

        # 2. 维护状态评分 (0-25分)
        maintenance_score = 25
        if asset.next_maintenance_date:
            days_overdue = (date.today() - asset.next_maintenance_date).days
            if days_overdue > 0:
                maintenance_score = max(0, 25 - days_overdue / 10)
                recommendations.append(f"维护已过期{days_overdue}天，请尽快安排维护")
            elif days_overdue > -30:
                maintenance_score = 20
                recommendations.append("维护即将到期，请做好准备")
        components['maintenance'] = maintenance_score

        # 3. 故障历史评分 (0-25分)
        records = await self.maintenance_manager.get_asset_records(asset_id)
        emergency_count = sum(
            1 for r in records
            if r.maintenance_type == MaintenanceType.EMERGENCY
        )
        corrective_count = sum(
            1 for r in records
            if r.maintenance_type == MaintenanceType.CORRECTIVE
        )

        failure_score = max(0, 25 - emergency_count * 5 - corrective_count * 2)
        if emergency_count > 2:
            recommendations.append("紧急维护次数较多，建议进行全面检查")
        components['failure_history'] = failure_score
        factors['emergency_maintenance_count'] = emergency_count
        factors['corrective_maintenance_count'] = corrective_count

        # 4. 状态评分 (0-25分)
        status_scores = {
            AssetStatus.IN_SERVICE: 25,
            AssetStatus.INSTALLED: 25,
            AssetStatus.MAINTENANCE: 15,
            AssetStatus.FAULT: 5,
            AssetStatus.DECOMMISSIONED: 0,
        }
        status_score = status_scores.get(asset.status, 10)
        components['status'] = status_score

        if asset.status == AssetStatus.FAULT:
            recommendations.append("资产处于故障状态，请及时修复")

        # 计算总分
        total_score = sum(components.values())

        return AssetHealthScore(
            asset_id=asset_id,
            score=round(total_score, 1),
            calculated_at=datetime.now(),
            components=components,
            factors=factors,
            recommendations=recommendations
        )


# ============================================================
# 报表生成器
# ============================================================

class AssetReportGenerator:
    """资产报表生成器"""

    def __init__(self, repository: AssetRepository):
        self.repository = repository

    async def generate_inventory_report(self) -> Dict[str, Any]:
        """生成库存报表"""
        total_count = len(self.repository.assets)
        total_value = sum(a.purchase_price for a in self.repository.assets.values())

        # 按类型统计
        by_type = defaultdict(lambda: {'count': 0, 'value': 0.0})
        for asset in self.repository.assets.values():
            by_type[asset.asset_type.value]['count'] += 1
            by_type[asset.asset_type.value]['value'] += asset.purchase_price

        # 按状态统计
        by_status = defaultdict(int)
        for asset in self.repository.assets.values():
            by_status[asset.status.value] += 1

        # 按位置统计
        by_location = defaultdict(int)
        for asset in self.repository.assets.values():
            loc = asset.location_id or 'unassigned'
            by_location[loc] += 1

        return {
            'report_type': 'inventory',
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_count': total_count,
                'total_value': round(total_value, 2)
            },
            'by_type': dict(by_type),
            'by_status': dict(by_status),
            'by_location': dict(by_location)
        }

    async def generate_depreciation_report(
        self,
        as_of_date: date = None
    ) -> Dict[str, Any]:
        """生成折旧报表"""
        if as_of_date is None:
            as_of_date = date.today()

        items = []
        total_original = 0.0
        total_accumulated = 0.0
        total_book = 0.0

        for asset in self.repository.assets.values():
            depreciation = asset.calculate_depreciation(as_of_date)

            items.append({
                'asset_id': asset.asset_id,
                'name': asset.name,
                'purchase_date': asset.purchase_date.isoformat() if asset.purchase_date else None,
                'original_cost': asset.purchase_price,
                'accumulated_depreciation': depreciation['accumulated_depreciation'],
                'book_value': depreciation['book_value'],
                'annual_depreciation': depreciation['annual_depreciation'],
                'method': asset.depreciation_method.value
            })

            total_original += asset.purchase_price
            total_accumulated += depreciation['accumulated_depreciation']
            total_book += depreciation['book_value']

        return {
            'report_type': 'depreciation',
            'as_of_date': as_of_date.isoformat(),
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_original_cost': round(total_original, 2),
                'total_accumulated_depreciation': round(total_accumulated, 2),
                'total_book_value': round(total_book, 2)
            },
            'items': items
        }

    async def generate_maintenance_report(
        self,
        maintenance_manager: MaintenanceManager,
        start_date: date = None,
        end_date: date = None
    ) -> Dict[str, Any]:
        """生成维护报表"""
        if start_date is None:
            start_date = date.today() - timedelta(days=30)
        if end_date is None:
            end_date = date.today()

        # 过滤时间范围内的记录
        records = [
            r for r in maintenance_manager.records.values()
            if start_date <= r.started_at.date() <= end_date
        ]

        # 按类型统计
        by_type = defaultdict(lambda: {'count': 0, 'cost': 0.0, 'hours': 0.0})
        for record in records:
            by_type[record.maintenance_type.value]['count'] += 1
            by_type[record.maintenance_type.value]['cost'] += record.cost
            by_type[record.maintenance_type.value]['hours'] += record.labor_hours

        # 统计摘要
        total_cost = sum(r.cost for r in records)
        total_hours = sum(r.labor_hours for r in records)

        return {
            'report_type': 'maintenance',
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_records': len(records),
                'total_cost': round(total_cost, 2),
                'total_labor_hours': round(total_hours, 1)
            },
            'by_type': dict(by_type)
        }


# ============================================================
# 资产管理服务
# ============================================================

class AssetManagementService:
    """资产管理服务"""

    def __init__(self):
        self.repository = AssetRepository()
        self.spare_part_manager = SparePartManager()
        self.maintenance_manager = MaintenanceManager(self.repository)
        self.health_assessor = AssetHealthAssessor(
            self.repository, self.maintenance_manager
        )
        self.report_generator = AssetReportGenerator(self.repository)

    async def register_asset(
        self,
        name: str,
        asset_type: AssetType,
        category_id: str,
        **kwargs
    ) -> str:
        """登记资产"""
        asset = Asset(
            asset_id=str(uuid.uuid4()),
            name=name,
            asset_type=asset_type,
            category_id=category_id,
            **kwargs
        )

        await self.repository.add_asset(asset)

        # 记录交易
        transaction = AssetTransaction(
            transaction_id=str(uuid.uuid4()),
            asset_id=asset.asset_id,
            transaction_type='purchase',
            to_status=asset.status,
            amount=asset.purchase_price,
            description=f"资产登记: {name}"
        )
        await self.repository.add_transaction(transaction)

        return asset.asset_id

    async def transfer_asset(
        self,
        asset_id: str,
        to_location_id: str,
        performed_by: str,
        description: str = ""
    ) -> bool:
        """转移资产"""
        asset = await self.repository.get_asset(asset_id)
        if not asset:
            return False

        from_location = asset.location_id
        asset.location_id = to_location_id
        await self.repository.update_asset(asset)

        # 记录交易
        transaction = AssetTransaction(
            transaction_id=str(uuid.uuid4()),
            asset_id=asset_id,
            transaction_type='transfer',
            from_location_id=from_location,
            to_location_id=to_location_id,
            performed_by=performed_by,
            description=description or f"资产转移至: {to_location_id}"
        )
        await self.repository.add_transaction(transaction)

        return True

    async def change_status(
        self,
        asset_id: str,
        new_status: AssetStatus,
        performed_by: str,
        description: str = ""
    ) -> bool:
        """更改资产状态"""
        asset = await self.repository.get_asset(asset_id)
        if not asset:
            return False

        from_status = asset.status
        asset.status = new_status
        await self.repository.update_asset(asset)

        # 记录交易
        transaction = AssetTransaction(
            transaction_id=str(uuid.uuid4()),
            asset_id=asset_id,
            transaction_type='status_change',
            from_status=from_status,
            to_status=new_status,
            performed_by=performed_by,
            description=description or f"状态变更: {from_status.value} -> {new_status.value}"
        )
        await self.repository.add_transaction(transaction)

        return True

    async def dispose_asset(
        self,
        asset_id: str,
        disposal_value: float,
        performed_by: str,
        description: str = ""
    ) -> bool:
        """处置资产"""
        asset = await self.repository.get_asset(asset_id)
        if not asset:
            return False

        from_status = asset.status
        asset.status = AssetStatus.DISPOSED
        await self.repository.update_asset(asset)

        # 记录交易
        transaction = AssetTransaction(
            transaction_id=str(uuid.uuid4()),
            asset_id=asset_id,
            transaction_type='disposal',
            from_status=from_status,
            to_status=AssetStatus.DISPOSED,
            amount=disposal_value,
            performed_by=performed_by,
            description=description or f"资产处置，处置价值: {disposal_value}"
        )
        await self.repository.add_transaction(transaction)

        return True

    async def schedule_maintenance(
        self,
        asset_id: str,
        maintenance_type: MaintenanceType,
        description: str,
        performed_by: str,
        scheduled_date: date = None
    ) -> str:
        """安排维护"""
        if scheduled_date is None:
            scheduled_date = date.today()

        record = MaintenanceRecord(
            record_id=str(uuid.uuid4()),
            asset_id=asset_id,
            maintenance_type=maintenance_type,
            description=description,
            performed_by=performed_by,
            started_at=datetime.combine(scheduled_date, datetime.min.time())
        )

        return await self.maintenance_manager.create_record(record)

    async def get_asset_health(self, asset_id: str) -> AssetHealthScore:
        """获取资产健康评分"""
        return await self.health_assessor.assess(asset_id)

    async def get_dashboard_data(self) -> Dict[str, Any]:
        """获取仪表板数据"""
        # 基本统计
        total_assets = len(self.repository.assets)
        total_value = sum(a.purchase_price for a in self.repository.assets.values())

        # 状态分布
        status_dist = defaultdict(int)
        for asset in self.repository.assets.values():
            status_dist[asset.status.value] += 1

        # 类型分布
        type_dist = defaultdict(int)
        for asset in self.repository.assets.values():
            type_dist[asset.asset_type.value] += 1

        # 维护提醒
        overdue = await self.maintenance_manager.get_overdue_maintenance()
        upcoming = await self.maintenance_manager.get_upcoming_maintenance(days=7)

        # 低库存备件
        low_stock = await self.spare_part_manager.get_low_stock_parts()

        return {
            'summary': {
                'total_assets': total_assets,
                'total_value': round(total_value, 2),
                'in_service': status_dist.get(AssetStatus.IN_SERVICE.value, 0),
                'in_maintenance': status_dist.get(AssetStatus.MAINTENANCE.value, 0),
                'fault': status_dist.get(AssetStatus.FAULT.value, 0)
            },
            'status_distribution': dict(status_dist),
            'type_distribution': dict(type_dist),
            'alerts': {
                'overdue_maintenance': len(overdue),
                'upcoming_maintenance': len(upcoming),
                'low_stock_parts': len(low_stock)
            },
            'generated_at': datetime.now().isoformat()
        }


# ============================================================
# 工厂函数
# ============================================================

def create_cyrp_asset_management() -> AssetManagementService:
    """创建CYRP资产管理服务实例

    Returns:
        AssetManagementService: 资产管理服务实例
    """
    service = AssetManagementService()

    # 添加默认类别
    async def setup():
        categories = [
            AssetCategory(
                category_id="pump",
                name="水泵设备",
                depreciation_method=DepreciationMethod.STRAIGHT_LINE,
                default_useful_life_years=15
            ),
            AssetCategory(
                category_id="valve",
                name="阀门设备",
                depreciation_method=DepreciationMethod.STRAIGHT_LINE,
                default_useful_life_years=20
            ),
            AssetCategory(
                category_id="sensor",
                name="传感器",
                depreciation_method=DepreciationMethod.STRAIGHT_LINE,
                default_useful_life_years=5
            ),
            AssetCategory(
                category_id="control",
                name="控制设备",
                depreciation_method=DepreciationMethod.DOUBLE_DECLINING,
                default_useful_life_years=8
            ),
            AssetCategory(
                category_id="instrument",
                name="仪器仪表",
                depreciation_method=DepreciationMethod.STRAIGHT_LINE,
                default_useful_life_years=10
            ),
        ]

        for category in categories:
            await service.repository.add_category(category)

        # 添加默认位置
        locations = [
            Location("loc_intake", "进水口", building="进水闸"),
            Location("loc_tunnel", "穿黄隧道", building="隧道"),
            Location("loc_outlet", "出水口", building="出水闸"),
            Location("loc_pump_station", "泵站", building="泵房"),
            Location("loc_control_room", "控制室", building="综合楼"),
        ]

        for location in locations:
            await service.repository.add_location(location)

    # 在事件循环中运行
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(setup())
    except RuntimeError:
        asyncio.run(setup())

    return service
