"""
Emergency Response Management Module for CYRP
穿黄工程应急响应管理模块

实现应急预案管理、事件响应、资源调度、应急演练等功能
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Callable, Set, Tuple
)
from collections import defaultdict
import logging
import json

logger = logging.getLogger(__name__)


# ============================================================
# 枚举定义
# ============================================================

class EmergencyLevel(Enum):
    """应急级别"""
    LEVEL_1 = 1  # 特别重大
    LEVEL_2 = 2  # 重大
    LEVEL_3 = 3  # 较大
    LEVEL_4 = 4  # 一般


class EmergencyType(Enum):
    """应急类型"""
    FLOOD = "flood"                    # 洪水
    LEAK = "leak"                      # 泄漏
    EQUIPMENT_FAILURE = "equipment_failure"  # 设备故障
    POWER_OUTAGE = "power_outage"      # 停电
    EARTHQUAKE = "earthquake"          # 地震
    FIRE = "fire"                      # 火灾
    SECURITY = "security"              # 安全事件
    ENVIRONMENTAL = "environmental"    # 环境事件
    OTHER = "other"                    # 其他


class IncidentStatus(Enum):
    """事件状态"""
    REPORTED = "reported"              # 已报告
    CONFIRMED = "confirmed"            # 已确认
    RESPONDING = "responding"          # 响应中
    CONTAINED = "contained"            # 已控制
    RESOLVED = "resolved"              # 已解决
    CLOSED = "closed"                  # 已关闭


class ResourceType(Enum):
    """资源类型"""
    PERSONNEL = "personnel"            # 人员
    EQUIPMENT = "equipment"            # 设备
    VEHICLE = "vehicle"                # 车辆
    MATERIAL = "material"              # 物资
    COMMUNICATION = "communication"    # 通信


class ResourceStatus(Enum):
    """资源状态"""
    AVAILABLE = "available"            # 可用
    DEPLOYED = "deployed"              # 已部署
    RETURNING = "returning"            # 返回中
    MAINTENANCE = "maintenance"        # 维护中
    UNAVAILABLE = "unavailable"        # 不可用


class ActionStatus(Enum):
    """行动状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ============================================================
# 数据类定义
# ============================================================

@dataclass
class EmergencyContact:
    """应急联系人"""
    contact_id: str
    name: str
    title: str
    organization: str
    phone: str
    mobile: str = ""
    email: str = ""
    roles: List[str] = field(default_factory=list)
    available: bool = True
    priority: int = 1


@dataclass
class EmergencyResource:
    """应急资源"""
    resource_id: str
    name: str
    resource_type: ResourceType
    description: str = ""
    location: str = ""
    quantity: int = 1
    status: ResourceStatus = ResourceStatus.AVAILABLE
    capabilities: List[str] = field(default_factory=list)
    contact_id: Optional[str] = None
    last_used: Optional[datetime] = None


@dataclass
class ResponseAction:
    """响应行动"""
    action_id: str
    name: str
    description: str
    sequence: int
    responsible_role: str
    estimated_duration_minutes: int = 30
    prerequisites: List[str] = field(default_factory=list)
    resources_required: List[str] = field(default_factory=list)
    checklist: List[str] = field(default_factory=list)
    status: ActionStatus = ActionStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    assigned_to: Optional[str] = None
    notes: str = ""


@dataclass
class EmergencyPlan:
    """应急预案"""
    plan_id: str
    name: str
    emergency_type: EmergencyType
    level: EmergencyLevel
    description: str
    scope: str = ""
    trigger_conditions: List[str] = field(default_factory=list)
    response_actions: List[ResponseAction] = field(default_factory=list)
    resources_required: List[str] = field(default_factory=list)
    contacts: List[str] = field(default_factory=list)
    notification_template: str = ""
    escalation_rules: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"
    effective_date: Optional[datetime] = None
    review_date: Optional[datetime] = None
    approved_by: str = ""
    enabled: bool = True


@dataclass
class Incident:
    """应急事件"""
    incident_id: str
    title: str
    emergency_type: EmergencyType
    level: EmergencyLevel
    status: IncidentStatus
    description: str
    location: str = ""
    reported_by: str = ""
    reported_at: datetime = field(default_factory=datetime.now)
    confirmed_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    plan_id: Optional[str] = None
    commander: Optional[str] = None
    affected_areas: List[str] = field(default_factory=list)
    casualties: Dict[str, int] = field(default_factory=dict)
    damage_assessment: str = ""
    actions: List[ResponseAction] = field(default_factory=list)
    resources_deployed: List[str] = field(default_factory=list)
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    attachments: List[str] = field(default_factory=list)

    def add_timeline_entry(self, action: str, details: str = "", user: str = ""):
        """添加时间线条目"""
        self.timeline.append({
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details,
            'user': user
        })


@dataclass
class DrillPlan:
    """演练计划"""
    drill_id: str
    name: str
    plan_id: str
    scheduled_date: datetime
    duration_hours: float
    objectives: List[str] = field(default_factory=list)
    participants: List[str] = field(default_factory=list)
    scenarios: List[Dict[str, Any]] = field(default_factory=list)
    evaluation_criteria: List[str] = field(default_factory=list)
    status: str = "planned"
    actual_start: Optional[datetime] = None
    actual_end: Optional[datetime] = None
    results: Dict[str, Any] = field(default_factory=dict)
    lessons_learned: List[str] = field(default_factory=list)


# ============================================================
# 应急预案管理器
# ============================================================

class EmergencyPlanManager:
    """应急预案管理器"""

    def __init__(self):
        self.plans: Dict[str, EmergencyPlan] = {}
        self._lock = asyncio.Lock()

    async def add_plan(self, plan: EmergencyPlan) -> str:
        """添加预案"""
        async with self._lock:
            self.plans[plan.plan_id] = plan
            logger.info(f"Emergency plan added: {plan.plan_id} - {plan.name}")
            return plan.plan_id

    async def update_plan(self, plan: EmergencyPlan) -> bool:
        """更新预案"""
        async with self._lock:
            if plan.plan_id not in self.plans:
                return False
            self.plans[plan.plan_id] = plan
            return True

    async def get_plan(self, plan_id: str) -> Optional[EmergencyPlan]:
        """获取预案"""
        return self.plans.get(plan_id)

    async def find_applicable_plans(
        self,
        emergency_type: EmergencyType,
        level: EmergencyLevel
    ) -> List[EmergencyPlan]:
        """查找适用的预案"""
        applicable = []
        for plan in self.plans.values():
            if not plan.enabled:
                continue
            if plan.emergency_type == emergency_type:
                if plan.level.value >= level.value:
                    applicable.append(plan)

        return sorted(applicable, key=lambda p: p.level.value)

    async def get_all_plans(self) -> List[EmergencyPlan]:
        """获取所有预案"""
        return list(self.plans.values())


# ============================================================
# 资源管理器
# ============================================================

class ResourceManager:
    """应急资源管理器"""

    def __init__(self):
        self.resources: Dict[str, EmergencyResource] = {}
        self.deployments: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def add_resource(self, resource: EmergencyResource) -> str:
        """添加资源"""
        async with self._lock:
            self.resources[resource.resource_id] = resource
            return resource.resource_id

    async def get_resource(self, resource_id: str) -> Optional[EmergencyResource]:
        """获取资源"""
        return self.resources.get(resource_id)

    async def get_available_resources(
        self,
        resource_type: Optional[ResourceType] = None,
        capability: Optional[str] = None
    ) -> List[EmergencyResource]:
        """获取可用资源"""
        available = []
        for resource in self.resources.values():
            if resource.status != ResourceStatus.AVAILABLE:
                continue
            if resource_type and resource.resource_type != resource_type:
                continue
            if capability and capability not in resource.capabilities:
                continue
            available.append(resource)
        return available

    async def deploy_resource(
        self,
        resource_id: str,
        incident_id: str,
        location: str,
        deployed_by: str
    ) -> bool:
        """部署资源"""
        async with self._lock:
            resource = self.resources.get(resource_id)
            if not resource or resource.status != ResourceStatus.AVAILABLE:
                return False

            resource.status = ResourceStatus.DEPLOYED
            self.deployments[resource_id] = {
                'incident_id': incident_id,
                'location': location,
                'deployed_by': deployed_by,
                'deployed_at': datetime.now()
            }

            logger.info(f"Resource {resource_id} deployed to incident {incident_id}")
            return True

    async def release_resource(
        self,
        resource_id: str,
        released_by: str
    ) -> bool:
        """释放资源"""
        async with self._lock:
            resource = self.resources.get(resource_id)
            if not resource:
                return False

            resource.status = ResourceStatus.AVAILABLE
            resource.last_used = datetime.now()

            if resource_id in self.deployments:
                del self.deployments[resource_id]

            logger.info(f"Resource {resource_id} released")
            return True

    async def get_deployment_status(self) -> Dict[str, Any]:
        """获取部署状态"""
        status = {
            'total_resources': len(self.resources),
            'available': 0,
            'deployed': 0,
            'by_type': defaultdict(lambda: {'total': 0, 'available': 0, 'deployed': 0})
        }

        for resource in self.resources.values():
            type_key = resource.resource_type.value
            status['by_type'][type_key]['total'] += 1

            if resource.status == ResourceStatus.AVAILABLE:
                status['available'] += 1
                status['by_type'][type_key]['available'] += 1
            elif resource.status == ResourceStatus.DEPLOYED:
                status['deployed'] += 1
                status['by_type'][type_key]['deployed'] += 1

        return status


# ============================================================
# 联系人管理器
# ============================================================

class ContactManager:
    """应急联系人管理器"""

    def __init__(self):
        self.contacts: Dict[str, EmergencyContact] = {}
        self.call_logs: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()

    async def add_contact(self, contact: EmergencyContact) -> str:
        """添加联系人"""
        async with self._lock:
            self.contacts[contact.contact_id] = contact
            return contact.contact_id

    async def get_contact(self, contact_id: str) -> Optional[EmergencyContact]:
        """获取联系人"""
        return self.contacts.get(contact_id)

    async def get_contacts_by_role(self, role: str) -> List[EmergencyContact]:
        """按角色获取联系人"""
        return [
            c for c in self.contacts.values()
            if role in c.roles and c.available
        ]

    async def get_emergency_contacts(
        self,
        level: EmergencyLevel
    ) -> List[EmergencyContact]:
        """获取应急联系人（按级别）"""
        contacts = list(self.contacts.values())
        # 按优先级排序
        contacts.sort(key=lambda c: c.priority)
        return contacts

    async def log_call(
        self,
        contact_id: str,
        incident_id: str,
        status: str,
        notes: str = ""
    ):
        """记录呼叫"""
        self.call_logs.append({
            'contact_id': contact_id,
            'incident_id': incident_id,
            'timestamp': datetime.now().isoformat(),
            'status': status,
            'notes': notes
        })


# ============================================================
# 事件管理器
# ============================================================

class IncidentManager:
    """事件管理器"""

    def __init__(
        self,
        plan_manager: EmergencyPlanManager,
        resource_manager: ResourceManager,
        contact_manager: ContactManager
    ):
        self.plan_manager = plan_manager
        self.resource_manager = resource_manager
        self.contact_manager = contact_manager
        self.incidents: Dict[str, Incident] = {}
        self._lock = asyncio.Lock()

    async def report_incident(
        self,
        title: str,
        emergency_type: EmergencyType,
        level: EmergencyLevel,
        description: str,
        location: str,
        reported_by: str
    ) -> Incident:
        """报告事件"""
        incident = Incident(
            incident_id=str(uuid.uuid4()),
            title=title,
            emergency_type=emergency_type,
            level=level,
            status=IncidentStatus.REPORTED,
            description=description,
            location=location,
            reported_by=reported_by
        )

        incident.add_timeline_entry(
            action="事件报告",
            details=f"报告人: {reported_by}，位置: {location}",
            user=reported_by
        )

        async with self._lock:
            self.incidents[incident.incident_id] = incident

        logger.info(f"Incident reported: {incident.incident_id} - {title}")
        return incident

    async def confirm_incident(
        self,
        incident_id: str,
        commander: str,
        level_adjustment: Optional[EmergencyLevel] = None
    ) -> bool:
        """确认事件"""
        async with self._lock:
            incident = self.incidents.get(incident_id)
            if not incident:
                return False

            incident.status = IncidentStatus.CONFIRMED
            incident.confirmed_at = datetime.now()
            incident.commander = commander

            if level_adjustment:
                incident.level = level_adjustment

            incident.add_timeline_entry(
                action="事件确认",
                details=f"指挥官: {commander}，级别: {incident.level.name}",
                user=commander
            )

        # 查找并激活预案
        plans = await self.plan_manager.find_applicable_plans(
            incident.emergency_type, incident.level
        )
        if plans:
            await self.activate_plan(incident_id, plans[0].plan_id, commander)

        return True

    async def activate_plan(
        self,
        incident_id: str,
        plan_id: str,
        activated_by: str
    ) -> bool:
        """激活预案"""
        incident = self.incidents.get(incident_id)
        plan = await self.plan_manager.get_plan(plan_id)

        if not incident or not plan:
            return False

        incident.plan_id = plan_id
        incident.status = IncidentStatus.RESPONDING

        # 复制响应行动
        for action in plan.response_actions:
            new_action = ResponseAction(
                action_id=str(uuid.uuid4()),
                name=action.name,
                description=action.description,
                sequence=action.sequence,
                responsible_role=action.responsible_role,
                estimated_duration_minutes=action.estimated_duration_minutes,
                prerequisites=action.prerequisites.copy(),
                resources_required=action.resources_required.copy(),
                checklist=action.checklist.copy()
            )
            incident.actions.append(new_action)

        incident.add_timeline_entry(
            action="预案激活",
            details=f"预案: {plan.name}",
            user=activated_by
        )

        logger.info(f"Plan {plan_id} activated for incident {incident_id}")
        return True

    async def start_action(
        self,
        incident_id: str,
        action_id: str,
        assigned_to: str
    ) -> bool:
        """开始行动"""
        incident = self.incidents.get(incident_id)
        if not incident:
            return False

        for action in incident.actions:
            if action.action_id == action_id:
                action.status = ActionStatus.IN_PROGRESS
                action.started_at = datetime.now()
                action.assigned_to = assigned_to

                incident.add_timeline_entry(
                    action="开始行动",
                    details=f"行动: {action.name}，执行人: {assigned_to}",
                    user=assigned_to
                )
                return True

        return False

    async def complete_action(
        self,
        incident_id: str,
        action_id: str,
        notes: str = ""
    ) -> bool:
        """完成行动"""
        incident = self.incidents.get(incident_id)
        if not incident:
            return False

        for action in incident.actions:
            if action.action_id == action_id:
                action.status = ActionStatus.COMPLETED
                action.completed_at = datetime.now()
                action.notes = notes

                incident.add_timeline_entry(
                    action="完成行动",
                    details=f"行动: {action.name}，备注: {notes}",
                    user=action.assigned_to or ""
                )
                return True

        return False

    async def contain_incident(
        self,
        incident_id: str,
        contained_by: str,
        notes: str = ""
    ) -> bool:
        """控制事件"""
        incident = self.incidents.get(incident_id)
        if not incident:
            return False

        incident.status = IncidentStatus.CONTAINED
        incident.add_timeline_entry(
            action="事件控制",
            details=notes,
            user=contained_by
        )

        return True

    async def resolve_incident(
        self,
        incident_id: str,
        resolved_by: str,
        damage_assessment: str = ""
    ) -> bool:
        """解决事件"""
        incident = self.incidents.get(incident_id)
        if not incident:
            return False

        incident.status = IncidentStatus.RESOLVED
        incident.resolved_at = datetime.now()
        incident.damage_assessment = damage_assessment

        incident.add_timeline_entry(
            action="事件解决",
            details=damage_assessment,
            user=resolved_by
        )

        # 释放部署的资源
        for resource_id in incident.resources_deployed:
            await self.resource_manager.release_resource(resource_id, resolved_by)

        return True

    async def close_incident(
        self,
        incident_id: str,
        closed_by: str,
        summary: str = ""
    ) -> bool:
        """关闭事件"""
        incident = self.incidents.get(incident_id)
        if not incident:
            return False

        incident.status = IncidentStatus.CLOSED
        incident.closed_at = datetime.now()

        incident.add_timeline_entry(
            action="事件关闭",
            details=summary,
            user=closed_by
        )

        return True

    async def get_active_incidents(self) -> List[Incident]:
        """获取活动事件"""
        active_statuses = [
            IncidentStatus.REPORTED,
            IncidentStatus.CONFIRMED,
            IncidentStatus.RESPONDING,
            IncidentStatus.CONTAINED
        ]
        return [
            i for i in self.incidents.values()
            if i.status in active_statuses
        ]

    async def get_incident_stats(self) -> Dict[str, Any]:
        """获取事件统计"""
        stats = {
            'total': len(self.incidents),
            'by_status': defaultdict(int),
            'by_type': defaultdict(int),
            'by_level': defaultdict(int),
            'active': 0
        }

        for incident in self.incidents.values():
            stats['by_status'][incident.status.value] += 1
            stats['by_type'][incident.emergency_type.value] += 1
            stats['by_level'][incident.level.name] += 1

            if incident.status not in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]:
                stats['active'] += 1

        return stats


# ============================================================
# 演练管理器
# ============================================================

class DrillManager:
    """演练管理器"""

    def __init__(self, plan_manager: EmergencyPlanManager):
        self.plan_manager = plan_manager
        self.drills: Dict[str, DrillPlan] = {}
        self._lock = asyncio.Lock()

    async def schedule_drill(
        self,
        name: str,
        plan_id: str,
        scheduled_date: datetime,
        duration_hours: float,
        objectives: List[str],
        participants: List[str]
    ) -> str:
        """安排演练"""
        drill = DrillPlan(
            drill_id=str(uuid.uuid4()),
            name=name,
            plan_id=plan_id,
            scheduled_date=scheduled_date,
            duration_hours=duration_hours,
            objectives=objectives,
            participants=participants
        )

        async with self._lock:
            self.drills[drill.drill_id] = drill

        logger.info(f"Drill scheduled: {drill.drill_id} - {name}")
        return drill.drill_id

    async def start_drill(self, drill_id: str) -> bool:
        """开始演练"""
        async with self._lock:
            drill = self.drills.get(drill_id)
            if not drill:
                return False

            drill.status = "in_progress"
            drill.actual_start = datetime.now()
            return True

    async def end_drill(
        self,
        drill_id: str,
        results: Dict[str, Any],
        lessons_learned: List[str]
    ) -> bool:
        """结束演练"""
        async with self._lock:
            drill = self.drills.get(drill_id)
            if not drill:
                return False

            drill.status = "completed"
            drill.actual_end = datetime.now()
            drill.results = results
            drill.lessons_learned = lessons_learned
            return True

    async def get_upcoming_drills(self, days: int = 30) -> List[DrillPlan]:
        """获取即将进行的演练"""
        threshold = datetime.now() + timedelta(days=days)
        return [
            d for d in self.drills.values()
            if d.status == "planned" and d.scheduled_date <= threshold
        ]

    async def generate_drill_report(self, drill_id: str) -> Optional[Dict[str, Any]]:
        """生成演练报告"""
        drill = self.drills.get(drill_id)
        if not drill or drill.status != "completed":
            return None

        plan = await self.plan_manager.get_plan(drill.plan_id)

        return {
            'drill_id': drill.drill_id,
            'name': drill.name,
            'plan': plan.name if plan else 'Unknown',
            'scheduled_date': drill.scheduled_date.isoformat(),
            'actual_start': drill.actual_start.isoformat() if drill.actual_start else None,
            'actual_end': drill.actual_end.isoformat() if drill.actual_end else None,
            'duration_planned': drill.duration_hours,
            'duration_actual': (
                (drill.actual_end - drill.actual_start).total_seconds() / 3600
                if drill.actual_start and drill.actual_end else None
            ),
            'objectives': drill.objectives,
            'participants': drill.participants,
            'results': drill.results,
            'lessons_learned': drill.lessons_learned,
            'recommendations': self._generate_recommendations(drill)
        }

    def _generate_recommendations(self, drill: DrillPlan) -> List[str]:
        """生成改进建议"""
        recommendations = []

        # 根据结果分析生成建议
        if drill.results.get('response_time_exceeded'):
            recommendations.append("优化响应流程，缩短响应时间")
        if drill.results.get('communication_issues'):
            recommendations.append("加强通信设备检查和人员培训")
        if drill.results.get('resource_shortage'):
            recommendations.append("补充应急资源储备")

        recommendations.extend(drill.lessons_learned)
        return recommendations


# ============================================================
# 应急响应服务
# ============================================================

class EmergencyResponseService:
    """应急响应服务"""

    def __init__(self):
        self.plan_manager = EmergencyPlanManager()
        self.resource_manager = ResourceManager()
        self.contact_manager = ContactManager()
        self.incident_manager = IncidentManager(
            self.plan_manager,
            self.resource_manager,
            self.contact_manager
        )
        self.drill_manager = DrillManager(self.plan_manager)

    async def initialize(self):
        """初始化服务"""
        # 添加默认预案
        await self._add_default_plans()
        # 添加默认资源
        await self._add_default_resources()
        # 添加默认联系人
        await self._add_default_contacts()

        logger.info("Emergency response service initialized")

    async def _add_default_plans(self):
        """添加默认预案"""
        plans = [
            EmergencyPlan(
                plan_id="plan_flood",
                name="洪水应急预案",
                emergency_type=EmergencyType.FLOOD,
                level=EmergencyLevel.LEVEL_2,
                description="穿黄工程洪水应急响应预案",
                trigger_conditions=[
                    "黄河水位超过警戒线",
                    "上游来水流量超过设计值",
                    "气象预报重大降雨"
                ],
                response_actions=[
                    ResponseAction(
                        action_id="a1",
                        name="启动应急响应",
                        description="启动应急指挥中心，通知相关人员",
                        sequence=1,
                        responsible_role="应急指挥官",
                        estimated_duration_minutes=15
                    ),
                    ResponseAction(
                        action_id="a2",
                        name="监测水位",
                        description="加密水位监测频率，每10分钟报告一次",
                        sequence=2,
                        responsible_role="监测人员",
                        estimated_duration_minutes=30
                    ),
                    ResponseAction(
                        action_id="a3",
                        name="检查闸门",
                        description="检查所有闸门状态，确保正常运行",
                        sequence=3,
                        responsible_role="运维人员",
                        estimated_duration_minutes=60
                    ),
                    ResponseAction(
                        action_id="a4",
                        name="疏散人员",
                        description="疏散危险区域人员至安全地点",
                        sequence=4,
                        responsible_role="安全人员",
                        estimated_duration_minutes=120
                    ),
                ]
            ),
            EmergencyPlan(
                plan_id="plan_leak",
                name="管道泄漏应急预案",
                emergency_type=EmergencyType.LEAK,
                level=EmergencyLevel.LEVEL_3,
                description="穿黄隧道管道泄漏应急响应预案",
                trigger_conditions=[
                    "监测到异常流量损失",
                    "压力传感器报警",
                    "巡检发现泄漏"
                ],
                response_actions=[
                    ResponseAction(
                        action_id="b1",
                        name="定位泄漏点",
                        description="使用监测系统定位泄漏位置",
                        sequence=1,
                        responsible_role="技术人员",
                        estimated_duration_minutes=30
                    ),
                    ResponseAction(
                        action_id="b2",
                        name="关闭阀门",
                        description="关闭上下游控制阀门，隔离泄漏段",
                        sequence=2,
                        responsible_role="运维人员",
                        estimated_duration_minutes=15
                    ),
                    ResponseAction(
                        action_id="b3",
                        name="抢修作业",
                        description="组织抢修队伍进行修复",
                        sequence=3,
                        responsible_role="抢修队",
                        estimated_duration_minutes=240
                    ),
                ]
            ),
            EmergencyPlan(
                plan_id="plan_power",
                name="停电应急预案",
                emergency_type=EmergencyType.POWER_OUTAGE,
                level=EmergencyLevel.LEVEL_3,
                description="穿黄工程停电应急响应预案",
                trigger_conditions=[
                    "主电源中断",
                    "备用电源故障",
                    "电网通知计划停电"
                ],
                response_actions=[
                    ResponseAction(
                        action_id="c1",
                        name="启动备用电源",
                        description="立即启动UPS和柴油发电机",
                        sequence=1,
                        responsible_role="电气人员",
                        estimated_duration_minutes=5
                    ),
                    ResponseAction(
                        action_id="c2",
                        name="确认关键设备",
                        description="确认关键监控和控制设备正常运行",
                        sequence=2,
                        responsible_role="运维人员",
                        estimated_duration_minutes=15
                    ),
                    ResponseAction(
                        action_id="c3",
                        name="联系供电部门",
                        description="联系电力公司了解恢复时间",
                        sequence=3,
                        responsible_role="调度人员",
                        estimated_duration_minutes=10
                    ),
                ]
            ),
        ]

        for plan in plans:
            await self.plan_manager.add_plan(plan)

    async def _add_default_resources(self):
        """添加默认资源"""
        resources = [
            EmergencyResource(
                resource_id="res_pump_1",
                name="移动泵站1号",
                resource_type=ResourceType.EQUIPMENT,
                description="大流量移动泵站",
                location="设备库",
                capabilities=["排水", "抽水"]
            ),
            EmergencyResource(
                resource_id="res_gen_1",
                name="应急发电车",
                resource_type=ResourceType.EQUIPMENT,
                description="500KW柴油发电车",
                location="停车场",
                capabilities=["供电"]
            ),
            EmergencyResource(
                resource_id="res_repair_team",
                name="抢修队",
                resource_type=ResourceType.PERSONNEL,
                description="专业抢修队伍",
                location="抢修中心",
                quantity=10,
                capabilities=["管道维修", "电气维修", "机械维修"]
            ),
            EmergencyResource(
                resource_id="res_boat_1",
                name="冲锋舟1号",
                resource_type=ResourceType.VEHICLE,
                description="应急救援冲锋舟",
                location="码头",
                capabilities=["水上救援", "物资运输"]
            ),
            EmergencyResource(
                resource_id="res_sandbag",
                name="沙袋储备",
                resource_type=ResourceType.MATERIAL,
                description="防汛沙袋",
                location="物资库",
                quantity=5000,
                capabilities=["防汛"]
            ),
        ]

        for resource in resources:
            await self.resource_manager.add_resource(resource)

    async def _add_default_contacts(self):
        """添加默认联系人"""
        contacts = [
            EmergencyContact(
                contact_id="contact_cmd",
                name="张指挥",
                title="应急总指挥",
                organization="穿黄工程管理处",
                phone="0371-12345678",
                mobile="13800138000",
                roles=["应急指挥官", "决策者"],
                priority=1
            ),
            EmergencyContact(
                contact_id="contact_tech",
                name="李工程师",
                title="技术负责人",
                organization="穿黄工程管理处",
                phone="0371-12345679",
                mobile="13800138001",
                roles=["技术人员", "监测人员"],
                priority=2
            ),
            EmergencyContact(
                contact_id="contact_safety",
                name="王安全员",
                title="安全主管",
                organization="穿黄工程管理处",
                phone="0371-12345680",
                mobile="13800138002",
                roles=["安全人员"],
                priority=2
            ),
            EmergencyContact(
                contact_id="contact_ops",
                name="赵运维",
                title="运维班长",
                organization="穿黄工程管理处",
                phone="0371-12345681",
                mobile="13800138003",
                roles=["运维人员", "电气人员"],
                priority=3
            ),
        ]

        for contact in contacts:
            await self.contact_manager.add_contact(contact)

    async def get_dashboard(self) -> Dict[str, Any]:
        """获取仪表板数据"""
        active_incidents = await self.incident_manager.get_active_incidents()
        incident_stats = await self.incident_manager.get_incident_stats()
        resource_status = await self.resource_manager.get_deployment_status()
        upcoming_drills = await self.drill_manager.get_upcoming_drills(days=30)

        return {
            'active_incidents': len(active_incidents),
            'incident_stats': incident_stats,
            'resource_status': resource_status,
            'upcoming_drills': len(upcoming_drills),
            'alerts': self._get_alerts(active_incidents, resource_status),
            'generated_at': datetime.now().isoformat()
        }

    def _get_alerts(
        self,
        incidents: List[Incident],
        resource_status: Dict
    ) -> List[Dict[str, Any]]:
        """生成告警"""
        alerts = []

        # 高级别事件告警
        for incident in incidents:
            if incident.level in [EmergencyLevel.LEVEL_1, EmergencyLevel.LEVEL_2]:
                alerts.append({
                    'type': 'high_priority_incident',
                    'message': f"高级别事件: {incident.title}",
                    'level': incident.level.name,
                    'incident_id': incident.incident_id
                })

        # 资源不足告警
        if resource_status['available'] < resource_status['total'] * 0.2:
            alerts.append({
                'type': 'low_resources',
                'message': "可用应急资源不足20%",
                'available': resource_status['available'],
                'total': resource_status['total']
            })

        return alerts


# ============================================================
# 工厂函数
# ============================================================

def create_cyrp_emergency_service() -> EmergencyResponseService:
    """创建CYRP应急响应服务实例

    Returns:
        EmergencyResponseService: 应急响应服务实例
    """
    return EmergencyResponseService()
