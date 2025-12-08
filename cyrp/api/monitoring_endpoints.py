"""
系统监控API端点 - System Monitoring API Endpoints

提供实时监控数据、告警管理、系统状态查询等API接口
Provides real-time monitoring data, alarm management, and system status query APIs
"""

import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

from cyrp.api.rest_api import (
    APIRouter,
    APIRequest,
    APIResponse,
    APIError,
    HTTPMethod,
)
from cyrp.monitoring.dashboard_data import (
    DashboardDataProvider,
    MetricsCollector,
    SystemStatus,
    MetricType as DashboardMetricType,
    Metric,
)


@dataclass
class AlertRule:
    """告警规则"""
    rule_id: str
    name: str
    metric_name: str
    condition: str  # 'gt', 'lt', 'eq', 'gte', 'lte'
    threshold: float
    severity: str  # 'info', 'warning', 'critical'
    enabled: bool = True
    cooldown_seconds: int = 300  # 冷却时间
    last_triggered: Optional[float] = None
    message_template: str = ""


@dataclass
class Alert:
    """告警实例"""
    alert_id: str
    rule_id: str
    name: str
    severity: str
    message: str
    metric_value: float
    threshold: float
    timestamp: float
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[float] = None
    resolved: bool = False
    resolved_at: Optional[float] = None


class AlertManager:
    """告警管理器"""

    def __init__(self):
        self._rules: Dict[str, AlertRule] = {}
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []
        self._alert_counter = 0

        # 创建默认规则
        self._create_default_rules()

    def _create_default_rules(self):
        """创建默认告警规则"""
        default_rules = [
            AlertRule(
                rule_id="RULE_FLOW_HIGH",
                name="流量过高告警",
                metric_name="flow_rate_total",
                condition="gt",
                threshold=300.0,
                severity="warning",
                message_template="总流量 {value:.1f} m³/s 超过阈值 {threshold:.1f} m³/s"
            ),
            AlertRule(
                rule_id="RULE_FLOW_CRITICAL",
                name="流量严重超标",
                metric_name="flow_rate_total",
                condition="gt",
                threshold=320.0,
                severity="critical",
                message_template="总流量 {value:.1f} m³/s 严重超标，阈值 {threshold:.1f} m³/s"
            ),
            AlertRule(
                rule_id="RULE_PRESSURE_HIGH",
                name="压力过高告警",
                metric_name="pressure_max",
                condition="gt",
                threshold=1e6,
                severity="warning",
                message_template="最大压力 {value:.0f} Pa 超过阈值"
            ),
            AlertRule(
                rule_id="RULE_HEALTH_LOW",
                name="系统健康度下降",
                metric_name="health_score",
                condition="lt",
                threshold=70.0,
                severity="warning",
                message_template="系统健康评分 {value:.1f}% 低于阈值 {threshold:.1f}%"
            ),
            AlertRule(
                rule_id="RULE_HEALTH_CRITICAL",
                name="系统健康严重告警",
                metric_name="health_score",
                condition="lt",
                threshold=50.0,
                severity="critical",
                message_template="系统健康评分 {value:.1f}% 严重偏低"
            ),
            AlertRule(
                rule_id="RULE_SENSOR_AVAIL",
                name="传感器可用率下降",
                metric_name="sensor_availability",
                condition="lt",
                threshold=90.0,
                severity="warning",
                message_template="传感器可用率 {value:.1f}% 低于阈值"
            ),
        ]

        for rule in default_rules:
            self._rules[rule.rule_id] = rule

    def add_rule(self, rule: AlertRule):
        """添加告警规则"""
        self._rules[rule.rule_id] = rule

    def remove_rule(self, rule_id: str) -> bool:
        """移除告警规则"""
        if rule_id in self._rules:
            del self._rules[rule_id]
            return True
        return False

    def get_rules(self) -> List[AlertRule]:
        """获取所有规则"""
        return list(self._rules.values())

    def evaluate_metrics(self, metrics: Dict[str, Metric]) -> List[Alert]:
        """评估指标并生成告警"""
        new_alerts = []
        current_time = time.time()

        for rule_id, rule in self._rules.items():
            if not rule.enabled:
                continue

            # 检查冷却时间
            if rule.last_triggered:
                if current_time - rule.last_triggered < rule.cooldown_seconds:
                    continue

            # 获取指标值
            metric = metrics.get(rule.metric_name)
            if not metric:
                continue

            value = metric.value
            triggered = False

            # 评估条件
            if rule.condition == "gt" and value > rule.threshold:
                triggered = True
            elif rule.condition == "lt" and value < rule.threshold:
                triggered = True
            elif rule.condition == "gte" and value >= rule.threshold:
                triggered = True
            elif rule.condition == "lte" and value <= rule.threshold:
                triggered = True
            elif rule.condition == "eq" and value == rule.threshold:
                triggered = True

            if triggered:
                self._alert_counter += 1
                alert_id = f"ALT{current_time:.0f}{self._alert_counter:04d}"

                message = rule.message_template.format(
                    value=value,
                    threshold=rule.threshold
                ) if rule.message_template else f"{rule.name}: {value}"

                alert = Alert(
                    alert_id=alert_id,
                    rule_id=rule_id,
                    name=rule.name,
                    severity=rule.severity,
                    message=message,
                    metric_value=value,
                    threshold=rule.threshold,
                    timestamp=current_time
                )

                self._active_alerts[alert_id] = alert
                self._alert_history.append(alert)
                new_alerts.append(alert)
                rule.last_triggered = current_time

        # 保持历史大小
        if len(self._alert_history) > 10000:
            self._alert_history = self._alert_history[-5000:]

        return new_alerts

    def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """确认告警"""
        if alert_id in self._active_alerts:
            alert = self._active_alerts[alert_id]
            alert.acknowledged = True
            alert.acknowledged_by = user_id
            alert.acknowledged_at = time.time()
            return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """解决告警"""
        if alert_id in self._active_alerts:
            alert = self._active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = time.time()
            del self._active_alerts[alert_id]
            return True
        return False

    def get_active_alerts(self, severity: Optional[str] = None) -> List[Alert]:
        """获取活动告警"""
        alerts = list(self._active_alerts.values())
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)

    def get_alert_history(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 100
    ) -> List[Alert]:
        """获取告警历史"""
        alerts = self._alert_history

        if start_time:
            alerts = [a for a in alerts if a.timestamp >= start_time]
        if end_time:
            alerts = [a for a in alerts if a.timestamp <= end_time]

        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)[:limit]

    def get_statistics(self) -> Dict[str, Any]:
        """获取告警统计"""
        active = self.get_active_alerts()
        return {
            "total_active": len(active),
            "critical_count": sum(1 for a in active if a.severity == "critical"),
            "warning_count": sum(1 for a in active if a.severity == "warning"),
            "info_count": sum(1 for a in active if a.severity == "info"),
            "unacknowledged": sum(1 for a in active if not a.acknowledged),
            "total_rules": len(self._rules),
            "enabled_rules": sum(1 for r in self._rules.values() if r.enabled),
            "total_history": len(self._alert_history)
        }


class MonitoringAPIModule:
    """监控API模块"""

    def __init__(
        self,
        dashboard: Optional[DashboardDataProvider] = None,
        alert_manager: Optional[AlertManager] = None
    ):
        self.dashboard = dashboard or DashboardDataProvider()
        self.alert_manager = alert_manager or AlertManager()

        # 创建路由器
        self.router = self._create_router()

    def _create_router(self) -> APIRouter:
        """创建监控API路由器"""
        router = APIRouter(prefix="/api/v1/monitoring")

        # ============ 实时监控 ============
        @router.get(
            "/realtime",
            summary="获取实时监控数据",
            description="返回系统实时状态快照，包括流量、压力、健康度等",
            tags=["监控"],
            auth_required=True
        )
        async def get_realtime_snapshot(request: APIRequest):
            snapshot = self.dashboard.get_realtime_snapshot()
            return {
                "success": True,
                "data": snapshot
            }

        @router.get(
            "/metrics",
            summary="获取当前指标值",
            description="返回所有监控指标的当前值",
            tags=["监控"],
            auth_required=True
        )
        async def get_current_metrics(request: APIRequest):
            metrics = self.dashboard.get_current_metrics()
            return {
                "success": True,
                "timestamp": time.time(),
                "metrics": {
                    name: metric.to_dict()
                    for name, metric in metrics.items()
                }
            }

        @router.get(
            "/metrics/{metric_name}/history",
            summary="获取指标历史",
            description="获取指定指标的历史数据",
            tags=["监控"],
            auth_required=True
        )
        async def get_metric_history(request: APIRequest):
            metric_name = request.path_params.get("metric_name")
            start_time = request.query_params.get("start")
            end_time = request.query_params.get("end")
            limit = int(request.query_params.get("limit", "1000"))

            start_ts = float(start_time) if start_time else None
            end_ts = float(end_time) if end_time else None

            history = self.dashboard.get_metric_history(
                metric_name, start_ts, end_ts, limit
            )

            return {
                "success": True,
                "metric_name": metric_name,
                "count": len(history),
                "data": [m.to_dict() for m in history]
            }

        # ============ Prometheus/Grafana 集成 ============
        @router.get(
            "/prometheus",
            summary="Prometheus格式指标",
            description="返回Prometheus兼容格式的指标数据",
            tags=["监控", "集成"],
            auth_required=False  # Prometheus抓取通常不带认证
        )
        async def get_prometheus_metrics(request: APIRequest):
            metrics_text = self.dashboard.get_prometheus_metrics()
            return APIResponse(
                status_code=200,
                body=metrics_text,
                headers={"Content-Type": "text/plain; charset=utf-8"}
            )

        @router.get(
            "/grafana/dashboard",
            summary="Grafana仪表板配置",
            description="返回Grafana仪表板JSON配置",
            tags=["监控", "集成"],
            auth_required=True
        )
        async def get_grafana_dashboard(request: APIRequest):
            dashboard_config = self.dashboard.get_grafana_dashboard()
            return {
                "success": True,
                "dashboard": dashboard_config
            }

        # ============ 系统状态 ============
        @router.get(
            "/status",
            summary="获取系统状态",
            description="获取完整的系统状态摘要",
            tags=["监控"],
            auth_required=True
        )
        async def get_system_status(request: APIRequest):
            api_response = self.dashboard.get_api_response()
            return {
                "success": True,
                "data": api_response
            }

        @router.post(
            "/metrics/record",
            summary="记录指标数据",
            description="批量记录监控指标",
            tags=["监控"],
            auth_required=True,
            request_schema={
                "metrics": {"type": "object", "required": True},
                "timestamp": {"type": "number", "required": False}
            }
        )
        async def record_metrics(request: APIRequest):
            metrics = request.body.get("metrics", {})
            timestamp = request.body.get("timestamp")

            self.dashboard.record_batch(metrics, timestamp)

            # 评估告警
            current_metrics = self.dashboard.get_current_metrics()
            new_alerts = self.alert_manager.evaluate_metrics(current_metrics)

            return {
                "success": True,
                "recorded": len(metrics),
                "new_alerts": len(new_alerts)
            }

        # ============ 告警管理 ============
        @router.get(
            "/alerts",
            summary="获取告警列表",
            description="获取活动告警列表",
            tags=["告警"],
            auth_required=True
        )
        async def get_alerts(request: APIRequest):
            severity = request.query_params.get("severity")
            alerts = self.alert_manager.get_active_alerts(severity)

            return {
                "success": True,
                "total": len(alerts),
                "alerts": [
                    {
                        "alert_id": a.alert_id,
                        "rule_id": a.rule_id,
                        "name": a.name,
                        "severity": a.severity,
                        "message": a.message,
                        "metric_value": a.metric_value,
                        "threshold": a.threshold,
                        "timestamp": a.timestamp,
                        "acknowledged": a.acknowledged,
                        "acknowledged_by": a.acknowledged_by
                    }
                    for a in alerts
                ]
            }

        @router.get(
            "/alerts/statistics",
            summary="获取告警统计",
            description="获取告警统计信息",
            tags=["告警"],
            auth_required=True
        )
        async def get_alert_statistics(request: APIRequest):
            stats = self.alert_manager.get_statistics()
            return {
                "success": True,
                "statistics": stats
            }

        @router.get(
            "/alerts/history",
            summary="获取告警历史",
            description="获取历史告警记录",
            tags=["告警"],
            auth_required=True
        )
        async def get_alert_history(request: APIRequest):
            start = request.query_params.get("start")
            end = request.query_params.get("end")
            limit = int(request.query_params.get("limit", "100"))

            start_ts = float(start) if start else None
            end_ts = float(end) if end else None

            history = self.alert_manager.get_alert_history(start_ts, end_ts, limit)

            return {
                "success": True,
                "total": len(history),
                "alerts": [
                    {
                        "alert_id": a.alert_id,
                        "name": a.name,
                        "severity": a.severity,
                        "message": a.message,
                        "timestamp": a.timestamp,
                        "resolved": a.resolved,
                        "resolved_at": a.resolved_at
                    }
                    for a in history
                ]
            }

        @router.post(
            "/alerts/{alert_id}/acknowledge",
            summary="确认告警",
            description="确认指定的告警",
            tags=["告警"],
            auth_required=True
        )
        async def acknowledge_alert(request: APIRequest):
            alert_id = request.path_params.get("alert_id")
            user_id = request.user_id or "anonymous"

            success = self.alert_manager.acknowledge_alert(alert_id, user_id)

            if success:
                return {"success": True, "message": f"告警 {alert_id} 已确认"}
            else:
                return APIError(
                    code="ALERT_NOT_FOUND",
                    message=f"告警 {alert_id} 不存在或已解决",
                    status_code=404
                ).to_response()

        @router.post(
            "/alerts/{alert_id}/resolve",
            summary="解决告警",
            description="标记告警为已解决",
            tags=["告警"],
            auth_required=True
        )
        async def resolve_alert(request: APIRequest):
            alert_id = request.path_params.get("alert_id")

            success = self.alert_manager.resolve_alert(alert_id)

            if success:
                return {"success": True, "message": f"告警 {alert_id} 已解决"}
            else:
                return APIError(
                    code="ALERT_NOT_FOUND",
                    message=f"告警 {alert_id} 不存在",
                    status_code=404
                ).to_response()

        # ============ 告警规则管理 ============
        @router.get(
            "/alerts/rules",
            summary="获取告警规则",
            description="获取所有告警规则",
            tags=["告警规则"],
            auth_required=True
        )
        async def get_alert_rules(request: APIRequest):
            rules = self.alert_manager.get_rules()
            return {
                "success": True,
                "total": len(rules),
                "rules": [
                    {
                        "rule_id": r.rule_id,
                        "name": r.name,
                        "metric_name": r.metric_name,
                        "condition": r.condition,
                        "threshold": r.threshold,
                        "severity": r.severity,
                        "enabled": r.enabled,
                        "cooldown_seconds": r.cooldown_seconds
                    }
                    for r in rules
                ]
            }

        @router.post(
            "/alerts/rules",
            summary="创建告警规则",
            description="创建新的告警规则",
            tags=["告警规则"],
            auth_required=True,
            request_schema={
                "rule_id": {"type": "string", "required": True},
                "name": {"type": "string", "required": True},
                "metric_name": {"type": "string", "required": True},
                "condition": {"type": "string", "required": True, "enum": ["gt", "lt", "gte", "lte", "eq"]},
                "threshold": {"type": "number", "required": True},
                "severity": {"type": "string", "required": True, "enum": ["info", "warning", "critical"]},
                "cooldown_seconds": {"type": "integer", "required": False},
                "message_template": {"type": "string", "required": False}
            }
        )
        async def create_alert_rule(request: APIRequest):
            data = request.body

            rule = AlertRule(
                rule_id=data["rule_id"],
                name=data["name"],
                metric_name=data["metric_name"],
                condition=data["condition"],
                threshold=data["threshold"],
                severity=data["severity"],
                cooldown_seconds=data.get("cooldown_seconds", 300),
                message_template=data.get("message_template", "")
            )

            self.alert_manager.add_rule(rule)

            return {
                "success": True,
                "message": f"告警规则 {rule.rule_id} 已创建"
            }

        @router.delete(
            "/alerts/rules/{rule_id}",
            summary="删除告警规则",
            description="删除指定的告警规则",
            tags=["告警规则"],
            auth_required=True
        )
        async def delete_alert_rule(request: APIRequest):
            rule_id = request.path_params.get("rule_id")

            success = self.alert_manager.remove_rule(rule_id)

            if success:
                return {"success": True, "message": f"告警规则 {rule_id} 已删除"}
            else:
                return APIError(
                    code="RULE_NOT_FOUND",
                    message=f"告警规则 {rule_id} 不存在",
                    status_code=404
                ).to_response()

        return router


def create_monitoring_api_module(
    dashboard: Optional[DashboardDataProvider] = None,
    alert_manager: Optional[AlertManager] = None
) -> MonitoringAPIModule:
    """创建监控API模块"""
    return MonitoringAPIModule(dashboard, alert_manager)


# 认证级别枚举 (与rest_api兼容)
class AuthLevel:
    """认证级别"""
    NONE = "none"
    USER = "user"
    ADMIN = "admin"
    SYSTEM = "system"


# 速率限制规则
@dataclass
class RateLimitRule:
    """速率限制规则"""
    requests_per_minute: int
    requests_per_hour: int = 0
    burst_limit: int = 0
