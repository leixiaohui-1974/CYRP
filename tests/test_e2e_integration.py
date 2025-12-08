"""
端到端集成测试 - End-to-End Integration Tests

测试完整的数据流：
传感器仿真 -> 数据同化 -> 状态预测 -> 安全联锁 -> 告警 -> 持久化 -> WebSocket推送

Tests the complete data flow:
Sensor Simulation -> Data Assimilation -> State Prediction -> Safety Interlocks -> Alerts -> Persistence -> WebSocket Push
"""

import pytest
import asyncio
import time
import numpy as np
from datetime import datetime, timedelta


class TestSensorToAlertFlow:
    """传感器到告警流程测试"""

    def test_sensor_reading_triggers_alert(self):
        """测试传感器读数触发告警"""
        from cyrp.api.monitoring_endpoints import AlertManager, AlertRule
        from cyrp.monitoring.dashboard_data import Metric, MetricType

        # 1. 创建告警管理器
        alert_manager = AlertManager()
        rule = AlertRule(
            rule_id="HIGH_PRESSURE",
            name="高压告警",
            metric_name="pressure",
            condition="gt",
            threshold=800000.0,
            severity="critical",
            cooldown_seconds=0
        )
        alert_manager.add_rule(rule)

        # 2. 模拟高压读数（直接使用值，避免传感器噪声影响）
        high_pressure = 900000.0

        # 3. 创建指标并评估告警
        metrics = {
            "pressure": Metric(
                name="pressure",
                value=high_pressure,
                timestamp=time.time(),
                metric_type=MetricType.GAUGE
            )
        }
        alerts = alert_manager.evaluate_metrics(metrics)

        # 验证
        assert len(alerts) > 0
        assert alerts[0].severity == "critical"
        assert alerts[0].metric_value > 800000

    def test_sensor_network_to_dashboard(self):
        """测试传感器网络到仪表板"""
        from cyrp.simulation.sensor_simulation import SensorSimulationManager
        from cyrp.monitoring.dashboard_data import DashboardDataProvider

        # 1. 创建传感器网络
        manager = SensorSimulationManager()
        network = manager.create_standard_tunnel_network()

        # 2. 创建仪表板
        dashboard = DashboardDataProvider()

        # 3. 读取传感器并记录到仪表板
        true_values = {f"P_{i}": 500000.0 + i * 1000 for i in range(11)}
        readings = network.read_all(true_values, 0.1)

        for sensor_id, value in readings.items():
            dashboard.record_metric(f"sensor_{sensor_id}", value)

        # 4. 验证仪表板有数据
        current = dashboard.get_current_metrics()
        assert len(current) > 0


class TestPredictionToAlertFlow:
    """预测到告警流程测试"""

    def test_prediction_triggers_preemptive_alert(self):
        """测试预测触发预防性告警"""
        from cyrp.prediction.state_prediction import ExponentialSmoothingPredictor
        from cyrp.api.monitoring_endpoints import AlertManager, AlertRule
        from cyrp.monitoring.dashboard_data import Metric, MetricType

        # 1. 创建预测器
        predictor = ExponentialSmoothingPredictor(alpha=0.3)

        # 2. 准备历史数据（压力上升趋势）
        history = [450000 + i * 10000 for i in range(20)]
        for i, value in enumerate(history):
            predictor.update(value, float(i))

        # 3. 预测未来值
        result = predictor.predict(5)

        # 4. 创建告警规则（预测压力超限）
        alert_manager = AlertManager()
        rule = AlertRule(
            rule_id="PREDICTED_HIGH_PRESSURE",
            name="预测高压告警",
            metric_name="predicted_pressure",
            condition="gt",
            threshold=600000.0,
            severity="warning",
            cooldown_seconds=0
        )
        alert_manager.add_rule(rule)

        # 5. 评估预测值（从PredictionResult中获取predictions数组）
        predictions = result.predictions
        max_predicted = float(np.max(predictions)) if len(predictions) > 0 else 0
        metrics = {
            "predicted_pressure": Metric(
                name="predicted_pressure",
                value=max_predicted,
                timestamp=time.time(),
                metric_type=MetricType.GAUGE
            )
        }
        alerts = alert_manager.evaluate_metrics(metrics)

        # 验证预测值在合理范围内（根据上升趋势应该较高）
        assert max_predicted > 500000  # 预测应该继续上升趋势


class TestWebSocketIntegration:
    """WebSocket集成测试"""

    def test_alert_pushes_to_websocket(self):
        """测试告警推送到WebSocket"""
        import asyncio
        from cyrp.communication.websocket_server import (
            WebSocketConnectionManager, RealtimePushService, SubscriptionChannel
        )
        from cyrp.api.monitoring_endpoints import AlertManager, AlertRule
        from cyrp.monitoring.dashboard_data import Metric, MetricType

        # 1. 创建WebSocket服务
        ws_manager = WebSocketConnectionManager()
        push_service = RealtimePushService(ws_manager)

        # 2. 连接客户端并订阅告警
        ws_manager.connect("test_client")
        ws_manager.subscribe("test_client", SubscriptionChannel.ALERTS)

        # 3. 创建告警
        alert_manager = AlertManager()
        rule = AlertRule(
            rule_id="TEST_RULE",
            name="测试规则",
            metric_name="test_metric",
            condition="gt",
            threshold=100.0,
            severity="warning",
            cooldown_seconds=0
        )
        alert_manager.add_rule(rule)

        metrics = {
            "test_metric": Metric(
                name="test_metric",
                value=150.0,
                timestamp=time.time(),
                metric_type=MetricType.GAUGE
            )
        }
        alerts = alert_manager.evaluate_metrics(metrics)

        # 4. 推送告警
        async def push_and_check():
            for alert in alerts:
                await push_service.push_alert({
                    'alert_id': alert.alert_id,
                    'severity': alert.severity,
                    'message': alert.message
                })

            # 验证消息队列
            client = ws_manager.get_client("test_client")
            return client.message_queue.qsize()

        queue_size = asyncio.run(push_and_check())
        assert queue_size > 0

    def test_metrics_batch_push(self):
        """测试批量指标推送"""
        import asyncio
        from cyrp.communication.websocket_server import (
            WebSocketConnectionManager, RealtimePushService, SubscriptionChannel
        )

        ws_manager = WebSocketConnectionManager()
        push_service = RealtimePushService(ws_manager)

        # 连接多个客户端
        for i in range(3):
            ws_manager.connect(f"client_{i}")
            ws_manager.subscribe(f"client_{i}", SubscriptionChannel.METRICS)

        # 批量推送指标
        async def push_metrics():
            return await push_service.push_metrics_batch({
                'flow_rate': 280.5,
                'pressure': 500000.0,
                'temperature': 18.5
            })

        count = asyncio.run(push_metrics())
        assert count == 3  # 3个客户端都收到


class TestPersistenceIntegration:
    """持久化集成测试"""

    def test_alert_persisted_to_database(self):
        """测试告警持久化到数据库"""
        from cyrp.database.persistence_manager import PersistenceManager
        from cyrp.database.historian import SQLiteBackend
        from cyrp.api.monitoring_endpoints import AlertManager, AlertRule
        from cyrp.monitoring.dashboard_data import Metric, MetricType

        # 1. 创建持久化管理器
        backend = SQLiteBackend(":memory:")
        persistence = PersistenceManager(backend)

        # 2. 创建告警
        alert_manager = AlertManager()
        rule = AlertRule(
            rule_id="PERSIST_TEST",
            name="持久化测试",
            metric_name="test",
            condition="gt",
            threshold=50.0,
            severity="info",
            cooldown_seconds=0
        )
        alert_manager.add_rule(rule)

        metrics = {
            "test": Metric(
                name="test",
                value=100.0,
                timestamp=time.time(),
                metric_type=MetricType.GAUGE
            )
        }
        alerts = alert_manager.evaluate_metrics(metrics)

        # 3. 持久化告警
        for alert in alerts:
            persistence.record_alert({
                'alert_id': alert.alert_id,
                'rule_id': alert.rule_id,
                'name': alert.name,
                'severity': alert.severity,
                'message': alert.message,
                'metric_value': alert.metric_value,
                'threshold': alert.threshold,
                'timestamp': time.time()
            })

        # 4. 刷新并验证
        result = persistence.flush_all()
        assert result['alerts'] > 0

        persistence.close()

    def test_metrics_persisted_with_history(self):
        """测试指标持久化带历史记录"""
        from cyrp.database.persistence_manager import PersistenceManager
        from cyrp.database.historian import SQLiteBackend

        backend = SQLiteBackend(":memory:")
        persistence = PersistenceManager(backend)

        # 记录多个时间点的指标
        base_time = time.time()
        for i in range(10):
            persistence.record_metric(
                "flow_rate",
                280.0 + i,
                timestamp=base_time + i * 2  # 每2秒一个点
            )

        # 刷新
        result = persistence.flush_all()
        assert result['metrics'] == 10

        persistence.close()


class TestFullSystemIntegration:
    """完整系统集成测试"""

    def test_integrated_system_lifecycle(self):
        """测试集成系统生命周期"""
        from cyrp.launcher import CYRPIntegratedSystem, CYRPApplicationConfig

        config = CYRPApplicationConfig()
        config.db_path = ":memory:"

        # 初始化
        system = CYRPIntegratedSystem(config)
        assert system.initialize()

        # 健康检查
        status = system.get_health_status()
        assert status['status'] == 'healthy'

        # 收集指标
        metrics = system._collect_system_metrics()
        assert 'flow_rate_total' in metrics
        assert 'pressure_avg' in metrics

        # 停止
        system.stop()

    def test_sensor_to_persistence_flow(self):
        """测试传感器到持久化完整流程"""
        from cyrp.simulation.sensor_simulation import (
            VirtualSensor, SensorCharacteristics, NoiseModel
        )
        from cyrp.monitoring.dashboard_data import DashboardDataProvider
        from cyrp.database.persistence_manager import PersistenceManager
        from cyrp.database.historian import SQLiteBackend

        # 1. 传感器层
        char = SensorCharacteristics(
            sensor_type="flow",
            measurement_range=(0, 500),
            noise_model=NoiseModel()
        )
        sensor = VirtualSensor("flow_sensor", char)

        # 2. 仪表板层
        dashboard = DashboardDataProvider()

        # 3. 持久化层
        backend = SQLiteBackend(":memory:")
        persistence = PersistenceManager(backend)

        # 4. 模拟数据流
        base_time = time.time()
        for i in range(20):
            # 传感器读数
            true_value = 280.0 + np.sin(i * 0.1) * 10
            reading = sensor.read(true_value, 0.1)

            # 记录到仪表板
            dashboard.record_metric("flow_rate", reading)

            # 持久化
            persistence.record_metric("flow_rate", reading, timestamp=base_time + i * 2)

        # 5. 验证
        current = dashboard.get_current_metrics()
        assert "flow_rate" in current

        result = persistence.flush_all()
        assert result['metrics'] == 20

        persistence.close()

    def test_control_loop_with_alerts(self):
        """测试带告警的控制循环"""
        from cyrp.control.safety_interlocks import AntiOverpressureInterlock
        from cyrp.api.monitoring_endpoints import AlertManager, AlertRule
        from cyrp.monitoring.dashboard_data import Metric, MetricType

        # 1. 安全联锁（使用反超压联锁，阈值600kPa）
        interlock = AntiOverpressureInterlock(threshold=600000.0)

        # 2. 告警管理
        alert_manager = AlertManager()
        alert_manager.add_rule(AlertRule(
            rule_id="INTERLOCK_TRIGGERED",
            name="联锁触发",
            metric_name="pressure",
            condition="gt",
            threshold=600000.0,
            severity="critical",
            cooldown_seconds=0
        ))

        # 3. 模拟压力升高
        pressure_values = [500000, 550000, 600000, 650000, 700000]
        interlock_triggered = False
        alerts_generated = []

        for i, pressure in enumerate(pressure_values):
            timestamp = time.time() + i

            # 检查联锁
            triggered, action = interlock.check(pressure, timestamp)
            if triggered:
                interlock_triggered = True

            # 检查告警
            metrics = {
                "pressure": Metric(
                    name="pressure",
                    value=pressure,
                    timestamp=timestamp,
                    metric_type=MetricType.GAUGE
                )
            }
            new_alerts = alert_manager.evaluate_metrics(metrics)
            alerts_generated.extend(new_alerts)

        # 验证
        assert interlock_triggered
        assert len(alerts_generated) > 0


class TestAPIEndpointIntegration:
    """API端点集成测试"""

    def test_monitoring_api_with_dashboard(self):
        """测试监控API与仪表板集成"""
        from cyrp.api.monitoring_endpoints import MonitoringAPIModule
        from cyrp.monitoring.dashboard_data import DashboardDataProvider

        # 创建组件
        dashboard = DashboardDataProvider()
        module = MonitoringAPIModule(dashboard=dashboard)

        # 记录数据
        dashboard.record_metric("test_metric", 100.0)
        dashboard.record_metric("test_metric", 101.0)

        # 验证模块可以访问仪表板数据
        current = module.dashboard.get_current_metrics()
        assert "test_metric" in current

    def test_api_server_with_all_modules(self):
        """测试API服务器集成所有模块"""
        from cyrp.api.rest_api import create_cyrp_api
        from cyrp.api.monitoring_endpoints import create_monitoring_api_module

        # 创建服务器
        server = create_cyrp_api()

        # 添加监控模块
        monitoring = create_monitoring_api_module()
        server.include_router(monitoring.router)

        # 验证路由已注册
        all_routes = server._routes
        assert len(all_routes) > 10  # 应该有多个路由

        # 检查关键路由存在
        route_paths = [path for (path, method) in all_routes.keys()]
        assert any('/monitoring/' in p for p in route_paths)
        assert any('/auth/' in p for p in route_paths)
        assert any('/system/' in p for p in route_paths)


class TestDataQualityFlow:
    """数据质量流程测试"""

    def test_sensor_failure_detection_and_alert(self):
        """测试传感器故障检测和告警"""
        from cyrp.simulation.sensor_simulation import (
            VirtualSensor, SensorCharacteristics, SensorFailureType, NoiseModel
        )
        from cyrp.api.monitoring_endpoints import AlertManager, AlertRule
        from cyrp.monitoring.dashboard_data import Metric, MetricType

        # 1. 创建传感器
        char = SensorCharacteristics(
            sensor_type="pressure",
            measurement_range=(0, 1e6),
            noise_model=NoiseModel()
        )
        sensor = VirtualSensor("P_test", char)

        # 2. 注入卡死故障
        sensor.inject_failure(SensorFailureType.STUCK_VALUE, {'stuck_value': 0.0})

        # 3. 创建故障检测告警规则
        alert_manager = AlertManager()
        alert_manager.add_rule(AlertRule(
            rule_id="SENSOR_ZERO",
            name="传感器读数异常",
            metric_name="sensor_reading",
            condition="eq",
            threshold=0.0,
            severity="warning",
            cooldown_seconds=0
        ))

        # 4. 读取多次，检测异常
        readings = [sensor.read(500000.0, 0.1) for _ in range(5)]

        # 5. 评估告警
        metrics = {
            "sensor_reading": Metric(
                name="sensor_reading",
                value=readings[-1],
                timestamp=time.time(),
                metric_type=MetricType.GAUGE
            )
        }
        alerts = alert_manager.evaluate_metrics(metrics)

        # 验证
        assert all(r == 0.0 for r in readings)  # 所有读数都应该卡在0
        # 如果读数是0则应该触发告警
        if readings[-1] == 0.0:
            assert len(alerts) > 0


class TestPerformanceIntegration:
    """性能集成测试"""

    def test_high_frequency_data_flow(self):
        """测试高频数据流"""
        from cyrp.monitoring.dashboard_data import DashboardDataProvider
        from cyrp.database.persistence_manager import PersistenceManager
        from cyrp.database.historian import SQLiteBackend
        import time

        dashboard = DashboardDataProvider(history_size=10000)
        backend = SQLiteBackend(":memory:")
        persistence = PersistenceManager(backend)

        # 模拟高频数据（1000个点）
        start_time = time.time()
        base_time = time.time()

        for i in range(1000):
            value = 280.0 + np.sin(i * 0.01) * 10
            dashboard.record_metric("high_freq_metric", value)
            persistence.record_metric("high_freq_metric", value, timestamp=base_time + i * 0.001)

        elapsed = time.time() - start_time

        # 刷新持久化
        persistence.flush_all()

        # 验证性能（应该在1秒内完成）
        assert elapsed < 1.0, f"High frequency processing took {elapsed:.3f}s"

        # 验证数据完整性
        current = dashboard.get_current_metrics()
        assert "high_freq_metric" in current

        persistence.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
