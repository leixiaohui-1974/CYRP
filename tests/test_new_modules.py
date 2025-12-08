"""
新增模块测试 - Tests for New Modules

测试传感器仿真、执行器仿真、数据治理、数据同化、
IDZ参数更新、状态评价和状态预测模块
"""

import pytest
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestSensorSimulation:
    """传感器仿真测试"""

    def test_noise_model_gaussian(self):
        """测试高斯噪声模型"""
        from cyrp.simulation.sensor_simulation import NoiseModel, NoiseType

        noise_model = NoiseModel(NoiseType.GAUSSIAN, amplitude=0.1)
        samples = [noise_model.generate() for _ in range(1000)]

        assert abs(np.mean(samples)) < 0.05  # 均值接近0
        assert 0.05 < np.std(samples) < 0.15  # 标准差接近amplitude

    def test_drift_model(self):
        """测试漂移模型"""
        from cyrp.simulation.sensor_simulation import DriftModel, DriftType

        drift_model = DriftModel(DriftType.LINEAR, rate=0.1)

        drift_1h = drift_model.compute(3600)  # 1小时
        assert abs(drift_1h - 0.1) < 0.01

    def test_virtual_sensor(self):
        """测试虚拟传感器"""
        from cyrp.simulation.sensor_simulation import (
            VirtualSensor, SensorCharacteristics, NoiseModel, DriftModel, NoiseType
        )

        char = SensorCharacteristics(
            sensor_type="pressure",
            measurement_range=(0, 1e6),
            resolution=100,
            accuracy=0.1,
            response_time=0.01,
            noise_model=NoiseModel(NoiseType.GAUSSIAN, amplitude=0.001),
            drift_model=DriftModel()
        )

        sensor = VirtualSensor("P_test", char)

        # 测试读数
        true_value = 500000.0
        reading = sensor.read(true_value, 0.01)

        # 验证读数是有效数值且在测量范围内
        assert not np.isnan(reading), "读数不应为NaN"
        assert not np.isinf(reading), "读数不应为无穷大"
        assert 0 <= reading <= 1e6, "读数应在测量范围内"

    def test_sensor_network(self):
        """测试传感器网络"""
        from cyrp.simulation.sensor_simulation import (
            SensorSimulationManager
        )

        manager = SensorSimulationManager()
        network = manager.create_standard_tunnel_network()

        assert len(network.sensors) > 0

        # 测试读取
        true_values = {f"P_{i}": 500000.0 + i * 1000 for i in range(11)}
        readings = network.read_all(true_values, 0.1)

        assert len(readings) > 0

    def test_failure_injection(self):
        """测试故障注入"""
        from cyrp.simulation.sensor_simulation import (
            VirtualSensor, SensorCharacteristics, SensorFailureType, NoiseModel
        )

        char = SensorCharacteristics(
            sensor_type="test",
            measurement_range=(0, 100),
            noise_model=NoiseModel()
        )
        sensor = VirtualSensor("test", char)

        # 注入卡死故障
        sensor.inject_failure(SensorFailureType.STUCK_VALUE, {'stuck_value': 50.0})

        reading = sensor.read(80.0, 0.1)
        assert reading == 50.0


class TestActuatorSimulation:
    """执行器仿真测试"""

    def test_valve_dynamics(self):
        """测试阀门动力学"""
        from cyrp.simulation.actuator_simulation import (
            VirtualValve, ValveCharacteristics, ActuatorType
        )

        char = ValveCharacteristics(
            valve_type=ActuatorType.GATE_VALVE,
            nominal_diameter=1.0,
            cv=1000,
            stroke_time=10.0,
            characteristic="linear"
        )

        valve = VirtualValve("test_valve", char)
        valve.position = 1.0  # 初始全开

        # 命令关闭
        for _ in range(100):
            valve.step(0.0, 0.1)

        # 阀门应该接近关闭
        assert valve.position < 0.1

    def test_pump_model(self):
        """测试泵模型"""
        from cyrp.simulation.actuator_simulation import (
            VirtualPump, PumpCharacteristics
        )

        char = PumpCharacteristics(
            rated_flow=1.0,
            rated_head=50.0,
            rated_speed=1480,
            rated_power=100,
            rated_efficiency=0.85
        )

        pump = VirtualPump("test_pump", char)

        # 启动泵
        for _ in range(200):
            pump.step(1.0, 0.1, 30.0)  # 30m系统扬程

        assert pump.is_running
        assert pump.speed > 1000

    def test_actuator_network(self):
        """测试执行器网络"""
        from cyrp.simulation.actuator_simulation import (
            ActuatorSimulationManager
        )

        manager = ActuatorSimulationManager()
        network = manager.create_standard_tunnel_network()

        assert len(network.actuators) > 0

        # 测试步进
        commands = {name: 0.5 for name in network.actuators}
        outputs = network.step_all(commands, 0.1)

        assert len(outputs) > 0


class TestDataGovernance:
    """数据治理测试"""

    def test_quality_rule(self):
        """测试质量规则"""
        from cyrp.governance.data_governance import (
            QualityRule, RuleType, QualityDimension
        )

        rule = QualityRule(
            rule_id="range_test",
            rule_type=RuleType.RANGE_CHECK,
            dimension=QualityDimension.VALIDITY,
            name="Range Test",
            description="Test range check",
            parameters={'min': 0, 'max': 100}
        )

        passed, msg = rule.validate(50)
        assert passed

        passed, msg = rule.validate(150)
        assert not passed

    def test_quality_engine(self):
        """测试质量引擎"""
        from cyrp.governance.data_governance import DataQualityEngine

        engine = DataQualityEngine()
        engine.create_standard_rules()

        # 测试数据
        data = np.array([100000, 200000, 300000, np.nan, 500000])
        metrics = engine.check_quality(data, rule_ids=["null_check"])

        assert metrics.completeness_score < 1.0  # 有空值

    def test_data_standardizer(self):
        """测试数据标准化器"""
        from cyrp.governance.data_governance import DataStandardizer

        standardizer = DataStandardizer()
        standardizer.create_sensor_standards()

        # 测试压力标准化
        data = np.array([0.5e6, 1.0e6, 1.5e6])
        result = standardizer.standardize(data, "pressure_pa")

        assert len(result) == 3


class TestDataAssimilation:
    """数据同化测试"""

    def test_kalman_filter(self):
        """测试卡尔曼滤波"""
        from cyrp.assimilation.data_assimilation import KalmanFilter

        kf = KalmanFilter(state_dim=2, obs_dim=1)
        kf.set_initial_state(np.array([0.0, 0.0]))

        # 模拟观测
        for i in range(10):
            obs = np.array([i * 1.0 + np.random.randn() * 0.1])
            result = kf.assimilate(obs, 1.0)

        assert result.state_estimate is not None
        assert len(result.state_estimate) == 2

    def test_extended_kalman_filter(self):
        """测试扩展卡尔曼滤波"""
        from cyrp.assimilation.data_assimilation import ExtendedKalmanFilter

        ekf = ExtendedKalmanFilter(state_dim=2, obs_dim=1)
        ekf.set_initial_state(np.array([0.0, 0.0]))

        for i in range(10):
            obs = np.array([i * 1.0])
            result = ekf.assimilate(obs, 1.0)

        assert result.state_estimate is not None

    def test_ensemble_kalman_filter(self):
        """测试集合卡尔曼滤波"""
        from cyrp.assimilation.data_assimilation import EnsembleKalmanFilter

        enkf = EnsembleKalmanFilter(state_dim=2, obs_dim=1, ensemble_size=20)
        enkf.set_initial_state(np.array([0.0, 0.0]))

        for i in range(10):
            obs = np.array([i * 1.0])
            result = enkf.assimilate(obs, 1.0)

        assert result.spread > 0  # 集合离散度

    def test_assimilation_manager(self):
        """测试同化管理器"""
        from cyrp.assimilation.data_assimilation import (
            DataAssimilationManager, AssimilationMethod
        )

        manager = DataAssimilationManager()
        manager.create_assimilator(
            "test_kf",
            AssimilationMethod.KALMAN,
            state_dim=2,
            obs_dim=1
        )

        result = manager.assimilate("test_kf", np.array([1.0]), 1.0)
        assert result is not None


class TestIDZParameterUpdater:
    """IDZ参数更新测试"""

    def test_recursive_least_squares(self):
        """测试递推最小二乘"""
        from cyrp.idz.parameter_updater import RecursiveLeastSquares

        rls = RecursiveLeastSquares(n_params=2)

        # 模拟数据: y = 2*x1 + 3*x2
        for _ in range(100):
            x1 = np.random.randn()
            x2 = np.random.randn()
            y = 2 * x1 + 3 * x2 + np.random.randn() * 0.1

            phi = np.array([x1, x2])
            theta, error = rls.update(phi, y)

        # 参数应该接近 [2, 3]
        assert abs(theta[0] - 2) < 0.5
        assert abs(theta[1] - 3) < 0.5

    def test_system_identification(self):
        """测试系统辨识"""
        from cyrp.idz.parameter_updater import SystemIdentification, ModelStructure

        sys_id = SystemIdentification(
            model_structure=ModelStructure.ARX,
            na=2, nb=2, nk=1
        )

        # 模拟数据
        for i in range(100):
            u = np.sin(i * 0.1)
            y = 0.5 * u + np.random.randn() * 0.01
            result = sys_id.update(y, u)

        assert result.parameters is not None
        assert result.model_fit > 0

    def test_parameter_updater(self):
        """测试参数更新器"""
        from cyrp.idz.parameter_updater import IDZParameterUpdater

        updater = IDZParameterUpdater()

        hifi_state = {
            'flow_rate': np.array([130, 135, 132, 128, 133]),
            'pressure': np.array([500000, 510000, 505000, 495000, 502000]),
            'water_level': np.array([6.0, 6.1, 6.0, 5.9, 6.0])
        }
        idz_state = {
            'pressure': np.array([500000, 510000, 505000, 495000, 502000])
        }
        control_input = np.array([0.8, 0.8])

        results = updater.update_from_hifi_model(hifi_state, idz_state, control_input, 1.0)
        assert 'hydraulic' in results


class TestStateEvaluation:
    """状态评价测试"""

    def test_objective_tracker(self):
        """测试目标跟踪器"""
        from cyrp.evaluation.state_evaluation import (
            ObjectiveTracker, ControlObjective, ObjectiveType
        )

        objective = ControlObjective(
            name="flow",
            objective_type=ObjectiveType.SETPOINT,
            target_value=265.0,
            tolerance=0.05
        )

        tracker = ObjectiveTracker(objective)

        # 模拟跟踪
        for i in range(100):
            actual = 265.0 + np.sin(i * 0.1) * 5
            metrics = tracker.update(actual, i * 0.1, 0.1)

        assert metrics.rmse > 0
        assert 0 <= metrics.overall_score <= 1

    def test_safety_evaluator(self):
        """测试安全评价器"""
        from cyrp.evaluation.state_evaluation import SafetyEvaluator, SafetyStatus

        evaluator = SafetyEvaluator()
        evaluator.set_limits("pressure", 0, 1000000, 0.1, 0.05)

        # 正常值
        status, warnings, alarms = evaluator.evaluate({"pressure": 500000}, 0)
        assert status == SafetyStatus.SAFE

        # 报警值
        status, warnings, alarms = evaluator.evaluate({"pressure": 980000}, 1)
        assert status in [SafetyStatus.WARNING, SafetyStatus.ALARM]

    def test_state_evaluator(self):
        """测试状态评价器"""
        from cyrp.evaluation.state_evaluation import StateEvaluator

        evaluator = StateEvaluator()

        state = {
            "flow_rate": 260.0,
            "pressure": 520000.0,
            "water_level": 5.9
        }
        targets = {
            "flow_rate": 265.0,
            "pressure": 500000.0,
            "water_level": 6.0
        }

        result = evaluator.evaluate(state, targets, 1.0, 0.1)

        assert result.overall_score > 0
        assert result.safety_status is not None


class TestStatePrediction:
    """状态预测测试"""

    def test_arima_predictor(self):
        """测试ARIMA预测器"""
        from cyrp.prediction.state_prediction import ARIMAPredictor

        predictor = ARIMAPredictor(p=2, d=1, q=1)

        # 生成测试数据
        data = np.cumsum(np.random.randn(200)) + 100

        predictor.fit(data)
        for v in data:
            predictor.update(v, 0)

        result = predictor.predict(horizon=10)

        assert len(result.predictions) == 10
        assert result.confidence_intervals is not None

    def test_exponential_smoothing(self):
        """测试指数平滑预测"""
        from cyrp.prediction.state_prediction import ExponentialSmoothingPredictor

        predictor = ExponentialSmoothingPredictor(alpha=0.3, beta=0.1)

        data = np.arange(100) + np.random.randn(100) * 2

        predictor.fit(data)
        for i, v in enumerate(data):
            predictor.update(v, i)

        result = predictor.predict(horizon=10)

        assert len(result.predictions) == 10
        # 趋势应该继续上升
        assert result.predictions[-1] > result.predictions[0]

    def test_physics_predictor(self):
        """测试物理预测器"""
        from cyrp.prediction.state_prediction import PhysicsBasedPredictor

        predictor = PhysicsBasedPredictor()
        predictor.set_state(np.array([130, 135, 6.0, 5.9]))

        result = predictor.predict(horizon=10, dt=1.0)

        assert len(result.predictions) == 10

    def test_ensemble_predictor(self):
        """测试集成预测器"""
        from cyrp.prediction.state_prediction import (
            EnsemblePredictor, ARIMAPredictor, ExponentialSmoothingPredictor
        )

        ensemble = EnsemblePredictor()
        ensemble.add_predictor(ARIMAPredictor(), weight=0.5)
        ensemble.add_predictor(ExponentialSmoothingPredictor(), weight=0.5)

        data = np.arange(100) + np.random.randn(100)
        ensemble.fit(data)

        for i, v in enumerate(data):
            ensemble.update(v, i)

        result = ensemble.predict(horizon=10)

        assert len(result.predictions) == 10
        assert result.uncertainty is not None

    def test_predictor_manager(self):
        """测试预测管理器"""
        from cyrp.prediction.state_prediction import StatePredictorManager

        manager = StatePredictorManager()

        # 更新数据
        for i in range(100):
            values = {'flow_rate': 265 + np.sin(i * 0.1) * 5}
            manager.update(values, i * 0.1)

        # 预测
        result = manager.predict('flow_rate', horizon=10)

        assert result is not None
        assert len(result.predictions) == 10


class TestUncertaintyQuantification:
    """不确定性量化测试"""

    def test_bayesian_quantifier(self):
        """测试贝叶斯不确定性量化器"""
        from cyrp.prediction.uncertainty_quantification import BayesianUncertaintyQuantifier

        quantifier = BayesianUncertaintyQuantifier(
            prior_variance=1.0,
            observation_noise=0.1
        )

        # 模拟预测和实际值
        for i in range(50):
            predicted = 100 + np.sin(i * 0.1) * 5
            actual = predicted + np.random.randn() * 2
            quantifier.update_posterior(predicted, actual)

        # 量化不确定性
        predictions = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        estimate = quantifier.quantify(predictions)

        assert estimate.mean is not None
        assert estimate.variance is not None
        assert len(estimate.aleatoric) == len(predictions)
        assert len(estimate.epistemic) == len(predictions)

        # 获取置信区间
        lower, upper = estimate.get_interval(0.95)
        assert np.all(lower < estimate.mean)
        assert np.all(upper > estimate.mean)

    def test_ensemble_fusion(self):
        """测试集成不确定性融合"""
        from cyrp.prediction.uncertainty_quantification import EnsembleUncertaintyFusion

        fusion = EnsembleUncertaintyFusion(n_models=3)

        # 三个模型的预测
        predictions = [
            np.array([100.0, 101.0, 102.0]),
            np.array([100.5, 101.5, 102.5]),
            np.array([99.5, 100.5, 101.5])
        ]
        variances = [
            np.array([1.0, 1.5, 2.0]),
            np.array([1.2, 1.8, 2.5]),
            np.array([0.8, 1.2, 1.8])
        ]

        estimate = fusion.fuse(predictions, variances)

        assert estimate.mean is not None
        assert len(estimate.mean) == 3
        assert estimate.epistemic is not None  # 模型间分歧

        # 更新权重
        fusion.update_weights([0.5, 1.0, 0.3])
        assert np.sum(fusion.model_weights) - 1.0 < 1e-6

    def test_conformal_predictor(self):
        """测试保型预测"""
        from cyrp.prediction.uncertainty_quantification import ConformalPredictor

        predictor = ConformalPredictor(coverage=0.95)

        # 模拟历史数据
        for i in range(100):
            predicted = 50 + i * 0.1
            actual = predicted + np.random.randn() * 2
            predictor.update(predicted, actual)

        # 预测区间
        lower, upper = predictor.predict_interval(60.0)
        assert lower < 60.0
        assert upper > 60.0

        # 批量预测
        predictions = np.array([60.0, 61.0, 62.0])
        lowers, uppers = predictor.predict_intervals_batch(predictions)
        assert len(lowers) == 3
        assert np.all(lowers < predictions)
        assert np.all(uppers > predictions)

    def test_uncertainty_monitor(self):
        """测试不确定性监控器"""
        from cyrp.prediction.uncertainty_quantification import UncertaintyMonitor

        monitor = UncertaintyMonitor(warning_threshold=2.0, critical_threshold=3.0)

        # 建立基线
        for i in range(100):
            monitor.update(1.0 + np.random.randn() * 0.2)

        # 正常状态
        assert monitor.check_status(1.0) == "normal"

        # 预警状态
        assert monitor.check_status(2.5) == "warning"

        # 严重状态
        assert monitor.check_status(4.0) == "critical"

        # 统计信息
        stats = monitor.get_statistics()
        assert 'mean' in stats
        assert 'baseline' in stats


class TestIntegration:
    """集成测试"""

    def test_sensor_actuator_loop(self):
        """测试传感器-执行器闭环"""
        from cyrp.simulation.sensor_simulation import SensorSimulationManager
        from cyrp.simulation.actuator_simulation import ActuatorSimulationManager

        sensor_mgr = SensorSimulationManager()
        sensor_network = sensor_mgr.create_standard_tunnel_network()

        actuator_mgr = ActuatorSimulationManager()
        actuator_network = actuator_mgr.create_standard_tunnel_network()

        # 模拟几步
        for step in range(10):
            # 执行器动作
            commands = {name: 0.8 for name in actuator_network.actuators}
            act_outputs = actuator_network.step_all(commands, 0.1)

            # 传感器读取
            true_values = {f"P_{i}": 500000.0 for i in range(11)}
            sensor_data = sensor_network.read_all(true_values, 0.1)

            assert len(act_outputs) > 0
            assert len(sensor_data) > 0

    def test_full_pipeline(self):
        """测试完整管道"""
        from cyrp.simulation.sensor_simulation import SensorSimulationManager
        from cyrp.governance.data_governance import DataGovernanceManager, DataCategory
        from cyrp.assimilation.data_assimilation import DataAssimilationManager, AssimilationMethod
        from cyrp.evaluation.state_evaluation import StateEvaluator
        from cyrp.prediction.state_prediction import StatePredictorManager

        # 初始化组件
        sensor_mgr = SensorSimulationManager()
        sensor_network = sensor_mgr.create_standard_tunnel_network()

        governance_mgr = DataGovernanceManager()
        assimilation_mgr = DataAssimilationManager()
        assimilation_mgr.create_assimilator("main", AssimilationMethod.EXTENDED_KALMAN, 4, 2)

        evaluator = StateEvaluator()
        predictor_mgr = StatePredictorManager()

        # 模拟运行
        for step in range(50):
            time = step * 0.1

            # 1. 传感器数据生成
            true_values = {
                f"P_{i}": 500000.0 + np.sin(time) * 10000
                for i in range(11)
            }
            sensor_data = sensor_network.read_all(true_values, 0.1)

            # 2. 数据治理
            pressure_data = np.array([sensor_data.get(f"P_{i}", 0) for i in range(11)])
            if not np.all(np.isnan(pressure_data)):
                _, metrics = governance_mgr.ingest_data(
                    pressure_data, "pressure_array", "sensors",
                    DataCategory.SENSOR
                )

            # 3. 数据同化
            obs = np.array([np.nanmean(pressure_data), 265.0])
            result = assimilation_mgr.assimilate("main", obs, 0.1)

            # 4. 状态评价
            state = {
                "flow_rate": 265.0,
                "pressure": np.nanmean(pressure_data),
                "water_level": 6.0
            }
            targets = {
                "flow_rate": 265.0,
                "pressure": 500000.0,
                "water_level": 6.0
            }
            eval_result = evaluator.evaluate(state, targets, time, 0.1)

            # 5. 状态预测
            predictor_mgr.update(state, time)

        # 验证
        pred_result = predictor_mgr.predict("flow_rate", 10)
        assert pred_result is not None

        accuracy = predictor_mgr.get_prediction_accuracy()
        assert 'mean_rmse' in accuracy


class TestDashboardData:
    """仪表板数据测试"""

    def test_dashboard_provider(self):
        """测试仪表板数据提供器"""
        from cyrp.monitoring.dashboard_data import DashboardDataProvider

        provider = DashboardDataProvider(history_size=1000)

        # 记录指标
        for i in range(100):
            provider.record_metric('flow_rate_total', 280 + np.sin(i * 0.1) * 10)
            provider.record_metric('pressure_avg', 500000 + np.cos(i * 0.1) * 10000)

        # 获取当前指标
        current = provider.get_current_metrics()
        assert 'flow_rate_total' in current
        assert 'pressure_avg' in current

        # 获取历史
        history = provider.get_metric_history('flow_rate_total', limit=50)
        assert len(history) <= 50

    def test_prometheus_format(self):
        """测试Prometheus格式输出"""
        from cyrp.monitoring.dashboard_data import DashboardDataProvider

        provider = DashboardDataProvider()
        provider.record_metric('flow_rate_total', 280.5, labels={'tunnel': 'north'})

        prometheus_output = provider.get_prometheus_metrics()
        assert 'flow_rate_total' in prometheus_output
        assert '280.5' in prometheus_output

    def test_system_status(self):
        """测试系统状态"""
        from cyrp.monitoring.dashboard_data import (
            DashboardDataProvider, MetricsCollector, SystemStatus
        )

        provider = DashboardDataProvider()
        collector = MetricsCollector(provider)

        # 创建系统状态
        status = collector.create_system_status(
            health_score=85.0,
            flow_rate=275.0,
            pressure_avg=500000.0,
            pressure_max=550000.0,
            water_level=6.0,
            active_alarms=0
        )

        assert status.overall_health == 'good'
        assert status.health_score == 85.0
        assert status.hydraulic_status == 'normal'

        # 更新到提供器
        provider.update_system_status(status)

        # 获取实时快照
        snapshot = provider.get_realtime_snapshot()
        assert snapshot['system']['health'] == 'good'

    def test_grafana_dashboard(self):
        """测试Grafana仪表板配置"""
        from cyrp.monitoring.dashboard_data import DashboardDataProvider

        provider = DashboardDataProvider()
        dashboard = provider.get_grafana_dashboard()

        assert 'title' in dashboard
        assert 'panels' in dashboard
        assert len(dashboard['panels']) > 0
        assert dashboard['title'] == 'CYRP 穿黄工程监控仪表板'

    def test_api_response(self):
        """测试API响应格式"""
        from cyrp.monitoring.dashboard_data import DashboardDataProvider

        provider = DashboardDataProvider()
        provider.record_batch({
            'flow_rate_total': 280.0,
            'pressure_avg': 500000.0,
            'health_score': 90.0
        })

        response = provider.get_api_response()
        assert 'timestamp' in response
        assert 'metrics' in response
        assert 'panels' in response


class TestAlertManager:
    """告警管理器测试"""

    def test_alert_rule_creation(self):
        """测试告警规则创建"""
        from cyrp.api.monitoring_endpoints import AlertManager, AlertRule

        manager = AlertManager()

        # 默认规则应已创建
        rules = manager.get_rules()
        assert len(rules) > 0

        # 添加自定义规则
        custom_rule = AlertRule(
            rule_id="CUSTOM_RULE_1",
            name="自定义测试规则",
            metric_name="test_metric",
            condition="gt",
            threshold=100.0,
            severity="warning"
        )
        manager.add_rule(custom_rule)

        rules = manager.get_rules()
        rule_ids = [r.rule_id for r in rules]
        assert "CUSTOM_RULE_1" in rule_ids

    def test_alert_evaluation(self):
        """测试告警评估"""
        from cyrp.api.monitoring_endpoints import AlertManager, AlertRule
        from cyrp.monitoring.dashboard_data import Metric, MetricType

        manager = AlertManager()

        # 添加测试规则
        rule = AlertRule(
            rule_id="TEST_HIGH_VALUE",
            name="高值告警",
            metric_name="test_value",
            condition="gt",
            threshold=50.0,
            severity="critical",
            cooldown_seconds=1  # 短冷却时间便于测试
        )
        manager.add_rule(rule)

        # 创建超阈值的指标
        metrics = {
            "test_value": Metric(
                name="test_value",
                value=75.0,  # 超过阈值50
                timestamp=1234567890.0,
                metric_type=MetricType.GAUGE
            )
        }

        # 评估
        new_alerts = manager.evaluate_metrics(metrics)
        assert len(new_alerts) > 0

        # 检查告警内容
        alert = new_alerts[0]
        assert alert.severity == "critical"
        assert alert.metric_value == 75.0

    def test_alert_acknowledge_and_resolve(self):
        """测试告警确认和解决"""
        from cyrp.api.monitoring_endpoints import AlertManager, AlertRule
        from cyrp.monitoring.dashboard_data import Metric, MetricType

        manager = AlertManager()

        # 添加规则
        rule = AlertRule(
            rule_id="TEST_RULE",
            name="测试规则",
            metric_name="test_metric",
            condition="lt",
            threshold=10.0,
            severity="warning",
            cooldown_seconds=0
        )
        manager.add_rule(rule)

        # 触发告警
        metrics = {
            "test_metric": Metric(
                name="test_metric",
                value=5.0,  # 低于阈值
                timestamp=1234567890.0,
                metric_type=MetricType.GAUGE
            )
        }
        alerts = manager.evaluate_metrics(metrics)
        assert len(alerts) > 0

        alert_id = alerts[0].alert_id

        # 确认告警
        success = manager.acknowledge_alert(alert_id, "test_user")
        assert success

        # 检查确认状态
        active = manager.get_active_alerts()
        alert = next((a for a in active if a.alert_id == alert_id), None)
        assert alert is not None
        assert alert.acknowledged
        assert alert.acknowledged_by == "test_user"

        # 解决告警
        success = manager.resolve_alert(alert_id)
        assert success

        # 确认已从活动列表移除
        active = manager.get_active_alerts()
        alert = next((a for a in active if a.alert_id == alert_id), None)
        assert alert is None

    def test_alert_statistics(self):
        """测试告警统计"""
        from cyrp.api.monitoring_endpoints import AlertManager

        manager = AlertManager()

        stats = manager.get_statistics()
        assert "total_active" in stats
        assert "critical_count" in stats
        assert "warning_count" in stats
        assert "total_rules" in stats
        assert stats["total_rules"] > 0

    def test_cooldown_mechanism(self):
        """测试冷却机制"""
        import time
        from cyrp.api.monitoring_endpoints import AlertManager, AlertRule
        from cyrp.monitoring.dashboard_data import Metric, MetricType

        manager = AlertManager()

        # 添加带冷却的规则
        rule = AlertRule(
            rule_id="COOLDOWN_TEST",
            name="冷却测试",
            metric_name="cool_metric",
            condition="gt",
            threshold=100.0,
            severity="info",
            cooldown_seconds=60  # 60秒冷却
        )
        manager.add_rule(rule)

        metrics = {
            "cool_metric": Metric(
                name="cool_metric",
                value=150.0,
                timestamp=time.time(),
                metric_type=MetricType.GAUGE
            )
        }

        # 第一次应触发
        alerts1 = manager.evaluate_metrics(metrics)
        triggered_count1 = sum(1 for a in alerts1 if a.rule_id == "COOLDOWN_TEST")
        assert triggered_count1 == 1

        # 立即再次评估，应被冷却阻止
        alerts2 = manager.evaluate_metrics(metrics)
        triggered_count2 = sum(1 for a in alerts2 if a.rule_id == "COOLDOWN_TEST")
        assert triggered_count2 == 0


class TestMonitoringAPI:
    """监控API模块测试"""

    def test_monitoring_module_creation(self):
        """测试监控API模块创建"""
        from cyrp.api.monitoring_endpoints import (
            MonitoringAPIModule, create_monitoring_api_module
        )
        from cyrp.monitoring.dashboard_data import DashboardDataProvider

        # 使用工厂函数
        module = create_monitoring_api_module()
        assert module.dashboard is not None
        assert module.alert_manager is not None
        assert module.router is not None

        # 使用自定义组件
        custom_dashboard = DashboardDataProvider(history_size=500)
        module2 = MonitoringAPIModule(dashboard=custom_dashboard)
        assert module2.dashboard == custom_dashboard

    def test_router_has_routes(self):
        """测试路由器包含路由"""
        from cyrp.api.monitoring_endpoints import create_monitoring_api_module

        module = create_monitoring_api_module()
        router = module.router

        assert len(router.routes) > 0

        # 检查关键路由存在
        route_paths = [r.path for r in router.routes]
        assert any("/monitoring/realtime" in p for p in route_paths)
        assert any("/monitoring/alerts" in p for p in route_paths)
        assert any("/monitoring/prometheus" in p for p in route_paths)

    def test_alert_rule_dataclass(self):
        """测试AlertRule数据类"""
        from cyrp.api.monitoring_endpoints import AlertRule

        rule = AlertRule(
            rule_id="TEST_RULE",
            name="测试规则",
            metric_name="pressure",
            condition="gt",
            threshold=1000000.0,
            severity="critical",
            message_template="压力 {value:.0f} Pa 超过阈值 {threshold:.0f} Pa"
        )

        assert rule.rule_id == "TEST_RULE"
        assert rule.enabled == True  # 默认值
        assert rule.cooldown_seconds == 300  # 默认值

    def test_alert_dataclass(self):
        """测试Alert数据类"""
        from cyrp.api.monitoring_endpoints import Alert
        import time

        alert = Alert(
            alert_id="ALT001",
            rule_id="RULE001",
            name="高压告警",
            severity="warning",
            message="压力过高",
            metric_value=1200000.0,
            threshold=1000000.0,
            timestamp=time.time()
        )

        assert alert.alert_id == "ALT001"
        assert alert.acknowledged == False  # 默认值
        assert alert.resolved == False  # 默认值


class TestAPIIntegration:
    """API集成测试"""

    def test_rest_api_module_import(self):
        """测试REST API模块导入"""
        from cyrp.api import (
            HTTPMethod,
            ContentType,
            APIRequest,
            APIResponse,
            JWTManager,
            RateLimiter,
            RequestValidator,
            APIRouter,
            APIServer,
            create_cyrp_api,
            AlertRule,
            Alert,
            AlertManager,
            MonitoringAPIModule,
            create_monitoring_api_module,
            AuthLevel,
            RateLimitRule,
        )

        # 验证所有导出正常
        assert HTTPMethod.GET.value == "GET"
        assert ContentType.JSON.value == "application/json"
        assert AuthLevel.ADMIN == "admin"

    def test_api_server_with_monitoring(self):
        """测试API服务器集成监控模块"""
        from cyrp.api import create_cyrp_api, create_monitoring_api_module

        # 创建API服务器
        server = create_cyrp_api()

        # 创建监控模块
        monitoring = create_monitoring_api_module()

        # 集成监控路由
        server.include_router(monitoring.router)

        # 检查路由已注册
        routes = server._routes
        monitoring_routes = [path for (path, method) in routes.keys()
                           if '/monitoring/' in path]
        assert len(monitoring_routes) > 0

    def test_jwt_manager(self):
        """测试JWT管理器"""
        from cyrp.api import JWTManager

        jwt = JWTManager("test-secret-key")

        # 创建令牌
        token = jwt.create_token(
            user_id="user123",
            username="testuser",
            roles=["operator"]
        )
        assert token is not None
        assert len(token.split('.')) == 3  # JWT格式: header.payload.signature

        # 验证令牌
        valid, payload, error = jwt.verify_token(token)
        assert valid
        assert payload['sub'] == "user123"
        assert payload['username'] == "testuser"
        assert "operator" in payload['roles']

    def test_rate_limiter(self):
        """测试速率限制器"""
        import asyncio
        from cyrp.api import RateLimiter

        limiter = RateLimiter()

        async def test_limit():
            key = "test_client"
            limit = 5

            # 前5次应该通过
            for i in range(5):
                allowed = await limiter.check(key, limit, window_seconds=60)
                assert allowed, f"Request {i+1} should be allowed"

            # 第6次应该被限制
            allowed = await limiter.check(key, limit, window_seconds=60)
            assert not allowed, "Request 6 should be rate limited"

        asyncio.run(test_limit())

    def test_request_validator(self):
        """测试请求验证器"""
        from cyrp.api import RequestValidator

        schema = {
            "name": {"type": "string", "required": True, "minLength": 1},
            "value": {"type": "number", "required": True, "minimum": 0},
            "severity": {"type": "string", "enum": ["low", "medium", "high"]}
        }

        # 有效数据
        valid_data = {"name": "test", "value": 100, "severity": "high"}
        is_valid, errors = RequestValidator.validate(valid_data, schema)
        assert is_valid
        assert len(errors) == 0

        # 缺少必填字段
        invalid_data1 = {"value": 100}
        is_valid, errors = RequestValidator.validate(invalid_data1, schema)
        assert not is_valid
        assert any("name" in e for e in errors)

        # 值超出范围
        invalid_data2 = {"name": "test", "value": -10}
        is_valid, errors = RequestValidator.validate(invalid_data2, schema)
        assert not is_valid

        # 枚举值错误
        invalid_data3 = {"name": "test", "value": 100, "severity": "invalid"}
        is_valid, errors = RequestValidator.validate(invalid_data3, schema)
        assert not is_valid


class TestWebSocketServer:
    """WebSocket服务器测试"""

    def test_connection_manager(self):
        """测试连接管理器"""
        from cyrp.communication.websocket_server import WebSocketConnectionManager

        manager = WebSocketConnectionManager()

        # 连接客户端
        client = manager.connect("client_1", user_id="user_1")
        assert client.client_id == "client_1"
        assert client.user_id == "user_1"
        assert client.is_connected

        # 获取客户端
        retrieved = manager.get_client("client_1")
        assert retrieved == client

        # 获取所有客户端
        all_clients = manager.get_all_clients()
        assert len(all_clients) == 1

        # 断开连接
        manager.disconnect("client_1")
        assert manager.get_client("client_1") is None

    def test_subscription(self):
        """测试订阅功能"""
        from cyrp.communication.websocket_server import (
            WebSocketConnectionManager, SubscriptionChannel
        )

        manager = WebSocketConnectionManager()

        # 连接客户端
        manager.connect("client_1")
        manager.connect("client_2")

        # 订阅频道
        success = manager.subscribe("client_1", SubscriptionChannel.ALERTS)
        assert success

        manager.subscribe("client_1", SubscriptionChannel.METRICS)
        manager.subscribe("client_2", SubscriptionChannel.ALERTS)

        # 检查订阅者
        alert_subs = manager.get_subscribers(SubscriptionChannel.ALERTS)
        assert "client_1" in alert_subs
        assert "client_2" in alert_subs

        metric_subs = manager.get_subscribers(SubscriptionChannel.METRICS)
        assert "client_1" in metric_subs
        assert "client_2" not in metric_subs

        # 取消订阅
        manager.unsubscribe("client_1", SubscriptionChannel.ALERTS)
        alert_subs = manager.get_subscribers(SubscriptionChannel.ALERTS)
        assert "client_1" not in alert_subs

    def test_websocket_message(self):
        """测试WebSocket消息"""
        from cyrp.communication.websocket_server import WebSocketMessage, MessageType

        # 创建消息
        msg = WebSocketMessage(
            type=MessageType.ALERT,
            payload={"alert_id": "ALT001", "severity": "critical"}
        )

        # 序列化
        json_str = msg.to_json()
        assert "alert" in json_str
        assert "ALT001" in json_str

        # 反序列化
        parsed = WebSocketMessage.from_json(json_str)
        assert parsed.type == MessageType.ALERT
        assert parsed.payload["alert_id"] == "ALT001"

    def test_message_handling(self):
        """测试消息处理"""
        import asyncio
        from cyrp.communication.websocket_server import (
            WebSocketConnectionManager, WebSocketMessage,
            MessageType, SubscriptionChannel
        )

        manager = WebSocketConnectionManager()
        manager.connect("client_1")

        async def test_handlers():
            # 测试订阅消息处理
            subscribe_msg = WebSocketMessage(
                type=MessageType.SUBSCRIBE,
                payload={"channel": "alerts"}
            )
            response = await manager.handle_message("client_1", subscribe_msg)
            assert response is not None
            assert response.type == MessageType.SUBSCRIBED
            assert response.payload["success"]

            # 验证订阅生效
            subs = manager.get_subscribers(SubscriptionChannel.ALERTS)
            assert "client_1" in subs

            # 测试心跳消息处理
            heartbeat_msg = WebSocketMessage(
                type=MessageType.HEARTBEAT,
                payload={}
            )
            response = await manager.handle_message("client_1", heartbeat_msg)
            assert response is not None
            assert response.type == MessageType.HEARTBEAT
            assert "server_time" in response.payload

        asyncio.run(test_handlers())

    def test_realtime_push_service(self):
        """测试实时推送服务"""
        import asyncio
        from cyrp.communication.websocket_server import (
            RealtimePushService, WebSocketConnectionManager, SubscriptionChannel
        )

        manager = WebSocketConnectionManager()
        service = RealtimePushService(manager)

        # 连接并订阅
        manager.connect("client_1")
        manager.subscribe("client_1", SubscriptionChannel.ALERTS)
        manager.subscribe("client_1", SubscriptionChannel.METRICS)

        async def test_push():
            # 推送告警
            count = await service.push_alert({
                "alert_id": "ALT001",
                "severity": "warning",
                "message": "Test alert"
            })
            assert count == 1

            # 推送指标
            count = await service.push_metric("flow_rate", 280.5)
            assert count == 1

            # 批量推送
            count = await service.push_metrics_batch({
                "flow_rate": 280.5,
                "pressure": 500000.0,
                "temperature": 18.5
            })
            assert count == 1

            # 检查消息队列
            client = manager.get_client("client_1")
            assert client.message_queue.qsize() == 3

        asyncio.run(test_push())

    def test_buffer_and_flush(self):
        """测试缓冲和刷新"""
        import asyncio
        from cyrp.communication.websocket_server import (
            RealtimePushService, WebSocketConnectionManager, SubscriptionChannel
        )

        manager = WebSocketConnectionManager()
        service = RealtimePushService(manager)

        manager.connect("client_1")
        manager.subscribe("client_1", SubscriptionChannel.ALERTS)
        manager.subscribe("client_1", SubscriptionChannel.METRICS)

        # 缓冲数据
        service.buffer_alert({"alert_id": "ALT001"})
        service.buffer_alert({"alert_id": "ALT002"})
        service.buffer_metric("flow_rate", 280.0)
        service.buffer_metric("pressure", 500000.0)

        async def test_flush():
            result = await service.flush_buffers()
            assert result["alerts"] == 2  # 2条告警
            assert result["metrics"] == 1  # 1次批量推送

        asyncio.run(test_flush())

    def test_stats(self):
        """测试统计信息"""
        from cyrp.communication.websocket_server import (
            WebSocketConnectionManager, SubscriptionChannel
        )

        manager = WebSocketConnectionManager()

        # 连接多个客户端
        manager.connect("client_1")
        manager.connect("client_2")
        manager.connect("client_3")

        manager.subscribe("client_1", SubscriptionChannel.ALERTS)
        manager.subscribe("client_2", SubscriptionChannel.ALERTS)
        manager.subscribe("client_2", SubscriptionChannel.METRICS)

        stats = manager.get_stats()
        assert stats["active_connections"] == 3
        assert stats["total_connections"] == 3
        assert stats["channels"]["alerts"] == 2
        assert stats["channels"]["metrics"] == 1

    def test_stale_connection_cleanup(self):
        """测试过期连接清理"""
        import time
        from cyrp.communication.websocket_server import WebSocketConnectionManager

        manager = WebSocketConnectionManager()

        # 连接客户端
        client = manager.connect("client_1")
        # 模拟心跳超时
        client.last_heartbeat = time.time() - 120  # 2分钟前

        # 清理过期连接（60秒超时）
        cleaned = manager.cleanup_stale_connections(timeout=60.0)
        assert "client_1" in cleaned
        assert manager.get_client("client_1") is None

    def test_create_realtime_push_system(self):
        """测试创建实时推送系统"""
        from cyrp.communication.websocket_server import create_realtime_push_system

        # 不带事件总线
        system = create_realtime_push_system()
        assert system["connection_manager"] is not None
        assert system["push_service"] is not None
        assert system["bridge"] is None

    def test_communication_module_import(self):
        """测试通信模块导入"""
        from cyrp.communication import (
            MessageType,
            SubscriptionChannel,
            WebSocketMessage,
            WebSocketClient,
            WebSocketConnectionManager,
            RealtimePushService,
            EventBusWebSocketBridge,
            create_realtime_push_system,
        )

        assert MessageType.ALERT.value == "alert"
        assert SubscriptionChannel.ALERTS.value == "alerts"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
