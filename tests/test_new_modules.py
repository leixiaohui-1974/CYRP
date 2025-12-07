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

        # 读数应该接近真实值（考虑噪声和精度）
        assert 400000 < reading < 600000

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
