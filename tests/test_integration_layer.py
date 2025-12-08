"""
Integration Layer Tests
集成层测试 - 测试新模块与现有系统的集成

测试内容：
1. IntegratedDigitalTwin - 数字孪生集成
2. IntegratedSafetyAgent - 安全智能体集成
3. SystemIntegrator - 系统集成器
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class MockHydraulicState:
    """模拟水力状态"""
    def __init__(self):
        self.pressure = np.array([5e5, 4.8e5, 4.6e5, 4.5e5])
        self.flow_rate = 280.0
        self.velocity = 7.3
        self.water_level = 3.5


class MockStructuralState:
    """模拟结构状态"""
    def __init__(self):
        self.displacement = np.zeros(10)
        self.stress = np.ones(10) * 1e6


class MockSystemState:
    """模拟系统状态"""
    def __init__(self):
        self.hydraulic = MockHydraulicState()
        self.structural = MockStructuralState()


class TestIntegratedDigitalTwin:
    """集成数字孪生测试"""

    def test_initialization(self):
        """测试初始化"""
        from cyrp.integration.integrated_digital_twin import IntegratedDigitalTwin

        twin = IntegratedDigitalTwin(
            enable_prediction=True,
            enable_idz_update=True,
            enable_assimilation=True
        )

        assert twin.enable_prediction == True
        assert twin.enable_assimilation == True

    def test_sync(self):
        """测试状态同步"""
        from cyrp.integration.integrated_digital_twin import IntegratedDigitalTwin

        twin = IntegratedDigitalTwin()
        state = MockSystemState()

        twin.sync(state, current_time=1000.0)

        assert twin.twin_state.timestamp == 1000.0
        assert twin.twin_state.sync_status == "synced"
        assert len(twin.state_history) == 1

    def test_prediction(self):
        """测试状态预测"""
        from cyrp.integration.integrated_digital_twin import IntegratedDigitalTwin

        twin = IntegratedDigitalTwin(enable_prediction=True)
        state = MockSystemState()

        # 同步多次以积累历史数据
        for i in range(20):
            twin.sync(state, current_time=1000.0 + i * 10)

        # 执行预测
        predictions = twin.predict(horizon=600)

        # 验证预测
        assert isinstance(predictions, list)

    def test_what_if_analysis(self):
        """测试What-if分析"""
        from cyrp.integration.integrated_digital_twin import IntegratedDigitalTwin

        twin = IntegratedDigitalTwin()
        state = MockSystemState()

        # 积累历史数据
        for i in range(15):
            twin.sync(state, current_time=1000.0 + i * 10)

        # What-if分析
        result = twin.what_if_analysis(
            scenario_changes={'friction_factor': 0.03},
            horizon=600
        )

        assert 'scenario_changes' in result
        assert 'risk_assessment' in result

    def test_statistics(self):
        """测试统计信息"""
        from cyrp.integration.integrated_digital_twin import IntegratedDigitalTwin

        twin = IntegratedDigitalTwin()
        state = MockSystemState()

        for i in range(5):
            twin.sync(state, current_time=1000.0 + i * 10)

        stats = twin.get_statistics()
        assert stats['sync_count'] == 5
        assert stats['state_history_length'] == 5


class TestIntegratedSafetyAgent:
    """集成安全智能体测试"""

    def test_initialization(self):
        """测试初始化"""
        from cyrp.integration.integrated_safety import IntegratedSafetyAgent

        agent = IntegratedSafetyAgent()

        assert agent.interlock_system is not None
        assert agent.safety_limits is not None

    def test_perceive(self):
        """测试感知"""
        from cyrp.integration.integrated_safety import IntegratedSafetyAgent

        agent = IntegratedSafetyAgent()
        state = MockSystemState()

        environment = {
            'system_state': state,
            'sensor_data': {
                'pressure_max': 5e5,
                'pressure_min': 4.5e5,
                'flow': 280.0
            },
            'time': 1000.0
        }

        perception = agent.perceive(environment)

        assert 'safety_assessment' in perception
        assessment = perception['safety_assessment']
        assert 1 <= assessment.risk_level <= 5

    def test_decide(self):
        """测试决策"""
        from cyrp.integration.integrated_safety import IntegratedSafetyAgent, SafetyAssessment

        agent = IntegratedSafetyAgent()

        # 正常情况
        perception = {
            'safety_assessment': SafetyAssessment(risk_level=1),
            'interlock_triggered': [],
            'interlock_actions': []
        }

        decision = agent.decide(perception)
        assert decision['action'] == 'monitor'

        # 高风险情况
        perception['safety_assessment'] = SafetyAssessment(risk_level=5)
        decision = agent.decide(perception)
        assert decision['action'] == 'emergency_stop'

    def test_safety_report(self):
        """测试安全报告"""
        from cyrp.integration.integrated_safety import IntegratedSafetyAgent

        agent = IntegratedSafetyAgent()
        state = MockSystemState()

        environment = {
            'system_state': state,
            'sensor_data': {'pressure_max': 5e5},
            'time': 1000.0
        }

        # 执行感知以生成评估
        agent.perceive(environment)

        report = agent.get_safety_report()
        assert 'risk_level' in report
        assert 'safety_margin' in report


class TestSystemIntegrator:
    """系统集成器测试"""

    def test_initialization(self):
        """测试初始化"""
        from cyrp.integration.system_integrator import SystemIntegrator

        integrator = SystemIntegrator(
            tunnel_length=4250.0,
            enable_simulation=True,
            enable_governance=True,
            enable_assimilation=True,
            enable_idz_update=True,
            enable_evaluation=True,
            enable_prediction=True
        )

        assert integrator.governance is not None
        assert integrator.kalman_filter is not None

    def test_process(self):
        """测试处理流程"""
        from cyrp.integration.system_integrator import SystemIntegrator

        integrator = SystemIntegrator()
        state = MockSystemState()

        result = integrator.process(
            physical_state=state,
            control_commands={'valve_1': 0.8},
            current_time=1000.0
        )

        assert result is not None
        assert result.timestamp == 1000.0
        assert isinstance(result.sensor_readings, dict)
        assert isinstance(result.actuator_states, dict)
        assert 0 <= result.data_quality_score <= 1.0

    def test_multiple_process(self):
        """测试多次处理"""
        from cyrp.integration.system_integrator import SystemIntegrator

        integrator = SystemIntegrator()
        state = MockSystemState()

        # 处理多次
        for i in range(30):
            result = integrator.process(state, current_time=1000.0 + i * 10)

        # 验证历史
        assert len(integrator.state_history) == 30

        # 验证统计
        stats = integrator.get_statistics()
        assert stats['process_count'] == 30

    def test_model_parameters(self):
        """测试模型参数获取"""
        from cyrp.integration.system_integrator import SystemIntegrator

        integrator = SystemIntegrator()
        params = integrator.get_model_parameters()

        assert 'friction_factor' in params
        assert 'wave_speed' in params

    def test_reset(self):
        """测试重置"""
        from cyrp.integration.system_integrator import SystemIntegrator

        integrator = SystemIntegrator()
        state = MockSystemState()

        # 处理一些数据
        for i in range(10):
            integrator.process(state, current_time=1000.0 + i * 10)

        # 重置
        integrator.reset()

        assert len(integrator.state_history) == 0
        assert integrator.stats['process_count'] == 0


class TestEndToEndIntegration:
    """端到端集成测试"""

    def test_full_pipeline(self):
        """测试完整流程"""
        from cyrp.integration.system_integrator import SystemIntegrator
        from cyrp.integration.integrated_safety import IntegratedSafetyAgent
        from cyrp.integration.integrated_digital_twin import IntegratedDigitalTwin

        # 初始化组件
        integrator = SystemIntegrator()
        safety_agent = IntegratedSafetyAgent()
        digital_twin = IntegratedDigitalTwin()

        state = MockSystemState()

        # 模拟运行周期
        for i in range(50):
            current_time = 1000.0 + i * 10

            # 1. 系统集成处理
            integrated_state = integrator.process(
                physical_state=state,
                current_time=current_time
            )

            # 2. 安全评估
            safety_perception = safety_agent.perceive({
                'system_state': state,
                'sensor_data': integrated_state.sensor_readings,
                'time': current_time
            })

            # 3. 数字孪生同步
            digital_twin.sync(state, current_time)

        # 验证最终状态
        assert integrator.stats['process_count'] == 50
        assert digital_twin.stats['sync_count'] == 50
        assert len(safety_agent.assessment_history) == 50

        # 执行预测
        predictions = digital_twin.predict(horizon=300)
        assert isinstance(predictions, list)

    def test_abnormal_scenario(self):
        """测试异常场景"""
        from cyrp.integration.integrated_safety import IntegratedSafetyAgent

        agent = IntegratedSafetyAgent()

        # 创建高压异常状态
        state = MockSystemState()
        state.hydraulic.pressure = np.array([1.2e6, 1.1e6, 1.0e6])  # 超压

        environment = {
            'system_state': state,
            'sensor_data': {
                'pressure_max': 1.2e6,
                'pressure_min': 1.0e6,
            },
            'time': 1000.0
        }

        perception = agent.perceive(environment)
        assessment = perception['safety_assessment']

        # 验证检测到风险
        assert assessment.risk_level >= 3
        assert len(assessment.boundary_violations) > 0


if __name__ == '__main__':
    # 运行测试
    print("=" * 60)
    print("Integration Layer Tests")
    print("=" * 60)

    test_classes = [
        TestIntegratedDigitalTwin,
        TestIntegratedSafetyAgent,
        TestSystemIntegrator,
        TestEndToEndIntegration,
    ]

    all_passed = True

    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        instance = test_class()

        for method_name in dir(instance):
            if method_name.startswith('test_'):
                try:
                    getattr(instance, method_name)()
                    print(f"  {method_name}: PASSED")
                except Exception as e:
                    print(f"  {method_name}: FAILED - {str(e)[:50]}")
                    all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL INTEGRATION TESTS PASSED!")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
