"""
Tests for HIL Test Framework.
在环测试框架测试
"""

import pytest
import numpy as np
from cyrp.hil import HILTestFramework, SimulationEngine
from cyrp.scenarios import ScenarioGenerator, ScenarioType


class TestHILFramework:
    """在环测试框架测试类"""

    def setup_method(self):
        """测试前设置"""
        self.hil = HILTestFramework()

    def test_initialization(self):
        """测试初始化"""
        assert self.hil.physical_system is not None
        assert self.hil.mas is not None
        assert self.hil.scenario_generator is not None

    def test_setup_teardown(self):
        """测试设置和清理"""
        self.hil.setup()
        assert len(self.hil.data_log) == 0

        self.hil.teardown()

    def test_run_scenario_test(self):
        """测试场景运行"""
        # 运行短时间测试
        generator = ScenarioGenerator()
        test_scenario = generator.generate_nominal_test(duration=10.0)

        result = self.hil.run_test(test_scenario)

        assert result.test_name == test_scenario.name
        assert result.duration > 0
        assert 'flow_tracking_error' in result.metrics

    def test_generate_report(self):
        """测试报告生成"""
        # 运行一个简单测试
        generator = ScenarioGenerator()
        test_scenario = generator.generate_nominal_test(duration=5.0)
        self.hil.run_test(test_scenario)

        report = self.hil.generate_report()

        assert "Test Report" in report
        assert test_scenario.name in report


class TestSimulationEngine:
    """仿真引擎测试类"""

    def setup_method(self):
        """测试前设置"""
        self.engine = SimulationEngine()

    def test_reset(self):
        """测试重置"""
        self.engine.reset(initial_flow=265.0)

        assert self.engine.sim_time == 0.0
        assert len(self.engine.history) == 0

    def test_step(self):
        """测试单步"""
        self.engine.reset()
        control = np.array([1.0, 1.0])

        state = self.engine.step(control)

        assert state is not None
        assert self.engine.sim_time > 0

    def test_run(self):
        """测试运行"""
        self.engine.reset()

        history = self.engine.run(
            duration=1.0,
            control_func=lambda t, s: np.array([1.0, 1.0])
        )

        assert len(history) > 0

    def test_inject_fault(self):
        """测试故障注入"""
        self.engine.reset()

        self.engine.inject_fault(
            'leakage',
            start_time=0.0,
            duration=10.0,
            parameters={'rate': 0.1}
        )

        # 验证故障已注入
        faults = self.engine.physical_system.fault_injector.fault_schedule
        assert 'leakage' in faults


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
