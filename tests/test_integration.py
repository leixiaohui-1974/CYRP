"""
Integration Tests for CYRP Core Modules.
穿黄工程核心模块集成测试
"""

import pytest
import numpy as np
from datetime import datetime

from cyrp import (
    PhysicalSystem,
    PerceptionSystem,
    HDMPCController,
    MultiAgentSystem,
    HILTestFramework,
    DigitalTwin,
    ScenarioManager,
)
from cyrp.core import TunnelParameters, HydraulicState


class TestPhysicalSystemIntegration:
    """物理系统集成测试"""

    def setup_method(self):
        """测试前设置"""
        self.system = PhysicalSystem()

    def test_system_initialization(self):
        """测试系统初始化"""
        self.system.reset(initial_flow=265.0)

        state = self.system.state
        assert state is not None
        assert state.hydraulic is not None

    def test_system_step(self):
        """测试系统单步执行"""
        self.system.reset(initial_flow=265.0)

        # 创建控制命令
        from cyrp.core.physical_system import ControlCommand
        cmd = ControlCommand(
            gate_inlet_1_target=0.8,
            gate_inlet_2_target=0.8
        )

        # 执行仿真步
        self.system.step(cmd, dt=0.1)

        # 验证状态更新
        state = self.system.state
        assert state is not None

    def test_multiple_steps(self):
        """测试多步仿真"""
        self.system.reset(initial_flow=265.0)

        from cyrp.core.physical_system import ControlCommand
        cmd = ControlCommand(
            gate_inlet_1_target=0.8,
            gate_inlet_2_target=0.8
        )

        # 执行多步仿真
        for _ in range(100):
            self.system.step(cmd, dt=0.1)

        # 系统应该稳定
        state = self.system.state
        assert state is not None


class TestMultiAgentSystemIntegration:
    """多智能体系统集成测试"""

    def setup_method(self):
        """测试前设置"""
        self.mas = MultiAgentSystem()

    def test_agent_initialization(self):
        """测试智能体初始化"""
        assert self.mas is not None

        # 检查所有智能体已创建
        status = self.mas.get_status()
        assert status is not None

    def test_agent_communication(self):
        """测试智能体间通信"""
        # 获取场景分类结果
        scenario = self.mas.get_scenario()
        assert scenario is not None

    def test_agent_step(self):
        """测试智能体执行步"""
        # 准备输入数据
        env = {
            'system_state': None,
            'sensor_data': {
                'pressure_max': 800.0,
                'pressure_min': 100.0,
                'gate_positions': [0.8, 0.8]
            },
            'time': 0.0,
            'dt': 0.1
        }

        # 执行步
        result = self.mas.step(env)

        # 验证输出
        assert result is not None

    def test_risk_assessment(self):
        """测试风险评估"""
        risk_level = self.mas.get_risk_level()

        # 风险级别应该是字符串或数值
        assert risk_level is not None


class TestHDMPCControllerIntegration:
    """分层分布式MPC控制器集成测试"""

    def setup_method(self):
        """测试前设置"""
        self.controller = HDMPCController()

    def test_controller_initialization(self):
        """测试控制器初始化"""
        assert self.controller is not None

    def test_control_computation(self):
        """测试控制计算"""
        # 准备状态
        state = {
            'flow_1': 132.5,
            'flow_2': 132.5,
            'pressure_inlet': 500.0,
            'pressure_outlet': 100.0,
        }
        reference = {
            'total_flow': 265.0,
            'pressure_target': 500.0
        }
        sensor_data = {
            'pressure_max': 800.0,
            'pressure_min': 100.0,
        }

        # 计算控制
        control = self.controller.compute(state, reference, sensor_data, current_time=0.0, dt=0.1)

        # 控制输出应该是数组或字典
        assert control is not None


class TestScenarioManagerIntegration:
    """场景管理器集成测试"""

    def test_scenario_loading(self):
        """测试场景加载"""
        manager = ScenarioManager()

        # 应该有场景定义
        assert manager is not None

    def test_scenario_transition(self):
        """测试场景切换"""
        manager = ScenarioManager()

        # 获取当前场景
        current = manager.current_scenario
        assert current is not None


class TestHILFrameworkIntegration:
    """HIL测试框架集成测试"""

    def test_framework_initialization(self):
        """测试框架初始化"""
        hil = HILTestFramework()

        assert hil is not None

    def test_setup_teardown(self):
        """测试设置和清理"""
        hil = HILTestFramework()

        # 设置测试环境
        hil.setup()

        # 清理测试环境
        hil.teardown()


class TestDigitalTwinIntegration:
    """数字孪生集成测试"""

    def test_twin_initialization(self):
        """测试数字孪生初始化"""
        twin = DigitalTwin()

        assert twin is not None

    def test_twin_sync(self):
        """测试数字孪生同步"""
        twin = DigitalTwin()
        physical = PhysicalSystem()
        physical.reset(initial_flow=265.0)

        # 同步物理系统状态
        twin.sync(physical.state)

        # 验证同步
        state = twin.get_state()
        assert state is not None


class TestEndToEndIntegration:
    """端到端集成测试"""

    def test_full_simulation_loop(self):
        """测试完整仿真循环"""
        # 初始化各子系统
        physical = PhysicalSystem()
        mas = MultiAgentSystem()

        # 初始化物理系统
        physical.reset(initial_flow=265.0)

        # 运行短时间仿真
        for t in range(10):
            # 准备智能体输入
            env = {
                'system_state': physical.state,
                'sensor_data': {
                    'pressure_max': 800.0,
                    'pressure_min': 100.0,
                    'gate_positions': [0.8, 0.8]
                },
                'time': t * 0.1,
                'dt': 0.1
            }

            # 执行智能体决策
            mas.step(env)

            # 获取控制输出
            control = mas.get_control_output()

            # 应用控制到物理系统
            from cyrp.core.physical_system import ControlCommand
            cmd = ControlCommand(
                gate_inlet_1_target=control.get('gate_1', 0.8),
                gate_inlet_2_target=control.get('gate_2', 0.8)
            )
            physical.step(cmd, dt=0.1)

        # 验证系统状态
        assert physical.state is not None
        assert mas.is_safe()


class TestPerformanceBenchmark:
    """性能基准测试"""

    def test_simulation_speed(self):
        """测试仿真速度"""
        import time

        physical = PhysicalSystem()
        physical.reset(initial_flow=265.0)

        from cyrp.core.physical_system import ControlCommand
        cmd = ControlCommand(
            gate_inlet_1_target=0.8,
            gate_inlet_2_target=0.8
        )

        # 测量1000步仿真时间
        start = time.time()
        for _ in range(1000):
            physical.step(cmd, dt=0.01)
        elapsed = time.time() - start

        # 仿真应该快于实时 (10s仿真时间)
        assert elapsed < 10.0, f"Simulation too slow: {elapsed}s for 10s simulated time"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
