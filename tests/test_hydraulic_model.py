"""
Tests for Hydraulic Model.
水力学模型测试
"""

import pytest
import numpy as np
from cyrp.core.hydraulic_model import HydraulicModel, HydraulicState, FlowRegime


class TestHydraulicModel:
    """水力学模型测试类"""

    def setup_method(self):
        """测试前设置"""
        self.model = HydraulicModel(
            length=4250.0,
            diameter=7.0,
            manning_n=0.014
        )

    def test_initialization(self):
        """测试初始化"""
        assert self.model.L == 4250.0
        assert self.model.D == 7.0
        assert abs(self.model.A - 38.48) < 0.1

    def test_friction_loss(self):
        """测试摩擦损失计算"""
        Q = 132.5  # m³/s
        h_f = self.model.compute_friction_loss(Q)
        assert h_f > 0
        assert h_f < 10  # 合理范围

    def test_local_loss(self):
        """测试局部损失计算"""
        Q = 132.5
        h_j = self.model.compute_local_loss(Q)
        assert h_j > 0
        assert h_j < 1

    def test_velocity(self):
        """测试流速计算"""
        Q = 132.5
        v = self.model.compute_velocity(Q)
        assert abs(v - 3.44) < 0.1  # 设计流速

    def test_pressure(self):
        """测试压力计算"""
        H = 106.0  # m
        z = 85.0  # m
        P = self.model.compute_pressure(H, z)
        expected = 1000 * 9.81 * (H - z)
        assert abs(P - expected) < 1000

    def test_steady_state(self):
        """测试稳态计算"""
        H_up = 106.05
        H_down = 104.79
        state = self.model.get_steady_state(H_up, H_down)

        assert state.Q1 > 0
        assert state.Q2 > 0
        assert abs(state.Q1 - state.Q2) < 1  # 双洞平衡

    def test_step(self):
        """测试单步仿真"""
        initial_state = HydraulicState(Q1=132.5, Q2=132.5)
        control = np.array([1.0, 1.0])

        new_state = self.model.step(initial_state, control, dt=0.1)

        assert new_state.time == 0.1
        assert new_state.Q1 > 0
        assert new_state.Q2 > 0


class TestHydraulicState:
    """水力状态测试类"""

    def test_total_flow(self):
        """测试总流量"""
        state = HydraulicState(Q1=132.5, Q2=132.5)
        assert state.total_flow == 265.0

    def test_flow_imbalance(self):
        """测试流量不平衡度"""
        state = HydraulicState(Q1=150, Q2=100)
        imbalance = state.flow_imbalance
        assert imbalance == 50 / 250

    def test_to_vector(self):
        """测试向量转换"""
        state = HydraulicState(Q1=132.5, Q2=132.5)
        vec = state.to_vector()
        assert len(vec) == 10
        assert vec[0] == 132.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
