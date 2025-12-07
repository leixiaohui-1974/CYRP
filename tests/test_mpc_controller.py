"""
Tests for MPC Controller.
MPC控制器测试
"""

import pytest
import numpy as np
from cyrp.control.mpc_controller import (
    LTVMPCController, NMPCController, RobustMPCController,
    MPCConfig, MPCResult
)


class TestLTVMPC:
    """LTV MPC测试类"""

    def setup_method(self):
        """测试前设置"""
        config = MPCConfig(
            prediction_horizon=10,
            control_horizon=5,
            sample_time=1.0
        )
        self.controller = LTVMPCController(config)

    def test_initialization(self):
        """测试初始化"""
        assert self.controller.Np == 10
        assert self.controller.Nc == 5
        assert self.controller.nx == 4
        assert self.controller.nu == 2

    def test_linearize(self):
        """测试线性化"""
        x_op = np.array([132.5, 132.5, 106.05, 104.79])
        u_op = np.array([1.0, 1.0])

        self.controller.linearize(x_op, u_op)

        assert self.controller.A.shape == (4, 4)
        assert self.controller.B.shape == (4, 2)

    def test_solve(self):
        """测试求解"""
        x0 = np.array([132.5, 132.5, 106.05, 104.79])
        x_ref = np.array([132.5, 132.5, 106.05, 104.79])

        result = self.controller.solve(x0, x_ref)

        assert isinstance(result, MPCResult)
        assert len(result.u_optimal) == 2
        assert 0 <= result.u_optimal[0] <= 1
        assert 0 <= result.u_optimal[1] <= 1

    def test_get_control(self):
        """测试获取控制"""
        x0 = np.array([130.0, 130.0, 106.0, 105.0])
        x_ref = np.array([132.5, 132.5, 106.05, 104.79])

        control = self.controller.get_control(x0, x_ref)

        assert len(control) == 2
        assert all(0 <= u <= 1 for u in control)


class TestRobustMPC:
    """鲁棒MPC测试类"""

    def setup_method(self):
        """测试前设置"""
        self.controller = RobustMPCController()

    def test_emergency_mode(self):
        """测试应急模式"""
        self.controller.set_emergency_mode(True, 'leakage')

        assert self.controller.emergency_mode
        assert self.controller.emergency_type == 'leakage'

    def test_solve_emergency(self):
        """测试应急求解"""
        self.controller.set_emergency_mode(True, 'leakage')

        x0 = np.array([100.0, 132.5, 106.0, 105.0])
        x_ref = np.array([132.5, 132.5, 106.05, 104.79])

        result = self.controller.solve(x0, x_ref)

        assert isinstance(result, MPCResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
