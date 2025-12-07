"""
Hierarchical Distributed MPC Controller for CYRP.
穿黄工程分层分布式模型预测控制器

架构:
- 上层: 全局优化层 (Global Optimization Layer) - 边缘服务器
- 下层: 局部执行层 (Local Execution Layer) - 现地PLC
- 调度核心: 场景状态机 (Scenario Supervisor)
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from enum import Enum

from cyrp.control.mpc_controller import (
    MPCController, LTVMPCController, NMPCController,
    RobustMPCController, MPCConfig, MPCResult
)
from cyrp.control.pid_controller import DualTunnelPIDController
from cyrp.control.safety_interlocks import SafetyInterlockSystem, InterlockAction
from cyrp.scenarios.scenario_definitions import ScenarioType, ScenarioDomain


class ControllerType(Enum):
    """控制器类型"""
    LTV_MPC = "ltv_mpc"
    NMPC = "nmpc"
    ROBUST_MPC = "robust_mpc"


@dataclass
class HDMPCConfig:
    """HD-MPC配置"""
    # 上层配置
    global_sample_time: float = 60.0  # 1分钟
    global_horizon: int = 24  # 预测24步 (24分钟)

    # 下层配置
    local_sample_time: float = 0.01  # 10ms
    local_pid_enabled: bool = True

    # 场景-模型映射
    scenario_model_map: Dict[ScenarioType, ControllerType] = field(default_factory=dict)


@dataclass
class HDMPCOutput:
    """HD-MPC输出"""
    timestamp: float = 0.0

    # 上层输出
    u_global: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0]))
    x_predicted: np.ndarray = field(default_factory=lambda: np.zeros((20, 4)))
    global_cost: float = 0.0

    # 下层输出
    u_local: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0]))
    pid_contribution: np.ndarray = field(default_factory=lambda: np.zeros(2))

    # 联锁状态
    interlock_triggered: bool = False
    interlock_actions: List[InterlockAction] = field(default_factory=list)

    # 场景信息
    active_scenario: str = "S1-A"
    active_controller: str = "LTV_MPC"


class HDMPCController:
    """
    分层分布式MPC控制器

    实现:
    1. 场景-模型动态映射
    2. 上层全局优化
    3. 下层局部执行
    4. 安全联锁集成
    5. 无扰切换
    """

    def __init__(self, config: Optional[HDMPCConfig] = None):
        """
        初始化HD-MPC

        Args:
            config: HD-MPC配置
        """
        self.config = config or HDMPCConfig()

        # 初始化模型库
        self._init_model_library()

        # 初始化场景-模型映射
        self._init_scenario_mapping()

        # 初始化下层控制器
        self.local_pid = DualTunnelPIDController()

        # 初始化安全联锁
        self.safety_system = SafetyInterlockSystem()

        # 当前状态
        self.current_scenario = ScenarioType.S1_A_DUAL_BALANCED
        self.current_controller_type = ControllerType.LTV_MPC
        self.current_controller = self.controllers[ControllerType.LTV_MPC]

        # 上一时刻输出
        self.u_prev_global = np.array([1.0, 1.0])
        self.u_prev_local = np.array([1.0, 1.0])

        # 时间管理
        self.last_global_update = 0.0

    def _init_model_library(self):
        """初始化模型库"""
        # LTV MPC (常态工况)
        ltv_config = MPCConfig(
            prediction_horizon=20,
            control_horizon=10,
            sample_time=self.config.global_sample_time,
            Q=np.diag([1.0, 1.0, 0.1, 0.1]),  # 流量权重高
            R=np.eye(2) * 0.01,
            Rd=np.eye(2) * 0.1
        )
        self.ltv_mpc = LTVMPCController(ltv_config)

        # NMPC (气液两相)
        nmpc_config = MPCConfig(
            prediction_horizon=15,
            control_horizon=8,
            sample_time=self.config.global_sample_time,
            Q=np.diag([0.5, 0.5, 1.0, 1.0]),  # 压力权重高
            R=np.eye(2) * 0.001
        )
        self.nmpc = NMPCController(nmpc_config)

        # Robust MPC (应急)
        robust_config = MPCConfig(
            prediction_horizon=10,
            control_horizon=5,
            sample_time=self.config.global_sample_time,
            Q=np.diag([0.1, 0.1, 10.0, 10.0]),  # 压力最重要
            R=np.eye(2) * 0.0001  # 允许激进控制
        )
        self.robust_mpc = RobustMPCController(robust_config)

        self.controllers = {
            ControllerType.LTV_MPC: self.ltv_mpc,
            ControllerType.NMPC: self.nmpc,
            ControllerType.ROBUST_MPC: self.robust_mpc
        }

    def _init_scenario_mapping(self):
        """初始化场景-模型映射"""
        self.scenario_model_map = {
            # 常态运行 -> LTV MPC
            ScenarioType.S1_A_DUAL_BALANCED: ControllerType.LTV_MPC,
            ScenarioType.S1_B_DYNAMIC_PEAK: ControllerType.LTV_MPC,
            ScenarioType.S2_A_SEDIMENT_FLUSH: ControllerType.LTV_MPC,
            ScenarioType.S2_B_MUSSEL_CONTROL: ControllerType.LTV_MPC,

            # 过渡态 -> NMPC
            ScenarioType.S3_A_FILLING: ControllerType.NMPC,
            ScenarioType.S3_B_DRAINING: ControllerType.NMPC,
            ScenarioType.S4_A_SWITCH_TUNNEL: ControllerType.LTV_MPC,
            ScenarioType.S4_B_ISOLATION: ControllerType.NMPC,

            # 应急 -> Robust MPC
            ScenarioType.S5_A_INNER_LEAK: ControllerType.ROBUST_MPC,
            ScenarioType.S5_B_OUTER_INTRUSION: ControllerType.ROBUST_MPC,
            ScenarioType.S5_C_JOINT_OFFSET: ControllerType.ROBUST_MPC,
            ScenarioType.S6_A_LIQUEFACTION: ControllerType.ROBUST_MPC,
            ScenarioType.S6_B_INTAKE_VORTEX: ControllerType.NMPC,
            ScenarioType.S7_A_PIPE_BURST: ControllerType.ROBUST_MPC,
            ScenarioType.S7_B_GATE_ASYNC: ControllerType.LTV_MPC,
        }

    def switch_scenario(
        self,
        new_scenario: ScenarioType,
        bumpless: bool = True
    ):
        """
        切换场景

        Args:
            new_scenario: 新场景
            bumpless: 是否无扰切换
        """
        if new_scenario == self.current_scenario:
            return

        old_controller = self.current_controller
        new_controller_type = self.scenario_model_map.get(
            new_scenario, ControllerType.LTV_MPC
        )
        new_controller = self.controllers[new_controller_type]

        # 状态继承 (用于热启动)
        if bumpless:
            new_controller.u_prev = old_controller.u_prev.copy()

        self.current_scenario = new_scenario
        self.current_controller_type = new_controller_type
        self.current_controller = new_controller

        # 应急模式特殊处理
        if new_controller_type == ControllerType.ROBUST_MPC:
            emergency_type = None
            if new_scenario in [ScenarioType.S5_A_INNER_LEAK, ScenarioType.S5_B_OUTER_INTRUSION]:
                emergency_type = 'leakage'
            elif new_scenario == ScenarioType.S6_A_LIQUEFACTION:
                emergency_type = 'liquefaction'
            self.robust_mpc.set_emergency_mode(True, emergency_type)
        else:
            self.robust_mpc.set_emergency_mode(False)

    def _compute_global(
        self,
        x0: np.ndarray,
        x_ref: np.ndarray
    ) -> MPCResult:
        """
        上层全局优化

        Args:
            x0: 当前状态
            x_ref: 参考轨迹

        Returns:
            MPC结果
        """
        result = self.current_controller.solve(x0, x_ref)
        return result

    def _compute_local(
        self,
        Q_sp: float,
        Q1: float,
        Q2: float,
        gate1: float,
        gate2: float,
        mpc_ff: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """
        下层局部控制

        Args:
            Q_sp: 流量设定值
            Q1, Q2: 实际流量
            gate1, gate2: 闸门位置
            mpc_ff: MPC前馈
            dt: 采样时间

        Returns:
            控制量
        """
        if self.config.local_pid_enabled:
            u1, u2 = self.local_pid.compute(
                Q_sp, Q1, Q2, gate1, gate2, mpc_ff, dt
            )
            return np.array([u1, u2])
        else:
            return mpc_ff

    def compute(
        self,
        state: Dict[str, Any],
        reference: Dict[str, Any],
        sensor_data: Dict[str, Any],
        current_time: float,
        dt: float
    ) -> HDMPCOutput:
        """
        计算HD-MPC输出

        Args:
            state: 系统状态
            reference: 参考目标
            sensor_data: 传感器数据
            current_time: 当前时间
            dt: 采样时间

        Returns:
            HD-MPC输出
        """
        output = HDMPCOutput(timestamp=current_time)

        # 提取状态
        Q1 = state.get('Q1', 132.5)
        Q2 = state.get('Q2', 132.5)
        H_in = state.get('H_inlet', 106.05)
        H_out = state.get('H_outlet', 104.79)
        gate1 = state.get('gate_1', 1.0)
        gate2 = state.get('gate_2', 1.0)

        x0 = np.array([Q1, Q2, H_in, H_out])

        # 提取参考
        Q_ref = reference.get('Q_total', 265.0)
        x_ref = np.array([Q_ref / 2, Q_ref / 2, H_in, H_out])

        # 1. 检查安全联锁
        interlock_triggered, interlock_actions = self.safety_system.check_all(
            sensor_data, current_time
        )
        output.interlock_triggered = interlock_triggered
        output.interlock_actions = interlock_actions

        # 2. 上层计算 (按周期)
        if current_time - self.last_global_update >= self.config.global_sample_time:
            mpc_result = self._compute_global(x0, x_ref)
            if mpc_result.success:
                self.u_prev_global = mpc_result.u_optimal
                output.x_predicted = mpc_result.x_predicted
                output.global_cost = mpc_result.cost
            self.last_global_update = current_time

        output.u_global = self.u_prev_global

        # 3. 下层计算
        u_local = self._compute_local(
            Q_ref, Q1, Q2, gate1, gate2,
            self.u_prev_global, dt
        )
        output.pid_contribution = u_local - self.u_prev_global

        # 4. 应用联锁
        if interlock_triggered:
            u_final = self.safety_system.apply_interlocks(u_local, interlock_actions)
        else:
            u_final = u_local

        # 限幅
        u_final = np.clip(u_final, 0.0, 1.0)

        output.u_local = u_final
        self.u_prev_local = u_final

        output.active_scenario = self.current_scenario.value
        output.active_controller = self.current_controller_type.value

        return output

    def get_status(self) -> Dict[str, Any]:
        """获取控制器状态"""
        return {
            'scenario': self.current_scenario.value,
            'controller': self.current_controller_type.value,
            'u_global': self.u_prev_global.tolist(),
            'u_local': self.u_prev_local.tolist(),
            'safety_status': self.safety_system.get_status()
        }

    def reset(self):
        """重置控制器"""
        self.u_prev_global = np.array([1.0, 1.0])
        self.u_prev_local = np.array([1.0, 1.0])
        self.last_global_update = 0.0

        for controller in self.controllers.values():
            controller.u_prev = np.array([1.0, 1.0])

        self.local_pid.reset()
        self.safety_system.reset_all()

        self.current_scenario = ScenarioType.S1_A_DUAL_BALANCED
        self.current_controller_type = ControllerType.LTV_MPC
        self.current_controller = self.controllers[ControllerType.LTV_MPC]
