"""
Simulation Engine for CYRP HIL Testing.
穿黄工程仿真引擎
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
import numpy as np
import time
from threading import Thread, Event

from cyrp.core import PhysicalSystem
from cyrp.core.physical_system import ControlCommand, SystemState


@dataclass
class SimulationConfig:
    """仿真配置"""
    dt: float = 0.1  # 时间步长 (s)
    realtime: bool = False  # 是否实时运行
    realtime_factor: float = 1.0  # 实时因子
    max_duration: float = 86400.0  # 最大仿真时长 (s)
    log_interval: float = 1.0  # 日志间隔 (s)


class SimulationEngine:
    """
    仿真引擎

    提供物理系统的高保真仿真能力
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        """
        初始化仿真引擎

        Args:
            config: 仿真配置
        """
        self.config = config or SimulationConfig()

        # 物理系统
        self.physical_system = PhysicalSystem()

        # 仿真状态
        self.sim_time = 0.0
        self.running = False
        self.paused = False

        # 控制输入回调
        self.control_callback: Optional[Callable] = None

        # 数据记录
        self.history: List[Dict] = []

        # 事件
        self.stop_event = Event()

        # 仿真线程
        self._sim_thread: Optional[Thread] = None

    def reset(self, initial_flow: float = 265.0):
        """重置仿真"""
        self.physical_system.reset(initial_flow=initial_flow)
        self.sim_time = 0.0
        self.history = []
        self.stop_event.clear()

    def step(self, control: Optional[np.ndarray] = None) -> SystemState:
        """
        单步仿真

        Args:
            control: 控制输入 [gate_1, gate_2]

        Returns:
            系统状态
        """
        if control is None:
            control = np.array([1.0, 1.0])

        cmd = ControlCommand(
            gate_inlet_1_target=control[0],
            gate_inlet_2_target=control[1]
        )

        state = self.physical_system.step(cmd, self.config.dt)
        self.sim_time += self.config.dt

        # 记录历史
        self._log_state(state)

        return state

    def _log_state(self, state: SystemState):
        """记录状态"""
        if len(self.history) == 0 or \
           self.sim_time - self.history[-1]['time'] >= self.config.log_interval:
            self.history.append({
                'time': self.sim_time,
                'Q1': state.hydraulic.Q1,
                'Q2': state.hydraulic.Q2,
                'H_inlet': state.hydraulic.H_inlet,
                'H_outlet': state.hydraulic.H_outlet,
                'gate_1': state.actuators.gate_inlet_1,
                'gate_2': state.actuators.gate_inlet_2,
                'mode': state.mode.value,
                'alarms': list(state.alarms.keys()) if state.alarms else []
            })

    def run(
        self,
        duration: float,
        control_func: Optional[Callable] = None
    ) -> List[Dict]:
        """
        运行仿真

        Args:
            duration: 仿真时长 (s)
            control_func: 控制函数 f(t, state) -> control

        Returns:
            历史记录
        """
        self.running = True
        end_time = self.sim_time + duration

        while self.sim_time < end_time and self.running:
            if self.paused:
                time.sleep(0.1)
                continue

            # 获取控制
            if control_func:
                control = control_func(self.sim_time, self.physical_system.state)
            else:
                control = np.array([1.0, 1.0])

            # 步进
            self.step(control)

            # 实时模式
            if self.config.realtime:
                time.sleep(self.config.dt / self.config.realtime_factor)

        self.running = False
        return self.history

    def run_async(
        self,
        duration: float,
        control_func: Optional[Callable] = None
    ):
        """
        异步运行仿真

        Args:
            duration: 仿真时长 (s)
            control_func: 控制函数
        """
        def _run():
            self.run(duration, control_func)

        self._sim_thread = Thread(target=_run)
        self._sim_thread.start()

    def stop(self):
        """停止仿真"""
        self.running = False
        self.stop_event.set()
        if self._sim_thread:
            self._sim_thread.join(timeout=2.0)

    def pause(self):
        """暂停仿真"""
        self.paused = True

    def resume(self):
        """恢复仿真"""
        self.paused = False

    def get_state(self) -> SystemState:
        """获取当前状态"""
        return self.physical_system.state

    def inject_fault(
        self,
        fault_type: str,
        start_time: float,
        duration: float,
        parameters: Dict[str, Any]
    ):
        """注入故障"""
        self.physical_system.fault_injector.schedule_fault(
            fault_type, start_time, duration, parameters
        )

    def get_history_array(self) -> Dict[str, np.ndarray]:
        """获取历史数据数组"""
        if not self.history:
            return {}

        return {
            'time': np.array([h['time'] for h in self.history]),
            'Q1': np.array([h['Q1'] for h in self.history]),
            'Q2': np.array([h['Q2'] for h in self.history]),
            'H_inlet': np.array([h['H_inlet'] for h in self.history]),
            'H_outlet': np.array([h['H_outlet'] for h in self.history]),
            'gate_1': np.array([h['gate_1'] for h in self.history]),
            'gate_2': np.array([h['gate_2'] for h in self.history])
        }
