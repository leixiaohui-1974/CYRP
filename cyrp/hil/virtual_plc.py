"""
Virtual PLC for CYRP HIL Testing.
穿黄工程虚拟PLC

模拟现地PLC的行为，用于在环测试
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np
from enum import Enum


class PLCMode(Enum):
    """PLC运行模式"""
    STOP = "stop"
    RUN = "run"
    PROGRAM = "program"
    FAULT = "fault"


@dataclass
class PLCRegisters:
    """PLC寄存器"""
    # 输入寄存器 (来自传感器)
    I_flow_1: float = 0.0
    I_flow_2: float = 0.0
    I_pressure_1: float = 0.0
    I_pressure_2: float = 0.0
    I_level_inlet: float = 0.0
    I_level_outlet: float = 0.0
    I_gate_pos_1: float = 0.0
    I_gate_pos_2: float = 0.0

    # 输出寄存器 (到执行器)
    O_gate_cmd_1: float = 0.0
    O_gate_cmd_2: float = 0.0
    O_valve_fill: float = 0.0
    O_valve_drain: float = 0.0
    O_pump_speed: float = 0.0

    # 控制字
    C_mode: int = 0
    C_enable: bool = True
    C_emergency: bool = False

    # 状态字
    S_running: bool = False
    S_fault: bool = False
    S_interlock: bool = False


class VirtualPLC:
    """
    虚拟PLC

    模拟Siemens S7-1500R或Schneider M580的行为
    实现:
    1. 扫描周期
    2. I/O处理
    3. 梯形图逻辑
    4. 安全联锁
    """

    def __init__(self, scan_time: float = 0.01):
        """
        初始化虚拟PLC

        Args:
            scan_time: 扫描周期 (s)
        """
        self.scan_time = scan_time
        self.mode = PLCMode.STOP

        # 寄存器
        self.registers = PLCRegisters()

        # 内部变量
        self.internal: Dict[str, Any] = {}

        # 定时器
        self.timers: Dict[str, float] = {}

        # 计数器
        self.counters: Dict[str, int] = {}

        # 报警
        self.alarms: List[str] = []

        # 上一扫描时间
        self.last_scan_time = 0.0

        # 梯形图程序
        self.programs: List[callable] = []

        # 安装默认程序
        self._install_default_programs()

    def _install_default_programs(self):
        """安装默认控制程序"""
        # 闸门控制程序
        self.programs.append(self._gate_control_program)
        # 安全联锁程序
        self.programs.append(self._safety_interlock_program)
        # 报警程序
        self.programs.append(self._alarm_program)

    def start(self):
        """启动PLC"""
        self.mode = PLCMode.RUN
        self.registers.S_running = True

    def stop(self):
        """停止PLC"""
        self.mode = PLCMode.STOP
        self.registers.S_running = False

    def scan(self, dt: float = None) -> Dict[str, float]:
        """
        执行扫描周期

        Args:
            dt: 时间步长

        Returns:
            输出寄存器
        """
        if self.mode != PLCMode.RUN:
            return self._get_outputs()

        dt = dt or self.scan_time

        # 1. 输入处理
        self._process_inputs()

        # 2. 执行程序
        for program in self.programs:
            try:
                program(dt)
            except Exception as e:
                self.alarms.append(f"Program error: {e}")
                self.registers.S_fault = True

        # 3. 输出处理
        self._process_outputs()

        self.last_scan_time += dt

        return self._get_outputs()

    def _process_inputs(self):
        """处理输入"""
        # 输入滤波 (一阶滤波)
        alpha = 0.1
        # 在实际应用中，这里会从I/O模块读取数据
        pass

    def _process_outputs(self):
        """处理输出"""
        # 输出限幅
        self.registers.O_gate_cmd_1 = np.clip(self.registers.O_gate_cmd_1, 0, 1)
        self.registers.O_gate_cmd_2 = np.clip(self.registers.O_gate_cmd_2, 0, 1)

    def _gate_control_program(self, dt: float):
        """闸门控制程序"""
        if not self.registers.C_enable:
            return

        if self.registers.C_emergency:
            # 紧急模式: 闸门保持
            return

        # PID控制逻辑由上位机完成
        # PLC仅执行位置环控制

        # 位置误差
        err_1 = self.registers.O_gate_cmd_1 - self.registers.I_gate_pos_1
        err_2 = self.registers.O_gate_cmd_2 - self.registers.I_gate_pos_2

        # 位置环增益
        Kp = 0.5

        # 更新执行器 (模拟)
        # 实际PLC中，这里会输出到变频器或伺服驱动器
        pass

    def _safety_interlock_program(self, dt: float):
        """安全联锁程序"""
        interlock_triggered = False

        # 防真空联锁
        if self.registers.I_pressure_1 < -5e4:
            self.alarms.append("Anti-vacuum interlock triggered")
            interlock_triggered = True

        # 防超压联锁
        if self.registers.I_pressure_1 > 1e6:
            self.alarms.append("Anti-overpressure interlock triggered")
            interlock_triggered = True

        # 防非同步联锁
        gate_diff = abs(self.registers.I_gate_pos_1 - self.registers.I_gate_pos_2)
        if gate_diff > 0.1:
            self.alarms.append("Anti-asymmetric interlock triggered")
            interlock_triggered = True

        self.registers.S_interlock = interlock_triggered

        if interlock_triggered:
            # 锁定输出
            # 实际实现中会根据联锁类型采取不同动作
            pass

    def _alarm_program(self, dt: float):
        """报警程序"""
        # 限制报警数量
        if len(self.alarms) > 100:
            self.alarms = self.alarms[-100:]

    def set_input(self, name: str, value: float):
        """设置输入寄存器"""
        if hasattr(self.registers, f'I_{name}'):
            setattr(self.registers, f'I_{name}', value)

    def set_output(self, name: str, value: float):
        """设置输出寄存器"""
        if hasattr(self.registers, f'O_{name}'):
            setattr(self.registers, f'O_{name}', value)

    def _get_outputs(self) -> Dict[str, float]:
        """获取输出寄存器"""
        return {
            'gate_cmd_1': self.registers.O_gate_cmd_1,
            'gate_cmd_2': self.registers.O_gate_cmd_2,
            'valve_fill': self.registers.O_valve_fill,
            'valve_drain': self.registers.O_valve_drain,
            'pump_speed': self.registers.O_pump_speed
        }

    def get_status(self) -> Dict[str, Any]:
        """获取PLC状态"""
        return {
            'mode': self.mode.value,
            'running': self.registers.S_running,
            'fault': self.registers.S_fault,
            'interlock': self.registers.S_interlock,
            'alarms': self.alarms[-10:]
        }

    def reset_alarms(self):
        """复位报警"""
        self.alarms = []
        self.registers.S_fault = False
