"""
全场景HIL集成测试系统 - Full-Scenario HIL Integration Test System

实现完整的闭环仿真测试、全自主运行验证
Implements complete closed-loop simulation testing and fully autonomous operation verification
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from collections import deque

# 导入各模块
from cyrp.core.high_fidelity_model import CoupledPhysicalModel, SimulationState
from cyrp.hil.sensor_models import InstrumentationSystem, SensorStatus
from cyrp.hil.actuator_models import ActuatorSystem, ActuatorStatus
from cyrp.perception.advanced_classifier import AdvancedScenarioClassifier
from cyrp.control.adaptive_mpc import ScenarioAdaptiveMPC, AdaptiveMPCController
from cyrp.control.adaptive_pid import DualTunnelAdaptivePID, AdaptivePIDController
from cyrp.scenarios.enhanced_scenario_generator import EnhancedScenarioGenerator


class HILMode(Enum):
    """HIL运行模式"""
    REALTIME = "realtime"           # 实时仿真
    ACCELERATED = "accelerated"     # 加速仿真
    STEP_BY_STEP = "step_by_step"  # 单步调试


class TestResult(Enum):
    """测试结果"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    RUNNING = "running"
    NOT_STARTED = "not_started"


@dataclass
class PerformanceMetrics:
    """性能指标"""
    # 控制性能
    flow_tracking_error: float = 0.0     # 流量跟踪误差 (m³/s)
    pressure_stability: float = 0.0       # 压力稳定性 (%)
    settling_time: float = 0.0            # 调节时间 (s)
    overshoot: float = 0.0                # 超调量 (%)

    # 响应性能
    scenario_detection_time: float = 0.0  # 场景检测时间 (s)
    response_time: float = 0.0            # 响应时间 (s)
    recovery_time: float = 0.0            # 恢复时间 (s)

    # 安全性能
    constraint_violations: int = 0        # 约束违反次数
    emergency_stops: int = 0              # 紧急停车次数
    safety_margin: float = 0.0            # 安全裕度

    # 系统性能
    control_loop_time: float = 0.0        # 控制循环时间 (ms)
    sensor_availability: float = 0.0      # 传感器可用率 (%)
    actuator_health: float = 0.0          # 执行器健康度 (%)


@dataclass
class HILTestCase:
    """HIL测试用例"""
    id: str
    name: str
    description: str
    scenario_sequence: List[Dict[str, Any]]
    duration: float
    success_criteria: Dict[str, float]
    fault_injections: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class HILTestResult:
    """HIL测试结果"""
    test_case_id: str
    result: TestResult
    metrics: PerformanceMetrics
    timeline: List[Dict[str, Any]]
    violations: List[str]
    duration: float
    timestamp: float


class ClosedLoopController:
    """闭环控制器 - 集成MPC和PID"""

    def __init__(self):
        # 自适应MPC (用于优化)
        self.mpc = ScenarioAdaptiveMPC()

        # 自适应PID (用于执行)
        self.pid = DualTunnelAdaptivePID()

        # 场景分类器
        self.classifier = AdvancedScenarioClassifier()

        # 控制模式
        self.mode = 'hybrid'  # 'mpc', 'pid', 'hybrid'
        self.current_scenario = 'S2-A'

        # 设定值
        self.flow_setpoint = 265.0
        self.pressure_setpoint = 0.5e6

        # 状态
        self.is_emergency = False

    def compute(self, sensor_data: Dict[str, float],
               signals: Optional[Dict[str, np.ndarray]] = None,
               dt: float = 0.1) -> Dict[str, float]:
        """
        计算控制输出

        Args:
            sensor_data: 传感器数据
            signals: 信号历史 (用于场景识别)
            dt: 时间步长

        Returns:
            执行器指令
        """
        # 1. 场景识别
        classification = self.classifier.classify(sensor_data, signals)
        new_scenario = classification.scenario_id

        # 场景切换
        if new_scenario != self.current_scenario:
            self._handle_scenario_transition(new_scenario)
            self.current_scenario = new_scenario

        # 2. 紧急检测
        if classification.is_anomaly and classification.anomaly_score > 3.0:
            self.is_emergency = True

        # 3. 控制计算
        commands = {}

        if self.is_emergency:
            commands = self._emergency_control(sensor_data)
        elif self.mode == 'mpc':
            commands = self._mpc_control(sensor_data, dt)
        elif self.mode == 'pid':
            commands = self._pid_control(sensor_data, dt)
        else:  # hybrid
            commands = self._hybrid_control(sensor_data, dt)

        return commands

    def _handle_scenario_transition(self, new_scenario: str):
        """处理场景切换"""
        self.mpc.set_scenario(new_scenario)

        # 调整PID模式
        if new_scenario.startswith('S5') or new_scenario.startswith('S6'):
            self.pid.set_all_modes('fuzzy')
        else:
            self.pid.set_all_modes('standard')

    def _emergency_control(self, sensor_data: Dict[str, float]) -> Dict[str, float]:
        """紧急控制"""
        return {
            'north_inlet': 0.0,      # 关闭进口
            'north_outlet': 1.0,     # 保持出口开
            'south_inlet': 1.0,      # 切换到备用
            'south_outlet': 1.0,
            'control_valve': 0.5,
            'emergency_shutoff': 0.0
        }

    def _mpc_control(self, sensor_data: Dict[str, float], dt: float) -> Dict[str, float]:
        """MPC控制"""
        flow = sensor_data.get('flow_rate', 0)

        # MPC计算
        y_ref = np.array([self.flow_setpoint])
        y_meas = np.array([flow])
        u = self.mpc.compute_control(y_ref, y_meas)

        # 转换为阀门指令
        valve_position = np.clip(0.5 + u[0] / 100, 0, 1)

        return {
            'north_inlet': 1.0,
            'north_outlet': 1.0,
            'south_inlet': 1.0,
            'south_outlet': 1.0,
            'control_valve': float(valve_position),
            'emergency_shutoff': 1.0
        }

    def _pid_control(self, sensor_data: Dict[str, float], dt: float) -> Dict[str, float]:
        """PID控制"""
        north_flow = sensor_data.get('north_flow', 132.5)
        south_flow = sensor_data.get('south_flow', 132.5)

        # 双洞PID
        north_out, south_out = self.pid.compute(
            self.flow_setpoint, north_flow, south_flow, dt
        )

        # 转换为阀门增量
        north_position = np.clip(0.5 + north_out / 100, 0, 1)
        south_position = np.clip(0.5 + south_out / 100, 0, 1)

        return {
            'north_inlet': float(north_position),
            'north_outlet': 1.0,
            'south_inlet': float(south_position),
            'south_outlet': 1.0,
            'control_valve': 0.8,
            'emergency_shutoff': 1.0
        }

    def _hybrid_control(self, sensor_data: Dict[str, float], dt: float) -> Dict[str, float]:
        """混合控制 - MPC优化 + PID执行"""
        # MPC提供设定值优化
        flow = sensor_data.get('flow_rate', 0)
        y_ref = np.array([self.flow_setpoint])
        y_meas = np.array([flow])
        u_mpc = self.mpc.compute_control(y_ref, y_meas)

        # 调整PID设定值
        optimized_setpoint = self.flow_setpoint + u_mpc[0]

        # PID执行
        north_flow = sensor_data.get('north_flow', flow / 2)
        south_flow = sensor_data.get('south_flow', flow / 2)

        north_out, south_out = self.pid.compute(
            optimized_setpoint, north_flow, south_flow, dt
        )

        return {
            'north_inlet': float(np.clip(0.8 + north_out / 200, 0, 1)),
            'north_outlet': 1.0,
            'south_inlet': float(np.clip(0.8 + south_out / 200, 0, 1)),
            'south_outlet': 1.0,
            'control_valve': 0.8,
            'emergency_shutoff': 1.0
        }

    def set_setpoints(self, flow: float, pressure: float):
        """设置设定值"""
        self.flow_setpoint = flow
        self.pressure_setpoint = pressure

    def reset_emergency(self):
        """复位紧急状态"""
        self.is_emergency = False


class FullHILSystem:
    """完整HIL系统"""

    def __init__(self, n_nodes: int = 100):
        # 物理模型
        self.physical_model = CoupledPhysicalModel(n_nodes)

        # 仪表系统
        self.instrumentation = InstrumentationSystem()

        # 执行器系统
        self.actuators = ActuatorSystem()

        # 控制器
        self.controller = ClosedLoopController()

        # 场景生成器
        self.scenario_generator = EnhancedScenarioGenerator()

        # 运行状态
        self.time = 0.0
        self.dt = 0.1
        self.mode = HILMode.ACCELERATED
        self.is_running = False

        # 数据记录
        self.history: List[Dict[str, Any]] = []
        self.max_history = 10000

        # 性能统计
        self.metrics = PerformanceMetrics()
        self.flow_errors = deque(maxlen=1000)
        self.pressure_readings = deque(maxlen=1000)

    def initialize(self, initial_flow: float = 265.0, initial_temp: float = 288.15):
        """初始化系统"""
        self.physical_model.reset(initial_flow, initial_temp)
        self.time = 0.0
        self.history.clear()
        self.flow_errors.clear()
        self.pressure_readings.clear()
        self.metrics = PerformanceMetrics()

        # 初始化执行器到运行位置
        for act_id in self.actuators.actuators:
            self.actuators.actuators[act_id].position = 1.0

    def step(self, external_disturbance: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        执行一步仿真

        Args:
            external_disturbance: 外部扰动

        Returns:
            当前步骤数据
        """
        start_time = time.time()

        # 1. 物理系统步进
        Q_upstream = 265.0  # 上游流量
        h_downstream = 6.0   # 下游水位

        # 应用外部扰动
        if external_disturbance:
            if 'flow_change' in external_disturbance:
                Q_upstream = external_disturbance['flow_change']
            if 'seismic' in external_disturbance:
                seismic_acc = external_disturbance['seismic']
            else:
                seismic_acc = 0.0
        else:
            seismic_acc = 0.0

        # 获取阀门状态以调整流量
        valve_positions = self.actuators.get_valve_positions()
        Q_north = Q_upstream * 0.5 * valve_positions.get('north_inlet', 1.0)
        Q_south = Q_upstream * 0.5 * valve_positions.get('south_inlet', 1.0)
        Q_effective = Q_north + Q_south

        # 物理模型步进
        state = self.physical_model.step(
            self.dt,
            Q_effective,
            h_downstream,
            seismic_acceleration=seismic_acc
        )

        # 2. 传感器读取
        physical_state = {
            'pressure': state.pressure,
            'flow_rate': state.flow_rate,
            'water_temperature': state.water_temperature,
            'strain': state.strain
        }
        sensor_data = self.instrumentation.read_all(physical_state, self.dt)

        # 提取关键测量值
        point_sensors = sensor_data.get('point_sensors', {})
        measurements = {
            'flow_rate': np.mean(state.flow_rate),
            'north_flow': Q_north,
            'south_flow': Q_south,
            'pressure': float(point_sensors.get('P_5', 0.5e6)),
            'temperature': float(point_sensors.get('T_5', 15.0)),
        }

        # 3. 控制计算
        commands = self.controller.compute(measurements, None, self.dt)

        # 4. 执行器步进
        actuator_outputs = self.actuators.step_all(commands, self.dt)

        # 5. 性能统计
        flow_error = abs(np.mean(state.flow_rate) - self.controller.flow_setpoint)
        self.flow_errors.append(flow_error)
        self.pressure_readings.append(np.mean(state.pressure))

        control_loop_time = (time.time() - start_time) * 1000

        # 6. 记录数据
        step_data = {
            'time': self.time,
            'state': {
                'flow_rate': float(np.mean(state.flow_rate)),
                'pressure': float(np.mean(state.pressure)),
                'water_level': float(np.mean(state.water_level)),
                'temperature': float(np.mean(state.water_temperature) - 273.15),
            },
            'measurements': measurements,
            'commands': commands,
            'actuator_outputs': actuator_outputs,
            'scenario': self.controller.current_scenario,
            'control_loop_time': control_loop_time
        }

        if len(self.history) < self.max_history:
            self.history.append(step_data)

        self.time += self.dt

        return step_data

    def run_test_case(self, test_case: HILTestCase,
                     progress_callback: Optional[Callable] = None) -> HILTestResult:
        """
        运行测试用例

        Args:
            test_case: 测试用例
            progress_callback: 进度回调

        Returns:
            测试结果
        """
        # 初始化
        self.initialize()
        timeline = []
        violations = []
        test_start = time.time()

        self.is_running = True

        try:
            # 执行场景序列
            scenario_idx = 0
            current_scenario = test_case.scenario_sequence[0] if test_case.scenario_sequence else {}
            scenario_start_time = 0.0

            # 应用故障注入
            fault_schedule = {f['time']: f for f in test_case.fault_injections}

            while self.time < test_case.duration and self.is_running:
                # 检查场景切换
                if scenario_idx < len(test_case.scenario_sequence) - 1:
                    next_scenario = test_case.scenario_sequence[scenario_idx + 1]
                    if self.time >= next_scenario.get('start_time', float('inf')):
                        scenario_idx += 1
                        current_scenario = next_scenario
                        scenario_start_time = self.time
                        timeline.append({
                            'time': self.time,
                            'event': 'scenario_change',
                            'scenario': current_scenario.get('id', 'unknown')
                        })

                # 构建扰动
                disturbance = {}
                if 'flow_setpoint' in current_scenario:
                    self.controller.set_setpoints(
                        current_scenario['flow_setpoint'],
                        current_scenario.get('pressure_setpoint', 0.5e6)
                    )
                if 'disturbance' in current_scenario:
                    disturbance = current_scenario['disturbance']

                # 检查故障注入
                for fault_time, fault in fault_schedule.items():
                    if abs(self.time - fault_time) < self.dt:
                        self._inject_fault(fault)
                        timeline.append({
                            'time': self.time,
                            'event': 'fault_injection',
                            'fault': fault
                        })

                # 执行步进
                step_data = self.step(disturbance)

                # 检查约束违反
                violation = self._check_constraints(step_data, test_case.success_criteria)
                if violation:
                    violations.append(f"t={self.time:.1f}s: {violation}")
                    timeline.append({
                        'time': self.time,
                        'event': 'constraint_violation',
                        'violation': violation
                    })

                # 进度回调
                if progress_callback:
                    progress = self.time / test_case.duration
                    progress_callback(progress, step_data)

            # 计算最终指标
            self._compute_metrics()

            # 判定结果
            result = self._evaluate_result(test_case.success_criteria, violations)

        except Exception as e:
            result = TestResult.FAILED
            violations.append(f"Exception: {str(e)}")

        self.is_running = False

        return HILTestResult(
            test_case_id=test_case.id,
            result=result,
            metrics=self.metrics,
            timeline=timeline,
            violations=violations,
            duration=time.time() - test_start,
            timestamp=time.time()
        )

    def _inject_fault(self, fault: Dict[str, Any]):
        """注入故障"""
        fault_type = fault.get('type', '')
        target = fault.get('target', '')
        params = fault.get('params', {})

        if fault_type == 'sensor':
            if target in self.instrumentation.sensors.sensors:
                from cyrp.hil.sensor_models import FailureMode
                mode = FailureMode[params.get('mode', 'STUCK').upper()]
                self.instrumentation.sensors.sensors[target].inject_failure(mode, params)

        elif fault_type == 'actuator':
            if target in self.actuators.actuators:
                from cyrp.hil.actuator_models import ActuatorFailureMode
                mode = ActuatorFailureMode[params.get('mode', 'STUCK_POSITION').upper()]
                self.actuators.actuators[target].inject_failure(mode, params)

        elif fault_type == 'physical':
            self.physical_model.inject_fault(params.get('fault_type', ''), params)

    def _check_constraints(self, step_data: Dict[str, Any],
                          criteria: Dict[str, float]) -> Optional[str]:
        """检查约束"""
        state = step_data.get('state', {})

        # 流量约束
        if 'max_flow_error' in criteria:
            flow = state.get('flow_rate', 0)
            error = abs(flow - self.controller.flow_setpoint)
            if error > criteria['max_flow_error']:
                return f"Flow error {error:.1f} > {criteria['max_flow_error']}"

        # 压力约束
        if 'max_pressure' in criteria:
            pressure = state.get('pressure', 0)
            if pressure > criteria['max_pressure']:
                return f"Pressure {pressure:.0f} > {criteria['max_pressure']}"

        if 'min_pressure' in criteria:
            pressure = state.get('pressure', 0)
            if pressure < criteria['min_pressure']:
                return f"Pressure {pressure:.0f} < {criteria['min_pressure']}"

        return None

    def _compute_metrics(self):
        """计算性能指标"""
        if self.flow_errors:
            self.metrics.flow_tracking_error = np.mean(list(self.flow_errors))

        if self.pressure_readings:
            pressures = np.array(list(self.pressure_readings))
            self.metrics.pressure_stability = np.std(pressures) / np.mean(pressures) * 100

        # 传感器可用率
        health_report = self.instrumentation.sensors.get_health_report()
        normal_count = sum(1 for h in health_report.values()
                         if h.get('status') == 'normal')
        self.metrics.sensor_availability = normal_count / len(health_report) * 100

        # 执行器健康度
        actuator_report = self.actuators.get_status_report()
        ready_count = sum(1 for a in actuator_report.values()
                        if a.get('status') in ['ready', 'operating'])
        self.metrics.actuator_health = ready_count / len(actuator_report) * 100

    def _evaluate_result(self, criteria: Dict[str, float],
                        violations: List[str]) -> TestResult:
        """评估测试结果"""
        if len(violations) > criteria.get('max_violations', 0):
            return TestResult.FAILED

        if self.metrics.flow_tracking_error > criteria.get('max_flow_error', float('inf')):
            return TestResult.FAILED

        if self.metrics.pressure_stability > criteria.get('max_pressure_variation', float('inf')):
            return TestResult.FAILED

        if violations:
            return TestResult.WARNING

        return TestResult.PASSED

    def generate_full_test_suite(self) -> List[HILTestCase]:
        """生成完整测试套件"""
        test_cases = []

        # 1. 常规运行测试
        test_cases.append(HILTestCase(
            id="TC_001",
            name="Nominal Operation",
            description="常规双洞运行测试",
            scenario_sequence=[
                {'id': 'S2-A', 'start_time': 0, 'flow_setpoint': 265},
            ],
            duration=300,
            success_criteria={
                'max_flow_error': 10.0,
                'max_pressure': 1.0e6,
                'min_pressure': 0.1e6,
                'max_violations': 0
            }
        ))

        # 2. 流量变化测试
        test_cases.append(HILTestCase(
            id="TC_002",
            name="Flow Change Response",
            description="流量阶跃变化响应测试",
            scenario_sequence=[
                {'id': 'S2-A', 'start_time': 0, 'flow_setpoint': 265},
                {'id': 'S2-A', 'start_time': 60, 'flow_setpoint': 200},
                {'id': 'S2-A', 'start_time': 120, 'flow_setpoint': 300},
                {'id': 'S2-A', 'start_time': 180, 'flow_setpoint': 265},
            ],
            duration=300,
            success_criteria={
                'max_flow_error': 20.0,
                'max_pressure': 1.0e6,
                'max_violations': 5
            }
        ))

        # 3. 隧道切换测试
        test_cases.append(HILTestCase(
            id="TC_003",
            name="Tunnel Switch",
            description="计划隧道切换测试",
            scenario_sequence=[
                {'id': 'S2-A', 'start_time': 0, 'flow_setpoint': 265},
                {'id': 'S3-A', 'start_time': 60, 'flow_setpoint': 200},
                {'id': 'S1-A', 'start_time': 180, 'flow_setpoint': 265},
            ],
            duration=300,
            success_criteria={
                'max_flow_error': 30.0,
                'max_violations': 10
            }
        ))

        # 4. 传感器故障测试
        test_cases.append(HILTestCase(
            id="TC_004",
            name="Sensor Failure Resilience",
            description="传感器故障容错测试",
            scenario_sequence=[
                {'id': 'S2-A', 'start_time': 0, 'flow_setpoint': 265},
            ],
            duration=300,
            success_criteria={
                'max_flow_error': 25.0,
                'max_violations': 5
            },
            fault_injections=[
                {'time': 60, 'type': 'sensor', 'target': 'P_3',
                 'params': {'mode': 'stuck', 'stuck_value': 0.5e6}},
                {'time': 120, 'type': 'sensor', 'target': 'P_5',
                 'params': {'mode': 'drift', 'drift_rate': 0.01}},
            ]
        ))

        # 5. 执行器故障测试
        test_cases.append(HILTestCase(
            id="TC_005",
            name="Actuator Failure Response",
            description="执行器故障响应测试",
            scenario_sequence=[
                {'id': 'S2-A', 'start_time': 0, 'flow_setpoint': 265},
            ],
            duration=300,
            success_criteria={
                'max_flow_error': 40.0,
                'max_violations': 10
            },
            fault_injections=[
                {'time': 60, 'type': 'actuator', 'target': 'control_valve',
                 'params': {'mode': 'slow_response', 'factor': 0.3}},
            ]
        ))

        # 6. 渗漏检测响应测试
        test_cases.append(HILTestCase(
            id="TC_006",
            name="Leakage Detection",
            description="渗漏检测与响应测试",
            scenario_sequence=[
                {'id': 'S2-A', 'start_time': 0, 'flow_setpoint': 265},
                {'id': 'S5-B', 'start_time': 60, 'flow_setpoint': 200},
            ],
            duration=300,
            success_criteria={
                'max_flow_error': 50.0,
                'max_violations': 15
            },
            fault_injections=[
                {'time': 60, 'type': 'physical',
                 'params': {'fault_type': 'leakage', 'location': 50, 'rate': 0.02}},
            ]
        ))

        # 7. 地震响应测试
        test_cases.append(HILTestCase(
            id="TC_007",
            name="Seismic Response",
            description="地震事件响应测试",
            scenario_sequence=[
                {'id': 'S2-A', 'start_time': 0, 'flow_setpoint': 265},
                {'id': 'S6-B', 'start_time': 30, 'flow_setpoint': 150,
                 'disturbance': {'seismic': 0.15}},
                {'id': 'S2-A', 'start_time': 120, 'flow_setpoint': 265},
            ],
            duration=300,
            success_criteria={
                'max_flow_error': 100.0,
                'max_violations': 20
            }
        ))

        # 8. 综合应急测试
        test_cases.append(HILTestCase(
            id="TC_008",
            name="Combined Emergency",
            description="多故障综合应急测试",
            scenario_sequence=[
                {'id': 'S2-A', 'start_time': 0, 'flow_setpoint': 265},
                {'id': 'S7', 'start_time': 60, 'flow_setpoint': 100},
            ],
            duration=300,
            success_criteria={
                'max_violations': 30
            },
            fault_injections=[
                {'time': 60, 'type': 'physical',
                 'params': {'fault_type': 'leakage', 'location': 50, 'rate': 0.01}},
                {'time': 60, 'type': 'sensor', 'target': 'P_5',
                 'params': {'mode': 'noise_increase', 'factor': 5}},
                {'time': 60, 'type': 'actuator', 'target': 'north_inlet',
                 'params': {'mode': 'slow_response', 'factor': 0.5}},
            ]
        ))

        return test_cases

    def run_full_test_suite(self,
                           progress_callback: Optional[Callable] = None) -> Dict[str, HILTestResult]:
        """运行完整测试套件"""
        test_cases = self.generate_full_test_suite()
        results = {}

        for i, test_case in enumerate(test_cases):
            print(f"Running test {i+1}/{len(test_cases)}: {test_case.name}")

            result = self.run_test_case(
                test_case,
                lambda p, d: progress_callback((i + p) / len(test_cases), d)
                if progress_callback else None
            )

            results[test_case.id] = result

            print(f"  Result: {result.result.value}")
            print(f"  Flow Error: {result.metrics.flow_tracking_error:.2f} m³/s")
            print(f"  Violations: {len(result.violations)}")

        return results

    def get_test_report(self, results: Dict[str, HILTestResult]) -> Dict[str, Any]:
        """生成测试报告"""
        total = len(results)
        passed = sum(1 for r in results.values() if r.result == TestResult.PASSED)
        failed = sum(1 for r in results.values() if r.result == TestResult.FAILED)
        warnings = sum(1 for r in results.values() if r.result == TestResult.WARNING)

        return {
            'summary': {
                'total_tests': total,
                'passed': passed,
                'failed': failed,
                'warnings': warnings,
                'pass_rate': passed / total * 100 if total > 0 else 0
            },
            'details': {
                tc_id: {
                    'result': r.result.value,
                    'metrics': {
                        'flow_tracking_error': r.metrics.flow_tracking_error,
                        'pressure_stability': r.metrics.pressure_stability,
                        'sensor_availability': r.metrics.sensor_availability,
                        'actuator_health': r.metrics.actuator_health,
                    },
                    'violations_count': len(r.violations),
                    'duration': r.duration
                }
                for tc_id, r in results.items()
            },
            'recommendations': self._generate_recommendations(results)
        }

    def _generate_recommendations(self, results: Dict[str, HILTestResult]) -> List[str]:
        """生成改进建议"""
        recommendations = []

        for tc_id, result in results.items():
            if result.result == TestResult.FAILED:
                if result.metrics.flow_tracking_error > 50:
                    recommendations.append(
                        f"{tc_id}: 流量跟踪误差过大，建议增加控制器增益"
                    )
                if result.metrics.sensor_availability < 90:
                    recommendations.append(
                        f"{tc_id}: 传感器可用率低，建议增加冗余传感器"
                    )
                if result.metrics.actuator_health < 90:
                    recommendations.append(
                        f"{tc_id}: 执行器健康度低，建议检查维护计划"
                    )

        return recommendations
