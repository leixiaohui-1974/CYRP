"""
Hardware-in-the-Loop Test Framework for CYRP.
穿黄工程在环测试框架
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
import time
import numpy as np

from cyrp.core import PhysicalSystem, TunnelParameters, EnvironmentParameters
from cyrp.perception import PerceptionSystem
from cyrp.control import HDMPCController
from cyrp.agents import MultiAgentSystem
from cyrp.scenarios import ScenarioManager, ScenarioGenerator, TestScenario, ScenarioType


class TestStatus(Enum):
    """测试状态"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class TestResult:
    """测试结果"""
    test_name: str
    status: TestStatus
    duration: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HILConfig:
    """在环测试配置"""
    # 仿真参数
    simulation_dt: float = 0.1  # 仿真时间步 (s)
    realtime_factor: float = 1.0  # 实时因子 (>1加速, <1减速)

    # 测试参数
    default_duration: float = 3600.0  # 默认测试时长 (s)
    warmup_time: float = 60.0  # 预热时间 (s)

    # 性能指标阈值
    flow_tracking_threshold: float = 0.05  # 流量跟踪误差阈值
    pressure_stability_threshold: float = 0.1  # 压力稳定性阈值
    response_time_threshold: float = 30.0  # 响应时间阈值 (s)

    # 安全阈值
    max_pressure: float = 1.0e6
    min_pressure: float = -5e4
    max_flow_imbalance: float = 0.1


class HILTestFramework:
    """
    在环测试框架

    实现:
    1. 全场景自动测试
    2. 多智能体协同验证
    3. 故障注入测试
    4. 性能评估
    5. 回归测试
    """

    def __init__(self, config: Optional[HILConfig] = None):
        """
        初始化在环测试框架

        Args:
            config: 测试配置
        """
        self.config = config or HILConfig()

        # 初始化物理系统
        self.physical_system = PhysicalSystem()

        # 初始化多智能体系统
        self.mas = MultiAgentSystem()

        # 场景生成器
        self.scenario_generator = ScenarioGenerator()

        # 测试结果
        self.results: List[TestResult] = []

        # 运行状态
        self.running = False
        self.current_test: Optional[str] = None

        # 数据记录
        self.data_log: List[Dict] = []

    def setup(self):
        """测试前设置"""
        self.physical_system.reset()
        self.mas.reset()
        self.data_log = []

    def teardown(self):
        """测试后清理"""
        pass

    def run_test(self, test_scenario: TestScenario) -> TestResult:
        """
        运行单个测试场景

        Args:
            test_scenario: 测试场景

        Returns:
            测试结果
        """
        result = TestResult(
            test_name=test_scenario.name,
            status=TestStatus.RUNNING
        )

        start_time = time.time()
        self.current_test = test_scenario.name

        try:
            # 设置
            self.setup()

            # 预热
            self._warmup(self.config.warmup_time)

            # 运行测试
            test_data = self._execute_test(test_scenario)

            # 评估结果
            metrics = self._evaluate_test(test_scenario, test_data)
            result.metrics = metrics

            # 判断通过/失败
            passed = self._check_pass_criteria(test_scenario, metrics)
            result.status = TestStatus.PASSED if passed else TestStatus.FAILED

            result.data = test_data

        except Exception as e:
            result.status = TestStatus.ERROR
            result.errors.append(str(e))

        finally:
            result.duration = time.time() - start_time
            self.teardown()
            self.current_test = None

        self.results.append(result)
        return result

    def _warmup(self, duration: float):
        """预热阶段"""
        t = 0.0
        dt = self.config.simulation_dt

        while t < duration:
            # 运行稳态
            from cyrp.core.physical_system import ControlCommand
            cmd = ControlCommand()
            self.physical_system.step(cmd, dt)
            t += dt

    def _execute_test(self, test_scenario: TestScenario) -> Dict[str, Any]:
        """执行测试"""
        from cyrp.core.physical_system import ControlCommand

        data = {
            'time': [],
            'Q1': [], 'Q2': [],
            'H_inlet': [], 'H_outlet': [],
            'gate_1': [], 'gate_2': [],
            'scenario': [],
            'risk_level': [],
            'alarms': []
        }

        # 获取扰动时间表
        disturbance_schedule = self.scenario_generator.get_disturbance_schedule(test_scenario)

        # 应用故障
        for fault in test_scenario.faults:
            self.physical_system.fault_injector.schedule_fault(
                fault.fault_type,
                fault.start_time,
                fault.duration,
                fault.parameters
            )

        t = 0.0
        dt = self.config.simulation_dt

        while t < test_scenario.duration:
            # 获取扰动
            disturbance = {}
            for t_dist, dist_val in disturbance_schedule.items():
                if abs(t - t_dist) < dt:
                    disturbance = dist_val
                    break

            # 构建环境
            environment = {
                'system_state': self.physical_system.state,
                'sensor_data': self._get_sensor_data(),
                'time': t,
                'dt': dt
            }

            # 多智能体系统步进
            mas_result = self.mas.step(environment)

            # 获取控制输出
            control = self.mas.get_control_output()
            cmd = ControlCommand(
                gate_inlet_1_target=control['gate_1'],
                gate_inlet_2_target=control['gate_2']
            )

            # 物理系统步进
            self.physical_system.step(cmd, dt, disturbance)

            # 记录数据
            state = self.physical_system.state
            data['time'].append(t)
            data['Q1'].append(state.hydraulic.Q1)
            data['Q2'].append(state.hydraulic.Q2)
            data['H_inlet'].append(state.hydraulic.H_inlet)
            data['H_outlet'].append(state.hydraulic.H_outlet)
            data['gate_1'].append(state.actuators.gate_inlet_1)
            data['gate_2'].append(state.actuators.gate_inlet_2)
            data['scenario'].append(self.mas.get_scenario())
            data['risk_level'].append(self.mas.get_risk_level())
            data['alarms'].append(list(state.alarms.keys()) if state.alarms else [])

            t += dt

        # 转换为numpy数组
        for key in ['time', 'Q1', 'Q2', 'H_inlet', 'H_outlet', 'gate_1', 'gate_2']:
            data[key] = np.array(data[key])

        return data

    def _get_sensor_data(self) -> Dict[str, Any]:
        """获取传感器数据"""
        state = self.physical_system.state
        return {
            'pressure_max': state.hydraulic.P_max,
            'pressure_min': state.hydraulic.P_min,
            'pressure': (state.hydraulic.P_max + state.hydraulic.P_min) / 2,
            'gate_positions': [
                state.actuators.gate_inlet_1,
                state.actuators.gate_inlet_2
            ],
            'flow_1': state.hydraulic.Q1,
            'flow_2': state.hydraulic.Q2
        }

    def _evaluate_test(
        self,
        test_scenario: TestScenario,
        data: Dict[str, Any]
    ) -> Dict[str, float]:
        """评估测试结果"""
        metrics = {}

        # 流量跟踪误差
        Q_total = data['Q1'] + data['Q2']
        Q_ref = 265.0  # 参考流量
        metrics['flow_tracking_error'] = np.mean(np.abs(Q_total - Q_ref) / Q_ref)

        # 流量平衡度
        imbalance = np.abs(data['Q1'] - data['Q2']) / (Q_total + 1e-10)
        metrics['max_flow_imbalance'] = np.max(imbalance)
        metrics['avg_flow_imbalance'] = np.mean(imbalance)

        # 压力稳定性
        H_diff = data['H_inlet'] - data['H_outlet']
        metrics['pressure_stability'] = np.std(H_diff) / np.mean(H_diff)

        # 闸门动作量
        gate_movement = np.sum(np.abs(np.diff(data['gate_1']))) + \
                       np.sum(np.abs(np.diff(data['gate_2'])))
        metrics['total_gate_movement'] = gate_movement

        # 风险等级统计
        metrics['max_risk_level'] = max(data['risk_level'])
        metrics['avg_risk_level'] = np.mean(data['risk_level'])

        # 检查预期结果
        expected = test_scenario.expected_outcomes
        if 'flow_imbalance_max' in expected:
            metrics['flow_imbalance_pass'] = (
                metrics['max_flow_imbalance'] <= expected['flow_imbalance_max']
            )

        return metrics

    def _check_pass_criteria(
        self,
        test_scenario: TestScenario,
        metrics: Dict[str, float]
    ) -> bool:
        """检查通过条件"""
        # 基本安全检查
        if metrics.get('max_risk_level', 0) >= 5:
            return False

        # 流量跟踪
        if metrics.get('flow_tracking_error', 1) > self.config.flow_tracking_threshold:
            return False

        # 流量平衡
        if metrics.get('max_flow_imbalance', 1) > self.config.max_flow_imbalance:
            # 某些场景允许不平衡
            if test_scenario.base_scenario not in [
                ScenarioType.S2_A_SEDIMENT_FLUSH,
                ScenarioType.S4_A_SWITCH_TUNNEL
            ]:
                return False

        # 检查场景特定条件
        expected = test_scenario.expected_outcomes
        for key, expected_val in expected.items():
            if key in metrics:
                if isinstance(expected_val, bool):
                    if metrics[key] != expected_val:
                        return False
                elif isinstance(expected_val, (int, float)):
                    if metrics[key] > expected_val:
                        return False

        return True

    def run_suite(
        self,
        test_scenarios: Optional[List[TestScenario]] = None
    ) -> List[TestResult]:
        """
        运行测试套件

        Args:
            test_scenarios: 测试场景列表

        Returns:
            测试结果列表
        """
        if test_scenarios is None:
            test_scenarios = self.scenario_generator.generate_full_coverage_suite()

        results = []
        for scenario in test_scenarios:
            print(f"Running test: {scenario.name}")
            result = self.run_test(scenario)
            results.append(result)
            print(f"  Status: {result.status.value}")

        return results

    def run_scenario_test(self, scenario_type: ScenarioType) -> TestResult:
        """
        运行特定场景测试

        Args:
            scenario_type: 场景类型

        Returns:
            测试结果
        """
        # 根据场景类型生成测试
        if scenario_type == ScenarioType.S1_A_DUAL_BALANCED:
            test = self.scenario_generator.generate_nominal_test()
        elif scenario_type == ScenarioType.S4_A_SWITCH_TUNNEL:
            test = self.scenario_generator.generate_tunnel_switch_test()
        elif scenario_type == ScenarioType.S3_A_FILLING:
            test = self.scenario_generator.generate_filling_test()
        elif scenario_type == ScenarioType.S5_A_INNER_LEAK:
            test = self.scenario_generator.generate_leakage_test()
        elif scenario_type == ScenarioType.S6_A_LIQUEFACTION:
            test = self.scenario_generator.generate_earthquake_test()
        elif scenario_type == ScenarioType.S7_B_GATE_ASYNC:
            test = self.scenario_generator.generate_gate_failure_test()
        else:
            test = self.scenario_generator.generate_random_scenario()

        return self.run_test(test)

    def generate_report(self) -> str:
        """生成测试报告"""
        report = []
        report.append("=" * 60)
        report.append("CYRP HIL Test Report")
        report.append("=" * 60)
        report.append("")

        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in self.results if r.status == TestStatus.FAILED)
        errors = sum(1 for r in self.results if r.status == TestStatus.ERROR)

        report.append(f"Total Tests: {total}")
        report.append(f"Passed: {passed}")
        report.append(f"Failed: {failed}")
        report.append(f"Errors: {errors}")
        report.append(f"Pass Rate: {passed/total*100:.1f}%" if total > 0 else "N/A")
        report.append("")

        report.append("-" * 60)
        report.append("Test Details:")
        report.append("-" * 60)

        for result in self.results:
            report.append(f"\n{result.test_name}")
            report.append(f"  Status: {result.status.value}")
            report.append(f"  Duration: {result.duration:.2f}s")
            if result.metrics:
                report.append("  Metrics:")
                for key, val in result.metrics.items():
                    if isinstance(val, float):
                        report.append(f"    {key}: {val:.4f}")
                    else:
                        report.append(f"    {key}: {val}")
            if result.errors:
                report.append("  Errors:")
                for err in result.errors:
                    report.append(f"    - {err}")

        report.append("")
        report.append("=" * 60)

        return "\n".join(report)
