"""
传感器仿真系统 - Sensor Simulation System

实现完整的传感器仿真功能，包括：
- 多类型传感器虚拟网络
- 真实噪声模型和漂移模型
- 故障注入与仿真
- 传感器数据生成与校验
- 传感器健康监测

Implements complete sensor simulation including:
- Multi-type virtual sensor network
- Realistic noise and drift models
- Failure injection and simulation
- Sensor data generation and validation
- Sensor health monitoring
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from collections import deque
import time
import threading
from concurrent.futures import ThreadPoolExecutor


class NoiseType(Enum):
    """噪声类型"""
    GAUSSIAN = "gaussian"           # 高斯白噪声
    COLORED = "colored"             # 有色噪声
    IMPULSE = "impulse"             # 脉冲噪声
    QUANTIZATION = "quantization"   # 量化噪声
    FLICKER = "flicker"             # 1/f噪声
    THERMAL = "thermal"             # 热噪声
    SHOT = "shot"                   # 散粒噪声


class DriftType(Enum):
    """漂移类型"""
    LINEAR = "linear"               # 线性漂移
    RANDOM_WALK = "random_walk"     # 随机游走
    SINUSOIDAL = "sinusoidal"       # 周期性漂移
    STEP = "step"                   # 阶跃漂移
    EXPONENTIAL = "exponential"     # 指数漂移
    TEMPERATURE = "temperature"     # 温度相关漂移


class SensorFailureType(Enum):
    """传感器故障类型"""
    NONE = "none"
    COMPLETE_FAILURE = "complete_failure"     # 完全失效
    STUCK_VALUE = "stuck_value"               # 卡死
    INTERMITTENT = "intermittent"             # 间歇性故障
    DRIFT_FAILURE = "drift_failure"           # 漂移故障
    BIAS_FAILURE = "bias_failure"             # 偏差故障
    NOISE_INCREASE = "noise_increase"         # 噪声增大
    DELAYED_RESPONSE = "delayed_response"     # 响应延迟
    SATURATION = "saturation"                 # 饱和
    NONLINEARITY = "nonlinearity"             # 非线性失真
    CROSS_SENSITIVITY = "cross_sensitivity"   # 交叉敏感


@dataclass
class NoiseModel:
    """噪声模型"""
    noise_type: NoiseType = NoiseType.GAUSSIAN
    amplitude: float = 0.01          # 噪声幅度 (相对于满量程)
    bandwidth: float = 100.0         # 带宽 (Hz)
    spectral_density: float = 1e-6   # 功率谱密度
    correlation_time: float = 0.01   # 相关时间 (s)
    impulse_probability: float = 0.001  # 脉冲噪声概率

    # 有色噪声参数
    ar_coefficients: List[float] = field(default_factory=lambda: [0.95])
    ma_coefficients: List[float] = field(default_factory=list)

    def __post_init__(self):
        self._state = 0.0
        self._history = deque(maxlen=100)

    def generate(self, dt: float = 0.01) -> float:
        """生成噪声样本"""
        if self.noise_type == NoiseType.GAUSSIAN:
            return np.random.normal(0, self.amplitude)

        elif self.noise_type == NoiseType.COLORED:
            # AR模型生成有色噪声
            white = np.random.normal(0, self.amplitude)
            self._history.append(white)

            colored = white
            for i, coef in enumerate(self.ar_coefficients):
                if len(self._history) > i + 1:
                    colored += coef * self._history[-(i + 2)]
            return colored

        elif self.noise_type == NoiseType.IMPULSE:
            if np.random.random() < self.impulse_probability:
                return np.random.choice([-1, 1]) * self.amplitude * 10
            return np.random.normal(0, self.amplitude * 0.1)

        elif self.noise_type == NoiseType.QUANTIZATION:
            # 量化噪声均匀分布
            return np.random.uniform(-self.amplitude/2, self.amplitude/2)

        elif self.noise_type == NoiseType.FLICKER:
            # 1/f噪声 (积分白噪声近似)
            self._state += np.random.normal(0, self.amplitude * np.sqrt(dt))
            self._state *= 0.999  # 缓慢衰减防止无界增长
            return self._state

        elif self.noise_type == NoiseType.THERMAL:
            # 热噪声 (温度相关)
            kB = 1.38e-23  # 玻尔兹曼常数
            T = 300  # 温度 (K)
            R = 1000  # 电阻 (Ohm)
            v_rms = np.sqrt(4 * kB * T * R * self.bandwidth)
            return np.random.normal(0, v_rms * self.amplitude / 1e-6)

        elif self.noise_type == NoiseType.SHOT:
            # 散粒噪声 (泊松近似)
            return np.random.normal(0, self.amplitude * np.sqrt(2 * 1.6e-19 * self.bandwidth))

        return np.random.normal(0, self.amplitude)


@dataclass
class DriftModel:
    """漂移模型"""
    drift_type: DriftType = DriftType.RANDOM_WALK
    rate: float = 0.001             # 漂移率 (%/h)
    amplitude: float = 0.01         # 漂移幅度
    period: float = 3600.0          # 周期 (s)
    temperature_coefficient: float = 0.02  # 温度系数 (%/°C)
    reference_temperature: float = 20.0    # 参考温度

    def __post_init__(self):
        self._cumulative_drift = 0.0
        self._last_time = 0.0
        self._step_occurred = False

    def compute(self, time: float, temperature: float = 20.0) -> float:
        """计算漂移值"""
        dt = time - self._last_time if self._last_time > 0 else 0
        self._last_time = time

        if self.drift_type == DriftType.LINEAR:
            return self.rate * time / 3600.0

        elif self.drift_type == DriftType.RANDOM_WALK:
            self._cumulative_drift += np.random.normal(0, self.rate * np.sqrt(dt / 3600.0))
            return self._cumulative_drift

        elif self.drift_type == DriftType.SINUSOIDAL:
            return self.amplitude * np.sin(2 * np.pi * time / self.period)

        elif self.drift_type == DriftType.STEP:
            if not self._step_occurred and time > self.period:
                self._step_occurred = True
                self._cumulative_drift = self.amplitude
            return self._cumulative_drift

        elif self.drift_type == DriftType.EXPONENTIAL:
            tau = self.period / 3
            return self.amplitude * (1 - np.exp(-time / tau))

        elif self.drift_type == DriftType.TEMPERATURE:
            delta_T = temperature - self.reference_temperature
            return self.temperature_coefficient * delta_T / 100.0

        return 0.0

    def reset(self):
        """重置漂移状态"""
        self._cumulative_drift = 0.0
        self._last_time = 0.0
        self._step_occurred = False


@dataclass
class SensorCharacteristics:
    """传感器特性参数"""
    # 基本参数
    sensor_type: str = "generic"
    measurement_range: Tuple[float, float] = (0.0, 100.0)
    resolution: float = 0.01
    accuracy: float = 0.1           # %FS

    # 动态响应
    response_time: float = 0.1      # 时间常数 (s)
    bandwidth: float = 100.0        # 带宽 (Hz)
    sampling_rate: float = 1000.0   # 采样率 (Hz)

    # 环境影响
    temperature_range: Tuple[float, float] = (-20.0, 80.0)
    temperature_coefficient: float = 0.02  # %/°C
    humidity_sensitivity: float = 0.01     # %/%RH

    # 可靠性
    mtbf: float = 50000.0           # 平均故障间隔 (h)
    mttr: float = 4.0               # 平均修复时间 (h)

    # 噪声和漂移
    noise_model: NoiseModel = field(default_factory=NoiseModel)
    drift_model: DriftModel = field(default_factory=DriftModel)


class VirtualSensor:
    """虚拟传感器"""

    def __init__(self, sensor_id: str, characteristics: SensorCharacteristics,
                 location: Tuple[float, float, float] = (0.0, 0.0, 0.0)):
        self.sensor_id = sensor_id
        self.characteristics = characteristics
        self.location = location

        # 状态
        self.is_active = True
        self.failure_type = SensorFailureType.NONE
        self.failure_params: Dict[str, Any] = {}

        # 内部状态
        self._output_value = 0.0
        self._internal_state = 0.0
        self._time = 0.0
        self._bias = 0.0
        self._delay_buffer: deque = deque(maxlen=100)

        # 统计
        self._samples_count = 0
        self._error_count = 0

    def read(self, true_value: float, dt: float,
             environment: Optional[Dict[str, float]] = None) -> float:
        """
        读取传感器值

        Args:
            true_value: 真实物理量
            dt: 时间步长
            environment: 环境条件 (温度、湿度等)

        Returns:
            传感器输出值
        """
        self._time += dt
        self._samples_count += 1
        environment = environment or {}

        # 检查是否激活
        if not self.is_active:
            return np.nan

        # 故障处理
        if self.failure_type != SensorFailureType.NONE:
            return self._apply_failure(true_value)

        # 1. 范围限制
        value = np.clip(true_value, *self.characteristics.measurement_range)

        # 2. 动态响应 (一阶滞后)
        tau = self.characteristics.response_time
        if tau > 0:
            alpha = dt / (tau + dt)
            self._internal_state = alpha * value + (1 - alpha) * self._internal_state
            value = self._internal_state
        else:
            self._internal_state = value

        # 3. 温度影响
        temp = environment.get('temperature', 20.0)
        temp_effect = self.characteristics.temperature_coefficient * (temp - 20.0) / 100.0
        value *= (1 + temp_effect)

        # 4. 漂移
        drift = self.characteristics.drift_model.compute(self._time, temp)
        span = self.characteristics.measurement_range[1] - self.characteristics.measurement_range[0]
        value += drift * span

        # 5. 偏差
        value += self._bias * span

        # 6. 噪声
        noise = self.characteristics.noise_model.generate(dt)
        value += noise * span

        # 7. 量化
        resolution = self.characteristics.resolution
        value = round(value / resolution) * resolution

        # 8. 饱和检查
        value = np.clip(value, *self.characteristics.measurement_range)

        self._output_value = value
        return value

    def _apply_failure(self, true_value: float) -> float:
        """应用故障模式"""
        span = self.characteristics.measurement_range[1] - self.characteristics.measurement_range[0]

        if self.failure_type == SensorFailureType.COMPLETE_FAILURE:
            return np.nan

        elif self.failure_type == SensorFailureType.STUCK_VALUE:
            return self.failure_params.get('stuck_value', self._output_value)

        elif self.failure_type == SensorFailureType.INTERMITTENT:
            if np.random.random() < self.failure_params.get('failure_probability', 0.1):
                return np.nan
            return true_value + np.random.normal(0, 0.01 * span)

        elif self.failure_type == SensorFailureType.DRIFT_FAILURE:
            drift_rate = self.failure_params.get('drift_rate', 0.1)
            self._bias += drift_rate * 0.01
            return true_value + self._bias * span

        elif self.failure_type == SensorFailureType.BIAS_FAILURE:
            bias = self.failure_params.get('bias', 0.1)
            return true_value + bias * span

        elif self.failure_type == SensorFailureType.NOISE_INCREASE:
            factor = self.failure_params.get('factor', 10)
            noise = np.random.normal(0, factor * 0.01 * span)
            return true_value + noise

        elif self.failure_type == SensorFailureType.DELAYED_RESPONSE:
            delay = self.failure_params.get('delay_samples', 10)
            self._delay_buffer.append(true_value)
            if len(self._delay_buffer) >= delay:
                return self._delay_buffer[0]
            return self._output_value

        elif self.failure_type == SensorFailureType.SATURATION:
            if true_value > 0.9 * self.characteristics.measurement_range[1]:
                return self.characteristics.measurement_range[1]
            if true_value < 0.1 * self.characteristics.measurement_range[0]:
                return self.characteristics.measurement_range[0]
            return true_value

        elif self.failure_type == SensorFailureType.NONLINEARITY:
            # 添加非线性失真
            normalized = (true_value - self.characteristics.measurement_range[0]) / span
            distortion = self.failure_params.get('distortion', 0.1)
            nonlinear = normalized + distortion * normalized * (1 - normalized)
            return self.characteristics.measurement_range[0] + nonlinear * span

        return true_value

    def inject_failure(self, failure_type: SensorFailureType,
                       params: Optional[Dict[str, Any]] = None):
        """注入故障"""
        self.failure_type = failure_type
        self.failure_params = params or {}

    def clear_failure(self):
        """清除故障"""
        self.failure_type = SensorFailureType.NONE
        self.failure_params = {}
        self._bias = 0.0

    def calibrate(self, reference_value: float):
        """校准传感器"""
        span = self.characteristics.measurement_range[1] - self.characteristics.measurement_range[0]
        self._bias = (reference_value - self._output_value) / span
        self.characteristics.drift_model.reset()

    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        return {
            'sensor_id': self.sensor_id,
            'is_active': self.is_active,
            'failure_type': self.failure_type.value,
            'current_value': self._output_value,
            'samples_count': self._samples_count,
            'error_rate': self._error_count / max(self._samples_count, 1),
            'bias': self._bias,
            'time': self._time
        }

    def reset(self):
        """重置传感器"""
        self._output_value = 0.0
        self._internal_state = 0.0
        self._time = 0.0
        self._bias = 0.0
        self._delay_buffer.clear()
        self._samples_count = 0
        self._error_count = 0
        self.characteristics.drift_model.reset()
        self.clear_failure()


class VirtualSensorNetwork:
    """虚拟传感器网络"""

    def __init__(self, name: str = "default_network"):
        self.name = name
        self.sensors: Dict[str, VirtualSensor] = {}
        self._lock = threading.Lock()

    def add_sensor(self, sensor: VirtualSensor):
        """添加传感器"""
        with self._lock:
            self.sensors[sensor.sensor_id] = sensor

    def remove_sensor(self, sensor_id: str):
        """移除传感器"""
        with self._lock:
            if sensor_id in self.sensors:
                del self.sensors[sensor_id]

    def read_all(self, true_values: Dict[str, float], dt: float,
                 environment: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """读取所有传感器"""
        readings = {}
        for sensor_id, sensor in self.sensors.items():
            if sensor_id in true_values:
                readings[sensor_id] = sensor.read(true_values[sensor_id], dt, environment)
            else:
                readings[sensor_id] = np.nan
        return readings

    def read_parallel(self, true_values: Dict[str, float], dt: float,
                      environment: Optional[Dict[str, float]] = None,
                      max_workers: int = 4) -> Dict[str, float]:
        """并行读取所有传感器"""
        readings = {}

        def read_sensor(args):
            sensor_id, sensor = args
            if sensor_id in true_values:
                return sensor_id, sensor.read(true_values[sensor_id], dt, environment)
            return sensor_id, np.nan

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(read_sensor, self.sensors.items())
            for sensor_id, value in results:
                readings[sensor_id] = value

        return readings

    def get_network_status(self) -> Dict[str, Any]:
        """获取网络状态"""
        active_count = sum(1 for s in self.sensors.values() if s.is_active)
        failed_count = sum(1 for s in self.sensors.values()
                         if s.failure_type != SensorFailureType.NONE)

        return {
            'name': self.name,
            'total_sensors': len(self.sensors),
            'active_sensors': active_count,
            'failed_sensors': failed_count,
            'sensors': {sid: s.get_health_status() for sid, s in self.sensors.items()}
        }

    def inject_random_failures(self, failure_probability: float = 0.05):
        """随机注入故障"""
        failure_types = list(SensorFailureType)
        failure_types.remove(SensorFailureType.NONE)

        for sensor in self.sensors.values():
            if np.random.random() < failure_probability:
                failure = np.random.choice(failure_types)
                sensor.inject_failure(failure)

    def clear_all_failures(self):
        """清除所有故障"""
        for sensor in self.sensors.values():
            sensor.clear_failure()

    def reset_all(self):
        """重置所有传感器"""
        for sensor in self.sensors.values():
            sensor.reset()


class SensorDataGenerator:
    """传感器数据生成器"""

    def __init__(self, sensor_network: VirtualSensorNetwork):
        self.network = sensor_network
        self._time = 0.0
        self._data_history: List[Dict[str, Any]] = []
        self._max_history = 10000

    def generate_step(self, physical_state: Dict[str, float], dt: float,
                      environment: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        生成一步传感器数据

        Args:
            physical_state: 物理状态 (真实值)
            dt: 时间步长
            environment: 环境条件

        Returns:
            传感器读数
        """
        self._time += dt

        readings = self.network.read_all(physical_state, dt, environment)

        data_point = {
            'timestamp': self._time,
            'readings': readings,
            'environment': environment or {},
            'physical_state': physical_state
        }

        # 保存历史
        self._data_history.append(data_point)
        if len(self._data_history) > self._max_history:
            self._data_history.pop(0)

        return data_point

    def generate_sequence(self, physical_states: List[Dict[str, float]],
                          dt: float, environments: Optional[List[Dict[str, float]]] = None
                          ) -> List[Dict[str, Any]]:
        """生成数据序列"""
        sequence = []
        environments = environments or [None] * len(physical_states)

        for state, env in zip(physical_states, environments):
            data = self.generate_step(state, dt, env)
            sequence.append(data)

        return sequence

    def get_history(self, n_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """获取历史数据"""
        if n_samples is None:
            return self._data_history.copy()
        return self._data_history[-n_samples:]

    def export_to_array(self, sensor_ids: Optional[List[str]] = None) -> np.ndarray:
        """导出为NumPy数组"""
        if not self._data_history:
            return np.array([])

        sensor_ids = sensor_ids or list(self.network.sensors.keys())
        data = []

        for record in self._data_history:
            row = [record['timestamp']]
            for sid in sensor_ids:
                row.append(record['readings'].get(sid, np.nan))
            data.append(row)

        return np.array(data)

    def reset(self):
        """重置生成器"""
        self._time = 0.0
        self._data_history.clear()
        self.network.reset_all()


class FailureInjector:
    """故障注入器"""

    def __init__(self, sensor_network: VirtualSensorNetwork):
        self.network = sensor_network
        self._scheduled_failures: List[Dict[str, Any]] = []
        self._active_failures: Dict[str, Dict[str, Any]] = {}

    def schedule_failure(self, sensor_id: str, failure_type: SensorFailureType,
                         start_time: float, duration: float = -1,
                         params: Optional[Dict[str, Any]] = None):
        """
        调度故障

        Args:
            sensor_id: 传感器ID
            failure_type: 故障类型
            start_time: 开始时间
            duration: 持续时间 (-1表示永久)
            params: 故障参数
        """
        self._scheduled_failures.append({
            'sensor_id': sensor_id,
            'failure_type': failure_type,
            'start_time': start_time,
            'duration': duration,
            'params': params or {},
            'activated': False
        })

    def update(self, current_time: float):
        """更新故障状态"""
        # 激活计划的故障
        for failure in self._scheduled_failures:
            if not failure['activated'] and current_time >= failure['start_time']:
                sensor_id = failure['sensor_id']
                if sensor_id in self.network.sensors:
                    self.network.sensors[sensor_id].inject_failure(
                        failure['failure_type'],
                        failure['params']
                    )
                    self._active_failures[sensor_id] = {
                        'start_time': failure['start_time'],
                        'duration': failure['duration'],
                        'failure_type': failure['failure_type']
                    }
                    failure['activated'] = True

        # 移除过期的故障
        for sensor_id in list(self._active_failures.keys()):
            info = self._active_failures[sensor_id]
            if info['duration'] > 0:
                if current_time >= info['start_time'] + info['duration']:
                    if sensor_id in self.network.sensors:
                        self.network.sensors[sensor_id].clear_failure()
                    del self._active_failures[sensor_id]

    def inject_immediate(self, sensor_id: str, failure_type: SensorFailureType,
                         params: Optional[Dict[str, Any]] = None):
        """立即注入故障"""
        if sensor_id in self.network.sensors:
            self.network.sensors[sensor_id].inject_failure(failure_type, params)

    def clear_failure(self, sensor_id: str):
        """清除指定传感器故障"""
        if sensor_id in self.network.sensors:
            self.network.sensors[sensor_id].clear_failure()
        if sensor_id in self._active_failures:
            del self._active_failures[sensor_id]

    def clear_all(self):
        """清除所有故障"""
        self.network.clear_all_failures()
        self._scheduled_failures.clear()
        self._active_failures.clear()

    def get_active_failures(self) -> Dict[str, Dict[str, Any]]:
        """获取活跃故障"""
        return self._active_failures.copy()


class SensorSimulationManager:
    """传感器仿真管理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.networks: Dict[str, VirtualSensorNetwork] = {}
        self.generators: Dict[str, SensorDataGenerator] = {}
        self.injectors: Dict[str, FailureInjector] = {}

        self._time = 0.0
        self._is_running = False

    def create_network(self, network_name: str) -> VirtualSensorNetwork:
        """创建传感器网络"""
        network = VirtualSensorNetwork(network_name)
        self.networks[network_name] = network
        self.generators[network_name] = SensorDataGenerator(network)
        self.injectors[network_name] = FailureInjector(network)
        return network

    def create_standard_tunnel_network(self, network_name: str = "tunnel",
                                        tunnel_length: float = 4250.0,
                                        n_sections: int = 10) -> VirtualSensorNetwork:
        """
        创建标准隧道传感器网络

        包含压力、流量、温度、加速度等传感器
        """
        network = self.create_network(network_name)
        section_length = tunnel_length / n_sections

        for i in range(n_sections + 1):
            location = (i * section_length, 0.0, 0.0)

            # 压力传感器
            pressure_char = SensorCharacteristics(
                sensor_type="pressure",
                measurement_range=(0.0, 2.0e6),
                resolution=100,
                accuracy=0.1,
                response_time=0.005,
                sampling_rate=100,
                noise_model=NoiseModel(NoiseType.GAUSSIAN, amplitude=0.001),
                drift_model=DriftModel(DriftType.RANDOM_WALK, rate=0.01)
            )
            network.add_sensor(VirtualSensor(f"P_{i}", pressure_char, location))

            # 流量计 (关键位置)
            if i in [0, n_sections // 2, n_sections]:
                flow_char = SensorCharacteristics(
                    sensor_type="flow",
                    measurement_range=(0.0, 400.0),
                    resolution=0.1,
                    accuracy=0.5,
                    response_time=0.1,
                    sampling_rate=10,
                    noise_model=NoiseModel(NoiseType.GAUSSIAN, amplitude=0.002),
                    drift_model=DriftModel(DriftType.LINEAR, rate=0.02)
                )
                network.add_sensor(VirtualSensor(f"Q_{i}", flow_char, location))

            # 温度传感器
            temp_char = SensorCharacteristics(
                sensor_type="temperature",
                measurement_range=(-20.0, 100.0),
                resolution=0.01,
                accuracy=0.1,
                response_time=2.0,
                sampling_rate=1,
                noise_model=NoiseModel(NoiseType.GAUSSIAN, amplitude=0.0005),
                drift_model=DriftModel(DriftType.TEMPERATURE, rate=0.001)
            )
            network.add_sensor(VirtualSensor(f"T_{i}", temp_char, location))

            # MEMS加速度传感器 (每隔两个断面)
            if i % 2 == 0:
                accel_char = SensorCharacteristics(
                    sensor_type="accelerometer",
                    measurement_range=(-5.0, 5.0),
                    resolution=0.0001,
                    accuracy=0.01,
                    response_time=0.001,
                    sampling_rate=1000,
                    noise_model=NoiseModel(NoiseType.THERMAL, amplitude=0.0001),
                    drift_model=DriftModel(DriftType.RANDOM_WALK, rate=0.001)
                )
                network.add_sensor(VirtualSensor(f"MEMS_{i}", accel_char, location))

        return network

    def step(self, physical_states: Dict[str, Dict[str, float]], dt: float,
             environment: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, Any]]:
        """
        执行一步仿真

        Args:
            physical_states: 各网络的物理状态
            dt: 时间步长
            environment: 环境条件

        Returns:
            各网络的传感器数据
        """
        self._time += dt
        results = {}

        for network_name, network in self.networks.items():
            # 更新故障状态
            if network_name in self.injectors:
                self.injectors[network_name].update(self._time)

            # 生成数据
            state = physical_states.get(network_name, {})
            if network_name in self.generators:
                results[network_name] = self.generators[network_name].generate_step(
                    state, dt, environment
                )
            else:
                results[network_name] = {
                    'timestamp': self._time,
                    'readings': network.read_all(state, dt, environment)
                }

        return results

    def get_status(self) -> Dict[str, Any]:
        """获取仿真状态"""
        return {
            'time': self._time,
            'is_running': self._is_running,
            'networks': {
                name: network.get_network_status()
                for name, network in self.networks.items()
            }
        }

    def reset(self):
        """重置仿真"""
        self._time = 0.0
        for network in self.networks.values():
            network.reset_all()
        for generator in self.generators.values():
            generator.reset()
        for injector in self.injectors.values():
            injector.clear_all()
