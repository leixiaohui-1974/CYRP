"""
传感器仿真模型 - Sensor Simulation Models

实现高保真传感器仿真：噪声、漂移、延迟、故障模式
Implements high-fidelity sensor simulation: noise, drift, delay, failure modes
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from abc import ABC, abstractmethod


class SensorStatus(Enum):
    """传感器状态"""
    NORMAL = "normal"
    DEGRADED = "degraded"
    FAILED = "failed"
    CALIBRATING = "calibrating"
    MAINTENANCE = "maintenance"


class FailureMode(Enum):
    """故障模式"""
    NONE = "none"
    STUCK = "stuck"                     # 卡死
    DRIFT = "drift"                     # 漂移
    BIAS = "bias"                       # 偏差
    NOISE_INCREASE = "noise_increase"   # 噪声增大
    SPIKE = "spike"                     # 尖峰
    DROPOUT = "dropout"                 # 信号丢失
    DELAY = "delay"                     # 延迟增大
    SATURATION = "saturation"          # 饱和
    DEAD_BAND = "dead_band"            # 死区


@dataclass
class SensorSpec:
    """传感器规格"""
    name: str
    measurement_range: Tuple[float, float]
    accuracy: float                    # 精度 (% of range)
    resolution: float                  # 分辨率
    response_time: float              # 响应时间 (s)
    sampling_rate: float              # 采样率 (Hz)
    noise_density: float              # 噪声密度
    drift_rate: float                 # 漂移率 (/h)
    temperature_coefficient: float    # 温度系数
    mtbf: float                       # 平均故障间隔 (h)


class SensorModel(ABC):
    """传感器基类"""

    def __init__(self, spec: SensorSpec, location: float = 0.0):
        self.spec = spec
        self.location = location

        # 状态
        self.status = SensorStatus.NORMAL
        self.failure_mode = FailureMode.NONE
        self.last_value = 0.0
        self.output_value = 0.0

        # 动态特性
        self.delay_buffer: deque = deque(maxlen=100)
        self.internal_state = 0.0

        # 误差模型
        self.bias = 0.0
        self.drift = 0.0
        self.noise_level = spec.noise_density

        # 时间
        self.time = 0.0
        self.dt = 1.0 / spec.sampling_rate

        # 故障注入
        self.failure_params: Dict[str, Any] = {}

    @abstractmethod
    def _measure(self, true_value: float) -> float:
        """获取真实测量值 (子类实现)"""
        pass

    def read(self, true_value: float, dt: float) -> float:
        """
        读取传感器值

        Args:
            true_value: 被测量的真实值
            dt: 时间步长

        Returns:
            传感器输出值
        """
        self.time += dt

        # 检查状态
        if self.status == SensorStatus.FAILED:
            return self._apply_failure_mode(true_value)

        # 基本测量
        measured = self._measure(true_value)

        # 动态响应 (一阶滞后)
        tau = self.spec.response_time
        alpha = dt / (tau + dt)
        self.internal_state = alpha * measured + (1 - alpha) * self.internal_state

        # 添加误差
        output = self._add_errors(self.internal_state, dt)

        # 量化
        output = self._quantize(output)

        # 范围限制
        output = np.clip(output, self.spec.measurement_range[0],
                        self.spec.measurement_range[1])

        # 延迟处理
        output = self._apply_delay(output)

        self.last_value = true_value
        self.output_value = output

        return output

    def _add_errors(self, value: float, dt: float) -> float:
        """添加误差"""
        # 偏差
        value += self.bias

        # 漂移 (随时间累积)
        self.drift += np.random.normal(0, self.spec.drift_rate * dt / 3600)
        value += self.drift

        # 随机噪声
        noise = np.random.normal(0, self.noise_level)
        value += noise

        return value

    def _quantize(self, value: float) -> float:
        """量化"""
        return round(value / self.spec.resolution) * self.spec.resolution

    def _apply_delay(self, value: float) -> float:
        """应用延迟"""
        delay_samples = int(self.spec.response_time / self.dt)
        self.delay_buffer.append(value)

        if len(self.delay_buffer) > delay_samples:
            return self.delay_buffer[0]
        return value

    def _apply_failure_mode(self, true_value: float) -> float:
        """应用故障模式"""
        if self.failure_mode == FailureMode.STUCK:
            return self.failure_params.get('stuck_value', self.last_value)

        elif self.failure_mode == FailureMode.DRIFT:
            drift_rate = self.failure_params.get('drift_rate', 0.01)
            self.drift += drift_rate * self.dt
            return true_value + self.drift

        elif self.failure_mode == FailureMode.BIAS:
            bias = self.failure_params.get('bias', 0.1)
            return true_value + bias

        elif self.failure_mode == FailureMode.NOISE_INCREASE:
            factor = self.failure_params.get('factor', 10)
            noise = np.random.normal(0, self.noise_level * factor)
            return true_value + noise

        elif self.failure_mode == FailureMode.SPIKE:
            if np.random.random() < self.failure_params.get('probability', 0.01):
                spike = self.failure_params.get('amplitude', 100)
                return true_value + spike * np.random.choice([-1, 1])
            return self.output_value

        elif self.failure_mode == FailureMode.DROPOUT:
            if np.random.random() < self.failure_params.get('probability', 0.1):
                return np.nan
            return true_value

        elif self.failure_mode == FailureMode.SATURATION:
            return self.spec.measurement_range[1]

        elif self.failure_mode == FailureMode.DEAD_BAND:
            dead_band = self.failure_params.get('width', 1.0)
            if abs(true_value - self.last_value) < dead_band:
                return self.output_value
            return true_value

        return true_value

    def inject_failure(self, mode: FailureMode, params: Optional[Dict[str, Any]] = None):
        """注入故障"""
        self.failure_mode = mode
        self.failure_params = params or {}
        if mode != FailureMode.NONE:
            self.status = SensorStatus.FAILED

    def clear_failure(self):
        """清除故障"""
        self.failure_mode = FailureMode.NONE
        self.failure_params = {}
        self.status = SensorStatus.NORMAL
        self.drift = 0.0

    def calibrate(self, reference_value: float):
        """校准"""
        self.bias = reference_value - self.output_value
        self.drift = 0.0
        self.status = SensorStatus.NORMAL

    def get_health(self) -> Dict[str, Any]:
        """获取健康状态"""
        return {
            'status': self.status.value,
            'failure_mode': self.failure_mode.value,
            'bias': self.bias,
            'drift': self.drift,
            'noise_level': self.noise_level,
            'output_value': self.output_value
        }


class PressureSensorModel(SensorModel):
    """压力传感器模型"""

    def __init__(self, location: float = 0.0, sensor_type: str = 'piezoresistive'):
        if sensor_type == 'piezoresistive':
            spec = SensorSpec(
                name="Pressure_Piezoresistive",
                measurement_range=(0, 2.0e6),   # 0-2 MPa
                accuracy=0.1,                    # 0.1%
                resolution=100,                  # 100 Pa
                response_time=0.005,             # 5 ms
                sampling_rate=100,               # 100 Hz
                noise_density=50,                # 50 Pa/√Hz
                drift_rate=0.01,                 # 0.01%/h
                temperature_coefficient=0.02,    # 0.02%/°C
                mtbf=50000                       # 50000 h
            )
        else:  # strain_gauge
            spec = SensorSpec(
                name="Pressure_StrainGauge",
                measurement_range=(0, 2.0e6),
                accuracy=0.05,
                resolution=50,
                response_time=0.001,
                sampling_rate=1000,
                noise_density=20,
                drift_rate=0.005,
                temperature_coefficient=0.01,
                mtbf=100000
            )
        super().__init__(spec, location)

        # 温度补偿
        self.temperature = 20.0  # °C
        self.reference_temperature = 20.0

    def _measure(self, true_value: float) -> float:
        """测量压力"""
        # 温度影响
        temp_effect = self.spec.temperature_coefficient * (self.temperature - self.reference_temperature)
        measured = true_value * (1 + temp_effect / 100)

        return measured

    def set_temperature(self, temperature: float):
        """设置环境温度"""
        self.temperature = temperature


class FlowMeterModel(SensorModel):
    """流量计模型"""

    def __init__(self, location: float = 0.0, meter_type: str = 'electromagnetic'):
        if meter_type == 'electromagnetic':
            spec = SensorSpec(
                name="FlowMeter_EM",
                measurement_range=(0, 400),      # 0-400 m³/s
                accuracy=0.5,                    # 0.5%
                resolution=0.1,                  # 0.1 m³/s
                response_time=0.1,               # 100 ms
                sampling_rate=10,                # 10 Hz
                noise_density=0.5,               # 0.5 m³/s/√Hz
                drift_rate=0.02,                 # 0.02%/h
                temperature_coefficient=0.01,
                mtbf=80000
            )
        elif meter_type == 'ultrasonic':
            spec = SensorSpec(
                name="FlowMeter_Ultrasonic",
                measurement_range=(0, 400),
                accuracy=1.0,
                resolution=0.5,
                response_time=0.2,
                sampling_rate=5,
                noise_density=1.0,
                drift_rate=0.05,
                temperature_coefficient=0.02,
                mtbf=60000
            )
        else:  # differential_pressure
            spec = SensorSpec(
                name="FlowMeter_DP",
                measurement_range=(0, 400),
                accuracy=2.0,
                resolution=1.0,
                response_time=0.5,
                sampling_rate=2,
                noise_density=2.0,
                drift_rate=0.1,
                temperature_coefficient=0.05,
                mtbf=40000
            )
        super().__init__(spec, location)

        # 流量计特性
        self.k_factor = 1.0
        self.reynolds_effect = True

    def _measure(self, true_value: float) -> float:
        """测量流量"""
        # K系数校正
        measured = true_value * self.k_factor

        # 雷诺数效应 (低流速时精度下降)
        if self.reynolds_effect and abs(true_value) < 10:
            error = np.random.normal(0, 0.02 * true_value)
            measured += error

        return measured


class TemperatureSensorModel(SensorModel):
    """温度传感器模型"""

    def __init__(self, location: float = 0.0, sensor_type: str = 'rtd'):
        if sensor_type == 'rtd':
            spec = SensorSpec(
                name="Temperature_RTD",
                measurement_range=(-50, 150),    # -50 to 150 °C
                accuracy=0.1,                    # 0.1°C
                resolution=0.01,                 # 0.01°C
                response_time=2.0,               # 2s (慢响应)
                sampling_rate=1,                 # 1 Hz
                noise_density=0.01,
                drift_rate=0.001,
                temperature_coefficient=0,
                mtbf=200000
            )
        else:  # thermocouple
            spec = SensorSpec(
                name="Temperature_TC",
                measurement_range=(-200, 500),
                accuracy=0.5,
                resolution=0.1,
                response_time=0.5,
                sampling_rate=10,
                noise_density=0.1,
                drift_rate=0.01,
                temperature_coefficient=0,
                mtbf=100000
            )
        super().__init__(spec, location)

        # 热惯性参数
        self.thermal_mass = 0.1  # 热容

    def _measure(self, true_value: float) -> float:
        """测量温度"""
        return true_value


class DASensorModel(SensorModel):
    """分布式声学传感器模型 (DAS)"""

    def __init__(self, fiber_length: float = 4250.0, spatial_resolution: float = 1.0):
        spec = SensorSpec(
            name="DAS_Fiber",
            measurement_range=(-1e-6, 1e-6),    # 应变范围
            accuracy=1.0,                        # 1 nε
            resolution=0.1e-9,                   # 0.1 nε
            response_time=0.001,                 # 1 ms
            sampling_rate=1000,                  # 1 kHz
            noise_density=0.5e-9,
            drift_rate=0.001,
            temperature_coefficient=0.05,
            mtbf=50000
        )
        super().__init__(spec)

        self.fiber_length = fiber_length
        self.spatial_resolution = spatial_resolution
        self.n_channels = int(fiber_length / spatial_resolution)

        # 光纤状态
        self.fiber_health = np.ones(self.n_channels)
        self.signal_to_noise = np.ones(self.n_channels) * 30  # dB

    def _measure(self, true_value: float) -> float:
        return true_value

    def read_distributed(self, strain_profile: np.ndarray) -> np.ndarray:
        """
        读取分布式应变

        Args:
            strain_profile: 真实应变分布

        Returns:
            测量的应变分布
        """
        n = len(strain_profile)
        measured = np.zeros(n)

        for i in range(n):
            if self.fiber_health[i % self.n_channels] > 0.5:
                # 正常测量
                measured[i] = strain_profile[i]

                # 添加噪声 (与SNR相关)
                snr = self.signal_to_noise[i % self.n_channels]
                noise_power = 10 ** (-snr / 10)
                measured[i] += np.random.normal(0, np.sqrt(noise_power) * 1e-9)
            else:
                # 光纤损坏
                measured[i] = np.nan

        return measured

    def inject_fiber_break(self, location: float):
        """注入光纤断裂"""
        idx = int(location / self.spatial_resolution)
        # 断点之后的所有通道失效
        self.fiber_health[idx:] = 0


class DTSensorModel(SensorModel):
    """分布式温度传感器模型 (DTS)"""

    def __init__(self, fiber_length: float = 4250.0, spatial_resolution: float = 1.0):
        spec = SensorSpec(
            name="DTS_Fiber",
            measurement_range=(-40, 300),       # °C
            accuracy=0.5,                        # 0.5°C
            resolution=0.1,                      # 0.1°C
            response_time=60.0,                  # 60s (积分时间)
            sampling_rate=0.017,                 # 1/60 Hz
            noise_density=0.1,
            drift_rate=0.01,
            temperature_coefficient=0,
            mtbf=50000
        )
        super().__init__(spec)

        self.fiber_length = fiber_length
        self.spatial_resolution = spatial_resolution
        self.n_channels = int(fiber_length / spatial_resolution)

        # 积分时间影响
        self.integration_time = 60.0  # s
        self.accumulated_samples = 0

    def _measure(self, true_value: float) -> float:
        return true_value

    def read_distributed(self, temperature_profile: np.ndarray) -> np.ndarray:
        """读取分布式温度"""
        n = len(temperature_profile)
        measured = np.zeros(n)

        # 根据积分时间计算噪声
        noise_reduction = np.sqrt(self.integration_time)

        for i in range(n):
            measured[i] = temperature_profile[i]
            measured[i] += np.random.normal(0, self.spec.noise_density / noise_reduction)

        return measured


class MEMSSensorModel(SensorModel):
    """MEMS加速度传感器模型"""

    def __init__(self, location: float = 0.0, axis: str = 'z'):
        spec = SensorSpec(
            name=f"MEMS_Accel_{axis}",
            measurement_range=(-5, 5),          # ±5 g
            accuracy=0.01,                       # 0.01 g
            resolution=0.0001,                   # 0.1 mg
            response_time=0.001,                 # 1 ms
            sampling_rate=1000,                  # 1 kHz
            noise_density=0.0001,                # 100 μg/√Hz
            drift_rate=0.001,                    # 0.001 g/h
            temperature_coefficient=0.1,         # 0.1%/°C
            mtbf=100000
        )
        super().__init__(spec, location)

        self.axis = axis

        # MEMS特性
        self.cross_axis_sensitivity = 0.02  # 2%
        self.nonlinearity = 0.001           # 0.1%

    def _measure(self, true_value: float) -> float:
        """测量加速度"""
        measured = true_value

        # 非线性
        measured += self.nonlinearity * true_value ** 2

        return measured

    def read_3axis(self, accel_x: float, accel_y: float, accel_z: float) -> Tuple[float, float, float]:
        """读取三轴加速度"""
        # 交叉轴灵敏度
        x = accel_x + self.cross_axis_sensitivity * (accel_y + accel_z)
        y = accel_y + self.cross_axis_sensitivity * (accel_x + accel_z)
        z = accel_z + self.cross_axis_sensitivity * (accel_x + accel_y)

        return self.read(x, self.dt), self.read(y, self.dt), self.read(z, self.dt)


class SensorArray:
    """传感器阵列"""

    def __init__(self):
        self.sensors: Dict[str, SensorModel] = {}
        self.sensor_locations: Dict[str, float] = {}

    def add_sensor(self, sensor_id: str, sensor: SensorModel):
        """添加传感器"""
        self.sensors[sensor_id] = sensor
        self.sensor_locations[sensor_id] = sensor.location

    def read_all(self, true_values: Dict[str, float], dt: float) -> Dict[str, float]:
        """读取所有传感器"""
        readings = {}
        for sensor_id, sensor in self.sensors.items():
            if sensor_id in true_values:
                readings[sensor_id] = sensor.read(true_values[sensor_id], dt)
            else:
                readings[sensor_id] = np.nan
        return readings

    def get_health_report(self) -> Dict[str, Dict[str, Any]]:
        """获取健康报告"""
        return {sensor_id: sensor.get_health()
                for sensor_id, sensor in self.sensors.items()}

    def inject_random_failures(self, failure_probability: float = 0.01):
        """随机注入故障"""
        for sensor_id, sensor in self.sensors.items():
            if np.random.random() < failure_probability:
                mode = np.random.choice(list(FailureMode))
                sensor.inject_failure(mode)

    def calibrate_all(self, reference_values: Dict[str, float]):
        """校准所有传感器"""
        for sensor_id, sensor in self.sensors.items():
            if sensor_id in reference_values:
                sensor.calibrate(reference_values[sensor_id])


class InstrumentationSystem:
    """完整仪表系统"""

    def __init__(self, tunnel_length: float = 4250.0, n_sections: int = 10):
        self.tunnel_length = tunnel_length
        self.n_sections = n_sections
        self.section_length = tunnel_length / n_sections

        # 创建传感器阵列
        self.sensors = SensorArray()
        self._setup_sensors()

        # 分布式传感器
        self.das = DASensorModel(tunnel_length)
        self.dts = DTSensorModel(tunnel_length)

    def _setup_sensors(self):
        """设置传感器"""
        # 每个断面的传感器
        for i in range(self.n_sections + 1):
            location = i * self.section_length

            # 压力传感器
            self.sensors.add_sensor(
                f"P_{i}",
                PressureSensorModel(location)
            )

            # 流量计 (关键位置)
            if i in [0, self.n_sections // 2, self.n_sections]:
                self.sensors.add_sensor(
                    f"F_{i}",
                    FlowMeterModel(location)
                )

            # 温度传感器
            self.sensors.add_sensor(
                f"T_{i}",
                TemperatureSensorModel(location)
            )

            # MEMS传感器 (每隔两个断面)
            if i % 2 == 0:
                self.sensors.add_sensor(
                    f"MEMS_{i}",
                    MEMSSensorModel(location)
                )

    def read_all(self, physical_state: Dict[str, np.ndarray], dt: float) -> Dict[str, Any]:
        """读取所有传感器数据"""
        # 提取各断面的真实值
        true_values = {}
        for i in range(self.n_sections + 1):
            idx = int(i * len(physical_state.get('pressure', [])) / (self.n_sections + 1))
            idx = min(idx, len(physical_state.get('pressure', [1])) - 1)

            if 'pressure' in physical_state and len(physical_state['pressure']) > 0:
                true_values[f"P_{i}"] = physical_state['pressure'][idx]
            if 'flow_rate' in physical_state and len(physical_state['flow_rate']) > 0:
                true_values[f"F_{i}"] = physical_state['flow_rate'][idx]
            if 'water_temperature' in physical_state and len(physical_state['water_temperature']) > 0:
                true_values[f"T_{i}"] = physical_state['water_temperature'][idx] - 273.15  # K to °C
            if 'acceleration' in physical_state:
                true_values[f"MEMS_{i}"] = physical_state['acceleration']

        # 读取点式传感器
        point_readings = self.sensors.read_all(true_values, dt)

        # 读取分布式传感器
        das_readings = None
        dts_readings = None

        if 'strain' in physical_state:
            das_readings = self.das.read_distributed(physical_state['strain'])
        if 'water_temperature' in physical_state:
            dts_readings = self.dts.read_distributed(physical_state['water_temperature'] - 273.15)

        return {
            'point_sensors': point_readings,
            'das': das_readings,
            'dts': dts_readings,
            'timestamp': dt,
            'health': self.sensors.get_health_report()
        }
