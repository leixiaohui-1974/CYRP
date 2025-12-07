"""
Sensor Models for CYRP Perception System.
穿黄工程传感器模型

包含:
- DAS: 分布式光纤声波传感 (泄漏检测、气爆监测)
- DTS: 分布式光纤测温 (渗漏定位)
- MEMS: 姿态/加速度传感 (地震响应、沉降)
- 传统传感器: 压力、流量、水质
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum
import numpy as np
from scipy.signal import butter, filtfilt, spectrogram
from scipy.fft import fft, fftfreq


class SensorStatus(Enum):
    """传感器状态"""
    NORMAL = "normal"
    DEGRADED = "degraded"
    FAULT = "fault"
    OFFLINE = "offline"


@dataclass
class SensorReading:
    """传感器读数基类"""
    timestamp: float
    value: Any
    quality: float = 1.0  # 数据质量 (0-1)
    status: SensorStatus = SensorStatus.NORMAL


@dataclass
class DASReading(SensorReading):
    """DAS传感器读数"""
    value: np.ndarray = field(default_factory=lambda: np.zeros(4250))  # 沿程声波信号
    leak_detected: bool = False
    leak_position: float = 0.0  # 泄漏位置 (m)
    leak_confidence: float = 0.0  # 置信度
    frequency_spectrum: np.ndarray = field(default_factory=lambda: np.zeros(100))


@dataclass
class DTSReading(SensorReading):
    """DTS传感器读数"""
    value: np.ndarray = field(default_factory=lambda: np.zeros(4250))  # 沿程温度
    anomaly_detected: bool = False
    anomaly_position: float = 0.0
    anomaly_type: str = ""  # "inner_leak", "outer_intrusion"


@dataclass
class MEMSReading(SensorReading):
    """MEMS传感器读数"""
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [ax, ay, az]
    angular_rate: np.ndarray = field(default_factory=lambda: np.zeros(3))  # [wx, wy, wz]
    tilt_angle: np.ndarray = field(default_factory=lambda: np.zeros(2))  # [pitch, roll]
    vibration_frequency: float = 0.0
    settlement: float = 0.0


class DASensor:
    """
    分布式光纤声波传感器 (Distributed Acoustic Sensing)

    基于瑞利散射原理，实现全长4.2km无死角监听
    功能:
    - 泄漏定位 (高压水泄漏的声发射信号 2-10kHz)
    - 气爆监测 (充水过程中气囊破裂)
    - 水锤检测 (压力波传播)
    """

    def __init__(
        self,
        length: float = 4250.0,
        spatial_resolution: float = 1.0,
        sampling_rate: float = 10000.0,
        noise_level: float = 0.01
    ):
        """
        初始化DAS传感器

        Args:
            length: 监测长度 (m)
            spatial_resolution: 空间分辨率 (m)
            sampling_rate: 采样率 (Hz)
            noise_level: 噪声水平 (0-1)
        """
        self.length = length
        self.dx = spatial_resolution
        self.fs = sampling_rate
        self.noise_level = noise_level

        self.num_points = int(length / spatial_resolution)
        self.x = np.linspace(0, length, self.num_points)

        # 滤波器参数
        self.filter_order = 4
        self.leak_freq_band = (2000, 10000)  # 泄漏频率带
        self.hammer_freq_band = (0.5, 100)  # 水锤频率带

        # 检测阈值
        self.leak_threshold = 0.3
        self.hammer_threshold = 0.5

        self.status = SensorStatus.NORMAL

    def _add_noise(self, signal: np.ndarray) -> np.ndarray:
        """添加高斯噪声"""
        noise = np.random.normal(0, self.noise_level, signal.shape)
        return signal + noise

    def _generate_leak_signal(
        self,
        position: float,
        intensity: float,
        duration: float = 0.1
    ) -> np.ndarray:
        """
        生成泄漏声发射信号

        Args:
            position: 泄漏位置 (m)
            intensity: 泄漏强度 (0-1)
            duration: 信号持续时间 (s)

        Returns:
            沿程信号分布
        """
        # 泄漏信号衰减模型 (指数衰减)
        distance = np.abs(self.x - position)
        attenuation = np.exp(-distance / 500)  # 衰减长度500m

        # 高频成分 (5kHz主频)
        t = np.arange(0, duration, 1 / self.fs)
        freq = 5000 + 1000 * np.random.randn()
        carrier = np.sin(2 * np.pi * freq * t)

        # 调制
        envelope = np.exp(-t / (duration / 3))
        signal_time = intensity * envelope * carrier

        # 空间分布
        signal = np.outer(attenuation, signal_time)

        return signal

    def measure(
        self,
        true_state: Dict[str, Any],
        duration: float = 0.1
    ) -> DASReading:
        """
        执行测量

        Args:
            true_state: 真实物理状态
            duration: 测量持续时间 (s)

        Returns:
            DAS读数
        """
        num_samples = int(duration * self.fs)
        signal = np.zeros((self.num_points, num_samples))

        # 背景噪声
        signal = self._add_noise(signal)

        leak_detected = False
        leak_position = 0.0
        leak_confidence = 0.0

        # 检查是否存在泄漏
        if 'leakage_rate' in true_state and true_state['leakage_rate'] > 0.01:
            leak_pos = true_state.get('leakage_location', 2000.0)
            leak_intensity = min(1.0, true_state['leakage_rate'] / 0.1)

            leak_signal = self._generate_leak_signal(leak_pos, leak_intensity, duration)
            signal += leak_signal

            leak_detected = True
            leak_position = leak_pos
            leak_confidence = 0.95

        # 检查水锤
        if 'water_hammer' in true_state and true_state['water_hammer'] > 0:
            # 水锤产生低频信号
            pass

        # 计算沿程RMS
        rms_profile = np.sqrt(np.mean(signal ** 2, axis=1))

        # 频谱分析
        avg_signal = np.mean(signal, axis=0)
        freq_spectrum = np.abs(fft(avg_signal))[:100]

        return DASReading(
            timestamp=true_state.get('time', 0.0),
            value=rms_profile,
            leak_detected=leak_detected,
            leak_position=leak_position,
            leak_confidence=leak_confidence,
            frequency_spectrum=freq_spectrum,
            status=self.status
        )

    def locate_leak(self, reading: DASReading) -> Tuple[bool, float, float]:
        """
        泄漏定位算法

        基于信号峰值定位，精度 ±5m

        Args:
            reading: DAS读数

        Returns:
            是否检测到泄漏, 位置 (m), 置信度
        """
        signal = reading.value

        # 归一化
        signal_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-10)

        # 寻找峰值
        max_idx = np.argmax(signal_norm)
        max_val = signal_norm[max_idx]

        if max_val > self.leak_threshold:
            position = self.x[max_idx]
            confidence = min(1.0, max_val / self.leak_threshold)
            return True, position, confidence

        return False, 0.0, 0.0


class DTSensor:
    """
    分布式光纤测温传感器 (Distributed Temperature Sensing)

    基于拉曼散射原理
    功能:
    - 渗漏类型判别 (内衬漏水 vs 外衬入侵)
    - 基于温度差异的泄漏定位
    """

    def __init__(
        self,
        length: float = 4250.0,
        spatial_resolution: float = 1.0,
        temperature_resolution: float = 0.1,
        noise_level: float = 0.05
    ):
        """
        初始化DTS传感器

        Args:
            length: 监测长度 (m)
            spatial_resolution: 空间分辨率 (m)
            temperature_resolution: 温度分辨率 (°C)
            noise_level: 噪声水平 (°C)
        """
        self.length = length
        self.dx = spatial_resolution
        self.temp_resolution = temperature_resolution
        self.noise_level = noise_level

        self.num_points = int(length / spatial_resolution)
        self.x = np.linspace(0, length, self.num_points)

        # 参考温度
        self.T_inner_water = 8.0  # 丹江口水温 (°C)
        self.T_yellow_river = 12.0  # 黄河水温 (°C)
        self.T_ground = 15.0  # 地温 (°C)

        self.status = SensorStatus.NORMAL

    def measure(self, true_state: Dict[str, Any]) -> DTSReading:
        """
        执行测量

        Args:
            true_state: 真实物理状态

        Returns:
            DTS读数
        """
        # 基础温度分布 (假设环形空腔温度接近地温)
        T_profile = np.ones(self.num_points) * self.T_ground

        # 添加噪声
        T_profile += np.random.normal(0, self.noise_level, self.num_points)

        anomaly_detected = False
        anomaly_position = 0.0
        anomaly_type = ""

        # 检查渗漏类型
        if 'leakage_rate' in true_state and true_state['leakage_rate'] > 0.01:
            leak_pos = true_state.get('leakage_location', 2000.0)
            leak_idx = int(leak_pos / self.dx)

            # 判断渗漏类型
            if true_state.get('leakage_source', 'inner') == 'inner':
                # 内衬渗漏：温度接近丹江口水温
                T_leak = self.T_inner_water
                anomaly_type = "inner_leak"
            else:
                # 外衬入侵：温度接近黄河水温
                T_leak = self.T_yellow_river
                anomaly_type = "outer_intrusion"

            # 温度扩散影响范围
            spread = 50  # m
            affected_range = np.arange(
                max(0, leak_idx - int(spread / self.dx)),
                min(self.num_points, leak_idx + int(spread / self.dx))
            )

            for idx in affected_range:
                dist = abs(idx - leak_idx) * self.dx
                influence = np.exp(-dist / 20)  # 扩散衰减
                T_profile[idx] = T_profile[idx] * (1 - influence) + T_leak * influence

            anomaly_detected = True
            anomaly_position = leak_pos

        return DTSReading(
            timestamp=true_state.get('time', 0.0),
            value=T_profile,
            anomaly_detected=anomaly_detected,
            anomaly_position=anomaly_position,
            anomaly_type=anomaly_type,
            status=self.status
        )

    def classify_leakage(self, reading: DTSReading) -> str:
        """
        分类渗漏类型

        基于温度特征判断是内衬渗漏还是外衬入侵

        Args:
            reading: DTS读数

        Returns:
            渗漏类型 ("inner_leak", "outer_intrusion", "none")
        """
        if not reading.anomaly_detected:
            return "none"

        # 获取异常点温度
        anomaly_idx = int(reading.anomaly_position / self.dx)
        T_anomaly = reading.value[anomaly_idx]

        # 温度判别
        diff_inner = abs(T_anomaly - self.T_inner_water)
        diff_outer = abs(T_anomaly - self.T_yellow_river)

        if diff_inner < diff_outer:
            return "inner_leak"
        else:
            return "outer_intrusion"


class MEMSSensor:
    """
    MEMS传感器阵列

    包含:
    - 加速度计 (振动监测)
    - 陀螺仪 (姿态检测)
    - 倾角计 (沉降/上浮)
    """

    def __init__(
        self,
        num_nodes: int = 20,
        sampling_rate: float = 100.0,
        noise_level: float = 0.001
    ):
        """
        初始化MEMS传感器阵列

        Args:
            num_nodes: 传感器节点数
            sampling_rate: 采样率 (Hz)
            noise_level: 噪声水平 (g)
        """
        self.num_nodes = num_nodes
        self.fs = sampling_rate
        self.noise_level = noise_level

        # 传感器位置 (均匀分布)
        self.positions = np.linspace(0, 4250, num_nodes)

        # 报警阈值
        self.vibration_threshold = 0.1  # g
        self.tilt_threshold = 0.005  # rad

        self.status = SensorStatus.NORMAL

    def measure(
        self,
        true_state: Dict[str, Any],
        duration: float = 0.1
    ) -> List[MEMSReading]:
        """
        执行测量

        Args:
            true_state: 真实物理状态
            duration: 测量持续时间 (s)

        Returns:
            各节点MEMS读数列表
        """
        readings = []

        for i, pos in enumerate(self.positions):
            # 基础读数
            acc = np.zeros(3)
            gyro = np.zeros(3)
            tilt = np.zeros(2)

            # 添加噪声
            acc += np.random.normal(0, self.noise_level, 3)
            gyro += np.random.normal(0, self.noise_level * 0.1, 3)
            tilt += np.random.normal(0, self.noise_level * 0.01, 2)

            # 地震响应
            if 'ground_acceleration' in true_state:
                pga = true_state['ground_acceleration']
                # 添加地震信号 (随机相位)
                phase = np.random.uniform(0, 2 * np.pi)
                acc[0] += pga * np.sin(phase)
                acc[1] += pga * np.cos(phase)
                acc[2] += pga * 0.5

            # 沉降/上浮
            settlement = true_state.get('settlement', 0.0)
            heave = true_state.get('heave', 0.0)

            # 倾斜
            if 'liquefaction_index' in true_state:
                liq = true_state['liquefaction_index']
                tilt[0] += liq * 0.01  # 液化引起倾斜

            # 振动频率分析
            vib_freq = 0.0
            if np.max(np.abs(acc)) > 0.01:
                vib_freq = 5.0 + np.random.uniform(-1, 1)  # 典型振动频率

            reading = MEMSReading(
                timestamp=true_state.get('time', 0.0),
                value=acc,
                acceleration=acc,
                angular_rate=gyro,
                tilt_angle=tilt,
                vibration_frequency=vib_freq,
                settlement=settlement + heave,
                status=self.status
            )
            readings.append(reading)

        return readings


class PressureSensor:
    """高频动态压力传感器"""

    def __init__(
        self,
        position: float,
        sampling_rate: float = 1000.0,
        range_max: float = 2.0e6,
        accuracy: float = 0.001
    ):
        self.position = position
        self.fs = sampling_rate
        self.range_max = range_max
        self.accuracy = accuracy
        self.status = SensorStatus.NORMAL

    def measure(self, true_pressure: float) -> SensorReading:
        """测量压力"""
        noise = np.random.normal(0, self.accuracy * self.range_max)
        measured = true_pressure + noise
        measured = np.clip(measured, 0, self.range_max)

        return SensorReading(
            timestamp=0.0,
            value=measured,
            quality=1.0 if abs(noise) < 3 * self.accuracy * self.range_max else 0.8,
            status=self.status
        )


class FlowMeter:
    """流量计"""

    def __init__(
        self,
        position: float,
        range_max: float = 300.0,
        accuracy: float = 0.005
    ):
        self.position = position
        self.range_max = range_max
        self.accuracy = accuracy
        self.status = SensorStatus.NORMAL

    def measure(self, true_flow: float) -> SensorReading:
        """测量流量"""
        noise = np.random.normal(0, self.accuracy * self.range_max)
        measured = true_flow + noise
        measured = max(0, measured)

        return SensorReading(
            timestamp=0.0,
            value=measured,
            quality=1.0,
            status=self.status
        )


class WaterQualitySensor:
    """水质传感器 (浊度、电导率)"""

    def __init__(self, position: float):
        self.position = position
        self.status = SensorStatus.NORMAL

    def measure(
        self,
        true_turbidity: float,
        true_conductivity: float
    ) -> Dict[str, SensorReading]:
        """测量水质"""
        return {
            'turbidity': SensorReading(
                timestamp=0.0,
                value=true_turbidity + np.random.normal(0, 1),
                quality=1.0,
                status=self.status
            ),
            'conductivity': SensorReading(
                timestamp=0.0,
                value=true_conductivity + np.random.normal(0, 10),
                quality=1.0,
                status=self.status
            )
        }
