"""
Integrated Perception System for CYRP.
穿黄工程多模态感知系统集成

整合所有传感器、数据融合和场景分类
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

from cyrp.perception.sensors import (
    DASensor, DTSensor, MEMSSensor,
    PressureSensor, FlowMeter, WaterQualitySensor,
    DASReading, DTSReading, MEMSReading, SensorReading
)
from cyrp.perception.fusion import DataFusionEngine, FusedState
from cyrp.perception.classifier import ScenarioClassifier, ScenarioClassification


@dataclass
class PerceptionOutput:
    """感知系统输出"""
    timestamp: float = 0.0

    # 传感器原始数据
    das_reading: Optional[DASReading] = None
    dts_reading: Optional[DTSReading] = None
    mems_readings: List[MEMSReading] = field(default_factory=list)
    pressure_readings: Dict[str, float] = field(default_factory=dict)
    flow_readings: Dict[str, float] = field(default_factory=dict)

    # 融合状态
    fused_state: Optional[FusedState] = None

    # 场景分类
    scenario: Optional[ScenarioClassification] = None

    # 报警
    alarms: Dict[str, bool] = field(default_factory=dict)

    # 健康状态
    sensor_health: Dict[str, str] = field(default_factory=dict)


class PerceptionSystem:
    """
    穿黄工程多模态感知系统

    五维一体的立体感知网:
    1. 听觉系统: DAS (分布式光纤声波传感)
    2. 触觉系统: DTS + MEMS
    3. 视觉系统: CV (计算机视觉)
    4. 水力专用传感器
    5. 岩土专用传感器
    """

    def __init__(
        self,
        tunnel_length: float = 4250.0,
        num_pressure_sensors: int = 8,
        num_mems_nodes: int = 20
    ):
        """
        初始化感知系统

        Args:
            tunnel_length: 隧洞长度 (m)
            num_pressure_sensors: 压力传感器数量
            num_mems_nodes: MEMS节点数
        """
        self.tunnel_length = tunnel_length

        # 初始化传感器
        self._init_sensors(num_pressure_sensors, num_mems_nodes)

        # 初始化数据融合引擎
        self.fusion_engine = DataFusionEngine()

        # 初始化场景分类器
        self.classifier = ScenarioClassifier()

        # 采样周期
        self.sample_period = 0.1  # 100ms

        # 上一次更新时间
        self.last_update_time = 0.0

    def _init_sensors(
        self,
        num_pressure: int,
        num_mems: int
    ):
        """初始化传感器阵列"""
        # DAS 光纤
        self.das = DASensor(
            length=self.tunnel_length,
            spatial_resolution=1.0,
            sampling_rate=10000.0
        )

        # DTS 光纤
        self.dts = DTSensor(
            length=self.tunnel_length,
            spatial_resolution=1.0,
            temperature_resolution=0.1
        )

        # MEMS 阵列
        self.mems = MEMSSensor(
            num_nodes=num_mems,
            sampling_rate=100.0
        )

        # 压力传感器
        self.pressure_sensors = {}
        positions = [0, 500, 1000, 2000, 3000, 3500, 4000, 4250]
        for i, pos in enumerate(positions[:num_pressure]):
            self.pressure_sensors[f'P{i+1}'] = PressureSensor(
                position=pos,
                sampling_rate=1000.0
            )

        # 流量计
        self.flow_meters = {
            'FM_inlet_1': FlowMeter(position=0, range_max=200),
            'FM_inlet_2': FlowMeter(position=0, range_max=200),
            'FM_outlet_1': FlowMeter(position=self.tunnel_length, range_max=200),
            'FM_outlet_2': FlowMeter(position=self.tunnel_length, range_max=200),
        }

        # 水质传感器
        self.water_quality = {
            'WQ_inlet': WaterQualitySensor(position=0),
            'WQ_outlet': WaterQualitySensor(position=self.tunnel_length),
            'WQ_cavity': WaterQualitySensor(position=self.tunnel_length / 2),
        }

    def _build_true_state(
        self,
        system_state: Any
    ) -> Dict[str, Any]:
        """
        从系统状态构建真实状态字典

        Args:
            system_state: 物理系统状态

        Returns:
            真实状态字典
        """
        true_state = {
            'time': getattr(system_state, 'time', 0.0),
            'Q1': getattr(system_state.hydraulic, 'Q1', 132.5),
            'Q2': getattr(system_state.hydraulic, 'Q2', 132.5),
            'P_inlet': getattr(system_state.hydraulic, 'P_max', 6e5),
            'P_outlet': getattr(system_state.hydraulic, 'P_min', 5e5),
            'leakage_rate': getattr(system_state.structural, 'leakage_rate', 0),
            'leakage_location': getattr(system_state.structural, 'leakage_location', 0),
            'leakage_source': 'inner',  # 默认
            'settlement': getattr(system_state.structural, 'settlement', 0),
            'heave': getattr(system_state.structural, 'heave', 0),
            'liquefaction_index': getattr(system_state.structural, 'liquefaction_index', 0),
            'ground_acceleration': 0,  # 从扰动获取
        }

        # 检查报警获取地震信息
        if hasattr(system_state, 'alarms') and system_state.alarms.get('liquefaction', False):
            true_state['ground_acceleration'] = 0.15 * 9.81

        return true_state

    def _collect_sensor_data(
        self,
        true_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        采集所有传感器数据

        Args:
            true_state: 真实状态

        Returns:
            传感器数据字典
        """
        sensor_data = {}

        # DAS 测量
        das_reading = self.das.measure(true_state, duration=0.1)
        sensor_data['das'] = das_reading

        # DTS 测量
        dts_reading = self.dts.measure(true_state)
        sensor_data['dts'] = dts_reading

        # MEMS 测量
        mems_readings = self.mems.measure(true_state, duration=0.1)
        sensor_data['mems'] = mems_readings

        # 压力测量
        pressures = {}
        for name, sensor in self.pressure_sensors.items():
            # 简化: 使用平均压力
            p = (true_state['P_inlet'] + true_state['P_outlet']) / 2
            reading = sensor.measure(p)
            pressures[name] = reading.value
        sensor_data['pressure'] = pressures

        # 流量测量
        flows = {}
        flows['FM_inlet_1'] = self.flow_meters['FM_inlet_1'].measure(true_state['Q1']).value
        flows['FM_inlet_2'] = self.flow_meters['FM_inlet_2'].measure(true_state['Q2']).value
        flows['FM_outlet_1'] = self.flow_meters['FM_outlet_1'].measure(true_state['Q1']).value
        flows['FM_outlet_2'] = self.flow_meters['FM_outlet_2'].measure(true_state['Q2']).value
        sensor_data['flow'] = flows

        return sensor_data

    def _prepare_fusion_input(
        self,
        sensor_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        准备融合引擎输入

        Args:
            sensor_data: 传感器数据

        Returns:
            融合输入字典
        """
        fusion_input = {}

        # 流量
        flows = sensor_data['flow']
        fusion_input['flow_1'] = flows['FM_inlet_1']
        fusion_input['flow_2'] = flows['FM_inlet_2']

        # 压力
        fusion_input['pressure'] = list(sensor_data['pressure'].values())

        # 空腔水位 (从DTS推断)
        dts = sensor_data['dts']
        if dts.anomaly_detected:
            fusion_input['cavity_level'] = 1.0  # 有异常
        else:
            fusion_input['cavity_level'] = 0.0

        # DAS
        das = sensor_data['das']
        fusion_input['das'] = {
            'leak_position': das.leak_position,
            'leak_intensity': 1.0 if das.leak_detected else 0.0
        }

        # DTS
        fusion_input['dts'] = {
            'anomaly_magnitude': 5.0 if dts.anomaly_detected else 0.0
        }

        # MEMS (取第一个节点)
        if sensor_data['mems']:
            mems0 = sensor_data['mems'][0]
            fusion_input['mems'] = {
                'acceleration': mems0.acceleration.tolist(),
                'tilt': mems0.tilt_angle.tolist()
            }

        return fusion_input

    def _check_alarms(
        self,
        fused_state: FusedState,
        scenario: ScenarioClassification
    ) -> Dict[str, bool]:
        """检查报警条件"""
        alarms = {}

        # 渗漏报警
        alarms['leakage'] = fused_state.leak_detected

        # 压力异常
        alarms['pressure_high'] = fused_state.P_inlet > 8e5
        alarms['pressure_low'] = fused_state.P_outlet < 3e5

        # 结构报警
        alarms['settlement'] = abs(fused_state.settlement) > 0.05
        alarms['tilt'] = abs(fused_state.tilt) > 0.005

        # 振动报警
        alarms['vibration'] = fused_state.vibration_level > 0.1

        # 紧急场景报警
        from cyrp.perception.classifier import ScenarioDomain
        alarms['emergency'] = scenario.domain == ScenarioDomain.EMERGENCY

        return alarms

    def _check_sensor_health(self) -> Dict[str, str]:
        """检查传感器健康状态"""
        health = {}

        health['DAS'] = self.das.status.value
        health['DTS'] = self.dts.status.value
        health['MEMS'] = self.mems.status.value

        for name, sensor in self.pressure_sensors.items():
            health[name] = sensor.status.value

        for name, sensor in self.flow_meters.items():
            health[name] = sensor.status.value

        return health

    def update(
        self,
        system_state: Any,
        dt: float = 0.1
    ) -> PerceptionOutput:
        """
        更新感知系统

        Args:
            system_state: 物理系统状态
            dt: 时间步长

        Returns:
            感知输出
        """
        current_time = getattr(system_state, 'time', self.last_update_time + dt)

        # 1. 构建真实状态
        true_state = self._build_true_state(system_state)

        # 2. 采集传感器数据
        sensor_data = self._collect_sensor_data(true_state)

        # 3. 数据融合
        fusion_input = self._prepare_fusion_input(sensor_data)
        fused_state = self.fusion_engine.fuse(fusion_input, dt)

        # 4. 场景分类
        scenario = self.classifier.classify(fused_state, current_time)

        # 5. 报警检查
        alarms = self._check_alarms(fused_state, scenario)

        # 6. 传感器健康检查
        sensor_health = self._check_sensor_health()

        # 7. 构建输出
        output = PerceptionOutput(
            timestamp=current_time,
            das_reading=sensor_data['das'],
            dts_reading=sensor_data['dts'],
            mems_readings=sensor_data['mems'],
            pressure_readings=sensor_data['pressure'],
            flow_readings=sensor_data['flow'],
            fused_state=fused_state,
            scenario=scenario,
            alarms=alarms,
            sensor_health=sensor_health
        )

        self.last_update_time = current_time
        return output

    def get_scenario(self) -> ScenarioClassification:
        """获取当前场景分类"""
        return self.classifier.classify(
            self.fusion_engine._to_fused_state(),
            self.last_update_time
        )

    def get_fused_state(self) -> FusedState:
        """获取当前融合状态"""
        return self.fusion_engine._to_fused_state()

    def inject_sensor_fault(
        self,
        sensor_name: str,
        fault_type: str
    ):
        """
        注入传感器故障

        Args:
            sensor_name: 传感器名称
            fault_type: 故障类型 ('offline', 'degraded', 'bias')
        """
        from cyrp.perception.sensors import SensorStatus

        if sensor_name == 'DAS':
            if fault_type == 'offline':
                self.das.status = SensorStatus.OFFLINE
            elif fault_type == 'degraded':
                self.das.status = SensorStatus.DEGRADED
                self.das.noise_level *= 5
        elif sensor_name == 'DTS':
            if fault_type == 'offline':
                self.dts.status = SensorStatus.OFFLINE
        elif sensor_name in self.pressure_sensors:
            sensor = self.pressure_sensors[sensor_name]
            if fault_type == 'offline':
                sensor.status = SensorStatus.OFFLINE

    def reset(self):
        """重置感知系统"""
        self.fusion_engine = DataFusionEngine()
        self.classifier = ScenarioClassifier()
        self.last_update_time = 0.0

        # 重置传感器状态
        from cyrp.perception.sensors import SensorStatus
        self.das.status = SensorStatus.NORMAL
        self.dts.status = SensorStatus.NORMAL
        self.mems.status = SensorStatus.NORMAL
        for sensor in self.pressure_sensors.values():
            sensor.status = SensorStatus.NORMAL
        for sensor in self.flow_meters.values():
            sensor.status = SensorStatus.NORMAL
