"""
Industrial Communication Protocols for CYRP
穿黄工程工业通信协议模块

支持的协议：
- OPC-UA (IEC 62541)
- Modbus TCP/RTU
- IEC 61850 (电力系统)
- DL/T 634.5104 (电力远动)
"""

import struct
import socket
import threading
import queue
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from enum import Enum, auto
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


# ============================================================================
# 通信协议基类
# ============================================================================

class ProtocolType(Enum):
    """协议类型"""
    OPC_UA = "opc_ua"
    MODBUS_TCP = "modbus_tcp"
    MODBUS_RTU = "modbus_rtu"
    IEC_61850 = "iec_61850"
    IEC_104 = "iec_104"


class ConnectionState(Enum):
    """连接状态"""
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    ERROR = auto()
    RECONNECTING = auto()


class DataQuality(Enum):
    """数据质量"""
    GOOD = 0
    UNCERTAIN = 1
    BAD = 2
    NOT_CONNECTED = 3


@dataclass
class DataPoint:
    """数据点"""
    tag_name: str
    value: Any
    timestamp: datetime
    quality: DataQuality = DataQuality.GOOD
    source: str = ""
    unit: str = ""

    def to_dict(self) -> Dict:
        return {
            'tag_name': self.tag_name,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'quality': self.quality.name,
            'source': self.source,
            'unit': self.unit
        }


@dataclass
class TagDefinition:
    """标签定义"""
    name: str
    address: str  # 协议特定地址
    data_type: str  # int16, int32, float32, bool, string
    access: str = "rw"  # r, w, rw
    description: str = ""
    unit: str = ""
    scale: float = 1.0
    offset: float = 0.0
    min_value: Optional[float] = None
    max_value: Optional[float] = None


class CommunicationProtocol(ABC):
    """通信协议抽象基类"""

    def __init__(self, name: str, protocol_type: ProtocolType):
        self.name = name
        self.protocol_type = protocol_type
        self.state = ConnectionState.DISCONNECTED
        self.tags: Dict[str, TagDefinition] = {}
        self.callbacks: Dict[str, List[Callable]] = {}
        self.last_error: Optional[str] = None
        self.statistics = {
            'messages_sent': 0,
            'messages_received': 0,
            'errors': 0,
            'reconnects': 0
        }

    @abstractmethod
    def connect(self) -> bool:
        """建立连接"""
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """断开连接"""
        pass

    @abstractmethod
    def read(self, tag_names: List[str]) -> Dict[str, DataPoint]:
        """读取数据"""
        pass

    @abstractmethod
    def write(self, tag_name: str, value: Any) -> bool:
        """写入数据"""
        pass

    def register_tag(self, tag: TagDefinition):
        """注册标签"""
        self.tags[tag.name] = tag

    def register_callback(self, tag_name: str, callback: Callable):
        """注册数据变化回调"""
        if tag_name not in self.callbacks:
            self.callbacks[tag_name] = []
        self.callbacks[tag_name].append(callback)

    def _notify_callbacks(self, tag_name: str, data_point: DataPoint):
        """通知回调"""
        if tag_name in self.callbacks:
            for callback in self.callbacks[tag_name]:
                try:
                    callback(data_point)
                except Exception as e:
                    logger.error(f"Callback error for {tag_name}: {e}")


# ============================================================================
# Modbus TCP 协议
# ============================================================================

class ModbusFunctionCode(Enum):
    """Modbus功能码"""
    READ_COILS = 0x01
    READ_DISCRETE_INPUTS = 0x02
    READ_HOLDING_REGISTERS = 0x03
    READ_INPUT_REGISTERS = 0x04
    WRITE_SINGLE_COIL = 0x05
    WRITE_SINGLE_REGISTER = 0x06
    WRITE_MULTIPLE_COILS = 0x0F
    WRITE_MULTIPLE_REGISTERS = 0x10


@dataclass
class ModbusConfig:
    """Modbus配置"""
    host: str = "127.0.0.1"
    port: int = 502
    unit_id: int = 1
    timeout: float = 3.0
    retry_count: int = 3
    retry_delay: float = 1.0


class ModbusTCPClient(CommunicationProtocol):
    """Modbus TCP客户端"""

    def __init__(self, name: str, config: ModbusConfig):
        super().__init__(name, ProtocolType.MODBUS_TCP)
        self.config = config
        self.socket: Optional[socket.socket] = None
        self.transaction_id = 0
        self.lock = threading.Lock()

    def connect(self) -> bool:
        """连接到Modbus服务器"""
        try:
            self.state = ConnectionState.CONNECTING
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.config.timeout)
            self.socket.connect((self.config.host, self.config.port))
            self.state = ConnectionState.CONNECTED
            logger.info(f"Modbus TCP connected to {self.config.host}:{self.config.port}")
            return True
        except Exception as e:
            self.state = ConnectionState.ERROR
            self.last_error = str(e)
            logger.error(f"Modbus TCP connection failed: {e}")
            return False

    def disconnect(self) -> bool:
        """断开连接"""
        try:
            if self.socket:
                self.socket.close()
                self.socket = None
            self.state = ConnectionState.DISCONNECTED
            logger.info("Modbus TCP disconnected")
            return True
        except Exception as e:
            self.last_error = str(e)
            return False

    def _build_request(self, function_code: int, start_address: int,
                       quantity: int) -> bytes:
        """构建Modbus请求"""
        self.transaction_id = (self.transaction_id + 1) % 65536

        # MBAP Header
        header = struct.pack('>HHHBB',
                            self.transaction_id,  # Transaction ID
                            0,                     # Protocol ID (0 for Modbus)
                            6,                     # Length
                            self.config.unit_id,   # Unit ID
                            function_code)         # Function Code

        # PDU
        pdu = struct.pack('>HH', start_address, quantity)

        return header + pdu

    def _parse_response(self, response: bytes, function_code: int) -> Optional[bytes]:
        """解析Modbus响应"""
        if len(response) < 9:
            return None

        # 解析MBAP Header
        trans_id, proto_id, length, unit_id, resp_fc = struct.unpack('>HHHBB', response[:8])

        # 检查错误
        if resp_fc & 0x80:
            error_code = response[8] if len(response) > 8 else 0
            logger.error(f"Modbus error: function={resp_fc & 0x7F}, code={error_code}")
            return None

        # 返回数据部分
        return response[8:]

    def read_holding_registers(self, start_address: int, quantity: int) -> Optional[List[int]]:
        """读取保持寄存器"""
        with self.lock:
            if self.state != ConnectionState.CONNECTED:
                return None

            try:
                request = self._build_request(
                    ModbusFunctionCode.READ_HOLDING_REGISTERS.value,
                    start_address, quantity
                )
                self.socket.send(request)
                self.statistics['messages_sent'] += 1

                response = self.socket.recv(256)
                self.statistics['messages_received'] += 1

                data = self._parse_response(response, ModbusFunctionCode.READ_HOLDING_REGISTERS.value)
                if data is None:
                    return None

                byte_count = data[0]
                registers = []
                for i in range(1, byte_count, 2):
                    reg_value = struct.unpack('>H', data[i:i+2])[0]
                    registers.append(reg_value)

                return registers

            except Exception as e:
                self.statistics['errors'] += 1
                self.last_error = str(e)
                logger.error(f"Modbus read error: {e}")
                return None

    def write_single_register(self, address: int, value: int) -> bool:
        """写入单个寄存器"""
        with self.lock:
            if self.state != ConnectionState.CONNECTED:
                return False

            try:
                self.transaction_id = (self.transaction_id + 1) % 65536

                # 构建写请求
                request = struct.pack('>HHHBBHH',
                                     self.transaction_id,
                                     0,  # Protocol ID
                                     6,  # Length
                                     self.config.unit_id,
                                     ModbusFunctionCode.WRITE_SINGLE_REGISTER.value,
                                     address,
                                     value)

                self.socket.send(request)
                self.statistics['messages_sent'] += 1

                response = self.socket.recv(256)
                self.statistics['messages_received'] += 1

                return len(response) >= 12

            except Exception as e:
                self.statistics['errors'] += 1
                self.last_error = str(e)
                return False

    def read(self, tag_names: List[str]) -> Dict[str, DataPoint]:
        """读取标签数据"""
        results = {}
        now = datetime.now()

        for tag_name in tag_names:
            if tag_name not in self.tags:
                continue

            tag = self.tags[tag_name]
            # 解析地址 (格式: HR:100 表示保持寄存器100)
            parts = tag.address.split(':')
            if len(parts) != 2:
                continue

            reg_type, address = parts[0], int(parts[1])

            if reg_type == 'HR':  # 保持寄存器
                regs = self.read_holding_registers(address, 1)
                if regs:
                    raw_value = regs[0]
                    # 应用缩放和偏移
                    value = raw_value * tag.scale + tag.offset
                    results[tag_name] = DataPoint(
                        tag_name=tag_name,
                        value=value,
                        timestamp=now,
                        quality=DataQuality.GOOD,
                        source=self.name,
                        unit=tag.unit
                    )
                else:
                    results[tag_name] = DataPoint(
                        tag_name=tag_name,
                        value=None,
                        timestamp=now,
                        quality=DataQuality.BAD,
                        source=self.name
                    )

        return results

    def write(self, tag_name: str, value: Any) -> bool:
        """写入标签数据"""
        if tag_name not in self.tags:
            return False

        tag = self.tags[tag_name]
        parts = tag.address.split(':')
        if len(parts) != 2:
            return False

        reg_type, address = parts[0], int(parts[1])

        # 逆向缩放
        raw_value = int((value - tag.offset) / tag.scale)

        if reg_type == 'HR':
            return self.write_single_register(address, raw_value)

        return False


# ============================================================================
# OPC-UA 协议
# ============================================================================

@dataclass
class OPCUAConfig:
    """OPC-UA配置"""
    endpoint: str = "opc.tcp://localhost:4840"
    security_policy: str = "None"  # None, Basic128Rsa15, Basic256, Basic256Sha256
    security_mode: str = "None"    # None, Sign, SignAndEncrypt
    username: Optional[str] = None
    password: Optional[str] = None
    certificate_path: Optional[str] = None
    private_key_path: Optional[str] = None
    application_name: str = "CYRP_OPC_Client"
    timeout: float = 5.0


class OPCUANodeId:
    """OPC-UA节点ID"""

    def __init__(self, namespace: int, identifier: Union[int, str]):
        self.namespace = namespace
        self.identifier = identifier

    def __str__(self):
        if isinstance(self.identifier, int):
            return f"ns={self.namespace};i={self.identifier}"
        else:
            return f"ns={self.namespace};s={self.identifier}"

    @classmethod
    def parse(cls, node_id_str: str) -> 'OPCUANodeId':
        """解析节点ID字符串"""
        parts = node_id_str.split(';')
        namespace = 0
        identifier: Union[int, str] = ""

        for part in parts:
            if part.startswith('ns='):
                namespace = int(part[3:])
            elif part.startswith('i='):
                identifier = int(part[2:])
            elif part.startswith('s='):
                identifier = part[2:]

        return cls(namespace, identifier)


class OPCUAClient(CommunicationProtocol):
    """OPC-UA客户端"""

    def __init__(self, name: str, config: OPCUAConfig):
        super().__init__(name, ProtocolType.OPC_UA)
        self.config = config
        self.session_id: Optional[str] = None
        self.subscriptions: Dict[str, int] = {}  # tag_name -> subscription_id
        self._polling_thread: Optional[threading.Thread] = None
        self._running = False

        # 模拟的节点值存储（实际实现需要真正的OPC-UA库）
        self._simulated_values: Dict[str, Any] = {}

    def connect(self) -> bool:
        """连接到OPC-UA服务器"""
        try:
            self.state = ConnectionState.CONNECTING

            # 注意：这是模拟实现
            # 实际项目需要使用 opcua-asyncio 或 python-opcua 库
            logger.info(f"OPC-UA connecting to {self.config.endpoint}")

            # 模拟连接过程
            self.session_id = f"session_{int(time.time())}"
            self.state = ConnectionState.CONNECTED

            # 启动轮询线程
            self._running = True
            self._polling_thread = threading.Thread(target=self._polling_loop, daemon=True)
            self._polling_thread.start()

            logger.info(f"OPC-UA connected, session: {self.session_id}")
            return True

        except Exception as e:
            self.state = ConnectionState.ERROR
            self.last_error = str(e)
            logger.error(f"OPC-UA connection failed: {e}")
            return False

    def disconnect(self) -> bool:
        """断开连接"""
        try:
            self._running = False
            if self._polling_thread:
                self._polling_thread.join(timeout=2.0)

            self.session_id = None
            self.state = ConnectionState.DISCONNECTED
            logger.info("OPC-UA disconnected")
            return True

        except Exception as e:
            self.last_error = str(e)
            return False

    def read(self, tag_names: List[str]) -> Dict[str, DataPoint]:
        """读取节点值"""
        results = {}
        now = datetime.now()

        for tag_name in tag_names:
            if tag_name not in self.tags:
                continue

            tag = self.tags[tag_name]

            try:
                # 模拟读取（实际需要调用OPC-UA读取服务）
                if tag_name in self._simulated_values:
                    value = self._simulated_values[tag_name]
                    quality = DataQuality.GOOD
                else:
                    # 生成模拟值
                    value = self._generate_simulated_value(tag)
                    self._simulated_values[tag_name] = value
                    quality = DataQuality.GOOD

                results[tag_name] = DataPoint(
                    tag_name=tag_name,
                    value=value,
                    timestamp=now,
                    quality=quality,
                    source=self.name,
                    unit=tag.unit
                )
                self.statistics['messages_received'] += 1

            except Exception as e:
                results[tag_name] = DataPoint(
                    tag_name=tag_name,
                    value=None,
                    timestamp=now,
                    quality=DataQuality.BAD,
                    source=self.name
                )
                self.statistics['errors'] += 1

        return results

    def write(self, tag_name: str, value: Any) -> bool:
        """写入节点值"""
        if tag_name not in self.tags:
            return False

        try:
            # 模拟写入
            self._simulated_values[tag_name] = value
            self.statistics['messages_sent'] += 1
            logger.debug(f"OPC-UA write: {tag_name} = {value}")
            return True

        except Exception as e:
            self.statistics['errors'] += 1
            self.last_error = str(e)
            return False

    def subscribe(self, tag_names: List[str], interval_ms: int = 1000) -> bool:
        """订阅数据变化"""
        for tag_name in tag_names:
            if tag_name in self.tags:
                self.subscriptions[tag_name] = interval_ms
        return True

    def _polling_loop(self):
        """轮询订阅的数据"""
        while self._running:
            try:
                for tag_name, interval in self.subscriptions.items():
                    if tag_name in self.tags:
                        result = self.read([tag_name])
                        if tag_name in result:
                            self._notify_callbacks(tag_name, result[tag_name])

                time.sleep(1.0)  # 基础轮询间隔

            except Exception as e:
                logger.error(f"OPC-UA polling error: {e}")

    def _generate_simulated_value(self, tag: TagDefinition) -> Any:
        """生成模拟值"""
        import random

        if tag.data_type == 'float32':
            base = (tag.min_value or 0) + (tag.max_value or 100) / 2
            return base + random.uniform(-10, 10)
        elif tag.data_type == 'int16' or tag.data_type == 'int32':
            return random.randint(int(tag.min_value or 0), int(tag.max_value or 100))
        elif tag.data_type == 'bool':
            return random.choice([True, False])
        else:
            return 0


# ============================================================================
# IEC 61850 协议 (电力系统)
# ============================================================================

@dataclass
class IEC61850Config:
    """IEC 61850配置"""
    host: str = "127.0.0.1"
    port: int = 102
    ap_title: str = "1.1.1.999.1"
    ae_qualifier: int = 12
    timeout: float = 5.0


class IEC61850Client(CommunicationProtocol):
    """IEC 61850 MMS客户端"""

    def __init__(self, name: str, config: IEC61850Config):
        super().__init__(name, ProtocolType.IEC_61850)
        self.config = config
        self.logical_devices: Dict[str, Dict] = {}
        self._connected = False

    def connect(self) -> bool:
        """连接到IED"""
        try:
            self.state = ConnectionState.CONNECTING
            logger.info(f"IEC 61850 connecting to {self.config.host}:{self.config.port}")

            # 模拟连接
            self._connected = True
            self.state = ConnectionState.CONNECTED

            # 模拟获取数据模型
            self._load_data_model()

            logger.info("IEC 61850 connected")
            return True

        except Exception as e:
            self.state = ConnectionState.ERROR
            self.last_error = str(e)
            return False

    def disconnect(self) -> bool:
        """断开连接"""
        self._connected = False
        self.state = ConnectionState.DISCONNECTED
        return True

    def _load_data_model(self):
        """加载IED数据模型"""
        # 模拟穿黄工程相关的逻辑节点
        self.logical_devices = {
            'CYRP_LD1': {
                'MMXU1': {  # 测量
                    'TotW': {'value': 0.0, 'unit': 'kW'},
                    'TotVAr': {'value': 0.0, 'unit': 'kVar'},
                    'Hz': {'value': 50.0, 'unit': 'Hz'},
                },
                'XCBR1': {  # 断路器
                    'Pos': {'stVal': True},
                    'BlkOpn': {'stVal': False},
                    'BlkCls': {'stVal': False},
                },
                'CSWI1': {  # 开关控制
                    'Pos': {'ctlVal': True},
                },
                'GGIO1': {  # 通用IO
                    'AnIn1': {'value': 0.0},  # 流量
                    'AnIn2': {'value': 0.0},  # 压力
                    'AnIn3': {'value': 0.0},  # 水位
                },
            }
        }

    def read(self, tag_names: List[str]) -> Dict[str, DataPoint]:
        """读取数据"""
        results = {}
        now = datetime.now()

        for tag_name in tag_names:
            # 解析IEC 61850地址 (格式: LD/LN.DO.DA)
            try:
                parts = tag_name.replace('/', '.').split('.')
                if len(parts) >= 3:
                    ld_name = parts[0]
                    ln_name = parts[1]
                    do_name = parts[2]

                    if ld_name in self.logical_devices:
                        ld = self.logical_devices[ld_name]
                        if ln_name in ld and do_name in ld[ln_name]:
                            do_data = ld[ln_name][do_name]
                            value = do_data.get('value', do_data.get('stVal', do_data.get('ctlVal')))

                            results[tag_name] = DataPoint(
                                tag_name=tag_name,
                                value=value,
                                timestamp=now,
                                quality=DataQuality.GOOD,
                                source=self.name
                            )
                            continue

                results[tag_name] = DataPoint(
                    tag_name=tag_name,
                    value=None,
                    timestamp=now,
                    quality=DataQuality.BAD,
                    source=self.name
                )

            except Exception as e:
                logger.error(f"IEC 61850 read error for {tag_name}: {e}")

        return results

    def write(self, tag_name: str, value: Any) -> bool:
        """写入控制命令"""
        try:
            parts = tag_name.replace('/', '.').split('.')
            if len(parts) >= 3:
                ld_name, ln_name, do_name = parts[0], parts[1], parts[2]
                if ld_name in self.logical_devices:
                    ld = self.logical_devices[ld_name]
                    if ln_name in ld and do_name in ld[ln_name]:
                        ld[ln_name][do_name]['ctlVal'] = value
                        return True
            return False

        except Exception as e:
            self.last_error = str(e)
            return False


# ============================================================================
# IEC 60870-5-104 协议 (电力远动)
# ============================================================================

class IEC104TypeId(Enum):
    """IEC 104 类型标识"""
    # 监视方向
    M_SP_NA_1 = 1    # 单点信息
    M_DP_NA_1 = 3    # 双点信息
    M_ME_NA_1 = 9    # 测量值，归一化值
    M_ME_NB_1 = 11   # 测量值，标度化值
    M_ME_NC_1 = 13   # 测量值，短浮点数
    M_IT_NA_1 = 15   # 累计量

    # 控制方向
    C_SC_NA_1 = 45   # 单点命令
    C_DC_NA_1 = 46   # 双点命令
    C_SE_NA_1 = 48   # 设定值命令，归一化值
    C_SE_NB_1 = 49   # 设定值命令，标度化值
    C_SE_NC_1 = 50   # 设定值命令，短浮点数


@dataclass
class IEC104Config:
    """IEC 104配置"""
    host: str = "127.0.0.1"
    port: int = 2404
    common_address: int = 1
    k: int = 12  # 发送序号模
    w: int = 8   # 接收确认阈值
    t1: float = 15.0  # 发送超时
    t2: float = 10.0  # 确认超时
    t3: float = 20.0  # 测试帧超时


class IEC104Client(CommunicationProtocol):
    """IEC 60870-5-104 客户端"""

    def __init__(self, name: str, config: IEC104Config):
        super().__init__(name, ProtocolType.IEC_104)
        self.config = config
        self.socket: Optional[socket.socket] = None
        self.send_sequence = 0
        self.recv_sequence = 0
        self._running = False
        self._recv_thread: Optional[threading.Thread] = None
        self._data_buffer: Dict[int, Any] = {}  # IOA -> value
        self.lock = threading.Lock()

    def connect(self) -> bool:
        """连接"""
        try:
            self.state = ConnectionState.CONNECTING
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.config.t1)
            self.socket.connect((self.config.host, self.config.port))

            # 发送STARTDT激活
            self._send_u_frame(0x07)  # STARTDT act

            self.state = ConnectionState.CONNECTED
            self._running = True

            # 启动接收线程
            self._recv_thread = threading.Thread(target=self._receive_loop, daemon=True)
            self._recv_thread.start()

            logger.info(f"IEC 104 connected to {self.config.host}:{self.config.port}")
            return True

        except Exception as e:
            self.state = ConnectionState.ERROR
            self.last_error = str(e)
            return False

    def disconnect(self) -> bool:
        """断开"""
        try:
            self._running = False
            if self.socket:
                # 发送STOPDT
                self._send_u_frame(0x13)  # STOPDT act
                self.socket.close()
            self.state = ConnectionState.DISCONNECTED
            return True
        except Exception as e:
            self.last_error = str(e)
            return False

    def _send_u_frame(self, control: int):
        """发送U格式帧"""
        frame = bytes([0x68, 0x04, control, 0x00, 0x00, 0x00])
        if self.socket:
            self.socket.send(frame)

    def _send_i_frame(self, asdu: bytes):
        """发送I格式帧"""
        with self.lock:
            control = struct.pack('<HH',
                                 (self.send_sequence << 1),
                                 (self.recv_sequence << 1))
            length = 4 + len(asdu)
            frame = bytes([0x68, length]) + control + asdu
            if self.socket:
                self.socket.send(frame)
            self.send_sequence = (self.send_sequence + 1) % 32768

    def _receive_loop(self):
        """接收循环"""
        while self._running and self.socket:
            try:
                self.socket.settimeout(1.0)
                header = self.socket.recv(2)
                if len(header) < 2:
                    continue

                if header[0] != 0x68:
                    continue

                length = header[1]
                data = self.socket.recv(length)
                if len(data) < length:
                    continue

                self._process_frame(data)

            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    logger.error(f"IEC 104 receive error: {e}")

    def _process_frame(self, data: bytes):
        """处理接收帧"""
        if len(data) < 4:
            return

        control = data[0]

        # U格式帧
        if control & 0x03 == 0x03:
            # 处理U帧（STARTDT/STOPDT确认等）
            pass

        # S格式帧
        elif control & 0x01 == 0x01:
            # 接收序号确认
            pass

        # I格式帧
        else:
            # 解析ASDU
            asdu = data[4:]
            self._process_asdu(asdu)

    def _process_asdu(self, asdu: bytes):
        """处理ASDU"""
        if len(asdu) < 6:
            return

        type_id = asdu[0]
        sq = (asdu[1] >> 7) & 0x01  # 序列标志
        num = asdu[1] & 0x7F        # 信息对象数量
        cot = asdu[2] & 0x3F        # 传送原因
        common_addr = struct.unpack('<H', asdu[4:6])[0]

        # 解析信息对象
        offset = 6
        for i in range(num):
            if offset + 3 > len(asdu):
                break

            ioa = struct.unpack('<I', asdu[offset:offset+3] + b'\x00')[0]
            offset += 3

            # 根据类型解析值
            if type_id == IEC104TypeId.M_ME_NC_1.value:  # 短浮点数
                if offset + 5 <= len(asdu):
                    value = struct.unpack('<f', asdu[offset:offset+4])[0]
                    quality = asdu[offset+4]
                    self._data_buffer[ioa] = value
                    offset += 5

            elif type_id == IEC104TypeId.M_SP_NA_1.value:  # 单点
                if offset + 1 <= len(asdu):
                    value = asdu[offset] & 0x01
                    self._data_buffer[ioa] = bool(value)
                    offset += 1

    def read(self, tag_names: List[str]) -> Dict[str, DataPoint]:
        """读取数据"""
        results = {}
        now = datetime.now()

        for tag_name in tag_names:
            if tag_name not in self.tags:
                continue

            tag = self.tags[tag_name]
            # 地址格式: IOA:类型，如 "100:float"
            parts = tag.address.split(':')
            if len(parts) < 1:
                continue

            ioa = int(parts[0])

            with self.lock:
                if ioa in self._data_buffer:
                    value = self._data_buffer[ioa]
                    results[tag_name] = DataPoint(
                        tag_name=tag_name,
                        value=value * tag.scale + tag.offset,
                        timestamp=now,
                        quality=DataQuality.GOOD,
                        source=self.name,
                        unit=tag.unit
                    )
                else:
                    results[tag_name] = DataPoint(
                        tag_name=tag_name,
                        value=None,
                        timestamp=now,
                        quality=DataQuality.NOT_CONNECTED,
                        source=self.name
                    )

        return results

    def write(self, tag_name: str, value: Any) -> bool:
        """写入控制命令"""
        if tag_name not in self.tags:
            return False

        tag = self.tags[tag_name]
        parts = tag.address.split(':')
        if len(parts) < 1:
            return False

        ioa = int(parts[0])
        data_type = parts[1] if len(parts) > 1 else 'float'

        try:
            # 构建ASDU
            if data_type == 'float':
                type_id = IEC104TypeId.C_SE_NC_1.value
                raw_value = (value - tag.offset) / tag.scale
                value_bytes = struct.pack('<f', raw_value) + bytes([0x00])
            else:
                type_id = IEC104TypeId.C_SC_NA_1.value
                value_bytes = bytes([0x01 if value else 0x00])

            asdu = bytes([
                type_id,
                0x01,  # 1个信息对象
                0x06,  # COT: 激活
                0x00,
            ]) + struct.pack('<H', self.config.common_address)

            asdu += struct.pack('<I', ioa)[:3]
            asdu += value_bytes

            self._send_i_frame(asdu)
            return True

        except Exception as e:
            self.last_error = str(e)
            return False


# ============================================================================
# 通信管理器
# ============================================================================

class CommunicationManager:
    """通信管理器"""

    def __init__(self):
        self.protocols: Dict[str, CommunicationProtocol] = {}
        self.tag_mapping: Dict[str, str] = {}  # tag_name -> protocol_name
        self._running = False
        self._update_thread: Optional[threading.Thread] = None
        self.data_cache: Dict[str, DataPoint] = {}
        self.update_interval = 1.0

    def add_protocol(self, protocol: CommunicationProtocol):
        """添加协议"""
        self.protocols[protocol.name] = protocol

    def register_tag(self, protocol_name: str, tag: TagDefinition):
        """注册标签"""
        if protocol_name in self.protocols:
            self.protocols[protocol_name].register_tag(tag)
            self.tag_mapping[tag.name] = protocol_name

    def connect_all(self) -> Dict[str, bool]:
        """连接所有协议"""
        results = {}
        for name, protocol in self.protocols.items():
            results[name] = protocol.connect()
        return results

    def disconnect_all(self):
        """断开所有连接"""
        for protocol in self.protocols.values():
            protocol.disconnect()

    def start_polling(self, interval: float = 1.0):
        """启动轮询"""
        self.update_interval = interval
        self._running = True
        self._update_thread = threading.Thread(target=self._polling_loop, daemon=True)
        self._update_thread.start()

    def stop_polling(self):
        """停止轮询"""
        self._running = False
        if self._update_thread:
            self._update_thread.join(timeout=2.0)

    def _polling_loop(self):
        """轮询循环"""
        while self._running:
            for protocol_name, protocol in self.protocols.items():
                if protocol.state == ConnectionState.CONNECTED:
                    try:
                        tags = list(protocol.tags.keys())
                        if tags:
                            data = protocol.read(tags)
                            self.data_cache.update(data)
                    except Exception as e:
                        logger.error(f"Polling error for {protocol_name}: {e}")

            time.sleep(self.update_interval)

    def read(self, tag_name: str) -> Optional[DataPoint]:
        """读取标签"""
        if tag_name in self.data_cache:
            return self.data_cache[tag_name]

        if tag_name in self.tag_mapping:
            protocol_name = self.tag_mapping[tag_name]
            protocol = self.protocols[protocol_name]
            result = protocol.read([tag_name])
            if tag_name in result:
                return result[tag_name]

        return None

    def write(self, tag_name: str, value: Any) -> bool:
        """写入标签"""
        if tag_name in self.tag_mapping:
            protocol_name = self.tag_mapping[tag_name]
            protocol = self.protocols[protocol_name]
            return protocol.write(tag_name, value)
        return False

    def get_statistics(self) -> Dict[str, Dict]:
        """获取统计信息"""
        return {name: p.statistics for name, p in self.protocols.items()}


# ============================================================================
# 便捷函数
# ============================================================================

def create_cyrp_communication_system() -> CommunicationManager:
    """创建穿黄工程通信系统"""
    manager = CommunicationManager()

    # Modbus TCP (现场PLC)
    modbus_config = ModbusConfig(host="192.168.1.10", port=502, unit_id=1)
    modbus = ModbusTCPClient("PLC_Main", modbus_config)

    # 注册标签
    modbus.register_tag(TagDefinition(
        name="inlet_flow", address="HR:100", data_type="float32",
        unit="m³/s", scale=0.1, description="进口流量"
    ))
    modbus.register_tag(TagDefinition(
        name="outlet_flow", address="HR:102", data_type="float32",
        unit="m³/s", scale=0.1, description="出口流量"
    ))
    modbus.register_tag(TagDefinition(
        name="inlet_pressure", address="HR:104", data_type="float32",
        unit="kPa", scale=1.0, description="进口压力"
    ))
    modbus.register_tag(TagDefinition(
        name="outlet_pressure", address="HR:106", data_type="float32",
        unit="kPa", scale=1.0, description="出口压力"
    ))
    modbus.register_tag(TagDefinition(
        name="inlet_valve_pos", address="HR:200", data_type="int16",
        scale=0.01, description="进口阀门开度"
    ))
    modbus.register_tag(TagDefinition(
        name="outlet_valve_pos", address="HR:202", data_type="int16",
        scale=0.01, description="出口阀门开度"
    ))

    manager.add_protocol(modbus)

    # OPC-UA (SCADA)
    opcua_config = OPCUAConfig(endpoint="opc.tcp://192.168.1.20:4840")
    opcua = OPCUAClient("SCADA_OPC", opcua_config)

    opcua.register_tag(TagDefinition(
        name="tunnel1_status", address="ns=2;s=Tunnel1.Status",
        data_type="int16", description="1号隧洞状态"
    ))
    opcua.register_tag(TagDefinition(
        name="tunnel2_status", address="ns=2;s=Tunnel2.Status",
        data_type="int16", description="2号隧洞状态"
    ))

    manager.add_protocol(opcua)

    # IEC 104 (调度)
    iec104_config = IEC104Config(host="192.168.1.30", port=2404)
    iec104 = IEC104Client("Dispatch_104", iec104_config)

    iec104.register_tag(TagDefinition(
        name="dispatch_flow_setpoint", address="1000:float",
        data_type="float32", unit="m³/s", description="调度流量设定值"
    ))
    iec104.register_tag(TagDefinition(
        name="dispatch_alarm", address="2000:bool",
        data_type="bool", description="调度告警"
    ))

    manager.add_protocol(iec104)

    return manager
