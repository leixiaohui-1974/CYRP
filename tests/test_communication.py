"""
Tests for Industrial Communication Module.
工业通信模块测试
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from cyrp.communication import (
    ProtocolType,
    ConnectionState,
    DataQuality,
    DataPoint,
    TagDefinition,
    ModbusConfig,
    ModbusTCPClient,
    ModbusFunctionCode,
    OPCUAConfig,
    OPCUAClient,
    OPCUANodeId,
    IEC61850Config,
    IEC61850Client,
    IEC104Config,
    IEC104Client,
    CommunicationManager,
    create_cyrp_communication_system,
)


class TestDataPoint:
    """数据点测试类"""

    def test_creation(self):
        """测试创建数据点"""
        point = DataPoint(
            tag_name="test_tag",
            value=123.45,
            quality=DataQuality.GOOD,
            timestamp=datetime.now()
        )

        assert point.tag_name == "test_tag"
        assert point.value == 123.45
        assert point.quality == DataQuality.GOOD

    def test_to_dict(self):
        """测试转换为字典"""
        point = DataPoint(
            tag_name="test_tag",
            value=100.0,
            quality=DataQuality.GOOD,
            timestamp=datetime.now()
        )

        data = point.to_dict()

        assert "tag_name" in data
        assert "value" in data
        assert "quality" in data
        assert data["value"] == 100.0


class TestTagDefinition:
    """标签定义测试类"""

    def test_creation(self):
        """测试创建标签定义"""
        tag = TagDefinition(
            name="flow_rate",
            address="HR:40001",
            data_type="float32",
            description="进口流量"
        )

        assert tag.name == "flow_rate"
        assert tag.address == "HR:40001"
        assert tag.data_type == "float32"


class TestModbusConfig:
    """Modbus配置测试类"""

    def test_default_config(self):
        """测试默认配置"""
        config = ModbusConfig(
            host="192.168.1.100",
            port=502
        )

        assert config.host == "192.168.1.100"
        assert config.port == 502
        assert config.unit_id == 1


class TestModbusTCPClient:
    """Modbus TCP客户端测试类"""

    def setup_method(self):
        """测试前设置"""
        self.config = ModbusConfig(host="127.0.0.1", port=502)
        self.client = ModbusTCPClient("test_modbus", self.config)

    def test_initialization(self):
        """测试初始化"""
        assert self.client.protocol_type == ProtocolType.MODBUS_TCP
        assert self.client.state == ConnectionState.DISCONNECTED

    def test_build_request(self):
        """测试请求构建"""
        request = self.client._build_request(0x03, 100, 10)

        # 检查请求长度 (7 MBAP + 1 FC + 4 PDU = 12)
        assert len(request) == 12
        # 检查协议标识符(0x0000)位于字节2-3
        assert request[2:4] == b'\x00\x00'

    def test_register_tag(self):
        """测试注册标签"""
        tag = TagDefinition(
            name="test_tag",
            address="HR:40001",
            data_type="int16"
        )

        self.client.register_tag(tag)

        assert "test_tag" in self.client.tags


class TestOPCUAClient:
    """OPC-UA客户端测试类"""

    def setup_method(self):
        """测试前设置"""
        self.config = OPCUAConfig(
            endpoint="opc.tcp://localhost:4840"
        )
        self.client = OPCUAClient("test_opcua", self.config)

    def test_initialization(self):
        """测试初始化"""
        assert self.client.protocol_type == ProtocolType.OPC_UA
        assert self.client.state == ConnectionState.DISCONNECTED

    def test_parse_node_id(self):
        """测试NodeId解析"""
        # 测试数值节点ID
        node_id = OPCUANodeId.parse("ns=2;i=1001")
        assert node_id.namespace == 2
        assert node_id.identifier == 1001

        # 测试字符串节点ID
        node_id = OPCUANodeId.parse("ns=1;s=Temperature")
        assert node_id.namespace == 1
        assert node_id.identifier == "Temperature"

    def test_register_tag(self):
        """测试注册标签"""
        tag = TagDefinition(
            name="temperature",
            address="ns=2;i=1001",
            data_type="float32"
        )

        self.client.register_tag(tag)

        assert "temperature" in self.client.tags


class TestIEC104Client:
    """IEC 60870-5-104客户端测试类"""

    def setup_method(self):
        """测试前设置"""
        self.config = IEC104Config(
            host="192.168.1.100",
            port=2404
        )
        self.client = IEC104Client("test_iec104", self.config)

    def test_initialization(self):
        """测试初始化"""
        assert self.client.protocol_type == ProtocolType.IEC_104
        assert self.client.state == ConnectionState.DISCONNECTED

    def test_send_u_frame(self):
        """测试U格式帧构建"""
        # 模拟socket来捕获发送的数据
        mock_socket = Mock()
        sent_data = []
        mock_socket.send = lambda x: sent_data.append(x)
        self.client.socket = mock_socket

        # 发送STARTDT帧
        self.client._send_u_frame(0x07)

        # 验证帧格式
        assert len(sent_data) == 1
        frame = sent_data[0]
        assert frame[0] == 0x68  # 起始字符
        assert frame[1] == 0x04  # APDU长度
        assert frame[2] == 0x07  # STARTDT act


class TestCommunicationManager:
    """通信管理器测试类"""

    def setup_method(self):
        """测试前设置"""
        self.manager = CommunicationManager()

    def test_add_protocol(self):
        """测试添加协议"""
        config = ModbusConfig(host="127.0.0.1", port=502)
        client = ModbusTCPClient("modbus1", config)

        self.manager.add_protocol(client)

        assert "modbus1" in self.manager.protocols

    def test_get_protocol(self):
        """测试获取协议"""
        config = ModbusConfig(host="127.0.0.1", port=502)
        client = ModbusTCPClient("modbus1", config)
        self.manager.add_protocol(client)

        retrieved = self.manager.protocols.get("modbus1")

        assert retrieved == client

    def test_register_tag(self):
        """测试注册标签"""
        config = ModbusConfig(host="127.0.0.1", port=502)
        client = ModbusTCPClient("modbus1", config)
        self.manager.add_protocol(client)

        tag = TagDefinition(
            name="test_tag",
            address="HR:40001",
            data_type="int16"
        )
        self.manager.register_tag("modbus1", tag)

        assert "test_tag" in client.tags
        assert "test_tag" in self.manager.tag_mapping

    def test_read_tag_not_connected(self):
        """测试读取标签(未连接)"""
        config = ModbusConfig(host="127.0.0.1", port=502)
        client = ModbusTCPClient("modbus1", config)
        self.manager.add_protocol(client)

        tag = TagDefinition(
            name="test_tag",
            address="HR:40001",
            data_type="int16"
        )
        self.manager.register_tag("modbus1", tag)

        # 由于没有实际连接,读取会返回None或BAD质量
        result = self.manager.read("test_tag")
        assert result is None or result.quality == DataQuality.BAD


class TestCreateCYRPCommunicationSystem:
    """测试创建穿黄工程通信系统"""

    def test_create_system(self):
        """测试创建系统"""
        manager = create_cyrp_communication_system()

        assert manager is not None
        assert isinstance(manager, CommunicationManager)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
