"""
Tests for Industrial Communication Module.
工业通信模块测试
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
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
            tag_name="flow_rate",
            address="40001",
            data_type="FLOAT",
            description="进口流量"
        )

        assert tag.tag_name == "flow_rate"
        assert tag.address == "40001"
        assert tag.data_type == "FLOAT"


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
        self.client = ModbusTCPClient(self.config)

    def test_initialization(self):
        """测试初始化"""
        assert self.client.protocol_type == ProtocolType.MODBUS_TCP
        assert self.client.connection_state == ConnectionState.DISCONNECTED

    def test_build_mbap_header(self):
        """测试MBAP头构建"""
        header = self.client._build_mbap_header(6)

        assert len(header) == 7
        # 检查协议标识符(0x0000)
        assert header[2:4] == b'\x00\x00'

    def test_parse_mbap_header(self):
        """测试MBAP头解析"""
        header = b'\x00\x01\x00\x00\x00\x06\x01'
        transaction_id, protocol_id, length, unit_id = self.client._parse_mbap_header(header)

        assert transaction_id == 1
        assert protocol_id == 0
        assert length == 6
        assert unit_id == 1

    @pytest.mark.asyncio
    async def test_add_tag(self):
        """测试添加标签"""
        tag = TagDefinition(
            tag_name="test_tag",
            address="40001",
            data_type="INT16"
        )

        await self.client.add_tag(tag)

        assert "test_tag" in self.client._tags


class TestOPCUAClient:
    """OPC-UA客户端测试类"""

    def setup_method(self):
        """测试前设置"""
        self.config = OPCUAConfig(
            endpoint="opc.tcp://localhost:4840"
        )
        self.client = OPCUAClient(self.config)

    def test_initialization(self):
        """测试初始化"""
        assert self.client.protocol_type == ProtocolType.OPC_UA
        assert self.client.connection_state == ConnectionState.DISCONNECTED

    def test_parse_node_id(self):
        """测试NodeId解析"""
        # 测试数值节点ID
        node_id = self.client._parse_node_id("ns=2;i=1001")
        assert node_id.namespace == 2
        assert node_id.identifier == 1001
        assert node_id.id_type == "i"

        # 测试字符串节点ID
        node_id = self.client._parse_node_id("ns=1;s=Temperature")
        assert node_id.namespace == 1
        assert node_id.identifier == "Temperature"
        assert node_id.id_type == "s"

    @pytest.mark.asyncio
    async def test_add_tag(self):
        """测试添加标签"""
        tag = TagDefinition(
            tag_name="temperature",
            address="ns=2;i=1001",
            data_type="FLOAT"
        )

        await self.client.add_tag(tag)

        assert "temperature" in self.client._tags


class TestIEC104Client:
    """IEC 60870-5-104客户端测试类"""

    def setup_method(self):
        """测试前设置"""
        self.config = IEC104Config(
            host="192.168.1.100",
            port=2404
        )
        self.client = IEC104Client(self.config)

    def test_initialization(self):
        """测试初始化"""
        assert self.client.protocol_type == ProtocolType.IEC104
        assert self.client.connection_state == ConnectionState.DISCONNECTED

    def test_build_start_frame(self):
        """测试起始帧构建"""
        frame = self.client._build_start_frame()

        assert frame[0] == 0x68  # 起始字符
        assert frame[1] == 0x04  # APDU长度
        assert frame[2] == 0x07  # STARTDT act

    def test_build_stop_frame(self):
        """测试停止帧构建"""
        frame = self.client._build_stop_frame()

        assert frame[0] == 0x68
        assert frame[2] == 0x13  # STOPDT act


class TestCommunicationManager:
    """通信管理器测试类"""

    def setup_method(self):
        """测试前设置"""
        self.manager = CommunicationManager()

    def test_add_client(self):
        """测试添加客户端"""
        config = ModbusConfig(host="127.0.0.1", port=502)
        client = ModbusTCPClient(config)

        self.manager.add_client("modbus1", client)

        assert "modbus1" in self.manager._clients

    def test_get_client(self):
        """测试获取客户端"""
        config = ModbusConfig(host="127.0.0.1", port=502)
        client = ModbusTCPClient(config)
        self.manager.add_client("modbus1", client)

        retrieved = self.manager.get_client("modbus1")

        assert retrieved == client

    def test_remove_client(self):
        """测试移除客户端"""
        config = ModbusConfig(host="127.0.0.1", port=502)
        client = ModbusTCPClient(config)
        self.manager.add_client("modbus1", client)

        self.manager.remove_client("modbus1")

        assert "modbus1" not in self.manager._clients

    @pytest.mark.asyncio
    async def test_read_tag(self):
        """测试读取标签(模拟)"""
        config = ModbusConfig(host="127.0.0.1", port=502)
        client = ModbusTCPClient(config)
        self.manager.add_client("modbus1", client)

        # 添加标签
        tag = TagDefinition(
            tag_name="test_tag",
            address="40001",
            data_type="INT16"
        )
        await client.add_tag(tag)

        # 由于没有实际连接,读取会返回None或缓存值
        result = await self.manager.read_tag("test_tag")
        # 未连接时返回None
        assert result is None


class TestCreateCYRPCommunicationSystem:
    """测试创建穿黄工程通信系统"""

    def test_create_system(self):
        """测试创建系统"""
        manager = create_cyrp_communication_system()

        assert manager is not None
        assert isinstance(manager, CommunicationManager)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
