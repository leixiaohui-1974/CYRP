"""
Industrial Communication Module for CYRP
穿黄工程工业通信模块

支持协议:
- Modbus TCP/RTU
- OPC-UA
- IEC 61850
- IEC 60870-5-104
- WebSocket 实时推送
"""

from cyrp.communication.protocols import (
    # 基类和枚举
    ProtocolType,
    ConnectionState,
    DataQuality,
    DataPoint,
    TagDefinition,
    CommunicationProtocol,
    # Modbus
    ModbusConfig,
    ModbusTCPClient,
    ModbusFunctionCode,
    # OPC-UA
    OPCUAConfig,
    OPCUAClient,
    OPCUANodeId,
    # IEC 61850
    IEC61850Config,
    IEC61850Client,
    # IEC 104
    IEC104Config,
    IEC104Client,
    IEC104TypeId,
    # 管理器
    CommunicationManager,
    create_cyrp_communication_system,
)

from cyrp.communication.websocket_server import (
    MessageType,
    SubscriptionChannel,
    WebSocketMessage,
    WebSocketClient,
    WebSocketConnectionManager,
    RealtimePushService,
    EventBusWebSocketBridge,
    create_realtime_push_system,
)

__all__ = [
    # 工业协议
    "ProtocolType",
    "ConnectionState",
    "DataQuality",
    "DataPoint",
    "TagDefinition",
    "CommunicationProtocol",
    "ModbusConfig",
    "ModbusTCPClient",
    "ModbusFunctionCode",
    "OPCUAConfig",
    "OPCUAClient",
    "OPCUANodeId",
    "IEC61850Config",
    "IEC61850Client",
    "IEC104Config",
    "IEC104Client",
    "IEC104TypeId",
    "CommunicationManager",
    "create_cyrp_communication_system",
    # WebSocket 实时推送
    "MessageType",
    "SubscriptionChannel",
    "WebSocketMessage",
    "WebSocketClient",
    "WebSocketConnectionManager",
    "RealtimePushService",
    "EventBusWebSocketBridge",
    "create_realtime_push_system",
]
