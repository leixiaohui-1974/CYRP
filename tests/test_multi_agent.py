"""
Tests for Multi-Agent System.
多智能体系统测试
"""

import pytest
import numpy as np
from cyrp.agents import MultiAgentSystem
from cyrp.agents.base_agent import AgentState, Message, MessageType
from cyrp.core import PhysicalSystem


class TestMultiAgentSystem:
    """多智能体系统测试类"""

    def setup_method(self):
        """测试前设置"""
        self.mas = MultiAgentSystem()

    def test_initialization(self):
        """测试初始化"""
        assert 'coordinator' in self.mas.agents
        assert 'perception' in self.mas.agents
        assert 'control' in self.mas.agents
        assert 'safety' in self.mas.agents
        assert 'scenario' in self.mas.agents

    def test_step(self):
        """测试单步执行"""
        # 创建物理系统
        physical_system = PhysicalSystem()
        physical_system.reset()

        environment = {
            'system_state': physical_system.state,
            'sensor_data': {},
            'time': 0.0,
            'dt': 0.1
        }

        result = self.mas.step(environment)

        assert 'coordinator' in result
        assert 'agents' in result
        assert 'cycle_time' in result

    def test_get_control_output(self):
        """测试获取控制输出"""
        control = self.mas.get_control_output()

        assert 'gate_1' in control
        assert 'gate_2' in control
        assert 0 <= control['gate_1'] <= 1
        assert 0 <= control['gate_2'] <= 1

    def test_get_scenario(self):
        """测试获取场景"""
        scenario = self.mas.get_scenario()
        assert isinstance(scenario, str)

    def test_is_safe(self):
        """测试安全检查"""
        is_safe = self.mas.is_safe()
        assert isinstance(is_safe, bool)

    def test_reset(self):
        """测试重置"""
        self.mas.reset()

        assert self.mas.cycle_count == 0
        for agent in self.mas.agents.values():
            assert agent.state == AgentState.IDLE


class TestAgentCommunication:
    """智能体通信测试类"""

    def test_message_creation(self):
        """测试消息创建"""
        msg = Message(
            msg_type=MessageType.COMMAND,
            sender="control",
            receiver="safety",
            content={'command': 'check'}
        )

        assert msg.sender == "control"
        assert msg.receiver == "safety"
        assert not msg.is_broadcast()

    def test_broadcast(self):
        """测试广播"""
        msg = Message(
            msg_type=MessageType.BROADCAST,
            sender="coordinator",
            receiver="*",
            content={'status': 'ok'}
        )

        assert msg.is_broadcast()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
