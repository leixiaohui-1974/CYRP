"""
Multi-Agent System for CYRP.
穿黄工程多智能体协同系统

实现全场景自主运行的智能体架构
"""

from cyrp.agents.base_agent import Agent, AgentRole, AgentState, Message
from cyrp.agents.perception_agent import PerceptionAgent
from cyrp.agents.control_agent import ControlAgent
from cyrp.agents.safety_agent import SafetyAgent
from cyrp.agents.scenario_agent import ScenarioAgent
from cyrp.agents.coordinator_agent import CoordinatorAgent
from cyrp.agents.multi_agent_system import MultiAgentSystem

__all__ = [
    "Agent",
    "AgentRole",
    "AgentState",
    "Message",
    "PerceptionAgent",
    "ControlAgent",
    "SafetyAgent",
    "ScenarioAgent",
    "CoordinatorAgent",
    "MultiAgentSystem",
]
