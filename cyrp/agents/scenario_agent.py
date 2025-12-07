"""
Scenario Agent for CYRP Multi-Agent System.
穿黄工程场景智能体
"""

from typing import Dict, Any, Optional

from cyrp.agents.base_agent import Agent, AgentRole, Message, MessageType
from cyrp.scenarios import ScenarioManager, ScenarioType, ScenarioDomain


class ScenarioAgent(Agent):
    """
    场景智能体

    负责:
    1. 场景识别与分类
    2. 场景切换调度
    3. 场景生命周期管理
    4. 控制策略配置
    """

    def __init__(self, name: str = "ScenarioAgent"):
        super().__init__(name, AgentRole.SCENARIO, priority=4)

        # 场景管理器
        self.scenario_manager = ScenarioManager()

        # 注册回调
        self.scenario_manager.on_scenario_change = self._on_scenario_change
        self.scenario_manager.on_emergency = self._on_emergency

    def perceive(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """感知场景相关信息"""
        current_time = environment.get('time', 0.0)

        # 从其他智能体获取感知结果
        perception_features = self.get_knowledge('perception_features', {})
        classified_scenario = self.get_knowledge('classified_scenario')

        # 更新场景管理器
        update_info = self.scenario_manager.update(current_time)

        return {
            'current_scenario': self.scenario_manager.current_type,
            'perception_features': perception_features,
            'classified_scenario': classified_scenario,
            'transition_state': update_info['transition_state'],
            'fade_progress': update_info['fade_progress'],
            'time': current_time
        }

    def decide(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """决策 - 确定场景操作"""
        current_scenario = perception['current_scenario']
        classified_scenario = perception.get('classified_scenario')

        # 检查是否需要切换场景
        should_switch = False
        target_scenario = None

        if classified_scenario and classified_scenario != current_scenario:
            # 验证切换是否合理
            if self.scenario_manager.can_transition_to(classified_scenario):
                should_switch = True
                target_scenario = classified_scenario

        # 获取混合参数 (用于无扰切换)
        blended_params = self.scenario_manager.get_blended_parameters()

        return {
            'current_scenario': current_scenario,
            'should_switch': should_switch,
            'target_scenario': target_scenario,
            'blended_params': blended_params,
            'transition_state': perception['transition_state']
        }

    def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """执行 - 执行场景切换"""
        if decision['should_switch'] and decision['target_scenario']:
            success = self.scenario_manager.request_transition(
                decision['target_scenario']
            )
            if success:
                self.broadcast({
                    'event': 'scenario_transition_started',
                    'from': decision['current_scenario'].value,
                    'to': decision['target_scenario'].value
                }, MessageType.STATUS)
        else:
            success = True

        # 广播当前场景状态
        status = self.scenario_manager.get_status()
        self.broadcast({
            'scenario_status': status,
            'blended_params': decision['blended_params']
        }, MessageType.STATUS)

        # 更新知识库
        self.update_knowledge('current_scenario', decision['current_scenario'])

        return {
            'success': success,
            'scenario': decision['current_scenario'].value,
            'transition_state': decision['transition_state']
        }

    def _on_scenario_change(self, new_scenario: ScenarioType):
        """场景变更回调"""
        self.broadcast({
            'event': 'scenario_changed',
            'scenario': new_scenario.value
        }, MessageType.STATUS)

    def _on_emergency(self, scenario: ScenarioType):
        """应急场景回调"""
        self.alert({
            'type': 'emergency_scenario',
            'scenario': scenario.value
        }, priority=5)

    def _handle_message(self, message: Message):
        """处理消息"""
        if message.msg_type == MessageType.OBSERVATION:
            # 更新感知特征
            if 'features' in message.content:
                self.update_knowledge('perception_features', message.content['features'])
            if 'scenario' in message.content:
                try:
                    scenario = ScenarioType(message.content['scenario'])
                    self.update_knowledge('classified_scenario', scenario)
                except ValueError:
                    pass

        elif message.msg_type == MessageType.COMMAND:
            if message.content.get('request') == 'switch_scenario':
                target = message.content.get('scenario')
                if target:
                    try:
                        scenario = ScenarioType(target)
                        if message.content.get('reason') == 'safety_triggered':
                            self.scenario_manager.trigger_emergency(scenario)
                        else:
                            self.scenario_manager.request_transition(scenario)
                    except ValueError:
                        pass

    def trigger_emergency(self, scenario_type: ScenarioType) -> bool:
        """触发应急场景"""
        return self.scenario_manager.trigger_emergency(scenario_type)

    def get_current_scenario(self) -> ScenarioType:
        """获取当前场景"""
        return self.scenario_manager.current_type
