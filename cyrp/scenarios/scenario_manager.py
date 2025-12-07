"""
Scenario Manager for CYRP.
穿黄工程场景管理器

实现场景的调度、切换和生命周期管理
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
import numpy as np
from collections import deque
import time

from cyrp.scenarios.scenario_definitions import (
    Scenario, ScenarioType, ScenarioDomain, ScenarioFamily,
    SCENARIO_REGISTRY
)


class TransitionState(Enum):
    """转换状态"""
    IDLE = "idle"  # 空闲
    PREPARING = "preparing"  # 准备中
    EXECUTING = "executing"  # 执行中
    STABILIZING = "stabilizing"  # 稳定中
    COMPLETED = "completed"  # 完成
    FAILED = "failed"  # 失败


@dataclass
class ScenarioTransition:
    """场景转换记录"""
    from_scenario: ScenarioType
    to_scenario: ScenarioType
    start_time: float
    end_time: Optional[float] = None
    state: TransitionState = TransitionState.IDLE
    fade_progress: float = 0.0  # 混合进度 (0-1)
    success: bool = False


class ScenarioManager:
    """
    场景管理器

    功能:
    1. 场景调度与切换
    2. 无扰切换 (Bumpless Transfer)
    3. 状态机管理
    4. 场景生命周期管理
    """

    def __init__(self):
        """初始化场景管理器"""
        # 当前场景
        self.current_scenario: Optional[Scenario] = None
        self.current_type: ScenarioType = ScenarioType.S1_A_DUAL_BALANCED

        # 目标场景 (用于转换)
        self.target_scenario: Optional[Scenario] = None
        self.target_type: Optional[ScenarioType] = None

        # 转换状态
        self.transition: Optional[ScenarioTransition] = None
        self.transition_state = TransitionState.IDLE

        # 转换参数
        self.fade_duration = 30.0  # 默认过渡时间 (s)
        self.min_scenario_duration = 10.0  # 最小场景持续时间

        # 场景开始时间
        self.scenario_start_time = 0.0

        # 历史记录
        self.history: deque = deque(maxlen=1000)

        # 转换矩阵
        self.transition_matrix = self._build_transition_matrix()

        # 回调函数
        self.on_scenario_change: Optional[Callable] = None
        self.on_emergency: Optional[Callable] = None

        # 加载默认场景
        self._load_scenario(ScenarioType.S1_A_DUAL_BALANCED)

    def _build_transition_matrix(self) -> Dict[ScenarioType, List[ScenarioType]]:
        """构建有效转换矩阵"""
        return {
            # 常态 -> 常态/过渡/应急
            ScenarioType.S1_A_DUAL_BALANCED: [
                ScenarioType.S1_B_DYNAMIC_PEAK,
                ScenarioType.S2_A_SEDIMENT_FLUSH,
                ScenarioType.S2_B_MUSSEL_CONTROL,
                ScenarioType.S3_A_FILLING,
                ScenarioType.S4_A_SWITCH_TUNNEL,
                # 应急场景总是可达
                ScenarioType.S5_A_INNER_LEAK,
                ScenarioType.S5_B_OUTER_INTRUSION,
                ScenarioType.S6_A_LIQUEFACTION,
                ScenarioType.S7_A_PIPE_BURST,
            ],
            ScenarioType.S1_B_DYNAMIC_PEAK: [
                ScenarioType.S1_A_DUAL_BALANCED,
                ScenarioType.S2_A_SEDIMENT_FLUSH,
                ScenarioType.S4_A_SWITCH_TUNNEL,
            ],
            # 过渡 -> 常态/过渡
            ScenarioType.S3_A_FILLING: [
                ScenarioType.S1_A_DUAL_BALANCED,
                ScenarioType.S4_A_SWITCH_TUNNEL,
            ],
            ScenarioType.S3_B_DRAINING: [
                ScenarioType.S4_B_ISOLATION,
            ],
            ScenarioType.S4_A_SWITCH_TUNNEL: [
                ScenarioType.S1_A_DUAL_BALANCED,
                ScenarioType.S3_A_FILLING,
                ScenarioType.S3_B_DRAINING,
                ScenarioType.S4_B_ISOLATION,
            ],
            # 应急 -> 恢复
            ScenarioType.S5_A_INNER_LEAK: [
                ScenarioType.S4_B_ISOLATION,
                ScenarioType.S7_A_PIPE_BURST,  # 恶化
            ],
            ScenarioType.S6_A_LIQUEFACTION: [
                ScenarioType.S1_A_DUAL_BALANCED,  # 恢复
            ],
        }

    def _load_scenario(self, scenario_type: ScenarioType) -> bool:
        """加载场景"""
        if scenario_type not in SCENARIO_REGISTRY:
            return False

        self.current_scenario = SCENARIO_REGISTRY[scenario_type]
        self.current_type = scenario_type
        self.scenario_start_time = time.time()

        return True

    def get_current_scenario(self) -> Optional[Scenario]:
        """获取当前场景"""
        return self.current_scenario

    def get_scenario_duration(self) -> float:
        """获取当前场景持续时间"""
        return time.time() - self.scenario_start_time

    def can_transition_to(self, target_type: ScenarioType) -> bool:
        """检查是否可以转换到目标场景"""
        # 应急场景总是可以触发
        target = SCENARIO_REGISTRY.get(target_type)
        if target and target.domain == ScenarioDomain.EMERGENCY:
            return True

        # 检查转换矩阵
        valid_targets = self.transition_matrix.get(self.current_type, [])
        return target_type in valid_targets

    def request_transition(
        self,
        target_type: ScenarioType,
        fade_duration: Optional[float] = None
    ) -> bool:
        """
        请求场景转换

        Args:
            target_type: 目标场景类型
            fade_duration: 过渡时间 (s)

        Returns:
            是否成功启动转换
        """
        # 检查是否正在转换
        if self.transition_state != TransitionState.IDLE:
            return False

        # 检查转换有效性
        if not self.can_transition_to(target_type):
            return False

        # 检查最小持续时间
        if self.get_scenario_duration() < self.min_scenario_duration:
            # 应急场景例外
            target = SCENARIO_REGISTRY.get(target_type)
            if not target or target.domain != ScenarioDomain.EMERGENCY:
                return False

        # 加载目标场景
        self.target_type = target_type
        self.target_scenario = SCENARIO_REGISTRY.get(target_type)

        if not self.target_scenario:
            return False

        # 创建转换记录
        self.transition = ScenarioTransition(
            from_scenario=self.current_type,
            to_scenario=target_type,
            start_time=time.time(),
            state=TransitionState.PREPARING
        )

        self.transition_state = TransitionState.PREPARING
        self.fade_duration = fade_duration or self._get_default_fade_duration(target_type)

        return True

    def _get_default_fade_duration(self, target_type: ScenarioType) -> float:
        """获取默认过渡时间"""
        target = SCENARIO_REGISTRY.get(target_type)
        if not target:
            return 30.0

        # 应急场景快速切换
        if target.domain == ScenarioDomain.EMERGENCY:
            return 5.0

        # 过渡场景中等速度
        if target.domain == ScenarioDomain.TRANSITION:
            return 60.0

        return 30.0

    def update(self, current_time: float) -> Dict[str, Any]:
        """
        更新场景管理器

        Args:
            current_time: 当前时间

        Returns:
            更新信息
        """
        result = {
            'scenario': self.current_type.value,
            'transition_state': self.transition_state.value,
            'fade_progress': 0.0
        }

        if self.transition_state == TransitionState.IDLE:
            return result

        if self.transition is None:
            return result

        # 计算过渡进度
        elapsed = current_time - self.transition.start_time

        if self.transition_state == TransitionState.PREPARING:
            # 准备阶段 (前10%)
            if elapsed >= self.fade_duration * 0.1:
                self.transition_state = TransitionState.EXECUTING
                self.transition.state = TransitionState.EXECUTING

        elif self.transition_state == TransitionState.EXECUTING:
            # 执行阶段 (10%-90%)
            progress = (elapsed - self.fade_duration * 0.1) / (self.fade_duration * 0.8)
            progress = np.clip(progress, 0, 1)

            # Sigmoid平滑函数
            self.transition.fade_progress = 1 / (1 + np.exp(-10 * (progress - 0.5)))
            result['fade_progress'] = self.transition.fade_progress

            if progress >= 1.0:
                self.transition_state = TransitionState.STABILIZING
                self.transition.state = TransitionState.STABILIZING

        elif self.transition_state == TransitionState.STABILIZING:
            # 稳定阶段 (后10%)
            stabilize_progress = (elapsed - self.fade_duration * 0.9) / (self.fade_duration * 0.1)
            if stabilize_progress >= 1.0:
                self._complete_transition()

        result['transition_state'] = self.transition_state.value
        return result

    def _complete_transition(self):
        """完成转换"""
        if self.transition and self.target_scenario:
            # 记录历史
            self.transition.end_time = time.time()
            self.transition.state = TransitionState.COMPLETED
            self.transition.success = True
            self.history.append(self.transition)

            # 切换场景
            self.current_scenario = self.target_scenario
            self.current_type = self.target_type
            self.scenario_start_time = time.time()

            # 回调
            if self.on_scenario_change:
                self.on_scenario_change(self.current_type)

            # 检查是否是应急场景
            if self.current_scenario.domain == ScenarioDomain.EMERGENCY:
                if self.on_emergency:
                    self.on_emergency(self.current_type)

        # 重置转换状态
        self.transition = None
        self.target_scenario = None
        self.target_type = None
        self.transition_state = TransitionState.IDLE

    def get_blended_parameters(self) -> Dict[str, Any]:
        """
        获取混合参数 (用于无扰切换)

        在转换过程中，返回当前场景和目标场景参数的加权平均

        Returns:
            混合参数
        """
        if self.transition_state == TransitionState.IDLE or self.transition is None:
            # 无转换，返回当前场景参数
            if self.current_scenario:
                return {
                    'constraints': self.current_scenario.constraints,
                    'objective': self.current_scenario.objective,
                    'mpc_model_type': self.current_scenario.mpc_model_type
                }
            return {}

        # 计算混合权重
        w = self.transition.fade_progress

        # 混合约束
        if self.current_scenario and self.target_scenario:
            current_c = self.current_scenario.constraints
            target_c = self.target_scenario.constraints

            blended_constraints = {
                'Q_min': (1 - w) * current_c.Q_min + w * target_c.Q_min,
                'Q_max': (1 - w) * current_c.Q_max + w * target_c.Q_max,
                'P_max': (1 - w) * current_c.P_max + w * target_c.P_max,
                'dP_dt_max': (1 - w) * current_c.dP_dt_max + w * target_c.dP_dt_max,
            }

            # 目标取目标场景
            blended_objective = self.target_scenario.objective if w > 0.5 else self.current_scenario.objective

            # MPC模型类型
            mpc_type = self.target_scenario.mpc_model_type if w > 0.5 else self.current_scenario.mpc_model_type

            return {
                'constraints': blended_constraints,
                'objective': blended_objective,
                'mpc_model_type': mpc_type,
                'blend_weight': w
            }

        return {}

    def trigger_emergency(self, scenario_type: ScenarioType) -> bool:
        """
        触发应急场景

        应急场景可以立即中断当前场景

        Args:
            scenario_type: 应急场景类型

        Returns:
            是否成功触发
        """
        scenario = SCENARIO_REGISTRY.get(scenario_type)
        if not scenario or scenario.domain != ScenarioDomain.EMERGENCY:
            return False

        # 中断当前转换
        if self.transition_state != TransitionState.IDLE:
            if self.transition:
                self.transition.state = TransitionState.FAILED
                self.history.append(self.transition)
            self.transition = None
            self.transition_state = TransitionState.IDLE

        # 快速切换
        return self.request_transition(scenario_type, fade_duration=3.0)

    def recover_to_nominal(self) -> bool:
        """从应急状态恢复到常态"""
        if self.current_scenario and self.current_scenario.domain != ScenarioDomain.EMERGENCY:
            return False

        return self.request_transition(
            ScenarioType.S1_A_DUAL_BALANCED,
            fade_duration=60.0
        )

    def get_status(self) -> Dict[str, Any]:
        """获取状态摘要"""
        return {
            'current_scenario': self.current_type.value,
            'current_domain': self.current_scenario.domain.value if self.current_scenario else None,
            'scenario_duration': self.get_scenario_duration(),
            'transition_state': self.transition_state.value,
            'target_scenario': self.target_type.value if self.target_type else None,
            'fade_progress': self.transition.fade_progress if self.transition else 0.0,
            'history_length': len(self.history)
        }
