"""
Fault Diagnosis Expert System for CYRP
穿黄工程故障诊断专家系统

功能：
- 基于规则的故障推理
- 故障树分析 (FTA)
- 贝叶斯网络诊断
- 案例推理 (CBR)
- 故障隔离与定位
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from datetime import datetime, timedelta
from enum import Enum, auto
from collections import defaultdict
import json
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# 故障定义
# ============================================================================

class FaultCategory(Enum):
    """故障类别"""
    HYDRAULIC = "hydraulic"         # 水力学故障
    STRUCTURAL = "structural"       # 结构故障
    MECHANICAL = "mechanical"       # 机械故障
    ELECTRICAL = "electrical"       # 电气故障
    INSTRUMENTATION = "instrumentation"  # 仪表故障
    CONTROL = "control"             # 控制故障
    ENVIRONMENTAL = "environmental" # 环境故障
    OPERATIONAL = "operational"     # 操作故障


class FaultSeverity(Enum):
    """故障严重性"""
    MINOR = 1       # 轻微
    MODERATE = 2    # 中等
    SERIOUS = 3     # 严重
    CRITICAL = 4    # 危急
    CATASTROPHIC = 5  # 灾难性


@dataclass
class Symptom:
    """症状"""
    symptom_id: str
    name: str
    description: str
    data_source: str            # 数据来源
    condition: str              # 条件表达式
    weight: float = 1.0         # 权重
    confidence: float = 1.0     # 置信度


@dataclass
class Fault:
    """故障"""
    fault_id: str
    name: str
    description: str
    category: FaultCategory
    severity: FaultSeverity
    symptoms: List[str]         # 症状ID列表
    causes: List[str]           # 可能原因
    effects: List[str]          # 可能后果
    remedies: List[str]         # 修复建议
    probability: float = 0.01   # 先验概率
    detection_time: float = 0.0  # 平均检测时间(秒)
    isolation_time: float = 0.0  # 平均隔离时间(秒)


@dataclass
class DiagnosisResult:
    """诊断结果"""
    fault_id: str
    fault_name: str
    probability: float          # 后验概率
    confidence: float           # 置信度
    matched_symptoms: List[str]
    unmatched_symptoms: List[str]
    evidence: Dict[str, Any]
    suggested_actions: List[str]
    estimated_severity: FaultSeverity
    diagnosis_time: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            'fault_id': self.fault_id,
            'fault_name': self.fault_name,
            'probability': self.probability,
            'confidence': self.confidence,
            'matched_symptoms': self.matched_symptoms,
            'suggested_actions': self.suggested_actions,
            'severity': self.estimated_severity.name,
            'diagnosis_time': self.diagnosis_time.isoformat()
        }


# ============================================================================
# 规则引擎
# ============================================================================

@dataclass
class Rule:
    """诊断规则"""
    rule_id: str
    name: str
    conditions: List[str]       # 条件列表 (AND关系)
    conclusion: str             # 结论 (故障ID)
    confidence: float = 1.0     # 规则置信度
    priority: int = 1           # 优先级
    enabled: bool = True


class RuleEngine:
    """规则引擎"""

    def __init__(self):
        self.rules: Dict[str, Rule] = {}
        self.facts: Dict[str, Any] = {}
        self._operators = {
            '>': lambda a, b: a > b,
            '>=': lambda a, b: a >= b,
            '<': lambda a, b: a < b,
            '<=': lambda a, b: a <= b,
            '==': lambda a, b: a == b,
            '!=': lambda a, b: a != b,
            'in': lambda a, b: a in b,
            'not_in': lambda a, b: a not in b,
        }

    def add_rule(self, rule: Rule):
        """添加规则"""
        self.rules[rule.rule_id] = rule

    def set_fact(self, name: str, value: Any):
        """设置事实"""
        self.facts[name] = value

    def set_facts(self, facts: Dict[str, Any]):
        """批量设置事实"""
        self.facts.update(facts)

    def clear_facts(self):
        """清除事实"""
        self.facts.clear()

    def evaluate(self) -> List[Tuple[str, float]]:
        """
        评估所有规则

        Returns:
            匹配的规则列表 [(fault_id, confidence), ...]
        """
        results = []

        # 按优先级排序
        sorted_rules = sorted(
            [r for r in self.rules.values() if r.enabled],
            key=lambda x: -x.priority
        )

        for rule in sorted_rules:
            if self._evaluate_rule(rule):
                results.append((rule.conclusion, rule.confidence))

        return results

    def _evaluate_rule(self, rule: Rule) -> bool:
        """评估单个规则"""
        for condition in rule.conditions:
            if not self._evaluate_condition(condition):
                return False
        return True

    def _evaluate_condition(self, condition: str) -> bool:
        """评估条件"""
        try:
            # 解析条件 (格式: "variable operator value")
            parts = condition.split()
            if len(parts) < 3:
                return False

            var_name = parts[0]
            operator = parts[1]
            value_str = ' '.join(parts[2:])

            if var_name not in self.facts:
                return False

            fact_value = self.facts[var_name]

            # 解析比较值
            try:
                compare_value = json.loads(value_str)
            except:
                compare_value = value_str

            # 执行比较
            if operator in self._operators:
                return self._operators[operator](fact_value, compare_value)

            return False

        except Exception as e:
            logger.error(f"Condition evaluation error: {condition}, {e}")
            return False


# ============================================================================
# 故障树分析
# ============================================================================

class FTANodeType(Enum):
    """故障树节点类型"""
    BASIC = "basic"         # 基本事件
    AND = "and"             # 与门
    OR = "or"               # 或门
    VOTE = "vote"           # 表决门
    INHIBIT = "inhibit"     # 抑制门


@dataclass
class FTANode:
    """故障树节点"""
    node_id: str
    name: str
    node_type: FTANodeType
    probability: float = 0.0    # 基本事件概率
    children: List[str] = field(default_factory=list)
    vote_threshold: int = 1     # 表决门阈值
    condition: Optional[str] = None  # 抑制门条件


class FaultTreeAnalyzer:
    """故障树分析器"""

    def __init__(self):
        self.nodes: Dict[str, FTANode] = {}
        self.top_event: Optional[str] = None

    def add_node(self, node: FTANode):
        """添加节点"""
        self.nodes[node.node_id] = node

    def set_top_event(self, node_id: str):
        """设置顶事件"""
        self.top_event = node_id

    def calculate_probability(self, node_id: Optional[str] = None,
                             basic_probabilities: Optional[Dict[str, float]] = None) -> float:
        """计算节点概率"""
        if node_id is None:
            node_id = self.top_event

        if node_id not in self.nodes:
            return 0.0

        node = self.nodes[node_id]

        # 更新基本事件概率
        if basic_probabilities and node.node_type == FTANodeType.BASIC:
            if node_id in basic_probabilities:
                node.probability = basic_probabilities[node_id]

        if node.node_type == FTANodeType.BASIC:
            return node.probability

        elif node.node_type == FTANodeType.AND:
            # P(A AND B) = P(A) * P(B)
            prob = 1.0
            for child_id in node.children:
                prob *= self.calculate_probability(child_id, basic_probabilities)
            return prob

        elif node.node_type == FTANodeType.OR:
            # P(A OR B) = 1 - (1-P(A)) * (1-P(B))
            prob = 1.0
            for child_id in node.children:
                child_prob = self.calculate_probability(child_id, basic_probabilities)
                prob *= (1 - child_prob)
            return 1 - prob

        elif node.node_type == FTANodeType.VOTE:
            # K-out-of-N
            child_probs = [
                self.calculate_probability(cid, basic_probabilities)
                for cid in node.children
            ]
            return self._vote_probability(child_probs, node.vote_threshold)

        return 0.0

    def _vote_probability(self, probs: List[float], k: int) -> float:
        """计算K/N表决概率"""
        n = len(probs)
        if k > n:
            return 0.0
        if k <= 0:
            return 1.0

        # 递归计算
        if n == 0:
            return 0.0 if k > 0 else 1.0

        p = probs[0]
        rest = probs[1:]

        # P(k/n) = p * P((k-1)/(n-1)) + (1-p) * P(k/(n-1))
        return p * self._vote_probability(rest, k-1) + (1-p) * self._vote_probability(rest, k)

    def get_minimal_cut_sets(self, node_id: Optional[str] = None) -> List[Set[str]]:
        """获取最小割集"""
        if node_id is None:
            node_id = self.top_event

        if node_id not in self.nodes:
            return []

        node = self.nodes[node_id]

        if node.node_type == FTANodeType.BASIC:
            return [{node_id}]

        elif node.node_type == FTANodeType.OR:
            # OR门：各子节点割集的并集
            result = []
            for child_id in node.children:
                result.extend(self.get_minimal_cut_sets(child_id))
            return self._minimize_cut_sets(result)

        elif node.node_type == FTANodeType.AND:
            # AND门：各子节点割集的笛卡尔积
            if not node.children:
                return []

            result = self.get_minimal_cut_sets(node.children[0])
            for child_id in node.children[1:]:
                child_sets = self.get_minimal_cut_sets(child_id)
                new_result = []
                for cs1 in result:
                    for cs2 in child_sets:
                        new_result.append(cs1 | cs2)
                result = new_result

            return self._minimize_cut_sets(result)

        return []

    def _minimize_cut_sets(self, cut_sets: List[Set[str]]) -> List[Set[str]]:
        """最小化割集"""
        result = []
        for cs in cut_sets:
            is_minimal = True
            for other in cut_sets:
                if other < cs:  # other是cs的真子集
                    is_minimal = False
                    break
            if is_minimal and cs not in result:
                result.append(cs)
        return result


# ============================================================================
# 贝叶斯网络诊断
# ============================================================================

@dataclass
class BayesNode:
    """贝叶斯网络节点"""
    node_id: str
    name: str
    states: List[str]           # 可能状态
    parents: List[str] = field(default_factory=list)
    cpt: Optional[np.ndarray] = None  # 条件概率表


class BayesianDiagnoser:
    """贝叶斯诊断器"""

    def __init__(self):
        self.nodes: Dict[str, BayesNode] = {}
        self.evidence: Dict[str, str] = {}

    def add_node(self, node: BayesNode):
        """添加节点"""
        self.nodes[node.node_id] = node

    def set_evidence(self, node_id: str, state: str):
        """设置证据"""
        if node_id in self.nodes:
            self.evidence[node_id] = state

    def clear_evidence(self):
        """清除证据"""
        self.evidence.clear()

    def infer(self, query_node: str) -> Dict[str, float]:
        """
        推理查询节点的后验概率

        Returns:
            各状态的概率 {state: probability}
        """
        if query_node not in self.nodes:
            return {}

        node = self.nodes[query_node]

        # 简化实现：基于似然加权的近似推理
        if query_node in self.evidence:
            # 已知节点
            result = {s: 0.0 for s in node.states}
            result[self.evidence[query_node]] = 1.0
            return result

        # 根据父节点计算
        if not node.parents:
            # 无父节点，返回先验
            if node.cpt is not None:
                return {s: node.cpt[i] for i, s in enumerate(node.states)}
            else:
                return {s: 1.0/len(node.states) for s in node.states}

        # 有父节点，需要边缘化
        # 简化实现：假设父节点独立
        parent_probs = {}
        for parent_id in node.parents:
            parent_probs[parent_id] = self.infer(parent_id)

        # 计算边缘概率
        result = {s: 0.0 for s in node.states}

        if node.cpt is not None:
            # 遍历父节点状态组合
            parent_states_list = [self.nodes[p].states for p in node.parents]

            def iterate_combinations(states_list, current_combo=[], index=0):
                if index == len(states_list):
                    yield current_combo.copy()
                    return
                for state in states_list[index]:
                    current_combo.append(state)
                    yield from iterate_combinations(states_list, current_combo, index + 1)
                    current_combo.pop()

            for combo in iterate_combinations(parent_states_list):
                combo_prob = 1.0
                for i, parent_id in enumerate(node.parents):
                    combo_prob *= parent_probs[parent_id].get(combo[i], 0.0)

                # 从CPT获取条件概率
                # 假设CPT格式: cpt[parent_states..., child_state]
                # 这里简化处理
                for j, child_state in enumerate(node.states):
                    result[child_state] += combo_prob * (1.0 / len(node.states))

        # 归一化
        total = sum(result.values())
        if total > 0:
            result = {k: v/total for k, v in result.items()}

        return result


# ============================================================================
# 案例推理
# ============================================================================

@dataclass
class DiagnosisCase:
    """诊断案例"""
    case_id: str
    description: str
    symptoms: Dict[str, Any]    # 症状特征
    diagnosis: str              # 诊断结果(故障ID)
    actions_taken: List[str]    # 采取的措施
    outcome: str                # 结果
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CaseBasedReasoner:
    """案例推理器"""

    def __init__(self):
        self.case_base: Dict[str, DiagnosisCase] = {}
        self.feature_weights: Dict[str, float] = {}

    def add_case(self, case: DiagnosisCase):
        """添加案例"""
        self.case_base[case.case_id] = case

    def set_feature_weight(self, feature: str, weight: float):
        """设置特征权重"""
        self.feature_weights[feature] = weight

    def retrieve(self, query_symptoms: Dict[str, Any], k: int = 5) -> List[Tuple[DiagnosisCase, float]]:
        """
        检索相似案例

        Returns:
            [(case, similarity), ...] 按相似度降序
        """
        similarities = []

        for case in self.case_base.values():
            sim = self._calculate_similarity(query_symptoms, case.symptoms)
            similarities.append((case, sim))

        # 按相似度排序
        similarities.sort(key=lambda x: -x[1])
        return similarities[:k]

    def _calculate_similarity(self, query: Dict[str, Any], case: Dict[str, Any]) -> float:
        """计算相似度"""
        if not query or not case:
            return 0.0

        all_features = set(query.keys()) | set(case.keys())
        if not all_features:
            return 0.0

        total_weight = 0.0
        weighted_sim = 0.0

        for feature in all_features:
            weight = self.feature_weights.get(feature, 1.0)
            total_weight += weight

            if feature in query and feature in case:
                sim = self._feature_similarity(query[feature], case[feature])
                weighted_sim += weight * sim

        return weighted_sim / total_weight if total_weight > 0 else 0.0

    def _feature_similarity(self, v1: Any, v2: Any) -> float:
        """计算特征相似度"""
        if v1 == v2:
            return 1.0

        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            # 数值相似度
            diff = abs(v1 - v2)
            max_val = max(abs(v1), abs(v2), 1)
            return max(0, 1 - diff / max_val)

        if isinstance(v1, str) and isinstance(v2, str):
            # 字符串相似度 (简单实现)
            s1, s2 = set(v1.lower()), set(v2.lower())
            if not s1 or not s2:
                return 0.0
            return len(s1 & s2) / len(s1 | s2)

        return 0.0

    def suggest(self, query_symptoms: Dict[str, Any]) -> Optional[Tuple[str, List[str], float]]:
        """
        根据症状建议诊断

        Returns:
            (fault_id, suggested_actions, confidence) 或 None
        """
        similar_cases = self.retrieve(query_symptoms, k=5)

        if not similar_cases:
            return None

        # 投票选择诊断
        diagnosis_votes: Dict[str, Tuple[float, List[str]]] = {}

        for case, similarity in similar_cases:
            if similarity < 0.3:  # 相似度阈值
                continue

            if case.diagnosis not in diagnosis_votes:
                diagnosis_votes[case.diagnosis] = (0.0, [])

            current_vote, current_actions = diagnosis_votes[case.diagnosis]
            diagnosis_votes[case.diagnosis] = (
                current_vote + similarity,
                current_actions + case.actions_taken
            )

        if not diagnosis_votes:
            return None

        # 选择最高票
        best_diagnosis = max(diagnosis_votes.items(), key=lambda x: x[1][0])
        diagnosis_id = best_diagnosis[0]
        confidence = best_diagnosis[1][0] / sum(v[0] for v in diagnosis_votes.values())
        actions = list(set(best_diagnosis[1][1]))  # 去重

        return (diagnosis_id, actions, confidence)


# ============================================================================
# 综合诊断系统
# ============================================================================

class IntegratedDiagnosisSystem:
    """综合故障诊断系统"""

    def __init__(self):
        self.faults: Dict[str, Fault] = {}
        self.symptoms: Dict[str, Symptom] = {}

        self.rule_engine = RuleEngine()
        self.fta = FaultTreeAnalyzer()
        self.bayesian = BayesianDiagnoser()
        self.cbr = CaseBasedReasoner()

        self._current_symptoms: Dict[str, bool] = {}
        self._diagnosis_history: List[DiagnosisResult] = []

    def register_fault(self, fault: Fault):
        """注册故障"""
        self.faults[fault.fault_id] = fault

    def register_symptom(self, symptom: Symptom):
        """注册症状"""
        self.symptoms[symptom.symptom_id] = symptom

    def add_rule(self, rule: Rule):
        """添加诊断规则"""
        self.rule_engine.add_rule(rule)

    def update_sensor_data(self, data: Dict[str, float]):
        """更新传感器数据"""
        # 评估症状
        for symptom_id, symptom in self.symptoms.items():
            try:
                # 评估条件
                is_present = self._evaluate_symptom(symptom, data)
                self._current_symptoms[symptom_id] = is_present
            except Exception as e:
                logger.error(f"Symptom evaluation error: {symptom_id}, {e}")

        # 更新规则引擎事实
        self.rule_engine.set_facts(data)
        for symptom_id, is_present in self._current_symptoms.items():
            self.rule_engine.set_fact(f"symptom_{symptom_id}", is_present)

    def _evaluate_symptom(self, symptom: Symptom, data: Dict[str, float]) -> bool:
        """评估症状"""
        condition = symptom.condition

        # 简单条件解析
        parts = condition.split()
        if len(parts) < 3:
            return False

        var_name = parts[0]
        operator = parts[1]
        threshold = float(parts[2])

        if var_name not in data:
            return False

        value = data[var_name]

        operators = {
            '>': lambda a, b: a > b,
            '>=': lambda a, b: a >= b,
            '<': lambda a, b: a < b,
            '<=': lambda a, b: a <= b,
            '==': lambda a, b: abs(a - b) < 0.001,
        }

        return operators.get(operator, lambda a, b: False)(value, threshold)

    def diagnose(self) -> List[DiagnosisResult]:
        """执行诊断"""
        results = []

        # 1. 规则推理
        rule_matches = self.rule_engine.evaluate()

        for fault_id, confidence in rule_matches:
            if fault_id in self.faults:
                fault = self.faults[fault_id]
                matched = [s for s in fault.symptoms if self._current_symptoms.get(s, False)]
                unmatched = [s for s in fault.symptoms if not self._current_symptoms.get(s, False)]

                result = DiagnosisResult(
                    fault_id=fault_id,
                    fault_name=fault.name,
                    probability=confidence,
                    confidence=confidence,
                    matched_symptoms=matched,
                    unmatched_symptoms=unmatched,
                    evidence={'method': 'rule_based'},
                    suggested_actions=fault.remedies,
                    estimated_severity=fault.severity
                )
                results.append(result)

        # 2. 案例推理补充
        symptom_features = {
            sid: 1 if present else 0
            for sid, present in self._current_symptoms.items()
        }

        cbr_result = self.cbr.suggest(symptom_features)
        if cbr_result:
            fault_id, actions, confidence = cbr_result
            if fault_id in self.faults and fault_id not in [r.fault_id for r in results]:
                fault = self.faults[fault_id]
                result = DiagnosisResult(
                    fault_id=fault_id,
                    fault_name=fault.name,
                    probability=confidence * 0.8,  # 降低权重
                    confidence=confidence,
                    matched_symptoms=[],
                    unmatched_symptoms=[],
                    evidence={'method': 'case_based'},
                    suggested_actions=actions,
                    estimated_severity=fault.severity
                )
                results.append(result)

        # 排序
        results.sort(key=lambda x: -x.probability)

        # 记录历史
        self._diagnosis_history.extend(results)

        return results

    def get_fault_probability(self, fault_id: str) -> float:
        """获取故障概率 (使用FTA)"""
        return self.fta.calculate_probability(fault_id)

    def get_diagnosis_history(self, limit: int = 100) -> List[DiagnosisResult]:
        """获取诊断历史"""
        return self._diagnosis_history[-limit:]


# ============================================================================
# 穿黄工程故障知识库
# ============================================================================

def create_cyrp_diagnosis_system() -> IntegratedDiagnosisSystem:
    """创建穿黄工程故障诊断系统"""
    system = IntegratedDiagnosisSystem()

    # 注册症状
    symptoms = [
        Symptom("S001", "流量下降", "进口流量低于正常值", "inlet_flow", "inlet_flow < 200"),
        Symptom("S002", "流量不对称", "两洞流量差异大", "flow_asymmetry", "flow_asymmetry > 20"),
        Symptom("S003", "压力升高", "进口压力异常升高", "inlet_pressure", "inlet_pressure > 800"),
        Symptom("S004", "压力下降", "进口压力异常下降", "inlet_pressure", "inlet_pressure < 300"),
        Symptom("S005", "负压", "出现负压", "min_pressure", "min_pressure < 0"),
        Symptom("S006", "渗漏增加", "渗漏量增加", "leakage_rate", "leakage_rate > 0.1"),
        Symptom("S007", "振动增大", "结构振动增大", "vibration_max", "vibration_max > 5"),
        Symptom("S008", "温度异常", "水温异常", "water_temperature", "water_temperature > 30"),
        Symptom("S009", "阀门响应慢", "阀门动作迟缓", "valve_response_time", "valve_response_time > 10"),
        Symptom("S010", "通信中断", "设备通信中断", "comm_status", "comm_status == 0"),
    ]

    for s in symptoms:
        system.register_symptom(s)

    # 注册故障
    faults = [
        Fault("F001", "管道堵塞", "隧洞内部堵塞导致过流能力下降",
              FaultCategory.HYDRAULIC, FaultSeverity.SERIOUS,
              symptoms=["S001", "S003"],
              causes=["泥沙淤积", "异物堵塞", "生物附着"],
              effects=["流量下降", "压力升高", "过流能力降低"],
              remedies=["清淤处理", "水下检查", "冲洗管道"]),

        Fault("F002", "阀门故障", "阀门动作异常或卡死",
              FaultCategory.MECHANICAL, FaultSeverity.SERIOUS,
              symptoms=["S009", "S002"],
              causes=["机械磨损", "液压系统故障", "电气故障"],
              effects=["流量调节失效", "紧急关闭失效"],
              remedies=["检查液压系统", "更换密封件", "润滑维护"]),

        Fault("F003", "结构渗漏", "隧洞衬砌渗漏",
              FaultCategory.STRUCTURAL, FaultSeverity.CRITICAL,
              symptoms=["S006", "S004"],
              causes=["衬砌裂缝", "止水带失效", "地下水压力"],
              effects=["水量损失", "结构安全隐患"],
              remedies=["灌浆堵漏", "结构加固", "降低水压"]),

        Fault("F004", "水锤冲击", "水锤压力波冲击",
              FaultCategory.HYDRAULIC, FaultSeverity.CRITICAL,
              symptoms=["S003", "S005", "S007"],
              causes=["快速关阀", "泵站停机", "流态突变"],
              effects=["压力振荡", "结构损伤风险"],
              remedies=["减缓阀门动作", "安装缓冲设施", "优化操作规程"]),

        Fault("F005", "空化气蚀", "低压区空化",
              FaultCategory.HYDRAULIC, FaultSeverity.SERIOUS,
              symptoms=["S005", "S007"],
              causes=["流速过高", "压力过低", "掺气不足"],
              effects=["衬砌损伤", "振动噪声"],
              remedies=["控制流速", "掺气设施", "优化流态"]),

        Fault("F006", "传感器故障", "测量传感器失效",
              FaultCategory.INSTRUMENTATION, FaultSeverity.MODERATE,
              symptoms=["S010"],
              causes=["元件老化", "电缆损坏", "供电故障"],
              effects=["数据缺失", "监测盲区"],
              remedies=["更换传感器", "检查线路", "备用冗余"]),

        Fault("F007", "控制系统故障", "自动控制失效",
              FaultCategory.CONTROL, FaultSeverity.SERIOUS,
              symptoms=["S010", "S009"],
              causes=["PLC故障", "程序错误", "网络中断"],
              effects=["自动化失效", "需人工干预"],
              remedies=["重启系统", "切换备用", "人工控制"]),
    ]

    for f in faults:
        system.register_fault(f)

    # 添加诊断规则
    rules = [
        Rule("R001", "堵塞诊断",
             ["inlet_flow < 200", "inlet_pressure > 700"],
             "F001", confidence=0.85),

        Rule("R002", "阀门故障诊断",
             ["valve_response_time > 10", "symptom_S009 == true"],
             "F002", confidence=0.9),

        Rule("R003", "渗漏诊断",
             ["leakage_rate > 0.1", "inlet_pressure < 400"],
             "F003", confidence=0.88),

        Rule("R004", "水锤诊断",
             ["inlet_pressure > 900", "vibration_max > 8"],
             "F004", confidence=0.92),

        Rule("R005", "空化诊断",
             ["min_pressure < -30", "vibration_max > 6"],
             "F005", confidence=0.85),

        Rule("R006", "传感器故障",
             ["comm_status == 0"],
             "F006", confidence=0.95),

        Rule("R007", "控制故障",
             ["comm_status == 0", "valve_response_time > 15"],
             "F007", confidence=0.88),
    ]

    for r in rules:
        system.add_rule(r)

    return system
