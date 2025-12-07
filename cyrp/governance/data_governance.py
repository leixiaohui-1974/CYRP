"""
数据治理系统 - Data Governance System

实现完整的数据治理功能，包括：
- 数据质量管理和评估
- 数据血缘追踪
- 数据标准化和规范化
- 数据校验和清洗
- 数据目录管理

Implements complete data governance including:
- Data quality management and assessment
- Data lineage tracking
- Data standardization and normalization
- Data validation and cleaning
- Data catalog management
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod
import hashlib
import json
import threading
from collections import defaultdict


class QualityDimension(Enum):
    """数据质量维度"""
    COMPLETENESS = "completeness"       # 完整性
    ACCURACY = "accuracy"               # 准确性
    CONSISTENCY = "consistency"         # 一致性
    TIMELINESS = "timeliness"          # 时效性
    VALIDITY = "validity"              # 有效性
    UNIQUENESS = "uniqueness"          # 唯一性
    INTEGRITY = "integrity"            # 完整性
    PRECISION = "precision"            # 精度


class RuleType(Enum):
    """规则类型"""
    RANGE_CHECK = "range_check"        # 范围检查
    NULL_CHECK = "null_check"          # 空值检查
    FORMAT_CHECK = "format_check"      # 格式检查
    CONSISTENCY_CHECK = "consistency_check"  # 一致性检查
    OUTLIER_CHECK = "outlier_check"    # 异常值检查
    TREND_CHECK = "trend_check"        # 趋势检查
    CUSTOM = "custom"                  # 自定义规则


class DataCategory(Enum):
    """数据类别"""
    SENSOR = "sensor"                  # 传感器数据
    ACTUATOR = "actuator"              # 执行器数据
    PROCESS = "process"                # 过程数据
    ALARM = "alarm"                    # 报警数据
    EVENT = "event"                    # 事件数据
    CONFIGURATION = "configuration"    # 配置数据
    DIAGNOSTIC = "diagnostic"          # 诊断数据


@dataclass
class QualityRule:
    """质量规则"""
    rule_id: str
    rule_type: RuleType
    dimension: QualityDimension
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    severity: str = "warning"          # error, warning, info
    is_active: bool = True

    def __post_init__(self):
        self._validator: Optional[Callable] = None

    def set_validator(self, validator: Callable):
        """设置自定义验证器"""
        self._validator = validator

    def validate(self, data: Any, context: Optional[Dict] = None) -> Tuple[bool, str]:
        """验证数据"""
        if self._validator:
            return self._validator(data, self.parameters, context)
        return self._default_validate(data, context)

    def _default_validate(self, data: Any, context: Optional[Dict] = None) -> Tuple[bool, str]:
        """默认验证"""
        if self.rule_type == RuleType.RANGE_CHECK:
            min_val = self.parameters.get('min', float('-inf'))
            max_val = self.parameters.get('max', float('inf'))
            if isinstance(data, (int, float)):
                if data < min_val or data > max_val:
                    return False, f"Value {data} out of range [{min_val}, {max_val}]"
            elif isinstance(data, np.ndarray):
                if np.any(data < min_val) or np.any(data > max_val):
                    return False, f"Array contains values out of range [{min_val}, {max_val}]"

        elif self.rule_type == RuleType.NULL_CHECK:
            if data is None or (isinstance(data, float) and np.isnan(data)):
                return False, "Null value detected"
            if isinstance(data, np.ndarray):
                nan_count = np.sum(np.isnan(data))
                max_nan_ratio = self.parameters.get('max_nan_ratio', 0.0)
                if nan_count / len(data) > max_nan_ratio:
                    return False, f"Too many null values: {nan_count}/{len(data)}"

        elif self.rule_type == RuleType.OUTLIER_CHECK:
            if isinstance(data, np.ndarray):
                mean = np.nanmean(data)
                std = np.nanstd(data)
                threshold = self.parameters.get('threshold', 3.0)
                outliers = np.abs(data - mean) > threshold * std
                if np.any(outliers):
                    return False, f"Outliers detected: {np.sum(outliers)} points"

        return True, "OK"


@dataclass
class QualityMetrics:
    """质量指标"""
    timestamp: datetime = field(default_factory=datetime.now)
    completeness_score: float = 1.0
    accuracy_score: float = 1.0
    consistency_score: float = 1.0
    timeliness_score: float = 1.0
    validity_score: float = 1.0
    overall_score: float = 1.0

    # 详细指标
    null_count: int = 0
    total_count: int = 0
    outlier_count: int = 0
    error_count: int = 0
    warning_count: int = 0

    # 规则执行结果
    rules_passed: int = 0
    rules_failed: int = 0
    failed_rules: List[str] = field(default_factory=list)

    def compute_overall(self, weights: Optional[Dict[str, float]] = None):
        """计算综合得分"""
        weights = weights or {
            'completeness': 0.25,
            'accuracy': 0.25,
            'consistency': 0.20,
            'timeliness': 0.15,
            'validity': 0.15
        }

        self.overall_score = (
            weights.get('completeness', 0.2) * self.completeness_score +
            weights.get('accuracy', 0.2) * self.accuracy_score +
            weights.get('consistency', 0.2) * self.consistency_score +
            weights.get('timeliness', 0.2) * self.timeliness_score +
            weights.get('validity', 0.2) * self.validity_score
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'completeness_score': self.completeness_score,
            'accuracy_score': self.accuracy_score,
            'consistency_score': self.consistency_score,
            'timeliness_score': self.timeliness_score,
            'validity_score': self.validity_score,
            'overall_score': self.overall_score,
            'null_count': self.null_count,
            'total_count': self.total_count,
            'outlier_count': self.outlier_count,
            'error_count': self.error_count,
            'warning_count': self.warning_count,
            'rules_passed': self.rules_passed,
            'rules_failed': self.rules_failed,
            'failed_rules': self.failed_rules
        }


@dataclass
class DataAsset:
    """数据资产"""
    asset_id: str
    name: str
    description: str
    category: DataCategory
    data_type: str
    unit: str = ""
    source: str = ""

    # 元数据
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    owner: str = ""
    tags: List[str] = field(default_factory=list)

    # 质量信息
    quality_rules: List[str] = field(default_factory=list)
    last_quality_check: Optional[datetime] = None
    quality_score: float = 1.0

    # 统计信息
    record_count: int = 0
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None
    std_value: Optional[float] = None


@dataclass
class LineageNode:
    """血缘节点"""
    node_id: str
    asset_id: str
    operation: str              # transform, aggregate, filter, join, etc.
    timestamp: datetime = field(default_factory=datetime.now)
    parameters: Dict[str, Any] = field(default_factory=dict)
    parent_nodes: List[str] = field(default_factory=list)


class DataQualityEngine:
    """数据质量引擎"""

    def __init__(self):
        self.rules: Dict[str, QualityRule] = {}
        self.rule_sets: Dict[str, List[str]] = {}
        self._history: List[QualityMetrics] = []
        self._max_history = 1000

    def add_rule(self, rule: QualityRule):
        """添加规则"""
        self.rules[rule.rule_id] = rule

    def remove_rule(self, rule_id: str):
        """移除规则"""
        if rule_id in self.rules:
            del self.rules[rule_id]

    def create_rule_set(self, set_name: str, rule_ids: List[str]):
        """创建规则集"""
        self.rule_sets[set_name] = rule_ids

    def check_quality(self, data: Any, rule_ids: Optional[List[str]] = None,
                     context: Optional[Dict] = None) -> QualityMetrics:
        """
        检查数据质量

        Args:
            data: 要检查的数据
            rule_ids: 要应用的规则ID列表
            context: 上下文信息

        Returns:
            质量指标
        """
        metrics = QualityMetrics()

        # 确定要应用的规则
        if rule_ids is None:
            rules_to_apply = list(self.rules.values())
        else:
            rules_to_apply = [self.rules[rid] for rid in rule_ids if rid in self.rules]

        # 基本统计
        if isinstance(data, np.ndarray):
            metrics.total_count = len(data)
            metrics.null_count = np.sum(np.isnan(data))
        elif isinstance(data, dict):
            metrics.total_count = len(data)
            metrics.null_count = sum(1 for v in data.values() if v is None)
        elif isinstance(data, list):
            metrics.total_count = len(data)
            metrics.null_count = sum(1 for v in data if v is None)

        # 完整性得分
        if metrics.total_count > 0:
            metrics.completeness_score = 1.0 - metrics.null_count / metrics.total_count
        else:
            metrics.completeness_score = 0.0

        # 应用规则
        dimension_scores = defaultdict(list)

        for rule in rules_to_apply:
            if not rule.is_active:
                continue

            passed, message = rule.validate(data, context)

            if passed:
                metrics.rules_passed += 1
                dimension_scores[rule.dimension].append(1.0)
            else:
                metrics.rules_failed += 1
                metrics.failed_rules.append(rule.rule_id)
                dimension_scores[rule.dimension].append(0.0)

                if rule.severity == "error":
                    metrics.error_count += 1
                elif rule.severity == "warning":
                    metrics.warning_count += 1

        # 计算维度得分
        for dimension, scores in dimension_scores.items():
            avg_score = np.mean(scores) if scores else 1.0

            if dimension == QualityDimension.ACCURACY:
                metrics.accuracy_score = avg_score
            elif dimension == QualityDimension.CONSISTENCY:
                metrics.consistency_score = avg_score
            elif dimension == QualityDimension.TIMELINESS:
                metrics.timeliness_score = avg_score
            elif dimension == QualityDimension.VALIDITY:
                metrics.validity_score = avg_score

        # 计算综合得分
        metrics.compute_overall()

        # 保存历史
        self._history.append(metrics)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        return metrics

    def check_with_rule_set(self, data: Any, set_name: str,
                           context: Optional[Dict] = None) -> QualityMetrics:
        """使用规则集检查质量"""
        rule_ids = self.rule_sets.get(set_name, [])
        return self.check_quality(data, rule_ids, context)

    def get_quality_trend(self, n_samples: int = 100) -> List[Dict[str, Any]]:
        """获取质量趋势"""
        samples = self._history[-n_samples:]
        return [m.to_dict() for m in samples]

    def create_standard_rules(self):
        """创建标准规则集"""
        # 压力传感器规则
        self.add_rule(QualityRule(
            rule_id="pressure_range",
            rule_type=RuleType.RANGE_CHECK,
            dimension=QualityDimension.VALIDITY,
            name="Pressure Range Check",
            description="Check if pressure is within valid range",
            parameters={'min': -0.05e6, 'max': 2.0e6},
            severity="error"
        ))

        # 流量传感器规则
        self.add_rule(QualityRule(
            rule_id="flow_range",
            rule_type=RuleType.RANGE_CHECK,
            dimension=QualityDimension.VALIDITY,
            name="Flow Range Check",
            description="Check if flow is within valid range",
            parameters={'min': 0, 'max': 400},
            severity="error"
        ))

        # 空值检查
        self.add_rule(QualityRule(
            rule_id="null_check",
            rule_type=RuleType.NULL_CHECK,
            dimension=QualityDimension.COMPLETENESS,
            name="Null Value Check",
            description="Check for null or missing values",
            parameters={'max_nan_ratio': 0.05},
            severity="warning"
        ))

        # 异常值检查
        self.add_rule(QualityRule(
            rule_id="outlier_check",
            rule_type=RuleType.OUTLIER_CHECK,
            dimension=QualityDimension.ACCURACY,
            name="Outlier Check",
            description="Check for statistical outliers",
            parameters={'threshold': 3.0},
            severity="warning"
        ))

        # 创建规则集
        self.create_rule_set("sensor_data", [
            "pressure_range", "flow_range", "null_check", "outlier_check"
        ])


class DataLineageTracker:
    """数据血缘追踪器"""

    def __init__(self):
        self.nodes: Dict[str, LineageNode] = {}
        self.edges: Dict[str, Set[str]] = defaultdict(set)  # parent -> children
        self._lock = threading.Lock()

    def record_transformation(self, source_ids: List[str], target_id: str,
                             operation: str, parameters: Optional[Dict] = None) -> str:
        """
        记录数据转换

        Args:
            source_ids: 源数据ID列表
            target_id: 目标数据ID
            operation: 操作类型
            parameters: 操作参数

        Returns:
            节点ID
        """
        node_id = self._generate_node_id(target_id, operation)

        with self._lock:
            node = LineageNode(
                node_id=node_id,
                asset_id=target_id,
                operation=operation,
                parameters=parameters or {},
                parent_nodes=source_ids
            )
            self.nodes[node_id] = node

            # 建立边关系
            for source_id in source_ids:
                self.edges[source_id].add(node_id)

        return node_id

    def get_upstream(self, asset_id: str, max_depth: int = 10) -> List[LineageNode]:
        """获取上游血缘"""
        result = []
        visited = set()

        def traverse(node_id: str, depth: int):
            if depth > max_depth or node_id in visited:
                return
            visited.add(node_id)

            if node_id in self.nodes:
                node = self.nodes[node_id]
                result.append(node)
                for parent in node.parent_nodes:
                    traverse(parent, depth + 1)

        # 找到与asset_id关联的节点
        for node_id, node in self.nodes.items():
            if node.asset_id == asset_id:
                traverse(node_id, 0)

        return result

    def get_downstream(self, asset_id: str, max_depth: int = 10) -> List[LineageNode]:
        """获取下游血缘"""
        result = []
        visited = set()

        def traverse(node_id: str, depth: int):
            if depth > max_depth or node_id in visited:
                return
            visited.add(node_id)

            if node_id in self.nodes:
                result.append(self.nodes[node_id])

            for child_id in self.edges.get(node_id, []):
                traverse(child_id, depth + 1)

        # 找到与asset_id关联的节点
        for node_id, node in self.nodes.items():
            if node.asset_id == asset_id:
                traverse(node_id, 0)

        return result

    def get_lineage_graph(self, asset_id: str) -> Dict[str, Any]:
        """获取血缘图"""
        upstream = self.get_upstream(asset_id)
        downstream = self.get_downstream(asset_id)

        nodes = []
        edges = []
        seen_nodes = set()

        for node in upstream + downstream:
            if node.node_id not in seen_nodes:
                nodes.append({
                    'id': node.node_id,
                    'asset_id': node.asset_id,
                    'operation': node.operation,
                    'timestamp': node.timestamp.isoformat()
                })
                seen_nodes.add(node.node_id)

            for parent in node.parent_nodes:
                edges.append({
                    'source': parent,
                    'target': node.node_id
                })

        return {
            'nodes': nodes,
            'edges': edges
        }

    def _generate_node_id(self, asset_id: str, operation: str) -> str:
        """生成节点ID"""
        timestamp = datetime.now().isoformat()
        content = f"{asset_id}_{operation}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


class DataStandardizer:
    """数据标准化器"""

    def __init__(self):
        self.standards: Dict[str, Dict[str, Any]] = {}
        self._converters: Dict[str, Callable] = {}

    def register_standard(self, standard_name: str, definition: Dict[str, Any]):
        """注册数据标准"""
        self.standards[standard_name] = definition

    def register_converter(self, from_format: str, to_format: str, converter: Callable):
        """注册转换器"""
        key = f"{from_format}_to_{to_format}"
        self._converters[key] = converter

    def standardize(self, data: Any, target_standard: str,
                   source_format: Optional[str] = None) -> Any:
        """
        标准化数据

        Args:
            data: 输入数据
            target_standard: 目标标准
            source_format: 源格式

        Returns:
            标准化后的数据
        """
        if target_standard not in self.standards:
            raise ValueError(f"Unknown standard: {target_standard}")

        standard = self.standards[target_standard]

        # 如果有转换器，先转换
        if source_format:
            key = f"{source_format}_to_{target_standard}"
            if key in self._converters:
                data = self._converters[key](data)

        # 应用标准化规则
        if isinstance(data, dict):
            return self._standardize_dict(data, standard)
        elif isinstance(data, np.ndarray):
            return self._standardize_array(data, standard)
        elif isinstance(data, list):
            return [self.standardize(item, target_standard) for item in data]

        return data

    def _standardize_dict(self, data: Dict, standard: Dict) -> Dict:
        """标准化字典数据"""
        result = {}

        field_mappings = standard.get('field_mappings', {})
        required_fields = standard.get('required_fields', [])
        default_values = standard.get('default_values', {})

        # 字段映射
        for src_field, tgt_field in field_mappings.items():
            if src_field in data:
                result[tgt_field] = data[src_field]

        # 复制未映射的字段
        for key, value in data.items():
            if key not in field_mappings and key not in result:
                result[key] = value

        # 添加缺失的必需字段
        for field in required_fields:
            if field not in result:
                result[field] = default_values.get(field)

        # 单位转换
        unit_conversions = standard.get('unit_conversions', {})
        for field, conversion in unit_conversions.items():
            if field in result and result[field] is not None:
                factor = conversion.get('factor', 1.0)
                offset = conversion.get('offset', 0.0)
                result[field] = result[field] * factor + offset

        return result

    def _standardize_array(self, data: np.ndarray, standard: Dict) -> np.ndarray:
        """标准化数组数据"""
        result = data.copy()

        # 单位转换
        factor = standard.get('factor', 1.0)
        offset = standard.get('offset', 0.0)
        result = result * factor + offset

        # 范围裁剪
        if 'range' in standard:
            min_val, max_val = standard['range']
            result = np.clip(result, min_val, max_val)

        # 精度处理
        if 'precision' in standard:
            result = np.round(result, standard['precision'])

        return result

    def create_sensor_standards(self):
        """创建传感器数据标准"""
        # 压力数据标准 (统一为Pa)
        self.register_standard("pressure_pa", {
            'unit': 'Pa',
            'range': (-1e5, 3e6),
            'precision': 0,
            'unit_conversions': {
                'pressure_mpa': {'factor': 1e6, 'offset': 0},
                'pressure_bar': {'factor': 1e5, 'offset': 0}
            }
        })

        # 流量数据标准 (统一为m³/s)
        self.register_standard("flow_m3s", {
            'unit': 'm³/s',
            'range': (0, 500),
            'precision': 2,
            'unit_conversions': {
                'flow_lps': {'factor': 0.001, 'offset': 0},
                'flow_m3h': {'factor': 1/3600, 'offset': 0}
            }
        })

        # 温度数据标准 (统一为K)
        self.register_standard("temperature_k", {
            'unit': 'K',
            'range': (200, 400),
            'precision': 2,
            'unit_conversions': {
                'temperature_c': {'factor': 1, 'offset': 273.15},
                'temperature_f': {'factor': 5/9, 'offset': 255.372}
            }
        })


class DataValidator:
    """数据验证器"""

    def __init__(self, quality_engine: DataQualityEngine):
        self.quality_engine = quality_engine
        self.validation_history: List[Dict[str, Any]] = []

    def validate(self, data: Any, schema: Dict[str, Any],
                context: Optional[Dict] = None) -> Tuple[bool, List[str]]:
        """
        验证数据

        Args:
            data: 要验证的数据
            schema: 验证模式
            context: 上下文信息

        Returns:
            (是否通过, 错误列表)
        """
        errors = []

        # 类型检查
        expected_type = schema.get('type')
        if expected_type:
            if not self._check_type(data, expected_type):
                errors.append(f"Type mismatch: expected {expected_type}")

        # 必需字段检查
        if isinstance(data, dict):
            required = schema.get('required', [])
            for field in required:
                if field not in data:
                    errors.append(f"Missing required field: {field}")

        # 范围检查
        if 'range' in schema:
            min_val, max_val = schema['range']
            if isinstance(data, (int, float)):
                if data < min_val or data > max_val:
                    errors.append(f"Value {data} out of range [{min_val}, {max_val}]")

        # 枚举检查
        if 'enum' in schema:
            allowed_values = schema['enum']
            if data not in allowed_values:
                errors.append(f"Value {data} not in allowed values: {allowed_values}")

        # 模式检查
        if 'pattern' in schema and isinstance(data, str):
            import re
            if not re.match(schema['pattern'], data):
                errors.append(f"Value does not match pattern: {schema['pattern']}")

        # 质量检查
        if 'quality_rules' in schema:
            metrics = self.quality_engine.check_quality(
                data, schema['quality_rules'], context
            )
            if metrics.error_count > 0:
                errors.extend([f"Quality rule failed: {r}" for r in metrics.failed_rules])

        # 记录验证历史
        self.validation_history.append({
            'timestamp': datetime.now().isoformat(),
            'passed': len(errors) == 0,
            'errors': errors,
            'schema': schema.get('name', 'unnamed')
        })

        return len(errors) == 0, errors

    def _check_type(self, data: Any, expected_type: str) -> bool:
        """检查类型"""
        type_mapping = {
            'int': (int, np.integer),
            'float': (float, np.floating),
            'number': (int, float, np.number),
            'string': str,
            'array': (list, np.ndarray),
            'object': dict,
            'boolean': bool
        }

        expected = type_mapping.get(expected_type)
        if expected:
            return isinstance(data, expected)
        return True

    def validate_batch(self, data_batch: List[Any], schema: Dict[str, Any]
                      ) -> Tuple[int, int, List[str]]:
        """批量验证"""
        passed = 0
        failed = 0
        all_errors = []

        for item in data_batch:
            is_valid, errors = self.validate(item, schema)
            if is_valid:
                passed += 1
            else:
                failed += 1
                all_errors.extend(errors)

        return passed, failed, all_errors


class DataCatalog:
    """数据目录"""

    def __init__(self):
        self.assets: Dict[str, DataAsset] = {}
        self.categories: Dict[DataCategory, List[str]] = defaultdict(list)
        self.tags_index: Dict[str, List[str]] = defaultdict(list)
        self._lock = threading.Lock()

    def register_asset(self, asset: DataAsset):
        """注册数据资产"""
        with self._lock:
            self.assets[asset.asset_id] = asset
            self.categories[asset.category].append(asset.asset_id)
            for tag in asset.tags:
                self.tags_index[tag].append(asset.asset_id)

    def get_asset(self, asset_id: str) -> Optional[DataAsset]:
        """获取数据资产"""
        return self.assets.get(asset_id)

    def search_by_category(self, category: DataCategory) -> List[DataAsset]:
        """按类别搜索"""
        asset_ids = self.categories.get(category, [])
        return [self.assets[aid] for aid in asset_ids if aid in self.assets]

    def search_by_tag(self, tag: str) -> List[DataAsset]:
        """按标签搜索"""
        asset_ids = self.tags_index.get(tag, [])
        return [self.assets[aid] for aid in asset_ids if aid in self.assets]

    def search(self, query: str) -> List[DataAsset]:
        """全文搜索"""
        results = []
        query_lower = query.lower()

        for asset in self.assets.values():
            if (query_lower in asset.name.lower() or
                query_lower in asset.description.lower() or
                any(query_lower in tag.lower() for tag in asset.tags)):
                results.append(asset)

        return results

    def update_statistics(self, asset_id: str, data: np.ndarray):
        """更新统计信息"""
        if asset_id in self.assets:
            asset = self.assets[asset_id]
            asset.record_count = len(data)
            asset.min_value = float(np.nanmin(data))
            asset.max_value = float(np.nanmax(data))
            asset.mean_value = float(np.nanmean(data))
            asset.std_value = float(np.nanstd(data))
            asset.updated_at = datetime.now()

    def get_catalog_summary(self) -> Dict[str, Any]:
        """获取目录摘要"""
        return {
            'total_assets': len(self.assets),
            'by_category': {cat.value: len(ids) for cat, ids in self.categories.items()},
            'tags': list(self.tags_index.keys()),
            'assets': [
                {
                    'id': a.asset_id,
                    'name': a.name,
                    'category': a.category.value,
                    'quality_score': a.quality_score
                }
                for a in self.assets.values()
            ]
        }


class DataGovernanceManager:
    """数据治理管理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # 初始化各组件
        self.quality_engine = DataQualityEngine()
        self.lineage_tracker = DataLineageTracker()
        self.standardizer = DataStandardizer()
        self.validator = DataValidator(self.quality_engine)
        self.catalog = DataCatalog()

        # 初始化标准规则
        self._init_defaults()

    def _init_defaults(self):
        """初始化默认配置"""
        self.quality_engine.create_standard_rules()
        self.standardizer.create_sensor_standards()

    def ingest_data(self, data: Any, asset_id: str, source: str,
                   category: DataCategory, metadata: Optional[Dict] = None
                   ) -> Tuple[Any, QualityMetrics]:
        """
        数据摄入

        Args:
            data: 数据
            asset_id: 资产ID
            source: 数据源
            category: 数据类别
            metadata: 元数据

        Returns:
            (处理后的数据, 质量指标)
        """
        metadata = metadata or {}

        # 1. 质量检查
        rule_set = self._get_rule_set_for_category(category)
        metrics = self.quality_engine.check_quality(data, rule_set)

        # 2. 标准化
        standard = self._get_standard_for_category(category)
        if standard:
            data = self.standardizer.standardize(data, standard)

        # 3. 记录血缘
        self.lineage_tracker.record_transformation(
            source_ids=[source],
            target_id=asset_id,
            operation="ingest",
            parameters={
                'category': category.value,
                'quality_score': metrics.overall_score
            }
        )

        # 4. 更新目录
        if asset_id not in self.catalog.assets:
            asset = DataAsset(
                asset_id=asset_id,
                name=metadata.get('name', asset_id),
                description=metadata.get('description', ''),
                category=category,
                data_type=str(type(data).__name__),
                source=source,
                tags=metadata.get('tags', [])
            )
            self.catalog.register_asset(asset)

        # 更新统计
        if isinstance(data, np.ndarray):
            self.catalog.update_statistics(asset_id, data)

        # 更新质量得分
        if asset_id in self.catalog.assets:
            self.catalog.assets[asset_id].quality_score = metrics.overall_score
            self.catalog.assets[asset_id].last_quality_check = datetime.now()

        return data, metrics

    def _get_rule_set_for_category(self, category: DataCategory) -> List[str]:
        """根据类别获取规则集"""
        if category == DataCategory.SENSOR:
            return self.quality_engine.rule_sets.get("sensor_data", [])
        return []

    def _get_standard_for_category(self, category: DataCategory) -> Optional[str]:
        """根据类别获取标准"""
        # 可以根据需要返回不同的标准
        return None

    def transform_data(self, source_ids: List[str], target_id: str,
                      transformation: Callable, params: Optional[Dict] = None
                      ) -> Any:
        """
        数据转换

        Args:
            source_ids: 源数据ID列表
            target_id: 目标数据ID
            transformation: 转换函数
            params: 转换参数

        Returns:
            转换后的数据
        """
        # 记录血缘
        self.lineage_tracker.record_transformation(
            source_ids=source_ids,
            target_id=target_id,
            operation="transform",
            parameters=params or {}
        )

        # 执行转换 (这里假设数据已经加载)
        # 实际实现中需要从数据存储中加载数据
        return None

    def get_data_quality_report(self, asset_id: Optional[str] = None) -> Dict[str, Any]:
        """获取数据质量报告"""
        if asset_id:
            asset = self.catalog.get_asset(asset_id)
            if asset:
                return {
                    'asset_id': asset_id,
                    'name': asset.name,
                    'quality_score': asset.quality_score,
                    'last_check': asset.last_quality_check.isoformat() if asset.last_quality_check else None,
                    'statistics': {
                        'min': asset.min_value,
                        'max': asset.max_value,
                        'mean': asset.mean_value,
                        'std': asset.std_value,
                        'count': asset.record_count
                    }
                }

        # 全局报告
        return {
            'total_assets': len(self.catalog.assets),
            'average_quality_score': np.mean([a.quality_score for a in self.catalog.assets.values()]) if self.catalog.assets else 0,
            'quality_trend': self.quality_engine.get_quality_trend(),
            'assets_by_quality': self._categorize_by_quality()
        }

    def _categorize_by_quality(self) -> Dict[str, int]:
        """按质量分类资产"""
        categories = {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0}

        for asset in self.catalog.assets.values():
            if asset.quality_score >= 0.9:
                categories['excellent'] += 1
            elif asset.quality_score >= 0.7:
                categories['good'] += 1
            elif asset.quality_score >= 0.5:
                categories['fair'] += 1
            else:
                categories['poor'] += 1

        return categories

    def get_lineage_report(self, asset_id: str) -> Dict[str, Any]:
        """获取血缘报告"""
        return self.lineage_tracker.get_lineage_graph(asset_id)

    def get_catalog_report(self) -> Dict[str, Any]:
        """获取目录报告"""
        return self.catalog.get_catalog_summary()
