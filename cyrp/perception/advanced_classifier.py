"""
高级场景识别算法 - Advanced Scenario Classification

实现基于机器学习的场景分类、模式识别、异常检测
Implements ML-based scenario classification, pattern recognition, anomaly detection
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import warnings


class PatternType(Enum):
    """模式类型"""
    STEADY_STATE = "steady_state"
    TRANSIENT = "transient"
    OSCILLATION = "oscillation"
    DRIFT = "drift"
    SPIKE = "spike"
    STEP_CHANGE = "step_change"
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"


@dataclass
class FeatureVector:
    """特征向量"""
    mean: float
    std: float
    min_val: float
    max_val: float
    range_val: float
    skewness: float
    kurtosis: float
    rms: float
    crest_factor: float
    zero_crossings: int
    trend_slope: float
    spectral_centroid: float
    spectral_bandwidth: float
    dominant_frequency: float
    energy: float


@dataclass
class ClassificationResult:
    """分类结果"""
    scenario_id: str
    confidence: float
    probabilities: Dict[str, float]
    features: FeatureVector
    pattern: PatternType
    anomaly_score: float
    is_anomaly: bool
    reasoning: List[str]


class FeatureExtractor:
    """特征提取器"""

    def __init__(self, sample_rate: float = 10.0):
        self.sample_rate = sample_rate

    def extract(self, signal: np.ndarray) -> FeatureVector:
        """提取信号特征"""
        if len(signal) < 2:
            return self._empty_features()

        # 时域特征
        mean = np.mean(signal)
        std = np.std(signal)
        min_val = np.min(signal)
        max_val = np.max(signal)
        range_val = max_val - min_val
        rms = np.sqrt(np.mean(signal ** 2))
        crest_factor = max_val / rms if rms > 0 else 0

        # 高阶统计量
        skewness = self._skewness(signal)
        kurtosis = self._kurtosis(signal)

        # 过零点
        zero_crossings = self._count_zero_crossings(signal - mean)

        # 趋势
        trend_slope = self._linear_trend(signal)

        # 频域特征
        spectral_features = self._spectral_features(signal)

        return FeatureVector(
            mean=mean,
            std=std,
            min_val=min_val,
            max_val=max_val,
            range_val=range_val,
            skewness=skewness,
            kurtosis=kurtosis,
            rms=rms,
            crest_factor=crest_factor,
            zero_crossings=zero_crossings,
            trend_slope=trend_slope,
            spectral_centroid=spectral_features['centroid'],
            spectral_bandwidth=spectral_features['bandwidth'],
            dominant_frequency=spectral_features['dominant_freq'],
            energy=spectral_features['energy']
        )

    def _empty_features(self) -> FeatureVector:
        return FeatureVector(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    def _skewness(self, x: np.ndarray) -> float:
        n = len(x)
        mean = np.mean(x)
        std = np.std(x)
        if std == 0:
            return 0
        return np.sum((x - mean) ** 3) / (n * std ** 3)

    def _kurtosis(self, x: np.ndarray) -> float:
        n = len(x)
        mean = np.mean(x)
        std = np.std(x)
        if std == 0:
            return 0
        return np.sum((x - mean) ** 4) / (n * std ** 4) - 3

    def _count_zero_crossings(self, x: np.ndarray) -> int:
        return int(np.sum(np.abs(np.diff(np.sign(x))) > 0))

    def _linear_trend(self, x: np.ndarray) -> float:
        n = len(x)
        if n < 2:
            return 0
        t = np.arange(n)
        # 线性回归斜率
        slope = (n * np.sum(t * x) - np.sum(t) * np.sum(x)) / \
                (n * np.sum(t ** 2) - np.sum(t) ** 2)
        return slope

    def _spectral_features(self, x: np.ndarray) -> Dict[str, float]:
        n = len(x)
        if n < 4:
            return {'centroid': 0, 'bandwidth': 0, 'dominant_freq': 0, 'energy': 0}

        # FFT
        fft = np.fft.rfft(x - np.mean(x))
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(n, 1.0 / self.sample_rate)

        # 能量
        energy = np.sum(magnitude ** 2)

        if energy == 0:
            return {'centroid': 0, 'bandwidth': 0, 'dominant_freq': 0, 'energy': 0}

        # 谱质心
        centroid = np.sum(freqs * magnitude ** 2) / energy

        # 谱带宽
        bandwidth = np.sqrt(np.sum((freqs - centroid) ** 2 * magnitude ** 2) / energy)

        # 主频
        dominant_freq = freqs[np.argmax(magnitude)]

        return {
            'centroid': centroid,
            'bandwidth': bandwidth,
            'dominant_freq': dominant_freq,
            'energy': energy
        }


class PatternRecognizer:
    """模式识别器"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size

    def recognize(self, signal: np.ndarray, features: FeatureVector) -> PatternType:
        """识别信号模式"""
        # 稳态判断
        if features.std < 0.01 * abs(features.mean) and abs(features.trend_slope) < 1e-4:
            return PatternType.STEADY_STATE

        # 趋势判断
        if abs(features.trend_slope) > 0.01:
            if features.trend_slope > 0:
                return PatternType.TREND_UP
            else:
                return PatternType.TREND_DOWN

        # 振荡判断
        if features.zero_crossings > len(signal) * 0.3:
            return PatternType.OSCILLATION

        # 阶跃判断
        if self._detect_step(signal):
            return PatternType.STEP_CHANGE

        # 尖峰判断
        if features.crest_factor > 3:
            return PatternType.SPIKE

        # 漂移判断
        if features.std > 0.1 * abs(features.mean):
            return PatternType.DRIFT

        return PatternType.TRANSIENT

    def _detect_step(self, signal: np.ndarray) -> bool:
        """检测阶跃变化"""
        n = len(signal)
        if n < 10:
            return False

        # 比较前后段均值差异
        mid = n // 2
        mean_before = np.mean(signal[:mid])
        mean_after = np.mean(signal[mid:])
        std_before = np.std(signal[:mid])
        std_after = np.std(signal[mid:])

        # 阶跃条件：均值变化大，段内标准差小
        mean_change = abs(mean_after - mean_before)
        avg_std = (std_before + std_after) / 2

        return mean_change > 3 * avg_std if avg_std > 0 else False


class AnomalyDetector:
    """异常检测器"""

    def __init__(self, history_size: int = 1000, threshold_sigma: float = 3.0):
        self.history_size = history_size
        self.threshold_sigma = threshold_sigma
        self.history: Dict[str, deque] = {}
        self.baselines: Dict[str, Tuple[float, float]] = {}

    def update_baseline(self, sensor_id: str, value: float):
        """更新基线"""
        if sensor_id not in self.history:
            self.history[sensor_id] = deque(maxlen=self.history_size)

        self.history[sensor_id].append(value)

        if len(self.history[sensor_id]) >= 100:
            values = np.array(self.history[sensor_id])
            self.baselines[sensor_id] = (np.mean(values), np.std(values))

    def detect(self, sensor_id: str, value: float) -> Tuple[float, bool]:
        """检测异常"""
        self.update_baseline(sensor_id, value)

        if sensor_id not in self.baselines:
            return 0.0, False

        mean, std = self.baselines[sensor_id]
        if std == 0:
            return 0.0, False

        z_score = abs(value - mean) / std
        is_anomaly = z_score > self.threshold_sigma

        return z_score, is_anomaly

    def detect_multivariate(self, values: Dict[str, float]) -> Tuple[float, bool]:
        """多变量异常检测 - Mahalanobis距离"""
        if len(values) < 2:
            return 0.0, False

        # 简化的多变量检测
        anomaly_scores = []
        for sensor_id, value in values.items():
            score, _ = self.detect(sensor_id, value)
            anomaly_scores.append(score)

        combined_score = np.sqrt(np.mean(np.array(anomaly_scores) ** 2))
        is_anomaly = combined_score > self.threshold_sigma

        return combined_score, is_anomaly


class IsolationForest:
    """孤立森林异常检测"""

    def __init__(self, n_trees: int = 100, sample_size: int = 256):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.trees: List[Dict] = []
        self.is_fitted = False

    def fit(self, X: np.ndarray):
        """训练模型"""
        n_samples = X.shape[0]
        self.trees = []

        for _ in range(self.n_trees):
            # 随机采样
            sample_idx = np.random.choice(n_samples, min(self.sample_size, n_samples), replace=False)
            sample = X[sample_idx]

            # 构建树
            tree = self._build_tree(sample, 0, int(np.ceil(np.log2(self.sample_size))))
            self.trees.append(tree)

        self.is_fitted = True

    def _build_tree(self, X: np.ndarray, depth: int, max_depth: int) -> Dict:
        """构建隔离树"""
        n_samples, n_features = X.shape

        if depth >= max_depth or n_samples <= 1:
            return {'type': 'leaf', 'size': n_samples}

        # 随机选择特征和分割点
        feature = np.random.randint(n_features)
        min_val, max_val = X[:, feature].min(), X[:, feature].max()

        if min_val == max_val:
            return {'type': 'leaf', 'size': n_samples}

        split = np.random.uniform(min_val, max_val)

        left_mask = X[:, feature] < split
        right_mask = ~left_mask

        return {
            'type': 'node',
            'feature': feature,
            'split': split,
            'left': self._build_tree(X[left_mask], depth + 1, max_depth),
            'right': self._build_tree(X[right_mask], depth + 1, max_depth)
        }

    def predict_score(self, x: np.ndarray) -> float:
        """预测异常分数"""
        if not self.is_fitted:
            return 0.5

        path_lengths = []
        for tree in self.trees:
            path_length = self._path_length(x, tree, 0)
            path_lengths.append(path_length)

        avg_path = np.mean(path_lengths)
        # 归一化
        c = self._c(self.sample_size)
        score = 2 ** (-avg_path / c)

        return score

    def _path_length(self, x: np.ndarray, tree: Dict, depth: int) -> float:
        if tree['type'] == 'leaf':
            return depth + self._c(tree['size'])

        if x[tree['feature']] < tree['split']:
            return self._path_length(x, tree['left'], depth + 1)
        else:
            return self._path_length(x, tree['right'], depth + 1)

    def _c(self, n: int) -> float:
        """平均路径长度"""
        if n <= 1:
            return 0
        return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n


class BayesianClassifier:
    """贝叶斯场景分类器"""

    def __init__(self, scenarios: List[str]):
        self.scenarios = scenarios
        self.n_scenarios = len(scenarios)

        # 先验概率
        self.priors: Dict[str, float] = {s: 1.0 / self.n_scenarios for s in scenarios}

        # 条件概率参数 (高斯分布)
        self.likelihoods: Dict[str, Dict[str, Tuple[float, float]]] = {}

        self._init_likelihoods()

    def _init_likelihoods(self):
        """初始化似然函数参数"""
        # 基于专家知识设置各场景下特征的期望分布
        scenario_params = {
            # 常规运行 S1-A (高流量)
            'S1-A': {
                'flow_rate': (280, 20), 'pressure': (0.6, 0.05),
                'velocity': (3.8, 0.3), 'std': (0.02, 0.01)
            },
            # 常规运行 S1-B (中流量)
            'S1-B': {
                'flow_rate': (200, 30), 'pressure': (0.45, 0.05),
                'velocity': (2.8, 0.3), 'std': (0.015, 0.008)
            },
            # 常规运行 S1-C (低流量)
            'S1-C': {
                'flow_rate': (120, 30), 'pressure': (0.3, 0.05),
                'velocity': (1.8, 0.3), 'std': (0.01, 0.005)
            },
            # 双洞运行 S2-A
            'S2-A': {
                'flow_rate': (265, 25), 'pressure': (0.5, 0.04),
                'velocity': (3.44, 0.2), 'std': (0.015, 0.008)
            },
            # 切换过程 S3-A
            'S3-A': {
                'flow_rate': (200, 50), 'pressure': (0.4, 0.1),
                'velocity': (2.5, 0.5), 'std': (0.05, 0.02)
            },
            # 紧急切换 S3-B
            'S3-B': {
                'flow_rate': (180, 60), 'pressure': (0.35, 0.15),
                'velocity': (2.3, 0.6), 'std': (0.1, 0.04)
            },
            # 检修模式 S4-A
            'S4-A': {
                'flow_rate': (100, 30), 'pressure': (0.25, 0.05),
                'velocity': (1.5, 0.3), 'std': (0.02, 0.01)
            },
            # 轻微渗漏 S5-A
            'S5-A': {
                'flow_rate': (260, 15), 'pressure': (0.48, 0.03),
                'velocity': (3.4, 0.2), 'std': (0.03, 0.01),
                'leakage_indicator': (0.01, 0.005)
            },
            # 中度渗漏 S5-B
            'S5-B': {
                'flow_rate': (255, 20), 'pressure': (0.45, 0.05),
                'velocity': (3.3, 0.25), 'std': (0.05, 0.02),
                'leakage_indicator': (0.03, 0.01)
            },
            # 严重渗漏 S5-C
            'S5-C': {
                'flow_rate': (240, 30), 'pressure': (0.4, 0.08),
                'velocity': (3.1, 0.3), 'std': (0.1, 0.04),
                'leakage_indicator': (0.08, 0.02)
            },
            # 地震 VI度 S6-A
            'S6-A': {
                'flow_rate': (265, 30), 'pressure': (0.5, 0.1),
                'velocity': (3.44, 0.4), 'std': (0.08, 0.03),
                'seismic_indicator': (0.05, 0.02)
            },
            # 地震 VII度 S6-B
            'S6-B': {
                'flow_rate': (250, 40), 'pressure': (0.48, 0.15),
                'velocity': (3.3, 0.5), 'std': (0.15, 0.05),
                'seismic_indicator': (0.15, 0.05)
            },
            # 地震 VIII度 S6-C
            'S6-C': {
                'flow_rate': (200, 60), 'pressure': (0.4, 0.2),
                'velocity': (2.8, 0.7), 'std': (0.25, 0.08),
                'seismic_indicator': (0.3, 0.1)
            },
        }

        for scenario_id in self.scenarios:
            if scenario_id in scenario_params:
                self.likelihoods[scenario_id] = scenario_params[scenario_id]
            else:
                # 默认参数
                self.likelihoods[scenario_id] = {
                    'flow_rate': (265, 50), 'pressure': (0.5, 0.1),
                    'velocity': (3.44, 0.5), 'std': (0.05, 0.03)
                }

    def update_prior(self, scenario_id: str, probability: float):
        """更新先验概率"""
        if scenario_id in self.priors:
            self.priors[scenario_id] = probability
            # 归一化
            total = sum(self.priors.values())
            for s in self.priors:
                self.priors[s] /= total

    def classify(self, features: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
        """贝叶斯分类"""
        posteriors = {}

        for scenario_id in self.scenarios:
            # 计算似然
            log_likelihood = 0
            for feature_name, value in features.items():
                if feature_name in self.likelihoods.get(scenario_id, {}):
                    mean, std = self.likelihoods[scenario_id][feature_name]
                    # 高斯似然
                    log_likelihood += self._log_gaussian(value, mean, std)

            # 后验 ∝ 先验 × 似然
            log_posterior = np.log(self.priors[scenario_id] + 1e-10) + log_likelihood
            posteriors[scenario_id] = log_posterior

        # 转换为概率
        max_log = max(posteriors.values())
        for s in posteriors:
            posteriors[s] = np.exp(posteriors[s] - max_log)

        total = sum(posteriors.values())
        for s in posteriors:
            posteriors[s] /= total

        # 最可能的场景
        best_scenario = max(posteriors, key=posteriors.get)

        return best_scenario, posteriors

    def _log_gaussian(self, x: float, mean: float, std: float) -> float:
        """对数高斯似然"""
        if std <= 0:
            std = 0.01
        return -0.5 * ((x - mean) / std) ** 2 - np.log(std)


class AdvancedScenarioClassifier:
    """高级场景分类器 - 集成多种方法"""

    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.pattern_recognizer = PatternRecognizer()
        self.anomaly_detector = AnomalyDetector()
        self.isolation_forest = IsolationForest()

        # 场景列表
        self.scenarios = [
            'S1-A', 'S1-B', 'S1-C',  # 单洞常规
            'S2-A', 'S2-B', 'S2-C',  # 双洞常规
            'S3-A', 'S3-B',          # 切换
            'S4-A', 'S4-B', 'S4-C',  # 检修
            'S5-A', 'S5-B', 'S5-C', 'S5-D',  # 渗漏
            'S6-A', 'S6-B', 'S6-C',  # 地震
            'S7',                     # 综合应急
        ]

        self.bayesian_classifier = BayesianClassifier(self.scenarios)

        # 规则引擎阈值
        self.thresholds = {
            'leakage_rate': {'minor': 0.01, 'moderate': 0.03, 'severe': 0.05, 'critical': 0.08},
            'seismic_intensity': {'vi': 0.05, 'vii': 0.15, 'viii': 0.3},
            'flow_rate': {'low': 100, 'medium': 200, 'high': 280},
            'pressure_drop': {'minor': 0.02, 'moderate': 0.05, 'severe': 0.1},
        }

        # 历史数据缓冲
        self.history_buffer: Dict[str, deque] = {}
        self.buffer_size = 100

    def classify(self, sensor_data: Dict[str, float],
                signals: Optional[Dict[str, np.ndarray]] = None) -> ClassificationResult:
        """综合分类"""

        # 1. 特征提取
        features = self._extract_features(sensor_data, signals)

        # 2. 模式识别
        pattern = self._recognize_pattern(signals)

        # 3. 异常检测
        anomaly_score, is_anomaly = self._detect_anomalies(sensor_data)

        # 4. 规则推理
        rule_result = self._apply_rules(sensor_data, features)

        # 5. 贝叶斯分类
        bayes_scenario, bayes_probs = self.bayesian_classifier.classify(features)

        # 6. 集成决策
        final_scenario, confidence, reasoning = self._ensemble_decision(
            rule_result, bayes_scenario, bayes_probs, pattern, anomaly_score
        )

        return ClassificationResult(
            scenario_id=final_scenario,
            confidence=confidence,
            probabilities=bayes_probs,
            features=self._make_feature_vector(features),
            pattern=pattern,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
            reasoning=reasoning
        )

    def _extract_features(self, sensor_data: Dict[str, float],
                         signals: Optional[Dict[str, np.ndarray]]) -> Dict[str, float]:
        """提取特征"""
        features = dict(sensor_data)

        if signals:
            for signal_name, signal_data in signals.items():
                fv = self.feature_extractor.extract(signal_data)
                features[f'{signal_name}_std'] = fv.std
                features[f'{signal_name}_trend'] = fv.trend_slope
                features[f'{signal_name}_rms'] = fv.rms

        return features

    def _recognize_pattern(self, signals: Optional[Dict[str, np.ndarray]]) -> PatternType:
        """识别模式"""
        if not signals:
            return PatternType.STEADY_STATE

        # 使用主要信号（流量）进行模式识别
        if 'flow_rate' in signals:
            signal = signals['flow_rate']
            fv = self.feature_extractor.extract(signal)
            return self.pattern_recognizer.recognize(signal, fv)

        return PatternType.STEADY_STATE

    def _detect_anomalies(self, sensor_data: Dict[str, float]) -> Tuple[float, bool]:
        """检测异常"""
        return self.anomaly_detector.detect_multivariate(sensor_data)

    def _apply_rules(self, sensor_data: Dict[str, float],
                    features: Dict[str, float]) -> Tuple[str, float]:
        """应用规则推理"""
        # 渗漏检测规则
        leakage_rate = sensor_data.get('leakage_rate', 0)
        if leakage_rate > self.thresholds['leakage_rate']['critical']:
            return 'S5-D', 0.95
        elif leakage_rate > self.thresholds['leakage_rate']['severe']:
            return 'S5-C', 0.9
        elif leakage_rate > self.thresholds['leakage_rate']['moderate']:
            return 'S5-B', 0.85
        elif leakage_rate > self.thresholds['leakage_rate']['minor']:
            return 'S5-A', 0.8

        # 地震检测规则
        seismic = sensor_data.get('seismic_intensity', 0)
        if seismic > self.thresholds['seismic_intensity']['viii']:
            return 'S6-C', 0.95
        elif seismic > self.thresholds['seismic_intensity']['vii']:
            return 'S6-B', 0.9
        elif seismic > self.thresholds['seismic_intensity']['vi']:
            return 'S6-A', 0.85

        # 流量模式规则
        flow = sensor_data.get('flow_rate', 265)
        tunnel_mode = sensor_data.get('tunnel_mode', 'dual')

        if tunnel_mode == 'single':
            if flow > self.thresholds['flow_rate']['high']:
                return 'S1-A', 0.8
            elif flow > self.thresholds['flow_rate']['medium']:
                return 'S1-B', 0.8
            else:
                return 'S1-C', 0.8
        else:  # dual
            if flow > self.thresholds['flow_rate']['high']:
                return 'S2-A', 0.8
            elif flow > self.thresholds['flow_rate']['medium']:
                return 'S2-B', 0.8
            else:
                return 'S2-C', 0.8

    def _ensemble_decision(self, rule_result: Tuple[str, float],
                          bayes_scenario: str,
                          bayes_probs: Dict[str, float],
                          pattern: PatternType,
                          anomaly_score: float) -> Tuple[str, float, List[str]]:
        """集成决策"""
        rule_scenario, rule_conf = rule_result
        reasoning = []

        # 紧急场景优先 (规则置信度高时)
        if rule_scenario.startswith('S5') or rule_scenario.startswith('S6'):
            if rule_conf > 0.85:
                reasoning.append(f"Emergency detected by rules: {rule_scenario}")
                return rule_scenario, rule_conf, reasoning

        # 异常分数高时增加贝叶斯紧急场景权重
        adjusted_probs = dict(bayes_probs)
        if anomaly_score > 2.0:
            for s in ['S5-A', 'S5-B', 'S5-C', 'S5-D', 'S6-A', 'S6-B', 'S6-C', 'S7']:
                if s in adjusted_probs:
                    adjusted_probs[s] *= (1 + anomaly_score / 3)
            # 重新归一化
            total = sum(adjusted_probs.values())
            for s in adjusted_probs:
                adjusted_probs[s] /= total
            reasoning.append(f"Anomaly detected (score={anomaly_score:.2f}), boosted emergency probabilities")

        # 模式信息整合
        if pattern == PatternType.OSCILLATION:
            reasoning.append("Oscillation pattern detected")
        elif pattern == PatternType.STEP_CHANGE:
            reasoning.append("Step change pattern detected")
            # 可能是切换
            for s in ['S3-A', 'S3-B']:
                if s in adjusted_probs:
                    adjusted_probs[s] *= 1.5
            total = sum(adjusted_probs.values())
            for s in adjusted_probs:
                adjusted_probs[s] /= total

        # 最终决策
        final_scenario = max(adjusted_probs, key=adjusted_probs.get)
        confidence = adjusted_probs[final_scenario]

        # 规则和贝叶斯一致性检查
        if rule_scenario == final_scenario:
            confidence = min(1.0, confidence * 1.2)
            reasoning.append(f"Rule and Bayesian agree: {final_scenario}")
        else:
            reasoning.append(f"Rule suggests {rule_scenario}, Bayesian suggests {final_scenario}")

        return final_scenario, confidence, reasoning

    def _make_feature_vector(self, features: Dict[str, float]) -> FeatureVector:
        """构造特征向量对象"""
        return FeatureVector(
            mean=features.get('flow_rate', 0),
            std=features.get('flow_rate_std', 0),
            min_val=features.get('min_flow', 0),
            max_val=features.get('max_flow', 0),
            range_val=features.get('flow_range', 0),
            skewness=0,
            kurtosis=0,
            rms=features.get('flow_rate_rms', 0),
            crest_factor=0,
            zero_crossings=0,
            trend_slope=features.get('flow_rate_trend', 0),
            spectral_centroid=0,
            spectral_bandwidth=0,
            dominant_frequency=0,
            energy=0
        )

    def train_isolation_forest(self, historical_data: np.ndarray):
        """训练孤立森林模型"""
        self.isolation_forest.fit(historical_data)

    def update(self, sensor_data: Dict[str, float]):
        """更新分类器状态"""
        for key, value in sensor_data.items():
            if key not in self.history_buffer:
                self.history_buffer[key] = deque(maxlen=self.buffer_size)
            self.history_buffer[key].append(value)

        # 更新异常检测基线
        for key, value in sensor_data.items():
            self.anomaly_detector.update_baseline(key, value)
