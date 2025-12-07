"""
Machine Learning Model Training and Deployment Module for CYRP
穿黄工程机器学习模型训练与部署模块

功能:
- 模型训练与评估
- 模型版本管理
- 在线推理服务
- 模型监控与漂移检测
- 自动重训练
"""

import asyncio
import json
import os
import pickle
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math
import statistics


class ModelType(Enum):
    """模型类型"""
    REGRESSION = auto()      # 回归
    CLASSIFICATION = auto()  # 分类
    CLUSTERING = auto()      # 聚类
    ANOMALY_DETECTION = auto()  # 异常检测
    TIME_SERIES = auto()     # 时间序列预测
    REINFORCEMENT = auto()   # 强化学习


class ModelStatus(Enum):
    """模型状态"""
    TRAINING = auto()
    TRAINED = auto()
    VALIDATED = auto()
    DEPLOYED = auto()
    DEPRECATED = auto()
    FAILED = auto()


class MetricType(Enum):
    """评估指标类型"""
    # 回归指标
    MSE = auto()
    RMSE = auto()
    MAE = auto()
    R2 = auto()
    # 分类指标
    ACCURACY = auto()
    PRECISION = auto()
    RECALL = auto()
    F1 = auto()
    AUC = auto()
    # 时间序列指标
    MAPE = auto()


@dataclass
class ModelMetrics:
    """模型评估指标"""
    metrics: Dict[str, float] = field(default_factory=dict)
    evaluated_at: datetime = field(default_factory=datetime.now)
    dataset_size: int = 0
    dataset_name: str = ""


@dataclass
class ModelVersion:
    """模型版本"""
    version_id: str
    model_id: str
    version: int
    created_at: datetime = field(default_factory=datetime.now)
    status: ModelStatus = ModelStatus.TRAINING
    metrics: Optional[ModelMetrics] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    model_path: str = ""
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class ModelDefinition:
    """模型定义"""
    model_id: str
    name: str
    model_type: ModelType
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    current_version: Optional[str] = None
    versions: List[ModelVersion] = field(default_factory=list)
    input_schema: Dict[str, str] = field(default_factory=dict)
    output_schema: Dict[str, str] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    owner: str = ""


@dataclass
class PredictionRequest:
    """预测请求"""
    request_id: str
    model_id: str
    version_id: Optional[str] = None  # None表示使用当前版本
    input_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PredictionResponse:
    """预测响应"""
    request_id: str
    model_id: str
    version_id: str
    predictions: Any = None
    confidence: Optional[float] = None
    processing_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseModel(ABC):
    """模型基类"""

    @abstractmethod
    def fit(self, X: List[List[float]], y: List[float]) -> 'BaseModel':
        """训练模型"""
        pass

    @abstractmethod
    def predict(self, X: List[List[float]]) -> List[float]:
        """预测"""
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """获取参数"""
        pass

    @abstractmethod
    def set_params(self, **params) -> 'BaseModel':
        """设置参数"""
        pass


class LinearRegressionModel(BaseModel):
    """线性回归模型"""

    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights: Optional[List[float]] = None
        self.bias: float = 0.0

    def fit(self, X: List[List[float]], y: List[float]) -> 'LinearRegressionModel':
        """使用梯度下降训练"""
        n_samples = len(X)
        n_features = len(X[0]) if X else 0

        self.weights = [0.0] * n_features
        self.bias = 0.0

        for _ in range(self.iterations):
            # 前向传播
            predictions = self.predict(X)

            # 计算梯度
            dw = [0.0] * n_features
            db = 0.0

            for i in range(n_samples):
                error = predictions[i] - y[i]
                for j in range(n_features):
                    dw[j] += error * X[i][j]
                db += error

            # 更新权重
            for j in range(n_features):
                self.weights[j] -= self.learning_rate * dw[j] / n_samples
            self.bias -= self.learning_rate * db / n_samples

        return self

    def predict(self, X: List[List[float]]) -> List[float]:
        """预测"""
        if self.weights is None:
            return [0.0] * len(X)

        predictions = []
        for x in X:
            pred = self.bias
            for j, w in enumerate(self.weights):
                pred += w * x[j]
            predictions.append(pred)

        return predictions

    def get_params(self) -> Dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "iterations": self.iterations,
            "weights": self.weights,
            "bias": self.bias
        }

    def set_params(self, **params) -> 'LinearRegressionModel':
        for key, value in params.items():
            setattr(self, key, value)
        return self


class KNNModel(BaseModel):
    """K近邻模型"""

    def __init__(self, k: int = 5):
        self.k = k
        self.X_train: List[List[float]] = []
        self.y_train: List[float] = []

    def fit(self, X: List[List[float]], y: List[float]) -> 'KNNModel':
        """存储训练数据"""
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X: List[List[float]]) -> List[float]:
        """预测"""
        predictions = []
        for x in X:
            # 计算距离
            distances = []
            for i, x_train in enumerate(self.X_train):
                dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(x, x_train)))
                distances.append((dist, self.y_train[i]))

            # 选择K个最近的
            distances.sort(key=lambda d: d[0])
            k_nearest = distances[:self.k]

            # 回归:取平均值
            pred = sum(d[1] for d in k_nearest) / len(k_nearest)
            predictions.append(pred)

        return predictions

    def get_params(self) -> Dict[str, Any]:
        return {"k": self.k}

    def set_params(self, **params) -> 'KNNModel':
        for key, value in params.items():
            setattr(self, key, value)
        return self


class SimpleAnomalyDetector(BaseModel):
    """简单异常检测模型(基于统计)"""

    def __init__(self, threshold_sigma: float = 3.0):
        self.threshold_sigma = threshold_sigma
        self.means: List[float] = []
        self.stds: List[float] = []

    def fit(self, X: List[List[float]], y: List[float] = None) -> 'SimpleAnomalyDetector':
        """计算每个特征的均值和标准差"""
        if not X:
            return self

        n_features = len(X[0])
        self.means = []
        self.stds = []

        for j in range(n_features):
            values = [x[j] for x in X]
            self.means.append(statistics.mean(values))
            self.stds.append(statistics.stdev(values) if len(values) > 1 else 1.0)

        return self

    def predict(self, X: List[List[float]]) -> List[float]:
        """返回异常分数(0=正常, 1=异常)"""
        predictions = []

        for x in X:
            max_zscore = 0
            for j, val in enumerate(x):
                if self.stds[j] > 0:
                    zscore = abs(val - self.means[j]) / self.stds[j]
                    max_zscore = max(max_zscore, zscore)

            # 如果最大Z分数超过阈值,则为异常
            is_anomaly = 1.0 if max_zscore > self.threshold_sigma else 0.0
            predictions.append(is_anomaly)

        return predictions

    def get_params(self) -> Dict[str, Any]:
        return {
            "threshold_sigma": self.threshold_sigma,
            "means": self.means,
            "stds": self.stds
        }

    def set_params(self, **params) -> 'SimpleAnomalyDetector':
        for key, value in params.items():
            setattr(self, key, value)
        return self


class ExponentialSmoothingModel(BaseModel):
    """指数平滑时间序列模型"""

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.last_value: float = 0.0
        self.last_level: float = 0.0

    def fit(self, X: List[List[float]], y: List[float]) -> 'ExponentialSmoothingModel':
        """训练(初始化)"""
        if y:
            self.last_value = y[-1]
            self.last_level = statistics.mean(y)
        return self

    def predict(self, X: List[List[float]]) -> List[float]:
        """预测未来值"""
        n_predictions = len(X) if X else 1
        predictions = []

        level = self.last_level
        for _ in range(n_predictions):
            # 简单指数平滑预测
            predictions.append(level)

        return predictions

    def update(self, new_value: float):
        """在线更新"""
        self.last_level = self.alpha * new_value + (1 - self.alpha) * self.last_level
        self.last_value = new_value

    def get_params(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "last_value": self.last_value,
            "last_level": self.last_level
        }

    def set_params(self, **params) -> 'ExponentialSmoothingModel':
        for key, value in params.items():
            setattr(self, key, value)
        return self


class ModelEvaluator:
    """模型评估器"""

    @staticmethod
    def calculate_mse(y_true: List[float], y_pred: List[float]) -> float:
        """均方误差"""
        if len(y_true) != len(y_pred):
            return float('inf')
        return sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true)

    @staticmethod
    def calculate_rmse(y_true: List[float], y_pred: List[float]) -> float:
        """均方根误差"""
        return math.sqrt(ModelEvaluator.calculate_mse(y_true, y_pred))

    @staticmethod
    def calculate_mae(y_true: List[float], y_pred: List[float]) -> float:
        """平均绝对误差"""
        if len(y_true) != len(y_pred):
            return float('inf')
        return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)

    @staticmethod
    def calculate_r2(y_true: List[float], y_pred: List[float]) -> float:
        """R²分数"""
        if len(y_true) != len(y_pred):
            return 0.0

        mean_true = statistics.mean(y_true)
        ss_tot = sum((t - mean_true) ** 2 for t in y_true)
        ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))

        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0

        return 1 - (ss_res / ss_tot)

    @staticmethod
    def calculate_accuracy(y_true: List[float], y_pred: List[float]) -> float:
        """准确率"""
        if len(y_true) != len(y_pred):
            return 0.0
        correct = sum(1 for t, p in zip(y_true, y_pred) if round(t) == round(p))
        return correct / len(y_true)

    @staticmethod
    def calculate_mape(y_true: List[float], y_pred: List[float]) -> float:
        """平均绝对百分比误差"""
        if len(y_true) != len(y_pred):
            return float('inf')

        total = 0.0
        count = 0
        for t, p in zip(y_true, y_pred):
            if t != 0:
                total += abs((t - p) / t)
                count += 1

        return (total / count * 100) if count > 0 else 0.0

    @staticmethod
    def evaluate(
        y_true: List[float],
        y_pred: List[float],
        model_type: ModelType
    ) -> ModelMetrics:
        """评估模型"""
        metrics = {}

        if model_type == ModelType.REGRESSION:
            metrics["mse"] = ModelEvaluator.calculate_mse(y_true, y_pred)
            metrics["rmse"] = ModelEvaluator.calculate_rmse(y_true, y_pred)
            metrics["mae"] = ModelEvaluator.calculate_mae(y_true, y_pred)
            metrics["r2"] = ModelEvaluator.calculate_r2(y_true, y_pred)
        elif model_type == ModelType.CLASSIFICATION:
            metrics["accuracy"] = ModelEvaluator.calculate_accuracy(y_true, y_pred)
        elif model_type == ModelType.TIME_SERIES:
            metrics["mse"] = ModelEvaluator.calculate_mse(y_true, y_pred)
            metrics["mape"] = ModelEvaluator.calculate_mape(y_true, y_pred)
        elif model_type == ModelType.ANOMALY_DETECTION:
            metrics["accuracy"] = ModelEvaluator.calculate_accuracy(y_true, y_pred)

        return ModelMetrics(
            metrics=metrics,
            dataset_size=len(y_true)
        )


class ModelRegistry:
    """模型注册表"""

    def __init__(self, storage_path: str = "./models"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._models: Dict[str, ModelDefinition] = {}
        self._loaded_models: Dict[str, BaseModel] = {}

    def register_model(self, definition: ModelDefinition):
        """注册模型定义"""
        self._models[definition.model_id] = definition

    def get_model_definition(self, model_id: str) -> Optional[ModelDefinition]:
        """获取模型定义"""
        return self._models.get(model_id)

    def save_model(
        self,
        model: BaseModel,
        model_id: str,
        version: ModelVersion
    ) -> str:
        """保存模型"""
        model_dir = self.storage_path / model_id / version.version_id
        model_dir.mkdir(parents=True, exist_ok=True)

        # 保存模型
        model_path = model_dir / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # 保存元数据
        meta_path = model_dir / "metadata.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({
                "version_id": version.version_id,
                "model_id": model_id,
                "version": version.version,
                "created_at": version.created_at.isoformat(),
                "hyperparameters": version.hyperparameters,
                "feature_names": version.feature_names,
                "metrics": version.metrics.metrics if version.metrics else {}
            }, f, indent=2, ensure_ascii=False)

        version.model_path = str(model_path)
        return str(model_path)

    def load_model(
        self,
        model_id: str,
        version_id: Optional[str] = None
    ) -> Optional[BaseModel]:
        """加载模型"""
        cache_key = f"{model_id}:{version_id or 'current'}"
        if cache_key in self._loaded_models:
            return self._loaded_models[cache_key]

        definition = self._models.get(model_id)
        if not definition:
            return None

        # 确定版本
        if version_id:
            version = next(
                (v for v in definition.versions if v.version_id == version_id),
                None
            )
        else:
            version = next(
                (v for v in definition.versions if v.version_id == definition.current_version),
                None
            )

        if not version or not version.model_path:
            return None

        try:
            with open(version.model_path, 'rb') as f:
                model = pickle.load(f)
            self._loaded_models[cache_key] = model
            return model
        except Exception:
            return None

    def list_models(self) -> List[ModelDefinition]:
        """列出所有模型"""
        return list(self._models.values())

    def list_versions(self, model_id: str) -> List[ModelVersion]:
        """列出模型版本"""
        definition = self._models.get(model_id)
        return definition.versions if definition else []


class ModelTrainer:
    """模型训练器"""

    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self._model_factories: Dict[str, Callable] = {
            "linear_regression": lambda **p: LinearRegressionModel(**p),
            "knn": lambda **p: KNNModel(**p),
            "anomaly_detector": lambda **p: SimpleAnomalyDetector(**p),
            "exponential_smoothing": lambda **p: ExponentialSmoothingModel(**p),
        }

    def register_model_factory(self, name: str, factory: Callable):
        """注册模型工厂"""
        self._model_factories[name] = factory

    async def train(
        self,
        model_id: str,
        algorithm: str,
        X_train: List[List[float]],
        y_train: List[float],
        X_val: Optional[List[List[float]]] = None,
        y_val: Optional[List[float]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        feature_names: Optional[List[str]] = None
    ) -> ModelVersion:
        """训练模型"""
        definition = self.registry.get_model_definition(model_id)
        if not definition:
            raise ValueError(f"模型不存在: {model_id}")

        if algorithm not in self._model_factories:
            raise ValueError(f"不支持的算法: {algorithm}")

        # 创建版本
        version_num = len(definition.versions) + 1
        version = ModelVersion(
            version_id=f"v{version_num}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            model_id=model_id,
            version=version_num,
            status=ModelStatus.TRAINING,
            hyperparameters=hyperparameters or {},
            feature_names=feature_names or []
        )

        try:
            # 创建并训练模型
            model = self._model_factories[algorithm](**(hyperparameters or {}))
            model.fit(X_train, y_train)

            # 评估模型
            if X_val and y_val:
                y_pred = model.predict(X_val)
                version.metrics = ModelEvaluator.evaluate(
                    y_val, y_pred, definition.model_type
                )
                version.status = ModelStatus.VALIDATED
            else:
                version.status = ModelStatus.TRAINED

            # 保存模型
            self.registry.save_model(model, model_id, version)

            # 更新定义
            definition.versions.append(version)
            definition.current_version = version.version_id
            definition.updated_at = datetime.now()

            return version

        except Exception as e:
            version.status = ModelStatus.FAILED
            version.description = str(e)
            definition.versions.append(version)
            raise


class PredictionService:
    """预测服务"""

    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self._prediction_log: List[Dict] = []
        self._performance_stats: Dict[str, Dict] = {}

    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        """执行预测"""
        import time
        start_time = time.perf_counter()

        # 加载模型
        model = self.registry.load_model(request.model_id, request.version_id)
        if not model:
            raise ValueError(f"模型不存在或未加载: {request.model_id}")

        definition = self.registry.get_model_definition(request.model_id)
        version_id = request.version_id or definition.current_version

        # 准备输入数据
        if isinstance(request.input_data, dict):
            # 从字典转换为特征列表
            X = [list(request.input_data.values())]
        elif isinstance(request.input_data, list):
            X = request.input_data if isinstance(request.input_data[0], list) else [request.input_data]
        else:
            X = [[request.input_data]]

        # 预测
        predictions = model.predict(X)

        processing_time = (time.perf_counter() - start_time) * 1000

        response = PredictionResponse(
            request_id=request.request_id,
            model_id=request.model_id,
            version_id=version_id,
            predictions=predictions[0] if len(predictions) == 1 else predictions,
            processing_time_ms=processing_time
        )

        # 记录预测日志
        self._log_prediction(request, response)

        return response

    def _log_prediction(
        self,
        request: PredictionRequest,
        response: PredictionResponse
    ):
        """记录预测日志"""
        self._prediction_log.append({
            "request_id": request.request_id,
            "model_id": request.model_id,
            "version_id": response.version_id,
            "input": request.input_data,
            "output": response.predictions,
            "processing_time_ms": response.processing_time_ms,
            "timestamp": response.timestamp.isoformat()
        })

        # 更新性能统计
        model_id = request.model_id
        if model_id not in self._performance_stats:
            self._performance_stats[model_id] = {
                "total_requests": 0,
                "total_time_ms": 0.0,
                "min_time_ms": float('inf'),
                "max_time_ms": 0.0
            }

        stats = self._performance_stats[model_id]
        stats["total_requests"] += 1
        stats["total_time_ms"] += response.processing_time_ms
        stats["min_time_ms"] = min(stats["min_time_ms"], response.processing_time_ms)
        stats["max_time_ms"] = max(stats["max_time_ms"], response.processing_time_ms)

    def get_performance_stats(self, model_id: str) -> Dict[str, Any]:
        """获取性能统计"""
        stats = self._performance_stats.get(model_id, {})
        if stats and stats.get("total_requests", 0) > 0:
            stats["avg_time_ms"] = stats["total_time_ms"] / stats["total_requests"]
        return stats


class ModelManager:
    """模型管理器(统一入口)"""

    def __init__(self, storage_path: str = "./models"):
        self.registry = ModelRegistry(storage_path)
        self.trainer = ModelTrainer(self.registry)
        self.prediction_service = PredictionService(self.registry)

    def register_model(
        self,
        model_id: str,
        name: str,
        model_type: ModelType,
        **kwargs
    ) -> ModelDefinition:
        """注册新模型"""
        definition = ModelDefinition(
            model_id=model_id,
            name=name,
            model_type=model_type,
            **kwargs
        )
        self.registry.register_model(definition)
        return definition

    async def train_model(
        self,
        model_id: str,
        algorithm: str,
        X_train: List[List[float]],
        y_train: List[float],
        **kwargs
    ) -> ModelVersion:
        """训练模型"""
        return await self.trainer.train(
            model_id, algorithm, X_train, y_train, **kwargs
        )

    async def predict(
        self,
        model_id: str,
        input_data: Any,
        version_id: Optional[str] = None
    ) -> PredictionResponse:
        """执行预测"""
        request = PredictionRequest(
            request_id=str(uuid.uuid4()),
            model_id=model_id,
            version_id=version_id,
            input_data=input_data
        )
        return await self.prediction_service.predict(request)


def create_cyrp_ml_system(storage_path: str = "./models") -> ModelManager:
    """创建穿黄工程机器学习系统"""
    manager = ModelManager(storage_path)

    # 注册预定义模型
    # 流量预测模型
    manager.register_model(
        model_id="flow_predictor",
        name="流量预测模型",
        model_type=ModelType.REGRESSION,
        description="基于历史数据预测流量",
        input_schema={
            "hour": "int",
            "day_of_week": "int",
            "month": "int",
            "upstream_flow": "float",
            "water_level": "float"
        },
        output_schema={"predicted_flow": "float"}
    )

    # 异常检测模型
    manager.register_model(
        model_id="anomaly_detector",
        name="异常检测模型",
        model_type=ModelType.ANOMALY_DETECTION,
        description="检测传感器数据异常",
        input_schema={
            "flow": "float",
            "pressure": "float",
            "temperature": "float"
        },
        output_schema={"is_anomaly": "bool", "anomaly_score": "float"}
    )

    # 设备故障预测模型
    manager.register_model(
        model_id="equipment_failure_predictor",
        name="设备故障预测模型",
        model_type=ModelType.CLASSIFICATION,
        description="预测设备是否会在未来7天内故障",
        input_schema={
            "vibration": "float",
            "temperature": "float",
            "runtime_hours": "float",
            "maintenance_days": "int"
        },
        output_schema={"failure_probability": "float"}
    )

    return manager
