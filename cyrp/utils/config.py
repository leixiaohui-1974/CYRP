"""
Configuration utilities for CYRP.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import json
import os


@dataclass
class Config:
    """系统配置"""
    # 物理系统参数
    tunnel_length: float = 4250.0
    tunnel_diameter: float = 7.0
    design_flow: float = 265.0

    # 仿真参数
    simulation_dt: float = 0.1
    mpc_sample_time: float = 60.0

    # 控制参数
    mpc_horizon: int = 20
    pid_enabled: bool = True

    # 安全参数
    max_pressure: float = 1.0e6
    min_pressure: float = -5e4
    max_flow_imbalance: float = 0.1

    # 日志
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # 其他配置
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'tunnel_length': self.tunnel_length,
            'tunnel_diameter': self.tunnel_diameter,
            'design_flow': self.design_flow,
            'simulation_dt': self.simulation_dt,
            'mpc_sample_time': self.mpc_sample_time,
            'mpc_horizon': self.mpc_horizon,
            'pid_enabled': self.pid_enabled,
            'max_pressure': self.max_pressure,
            'min_pressure': self.min_pressure,
            'max_flow_imbalance': self.max_flow_imbalance,
            'log_level': self.log_level,
            'log_file': self.log_file,
            'extra': self.extra
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """从字典创建"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, filepath: str):
        """保存配置到文件"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'Config':
        """从文件加载配置"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


def load_config(filepath: Optional[str] = None) -> Config:
    """
    加载配置

    Args:
        filepath: 配置文件路径

    Returns:
        配置对象
    """
    if filepath and os.path.exists(filepath):
        return Config.load(filepath)

    # 检查默认路径
    default_paths = [
        'config/cyrp_config.json',
        'cyrp_config.json',
        os.path.expanduser('~/.cyrp/config.json')
    ]

    for path in default_paths:
        if os.path.exists(path):
            return Config.load(path)

    # 返回默认配置
    return Config()
