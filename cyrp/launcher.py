"""
CYRP 集成应用启动器 - Integrated Application Launcher

集成所有模块的统一入口点:
- REST API 服务
- WebSocket 实时推送
- 监控告警系统
- 数据持久化

Unified entry point integrating all modules:
- REST API service
- WebSocket real-time push
- Monitoring and alert system
- Data persistence
"""

import asyncio
import signal
import sys
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("cyrp.launcher")


class CYRPApplicationConfig:
    """应用配置"""

    def __init__(self):
        # 服务端口
        self.api_port = int(os.getenv('CYRP_API_PORT', '8080'))
        self.ws_port = int(os.getenv('CYRP_WS_PORT', '8081'))

        # 数据库配置
        self.db_path = os.getenv('CYRP_DB_PATH', './data/cyrp_history.db')

        # 环境
        self.env = os.getenv('CYRP_ENV', 'development')
        self.debug = os.getenv('CYRP_DEBUG', 'false').lower() == 'true'

        # 持久化配置
        self.flush_interval = float(os.getenv('CYRP_FLUSH_INTERVAL', '10.0'))
        self.buffer_size = int(os.getenv('CYRP_BUFFER_SIZE', '1000'))

        # 告警配置
        self.alert_cooldown = int(os.getenv('CYRP_ALERT_COOLDOWN', '300'))


class CYRPIntegratedSystem:
    """CYRP集成系统"""

    def __init__(self, config: CYRPApplicationConfig = None):
        self.config = config or CYRPApplicationConfig()
        self._running = False
        self._components: Dict[str, Any] = {}

        logger.info(f"CYRP Integrated System initializing (env: {self.config.env})")

    def initialize(self) -> bool:
        """初始化所有组件"""
        try:
            # 1. 初始化数据持久化
            self._init_persistence()

            # 2. 初始化API服务
            self._init_api()

            # 3. 初始化WebSocket服务
            self._init_websocket()

            # 4. 初始化监控告警
            self._init_monitoring()

            # 5. 连接组件
            self._connect_components()

            logger.info("All components initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    def _init_persistence(self):
        """初始化持久化层"""
        from cyrp.database.persistence_manager import (
            PersistenceManager, PersistenceConfig, PersistenceLevel
        )
        from cyrp.database.historian import SQLiteBackend

        # 确保数据目录存在
        db_dir = os.path.dirname(self.config.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        config = PersistenceConfig(
            level=PersistenceLevel.BUFFERED,
            buffer_size=self.config.buffer_size,
            flush_interval=self.config.flush_interval
        )

        backend = SQLiteBackend(self.config.db_path)
        self._components['persistence'] = PersistenceManager(backend, config)

        logger.info(f"Persistence initialized: {self.config.db_path}")

    def _init_api(self):
        """初始化API服务"""
        from cyrp.api.rest_api import create_cyrp_api
        from cyrp.api.monitoring_endpoints import create_monitoring_api_module

        # 创建API服务器
        api_server = create_cyrp_api()

        # 创建监控API模块
        monitoring_module = create_monitoring_api_module()

        # 集成监控路由
        api_server.include_router(monitoring_module.router)

        self._components['api_server'] = api_server
        self._components['monitoring_module'] = monitoring_module

        logger.info(f"API server initialized (port: {self.config.api_port})")

    def _init_websocket(self):
        """初始化WebSocket服务"""
        from cyrp.communication.websocket_server import create_realtime_push_system

        ws_system = create_realtime_push_system(heartbeat_interval=30.0)

        self._components['ws_manager'] = ws_system['connection_manager']
        self._components['push_service'] = ws_system['push_service']

        logger.info(f"WebSocket service initialized (port: {self.config.ws_port})")

    def _init_monitoring(self):
        """初始化监控告警"""
        from cyrp.monitoring.dashboard_data import DashboardDataProvider, MetricsCollector

        dashboard = DashboardDataProvider(history_size=10000)
        collector = MetricsCollector(dashboard)

        self._components['dashboard'] = dashboard
        self._components['metrics_collector'] = collector

        logger.info("Monitoring dashboard initialized")

    def _connect_components(self):
        """连接各组件"""
        # 将仪表板连接到监控API
        monitoring = self._components.get('monitoring_module')
        dashboard = self._components.get('dashboard')

        if monitoring and dashboard:
            monitoring.dashboard = dashboard
            logger.info("Dashboard connected to monitoring API")

        # 连接持久化到监控
        persistence = self._components.get('persistence')
        if persistence:
            persistence.start_background_flush()
            logger.info("Persistence background flush started")

    async def start(self):
        """启动系统"""
        if self._running:
            return

        self._running = True
        logger.info("CYRP Integrated System starting...")

        # 启动各服务
        tasks = []

        # 启动健康检查循环
        tasks.append(asyncio.create_task(self._health_check_loop()))

        # 启动指标收集循环
        tasks.append(asyncio.create_task(self._metrics_collection_loop()))

        # 等待所有任务
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Tasks cancelled, shutting down...")

    async def _health_check_loop(self):
        """健康检查循环"""
        while self._running:
            try:
                status = self.get_health_status()
                logger.debug(f"Health check: {status['status']}")
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(5)

    async def _metrics_collection_loop(self):
        """指标收集循环"""
        dashboard = self._components.get('dashboard')
        persistence = self._components.get('persistence')
        push_service = self._components.get('push_service')

        while self._running:
            try:
                # 收集系统指标
                metrics = self._collect_system_metrics()

                # 记录到仪表板
                if dashboard:
                    for name, value in metrics.items():
                        dashboard.record_metric(name, value)

                # 持久化
                if persistence:
                    persistence.record_metrics_batch(metrics)

                # 推送到WebSocket客户端
                if push_service:
                    await push_service.push_metrics_batch(metrics)

                await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(5)

    def _collect_system_metrics(self) -> Dict[str, float]:
        """收集系统指标"""
        import random

        # 模拟系统指标（实际应从传感器/控制系统获取）
        return {
            'flow_rate_total': 280.0 + random.uniform(-5, 5),
            'pressure_avg': 500000.0 + random.uniform(-10000, 10000),
            'pressure_max': 550000.0 + random.uniform(-5000, 5000),
            'health_score': 95.0 + random.uniform(-3, 2),
            'sensor_availability': 98.0 + random.uniform(-2, 2),
            'control_performance': 92.0 + random.uniform(-4, 4),
        }

    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        components_status = {}

        for name, component in self._components.items():
            if component is not None:
                components_status[name] = 'ok'
            else:
                components_status[name] = 'not_initialized'

        all_ok = all(s == 'ok' for s in components_status.values())

        return {
            'status': 'healthy' if all_ok else 'degraded',
            'timestamp': datetime.now().isoformat(),
            'environment': self.config.env,
            'components': components_status
        }

    def stop(self):
        """停止系统"""
        self._running = False
        logger.info("CYRP Integrated System stopping...")

        # 停止持久化
        persistence = self._components.get('persistence')
        if persistence:
            persistence.close()

        # 清理WebSocket连接
        ws_manager = self._components.get('ws_manager')
        if ws_manager:
            for client in ws_manager.get_all_clients():
                ws_manager.disconnect(client.client_id)

        logger.info("CYRP Integrated System stopped")


def create_integrated_system(config: CYRPApplicationConfig = None) -> CYRPIntegratedSystem:
    """创建集成系统实例"""
    system = CYRPIntegratedSystem(config)
    if system.initialize():
        return system
    raise RuntimeError("Failed to initialize CYRP Integrated System")


async def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("CYRP - 穿黄工程智能管控系统")
    logger.info("Yellow River Crossing Project Intelligent Control System")
    logger.info("=" * 60)

    # 创建系统
    try:
        system = create_integrated_system()
    except Exception as e:
        logger.error(f"Failed to create system: {e}")
        sys.exit(1)

    # 设置信号处理
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Received shutdown signal")
        system.stop()
        loop.stop()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    # 启动系统
    try:
        await system.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        system.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
