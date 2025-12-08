"""
CYRP Command Line Interface (CLI)
穿黄工程数字孪生平台命令行管理工具

提供统一的命令行接口用于:
- 系统启动/停止
- 健康检查和诊断
- 配置管理
- 性能监控
- 数据管理
"""

import argparse
import sys
import os
import json
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime


@dataclass
class CLIConfig:
    """CLI配置"""
    verbose: bool = False
    output_format: str = "text"  # text, json
    config_path: str = "cyrp.yaml"
    db_path: str = ":memory:"


class OutputFormatter:
    """输出格式化器"""

    def __init__(self, format_type: str = "text"):
        self.format_type = format_type

    def print_header(self, title: str):
        """打印标题"""
        if self.format_type == "text":
            print(f"\n{'='*60}")
            print(f"  {title}")
            print(f"{'='*60}")

    def print_section(self, title: str):
        """打印章节"""
        if self.format_type == "text":
            print(f"\n--- {title} ---")

    def print_item(self, key: str, value: Any, indent: int = 0):
        """打印项目"""
        prefix = "  " * indent
        if self.format_type == "text":
            print(f"{prefix}{key}: {value}")

    def print_table(self, headers: List[str], rows: List[List[Any]]):
        """打印表格"""
        if self.format_type == "text":
            # 计算列宽
            widths = [len(h) for h in headers]
            for row in rows:
                for i, cell in enumerate(row):
                    widths[i] = max(widths[i], len(str(cell)))

            # 打印表头
            header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
            print(header_line)
            print("-" * len(header_line))

            # 打印数据行
            for row in rows:
                row_line = " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))
                print(row_line)
        else:
            print(json.dumps({"headers": headers, "rows": rows}, indent=2))

    def print_status(self, status: str, message: str):
        """打印状态"""
        icons = {
            "ok": "✓",
            "error": "✗",
            "warning": "⚠",
            "info": "ℹ"
        }
        icon = icons.get(status, "•")
        if self.format_type == "text":
            print(f"  [{icon}] {message}")

    def print_json(self, data: Any):
        """打印JSON"""
        print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


class SystemCommands:
    """系统命令"""

    def __init__(self, formatter: OutputFormatter):
        self.formatter = formatter

    def status(self) -> Dict[str, Any]:
        """获取系统状态"""
        from cyrp.launcher import CYRPIntegratedSystem, CYRPApplicationConfig

        self.formatter.print_header("CYRP System Status")

        config = CYRPApplicationConfig()
        config.db_path = ":memory:"
        system = CYRPIntegratedSystem(config)

        try:
            initialized = system.initialize()
            health = system.get_health_status()

            self.formatter.print_section("Health Status")
            self.formatter.print_item("Status", health.get('status', 'unknown'))
            self.formatter.print_item("Uptime", f"{health.get('uptime', 0):.1f}s")

            self.formatter.print_section("Components")
            for component, status in health.get('components', {}).items():
                status_str = "ok" if status else "error"
                self.formatter.print_status(status_str, component)

            system.stop()
            return health

        except Exception as e:
            self.formatter.print_status("error", f"Failed to get status: {e}")
            return {"status": "error", "error": str(e)}

    def start(self, background: bool = False) -> bool:
        """启动系统"""
        from cyrp.launcher import CYRPIntegratedSystem, CYRPApplicationConfig

        self.formatter.print_header("Starting CYRP System")

        config = CYRPApplicationConfig()
        system = CYRPIntegratedSystem(config)

        try:
            self.formatter.print_status("info", "Initializing components...")
            if system.initialize():
                self.formatter.print_status("ok", "System initialized successfully")

                if not background:
                    self.formatter.print_status("info", "Running in foreground (Ctrl+C to stop)")
                    try:
                        system.run()
                    except KeyboardInterrupt:
                        self.formatter.print_status("info", "Shutting down...")
                        system.stop()
                return True
            else:
                self.formatter.print_status("error", "Failed to initialize system")
                return False

        except Exception as e:
            self.formatter.print_status("error", f"Startup failed: {e}")
            return False

    def version(self) -> str:
        """显示版本信息"""
        self.formatter.print_header("CYRP Version Information")

        try:
            from cyrp import __version__
            version = __version__
        except ImportError:
            version = "1.0.0"

        self.formatter.print_item("Version", version)
        self.formatter.print_item("Python", sys.version.split()[0])
        self.formatter.print_item("Platform", sys.platform)

        # 检查依赖
        self.formatter.print_section("Dependencies")
        deps = ["numpy", "scipy", "aiohttp", "pyyaml"]
        for dep in deps:
            try:
                mod = __import__(dep)
                ver = getattr(mod, "__version__", "unknown")
                self.formatter.print_status("ok", f"{dep} {ver}")
            except ImportError:
                self.formatter.print_status("error", f"{dep} not installed")

        return version


class DiagnosticsCommands:
    """诊断命令"""

    def __init__(self, formatter: OutputFormatter):
        self.formatter = formatter

    def check_modules(self) -> Dict[str, bool]:
        """检查所有模块"""
        self.formatter.print_header("Module Health Check")

        modules = [
            ("cyrp.simulation.sensor_simulation", "Sensor Simulation"),
            ("cyrp.assimilation.data_assimilation", "Data Assimilation"),
            ("cyrp.prediction.state_prediction", "State Prediction"),
            ("cyrp.control.safety_interlocks", "Safety Interlocks"),
            ("cyrp.api.rest_api", "REST API"),
            ("cyrp.api.monitoring_endpoints", "Monitoring API"),
            ("cyrp.communication.websocket_server", "WebSocket Server"),
            ("cyrp.database.persistence_manager", "Persistence Manager"),
            ("cyrp.database.historian", "Historian Database"),
            ("cyrp.monitoring.dashboard_data", "Dashboard Data"),
            ("cyrp.eventbus.event_bus", "Event Bus"),
            ("cyrp.notification.notification_service", "Notification Service"),
            ("cyrp.launcher", "Integrated Launcher"),
        ]

        results = {}
        for module_path, name in modules:
            try:
                __import__(module_path)
                self.formatter.print_status("ok", name)
                results[name] = True
            except Exception as e:
                self.formatter.print_status("error", f"{name}: {e}")
                results[name] = False

        # 统计
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        self.formatter.print_section("Summary")
        self.formatter.print_item("Passed", f"{passed}/{total}")

        return results

    def check_sensors(self) -> Dict[str, Any]:
        """检查传感器系统"""
        self.formatter.print_header("Sensor System Diagnostics")

        try:
            from cyrp.simulation.sensor_simulation import (
                SensorSimulationManager, VirtualSensor,
                SensorCharacteristics, NoiseModel
            )

            # 创建测试传感器网络
            manager = SensorSimulationManager()
            network = manager.create_standard_tunnel_network()

            self.formatter.print_section("Sensor Network")
            self.formatter.print_item("Total Sensors", len(network.sensors))

            # 测试读取
            true_values = {f"P_{i}": 500000.0 for i in range(11)}
            readings = network.read_all(true_values, 0.1)

            self.formatter.print_section("Reading Test")
            self.formatter.print_item("Readings Obtained", len(readings))

            # 检查网络状态
            status = network.get_network_status()
            self.formatter.print_section("Network Status")
            self.formatter.print_item("Total Sensors", status.get('total_sensors', 0))
            self.formatter.print_item("Healthy Sensors", status.get('healthy_sensors', 0))

            return {
                "status": "ok",
                "sensor_count": len(network.sensors),
                "readings": len(readings)
            }

        except Exception as e:
            self.formatter.print_status("error", f"Sensor check failed: {e}")
            return {"status": "error", "error": str(e)}

    def check_prediction(self) -> Dict[str, Any]:
        """检查预测系统"""
        self.formatter.print_header("Prediction System Diagnostics")

        try:
            from cyrp.prediction.state_prediction import (
                ExponentialSmoothingPredictor, ARIMAPredictor,
                PhysicsBasedPredictor, EnsemblePredictor
            )
            import numpy as np

            # 测试数据
            test_data = [100 + i * 2 + np.random.normal(0, 5) for i in range(50)]

            predictors = [
                ("Exponential Smoothing", ExponentialSmoothingPredictor(alpha=0.3)),
                ("ARIMA", ARIMAPredictor()),
                ("Physics-Based", PhysicsBasedPredictor()),
            ]

            results = {}
            for name, predictor in predictors:
                try:
                    # 训练
                    for i, value in enumerate(test_data):
                        predictor.update(value, float(i))

                    # 预测
                    result = predictor.predict(5)
                    predictions = result.predictions

                    self.formatter.print_status("ok", f"{name}: {len(predictions)} predictions")
                    results[name] = {"status": "ok", "predictions": len(predictions)}

                except Exception as e:
                    self.formatter.print_status("error", f"{name}: {e}")
                    results[name] = {"status": "error", "error": str(e)}

            return results

        except Exception as e:
            self.formatter.print_status("error", f"Prediction check failed: {e}")
            return {"status": "error", "error": str(e)}

    def check_interlocks(self) -> Dict[str, Any]:
        """检查安全联锁系统"""
        self.formatter.print_header("Safety Interlock Diagnostics")

        try:
            from cyrp.control.safety_interlocks import (
                AntiVacuumInterlock, AntiOverpressureInterlock,
                AntiSurgeInterlock, InterlockCoordinator
            )

            interlocks = [
                ("Anti-Vacuum", AntiVacuumInterlock()),
                ("Anti-Overpressure", AntiOverpressureInterlock()),
                ("Anti-Surge", AntiSurgeInterlock()),
            ]

            results = {}
            for name, interlock in interlocks:
                try:
                    # 测试正常条件
                    triggered, action = interlock.check(500000.0, time.time())
                    status = "triggered" if triggered else "normal"
                    self.formatter.print_status("ok", f"{name}: {status}")
                    results[name] = {"status": "ok", "state": status}

                except Exception as e:
                    self.formatter.print_status("error", f"{name}: {e}")
                    results[name] = {"status": "error", "error": str(e)}

            return results

        except Exception as e:
            self.formatter.print_status("error", f"Interlock check failed: {e}")
            return {"status": "error", "error": str(e)}


class BenchmarkCommands:
    """性能基准测试命令"""

    def __init__(self, formatter: OutputFormatter):
        self.formatter = formatter

    def run_all(self) -> Dict[str, Any]:
        """运行所有基准测试"""
        self.formatter.print_header("CYRP Performance Benchmarks")

        results = {}
        results["sensor"] = self.benchmark_sensors()
        results["prediction"] = self.benchmark_prediction()
        results["persistence"] = self.benchmark_persistence()

        return results

    def benchmark_sensors(self, iterations: int = 1000) -> Dict[str, float]:
        """传感器性能基准"""
        self.formatter.print_section("Sensor Benchmark")

        try:
            from cyrp.simulation.sensor_simulation import (
                VirtualSensor, SensorCharacteristics, NoiseModel
            )

            char = SensorCharacteristics(
                sensor_type="pressure",
                measurement_range=(0, 1e6),
                noise_model=NoiseModel()
            )
            sensor = VirtualSensor("benchmark_sensor", char)

            # 单次读取
            start = time.perf_counter()
            for _ in range(iterations):
                sensor.read(500000.0, 0.1)
            elapsed = time.perf_counter() - start

            rate = iterations / elapsed
            self.formatter.print_item("Single Read Rate", f"{rate:.0f} reads/sec")
            self.formatter.print_item("Avg Latency", f"{elapsed/iterations*1000:.3f} ms")

            return {"rate": rate, "latency_ms": elapsed/iterations*1000}

        except Exception as e:
            self.formatter.print_status("error", f"Benchmark failed: {e}")
            return {"error": str(e)}

    def benchmark_prediction(self, iterations: int = 100) -> Dict[str, float]:
        """预测性能基准"""
        self.formatter.print_section("Prediction Benchmark")

        try:
            from cyrp.prediction.state_prediction import ExponentialSmoothingPredictor
            import numpy as np

            predictor = ExponentialSmoothingPredictor(alpha=0.3)

            # 准备数据
            for i in range(50):
                predictor.update(500000.0 + np.random.normal(0, 1000), float(i))

            # 预测基准
            start = time.perf_counter()
            for _ in range(iterations):
                predictor.predict(10)
            elapsed = time.perf_counter() - start

            rate = iterations / elapsed
            self.formatter.print_item("Prediction Rate", f"{rate:.0f} predictions/sec")
            self.formatter.print_item("Avg Latency", f"{elapsed/iterations*1000:.3f} ms")

            return {"rate": rate, "latency_ms": elapsed/iterations*1000}

        except Exception as e:
            self.formatter.print_status("error", f"Benchmark failed: {e}")
            return {"error": str(e)}

    def benchmark_persistence(self, iterations: int = 1000) -> Dict[str, float]:
        """持久化性能基准"""
        self.formatter.print_section("Persistence Benchmark")

        try:
            from cyrp.database.persistence_manager import PersistenceManager
            from cyrp.database.historian import SQLiteBackend

            backend = SQLiteBackend(":memory:")
            persistence = PersistenceManager(backend)

            # 写入基准
            start = time.perf_counter()
            base_time = time.time()
            for i in range(iterations):
                persistence.record_metric("benchmark", 100.0 + i, timestamp=base_time + i)
            elapsed = time.perf_counter() - start

            write_rate = iterations / elapsed
            self.formatter.print_item("Write Rate", f"{write_rate:.0f} writes/sec")

            # 刷新基准
            start = time.perf_counter()
            persistence.flush_all()
            flush_time = time.perf_counter() - start
            self.formatter.print_item("Flush Time", f"{flush_time*1000:.1f} ms")

            persistence.close()

            return {"write_rate": write_rate, "flush_ms": flush_time*1000}

        except Exception as e:
            self.formatter.print_status("error", f"Benchmark failed: {e}")
            return {"error": str(e)}


class DataCommands:
    """数据管理命令"""

    def __init__(self, formatter: OutputFormatter):
        self.formatter = formatter

    def export_metrics(self, output_path: str, format: str = "json") -> bool:
        """导出指标数据"""
        self.formatter.print_header("Export Metrics")

        try:
            from cyrp.monitoring.dashboard_data import DashboardDataProvider

            dashboard = DashboardDataProvider()

            # 模拟一些数据
            import time
            for i in range(10):
                dashboard.record_metric("demo_metric", 100.0 + i)

            current = dashboard.get_current_metrics()

            if format == "json":
                data = {
                    name: {
                        "value": metric.value,
                        "timestamp": metric.timestamp
                    }
                    for name, metric in current.items()
                }

                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2)

                self.formatter.print_status("ok", f"Exported to {output_path}")
                return True

        except Exception as e:
            self.formatter.print_status("error", f"Export failed: {e}")
            return False

    def stats(self) -> Dict[str, Any]:
        """显示数据统计"""
        self.formatter.print_header("Data Statistics")

        try:
            from cyrp.database.persistence_manager import PersistenceManager
            from cyrp.database.historian import SQLiteBackend

            backend = SQLiteBackend(":memory:")
            persistence = PersistenceManager(backend)

            stats = persistence.get_stats()

            self.formatter.print_section("Buffer Statistics")
            for key, value in stats.items():
                self.formatter.print_item(key, value)

            persistence.close()
            return stats

        except Exception as e:
            self.formatter.print_status("error", f"Stats failed: {e}")
            return {"error": str(e)}


class CYRPCLI:
    """CYRP命令行接口主类"""

    def __init__(self):
        self.config = CLIConfig()
        self.formatter = OutputFormatter()

        # 命令处理器
        self.system = SystemCommands(self.formatter)
        self.diagnostics = DiagnosticsCommands(self.formatter)
        self.benchmark = BenchmarkCommands(self.formatter)
        self.data = DataCommands(self.formatter)

    def create_parser(self) -> argparse.ArgumentParser:
        """创建命令行解析器"""
        parser = argparse.ArgumentParser(
            prog="cyrp",
            description="CYRP Digital Twin Platform CLI - 穿黄工程数字孪生平台命令行工具",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  cyrp status          Show system status
  cyrp start           Start the CYRP system
  cyrp diag modules    Check all modules
  cyrp bench all       Run all benchmarks
  cyrp version         Show version info
            """
        )

        parser.add_argument(
            "-v", "--verbose",
            action="store_true",
            help="Enable verbose output"
        )
        parser.add_argument(
            "--json",
            action="store_true",
            help="Output in JSON format"
        )

        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # system commands
        subparsers.add_parser("status", help="Show system status")
        subparsers.add_parser("version", help="Show version information")

        start_parser = subparsers.add_parser("start", help="Start the system")
        start_parser.add_argument(
            "-d", "--daemon",
            action="store_true",
            help="Run in background"
        )

        # diagnostics commands
        diag_parser = subparsers.add_parser("diag", help="Run diagnostics")
        diag_parser.add_argument(
            "target",
            choices=["modules", "sensors", "prediction", "interlocks", "all"],
            help="Diagnostic target"
        )

        # benchmark commands
        bench_parser = subparsers.add_parser("bench", help="Run benchmarks")
        bench_parser.add_argument(
            "target",
            choices=["sensors", "prediction", "persistence", "all"],
            help="Benchmark target"
        )
        bench_parser.add_argument(
            "-n", "--iterations",
            type=int,
            default=1000,
            help="Number of iterations"
        )

        # data commands
        data_parser = subparsers.add_parser("data", help="Data management")
        data_subparsers = data_parser.add_subparsers(dest="data_command")

        export_parser = data_subparsers.add_parser("export", help="Export data")
        export_parser.add_argument("output", help="Output file path")
        export_parser.add_argument(
            "-f", "--format",
            choices=["json", "csv"],
            default="json",
            help="Output format"
        )

        data_subparsers.add_parser("stats", help="Show data statistics")

        return parser

    def run(self, args: Optional[List[str]] = None) -> int:
        """运行CLI"""
        parser = self.create_parser()
        parsed = parser.parse_args(args)

        # 配置
        if parsed.verbose:
            self.config.verbose = True
        if parsed.json:
            self.config.output_format = "json"
            self.formatter = OutputFormatter("json")
            self.system.formatter = self.formatter
            self.diagnostics.formatter = self.formatter
            self.benchmark.formatter = self.formatter
            self.data.formatter = self.formatter

        # 执行命令
        if parsed.command == "status":
            self.system.status()
        elif parsed.command == "version":
            self.system.version()
        elif parsed.command == "start":
            self.system.start(background=parsed.daemon)
        elif parsed.command == "diag":
            if parsed.target == "modules":
                self.diagnostics.check_modules()
            elif parsed.target == "sensors":
                self.diagnostics.check_sensors()
            elif parsed.target == "prediction":
                self.diagnostics.check_prediction()
            elif parsed.target == "interlocks":
                self.diagnostics.check_interlocks()
            elif parsed.target == "all":
                self.diagnostics.check_modules()
                self.diagnostics.check_sensors()
                self.diagnostics.check_prediction()
                self.diagnostics.check_interlocks()
        elif parsed.command == "bench":
            if parsed.target == "sensors":
                self.benchmark.benchmark_sensors(parsed.iterations)
            elif parsed.target == "prediction":
                self.benchmark.benchmark_prediction(parsed.iterations)
            elif parsed.target == "persistence":
                self.benchmark.benchmark_persistence(parsed.iterations)
            elif parsed.target == "all":
                self.benchmark.run_all()
        elif parsed.command == "data":
            if parsed.data_command == "export":
                self.data.export_metrics(parsed.output, parsed.format)
            elif parsed.data_command == "stats":
                self.data.stats()
        else:
            parser.print_help()
            return 1

        return 0


def main():
    """CLI入口点"""
    cli = CYRPCLI()
    sys.exit(cli.run())


if __name__ == "__main__":
    main()
