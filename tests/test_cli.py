"""
CLI命令行工具测试
Tests for CYRP Command Line Interface
"""

import pytest
import json
import tempfile
import os
from io import StringIO
from unittest.mock import patch


class TestOutputFormatter:
    """测试输出格式化器"""

    def test_text_format_header(self, capsys):
        """测试文本格式标题"""
        from cyrp.cli import OutputFormatter

        formatter = OutputFormatter("text")
        formatter.print_header("Test Header")

        captured = capsys.readouterr()
        assert "Test Header" in captured.out
        assert "=" in captured.out

    def test_text_format_status(self, capsys):
        """测试状态输出"""
        from cyrp.cli import OutputFormatter

        formatter = OutputFormatter("text")
        formatter.print_status("ok", "Test passed")

        captured = capsys.readouterr()
        assert "Test passed" in captured.out
        assert "✓" in captured.out

    def test_text_format_item(self, capsys):
        """测试项目输出"""
        from cyrp.cli import OutputFormatter

        formatter = OutputFormatter("text")
        formatter.print_item("Key", "Value")

        captured = capsys.readouterr()
        assert "Key: Value" in captured.out

    def test_json_format(self, capsys):
        """测试JSON格式输出"""
        from cyrp.cli import OutputFormatter

        formatter = OutputFormatter("json")
        formatter.print_json({"test": "data"})

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["test"] == "data"

    def test_table_output(self, capsys):
        """测试表格输出"""
        from cyrp.cli import OutputFormatter

        formatter = OutputFormatter("text")
        formatter.print_table(
            ["Name", "Value"],
            [["test1", 100], ["test2", 200]]
        )

        captured = capsys.readouterr()
        assert "Name" in captured.out
        assert "test1" in captured.out
        assert "100" in captured.out


class TestCLIConfig:
    """测试CLI配置"""

    def test_default_config(self):
        """测试默认配置"""
        from cyrp.cli import CLIConfig

        config = CLIConfig()
        assert config.verbose is False
        assert config.output_format == "text"

    def test_custom_config(self):
        """测试自定义配置"""
        from cyrp.cli import CLIConfig

        config = CLIConfig(verbose=True, output_format="json")
        assert config.verbose is True
        assert config.output_format == "json"


class TestSystemCommands:
    """测试系统命令"""

    def test_version_command(self, capsys):
        """测试版本命令"""
        from cyrp.cli import SystemCommands, OutputFormatter

        formatter = OutputFormatter("text")
        cmd = SystemCommands(formatter)
        version = cmd.version()

        captured = capsys.readouterr()
        assert "Version" in captured.out
        assert "Python" in captured.out

    def test_status_command(self, capsys):
        """测试状态命令"""
        from cyrp.cli import SystemCommands, OutputFormatter

        formatter = OutputFormatter("text")
        cmd = SystemCommands(formatter)
        status = cmd.status()

        assert "status" in status
        captured = capsys.readouterr()
        assert "Status" in captured.out


class TestDiagnosticsCommands:
    """测试诊断命令"""

    def test_check_modules(self, capsys):
        """测试模块检查"""
        from cyrp.cli import DiagnosticsCommands, OutputFormatter

        formatter = OutputFormatter("text")
        cmd = DiagnosticsCommands(formatter)
        results = cmd.check_modules()

        assert isinstance(results, dict)
        assert len(results) > 0

        captured = capsys.readouterr()
        assert "Module Health Check" in captured.out

    def test_check_sensors(self, capsys):
        """测试传感器检查"""
        from cyrp.cli import DiagnosticsCommands, OutputFormatter

        formatter = OutputFormatter("text")
        cmd = DiagnosticsCommands(formatter)
        results = cmd.check_sensors()

        assert "status" in results
        assert results["status"] == "ok"

    def test_check_prediction(self, capsys):
        """测试预测系统检查"""
        from cyrp.cli import DiagnosticsCommands, OutputFormatter

        formatter = OutputFormatter("text")
        cmd = DiagnosticsCommands(formatter)
        results = cmd.check_prediction()

        assert isinstance(results, dict)
        # 至少有一个预测器
        assert len(results) > 0

    def test_check_interlocks(self, capsys):
        """测试安全联锁检查"""
        from cyrp.cli import DiagnosticsCommands, OutputFormatter

        formatter = OutputFormatter("text")
        cmd = DiagnosticsCommands(formatter)
        results = cmd.check_interlocks()

        assert isinstance(results, dict)
        assert len(results) > 0


class TestBenchmarkCommands:
    """测试基准测试命令"""

    def test_sensor_benchmark(self, capsys):
        """测试传感器基准"""
        from cyrp.cli import BenchmarkCommands, OutputFormatter

        formatter = OutputFormatter("text")
        cmd = BenchmarkCommands(formatter)
        results = cmd.benchmark_sensors(iterations=100)

        assert "rate" in results
        assert results["rate"] > 0

    def test_prediction_benchmark(self, capsys):
        """测试预测基准"""
        from cyrp.cli import BenchmarkCommands, OutputFormatter

        formatter = OutputFormatter("text")
        cmd = BenchmarkCommands(formatter)
        results = cmd.benchmark_prediction(iterations=50)

        assert "rate" in results
        assert results["rate"] > 0

    def test_persistence_benchmark(self, capsys):
        """测试持久化基准"""
        from cyrp.cli import BenchmarkCommands, OutputFormatter

        formatter = OutputFormatter("text")
        cmd = BenchmarkCommands(formatter)
        results = cmd.benchmark_persistence(iterations=100)

        assert "write_rate" in results
        assert results["write_rate"] > 0

    def test_run_all_benchmarks(self, capsys):
        """测试运行所有基准"""
        from cyrp.cli import BenchmarkCommands, OutputFormatter

        formatter = OutputFormatter("text")
        cmd = BenchmarkCommands(formatter)
        results = cmd.run_all()

        assert "sensor" in results
        assert "prediction" in results
        assert "persistence" in results


class TestDataCommands:
    """测试数据管理命令"""

    def test_data_stats(self, capsys):
        """测试数据统计"""
        from cyrp.cli import DataCommands, OutputFormatter

        formatter = OutputFormatter("text")
        cmd = DataCommands(formatter)
        stats = cmd.stats()

        assert isinstance(stats, dict)

    def test_export_metrics(self, capsys):
        """测试导出指标"""
        from cyrp.cli import DataCommands, OutputFormatter

        formatter = OutputFormatter("text")
        cmd = DataCommands(formatter)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name

        try:
            result = cmd.export_metrics(output_path, format="json")
            assert result is True

            # 验证文件内容
            with open(output_path, 'r') as f:
                data = json.load(f)
            assert isinstance(data, dict)

        finally:
            os.unlink(output_path)


class TestCYRPCLI:
    """测试CLI主类"""

    def test_parser_creation(self):
        """测试解析器创建"""
        from cyrp.cli import CYRPCLI

        cli = CYRPCLI()
        parser = cli.create_parser()

        assert parser is not None
        assert parser.prog == "cyrp"

    def test_version_via_cli(self, capsys):
        """测试通过CLI运行版本命令"""
        from cyrp.cli import CYRPCLI

        cli = CYRPCLI()
        result = cli.run(["version"])

        assert result == 0
        captured = capsys.readouterr()
        assert "Version" in captured.out

    def test_diag_modules_via_cli(self, capsys):
        """测试通过CLI运行模块诊断"""
        from cyrp.cli import CYRPCLI

        cli = CYRPCLI()
        result = cli.run(["diag", "modules"])

        assert result == 0
        captured = capsys.readouterr()
        assert "Module Health Check" in captured.out

    def test_bench_sensors_via_cli(self, capsys):
        """测试通过CLI运行传感器基准"""
        from cyrp.cli import CYRPCLI

        cli = CYRPCLI()
        result = cli.run(["bench", "sensors", "-n", "100"])

        assert result == 0
        captured = capsys.readouterr()
        assert "Sensor Benchmark" in captured.out

    def test_json_output_flag(self, capsys):
        """测试JSON输出标志"""
        from cyrp.cli import CYRPCLI

        cli = CYRPCLI()
        cli.run(["--json", "data", "stats"])

        # JSON模式不会打印文本格式
        captured = capsys.readouterr()
        # 输出应该是JSON格式或空（依赖具体实现）

    def test_help_on_no_command(self, capsys):
        """测试无命令时显示帮助"""
        from cyrp.cli import CYRPCLI

        cli = CYRPCLI()
        result = cli.run([])

        assert result == 1  # 无命令应返回1


class TestCLIIntegration:
    """CLI集成测试"""

    def test_full_diagnostic_workflow(self, capsys):
        """测试完整诊断工作流"""
        from cyrp.cli import CYRPCLI

        cli = CYRPCLI()

        # 运行所有诊断
        result = cli.run(["diag", "all"])
        assert result == 0

        captured = capsys.readouterr()
        # 验证所有诊断都运行了
        assert "Module Health Check" in captured.out
        assert "Sensor System" in captured.out
        assert "Prediction System" in captured.out
        assert "Safety Interlock" in captured.out

    def test_benchmark_workflow(self, capsys):
        """测试基准测试工作流"""
        from cyrp.cli import CYRPCLI

        cli = CYRPCLI()
        result = cli.run(["bench", "all"])

        assert result == 0
        captured = capsys.readouterr()
        assert "Sensor Benchmark" in captured.out
        assert "Prediction Benchmark" in captured.out
        assert "Persistence Benchmark" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
