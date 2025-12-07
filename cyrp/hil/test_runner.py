"""
Test Runner for CYRP HIL Testing.
穿黄工程测试运行器
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from enum import Enum
import time


class TestStatus(Enum):
    """测试状态"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class TestCase:
    """测试用例"""
    name: str
    description: str = ""
    setup: Optional[Callable] = None
    test_func: Optional[Callable] = None
    teardown: Optional[Callable] = None
    timeout: float = 300.0
    tags: List[str] = field(default_factory=list)


@dataclass
class TestResult:
    """测试结果"""
    test_name: str
    status: TestStatus = TestStatus.PENDING
    duration: float = 0.0
    message: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class TestRunner:
    """
    测试运行器

    管理和执行测试用例
    """

    def __init__(self):
        """初始化测试运行器"""
        self.test_cases: List[TestCase] = []
        self.results: List[TestResult] = []

        # 全局设置/清理
        self.global_setup: Optional[Callable] = None
        self.global_teardown: Optional[Callable] = None

        # 回调
        self.on_test_start: Optional[Callable] = None
        self.on_test_end: Optional[Callable] = None

    def add_test(self, test_case: TestCase):
        """添加测试用例"""
        self.test_cases.append(test_case)

    def add_tests(self, test_cases: List[TestCase]):
        """批量添加测试用例"""
        self.test_cases.extend(test_cases)

    def run_test(self, test_case: TestCase) -> TestResult:
        """
        运行单个测试

        Args:
            test_case: 测试用例

        Returns:
            测试结果
        """
        result = TestResult(test_name=test_case.name)

        if self.on_test_start:
            self.on_test_start(test_case)

        start_time = time.time()
        result.status = TestStatus.RUNNING

        try:
            # Setup
            if test_case.setup:
                test_case.setup()

            # Run test
            if test_case.test_func:
                test_result = test_case.test_func()
                if isinstance(test_result, dict):
                    result.metrics.update(test_result)

            result.status = TestStatus.PASSED
            result.message = "Test passed"

        except AssertionError as e:
            result.status = TestStatus.FAILED
            result.message = str(e)
            result.errors.append(str(e))

        except Exception as e:
            result.status = TestStatus.ERROR
            result.message = f"Error: {str(e)}"
            result.errors.append(str(e))

        finally:
            # Teardown
            try:
                if test_case.teardown:
                    test_case.teardown()
            except Exception as e:
                result.logs.append(f"Teardown error: {str(e)}")

            result.duration = time.time() - start_time

        if self.on_test_end:
            self.on_test_end(test_case, result)

        return result

    def run_all(self, tags: Optional[List[str]] = None) -> List[TestResult]:
        """
        运行所有测试

        Args:
            tags: 过滤标签

        Returns:
            测试结果列表
        """
        self.results = []

        # 全局设置
        if self.global_setup:
            try:
                self.global_setup()
            except Exception as e:
                print(f"Global setup failed: {e}")
                return self.results

        # 过滤测试
        tests_to_run = self.test_cases
        if tags:
            tests_to_run = [
                t for t in self.test_cases
                if any(tag in t.tags for tag in tags)
            ]

        # 运行测试
        for test_case in tests_to_run:
            result = self.run_test(test_case)
            self.results.append(result)

        # 全局清理
        if self.global_teardown:
            try:
                self.global_teardown()
            except Exception as e:
                print(f"Global teardown failed: {e}")

        return self.results

    def get_summary(self) -> Dict[str, Any]:
        """获取测试摘要"""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == TestStatus.PASSED)
        failed = sum(1 for r in self.results if r.status == TestStatus.FAILED)
        errors = sum(1 for r in self.results if r.status == TestStatus.ERROR)
        skipped = sum(1 for r in self.results if r.status == TestStatus.SKIPPED)

        total_duration = sum(r.duration for r in self.results)

        return {
            'total': total,
            'passed': passed,
            'failed': failed,
            'errors': errors,
            'skipped': skipped,
            'pass_rate': passed / total if total > 0 else 0,
            'total_duration': total_duration
        }

    def print_report(self):
        """打印测试报告"""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("Test Results")
        print("=" * 60)

        for result in self.results:
            status_icon = {
                TestStatus.PASSED: "[PASS]",
                TestStatus.FAILED: "[FAIL]",
                TestStatus.ERROR: "[ERR ]",
                TestStatus.SKIPPED: "[SKIP]"
            }.get(result.status, "[????]")

            print(f"{status_icon} {result.test_name} ({result.duration:.2f}s)")
            if result.status != TestStatus.PASSED and result.message:
                print(f"       {result.message}")

        print("-" * 60)
        print(f"Total: {summary['total']} | "
              f"Passed: {summary['passed']} | "
              f"Failed: {summary['failed']} | "
              f"Errors: {summary['errors']}")
        print(f"Pass Rate: {summary['pass_rate']*100:.1f}%")
        print(f"Duration: {summary['total_duration']:.2f}s")
        print("=" * 60)
