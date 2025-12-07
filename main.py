#!/usr/bin/env python3
"""
CYRP - Crossing Yellow River Project
南水北调中线穿黄工程全场景自主运行在环测试与多智能体系统平台

主入口文件
"""

import argparse
import sys
from typing import Optional

from cyrp import (
    PhysicalSystem,
    PerceptionSystem,
    HDMPCController,
    MultiAgentSystem,
    HILTestFramework,
    DigitalTwin,
    ScenarioManager,
    ScenarioGenerator,
)
from cyrp.scenarios import ScenarioType
from cyrp.utils import setup_logger, load_config


def run_simulation(duration: float = 3600.0, scenario: str = "S1-A"):
    """
    运行仿真

    Args:
        duration: 仿真时长 (s)
        scenario: 场景类型
    """
    logger = setup_logger("cyrp")
    logger.info(f"Starting simulation: duration={duration}s, scenario={scenario}")

    # 加载配置
    config = load_config()

    # 初始化系统
    physical_system = PhysicalSystem()
    physical_system.reset(initial_flow=config.design_flow)

    # 初始化多智能体系统
    mas = MultiAgentSystem()

    # 运行仿真
    def environment_generator(t):
        return {
            'system_state': physical_system.state,
            'sensor_data': {
                'pressure_max': physical_system.state.hydraulic.P_max,
                'pressure_min': physical_system.state.hydraulic.P_min,
                'gate_positions': [
                    physical_system.state.actuators.gate_inlet_1,
                    physical_system.state.actuators.gate_inlet_2
                ]
            },
            'time': t,
            'dt': config.simulation_dt
        }

    def step_callback(t, result):
        if int(t) % 60 == 0:
            logger.info(
                f"t={t:.0f}s: Q={physical_system.state.hydraulic.total_flow:.1f} m³/s, "
                f"scenario={mas.get_scenario()}, risk={mas.get_risk_level()}"
            )

        # 应用控制
        from cyrp.core.physical_system import ControlCommand
        control = mas.get_control_output()
        cmd = ControlCommand(
            gate_inlet_1_target=control['gate_1'],
            gate_inlet_2_target=control['gate_2']
        )
        physical_system.step(cmd, config.simulation_dt)

    results = mas.run(
        environment_generator,
        duration=duration,
        dt=config.simulation_dt,
        callback=step_callback
    )

    logger.info(f"Simulation completed: {len(results)} steps")

    # 打印统计
    status = mas.get_status()
    logger.info(f"Final status: {status}")

    return results


def run_hil_test(test_type: str = "all"):
    """
    运行在环测试

    Args:
        test_type: 测试类型 ("all", "nominal", "emergency", ...)
    """
    logger = setup_logger("cyrp")
    logger.info(f"Starting HIL test: type={test_type}")

    # 初始化测试框架
    hil = HILTestFramework()
    generator = ScenarioGenerator()

    if test_type == "all":
        tests = generator.generate_full_coverage_suite()
    elif test_type == "nominal":
        tests = [generator.generate_nominal_test()]
    elif test_type == "switch":
        tests = [generator.generate_tunnel_switch_test()]
    elif test_type == "leakage":
        tests = [generator.generate_leakage_test()]
    elif test_type == "earthquake":
        tests = [generator.generate_earthquake_test()]
    else:
        tests = [generator.generate_random_scenario()]

    # 运行测试
    results = hil.run_suite(tests)

    # 生成报告
    report = hil.generate_report()
    print(report)

    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="CYRP - 穿黄工程全场景自主运行平台"
    )
    subparsers = parser.add_subparsers(dest='command', help='命令')

    # 仿真命令
    sim_parser = subparsers.add_parser('simulate', help='运行仿真')
    sim_parser.add_argument(
        '--duration', '-d', type=float, default=3600.0,
        help='仿真时长 (秒)'
    )
    sim_parser.add_argument(
        '--scenario', '-s', type=str, default='S1-A',
        help='场景类型'
    )

    # 测试命令
    test_parser = subparsers.add_parser('test', help='运行在环测试')
    test_parser.add_argument(
        '--type', '-t', type=str, default='all',
        choices=['all', 'nominal', 'switch', 'leakage', 'earthquake', 'random'],
        help='测试类型'
    )

    # 信息命令
    info_parser = subparsers.add_parser('info', help='显示系统信息')

    args = parser.parse_args()

    if args.command == 'simulate':
        run_simulation(args.duration, args.scenario)
    elif args.command == 'test':
        run_hil_test(args.type)
    elif args.command == 'info':
        print("CYRP - Crossing Yellow River Project")
        print("南水北调中线穿黄工程全场景自主运行在环测试与多智能体系统平台")
        print()
        print("版本: 1.0.0")
        print()
        print("模块:")
        print("  - 核心物理模型 (水力学、结构动力学)")
        print("  - 多模态感知系统 (DAS, DTS, MEMS, CV)")
        print("  - 场景管理器 (32种场景)")
        print("  - 分层分布式MPC控制器")
        print("  - 多智能体协同系统")
        print("  - 在环测试框架")
        print("  - 数字孪生")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
