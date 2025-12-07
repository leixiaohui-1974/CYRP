# CYRP - Crossing Yellow River Project

南水北调中线穿黄工程全场景自主运行在环测试与多智能体系统平台

## Overview

CYRP is a comprehensive simulation and control platform for the South-to-North Water Diversion Middle Route Yellow River Crossing Project. The platform provides:

- **High-fidelity Physical Simulation**: Saint-Venant equations, Preissmann slot model, water hammer MOC
- **Multi-modal Perception System**: DAS, DTS, MEMS, CV sensor fusion with EKF
- **32 Scenario Coverage**: Nominal, Transition, and Emergency operational domains
- **Hierarchical Distributed MPC**: Global optimization with local execution
- **Multi-Agent Autonomous Operation**: Coordinator, Perception, Control, Safety, Scenario agents
- **Hardware-in-the-Loop Testing**: Complete HIL framework with virtual PLC
- **Digital Twin**: Real-time synchronization, prediction, and what-if analysis

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Multi-Agent System                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐  │
│  │ Coordinator │ │  Perception │ │   Control   │ │  Safety   │  │
│  │   Agent     │ │   Agent     │ │   Agent     │ │  Agent    │  │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                    Control Layer (HD-MPC)                        │
│  ┌───────────────────────┐  ┌─────────────────────────────────┐ │
│  │   Global Optimizer    │  │      Safety Interlocks          │ │
│  │   (LTV/NMPC/Robust)   │  │  Anti-vacuum/Surge/Overpressure │ │
│  └───────────────────────┘  └─────────────────────────────────┘ │
│  ┌───────────────────────┐  ┌─────────────────────────────────┐ │
│  │   Local Executors     │  │      Cascade PID Control        │ │
│  │   (Per-tunnel MPC)    │  │      with Anti-windup           │ │
│  └───────────────────────┘  └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                    Perception Layer                              │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌───────────┐  │
│  │   DAS   │ │   DTS   │ │  MEMS   │ │ Pressure│ │ Flow Meter│  │
│  │ Sensor  │ │ Sensor  │ │ Array   │ │ Sensors │ │           │  │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └───────────┘  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              EKF-based Multi-source Fusion                  ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                    Physical System Layer                         │
│  ┌─────────────────────────┐  ┌───────────────────────────────┐ │
│  │    Hydraulic Model      │  │    Structural Model           │ │
│  │  - Saint-Venant Eqs     │  │  - Stress Analysis            │ │
│  │  - Preissmann Slot      │  │  - Buckling Assessment        │ │
│  │  - Water Hammer MOC     │  │  - Seismic Response           │ │
│  └─────────────────────────┘  └───────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/cyrp/cyrp.git
cd cyrp

# Install dependencies
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

## Usage

### Command Line Interface

```bash
# Run simulation
python main.py simulate --duration 3600 --scenario S1-A

# Run HIL tests
python main.py test --type all

# Show system information
python main.py info
```

### Python API

```python
from cyrp import (
    PhysicalSystem,
    PerceptionSystem,
    HDMPCController,
    MultiAgentSystem,
    HILTestFramework,
    DigitalTwin,
)

# Initialize physical system
physical_system = PhysicalSystem()
physical_system.reset(initial_flow=265.0)

# Initialize multi-agent system
mas = MultiAgentSystem()

# Run autonomous operation
results = mas.run(
    environment_generator=lambda t: {...},
    duration=3600,
    dt=0.1
)
```

## Scenario Coverage

### Domain 1: Nominal Operation (S1-S2)
- **S1-A/B/C**: Single-tunnel nominal operation (high/medium/low flow)
- **S2-A/B/C**: Dual-tunnel nominal operation (high/medium/low flow)

### Domain 2: Transition Operation (S3-S4)
- **S3-A**: Planned switch (primary to backup)
- **S3-B**: Emergency switch
- **S4-A/B/C**: Maintenance modes

### Domain 3: Emergency Operation (S5-S7)
- **S5-A/B/C/D**: Leakage scenarios (minor to major)
- **S6-A/B/C**: Seismic events (VI/VII/VIII intensity)
- **S7**: Integrated emergency response

## Key Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| Tunnel Length | 4,250 | m |
| Tunnel Diameter | 7.0 | m |
| Design Flow | 265 | m³/s |
| Design Velocity | 3.44 | m/s |
| Burial Depth | 30-70 | m |
| Max Pressure | 1.0 | MPa |
| Wave Speed | 1,000 | m/s |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_hydraulic_model.py -v

# Run with coverage
pytest tests/ --cov=cyrp --cov-report=html
```

## Project Structure

```
cyrp/
├── core/               # Physical models
│   ├── hydraulic_model.py    # Saint-Venant, Preissmann, MOC
│   ├── structural_model.py   # Stress, buckling, seismic
│   └── physical_system.py    # Integrated system
├── perception/         # Sensor and fusion
│   ├── sensors.py           # DAS, DTS, MEMS, etc.
│   ├── fusion.py            # EKF-based fusion
│   └── classifier.py        # Scenario classification
├── control/            # Control algorithms
│   ├── mpc_controller.py    # LTV, NMPC, Robust MPC
│   ├── pid_controller.py    # Cascade PID
│   ├── safety_interlocks.py # Safety systems
│   └── hdmpc.py             # Hierarchical distributed MPC
├── agents/             # Multi-agent system
│   ├── coordinator_agent.py # Master scheduler
│   ├── perception_agent.py  # Perception processing
│   ├── control_agent.py     # Control execution
│   └── safety_agent.py      # Safety monitoring
├── scenarios/          # Scenario management
│   ├── scenario_definitions.py  # 32 scenario specs
│   ├── scenario_manager.py      # State machine
│   └── scenario_generator.py    # Test generation
├── hil/                # Hardware-in-the-loop
│   ├── hil_framework.py     # HIL test framework
│   ├── simulation_engine.py # Simulation core
│   └── virtual_plc.py       # Virtual Siemens S7-1500R
├── digital_twin/       # Digital twin
│   └── digital_twin.py      # Sync, predict, what-if
└── utils/              # Utilities
    ├── logger.py            # Logging
    └── config.py            # Configuration
```

## License

MIT License

## References

1. 南水北调中线穿黄工程运行设计域(ODD)定义与多模态感知系统
2. 分层分布式MPC控制架构设计
3. 工程实际操作流程规范
