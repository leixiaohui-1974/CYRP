# Report I: South-to-North Water Diversion Middle Route Yellow River Crossing Project
# Data Source: "Yellow River Crossing Project: Engineering Parameters and Modeling Basic Data Manual"

import math

# 1. Constants
G = 9.81  # m/s^2
RHO_WATER = 1000.0  # kg/m^3
PATM = 101325.0  # Pa (Standard Atmosphere)

# 2. Hydraulic Design Parameters
Q_DESIGN_TOTAL = 265.0  # m^3/s (Total Design Flow)
Q_MAX_TOTAL = 305.0     # m^3/s (Total Max Flow)
Q_DESIGN_SINGLE = 132.5 # m^3/s (Single Tube Design)
Q_MAX_SINGLE = 152.5    # m^3/s (Single Tube Max)

Z_IN_DESIGN = 106.05    # m (Inlet Design Water Level)
Z_OUT_DESIGN = 104.79   # m (Outlet Design Water Level)
Z_IN_MAX = 107.06       # m (Inlet Max Level)
Z_OUT_MAX = 104.43      # m (Outlet Max Level)

HEAD_LOSS_DESIGN = 1.26 # m (Design Head Loss)
HEAD_LOSS_CHECK = 2.63  # m (Check Head Loss)

# 3. Tunnel Geometry
NUM_TUBES = 2
TUBE_LENGTH = 4250.0    # m (Total Length including transitions)
RIVER_CROSSING_LENGTH = 3450.0 # m
DIAMETER = 7.0          # m (Inner Diameter)
RADIUS = DIAMETER / 2.0
AREA = math.pi * RADIUS**2 # 38.4845 m^2

# Elevations (Slope 1:12 Inlet, 1:3000 Body)
# Simplified Profile for 1D Model:
Z_BOTTOM_INLET = 85.08  # m
Z_BOTTOM_OUTLET = 84.60 # m
SLOPE_AVG = (Z_BOTTOM_INLET - Z_BOTTOM_OUTLET) / TUBE_LENGTH # approx 0.000113

# Roughness
MANNING_N_CONCRETE = 0.014
MANNING_N_MUSSELS = 0.018

# 4. Actuators
# Inlet Gate
GATE_WIDTH = 7.0        # m
GATE_HEIGHT = 7.5       # m
GATE_SPEED_NORMAL = 0.2 / 60.0 # m/s (0.2 m/min)
GATE_SPEED_FAST = 0.5 / 60.0   # m/s (Emergency)

# Bypass Valve (Filling)
VALVE_BYPASS_DIAMETER = 1.4 # m (DN1400)
VALVE_BYPASS_AREA = math.pi * (VALVE_BYPASS_DIAMETER/2)**2

# Drain Pump
PUMP_CAPACITY = 500.0 / 3600.0 # m^3/s (500 m^3/h)

# 5. Environment & Geotechnics
RIVER_BED_DEPTH = 70.0  # m (Depth from river bed to tunnel)
SOIL_COVER_MIN = 23.0   # m
YELLOW_RIVER_LEVEL_DESIGN = 96.22 # m (300-year)
YELLOW_RIVER_LEVEL_CHECK = 98.70  # m (1000-year)

# 6. Advanced Simulation Parameters (Preissmann Slot)
WAVE_SPEED_A = 1000.0   # m/s (Equivalent wave speed in pressurized pipe)
# Slot Width B_slot = gA / a^2
SLOT_WIDTH = (G * AREA) / (WAVE_SPEED_A**2) # approx 0.000377 m

# Discretization
NODES = 50             # Number of spatial nodes (Higher for fidelity)
DX = TUBE_LENGTH / NODES
DT = 0.05              # s (Time step, satisfy CFL condition: dt < dx / (v+a))
# CFL check: dx = 4250/50 = 85m. (v+a) ~ 1005 m/s. dt < 0.08s.
