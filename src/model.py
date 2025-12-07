import numpy as np
import math
from src.config import *

# --- Helper Classes ---

class HydraulicNode:
    """
    Represents a specific cross-section of the tunnel.
    """
    def __init__(self, x_pos, z_bottom, section_type='tunnel'):
        self.x = x_pos
        self.z_bottom = z_bottom
        self.section_type = section_type # 'inlet_ramp', 'tunnel', 'outlet_shaft'

        # Hydraulic State (Initialize as Full Static Water)
        # Assume static level at Z_IN_DESIGN
        self.H = max(z_bottom + 0.1, Z_IN_DESIGN) # Piezometric Head (m)
        self.Q = 0.0 # Flow rate (m^3/s)
        self.A = AREA
        self.B = SLOT_WIDTH # Start pressurized

        # Structural/Geo State
        self.P_int = 0.0 # Internal Pressure (Pa)
        self.P_ext = 0.0 # External Pore Pressure (Pa)
        self.P_gap = 101325.0 # Annular Gap Pressure (Pa)
        self.Z_gap = 0.0 # Water level in annular gap (m from bottom)

        # Thermal State
        self.T_water = 8.0 # C
        self.T_gap = 15.0 # C
        self.T_soil = 15.0 # C

        # Dynamic Roughness (Bio-fouling)
        self.roughness = MANNING_N_CONCRETE

    def update_geometry(self):
        # Calculate A and B based on H (Preissmann Slot Model)
        y = self.H - self.z_bottom # Water depth

        if self.section_type == 'outlet_shaft':
             d_shaft = 16.4
             self.A = math.pi * (d_shaft/2)**2
             self.B = d_shaft
        else:
            # Tunnel Geometry
            if y < DIAMETER:
                # Free surface flow in circular pipe
                # Clamp y to avoid negative sqrt
                y = max(0.01, min(DIAMETER - 0.01, y))
                arg = max(-1.0, min(1.0, 1.0 - y/RADIUS))
                theta = 2.0 * math.acos(arg)
                self.A = (RADIUS**2 / 2.0) * (theta - math.sin(theta))
                self.B = 2.0 * RADIUS * math.sin(theta / 2.0)
            else:
                # Pressurized flow (Slot)
                self.A = AREA + (y - DIAMETER) * SLOT_WIDTH
                self.B = SLOT_WIDTH

        # Clamp A to avoid division by zero
        if self.A < 1e-3: self.A = 1e-3

        # Update Pressures
        self.P_int = (self.H - self.z_bottom) * RHO_WATER * G
        if self.P_int < 0: self.P_int = 0

class Actuator:
    def __init__(self, name, max_speed):
        self.name = name
        self.position = 0.0 # 0.0 to 1.0
        self.target = 0.0
        self.max_speed = max_speed
        self.is_stuck = False
        self.stuck_pos = 0.0

    def step(self, dt):
        if self.is_stuck:
            return

        diff = self.target - self.position
        step = np.sign(diff) * min(abs(diff), self.max_speed * dt)
        self.position += step

# --- Main Physics Engine ---

class TubeModel:
    def __init__(self, id):
        self.id = id
        self.nodes = []

        # Geometry Generation
        for i in range(NODES + 1):
            x = i * DX
            if x < 300:
                # Ramp Down
                progress = x / 300.0
                z = 95.0 - progress * (95.0 - 85.08)
                sect = 'inlet_ramp'
            elif x < 3750:
                # Main Body
                progress = (x - 300) / (3750 - 300)
                z = 85.08 - progress * (85.08 - 84.60)
                sect = 'tunnel'
            else:
                # Outlet Shaft
                progress = (x - 3750) / (TUBE_LENGTH - 3750)
                z = 84.60 + progress * (95.0 - 84.60)
                sect = 'outlet_shaft'

            self.nodes.append(HydraulicNode(x, z, sect))

        # Init Geometry once
        for n in self.nodes:
            n.update_geometry()

        # Actuators
        self.gate_in = Actuator("GateIn", GATE_SPEED_NORMAL)
        self.gate_out = Actuator("GateOut", GATE_SPEED_NORMAL)
        self.valve_fill = Actuator("ValveFill", 0.1)
        self.valve_vac = Actuator("ValveVac", 1.0)
        self.pump_drain = Actuator("PumpDrain", 1.0)

        # Initialize Gates Closed
        self.gate_in.position = 0.0
        self.gate_in.target = 0.0
        self.gate_out.position = 1.0 # Outlet usually open? No, let's say closed for start up or full for static.
        # For a stable start, let's assume system is running at steady state or full static.
        # Let's start STATIC FULL.
        self.gate_in.position = 0.0
        self.gate_out.position = 0.0

        # Anomalies
        self.leak_node = -1
        self.leak_area = 0.0

    def step_physics(self, dt, env_state):
        # 1. Update Geometry & Pressures
        for n in self.nodes:
            n.update_geometry()
            # External Pressure
            depth_from_river_surface = env_state['river_level'] - n.z_bottom
            n.P_ext = max(0, depth_from_river_surface * RHO_WATER * G)

            # Annular Gap Dynamics
            if self.leak_node != -1 and abs(n.x - self.nodes[self.leak_node].x) < DX:
                n.P_gap = 0.9 * n.P_int
                n.T_gap = n.T_water
            else:
                n.P_gap = n.P_ext * 0.1

        # 2. Hydraulic Solver (Explicit / Semi-Implicit - Robust Version)
        # Using a very simplified diffusion-wave or dampened momentum eq to prevent explosion

        new_As = []
        new_Qs = []

        # Mass Conservation
        for i in range(1, NODES):
            dQdx = (self.nodes[i+1].Q - self.nodes[i-1].Q) / (2*DX)
            q_lat = 0.0

            # Leak
            if i == self.leak_node:
                 delta_P = self.nodes[i].P_int - self.nodes[i].P_gap
                 if delta_P > 0:
                     q_lat = -self.leak_area * math.sqrt(2*G*delta_P/ (RHO_WATER*G)) / DX

            A_new = self.nodes[i].A - dt * (dQdx - q_lat)
            # Clamp A
            A_new = max(0.1, min(A_new, AREA * 2.0)) # Don't allow massive expansion
            new_As.append(A_new)

        # Momentum Conservation
        for i in range(1, NODES):
            node = self.nodes[i]
            # Friction
            R = node.A / (math.pi * DIAMETER)
            Sf = (node.roughness**2 * node.Q * abs(node.Q)) / (node.A**2 * R**(4/3))

            # Gradients
            dHdx = (self.nodes[i+1].H - self.nodes[i-1].H) / (2*DX)

            # Simplified Momentum: dQ/dt = -gA(dH/dx + Sf)
            # Add some damping for stability
            damping = 0.1 * node.Q

            Q_new = node.Q - dt * (G * node.A * (dHdx + Sf)) - damping * dt

            # Clamp Q
            Q_new = max(-500.0, min(500.0, Q_new))
            new_Qs.append(Q_new)

        # Apply Updates
        for i in range(1, NODES):
            self.nodes[i].A = new_As[i-1]
            self.nodes[i].Q = new_Qs[i-1]

            # Back calc H (Robust)
            if self.nodes[i].A < AREA:
                 ratio = self.nodes[i].A / AREA
                 y = DIAMETER * ratio
                 self.nodes[i].H = self.nodes[i].z_bottom + y
            else:
                 self.nodes[i].H = self.nodes[i].z_bottom + DIAMETER + (self.nodes[i].A - AREA)/SLOT_WIDTH

        # Boundaries (Robust)
        # Inlet (Fixed Head Z_res)
        # H_0 = Z_res - loss
        # Let's just fix H for stability if Gate Open, or Static if Closed
        if self.gate_in.position > 0.01:
            self.nodes[0].H = Z_IN_DESIGN
            # Q is extrapolated
            self.nodes[0].Q = self.nodes[1].Q
        else:
            self.nodes[0].Q = 0

        # Outlet (Fixed Head Z_tail)
        if self.gate_out.position > 0.01:
            self.nodes[-1].H = Z_OUT_DESIGN
            self.nodes[-1].Q = self.nodes[-2].Q
        else:
            self.nodes[-1].Q = 0


class SimulationEngine:
    def __init__(self):
        self.tubes = [TubeModel(1), TubeModel(2)]
        self.time = 0.0

        # Environment
        self.env = {
            'z_res': Z_IN_DESIGN,
            'z_tail': Z_OUT_DESIGN,
            'river_level': YELLOW_RIVER_LEVEL_DESIGN,
            'seismic_accel': 0.0,
            'soil_liquefied': False
        }

    def step(self, dt, controls):
        self.time += dt

        for t in self.tubes:
            ctrl = controls.get(t.id, {})
            t.gate_in.target = ctrl.get('gate_in', t.gate_in.target)
            t.gate_out.target = ctrl.get('gate_out', t.gate_out.target)
            t.valve_fill.target = ctrl.get('valve_fill', t.valve_fill.target)

            t.gate_in.step(dt)
            t.gate_out.step(dt)
            t.valve_fill.step(dt)

            t.step_physics(dt, self.env)

    def get_public_state(self):
        return {
            'time': self.time,
            'env': self.env,
            'tubes': [{
                'id': t.id,
                'gate_in': t.gate_in.position,
                'nodes': [{'x':n.x, 'H':n.H, 'Q':n.Q, 'P_int':n.P_int, 'P_gap':n.P_gap} for n in t.nodes]
            } for t in self.tubes]
        }
