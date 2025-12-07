from src.config import *

class Controller:
    def __init__(self):
        pass

    def execute(self, objectives, physics_state):
        # Generate Control Signals (u) based on Objectives

        controls = {1: {}, 2: {}}

        strategy = objectives.get('strategy', 'OPTIMAL_FLOW')

        if strategy == 'OPTIMAL_FLOW':
            target_Q = objectives.get('target_Q', 265.0)
            # Simple Proportional Control for Demo
            # Target per tube
            q_per_tube = target_Q / 2.0

            for t in physics_state['tubes']:
                # Current Q
                q_curr = t['nodes'][10]['Q'] # Mid-point Q

                # Feedback
                err = q_per_tube - q_curr
                gain = 0.0001

                curr_gate = t['gate_in']
                new_gate = curr_gate + gain * err

                controls[t['id']]['gate_in'] = max(0, min(1, new_gate))
                controls[t['id']]['gate_out'] = 1.0 # Full open

        elif strategy == 'MAX_PRESSURE':
            # Earthquake Mode
            # Close Outlet, Open Inlet
            for t in physics_state['tubes']:
                controls[t['id']]['gate_in'] = 1.0
                controls[t['id']]['gate_out'] = 0.2 # Throttled

        elif strategy == 'SINGLE_TUBE':
            active = objectives.get('active_tubes', [])
            target_Q = objectives.get('target_Q', 132.5)

            for t in physics_state['tubes']:
                if t['id'] in active:
                    # Control this tube
                    q_curr = t['nodes'][10]['Q']
                    err = target_Q - q_curr
                    gain = 0.0001
                    new_gate = t['gate_in'] + gain * err
                    controls[t['id']]['gate_in'] = max(0, min(1, new_gate))
                    controls[t['id']]['gate_out'] = 1.0
                else:
                    # Close this tube
                    controls[t['id']]['gate_in'] = 0.0
                    controls[t['id']]['gate_out'] = 0.0

        return controls
