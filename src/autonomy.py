class AutonomousManager:
    def __init__(self):
        self.state = "INIT"
        self.target_mode = "S1"
        self.control_objectives = {}

    def decide(self, fused_state, external_data):
        # fused_state: "S1_NOMINAL", "S6_EARTHQUAKE", etc.
        # external_data: demand plan, etc.

        # 1. State Machine Transition
        if fused_state == "S6_EARTHQUAKE":
            self.target_mode = "S6"
            # Strategy: Pressurize
            self.control_objectives = {
                'strategy': 'MAX_PRESSURE',
                'target_P': 500000.0, # 5 bar
                'allow_gate_motion': True
            }

        elif fused_state == "S5_LEAKAGE":
            self.target_mode = "S5"
            # Strategy: Pressure Sealing (Keep P_int > P_ext but minimize leak flow?)
            # Actually Report III says: "High Pressure Self-Sealing" usually.
            self.control_objectives = {
                'strategy': 'PRESSURE_SEAL',
                'target_P': 300000.0
            }

        elif fused_state == "S4_MAINTENANCE":
            self.target_mode = "S4"
            # Check which tube is out
            outage_tube = next((m['tube_id'] for m in external_data['maintenance']), None)
            if outage_tube:
                self.control_objectives = {
                    'strategy': 'SINGLE_TUBE',
                    'active_tubes': [1, 2], # Remove outage_tube
                    'target_Q': external_data['demand'].get('target_Q', 100.0)
                }
                self.control_objectives['active_tubes'].remove(outage_tube)

        else:
            self.target_mode = "S1"
            self.control_objectives = {
                'strategy': 'OPTIMAL_FLOW',
                'target_Q': external_data['demand'].get('target_Q', 265.0)
            }

        return self.target_mode, self.control_objectives
