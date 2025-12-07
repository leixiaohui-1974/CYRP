import numpy as np
import json
import random

class SensorSuite:
    def __init__(self, tube_id):
        self.tube_id = tube_id
        # Physical Sensor Buffers
        self.das_spectrum = [] # PSD array
        self.dts_temp = [] # Temperature array
        self.mems_accel = [] # G-force array
        self.scada_flow = 0.0
        self.scada_pressure = 0.0

    def measure_physics(self, tube_model, env_state):
        # 1. DAS (Acoustic)
        nodes = tube_model.nodes
        self.das_spectrum = np.zeros(len(nodes))

        for i, n in enumerate(nodes):
            # Flow Noise ~ v^3
            v = 0.0
            if n.A > 0.001:
                v = n.Q / n.A

            # Clamp v to avoid overflow in v^3
            v = max(-50.0, min(50.0, v))

            noise = 10 * np.log10(abs(v**3) + 1e-6) + 20

            # Leak Noise
            if tube_model.leak_node != -1 and abs(i - tube_model.leak_node) < 2:
                noise += 50.0 # dB

            self.das_spectrum[i] = max(0, noise + np.random.normal(0, 1))

        # 2. DTS (Temperature)
        self.dts_temp = np.zeros(len(nodes))
        for i, n in enumerate(nodes):
            self.dts_temp[i] = n.T_gap + np.random.normal(0, 0.2)

        # 3. MEMS (Vibration/Tilt)
        self.mems_accel = np.zeros(len(nodes))
        for i, n in enumerate(nodes):
            v = 0.0
            if n.A > 0.001: v = n.Q / n.A
            v = max(-50.0, min(50.0, v))

            base_vib = 0.001 * (v)**2
            seismic = env_state['seismic_accel']
            self.mems_accel[i] = base_vib + seismic + np.random.normal(0, 0.001)

class ExternalDataFeed:
    def __init__(self):
        self.maintenance_plan = []
        self.weather_forecast = {}
        self.demand_plan = []

    def fetch_feeds(self, current_time):
        self.maintenance_plan = []
        if 100 < current_time < 500:
             self.maintenance_plan.append({'tube_id': 2, 'status': 'PLANNED_OUTAGE'})

        self.weather_forecast = {'flood_risk': 'LOW', 'seismic_alert': 'NONE'}
        if current_time > 800:
            self.weather_forecast['seismic_alert'] = 'WARNING_LEVEL_1'

        self.demand_plan = {'target_Q': 200.0}

        return {
            'maintenance': self.maintenance_plan,
            'weather': self.weather_forecast,
            'demand': self.demand_plan
        }
