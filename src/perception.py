import numpy as np

class FusionEngine:
    def __init__(self):
        self.current_mode = "S1_NOMINAL"
        self.health_score = 1.0

    def process(self, sensor_data, external_data):
        # sensor_data: list of SensorSuite objects
        # external_data: dict from ExternalDataFeed

        # 1. Physical Anomaly Detection
        anomalies = []
        for s in sensor_data:
            # Leak Detection (DAS + DTS)
            max_noise = np.max(s.das_spectrum)
            min_temp = np.min(s.dts_temp)

            if max_noise > 60.0 and min_temp < 12.0:
                anomalies.append(f"TUBE_{s.tube_id}_LEAK_SUSPECTED")

            # Vibration
            if np.max(s.mems_accel) > 0.1:
                anomalies.append(f"TUBE_{s.tube_id}_HIGH_VIB")

        # 2. Contextual Filtering (The "Plan" Aspect)
        # If High Vib is detected BUT Seismic Alert is Active -> It's Earthquake, not Mechanical Fault

        seismic_alert = external_data['weather'].get('seismic_alert') == 'WARNING_LEVEL_1'
        maintenance_active = any(m['status'] == 'PLANNED_OUTAGE' for m in external_data['maintenance'])

        final_state = "S1_NOMINAL"

        # Logic Tree
        if seismic_alert or "TUBE_1_HIGH_VIB" in anomalies and "TUBE_2_HIGH_VIB" in anomalies:
            final_state = "S6_EARTHQUAKE"

        elif any("LEAK" in a for a in anomalies):
            final_state = "S5_LEAKAGE"

        elif maintenance_active:
            final_state = "S4_MAINTENANCE"

        else:
            final_state = "S1_NOMINAL"

        return final_state, anomalies
