from src.model import SimulationEngine
from src.sensors import SensorSuite, ExternalDataFeed
from src.perception import FusionEngine
from src.autonomy import AutonomousManager
from src.control import Controller
import random

class ScenarioOrchestrator:
    def __init__(self):
        self.sim = SimulationEngine()
        self.sensors = [SensorSuite(1), SensorSuite(2)]
        self.ext_feed = ExternalDataFeed()
        self.perception = FusionEngine()
        self.brain = AutonomousManager()
        self.plc = Controller()

        self.active_drill = None
        self.last_state = None # Cache for read-only access

    def set_drill(self, drill_name):
        self.active_drill = drill_name
        print(f"Drill Started: {drill_name}")

    def step(self):
        # 1. Physics Step
        if not hasattr(self, 'last_controls'): self.last_controls = {1:{}, 2:{}}
        self.sim.step(0.5, self.last_controls)

        phys_state = self.sim.get_public_state()

        # 2. Inject Drill Anomalies
        if self.active_drill == "S6_EARTHQUAKE":
            if self.sim.time > 10.0:
                self.sim.env['seismic_accel'] = 0.4
                self.ext_feed.weather_forecast['seismic_alert'] = 'WARNING_LEVEL_1'

        elif self.active_drill == "S5_LEAK":
            if self.sim.time > 10.0:
                self.sim.tubes[0].leak_node = 25
                self.sim.tubes[0].leak_area = 0.05

        elif self.active_drill == "S4_MAINTENANCE":
            if self.sim.time > 5.0:
                self.ext_feed.maintenance_plan = [{'tube_id': 2, 'status': 'PLANNED_OUTAGE'}]

        # 3. Sensor Sim
        ext_data = self.ext_feed.fetch_feeds(self.sim.time)

        sensor_readings = []
        for i, s in enumerate(self.sensors):
            s.measure_physics(self.sim.tubes[i], self.sim.env)
            sensor_readings.append(s)

        # 4. Perception
        fused_mode, anomalies = self.perception.process(sensor_readings, ext_data)

        # 5. Autonomy
        target_mode, objectives = self.brain.decide(fused_mode, ext_data)

        # 6. Control
        self.last_controls = self.plc.execute(objectives, phys_state)

        # Cache State
        self.last_state = {
            'time': self.sim.time,
            'mode': fused_mode,
            'target': target_mode,
            'anomalies': anomalies,
            'objectives': objectives,
            'physics': phys_state
        }

        return self.last_state

    def get_current_state(self):
        # Read-only access to last computed state
        # If simulation hasn't started, compute initial state
        if self.last_state is None:
            # We don't step physics, just return initial snapshot
            phys_state = self.sim.get_public_state()
            return {
                'time': 0.0,
                'mode': "INIT",
                'target': "INIT",
                'anomalies': [],
                'objectives': {},
                'physics': phys_state
            }
        return self.last_state
