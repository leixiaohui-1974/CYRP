from flask import Flask, render_template, jsonify, request
from src.scenarios import ScenarioOrchestrator
import threading
import time

app = Flask(__name__)

orch = ScenarioOrchestrator()
running = False
sim_thread = None

def loop():
    global running
    while running:
        orch.step()
        time.sleep(0.5) # Match simulation DT roughly for real-time feel

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def status():
    # Read-only access prevents double-stepping
    state = orch.get_current_state()
    return jsonify(state)

@app.route('/api/control', methods=['POST'])
def control():
    cmd = request.json.get('command')
    global running, sim_thread, orch

    if cmd == 'start':
        if not running:
            running = True
            sim_thread = threading.Thread(target=loop)
            sim_thread.start()
    elif cmd == 'stop':
        running = False
    elif cmd == 'reset':
        running = False
        time.sleep(0.6) # Wait for loop to exit
        orch = ScenarioOrchestrator()

    return jsonify({'status': 'ok'})

@app.route('/api/drill', methods=['POST'])
def drill():
    drill_name = request.json.get('name')
    orch.set_drill(drill_name)
    return jsonify({'status': 'ok', 'drill': drill_name})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
