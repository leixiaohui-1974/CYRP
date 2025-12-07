"""
Web可视化界面 - Web Visualization Dashboard

提供实时监控、中间结果展示、场景测试的Web界面
Provides real-time monitoring, intermediate results display, and scenario testing web interface
"""

import json
import time
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any, Optional, List
import numpy as np


class CYRPWebServer:
    """CYRP Web服务器"""

    def __init__(self, host: str = '0.0.0.0', port: int = 8080):
        self.host = host
        self.port = port

        # 实时数据存储
        self.realtime_data: Dict[str, Any] = {
            'timestamp': 0,
            'state': {},
            'sensors': {},
            'actuators': {},
            'control': {},
            'scenario': {},
            'alarms': []
        }

        # 历史数据
        self.history_data: List[Dict] = []
        self.max_history = 1000

        # 测试结果
        self.test_results: Dict[str, Any] = {}

        # MPC配置
        self.mpc_config: Dict[str, Any] = {}

        # 服务器线程
        self.server: Optional[HTTPServer] = None
        self.server_thread: Optional[threading.Thread] = None
        self.running = False

    def update_data(self, data: Dict[str, Any]):
        """更新实时数据"""
        self.realtime_data.update(data)
        self.realtime_data['timestamp'] = time.time()

        # 保存历史
        if len(self.history_data) >= self.max_history:
            self.history_data.pop(0)
        self.history_data.append({
            't': self.realtime_data['timestamp'],
            **{k: v for k, v in data.items() if isinstance(v, (int, float, str))}
        })

    def update_mpc_config(self, config: Dict[str, Any]):
        """更新MPC配置"""
        self.mpc_config = config

    def update_test_results(self, results: Dict[str, Any]):
        """更新测试结果"""
        self.test_results = results

    def get_html_page(self) -> str:
        """生成主页HTML"""
        return '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>穿黄工程HIL监控系统</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Microsoft YaHei', sans-serif; background: #1a1a2e; color: #eee; }
        .header { background: linear-gradient(135deg, #16213e, #0f3460); padding: 20px; text-align: center; }
        .header h1 { font-size: 24px; color: #00d4ff; }
        .header .subtitle { color: #888; margin-top: 5px; }
        .container { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; padding: 15px; }
        .card { background: #16213e; border-radius: 10px; padding: 15px; }
        .card h3 { color: #00d4ff; font-size: 14px; margin-bottom: 10px; border-bottom: 1px solid #333; padding-bottom: 5px; }
        .card.wide { grid-column: span 2; }
        .card.full { grid-column: span 4; }
        .metric { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #222; }
        .metric-label { color: #888; }
        .metric-value { font-weight: bold; color: #00ff88; }
        .metric-value.warning { color: #ffaa00; }
        .metric-value.danger { color: #ff4444; }
        .scenario-badge { display: inline-block; padding: 5px 15px; border-radius: 20px; font-weight: bold; }
        .scenario-normal { background: #00aa44; }
        .scenario-transition { background: #ff8800; }
        .scenario-emergency { background: #ff0044; }
        .chart-container { height: 200px; }
        .alarm-list { max-height: 150px; overflow-y: auto; }
        .alarm-item { padding: 5px 10px; margin: 5px 0; border-radius: 5px; font-size: 12px; }
        .alarm-warning { background: #664400; }
        .alarm-critical { background: #660022; }
        .mpc-config { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
        .mpc-param { background: #0f3460; padding: 10px; border-radius: 5px; text-align: center; }
        .mpc-param-label { font-size: 11px; color: #888; }
        .mpc-param-value { font-size: 18px; color: #00d4ff; margin-top: 5px; }
        .test-result { display: flex; align-items: center; padding: 10px; margin: 5px 0; background: #0f3460; border-radius: 5px; }
        .test-passed { border-left: 4px solid #00aa44; }
        .test-failed { border-left: 4px solid #ff0044; }
        .test-warning { border-left: 4px solid #ff8800; }
        .progress-bar { height: 6px; background: #333; border-radius: 3px; margin-top: 10px; }
        .progress-fill { height: 100%; background: #00d4ff; border-radius: 3px; transition: width 0.3s; }
        .status-indicator { width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 5px; }
        .status-ok { background: #00ff88; }
        .status-warn { background: #ffaa00; }
        .status-error { background: #ff4444; }
        .tabs { display: flex; gap: 10px; margin-bottom: 15px; }
        .tab { padding: 10px 20px; background: #0f3460; border-radius: 5px; cursor: pointer; }
        .tab.active { background: #00d4ff; color: #000; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🌊 南水北调中线穿黄工程</h1>
        <div class="subtitle">全场景自主运行HIL测试监控系统 | Real-time HIL Monitoring Dashboard</div>
    </div>

    <div class="tabs" style="padding: 15px 15px 0;">
        <div class="tab active" onclick="showTab('realtime')">实时监控</div>
        <div class="tab" onclick="showTab('mpc')">MPC配置</div>
        <div class="tab" onclick="showTab('test')">测试结果</div>
        <div class="tab" onclick="showTab('history')">历史数据</div>
    </div>

    <div id="realtime" class="tab-content active">
        <div class="container">
            <!-- 场景状态 -->
            <div class="card">
                <h3>🎯 当前场景</h3>
                <div style="text-align: center; padding: 20px;">
                    <span id="scenario-badge" class="scenario-badge scenario-normal">S2-A</span>
                    <p id="scenario-desc" style="margin-top: 10px; color: #888;">双洞均衡运行</p>
                </div>
                <div class="metric">
                    <span class="metric-label">置信度</span>
                    <span id="scenario-confidence" class="metric-value">95.2%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">优先级</span>
                    <span id="scenario-priority" class="metric-value">Normal</span>
                </div>
            </div>

            <!-- 水力学状态 -->
            <div class="card">
                <h3>💧 水力学状态</h3>
                <div class="metric">
                    <span class="metric-label">总流量</span>
                    <span id="flow-rate" class="metric-value">265.0 m³/s</span>
                </div>
                <div class="metric">
                    <span class="metric-label">北洞流量</span>
                    <span id="north-flow" class="metric-value">132.5 m³/s</span>
                </div>
                <div class="metric">
                    <span class="metric-label">南洞流量</span>
                    <span id="south-flow" class="metric-value">132.5 m³/s</span>
                </div>
                <div class="metric">
                    <span class="metric-label">平均压力</span>
                    <span id="pressure" class="metric-value">0.50 MPa</span>
                </div>
                <div class="metric">
                    <span class="metric-label">平均流速</span>
                    <span id="velocity" class="metric-value">3.44 m/s</span>
                </div>
            </div>

            <!-- 控制状态 -->
            <div class="card">
                <h3>🎛️ 控制状态</h3>
                <div class="metric">
                    <span class="metric-label">控制模式</span>
                    <span id="control-mode" class="metric-value">Hybrid MPC+PID</span>
                </div>
                <div class="metric">
                    <span class="metric-label">流量设定值</span>
                    <span id="flow-setpoint" class="metric-value">265.0 m³/s</span>
                </div>
                <div class="metric">
                    <span class="metric-label">跟踪误差</span>
                    <span id="tracking-error" class="metric-value">0.5 m³/s</span>
                </div>
                <div class="metric">
                    <span class="metric-label">控制循环</span>
                    <span id="control-loop" class="metric-value">5.2 ms</span>
                </div>
            </div>

            <!-- 执行器状态 -->
            <div class="card">
                <h3>🔧 执行器状态</h3>
                <div class="metric">
                    <span class="metric-label">北洞进口阀</span>
                    <span id="north-inlet-valve" class="metric-value">100%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">南洞进口阀</span>
                    <span id="south-inlet-valve" class="metric-value">100%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">调节阀开度</span>
                    <span id="control-valve" class="metric-value">80%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">紧急切断阀</span>
                    <span id="emergency-valve" class="metric-value">开</span>
                </div>
            </div>

            <!-- 流量曲线 -->
            <div class="card wide">
                <h3>📈 流量趋势</h3>
                <div class="chart-container">
                    <canvas id="flowChart"></canvas>
                </div>
            </div>

            <!-- 压力曲线 -->
            <div class="card wide">
                <h3>📊 压力分布</h3>
                <div class="chart-container">
                    <canvas id="pressureChart"></canvas>
                </div>
            </div>

            <!-- 传感器健康 -->
            <div class="card">
                <h3>📡 传感器状态</h3>
                <div class="metric">
                    <span class="metric-label"><span class="status-indicator status-ok"></span>压力传感器</span>
                    <span class="metric-value">11/11</span>
                </div>
                <div class="metric">
                    <span class="metric-label"><span class="status-indicator status-ok"></span>流量计</span>
                    <span class="metric-value">3/3</span>
                </div>
                <div class="metric">
                    <span class="metric-label"><span class="status-indicator status-ok"></span>DAS光纤</span>
                    <span class="metric-value">正常</span>
                </div>
                <div class="metric">
                    <span class="metric-label"><span class="status-indicator status-ok"></span>DTS光纤</span>
                    <span class="metric-value">正常</span>
                </div>
                <div class="metric">
                    <span class="metric-label">可用率</span>
                    <span id="sensor-availability" class="metric-value">100%</span>
                </div>
            </div>

            <!-- 告警列表 -->
            <div class="card">
                <h3>⚠️ 告警信息</h3>
                <div id="alarm-list" class="alarm-list">
                    <div class="alarm-item" style="color: #888; text-align: center;">无告警</div>
                </div>
            </div>

            <!-- 场景识别 -->
            <div class="card wide">
                <h3>🔍 场景识别详情</h3>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;">
                    <div class="mpc-param">
                        <div class="mpc-param-label">模式类型</div>
                        <div id="pattern-type" class="mpc-param-value">稳态</div>
                    </div>
                    <div class="mpc-param">
                        <div class="mpc-param-label">异常分数</div>
                        <div id="anomaly-score" class="mpc-param-value">0.12</div>
                    </div>
                    <div class="mpc-param">
                        <div class="mpc-param-label">特征-均值</div>
                        <div id="feature-mean" class="mpc-param-value">265.0</div>
                    </div>
                    <div class="mpc-param">
                        <div class="mpc-param-label">特征-标准差</div>
                        <div id="feature-std" class="mpc-param-value">2.1</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div id="mpc" class="tab-content">
        <div class="container">
            <div class="card full">
                <h3>⚙️ MPC配置 - 场景自适应参数</h3>
                <div class="mpc-config" style="margin-top: 15px;">
                    <div class="mpc-param">
                        <div class="mpc-param-label">预测时域</div>
                        <div id="mpc-horizon" class="mpc-param-value">30</div>
                    </div>
                    <div class="mpc-param">
                        <div class="mpc-param-label">控制时域</div>
                        <div id="mpc-control-horizon" class="mpc-param-value">10</div>
                    </div>
                    <div class="mpc-param">
                        <div class="mpc-param-label">采样周期</div>
                        <div id="mpc-sampling" class="mpc-param-value">1.0s</div>
                    </div>
                    <div class="mpc-param">
                        <div class="mpc-param-label">Q_flow</div>
                        <div id="mpc-q-flow" class="mpc-param-value">100</div>
                    </div>
                    <div class="mpc-param">
                        <div class="mpc-param-label">Q_pressure</div>
                        <div id="mpc-q-pressure" class="mpc-param-value">50</div>
                    </div>
                    <div class="mpc-param">
                        <div class="mpc-param-label">Q_asymmetric</div>
                        <div id="mpc-q-asym" class="mpc-param-value">200</div>
                    </div>
                    <div class="mpc-param">
                        <div class="mpc-param-label">R_valve</div>
                        <div id="mpc-r-valve" class="mpc-param-value">1.0</div>
                    </div>
                    <div class="mpc-param">
                        <div class="mpc-param-label">R_delta</div>
                        <div id="mpc-r-delta" class="mpc-param-value">10</div>
                    </div>
                    <div class="mpc-param">
                        <div class="mpc-param-label">过渡状态</div>
                        <div id="mpc-transition" class="mpc-param-value">无</div>
                    </div>
                </div>
            </div>
            <div class="card wide">
                <h3>📋 约束配置</h3>
                <div class="metric"><span class="metric-label">流量范围</span><span id="cons-flow" class="metric-value">0 - 320 m³/s</span></div>
                <div class="metric"><span class="metric-label">压力范围</span><span id="cons-pressure" class="metric-value">-0.05 - 1.0 MPa</span></div>
                <div class="metric"><span class="metric-label">阀门速率</span><span id="cons-valve-rate" class="metric-value">≤ 1%/s</span></div>
                <div class="metric"><span class="metric-label">不对称限制</span><span id="cons-asym" class="metric-value">≤ 10%</span></div>
            </div>
            <div class="card wide">
                <h3>🎯 设定值</h3>
                <div class="metric"><span class="metric-label">流量设定值</span><span id="sp-flow" class="metric-value">265.0 m³/s</span></div>
                <div class="metric"><span class="metric-label">压力设定值</span><span id="sp-pressure" class="metric-value">0.5 MPa</span></div>
                <div class="metric"><span class="metric-label">北洞比例</span><span id="sp-north" class="metric-value">50%</span></div>
                <div class="metric"><span class="metric-label">南洞比例</span><span id="sp-south" class="metric-value">50%</span></div>
            </div>
        </div>
    </div>

    <div id="test" class="tab-content">
        <div class="container">
            <div class="card full">
                <h3>🧪 测试结果汇总</h3>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 15px 0;">
                    <div class="mpc-param">
                        <div class="mpc-param-label">总测试数</div>
                        <div id="test-total" class="mpc-param-value">8</div>
                    </div>
                    <div class="mpc-param" style="background: #003311;">
                        <div class="mpc-param-label">通过</div>
                        <div id="test-passed" class="mpc-param-value" style="color: #00ff88;">6</div>
                    </div>
                    <div class="mpc-param" style="background: #331100;">
                        <div class="mpc-param-label">失败</div>
                        <div id="test-failed" class="mpc-param-value" style="color: #ff4444;">1</div>
                    </div>
                    <div class="mpc-param" style="background: #332200;">
                        <div class="mpc-param-label">警告</div>
                        <div id="test-warning" class="mpc-param-value" style="color: #ffaa00;">1</div>
                    </div>
                </div>
                <div class="progress-bar">
                    <div id="test-progress" class="progress-fill" style="width: 75%;"></div>
                </div>
            </div>
            <div class="card full">
                <h3>📝 测试详情</h3>
                <div id="test-details">
                    <div class="test-result test-passed">
                        <div style="flex: 1;"><b>TC_001</b> - 常规运行测试</div>
                        <div style="color: #00ff88;">✓ PASSED</div>
                    </div>
                    <div class="test-result test-passed">
                        <div style="flex: 1;"><b>TC_002</b> - 流量变化响应</div>
                        <div style="color: #00ff88;">✓ PASSED</div>
                    </div>
                    <div class="test-result test-warning">
                        <div style="flex: 1;"><b>TC_003</b> - 隧道切换测试</div>
                        <div style="color: #ffaa00;">⚠ WARNING</div>
                    </div>
                    <div class="test-result test-passed">
                        <div style="flex: 1;"><b>TC_004</b> - 传感器故障容错</div>
                        <div style="color: #00ff88;">✓ PASSED</div>
                    </div>
                    <div class="test-result test-failed">
                        <div style="flex: 1;"><b>TC_005</b> - 执行器故障响应</div>
                        <div style="color: #ff4444;">✗ FAILED</div>
                    </div>
                    <div class="test-result test-passed">
                        <div style="flex: 1;"><b>TC_006</b> - 渗漏检测响应</div>
                        <div style="color: #00ff88;">✓ PASSED</div>
                    </div>
                    <div class="test-result test-passed">
                        <div style="flex: 1;"><b>TC_007</b> - 地震响应测试</div>
                        <div style="color: #00ff88;">✓ PASSED</div>
                    </div>
                    <div class="test-result test-passed">
                        <div style="flex: 1;"><b>TC_008</b> - 综合应急测试</div>
                        <div style="color: #00ff88;">✓ PASSED</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div id="history" class="tab-content">
        <div class="container">
            <div class="card full">
                <h3>📜 历史数据</h3>
                <div class="chart-container" style="height: 400px;">
                    <canvas id="historyChart"></canvas>
                </div>
            </div>
        </div>
    </div>

    <script>
        // 初始化图表
        const flowCtx = document.getElementById('flowChart').getContext('2d');
        const flowChart = new Chart(flowCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: '总流量',
                    data: [],
                    borderColor: '#00d4ff',
                    tension: 0.4,
                    fill: false
                }, {
                    label: '设定值',
                    data: [],
                    borderColor: '#ff8800',
                    borderDash: [5, 5],
                    tension: 0,
                    fill: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: { y: { beginAtZero: false } },
                plugins: { legend: { labels: { color: '#888' } } }
            }
        });

        const pressureCtx = document.getElementById('pressureChart').getContext('2d');
        const pressureChart = new Chart(pressureCtx, {
            type: 'bar',
            data: {
                labels: ['0m', '425m', '850m', '1275m', '1700m', '2125m', '2550m', '2975m', '3400m', '3825m', '4250m'],
                datasets: [{
                    label: '压力 (MPa)',
                    data: [0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.55, 0.52, 0.50, 0.48, 0.45],
                    backgroundColor: '#00d4ff88'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: { y: { beginAtZero: true, max: 1.0 } },
                plugins: { legend: { labels: { color: '#888' } } }
            }
        });

        // Tab切换
        function showTab(tabId) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            document.querySelector(`.tab-content#${tabId}`).classList.add('active');
            event.target.classList.add('active');
        }

        // 更新数据
        function updateData() {
            fetch('/api/data')
                .then(r => r.json())
                .then(data => {
                    // 更新显示
                    if (data.state) {
                        document.getElementById('flow-rate').textContent =
                            (data.state.flow_rate || 265).toFixed(1) + ' m³/s';
                        document.getElementById('pressure').textContent =
                            ((data.state.pressure || 500000) / 1e6).toFixed(2) + ' MPa';
                    }
                    if (data.scenario) {
                        document.getElementById('scenario-badge').textContent = data.scenario.id || 'S2-A';
                    }
                })
                .catch(e => console.log('Data fetch error'));
        }

        // 定时更新
        setInterval(updateData, 1000);
    </script>
</body>
</html>'''

    def create_handler(self):
        """创建请求处理器"""
        server = self

        class CYRPHandler(SimpleHTTPRequestHandler):
            def do_GET(self):
                parsed = urlparse(self.path)

                if parsed.path == '/' or parsed.path == '/index.html':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html; charset=utf-8')
                    self.end_headers()
                    self.wfile.write(server.get_html_page().encode('utf-8'))

                elif parsed.path == '/api/data':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps(server.realtime_data).encode())

                elif parsed.path == '/api/mpc':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(server.mpc_config).encode())

                elif parsed.path == '/api/tests':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(server.test_results).encode())

                elif parsed.path == '/api/history':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(server.history_data[-100:]).encode())

                else:
                    self.send_error(404)

            def log_message(self, format, *args):
                pass  # 禁用日志

        return CYRPHandler

    def start(self):
        """启动服务器"""
        handler = self.create_handler()
        self.server = HTTPServer((self.host, self.port), handler)
        self.running = True

        def serve():
            while self.running:
                self.server.handle_request()

        self.server_thread = threading.Thread(target=serve, daemon=True)
        self.server_thread.start()
        print(f"Web server started at http://{self.host}:{self.port}")

    def stop(self):
        """停止服务器"""
        self.running = False
        if self.server:
            self.server.shutdown()


# 全局Web服务器实例
_web_server: Optional[CYRPWebServer] = None


def start_web_server(port: int = 8080) -> CYRPWebServer:
    """启动Web服务器"""
    global _web_server
    if _web_server is None:
        _web_server = CYRPWebServer(port=port)
        _web_server.start()
    return _web_server


def get_web_server() -> Optional[CYRPWebServer]:
    """获取Web服务器实例"""
    return _web_server
