#!/usr/bin/env python3
"""
CodeForge Live Dashboard Server.

Runs experiments and serves a live dashboard at http://localhost:8501

Usage:
    # Start dashboard with existing results
    python scripts/serve_dashboard.py

    # Run experiments AND serve dashboard simultaneously
    python scripts/serve_dashboard.py --run --strategy mcts --budget 16

    # Full ablation with live dashboard
    python scripts/serve_dashboard.py --run --ablation all --budget 16
"""

import argparse
import json
import os
import sys
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DASHBOARD_PORT = 8501


class DashboardHandler(SimpleHTTPRequestHandler):
    """Serves the dashboard HTML and results JSON API."""
    
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(get_dashboard_html().encode())
        
        elif self.path == "/api/results":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(load_all_results()).encode())
        
        elif self.path == "/api/live":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(load_live_status()).encode())
        
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass  # Suppress HTTP logs


def load_all_results() -> dict:
    """Load all experiment results from the results directory."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    results = {
        "experiments": [],
        "scaling": [],
        "ablations": [],
        "strategies": [],
        "live_log": [],
        "timestamp": time.time(),
    }
    
    for f in sorted(Path(RESULTS_DIR).glob("*.json")):
        try:
            data = json.loads(f.read_text())
            
            config = data.get("config", {})
            strategy = config.get("strategy", "unknown")
            name = config.get("name", f.stem)
            
            problem_results = data.get("problem_results", [])
            solved = sum(1 for p in problem_results if p.get("solved", False))
            total = len(problem_results)
            pass1 = solved / total if total > 0 else 0
            avg_score = sum(p.get("best_score", 0) for p in problem_results) / total if total > 0 else 0
            avg_gens = sum(p.get("total_generations", 0) for p in problem_results) / total if total > 0 else 0
            
            exp = {
                "name": name,
                "strategy": strategy,
                "pass1": round(pass1, 4),
                "avg_score": round(avg_score, 4),
                "avg_gens": round(avg_gens, 1),
                "solved": solved,
                "total": total,
                "budget": config.get("max_generations", 0),
                "problems": [],
            }
            
            for p in problem_results:
                exp["problems"].append({
                    "id": p.get("problem_id", ""),
                    "solved": p.get("solved", False),
                    "score": round(p.get("best_score", 0), 4),
                    "gens": p.get("total_generations", 0),
                    "time": round(p.get("wall_time_seconds", 0), 2),
                })
            
            results["experiments"].append(exp)
            
            # Categorize
            if "scaling" in name:
                results["scaling"].append(exp)
            elif strategy in ("best_of_n", "beam_search", "best_first", "mcts"):
                results["strategies"].append(exp)
            
        except Exception as e:
            pass
    
    # Load live log if exists
    log_path = os.path.join(RESULTS_DIR, "live.log")
    if os.path.exists(log_path):
        try:
            lines = open(log_path).readlines()[-50:]  # Last 50 lines
            results["live_log"] = [l.strip() for l in lines]
        except:
            pass
    
    return results


def load_live_status() -> dict:
    """Load current running experiment status."""
    status_path = os.path.join(RESULTS_DIR, "live_status.json")
    if os.path.exists(status_path):
        try:
            return json.loads(open(status_path).read())
        except:
            pass
    return {"running": False}


def get_dashboard_html() -> str:
    """Generate the complete dashboard HTML. Single file, no dependencies."""
    
    return '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CodeForge — Live Dashboard</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --bg: #0a0e17;
    --card: #111827;
    --border: #1e293b;
    --accent: #06b6d4;
    --green: #10b981;
    --red: #ef4444;
    --yellow: #f59e0b;
    --purple: #8b5cf6;
    --orange: #f97316;
    --text: #e2e8f0;
    --dim: #64748b;
    --grid: #1e293b;
}

* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text); font-family: 'JetBrains Mono', monospace; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

.header {
    border-bottom: 1px solid var(--border);
    padding: 16px 24px;
    display: flex; align-items: center; justify-content: space-between;
}
.logo { font-size: 18px; font-weight: 700; }
.logo span { color: var(--accent); }
.logo-sub { color: var(--dim); font-size: 10px; letter-spacing: 1px; margin-top: 2px; }
.status-dot {
    width: 8px; height: 8px; border-radius: 50%; display: inline-block;
    margin-right: 6px; animation: pulse 2s infinite;
}
.status-dot.live { background: var(--green); }
.status-dot.idle { background: var(--dim); }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }

.tabs {
    display: flex; border-bottom: 1px solid var(--border);
    padding: 0 24px; overflow-x: auto;
}
.tab {
    background: none; border: none; color: var(--dim);
    padding: 12px 16px; cursor: pointer; font-size: 11px; font-weight: 600;
    font-family: 'JetBrains Mono', monospace; letter-spacing: 0.5px;
    border-bottom: 2px solid transparent; transition: all 0.2s; white-space: nowrap;
}
.tab.active { color: var(--accent); border-bottom-color: var(--accent); }
.tab:hover { color: var(--text); }

.content { padding: 20px 24px; max-width: 1200px; margin: 0 auto; }

.metrics { display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }
.metric-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px 20px; flex: 1; min-width: 160;
}
.metric-label { color: var(--dim); font-size: 10px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
.metric-value { font-size: 28px; font-weight: 700; line-height: 1; }
.metric-sub { color: var(--dim); font-size: 10px; margin-top: 4px; }
.metric-trend { font-size: 10px; margin-top: 2px; }
.metric-trend.up { color: var(--green); }
.metric-trend.down { color: var(--red); }

.card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 8px; padding: 20px; margin-bottom: 20px;
}
.card-title {
    color: var(--dim); font-size: 10px; letter-spacing: 1px;
    text-transform: uppercase; margin-bottom: 14px;
}

.chart-container { position: relative; width: 100%; height: 350px; }
canvas { width: 100% !important; height: 100% !important; }

.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 11px; }
th { padding: 8px 12px; text-align: left; color: var(--dim); font-weight: 600; border-bottom: 1px solid var(--border); }
td { padding: 8px 12px; border-bottom: 1px solid var(--border); }
tr.highlight { background: rgba(16, 185, 129, 0.05); }
tr.highlight td:first-child { color: var(--green); font-weight: 700; }

.bar-row { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }
.bar-label { width: 110px; font-size: 10px; color: var(--dim); }
.bar-track { flex: 1; height: 18px; background: var(--bg); border-radius: 4px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 4px; transition: width 0.8s ease; }
.bar-value { width: 45px; font-size: 11px; text-align: right; }

.log-box {
    background: var(--bg); border: 1px solid var(--border); border-radius: 6px;
    padding: 12px; max-height: 400px; overflow-y: auto; font-size: 10px; line-height: 1.8;
}
.log-line { color: var(--dim); }
.log-line .pass { color: var(--green); }
.log-line .fail { color: var(--red); }
.log-line .info { color: var(--accent); }
.log-line .warn { color: var(--yellow); }

.problem-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(36px, 1fr)); gap: 4px; }
.problem-cell {
    width: 36px; height: 36px; border-radius: 4px; display: flex;
    align-items: center; justify-content: center; font-size: 8px; font-weight: 600;
    cursor: pointer; transition: transform 0.1s;
}
.problem-cell:hover { transform: scale(1.2); }
.problem-cell.solved { background: rgba(16, 185, 129, 0.2); color: var(--green); border: 1px solid rgba(16, 185, 129, 0.3); }
.problem-cell.failed { background: rgba(239, 68, 68, 0.15); color: var(--red); border: 1px solid rgba(239, 68, 68, 0.2); }
.problem-cell.partial { background: rgba(245, 158, 11, 0.15); color: var(--yellow); border: 1px solid rgba(245, 158, 11, 0.2); }

.refresh-badge {
    display: inline-flex; align-items: center; gap: 4px;
    color: var(--dim); font-size: 9px;
}

.two-col { display: flex; gap: 16px; flex-wrap: wrap; }
.two-col > * { flex: 1; min-width: 300px; }

.empty-state {
    text-align: center; padding: 60px 20px; color: var(--dim);
}
.empty-state h3 { font-size: 14px; margin-bottom: 8px; color: var(--text); }
.empty-state p { font-size: 11px; line-height: 1.8; }
.empty-state code { background: var(--bg); padding: 2px 6px; border-radius: 3px; color: var(--accent); }

.tooltip {
    position: absolute; background: var(--card); border: 1px solid var(--border);
    border-radius: 6px; padding: 8px 12px; font-size: 10px; pointer-events: none;
    z-index: 100; box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
</style>
</head>
<body>

<div class="header">
    <div>
        <div class="logo"><span>CodeForge</span> / Dashboard</div>
        <div class="logo-sub">TEST-TIME SCALING FOR CODE GENERATION</div>
    </div>
    <div class="refresh-badge">
        <span class="status-dot idle" id="statusDot"></span>
        <span id="statusText">Idle</span>
        <span style="margin-left:8px" id="refreshTimer">Auto-refresh: 3s</span>
    </div>
</div>

<div class="tabs">
    <button class="tab active" onclick="switchTab('overview')">Overview</button>
    <button class="tab" onclick="switchTab('scaling')">Compute Scaling</button>
    <button class="tab" onclick="switchTab('problems')">Problem Map</button>
    <button class="tab" onclick="switchTab('ablation')">Ablations</button>
    <button class="tab" onclick="switchTab('live')">Live Log</button>
</div>

<div class="content" id="content"></div>

<script>
// ═══════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════
let DATA = { experiments: [], scaling: [], strategies: [], live_log: [], timestamp: 0 };
let CURRENT_TAB = 'overview';
let REFRESH_INTERVAL = 3000;

// ═══════════════════════════════════════════
// DATA FETCHING
// ═══════════════════════════════════════════
async function fetchData() {
    try {
        const res = await fetch('/api/results');
        DATA = await res.json();
        
        // Update status indicator
        const live = await fetch('/api/live').then(r => r.json()).catch(() => ({running: false}));
        const dot = document.getElementById('statusDot');
        const text = document.getElementById('statusText');
        if (live.running) {
            dot.className = 'status-dot live';
            text.textContent = 'Running: ' + (live.current_problem || '...');
        } else {
            dot.className = 'status-dot idle';
            text.textContent = DATA.experiments.length > 0 ? 
                `${DATA.experiments.length} experiments loaded` : 'No results yet';
        }
        
        render();
    } catch(e) {
        console.log('Fetch failed, retrying...', e);
    }
}

setInterval(fetchData, REFRESH_INTERVAL);
fetchData();

// ═══════════════════════════════════════════
// TAB SWITCHING
// ═══════════════════════════════════════════
function switchTab(tab) {
    CURRENT_TAB = tab;
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    event.target.classList.add('active');
    render();
}

// ═══════════════════════════════════════════
// RENDER
// ═══════════════════════════════════════════
function render() {
    const c = document.getElementById('content');
    
    if (DATA.experiments.length === 0) {
        c.innerHTML = renderEmpty();
        return;
    }
    
    switch(CURRENT_TAB) {
        case 'overview': c.innerHTML = renderOverview(); break;
        case 'scaling': c.innerHTML = renderScaling(); break;
        case 'problems': c.innerHTML = renderProblems(); break;
        case 'ablation': c.innerHTML = renderAblation(); break;
        case 'live': c.innerHTML = renderLive(); break;
    }
    
    // Draw charts after DOM update
    requestAnimationFrame(() => {
        if (CURRENT_TAB === 'scaling') drawScalingChart();
        if (CURRENT_TAB === 'overview') drawOverviewChart();
    });
}

function renderEmpty() {
    return `<div class="empty-state">
        <h3>No experiment results yet</h3>
        <p>Run an experiment to see results here:</p>
        <p style="margin-top:12px">
            <code>python scripts/run_experiment.py --strategy mcts --budget 16</code>
        </p>
        <p style="margin-top:8px">
            <code>python scripts/run_experiment.py --ablation all --budget 16</code>
        </p>
        <p style="margin-top:16px">Results will appear here automatically (polling every 3s)</p>
    </div>`;
}

// ═══════════════════════════════════════════
// OVERVIEW TAB
// ═══════════════════════════════════════════
function renderOverview() {
    const best = DATA.experiments.reduce((a, b) => a.pass1 > b.pass1 ? a : b, DATA.experiments[0]);
    const baseline = DATA.experiments.find(e => e.strategy === 'best_of_n') || DATA.experiments[0];
    const improvement = baseline ? ((best.pass1 - baseline.pass1) / Math.max(baseline.pass1, 0.01) * 100).toFixed(0) : '?';
    const totalProblems = DATA.experiments.reduce((s, e) => s + e.total, 0);
    const totalSolved = DATA.experiments.reduce((s, e) => s + e.solved, 0);
    
    return `
    <div class="metrics">
        <div class="metric-card">
            <div class="metric-label">Best pass@1</div>
            <div class="metric-value" style="color:var(--green)">${best.pass1.toFixed(3)}</div>
            <div class="metric-sub">${best.name}</div>
            ${baseline && best !== baseline ? `<div class="metric-trend up">▲ ${improvement}% vs baseline</div>` : ''}
        </div>
        <div class="metric-card">
            <div class="metric-label">Experiments</div>
            <div class="metric-value" style="color:var(--accent)">${DATA.experiments.length}</div>
            <div class="metric-sub">${totalSolved}/${totalProblems} problems solved</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Best Strategy</div>
            <div class="metric-value" style="color:var(--purple);font-size:18px">${best.strategy.replace('_',' ')}</div>
            <div class="metric-sub">budget=${best.budget}, avg_gens=${best.avg_gens}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Avg Score</div>
            <div class="metric-value" style="color:var(--yellow)">${best.avg_score.toFixed(3)}</div>
            <div class="metric-sub">composite verifier score</div>
        </div>
    </div>
    
    <div class="two-col">
        <div class="card">
            <div class="card-title">Strategy Comparison — pass@1</div>
            <canvas id="overviewChart" height="280"></canvas>
        </div>
        <div class="card">
            <div class="card-title">All Experiments</div>
            <div class="table-wrap">
                <table>
                    <tr><th>Name</th><th>Strategy</th><th>pass@1</th><th>Avg Score</th><th>Solved</th><th>Budget</th></tr>
                    ${DATA.experiments.map((e, i) => `
                        <tr class="${e === best ? 'highlight' : ''}">
                            <td>${e.name}</td>
                            <td>${e.strategy}</td>
                            <td>${e.pass1.toFixed(3)}</td>
                            <td>${e.avg_score.toFixed(3)}</td>
                            <td>${e.solved}/${e.total}</td>
                            <td>${e.budget}</td>
                        </tr>
                    `).join('')}
                </table>
            </div>
        </div>
    </div>`;
}

function drawOverviewChart() {
    const canvas = document.getElementById('overviewChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width = canvas.parentElement.clientWidth - 40;
    const H = canvas.height = 260;
    ctx.clearRect(0, 0, W, H);
    
    const exps = DATA.experiments.slice(0, 8);
    if (exps.length === 0) return;
    
    const barW = Math.min(60, (W - 80) / exps.length - 10);
    const colors = ['#64748b', '#f59e0b', '#06b6d4', '#10b981', '#8b5cf6', '#f97316', '#ef4444', '#06b6d4'];
    
    const maxVal = 1.0;
    const chartH = H - 60;
    const chartY = 20;
    
    // Grid lines
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
        const y = chartY + chartH - (i / 4) * chartH;
        ctx.beginPath(); ctx.moveTo(50, y); ctx.lineTo(W, y); ctx.stroke();
        ctx.fillStyle = '#64748b'; ctx.font = '9px JetBrains Mono';
        ctx.fillText((i * 0.25).toFixed(2), 10, y + 3);
    }
    
    exps.forEach((e, i) => {
        const x = 60 + i * ((W - 80) / exps.length);
        const h = (e.pass1 / maxVal) * chartH;
        
        ctx.fillStyle = colors[i % colors.length];
        ctx.beginPath();
        ctx.roundRect(x, chartY + chartH - h, barW, h, [4, 4, 0, 0]);
        ctx.fill();
        
        // Value label
        ctx.fillStyle = '#e2e8f0'; ctx.font = 'bold 10px JetBrains Mono';
        ctx.textAlign = 'center';
        ctx.fillText(e.pass1.toFixed(3), x + barW/2, chartY + chartH - h - 6);
        
        // Name label
        ctx.fillStyle = '#64748b'; ctx.font = '8px JetBrains Mono';
        ctx.save();
        ctx.translate(x + barW/2, H - 2);
        ctx.rotate(-0.4);
        ctx.fillText(e.strategy.replace('_',' '), 0, 0);
        ctx.restore();
    });
}

// ═══════════════════════════════════════════
// COMPUTE SCALING TAB
// ═══════════════════════════════════════════
function renderScaling() {
    // Group experiments by strategy, sort by budget
    const byStrategy = {};
    DATA.experiments.forEach(e => {
        if (!byStrategy[e.strategy]) byStrategy[e.strategy] = [];
        byStrategy[e.strategy].push(e);
    });
    Object.values(byStrategy).forEach(arr => arr.sort((a, b) => a.budget - b.budget));
    
    return `
    <div class="card">
        <div class="card-title">Compute-vs-Success Curve — pass@1 by generation budget</div>
        <canvas id="scalingChart" height="380"></canvas>
    </div>
    <div class="card">
        <div class="card-title">Efficiency Analysis — pass@1 per generation</div>
        ${DATA.experiments.map(e => `
            <div class="bar-row">
                <div class="bar-label">${e.name.substring(0, 18)}</div>
                <div class="bar-track">
                    <div class="bar-fill" style="width:${e.pass1*100}%;background:${
                        e.strategy==='mcts'?'var(--green)':e.strategy==='best_first'?'var(--accent)':
                        e.strategy==='beam_search'?'var(--yellow)':'var(--dim)'}"></div>
                </div>
                <div class="bar-value">${e.pass1.toFixed(3)}</div>
            </div>
        `).join('')}
    </div>`;
}

function drawScalingChart() {
    const canvas = document.getElementById('scalingChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width = canvas.parentElement.clientWidth - 40;
    const H = canvas.height = 350;
    ctx.clearRect(0, 0, W, H);
    
    const byStrategy = {};
    DATA.experiments.forEach(e => {
        if (!byStrategy[e.strategy]) byStrategy[e.strategy] = [];
        byStrategy[e.strategy].push(e);
    });
    Object.values(byStrategy).forEach(arr => arr.sort((a, b) => a.budget - b.budget));
    
    const stratColors = { best_of_n:'#64748b', beam_search:'#f59e0b', best_first:'#06b6d4', mcts:'#10b981' };
    
    const chartX = 50, chartY = 20, chartW = W - 80, chartH = H - 70;
    
    // Grid
    ctx.strokeStyle = '#1e293b'; ctx.lineWidth = 0.5;
    for (let i = 0; i <= 5; i++) {
        const y = chartY + chartH - (i/5) * chartH;
        ctx.beginPath(); ctx.moveTo(chartX, y); ctx.lineTo(chartX+chartW, y); ctx.stroke();
        ctx.fillStyle = '#64748b'; ctx.font = '9px JetBrains Mono';
        ctx.fillText((i*0.2).toFixed(1), 10, y+3);
    }
    
    // Find budget range
    const allBudgets = DATA.experiments.map(e => e.budget);
    const minB = Math.min(...allBudgets), maxB = Math.max(...allBudgets);
    
    // Lines
    Object.entries(byStrategy).forEach(([strat, exps]) => {
        if (exps.length < 1) return;
        const color = stratColors[strat] || '#64748b';
        
        ctx.strokeStyle = color; ctx.lineWidth = strat === 'mcts' ? 3 : 2;
        ctx.beginPath();
        
        exps.forEach((e, i) => {
            const x = chartX + ((e.budget - minB) / Math.max(maxB - minB, 1)) * chartW;
            const y = chartY + chartH - (e.pass1) * chartH;
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.stroke();
        
        // Dots + labels
        exps.forEach(e => {
            const x = chartX + ((e.budget - minB) / Math.max(maxB - minB, 1)) * chartW;
            const y = chartY + chartH - (e.pass1) * chartH;
            ctx.fillStyle = color;
            ctx.beginPath(); ctx.arc(x, y, strat==='mcts'?5:4, 0, Math.PI*2); ctx.fill();
            ctx.fillStyle = '#e2e8f0'; ctx.font = '9px JetBrains Mono'; ctx.textAlign = 'center';
            ctx.fillText(e.pass1.toFixed(3), x, y - 10);
        });
        
        // Legend
    });
    
    // Legend
    let lx = chartX + 10;
    Object.entries(stratColors).forEach(([strat, color]) => {
        if (!byStrategy[strat]) return;
        ctx.fillStyle = color;
        ctx.fillRect(lx, H - 20, 12, 3); lx += 16;
        ctx.fillStyle = '#e2e8f0'; ctx.font = '9px JetBrains Mono'; ctx.textAlign = 'left';
        ctx.fillText(strat.replace('_',' '), lx, H - 16); lx += 80;
    });
    
    // Axis labels
    ctx.fillStyle = '#64748b'; ctx.font = '10px JetBrains Mono'; ctx.textAlign = 'center';
    ctx.fillText('Compute Budget (# generations)', chartX + chartW/2, H - 2);
}

// ═══════════════════════════════════════════
// PROBLEM MAP TAB
// ═══════════════════════════════════════════
function renderProblems() {
    const best = DATA.experiments.reduce((a, b) => a.pass1 > b.pass1 ? a : b, DATA.experiments[0]);
    if (!best || !best.problems) return '<div class="empty-state"><h3>No problem-level data</h3></div>';
    
    return `
    <div class="card">
        <div class="card-title">Problem Map — ${best.name} (${best.solved}/${best.total} solved)</div>
        <div style="margin-bottom:12px;font-size:10px;color:var(--dim)">
            Each cell = one problem. Green = solved, Red = failed, Yellow = partial (score > 0.3).
        </div>
        <div class="problem-grid">
            ${best.problems.map(p => {
                const cls = p.solved ? 'solved' : p.score > 0.3 ? 'partial' : 'failed';
                return `<div class="problem-cell ${cls}" title="${p.id}\\nScore: ${p.score}\\nGens: ${p.gens}\\nTime: ${p.time}s">
                    ${p.score > 0 ? (p.score * 100).toFixed(0) : '✗'}
                </div>`;
            }).join('')}
        </div>
    </div>
    <div class="card">
        <div class="card-title">Problem-Level Results</div>
        <div class="table-wrap">
            <table>
                <tr><th>Problem</th><th>Solved</th><th>Score</th><th>Generations</th><th>Time</th></tr>
                ${best.problems.map(p => `
                    <tr>
                        <td>${p.id}</td>
                        <td style="color:${p.solved?'var(--green)':'var(--red)'}">${p.solved?'✓':'✗'}</td>
                        <td>${p.score.toFixed(3)}</td>
                        <td>${p.gens}</td>
                        <td>${p.time}s</td>
                    </tr>
                `).join('')}
            </table>
        </div>
    </div>`;
}

// ═══════════════════════════════════════════
// ABLATION TAB
// ═══════════════════════════════════════════
function renderAblation() {
    const sorted = [...DATA.experiments].sort((a, b) => b.pass1 - a.pass1);
    const best = sorted[0];
    
    return `
    <div class="card">
        <div class="card-title">Strategy Ablation — Fixed Budget Comparison</div>
        ${sorted.map((e, i) => {
            const color = i===0?'var(--green)':i<3?'var(--accent)':'var(--dim)';
            return `<div class="bar-row">
                <div class="bar-label" style="${i===0?'color:var(--green);font-weight:700':''}">${e.name.substring(0,20)}</div>
                <div class="bar-track">
                    <div class="bar-fill" style="width:${e.pass1*100}%;background:${color}"></div>
                </div>
                <div class="bar-value" style="color:${color}">${e.pass1.toFixed(3)}</div>
            </div>`;
        }).join('')}
    </div>
    <div class="card">
        <div class="card-title">Full Ablation Table</div>
        <div class="table-wrap">
            <table>
                <tr><th>Configuration</th><th>Strategy</th><th>pass@1</th><th>Avg Score</th><th>Solved/Total</th><th>Avg Gens</th><th>Δ vs Best</th></tr>
                ${sorted.map((e, i) => `
                    <tr class="${i===0?'highlight':''}">
                        <td>${e.name}</td>
                        <td>${e.strategy}</td>
                        <td>${e.pass1.toFixed(3)}</td>
                        <td>${e.avg_score.toFixed(3)}</td>
                        <td>${e.solved}/${e.total}</td>
                        <td>${e.avg_gens}</td>
                        <td style="color:${i===0?'var(--green)':'var(--red)'}">${i===0?'—':((e.pass1-best.pass1)*100).toFixed(1)+'%'}</td>
                    </tr>
                `).join('')}
            </table>
        </div>
    </div>`;
}

// ═══════════════════════════════════════════
// LIVE LOG TAB
// ═══════════════════════════════════════════
function renderLive() {
    const logs = DATA.live_log || [];
    return `
    <div class="card">
        <div class="card-title">Live Experiment Log (last 50 lines, auto-refreshing)</div>
        <div class="log-box" id="logBox">
            ${logs.length === 0 ? '<div class="log-line" style="color:var(--dim)">No active experiment. Run:</div><div class="log-line"><span class="info">python scripts/serve_dashboard.py --run --strategy mcts --budget 16</span></div>' :
            logs.map(l => {
                let cls = '';
                if (l.includes('SOLVED') || l.includes('PASS')) cls = 'pass';
                else if (l.includes('FAILED') || l.includes('ERROR')) cls = 'fail';
                else if (l.includes('Expanding') || l.includes('SELECT')) cls = 'info';
                else if (l.includes('PRUNE')) cls = 'warn';
                return `<div class="log-line"><span class="${cls}">${escapeHtml(l)}</span></div>`;
            }).join('')}
        </div>
    </div>`;
}

function escapeHtml(s) {
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
</script>
</body>
</html>'''


def run_experiment_thread(args):
    """Run experiment in background thread, writing results to results/ for dashboard to pick up."""
    import importlib
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Write live status
    status_path = os.path.join(RESULTS_DIR, "live_status.json")
    log_path = os.path.join(RESULTS_DIR, "live.log")
    
    with open(status_path, 'w') as f:
        json.dump({"running": True, "strategy": args.strategy, "budget": args.budget}, f)
    
    # Redirect loguru to file for live log
    from loguru import logger
    logger.add(log_path, format="{time:HH:mm:ss} | {message}", level="INFO", rotation="1 MB")
    
    # Import and run
    sys.argv = ['run_experiment.py',
        '--strategy', args.strategy,
        '--budget', str(args.budget),
        '--benchmark', args.benchmark,
        '--output-dir', RESULTS_DIR,
    ]
    
    if args.max_problems:
        sys.argv.extend(['--max-problems', str(args.max_problems)])
    if args.ablation:
        sys.argv.extend(['--ablation', args.ablation])
    
    try:
        from scripts.run_experiment import main
        main()
    except Exception as e:
        with open(log_path, 'a') as f:
            f.write(f"ERROR: {e}\n")
    finally:
        with open(status_path, 'w') as f:
            json.dump({"running": False}, f)


def main():
    parser = argparse.ArgumentParser(description="CodeForge Live Dashboard")
    parser.add_argument("--port", type=int, default=DASHBOARD_PORT)
    parser.add_argument("--run", action="store_true", help="Run experiment alongside dashboard")
    parser.add_argument("--strategy", default="mcts")
    parser.add_argument("--budget", type=int, default=16)
    parser.add_argument("--benchmark", default="sample")
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--ablation", default=None)
    
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Start experiment in background if requested
    if args.run:
        print(f"\n  Starting experiment: strategy={args.strategy}, budget={args.budget}")
        t = threading.Thread(target=run_experiment_thread, args=(args,), daemon=True)
        t.start()
    
    # Start dashboard server
    server = HTTPServer(("0.0.0.0", args.port), DashboardHandler)
    print(f"""
╔══════════════════════════════════════════════════╗
║          CodeForge Live Dashboard                ║
║                                                  ║
║   → http://localhost:{args.port}                     ║
║                                                  ║
║   Auto-refreshes every 3 seconds                 ║
║   Results dir: {RESULTS_DIR:<30} ║
╚══════════════════════════════════════════════════╝
""")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down dashboard...")
        server.shutdown()


if __name__ == "__main__":
    main()
