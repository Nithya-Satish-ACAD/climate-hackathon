# Distributed Energy Management Simulation

This project implements a distributed, multi-agent energy management system for smart grids. Each agent runs as an independent FastAPI server and communicates via REST APIs, simulating a realistic distributed energy scenario.

---

## Table of Contents
- [Architecture](#architecture)
- [Features](#features)
- [Setup & Installation](#setup--installation)
- [Running the Simulation](#running-the-simulation)
- [API Endpoints](#api-endpoints)
- [Simulation Flow & Anomaly Logic](#simulation-flow--anomaly-logic)
- [Testing](#testing)
- [Logging](#logging)
- [Troubleshooting](#troubleshooting)
- [Extending the System](#extending-the-system)
- [Demo Video](#demo-video)
- [Screenshots](#screenshots)

---

## Architecture

The system consists of three main agent types:

1. **Grid Agent** (Port 8000): Coordinates the system, collects status, detects anomalies, and issues curtailment commands.
2. **Home Agent** (Port 8001, 8002, ...): Simulates a smart home with solar, loads, and batteries. Detects local anomalies and reports them.
3. **Battery Agent** (Port 8004, 8005, ...): Manages battery storage systems.

Agents communicate via HTTP REST APIs. Each agent can be run independently, allowing for distributed deployment.

---

## Features
- **Distributed, modular architecture** (each agent is a server)
- **AI-powered decision making** (OpenAI GPT-3.5-turbo)
- **Real-time anomaly detection and curtailment**
- **Scenario-based simulation and testing**
- **Comprehensive logging**
- **Easy extensibility for new agent/module types**

---

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd climate-hackathon
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Prepare data files:**
   - Place `data/historical_data.csv` and `data/battery_historical_data.csv` in the `data/` directory.
4. **Set API keys:**
   - Set your OpenAI and OpenWeatherMap API keys in a `.env` file or as environment variables.

---

## Running the Simulation

### Option 1: Run All Agents Together (Recommended)
Use the coordinator script to start all agents and run the simulation:
```bash
python scripts/run_distributed_simulation.py
```

### Option 2: Run Agents Individually
Open separate terminals for each agent:

**Grid Agent:**
```bash
python agents/grid_agent/server.py --port 8000
```
**Home Agents:**
```bash
python agents/home_agent/server.py --id home_1 --port 8001 --transformer transformer_A
python agents/home_agent/server.py --id home_2 --port 8002 --transformer transformer_A
python agents/home_agent/server.py --id home_3 --port 8003 --transformer transformer_B
```
**Battery Agents:**
```bash
python agents/battery_agent/server.py --id battery_agent_1 --port 8004 --transformer transformer_A
python agents/battery_agent/server.py --id battery_agent_2 --port 8005 --transformer transformer_B
```

---

## API Endpoints

### Grid Agent (http://localhost:8000)
- `GET /` — Server info
- `GET /status` — Current transformer status
- `POST /coordinate` — Run grid coordination
- `GET /agents` — List agent endpoints
- `POST /agents/{agent_id}/status` — Receive agent status
- `GET /health` — Health check

### Home Agent (http://localhost:8001, 8002, ...)
- `GET /` — Server info
- `GET /status` — Current module status
- `POST /simulate` — Run simulation cycle (accepts scenario for testing)
- `POST /curtail` — Apply curtailment
- `GET /health` — Health check

### Battery Agent (http://localhost:8004, 8005, ...)
- `GET /` — Server info
- `GET /status` — Current battery status
- `POST /simulate` — Run simulation cycle
- `POST /curtail` — Apply curtailment
- `GET /health` — Health check

---

## Simulation Flow & Anomaly Logic

1. **Initialization:** All agents start and register with the grid agent.
2. **Status Collection:** Grid agent collects status from all agents.
3. **Anomaly Detection:**
   - **Home agents** compute and report an `anomaly` field (`True` if generation < demand or generation > demand).
   - **Grid agent** acts only on the anomaly field reported by home agents.
4. **Curtailment Planning:** If an anomaly is detected, the grid agent uses AI to generate a curtailment plan (prioritizing loads, then solar, then batteries).
5. **Action Execution:** Curtailment actions are sent to respective agents.
6. **Logging:** All actions and status pings are logged for analysis.

---

## Testing

To run the scenario-based test script (make sure all agents are running):
```bash
python tests/test_grid_flows.py
```
This will run several scenarios and print pass/fail results for each.

---

## Logging
- Each agent writes logs to the `logs/` directory (e.g., `home_1_log.csv`, `grid_agent_log.csv`).
- Logs include timestamps, status pings, and curtailment actions.

---

## Troubleshooting
- **500 Internal Server Error:** Check agent logs for stack traces (often due to missing required fields or misconfiguration).
- **Port Already in Use:** Make sure the required ports are free.
- **API Key Issues:** Ensure your API keys are set correctly.
- **Missing Data Files:** Ensure all required CSV files are present in the `data/` directory.

---

## Extending the System

### Adding New Agents
1. Create a new agent server (see `agents/` for examples).
2. Implement required endpoints (`/status`, `/simulate`, `/curtail`, `/health`).
3. Register the agent with the grid agent (update config or CLI args).

### Adding New Module Types
1. Create new module classes (e.g., `WindModule`).
2. Update Pydantic models for API serialization.
3. Implement required methods (e.g., `status()`).

---

## Demo Video

*Coming soon!*

<!--
Insert a link to your demo video here, e.g.:
[![Watch the demo](demo_screenshot.png)](https://your-demo-video-link)
-->

---

## Screenshots

*Coming soon!*

<!--
Insert screenshots of the running system, logs, or UI here.
Example:
![Grid Agent Output](screenshots/grid_agent_output.png)
![Home Agent Log](screenshots/home_agent_log.png)
--> 