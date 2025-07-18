import argparse
import threading
import time
import ntplib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import openai
import requests
import csv
import json
from datetime import datetime, timedelta
import uvicorn
from langgraph.graph import StateGraph
import os
from dotenv import load_dotenv
load_dotenv()

# --- NTP Helper ---
ntp_offset = timedelta(0)

def sync_ntp():
    global ntp_offset
    try:
        c = ntplib.NTPClient()
        response = c.request('pool.ntp.org', version=3)
        ntp_time = datetime.utcfromtimestamp(response.tx_time)
        system_time = datetime.utcnow()
        ntp_offset = ntp_time - system_time
        print(f"[NTP] Synced. Offset: {ntp_offset.total_seconds():.3f} seconds")
    except Exception as e:
        print(f"[NTP] Sync failed: {e}")

def ntp_sync_thread():
    while True:
        sync_ntp()
        time.sleep(300)  # Re-sync every 5 minutes

# On startup
sync_ntp()
threading.Thread(target=ntp_sync_thread, daemon=True).start()

def get_synced_time():
    return datetime.utcnow() + ntp_offset

def seconds_until_next_minute():
    now = get_synced_time()
    next_minute = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
    return (next_minute - now).total_seconds()

# --- CLI Args ---
def parse_args():
    parser = argparse.ArgumentParser(description="Grid Agent Server")
    parser.add_argument('--home_agents', type=str, nargs='*', default=['home_1:http://localhost:8001'], help='List of home agent_id:endpoint')
    parser.add_argument('--battery_agents', type=str, nargs='*', default=['battery_agent_1:http://localhost:8002'], help='List of battery agent_id:endpoint')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the server on')
    return parser.parse_args()

args = parse_args()
PORT = args.port

# Build AGENT_ENDPOINTS from CLI args
AGENT_ENDPOINTS = {}
for entry in args.home_agents + args.battery_agents:
    if ':' in entry:
        agent_id, endpoint = entry.split(':', 1)
        AGENT_ENDPOINTS[agent_id] = endpoint

# Configuration
# Replace hardcoded secrets
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Agent endpoints configuration
# AGENT_ENDPOINTS = {
#     "home_1": "http://localhost:8001",
#     "battery_agent_1": "http://localhost:8002"
# }

# Pydantic models for API
class ModuleStatus(BaseModel):
    type: str
    id: str
    soc: Optional[float] = None
    available: Optional[bool] = None
    available_to_charge: Optional[float] = None
    predicted_generation: Optional[float] = None
    predicted_demand: Optional[float] = None

class AgentStatus(BaseModel):
    home_agent_id: Optional[str] = None
    battery_agent_id: Optional[str] = None
    transformer_id: str
    modules: List[ModuleStatus]
    anomaly: Optional[bool] = None

class CurtailmentAction(BaseModel):
    type: str  # "load" or "solar"
    id: str
    curtailment_kw: float

class CurtailmentPlan(BaseModel):
    actions: List[CurtailmentAction]

# Helper functions
def openai_decision(prompt, api_key):
    client = openai.OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"OpenAI error: {e}")
        return ""

# Transformer module
class Transformer:
    def __init__(self, id, neighbor_ids=None):
        self.id = id
        self.agents = {}  # agent_id -> last status ping
        self.neighbor_ids = neighbor_ids or []
    
    def receive_status_ping(self, agent_id, status):
        self.agents[agent_id] = status
    
    def detect_anomaly(self):
        # Use the anomaly field from each agent's status, if present
        any_anomaly = False
        total_generation = 0
        total_demand = 0
        for agent_status in self.agents.values():
            # Use anomaly field if present
            if 'anomaly' in agent_status and agent_status['anomaly']:
                any_anomaly = True
            # Still sum for logging/visibility
            for module in agent_status.get("modules", []):
                if module["type"] == "solar":
                    total_generation += module.get("predicted_generation", 0) or 0
                if module["type"] == "load":
                    total_demand += module.get("predicted_demand", 0) or 0
        return any_anomaly, total_generation, total_demand
    
    def plan_curtailment(self, curtail_loads_only=False):
        prompt = f"""
Given the following agent statuses: {self.agents}.
There is a generation-demand anomaly (deficit).
Curtailment/dispatch priority:
1. First, use solar generation to meet demand or curtailment needs.
2. Only use (discharge) the battery if solar is insufficient.
3. Battery should primarily store energy for backup or night-time use, not be used if solar is available.
4. If there is excess solar, charge the battery (if not full).

Batteries should only be discharged if solar generation is insufficient to meet demand. Prioritize using solar first, then battery if needed. If there is excess solar, charge the battery.

Respond ONLY with a JSON array of curtailment actions. Each action should have:
- 'type': 'battery', 'solar', or 'load'
- 'id': the full unique id (e.g., home_1_battery_1, home_2_solar_2, home_3_load_1)
- 'action': for battery: 'charge' or 'discharge'; for solar/load: 'turn_on', 'turn_off', or 'curtail'
- 'amount': a percentage (0.0 to 1.0) for curtail/charge/discharge, or null for on/off
- 'duration_minutes': integer, how long to apply the action
Example:
[
  {{"type": "battery", "id": "home_1_battery_1", "action": "discharge", "amount": 0.2, "duration_minutes": 10}},
  {{"type": "solar", "id": "home_2_solar_2", "action": "turn_off", "amount": null, "duration_minutes": 5}},
  {{"type": "load", "id": "home_3_load_1", "action": "curtail", "amount": 0.5, "duration_minutes": 15}}
]
Curtail loads first, then solar, then batteries if needed. Do not include any explanation or extra text.
"""
        plan_text = openai_decision(prompt, OPENAI_API_KEY)
        try:
            plan = json.loads(plan_text)
            # Validate and normalize actions
            valid_actions = []
            for action in plan:
                if all(k in action for k in ("type", "id", "action", "amount", "duration_minutes")):
                    valid_actions.append(action)
                else:
                    print(f"[WARNING] Skipping invalid curtailment action: {action}")
            return valid_actions
        except Exception as e:
            print(f"Failed to parse curtailment plan as JSON: {e}\nAI response: {plan_text}")
            return []

# GridAgent class
class GridAgent:
    def __init__(self, transformers=None):
        self.transformers = transformers or {}
    
    def log_status(self, log_data):
        log_dir = os.path.join(os.path.dirname(__file__), '../../logs')
        os.makedirs(log_dir, exist_ok=True)
        log_filename = os.path.join(log_dir, "grid_agent_log.csv")
        with open(log_filename, mode="a", newline="") as log_file:
            writer = csv.writer(log_file)
            writer.writerow([
                datetime.now().isoformat(),
                json.dumps(log_data)
            ])
    
    def collect_agent_statuses(self):
        """Collect status from all agents via HTTP"""
        agent_statuses = {}
        scenario = os.environ.get("TEST_SCENARIO")
        for agent_id, endpoint in AGENT_ENDPOINTS.items():
            try:
                url = f"{endpoint}/simulate"
                if scenario:
                    url += f"?scenario={scenario}"
                response = requests.post(url, timeout=10)
                if response.status_code == 200:
                    status_data = response.json()
                    agent_statuses[agent_id] = status_data
                    print(f"Collected status from {agent_id}: {status_data}")
                else:
                    print(f"Failed to get status from {agent_id}: {response.status_code}")
            except Exception as e:
                print(f"Error collecting status from {agent_id}: {e}")
        return agent_statuses
    
    def send_curtailment_plan(self, plan, agent_lookup=None):
        # Print the plan in natural language for the user
        if plan:
            print("Curtailment Plan (user-friendly):")
            for action in plan:
                t = action.get("type")
                i = action.get("id")
                act = action.get("action")
                amt = action.get("amount")
                dur = action.get("duration_minutes")
                print(f"  {t} {i}: {act} {amt} for {dur} min")
        else:
            print("No curtailment actions required.")
        # Send curtailment plan to respective agents
        for action in plan:
            agent_id = None
            for aid in AGENT_ENDPOINTS:
                if action.get("id", "").startswith(aid):
                    agent_id = aid
                    break
            if agent_id and agent_lookup and agent_id in agent_lookup:
                agent_lookup[agent_id].apply_curtailment(action)
            elif agent_id:
                # Send via HTTP
                endpoint = AGENT_ENDPOINTS[agent_id]
                try:
                    response = requests.post(f"{endpoint}/curtail", json=action, timeout=10)
                    print(f"Sent curtailment to {endpoint}: {action}")
                except Exception as e:
                    print(f"Error sending curtailment to {endpoint}: {e}")
            else:
                print(f"No agent endpoint found for module {action.get('id')}")
    
    def run_coordination(self):
        """Run the grid coordination logic"""
        # No longer collect statuses from all agents here; use in-memory statuses from POSTs
        # agent_statuses = self.collect_agent_statuses()  # REMOVED
        # Update transformers with latest pings (REMOVED)
        # for agent_id, status in agent_statuses.items():
        #     transformer_id = status.get("transformer_id")
        #     if transformer_id and transformer_id in self.transformers:
        #         self.transformers[transformer_id].receive_status_ping(agent_id, status)
        # The rest of the logic is now handled in timed_coordination
        
        log_data = {}
        curtailment_results = []
        
        # Process each transformer
        for transformer_id, transformer in self.transformers.items():
            anomaly, gen, demand = transformer.detect_anomaly()
            print(f"Transformer {transformer_id}: Generation={gen}, Demand={demand}, Anomaly={anomaly}")
            
            log_data[transformer_id] = {
                "generation": gen,
                "demand": demand,
                "anomaly": anomaly
            }
            
            if anomaly:
                print(f"Anomaly detected in Transformer {transformer_id}. Planning curtailment...")
                
                # Check neighbors for surplus
                found_surplus = False
                for neighbor_id in transformer.neighbor_ids:
                    neighbor = self.transformers.get(neighbor_id)
                    if neighbor:
                        n_anomaly, n_gen, n_demand = neighbor.detect_anomaly()
                        if n_gen > n_demand:
                            transfer_amount = min(n_gen - n_demand, demand - gen)
                            print(f"Transferring {transfer_amount} kW from {neighbor_id} to {transformer_id}")
                            log_data[transformer_id]["transfer_from_neighbor"] = {
                                "from": neighbor_id,
                                "amount": transfer_amount
                            }
                            found_surplus = True
                            break
                
                if not found_surplus:
                    # No surplus found, create curtailment plan
                    plan = transformer.plan_curtailment(curtail_loads_only=True)
                    print(f"Curtailment Plan for {transformer_id}: {plan}")
                    log_data[transformer_id]["curtailment_plan"] = plan
                    
                    # Send curtailment plan to agents
                    if plan:
                        self.send_curtailment_plan(plan)
                        log_data[transformer_id]["curtailment_results"] = plan
        
        self.log_status(log_data)
        return {
            "log_data": log_data,
            "curtailment_results": curtailment_results
        }

# Initialize FastAPI app
app = FastAPI(title="Grid Agent Server", version="1.0.0")

# Initialize transformers
transformer_A = Transformer("transformer_A", neighbor_ids=["transformer_B"])
transformer_B = Transformer("transformer_B", neighbor_ids=["transformer_A"])
transformers = {
    "transformer_A": transformer_A,
    "transformer_B": transformer_B
}

grid_agent = GridAgent(transformers=transformers)

# --- In-memory status storage ---
# Each transformer keeps a dict of agent_id -> last status ping (already in Transformer.agents)

# --- Timed Coordination ---
def timed_coordination():
    while True:
        wait = seconds_until_next_minute() + 5  # 5 seconds after top of minute
        print(f"[GridAgent] Waiting {wait:.2f}s for next coordination...")
        time.sleep(wait)
        print(f"[GridAgent] Running coordination at {get_synced_time().isoformat()} (5s after minute)")
        try:
            # Use the latest statuses received via POSTs
            log_data = {}
            curtailment_results = []
            for transformer_id, transformer in transformers.items():
                # Print/log the statuses used
                print(f"[GridAgent] Statuses for {transformer_id}:")
                for agent_id, status in transformer.agents.items():
                    print(f"  {agent_id}: {status}")
                anomaly, gen, demand = transformer.detect_anomaly()
                print(f"Transformer {transformer_id}: Generation={gen}, Demand={demand}, Anomaly={anomaly}")
                log_data[transformer_id] = {
                    "generation": gen,
                    "demand": demand,
                    "anomaly": anomaly,
                    "agents": list(transformer.agents.keys())
                }
                if anomaly:
                    print(f"Anomaly detected in Transformer {transformer_id}. Planning curtailment...")
                    plan = transformer.plan_curtailment(curtail_loads_only=True)
                    print(f"Curtailment Plan for {transformer_id}: {plan}")
                    log_data[transformer_id]["curtailment_plan"] = plan
                    if plan:
                        grid_agent.send_curtailment_plan(plan)
                        log_data[transformer_id]["curtailment_results"] = plan
            grid_agent.log_status(log_data)
        except Exception as e:
            print(f"[GridAgent] Error in timed coordination: {e}")

# Start background thread for timed coordination
threading.Thread(target=timed_coordination, daemon=True).start()

@app.get("/")
async def root():
    return {"message": "Grid Agent Server is running"}

@app.get("/status")
async def get_status():
    """Get current status of all transformers"""
    status_data = {}
    for transformer_id, transformer in transformers.items():
        anomaly, gen, demand = transformer.detect_anomaly()
        status_data[transformer_id] = {
            "generation": gen,
            "demand": demand,
            "anomaly": anomaly,
            "agents": list(transformer.agents.keys())
        }
    return status_data

@app.post("/coordinate")
async def run_coordination():
    """Run grid coordination and return results"""
    try:
        results = grid_agent.run_coordination()
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents")
async def get_agent_endpoints():
    """Get list of configured agent endpoints"""
    return AGENT_ENDPOINTS

@app.post("/agents/{agent_id}/status")
async def receive_agent_status(agent_id: str, status: AgentStatus):
    """Receive status ping from an agent"""
    try:
        transformer_id = status.transformer_id
        if transformer_id in transformers:
            transformers[transformer_id].receive_status_ping(agent_id, status.model_dump())
            return {"status": "success", "message": f"Status received from {agent_id}"}
        else:
            raise HTTPException(status_code=400, detail=f"Unknown transformer {transformer_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    print(f"Starting Grid Agent Server on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT) 