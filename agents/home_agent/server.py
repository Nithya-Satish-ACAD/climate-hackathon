import argparse
import threading
import time
import ntplib
from fastapi import FastAPI, HTTPException, Request, Query, Body
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import requests
import openai
import random
import csv
import json
from datetime import datetime, timedelta
import math
import uvicorn
from dotenv import load_dotenv
import os
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
    parser = argparse.ArgumentParser(description="Home Agent Server")
    parser.add_argument('--id', type=str, default='home_1', help='Home agent ID')
    parser.add_argument('--port', type=int, default=8001, help='Port to run the server on')
    parser.add_argument('--transformer', type=str, default='transformer_A', help='Transformer ID')
    return parser.parse_args()

args = parse_args()
HOME_AGENT_ID = args.id
PORT = args.port
TRANSFORMER_ID = args.transformer

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
LOCATION = "Bangalore"
HISTORICAL_CSV_PATH = os.path.join(os.path.dirname(__file__), '../../data/historical_data.csv')

# Pydantic models for API
class ModuleStatus(BaseModel):
    type: str
    id: str
    soc: Optional[float] = None
    available: Optional[bool] = None
    available_to_charge: Optional[float] = None
    predicted_generation: Optional[float] = None
    predicted_demand: Optional[float] = None

class StatusPing(BaseModel):
    home_agent_id: str
    transformer_id: str
    modules: List[ModuleStatus]

class CurtailmentRequest(BaseModel):
    curtailment_kw: float
    target_type: Optional[str] = None
    target_id: Optional[str] = None

class BatteryModule:
    def __init__(self, id, threshold):
        self.id = id
        self.threshold = threshold
        self.soc: float | None = None
        self.available: bool | None = None
        self.available_to_charge: float | None = None
    
    def status(self):
        return ModuleStatus(
            type="battery",
            id=self.id,
            soc=self.soc,
            available=self.available,
            available_to_charge=self.available_to_charge
        )

class SolarModule:
    def __init__(self, id):
        self.id = id
        self.predicted_generation: float | None = None
    
    def status(self):
        return ModuleStatus(
            type="solar", 
            id=self.id, 
            predicted_generation=self.predicted_generation
        )

class LoadModule:
    def __init__(self, id):
        self.id = id
        self.predicted_demand: float | None = None
    
    def status(self):
        return ModuleStatus(
            type="load", 
            id=self.id, 
            predicted_demand=self.predicted_demand
        )

# Helper functions
def load_historical_data(csv_path):
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return pd.DataFrame()

def fetch_weather(api_key, location):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        return response.json()
    except Exception as e:
        print(f"Error fetching weather: {e}")
        return {}

def predict_with_openai(prompt, api_key):
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

# HomeAgent class
class HomeAgent:
    def __init__(self, csv_path, weather_api_key, openai_api_key, location, home_agent_id, transformer_id):
        self.csv_path = csv_path
        self.weather_api_key = weather_api_key
        self.openai_api_key = openai_api_key
        self.location = location
        self.home_agent_id = home_agent_id
        self.transformer_id = transformer_id
        self.historical_data = load_historical_data(csv_path)
        
        # Initialize modules with unique IDs
        self.batteries = [
            BatteryModule(id="battery_1", threshold=0.2),
            BatteryModule(id="battery_2", threshold=0.25)
        ]
        self.solars = [
            SolarModule(id=f"{home_agent_id}_solar_1"),
            SolarModule(id=f"{home_agent_id}_solar_2")
        ]
        self.loads = [
            LoadModule(id=f"{home_agent_id}_load_1"),
            LoadModule(id=f"{home_agent_id}_load_2")
        ]

    def log_status(self, status_ping):
        log_dir = os.path.join(os.path.dirname(__file__), '../../logs')
        os.makedirs(log_dir, exist_ok=True)
        log_filename = os.path.join(log_dir, f"{self.home_agent_id}_log.csv")
        with open(log_filename, mode="a", newline="") as log_file:
            writer = csv.writer(log_file)
            writer.writerow([
                datetime.now().isoformat(),
                json.dumps(status_ping.model_dump())
            ])

    def run_simulation(self):
        """Run the simulation and return status ping"""
        data = self.historical_data
        weather = fetch_weather(self.weather_api_key, self.location)
        now = get_synced_time()
        hour = now.hour
        # Predict solar generation (higher during 7am-6pm, lower otherwise)
        for solar in self.solars:
            if 7 <= hour <= 18:
                base = 2.0 + 2.0 * math.sin(math.pi * (hour-7)/11)  # peak at noon
                variation = random.uniform(-0.3, 0.3)
                solar.predicted_generation = max(0.0, base + variation)
            else:
                solar.predicted_generation = max(0.0, random.uniform(0, 0.2))
        # Predict load demand (higher in morning/evening, lower at night)
        for load in self.loads:
            if 6 <= hour <= 9 or 18 <= hour <= 22:
                base = 2.5
            elif 10 <= hour <= 17:
                base = 1.5
            else:
                base = 1.0
            variation = random.uniform(-0.3, 0.3)
            load.predicted_demand = max(0.5, base + variation)
        # Generate battery SOC (random, but trend: higher at night, lower at end of day)
        for battery in self.batteries:
            if 0 <= hour <= 6:
                soc = random.uniform(0.7, 1.0)
            elif 7 <= hour <= 17:
                soc = random.uniform(0.3, 0.8)
            else:
                soc = random.uniform(0.5, 0.9)
            battery.soc = round(soc, 2)
            battery.available_to_charge = max(0.0, round(0.9 - battery.soc, 2))
            # Use OpenAI to decide availability (as before)
            total_predicted_demand = sum(l.predicted_demand or 0 for l in self.loads)
            total_predicted_generation = sum(s.predicted_generation or 0 for s in self.solars)
            prompt = f"""
            Battery SOC: {battery.soc}, Threshold: {battery.threshold}, Max SOC: 0.9, Available to charge: {battery.available_to_charge},
            Predicted total demand: {total_predicted_demand}, Predicted total generation: {total_predicted_generation}, Weather: {weather}.
            Should the battery (ID: {battery.id}) be available for charge/discharge? Respond with 'True' or 'False' and a brief reason.
            """
            result = predict_with_openai(prompt, self.openai_api_key)
            battery.available = "true" in result.lower()

        # Create status ping
        modules_status = [b.status() for b in self.batteries] + \
                         [s.status() for s in self.solars] + \
                         [l.status() for l in self.loads]
        
        status_ping = StatusPing(
            home_agent_id=self.home_agent_id,
            transformer_id=self.transformer_id,
            modules=modules_status
        )
        
        print("Status Ping:", status_ping.model_dump())
        self.log_status(status_ping)
        return status_ping

    def apply_curtailment(self, curtailment_kw, target_type=None, target_id=None):
        if target_type == "load":
            for load in self.loads:
                if load.id == target_id:
                    print(f"Curtailing {curtailment_kw} kW from load {target_id} in HomeAgent {self.home_agent_id}")
                    return {"status": "success", "message": f"Curtailed {curtailment_kw} kW from load {target_id}"}
            return {"status": "error", "message": f"Load {target_id} not found"}
        elif target_type == "solar":
            for solar in self.solars:
                if solar.id == target_id:
                    print(f"Curtailing {curtailment_kw} kW from solar {target_id} in HomeAgent {self.home_agent_id}")
                    return {"status": "success", "message": f"Curtailed {curtailment_kw} kW from solar {target_id}"}
            return {"status": "error", "message": f"Solar {target_id} not found"}
        else:
            print(f"HomeAgent {self.home_agent_id} curtailing {curtailment_kw} kW (no specific target)")
            return {"status": "success", "message": f"Curtailed {curtailment_kw} kW"}

# Initialize FastAPI app
app = FastAPI(title="Home Agent Server", version="1.0.0")
home_agent = HomeAgent(
    csv_path=HISTORICAL_CSV_PATH,
    weather_api_key=OPENWEATHERMAP_API_KEY,
    openai_api_key=OPENAI_API_KEY,
    location=LOCATION,
    home_agent_id=HOME_AGENT_ID,
    transformer_id=TRANSFORMER_ID
)

# --- Timed Status Reporting ---
def timed_status_report():
    while True:
        wait = seconds_until_next_minute()
        print(f"[{HOME_AGENT_ID}] Waiting {wait:.2f}s for next minute...")
        time.sleep(wait)
        # At top of minute, send status
        print(f"[{HOME_AGENT_ID}] Sending status report at {get_synced_time().isoformat()} (top of minute)")
        try:
            status_ping = home_agent.run_simulation()
            import requests
            try:
                resp = requests.post(
                    f"http://localhost:8000/agents/{HOME_AGENT_ID}/status",
                    json=status_ping.model_dump(),
                    timeout=5
                )
                print(f"[{HOME_AGENT_ID}] Sent status to grid agent, response: {resp.status_code}")
            except Exception as e:
                print(f"[{HOME_AGENT_ID}] Failed to send status to grid agent: {e}")
        except Exception as e:
            print(f"[{HOME_AGENT_ID}] Error in timed status report: {e}")

# Start background thread for timed reporting
threading.Thread(target=timed_status_report, daemon=True).start()

@app.get("/")
async def root():
    return {"message": "Home Agent Server is running", "agent_id": HOME_AGENT_ID}

@app.get("/status")
async def get_status():
    """Get current status of all modules"""
    modules_status = [b.status() for b in home_agent.batteries] + \
                     [s.status() for s in home_agent.solars] + \
                     [l.status() for l in home_agent.loads]
    
    return StatusPing(
        home_agent_id=home_agent.home_agent_id,
        transformer_id=home_agent.transformer_id,
        modules=modules_status
    )

@app.post("/simulate")
async def run_simulation(request: Request, scenario: int = Query(None)):
    """Run simulation and return status ping. If scenario is set, use test data."""
    if scenario:
        # TEST SCENARIOS
        if scenario == 1:
            # All local resources sufficient
            for solar in home_agent.solars:
                solar.predicted_generation = 3.0
            for load in home_agent.loads:
                load.predicted_demand = 1.0
            for battery in home_agent.batteries:
                battery.soc = 0.8
                battery.available_to_charge = 0.1
                battery.available = True
        elif scenario == 2:
            # Not enough local resources
            for solar in home_agent.solars:
                solar.predicted_generation = 0.1
            for load in home_agent.loads:
                load.predicted_demand = 3.0
            for battery in home_agent.batteries:
                battery.soc = 0.2
                battery.available_to_charge = 0.7
                battery.available = True
        elif scenario == 3:
            # Generation > demand
            for solar in home_agent.solars:
                solar.predicted_generation = 4.0
            for load in home_agent.loads:
                load.predicted_demand = 1.0
            for battery in home_agent.batteries:
                battery.soc = 0.5
                battery.available_to_charge = 0.4
                battery.available = True
        elif scenario == 4:
            # Generation < demand
            for solar in home_agent.solars:
                solar.predicted_generation = 0.2
            for load in home_agent.loads:
                load.predicted_demand = 3.0
            for battery in home_agent.batteries:
                battery.soc = 0.7
                battery.available_to_charge = 0.2
                battery.available = True
        # Skip normal simulation logic, just return status
        modules_status = [b.status() for b in home_agent.batteries] + \
                         [s.status() for s in home_agent.solars] + \
                         [l.status() for l in home_agent.loads]
        status_ping = StatusPing(
            home_agent_id=home_agent.home_agent_id,
            transformer_id=home_agent.transformer_id,
            modules=modules_status
        )
        return status_ping
    # Normal simulation
    try:
        status_ping = home_agent.run_simulation()
        return status_ping
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/curtail")
async def apply_curtailment(action: dict = Body(...)):
    """
    Apply curtailment to specific modules using the new action structure.
    """
    target_type = action.get("type")
    target_id = action.get("id")
    curtailment_action = action.get("action")
    amount = action.get("amount")
    duration = action.get("duration_minutes")
    # Implement logic for battery, solar, load based on action
    print(f"Applying curtailment: {action}")
    return {"status": "success", "message": f"Applied {curtailment_action} to {target_type} {target_id} for {duration} min"}

@app.get("/simulate")
async def simulate_get(request: Request):
    user_agent = request.headers.get("user-agent", "unknown")
    referrer = request.headers.get("referer", "unknown")
    print(f"[WARNING] GET /simulate called. User-Agent: {user_agent}, Referrer: {referrer}")
    return {"detail": "This endpoint only supports POST. Please use POST for /simulate."}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "agent_id": HOME_AGENT_ID}

if __name__ == "__main__":
    print(f"Starting Home Agent Server {HOME_AGENT_ID} on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT) 