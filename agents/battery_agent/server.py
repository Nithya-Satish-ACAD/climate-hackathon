from fastapi import FastAPI, HTTPException, Request, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import requests
import openai
import random
import csv
import json
from datetime import datetime, timedelta
import uvicorn
import argparse
import math
import ntplib
from dotenv import load_dotenv
import os
import threading
import time
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
LOCATION = "Bangalore"
HISTORICAL_CSV_PATH = os.path.join(os.path.dirname(__file__), '../../data/battery_historical_data.csv')

# --- CLI Args ---
def parse_args():
    parser = argparse.ArgumentParser(description="Battery Agent Server")
    parser.add_argument('--id', type=str, default='battery_agent_1', help='Battery agent ID')
    parser.add_argument('--port', type=int, default=8004, help='Port to run the server on')
    parser.add_argument('--transformer', type=str, default='transformer_B', help='Transformer ID')
    return parser.parse_args()

args = parse_args()
BATTERY_AGENT_ID = args.id
PORT = args.port
TRANSFORMER_ID = args.transformer

# Pydantic models for API
class ModuleStatus(BaseModel):
    type: str
    id: str
    soc: Optional[float] = None
    available: Optional[bool] = None
    available_to_charge: Optional[float] = None

class StatusPing(BaseModel):
    battery_agent_id: str
    transformer_id: str
    modules: List[ModuleStatus]

class CurtailmentRequest(BaseModel):
    curtailment_kw: float

class BatteryModule:
    def __init__(self, id):
        self.id = id
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

def timed_status_report():
    while True:
        wait = seconds_until_next_minute()
        print(f"[{BATTERY_AGENT_ID}] Waiting {wait:.2f}s for next minute...")
        time.sleep(wait)
        print(f"[{BATTERY_AGENT_ID}] Sending status report at {get_synced_time().isoformat()} (top of minute)")
        try:
            status_ping = battery_agent.run_simulation()
            import requests
            try:
                resp = requests.post(
                    f"http://localhost:8000/agents/{BATTERY_AGENT_ID}/status",
                    json=status_ping.model_dump(),
                    timeout=5
                )
                print(f"[{BATTERY_AGENT_ID}] Sent status to grid agent, response: {resp.status_code}")
            except Exception as e:
                print(f"[{BATTERY_AGENT_ID}] Failed to send status to grid agent: {e}")
        except Exception as e:
            print(f"[{BATTERY_AGENT_ID}] Error in timed status report: {e}")

# BatteryAgent class
class BatteryAgent:
    def __init__(self, csv_path, weather_api_key, openai_api_key, location, battery_agent_id, transformer_id):
        self.csv_path = csv_path
        self.weather_api_key = weather_api_key
        self.openai_api_key = openai_api_key
        self.location = location
        self.battery_agent_id = battery_agent_id
        self.transformer_id = transformer_id
        self.historical_data = load_historical_data(csv_path)
        
        # Initialize battery modules
        self.batteries = [
            BatteryModule(id="battery_park_1"),
            BatteryModule(id="battery_park_2")
        ]

    def log_status(self, status_ping):
        log_dir = os.path.join(os.path.dirname(__file__), '../../logs')
        os.makedirs(log_dir, exist_ok=True)
        log_filename = os.path.join(log_dir, f"{self.battery_agent_id}_log.csv")
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
        # Generate battery SOC (random, but trend: higher at night, lower at end of day)
        for battery in self.batteries:
            if 0 <= hour <= 6:
                soc = random.uniform(0.7, 1.0)
            elif 7 <= hour <= 17:
                soc = random.uniform(0.3, 0.8)
            else:
                soc = random.uniform(0.5, 0.9)
            battery.soc = round(soc, 2)
            battery.available_to_charge = max(0.0, round(0.9 - battery.soc + random.uniform(-0.05, 0.05), 2))
            # Use OpenAI to decide availability (as before)
            prompt = f"""
            Battery SOC: {battery.soc}, Threshold: 0.9, Max SOC: 0.9, Available to charge: {battery.available_to_charge}, Weather: {weather}.
            Should the battery (ID: {battery.id}) be available for charge/discharge? Respond with 'True' or 'False' and a brief reason.
            """
            result = predict_with_openai(prompt, self.openai_api_key)
            print(f"OpenAI response for battery {battery.id}: {result}")
            try:
                battery.available = "true" in result.lower()
            except Exception as e:
                print(f"Failed to parse OpenAI response for battery {battery.id}: {e}")
                battery.available = False

        # Create status ping
        modules_status = [b.status() for b in self.batteries]
        status_ping = StatusPing(
            battery_agent_id=self.battery_agent_id,
            transformer_id=self.transformer_id,
            modules=modules_status
        )
        
        print("Status Ping:", status_ping.model_dump())
        self.log_status(status_ping)
        return status_ping

    def apply_curtailment(self, curtailment_kw):
        print(f"BatteryAgent {self.battery_agent_id} curtailing {curtailment_kw} kW")
        return {"status": "success", "message": f"Curtailed {curtailment_kw} kW from battery system"}

# Initialize FastAPI app
app = FastAPI(title="Battery Agent Server", version="1.0.0")
battery_agent = BatteryAgent(
    csv_path=HISTORICAL_CSV_PATH,
    weather_api_key=OPENWEATHERMAP_API_KEY,
    openai_api_key=OPENAI_API_KEY,
    location=LOCATION,
    battery_agent_id=BATTERY_AGENT_ID,
    transformer_id=TRANSFORMER_ID
)

threading.Thread(target=timed_status_report, daemon=True).start()

@app.get("/")
async def root():
    return {"message": "Battery Agent Server is running", "agent_id": BATTERY_AGENT_ID}

@app.get("/status")
async def get_status():
    """Get current status of all battery modules"""
    modules_status = [b.status() for b in battery_agent.batteries]
    
    return StatusPing(
        battery_agent_id=battery_agent.battery_agent_id,
        transformer_id=battery_agent.transformer_id,
        modules=modules_status
    )

@app.post("/simulate")
async def run_simulation(request: Request, scenario: int = Query(None)):
    """Run simulation and return status ping. If scenario is set, use test data."""
    if scenario:
        # TEST SCENARIOS
        if scenario == 1:
            # All local resources sufficient
            for battery in battery_agent.batteries:
                battery.soc = 0.8
                battery.available_to_charge = 0.1
                battery.available = True
        elif scenario == 2:
            # Not enough local resources
            for battery in battery_agent.batteries:
                battery.soc = 0.2
                battery.available_to_charge = 0.7
                battery.available = True
        elif scenario == 3:
            # Generation > demand (battery available to charge)
            for battery in battery_agent.batteries:
                battery.soc = 0.5
                battery.available_to_charge = 0.4
                battery.available = True
        elif scenario == 4:
            # Generation < demand (battery available to discharge)
            for battery in battery_agent.batteries:
                battery.soc = 0.7
                battery.available_to_charge = 0.2
                battery.available = True
        # Skip normal simulation logic, just return status
        modules_status = [b.status() for b in battery_agent.batteries]
        status_ping = StatusPing(
            battery_agent_id=battery_agent.battery_agent_id,
            transformer_id=battery_agent.transformer_id,
            modules=modules_status
        )
        return status_ping
    # Normal simulation
    try:
        status_ping = battery_agent.run_simulation()
        return status_ping
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/curtail")
async def apply_curtailment(request: CurtailmentRequest):
    """Apply curtailment to battery system"""
    try:
        result = battery_agent.apply_curtailment(request.curtailment_kw)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "agent_id": BATTERY_AGENT_ID}

if __name__ == "__main__":
    print(f"Starting Battery Agent Server on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT) 