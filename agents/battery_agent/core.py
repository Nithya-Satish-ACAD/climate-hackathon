import pandas as pd
import requests
import openai
import random
import csv
import json
from datetime import datetime
from langgraph.graph import StateGraph
from typing import TypedDict
import os
from dotenv import load_dotenv
load_dotenv()

# Placeholder for configuration (API keys, file paths, etc.)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
LOCATION = "Bangalore"  # e.g., city name or coordinates
HISTORICAL_CSV_PATH = "battery_historical_data.csv"
BATTERY_AGENT_ID = "battery_agent_1"
TRANSFORMER_ID = "transformer_B"

# All batteries have a threshold of 0.9 (90%)
default_batteries = [
    {"id": "battery_park_1"},
    {"id": "battery_park_2"}
]

# Set OpenAI API key
# def set_openai_key(key):
#     openai.api_key = key

# Placeholder: Load historical data from CSV
def load_historical_data(csv_path):
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return pd.DataFrame()

# Placeholder: Fetch weather data from OpenWeatherMap
def fetch_weather(api_key, location):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        return response.json()
    except Exception as e:
        print(f"Error fetching weather: {e}")
        return {}

# Placeholder: Use OpenAI to predict demand/availability
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

class BatteryModule:
    def __init__(self, id):
        self.id = id
        self.soc: float | None = None
        self.available: bool | None = None
        self.available_to_charge: float | None = None
    def status(self):
        return {
            "type": "battery",
            "id": self.id,
            "soc": self.soc,
            "available": self.available,
            "available_to_charge": self.available_to_charge
        }

# BatteryAgent Node definition
class BatteryAgent:
    def __init__(self, csv_path, weather_api_key, openai_api_key, location, battery_agent_id, transformer_id, batteries=None):
        self.csv_path = csv_path
        self.weather_api_key = weather_api_key
        self.openai_api_key = openai_api_key
        self.location = location
        self.battery_agent_id = battery_agent_id
        self.transformer_id = transformer_id
        self.historical_data = load_historical_data(csv_path)
        self.batteries = [BatteryModule(**b) for b in (batteries or default_batteries)]

    def log_status(self, status_ping):
        log_filename = f"{self.battery_agent_id}_log.csv"
        with open(log_filename, mode="a", newline="") as log_file:
            writer = csv.writer(log_file)
            writer.writerow([
                datetime.now().isoformat(),
                json.dumps(status_ping)
            ])

    def run(self, state):
        data = self.historical_data
        weather = fetch_weather(self.weather_api_key, self.location)
        # Generate random SOC for each battery and decide availability using OpenAI
        for battery in self.batteries:
            battery.soc = round(random.uniform(0, 1), 2)
            threshold = 0.9
            soc_val = battery.soc if battery.soc is not None else 0.0
            battery.available_to_charge = max(0.0, round(threshold - soc_val, 2))
            prompt = f"""
            Battery SOC: {battery.soc}, Threshold: {threshold}, Max SOC: 0.9, Available to charge: {battery.available_to_charge}, Weather: {weather}.
            Should the battery (ID: {battery.id}) be available for charge/discharge? Respond with 'True' or 'False' and a brief reason.
            """
            result = predict_with_openai(prompt, self.openai_api_key)
            print(f"OpenAI response for battery {battery.id}: {result}")  # Debug print
            try:
                battery.available = "true" in result.lower()
            except Exception as e:
                print(f"Failed to parse OpenAI response for battery {battery.id}: {e}")
                battery.available = False  # Fallback to False
        modules_status = [b.status() for b in self.batteries]
        status_ping = {
            "battery_agent_id": self.battery_agent_id,
            "transformer_id": self.transformer_id,
            "modules": modules_status
        }
        print("Status Ping to Grid Agent:", status_ping)
        self.log_status(status_ping)
        return state

    def apply_curtailment(self, curtailment_kw):
        print(f"BatteryAgent {self.battery_agent_id} curtailing {curtailment_kw} kW")
        # Implement logic to reduce battery output as needed

    def __call__(self, state):
        return self.run(state)

class BatteryAgentState(TypedDict, total=False):
    pass

# Example: Add BatteryAgent to a LangGraph StateGraph
def build_battery_agent_graph():
    graph = StateGraph(BatteryAgentState)
    battery_agent = BatteryAgent(
        csv_path=HISTORICAL_CSV_PATH,
        weather_api_key=OPENWEATHERMAP_API_KEY,
        openai_api_key=OPENAI_API_KEY,
        location=LOCATION,
        battery_agent_id=BATTERY_AGENT_ID,
        transformer_id=TRANSFORMER_ID
    )
    graph.add_node("battery_agent", battery_agent)
    return graph

if __name__ == "__main__":
    graph = build_battery_agent_graph()
    runnable = graph.compile()
    runnable.invoke({}) 