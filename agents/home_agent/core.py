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
HISTORICAL_CSV_PATH = "historical_data.csv"
HOME_AGENT_ID = "home_1"
TRANSFORMER_ID = "transformer_A"

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

# Placeholder: Use OpenAI to predict generation/demand or decide availability
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

default_batteries = [
    {"id": "battery_1", "threshold": 0.2},
    {"id": "battery_2", "threshold": 0.25}
]
default_solars = [
    {"id": "solar_1"},
    {"id": "solar_2"}
]
default_loads = [
    {"id": "load_1"},
    {"id": "load_2"}
]

class BatteryModule:
    def __init__(self, id, threshold):
        self.id = id
        self.threshold = threshold
        self.soc: float | None = None
        self.available: bool | None = None
        self.available_to_charge: float | None = None  # New field
    def status(self):
        return {
            "type": "battery",
            "id": self.id,
            "soc": self.soc,
            "available": self.available,
            "available_to_charge": self.available_to_charge
        }

class SolarModule:
    def __init__(self, id):
        self.id = id
        self.predicted_generation: float | None = None
    def status(self):
        return {"type": "solar", "id": self.id, "predicted_generation": self.predicted_generation}

class LoadModule:
    def __init__(self, id):
        self.id = id
        self.predicted_demand: float | None = None
    def status(self):
        return {"type": "load", "id": self.id, "predicted_demand": self.predicted_demand}

class HomeAgentState(TypedDict, total=False):
    pass

class HomeAgent:
    def __init__(self, csv_path, weather_api_key, openai_api_key, location, home_agent_id, transformer_id,
                 batteries=None, solars=None, loads=None):
        self.csv_path = csv_path
        self.weather_api_key = weather_api_key
        self.openai_api_key = openai_api_key
        self.location = location
        self.home_agent_id = home_agent_id
        self.transformer_id = transformer_id
        self.historical_data = load_historical_data(csv_path)
        self.batteries = [BatteryModule(**b) for b in (batteries or default_batteries)]
        # Assign unique IDs to solars and loads
        self.solars = [SolarModule(id=f"{home_agent_id}_{s['id']}") for s in (solars or default_solars)]
        self.loads = [LoadModule(id=f"{home_agent_id}_{l['id']}") for l in (loads or default_loads)]

    def log_status(self, status_ping):
        log_filename = f"{self.home_agent_id}_log.csv"
        with open(log_filename, mode="a", newline="") as log_file:
            writer = csv.writer(log_file)
            writer.writerow([
                datetime.now().isoformat(),
                json.dumps(status_ping)
            ])

    def run(self, state):
        data = self.historical_data
        weather = fetch_weather(self.weather_api_key, self.location)
        # Predict solar generation
        for solar in self.solars:
            prompt = f"""
            Given the following historical solar data: {data.tail(10).to_dict()} and current weather: {weather},
            predict the solar generation for the next 5 minutes for solar module {solar.id}.
            Respond with only a single number (no units, no explanation).
            """
            result = predict_with_openai(prompt, self.openai_api_key)
            print(f"OpenAI response for solar {solar.id}: {result}")  # Debug print
            try:
                solar.predicted_generation = float(result.split()[0])
            except Exception as e:
                print(f"Failed to parse OpenAI response for solar {solar.id}: {e}")
                solar.predicted_generation = 0.0  # Fallback to 0.0
        # Predict load demand
        for load in self.loads:
            prompt = f"""
            Given the following historical load data: {data.tail(10).to_dict()} and current weather: {weather},
            predict the load demand for the next 5 minutes for load module {load.id}.
            Respond with only a single number (no units, no explanation).
            """
            result = predict_with_openai(prompt, self.openai_api_key)
            print(f"OpenAI response for load {load.id}: {result}")  # Debug print
            try:
                load.predicted_demand = float(result.split()[0])
            except Exception as e:
                print(f"Failed to parse OpenAI response for load {load.id}: {e}")
                load.predicted_demand = 0.0  # Fallback to 0.0
        # Generate random SOC for each battery and decide availability using OpenAI
        for battery in self.batteries:
            battery.soc = round(random.uniform(0, 1), 2)
            # Calculate available_to_charge based on max SOC of 0.9
            soc_val = battery.soc if battery.soc is not None else 0.0
            battery.available_to_charge = max(0.0, round(0.9 - soc_val, 2))
            # Prepare context for OpenAI to decide availability
            total_predicted_demand = sum(l.predicted_demand or 0 for l in self.loads)
            total_predicted_generation = sum(s.predicted_generation or 0 for s in self.solars)
            prompt = f"""
            Battery SOC: {battery.soc}, Threshold: {battery.threshold}, Max SOC: 0.9, Available to charge: {battery.available_to_charge},
            Predicted total demand: {total_predicted_demand}, Predicted total generation: {total_predicted_generation}, Weather: {weather}.
            Should the battery (ID: {battery.id}) be available for charge/discharge? Respond with 'True' or 'False' and a brief reason.
            """
            result = predict_with_openai(prompt, self.openai_api_key)
            battery.available = "true" in result.lower()
        modules_status = [b.status() for b in self.batteries] + \
                         [s.status() for s in self.solars] + \
                         [l.status() for l in self.loads]
        status_ping = {
            "home_agent_id": self.home_agent_id,
            "transformer_id": self.transformer_id,
            "modules": modules_status
        }
        print("Status Ping to Grid Agent:", status_ping)
        self.log_status(status_ping)
        return state

    def apply_curtailment(self, curtailment_kw, target_type=None, target_id=None):
        if target_type == "load":
            for load in self.loads:
                if load.id == target_id:
                    print(f"Curtailing {curtailment_kw} kW from load {target_id} in HomeAgent {self.home_agent_id}")
                    # Implement logic to reduce load here
                    return
            print(f"Load {target_id} not found in HomeAgent {self.home_agent_id}")
        elif target_type == "solar":
            for solar in self.solars:
                if solar.id == target_id:
                    print(f"Curtailing {curtailment_kw} kW from solar {target_id} in HomeAgent {self.home_agent_id}")
                    # Implement logic to reduce solar generation here
                    return
            print(f"Solar {target_id} not found in HomeAgent {self.home_agent_id}")
        else:
            print(f"HomeAgent {self.home_agent_id} curtailing {curtailment_kw} kW (no specific target)")

    def __call__(self, state):
        return self.run(state)

# Example: Add HomeAgent to a LangGraph StateGraph
def build_home_agent_graph():
    graph = StateGraph(HomeAgentState)
    home_agent = HomeAgent(
        csv_path=HISTORICAL_CSV_PATH,
        weather_api_key=OPENWEATHERMAP_API_KEY,
        openai_api_key=OPENAI_API_KEY,
        location=LOCATION,
        home_agent_id=HOME_AGENT_ID,
        transformer_id=TRANSFORMER_ID
    )
    graph.add_node("home_agent", home_agent)
    return graph

if __name__ == "__main__":
    graph = build_home_agent_graph()
    runnable = graph.compile()
    runnable.invoke({}) 