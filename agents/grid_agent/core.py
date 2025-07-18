import openai
from langgraph.graph import StateGraph
from collections import defaultdict
import csv
import json
from datetime import datetime
from typing import TypedDict, Dict, Any, cast
import os
from dotenv import load_dotenv
load_dotenv()

# Placeholder for configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set OpenAI API key
# def set_openai_key(key):
#     openai.api_key = key

# Placeholder: Use OpenAI to detect anomalies and plan curtailment
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
        # Placeholder: Simple anomaly detection (to be replaced with OpenAI logic)
        # For now, just check if any battery is not available and demand > generation
        total_generation = 0
        total_demand = 0
        for agent_status in self.agents.values():
            for module in agent_status.get("modules", []):
                if module["type"] == "solar":
                    total_generation += module.get("predicted_generation", 0) or 0
                if module["type"] == "load":
                    total_demand += module.get("predicted_demand", 0) or 0
        anomaly = total_generation < total_demand
        return anomaly, total_generation, total_demand
    def plan_curtailment(self, agent_lookup=None, curtail_loads_only=False):
        # Prompt OpenAI for a strict JSON curtailment plan (loads first, then solar if needed)
        prompt = f"""
Given the following agent statuses: {self.agents}.
There is a generation-demand anomaly (deficit). Respond ONLY with a JSON array of curtailment actions. Curtail loads first, and if that is not enough, curtail solar. Each action should have a 'type' (either 'load' or 'solar'), an 'id' (the full unique id, e.g., home_1_load_1), and 'curtailment_kw'. For example:
[
  {{"type": "load", "id": "home_1_load_1", "curtailment_kw": 1.0}},
  {{"type": "solar", "id": "home_2_solar_2", "curtailment_kw": 0.5}}
]
Do not include any explanation or extra text.
"""
        plan_text = openai_decision(prompt, OPENAI_API_KEY)
        try:
            plan = json.loads(plan_text)
        except Exception as e:
            print(f"Failed to parse curtailment plan as JSON: {e}\nAI response: {plan_text}")
            plan = []
        self.send_curtailment_plan(plan, agent_lookup)
        return plan

    def send_curtailment_plan(self, plan, agent_lookup=None):
        # Print the plan in natural language for the user
        if plan:
            print("Curtailment Plan (user-friendly):")
            for action in plan:
                t = action.get("type")
                i = action.get("id")
                k = action.get("curtailment_kw")
                print(f"  Curtail {k} kW from {t} {i}")
        else:
            print("No curtailment actions required.")
        # Send curtailment plan to respective agents
        for action in plan:
            agent_id = action.get("agent_id")  # for backward compatibility
            curtailment_kw = action.get("curtailment_kw")
            target_type = action.get("type")
            target_id = action.get("id")
            # Try to find the agent by id (for home/battery agents)
            if agent_id and agent_lookup and agent_id in agent_lookup:
                agent_lookup[agent_id].apply_curtailment(curtailment_kw)
            # Try to find the parent agent for the module (load/solar) by target_id
            elif target_type and target_id and agent_lookup:
                agent = agent_lookup.get(target_id)
                if agent and hasattr(agent, "apply_curtailment"):
                    agent.apply_curtailment(curtailment_kw, target_type=target_type, target_id=target_id)
                else:
                    print(f"No agent found for {target_type} {target_id} (curtail {curtailment_kw} kW)")
            else:
                print(f"Agent or module not found for curtailment action: {action}")

# GridAgent Node definition
class GridAgent:
    def __init__(self, transformers=None):
        # transformers: dict of transformer_id -> Transformer
        self.transformers = transformers or {}
    def log_status(self, log_data):
        log_filename = "grid_agent_log.csv"
        with open(log_filename, mode="a", newline="") as log_file:
            writer = csv.writer(log_file)
            writer.writerow([
                datetime.now().isoformat(),
                json.dumps(log_data)
            ])
    def run(self, state, agent_lookup=None):
        log_data = {}
        # state should contain status pings: {transformer_id: {agent_id: status_ping}}
        for transformer_id, transformer in self.transformers.items():
            # Update transformer with latest pings
            if transformer_id in state:
                for agent_id, status in state[transformer_id].items():
                    transformer.receive_status_ping(agent_id, status)
            # Detect anomaly
            anomaly, gen, demand = transformer.detect_anomaly()
            print(f"Transformer {transformer_id}: Generation={gen}, Demand={demand}, Anomaly={anomaly}")
            log_data[transformer_id] = {
                "generation": gen,
                "demand": demand,
                "anomaly": anomaly
            }
            if anomaly:
                print(f"Anomaly detected in Transformer {transformer_id}. Attempting local curtailment...")
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
                    # No surplus found, ask AI for curtailment plan (loads only)
                    plan = transformer.plan_curtailment(agent_lookup=agent_lookup, curtail_loads_only=True)
                    print(f"Curtailment Plan for {transformer_id}: {plan}")
                    log_data[transformer_id]["curtailment_plan"] = plan
                    # If plan is insufficient, escalate to neighbors (placeholder)
                    if not plan or (isinstance(plan, str) and "cannot resolve" in plan.lower()):
                        for neighbor_id in transformer.neighbor_ids:
                            print(f"Escalating to neighbor Transformer {neighbor_id}")
        self.log_status(log_data)
        return state

    def __call__(self, state):
        return self.run(state)

class ModuleStatus(TypedDict, total=False):
    type: str
    id: str
    soc: float
    available: bool
    available_to_charge: float
    predicted_generation: float
    predicted_demand: float

class AgentStatus(TypedDict, total=False):
    home_agent_id: str
    transformer_id: str
    battery_agent_id: str
    modules: list[ModuleStatus]

class GridAgentState(TypedDict, total=False):
    transformer_A: Dict[str, AgentStatus]
    transformer_B: Dict[str, AgentStatus]

# Example: Add GridAgent to a LangGraph StateGraph
def build_grid_agent_graph():
    # Example transformer setup
    transformer_A = Transformer("transformer_A", neighbor_ids=["transformer_B"])
    transformer_B = Transformer("transformer_B", neighbor_ids=["transformer_A"])
    transformers = {
        "transformer_A": transformer_A,
        "transformer_B": transformer_B
    }
    graph = StateGraph(GridAgentState)
    grid_agent = GridAgent(transformers=transformers)
    graph.add_node("grid_agent", grid_agent)
    graph.add_edge("__start__", "grid_agent")  # Use the string "__start__" for entrypoint
    return graph

if __name__ == "__main__":
    graph = build_grid_agent_graph()
    # Example state: {transformer_id: {agent_id: status_ping}}
    example_state = {
        "transformer_A": {
            "home_1": {
                "home_agent_id": "home_1",
                "transformer_id": "transformer_A",
                "modules": [
                    {"type": "battery", "id": "battery_1", "soc": 0.75, "available": True, "available_to_charge": 0.15},
                    {"type": "solar", "id": "solar_1", "predicted_generation": 2.5},
                    {"type": "load", "id": "load_1", "predicted_demand": 3.0}
                ]
            },
            "home_2": {
                "home_agent_id": "home_2",
                "transformer_id": "transformer_A",
                "modules": [
                    {"type": "battery", "id": "battery_2", "soc": 0.60, "available": True, "available_to_charge": 0.30},
                    {"type": "solar", "id": "solar_2", "predicted_generation": 1.8},
                    {"type": "load", "id": "load_2", "predicted_demand": 2.2}
                ]
            },
            "battery_agent_1": {
                "battery_agent_id": "battery_agent_1",
                "transformer_id": "transformer_A",
                "modules": [
                    {"type": "battery", "id": "battery_park_1", "soc": 0.80, "available": True, "available_to_charge": 0.10}
                ]
            }
        },
        "transformer_B": {
            "home_3": {
                "home_agent_id": "home_3",
                "transformer_id": "transformer_B",
                "modules": [
                    {"type": "battery", "id": "battery_3", "soc": 0.50, "available": True, "available_to_charge": 0.40},
                    {"type": "solar", "id": "solar_3", "predicted_generation": 2.0},
                    {"type": "load", "id": "load_3", "predicted_demand": 2.5}
                ]
            },
            "battery_agent_2": {
                "battery_agent_id": "battery_agent_2",
                "transformer_id": "transformer_B",
                "modules": [
                    {"type": "battery", "id": "battery_park_2", "soc": 0.65, "available": True, "available_to_charge": 0.25}
                ]
            }
        }
    }
    runnable = graph.compile()
    runnable.invoke(cast(GridAgentState, example_state)) 