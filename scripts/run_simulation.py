from agents.home_agent.core import HomeAgent
from agents.battery_agent.core import BatteryAgent
from agents.grid_agent.core import build_grid_agent_graph, GridAgentState
from typing import cast
import os
from dotenv import load_dotenv
load_dotenv()

# Configuration (update as needed)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
LOCATION = "Bangalore"
HISTORICAL_CSV_PATH = "historical_data.csv"
BATTERY_HISTORICAL_CSV_PATH = "battery_historical_data.csv"

# Instantiate Home Agents
home_1 = HomeAgent(
    csv_path=HISTORICAL_CSV_PATH,
    weather_api_key=OPENWEATHERMAP_API_KEY,
    openai_api_key=OPENAI_API_KEY,
    location=LOCATION,
    home_agent_id="home_1",
    transformer_id="transformer_A"
)
home_2 = HomeAgent(
    csv_path=HISTORICAL_CSV_PATH,
    weather_api_key=OPENWEATHERMAP_API_KEY,
    openai_api_key=OPENAI_API_KEY,
    location=LOCATION,
    home_agent_id="home_2",
    transformer_id="transformer_A"
)
home_3 = HomeAgent(
    csv_path=HISTORICAL_CSV_PATH,
    weather_api_key=OPENWEATHERMAP_API_KEY,
    openai_api_key=OPENAI_API_KEY,
    location=LOCATION,
    home_agent_id="home_3",
    transformer_id="transformer_B"
)

# Instantiate Battery Agents
battery_agent_1 = BatteryAgent(
    csv_path=BATTERY_HISTORICAL_CSV_PATH,
    weather_api_key=OPENWEATHERMAP_API_KEY,
    openai_api_key=OPENAI_API_KEY,
    location=LOCATION,
    battery_agent_id="battery_agent_1",
    transformer_id="transformer_A"
)
battery_agent_2 = BatteryAgent(
    csv_path=BATTERY_HISTORICAL_CSV_PATH,
    weather_api_key=OPENWEATHERMAP_API_KEY,
    openai_api_key=OPENAI_API_KEY,
    location=LOCATION,
    battery_agent_id="battery_agent_2",
    transformer_id="transformer_B"
)

# Run each agent and collect status pings
def get_status(agent, agent_type):
    # Patch: return the status ping instead of state
    import io
    import sys
    captured_output = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = captured_output
    agent.run({})
    sys.stdout = sys_stdout
    # Extract the last printed status ping
    lines = captured_output.getvalue().splitlines()
    for line in reversed(lines):
        if "Status Ping to Grid Agent:" in line:
            # Evaluate the dict from the print statement
            import ast
            return ast.literal_eval(line.split("Status Ping to Grid Agent:", 1)[1].strip())
    return {}

state = {
    "transformer_A": {
        "home_1": get_status(home_1, "home"),
        "home_2": get_status(home_2, "home"),
        "battery_agent_1": get_status(battery_agent_1, "battery")
    },
    "transformer_B": {
        "home_3": get_status(home_3, "home"),
        "battery_agent_2": get_status(battery_agent_2, "battery")
    }
}

# Build agent lookup for curtailment communication
agent_lookup = {
    "home_1": home_1,
    "home_2": home_2,
    "home_3": home_3,
    "battery_agent_1": battery_agent_1,
    "battery_agent_2": battery_agent_2,
}
# Add loads and solar modules to agent_lookup using their unique IDs
for home in [home_1, home_2, home_3]:
    for load in home.loads:
        agent_lookup[load.id] = home
    for solar in home.solars:
        agent_lookup[solar.id] = home

# Build and run the Grid Agent
graph = build_grid_agent_graph()
runnable = graph.compile()
result = runnable.invoke(cast(GridAgentState, state))

# After running the grid agent, trigger curtailment plan (if needed)
# Assume the grid agent node is accessible as a node in the graph
# (If not, you may need to refactor to keep a reference to the grid agent object)

# Example: If you have a reference to the grid agent object
# grid_agent.plan_curtailment(agent_lookup)
#
# If not, you may need to extract the plan from the result or refactor the code to support this.
#
# For demonstration, here's a stub call (replace with actual reference as needed):
# grid_agent = ...
# grid_agent.plan_curtailment(agent_lookup) 