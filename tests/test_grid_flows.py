import requests
import json
import time
import os

def run_test_scenario(scenario_name, agent_data, expected_checks, scenario_num):
    print(f"\n=== Running Scenario: {scenario_name} ===")
    # Prime each agent with the scenario data
    for agent_id, (endpoint, _) in agent_data.items():
        try:
            resp = requests.post(f"{endpoint}/simulate?scenario={scenario_num}", timeout=5)
            if resp.status_code != 200:
                print(f"[WARN] {agent_id} /simulate?scenario={scenario_num} returned {resp.status_code}")
        except Exception as e:
            print(f"[FAIL] Could not reach {agent_id} at {endpoint}: {e}")
            return
    # Give agents a moment to update
    time.sleep(1)
    # Set scenario for grid agent
    os.environ["TEST_SCENARIO"] = str(scenario_num)
    # Call grid agent's /coordinate endpoint
    try:
        resp = requests.post("http://localhost:8000/coordinate", timeout=15)
        if resp.status_code == 200:
            result = resp.json()
            curtailment = result.get("curtailment_results") or result.get("log_data", {})
            # Run expected checks
            passed = True
            for check in expected_checks:
                if not check(curtailment):
                    print(f"[FAIL] {check.__doc__}")
                    passed = False
            if passed:
                print("[PASS] All checks passed.")
        else:
            print(f"[FAIL] Grid agent returned status {resp.status_code}")
    except Exception as e:
        print(f"[FAIL] Exception: {e}")

# --- Scenario Definitions ---
all_agents = {
    "home_1": ("http://localhost:8001", {}),
    "home_2": ("http://localhost:8002", {}),
    "home_3": ("http://localhost:8003", {}),
    "battery_agent_1": ("http://localhost:8004", {}),
    "battery_agent_2": ("http://localhost:8005", {}),
}

scenario1_agents = all_agents
def check_no_escalation(curtailment):
    "No escalation to neighbor transformer."
    if isinstance(curtailment, dict):
        for t in curtailment.values():
            if "transfer_from_neighbor" in t:
                return False
    return True

scenario2_agents = all_agents
def check_escalation(curtailment):
    "Escalation to neighbor transformer occurs."
    if isinstance(curtailment, dict):
        for t in curtailment.values():
            if "transfer_from_neighbor" in t:
                return True
    return False

scenario3_agents = all_agents
def check_battery_charging(curtailment):
    "Battery is charged or solar is curtailed when generation > demand."
    if isinstance(curtailment, list):
        for action in curtailment:
            if (action.get("type") == "battery" and action.get("action") == "charge") or \
               (action.get("type") == "solar" and action.get("action") == "curtail"):
                return True
    return False

scenario4_agents = all_agents
def check_load_curtailment(curtailment):
    "Load is curtailed when generation < demand."
    if isinstance(curtailment, list):
        for action in curtailment:
            if action.get("type") == "load" and action.get("action") in ("curtail", "turn_off"):
                return True
    return False

if __name__ == "__main__":
    run_test_scenario("Resolve within transformer", scenario1_agents, [check_no_escalation], 1)
    run_test_scenario("Escalate to neighbor transformer", scenario2_agents, [check_escalation], 2)
    run_test_scenario("Generation > Demand", scenario3_agents, [check_battery_charging], 3)
    run_test_scenario("Generation < Demand", scenario4_agents, [check_load_curtailment], 4) 