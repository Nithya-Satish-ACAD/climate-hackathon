import subprocess
import time
import requests
import json
from typing import Dict, List
import signal
import sys

# Configuration
AGENT_CONFIGS = {
    "grid_agent": {
        "script": "grid_agent_server.py",
        "port": 8000,
        "name": "Grid Agent"
    },
    "home_agent": {
        "script": "home_agent_server.py", 
        "port": 8001,
        "name": "Home Agent"
    },
    "battery_agent": {
        "script": "battery_agent_server.py",
        "port": 8002,
        "name": "Battery Agent"
    }
}

class DistributedSimulationCoordinator:
    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.agent_endpoints = {
            "home_1": "http://localhost:8001",
            "battery_agent_1": "http://localhost:8002"
        }
    
    def start_agent(self, agent_name: str, config: Dict):
        """Start an agent server"""
        try:
            print(f"Starting {config['name']} on port {config['port']}...")
            process = subprocess.Popen(
                [sys.executable, config['script']],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.processes[agent_name] = process
            print(f"✅ {config['name']} started (PID: {process.pid})")
            return True
        except Exception as e:
            print(f"❌ Failed to start {config['name']}: {e}")
            return False
    
    def wait_for_agent_ready(self, port: int, timeout: int = 30) -> bool:
        """Wait for an agent to be ready by checking its health endpoint"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"http://localhost:{port}/health", timeout=5)
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
        return False
    
    def start_all_agents(self):
        """Start all agent servers"""
        print("🚀 Starting distributed energy management simulation...")
        print("=" * 60)
        
        # Start agents in order
        for agent_name, config in AGENT_CONFIGS.items():
            if not self.start_agent(agent_name, config):
                print(f"Failed to start {agent_name}. Stopping all agents...")
                self.stop_all_agents()
                return False
            
            # Wait for agent to be ready
            if not self.wait_for_agent_ready(config['port']):
                print(f"❌ {config['name']} failed to start within timeout")
                self.stop_all_agents()
                return False
            
            print(f"✅ {config['name']} is ready")
            time.sleep(2)  # Brief pause between starts
        
        print("=" * 60)
        print("🎉 All agents started successfully!")
        return True
    
    def run_simulation_cycle(self, cycle_count: int = 1):
        """Run simulation cycles"""
        print(f"\n🔄 Running {cycle_count} simulation cycle(s)...")
        
        for cycle in range(1, cycle_count + 1):
            print(f"\n--- Simulation Cycle {cycle} ---")
            
            try:
                # Trigger grid coordination
                response = requests.post("http://localhost:8000/coordinate", timeout=30)
                if response.status_code == 200:
                    results = response.json()
                    print("📊 Grid coordination completed successfully")
                    
                    # Log results
                    log_data = results.get("log_data", {})
                    for transformer_id, data in log_data.items():
                        print(f"  Transformer {transformer_id}:")
                        print(f"    Generation: {data.get('generation', 0):.2f} kW")
                        print(f"    Demand: {data.get('demand', 0):.2f} kW")
                        print(f"    Anomaly: {data.get('anomaly', False)}")
                        
                        if data.get('curtailment_plan'):
                            print(f"    Curtailment actions: {len(data['curtailment_plan']['actions'])}")
                    
                    curtailment_results = results.get("curtailment_results", [])
                    if curtailment_results:
                        print(f"  Curtailment results: {len(curtailment_results)} actions")
                        for result in curtailment_results:
                            action = result.get("action", {})
                            status = result.get("status")
                            print(f"    {action.get('type', 'unknown')} {action.get('id', 'unknown')}: {status}")
                else:
                    print(f"❌ Grid coordination failed: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Error during simulation cycle {cycle}: {e}")
            
            if cycle < cycle_count:
                print("⏳ Waiting 5 seconds before next cycle...")
                time.sleep(5)
    
    def get_agent_statuses(self):
        """Get status from all agents"""
        print("\n📋 Agent Status Report:")
        print("-" * 40)
        
        for agent_name, config in AGENT_CONFIGS.items():
            try:
                response = requests.get(f"http://localhost:{config['port']}/status", timeout=5)
                if response.status_code == 200:
                    status = response.json()
                    print(f"✅ {config['name']}: {status}")
                else:
                    print(f"❌ {config['name']}: HTTP {response.status_code}")
            except Exception as e:
                print(f"❌ {config['name']}: {e}")
    
    def stop_agent(self, agent_name: str):
        """Stop a specific agent"""
        if agent_name in self.processes:
            process = self.processes[agent_name]
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"✅ {AGENT_CONFIGS[agent_name]['name']} stopped")
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"⚠️  {AGENT_CONFIGS[agent_name]['name']} force killed")
            except Exception as e:
                print(f"❌ Error stopping {agent_name}: {e}")
    
    def stop_all_agents(self):
        """Stop all agent servers"""
        print("\n🛑 Stopping all agents...")
        for agent_name in list(self.processes.keys()):
            self.stop_agent(agent_name)
        self.processes.clear()
        print("✅ All agents stopped")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"\n🛑 Received signal {signum}. Shutting down...")
        self.stop_all_agents()
        sys.exit(0)

def main():
    coordinator = DistributedSimulationCoordinator()
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, coordinator.signal_handler)
    signal.signal(signal.SIGTERM, coordinator.signal_handler)
    
    try:
        # Start all agents
        if not coordinator.start_all_agents():
            return
        
        # Show initial status
        coordinator.get_agent_statuses()
        
        # Run simulation cycles
        cycles = 3  # Run 3 cycles by default
        coordinator.run_simulation_cycle(cycles)
        
        # Show final status
        coordinator.get_agent_statuses()
        
        print("\n🎯 Simulation completed successfully!")
        print("Press Ctrl+C to stop all agents...")
        
        # Keep running until interrupted
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    finally:
        coordinator.stop_all_agents()

if __name__ == "__main__":
    main() 