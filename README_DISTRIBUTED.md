# Distributed Energy Management Simulation

This project implements a distributed multi-agent energy management system where each agent runs as an independent HTTP server, communicating via REST APIs for a more realistic distributed simulation.

## Architecture

The system consists of three main agents:

1. **Grid Agent** (Port 8000) - Coordinates the entire system and makes curtailment decisions
2. **Home Agent** (Port 8001) - Manages home energy systems (solar, loads, batteries)
3. **Battery Agent** (Port 8002) - Manages battery storage systems

## Features

- **Distributed Architecture**: Each agent runs as an independent FastAPI server
- **HTTP Communication**: Agents communicate via REST APIs
- **AI-Powered Decisions**: Uses OpenAI GPT-3.5-turbo for intelligent decision making
- **Real-time Coordination**: Grid agent coordinates all other agents
- **Curtailment Planning**: Intelligent load and solar curtailment based on demand/supply
- **Health Monitoring**: Each agent provides health check endpoints
- **Logging**: Comprehensive logging of all operations

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the required CSV files:
   - `historical_data.csv` - for home agent
   - `battery_historical_data.csv` - for battery agent

## Running the Simulation

### Option 1: Run All Agents Together (Recommended)

Use the coordinator script to start all agents and run the simulation:

```bash
python run_distributed_simulation.py
```

This will:
- Start all three agent servers
- Run 3 simulation cycles
- Display results and status
- Keep running until you press Ctrl+C

### Option 2: Run Agents Individually

You can also run each agent separately in different terminals:

**Terminal 1 - Grid Agent:**
```bash
python grid_agent_server.py
```

**Terminal 2 - Home Agent:**
```bash
python home_agent_server.py
```

**Terminal 3 - Battery Agent:**
```bash
python battery_agent_server.py
```

## API Endpoints

### Grid Agent (http://localhost:8000)
- `GET /` - Server info
- `GET /status` - Current transformer status
- `POST /coordinate` - Run grid coordination
- `GET /agents` - List agent endpoints
- `POST /agents/{agent_id}/status` - Receive agent status
- `GET /health` - Health check

### Home Agent (http://localhost:8001)
- `GET /` - Server info
- `GET /status` - Current module status
- `POST /simulate` - Run simulation cycle
- `POST /curtail` - Apply curtailment
- `GET /health` - Health check

### Battery Agent (http://localhost:8002)
- `GET /` - Server info
- `GET /status` - Current battery status
- `POST /simulate` - Run simulation cycle
- `POST /curtail` - Apply curtailment
- `GET /health` - Health check

## Simulation Flow

1. **Initialization**: All agents start and register with the grid agent
2. **Status Collection**: Grid agent collects status from all agents
3. **Anomaly Detection**: Grid agent detects generation-demand imbalances
4. **Curtailment Planning**: AI generates curtailment plans if needed
5. **Action Execution**: Curtailment actions are sent to respective agents
6. **Logging**: All actions are logged for analysis

## Configuration

Key configuration parameters can be modified in each agent file:

- **API Keys**: OpenAI and OpenWeatherMap API keys
- **Ports**: Server ports for each agent
- **Agent IDs**: Unique identifiers for each agent
- **Transformer IDs**: Grid transformer assignments

## Logging

Each agent creates its own log file:
- `home_1_log.csv` - Home agent logs
- `battery_agent_1_log.csv` - Battery agent logs
- `grid_agent_log.csv` - Grid agent logs

## Troubleshooting

### Common Issues

1. **Port Already in Use**: Make sure ports 8000, 8001, and 8002 are available
2. **API Key Issues**: Verify your OpenAI API key is valid
3. **Missing CSV Files**: Ensure historical data files exist
4. **Network Issues**: Check if localhost is accessible

### Debug Mode

To see detailed logs, check the console output of each agent server. The coordinator script provides a summary of all operations.

## Extending the System

### Adding New Agents

1. Create a new agent server file (e.g., `new_agent_server.py`)
2. Implement the required endpoints (`/status`, `/simulate`, `/curtail`, `/health`)
3. Update the coordinator configuration in `run_distributed_simulation.py`
4. Add the agent endpoint to the grid agent configuration

### Adding New Module Types

1. Create new module classes (similar to `BatteryModule`, `SolarModule`, `LoadModule`)
2. Update the Pydantic models for API serialization
3. Implement the required methods (`status()`, etc.)

## Performance Considerations

- The simulation runs with realistic timing (5-second intervals between cycles)
- Each agent operates independently, allowing for true distributed operation
- HTTP communication adds realistic network latency
- All operations are logged for performance analysis

## Security Notes

- This is a development/demo system
- API keys are hardcoded for simplicity
- In production, use environment variables for sensitive data
- Consider adding authentication for agent communication 