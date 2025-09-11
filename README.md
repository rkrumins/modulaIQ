# AI Multi-Agent Supervisor Framework

A comprehensive Python framework for creating hierarchical multi-agent systems using LangGraph and other LLM frameworks. The framework features a central Supervisor agent that orchestrates tasks among specialized agents using an enhanced ReAct (Reasoning, Acting, Observing) paradigm with deterministic multi-agent task decomposition.

## Features

- **Enhanced ReAct Supervisor**: Central orchestrator with deterministic reasoning and intelligent multi-agent task decomposition
- **Multi-Agent Task Decomposition**: Automatically breaks down complex queries into specialized agent tasks
- **Microservices Architecture**: Each agent runs as an independent FastAPI service
- **Unified API**: Consistent REST API for all agent communications
- **Dynamic Agent Registration**: Agents can register and unregister dynamically
- **Memory Management**: In-memory conversation memory with SQLite, dictionary, or Redis backends
- **Configurable LLM Providers**: Support for Groq, OpenAI, and Google Gemini
- **Extensible Design**: Easy to add new agents and capabilities
- **YAML Configuration**: All settings configurable via YAML files
- **Robust Error Handling**: Intelligent fallback mechanisms and context-aware agent selection

## Architecture

```text
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Supervisor    │    │   General       │    │   Code          │
│   (Port 8000)   │◄──►│   Agent         │    │   Agent         │
│                 │    │   (Port 8002)   │    │   (Port 8003)   │
│  Enhanced ReAct │    │                 │    │                 │
│  Multi-Agent    │    │                 │    │                 │
│  Orchestration  │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐    ┌─────────────────┐
         └──────────────►│   Research      │    │   Creative      │
                        │   Agent         │    │   Agent         │
                        │   (Port 8004)   │    │   (Port 8005)   │
                        └─────────────────┘    └─────────────────┘
```

### Multi-Agent Task Decomposition

The Supervisor automatically detects and decomposes complex queries:

- **Single-Part Queries**: "Write a Python function" → Uses appropriate specialized agent
- **Multi-Part Queries**: "Research Fibonacci origins and code it in Python" → Uses multiple agents sequentially
  - Step 1: Research Agent (origins/history)
  - Step 2: Code Agent (Python implementation)
  - Step 3: Synthesis (combines responses)

### Enhanced ReAct Paradigm

1. **Reasoning**: Deterministic analysis of query complexity and agent requirements
2. **Acting**: Intelligent agent selection and task decomposition
3. **Observing**: Context-aware monitoring and response synthesis

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd agent_orchestrator

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Copy the example environment file and configure your API keys:

```bash
cp env.example .env
```

Edit `.env` and add your API keys:

```bash
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

### 3. Start the System

#### Option A: Automated Startup (Recommended)

```bash
# Start everything with automatic registration
python start_with_registration.py
```

#### Option B: Manual Startup

```bash
# Terminal 1: Start the Supervisor first
python run_supervisor.py

# Terminal 2: Start all agents (they will auto-register)
python run_agents.py
```

#### Option C: Individual Agents

```bash
# Terminal 1: Start the Supervisor first
python run_supervisor.py

# Terminal 2: Start individual agents
python agents/examples/general_agent.py
python agents/examples/code_agent.py
python agents/examples/research_agent.py
python agents/examples/creative_agent.py
```

**Important**: Always start the supervisor first, then the agents. The agents will automatically register with the supervisor.

### 4. Test the System

```bash
# Test supervisor health
curl http://localhost:8000/health

# List registered agents
curl http://localhost:8000/agents

# Test single-agent query
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test-session",
    "query": "Help me write a Python function to calculate fibonacci numbers"
  }'

# Test multi-agent query (will use both research and code agents)
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test-session",
    "query": "Research the origins of Fibonacci and code this up in Python"
  }'
```

### 5. Run Tests

```bash
# Run comprehensive system tests
python test_system.py

# Test multi-agent workflows
python test_multi_agent_workflow.py

# Test agent selection
python test_llm_agent_selection.py
```

## Configuration

The system is configured via `config.yaml`. Key configuration options:

### LLM Providers

```yaml
llm_providers:
  groq:
    api_key_env: "GROQ_API_KEY"
    default_model: "llama-3.1-8b-instant"
  openai:
    api_key_env: "OPENAI_API_KEY"
    default_model: "gpt-4"
  google:
    api_key_env: "GOOGLE_API_KEY"
    default_model: "gemini-pro"
```

### Memory Backend

```yaml
memory:
  # In-memory backends (default)
  in_memory_sqlite:
    # No configuration needed - uses SQLite in-memory mode
  
  in_memory_dict:
    # No configuration needed - uses Python dictionaries
  
  # Persistent backends (optional)
  sqlite:
    database_url: "sqlite:///./agent_memory.db"
  
  redis:
    host: "localhost"
    port: 6379
```

### Supervisor Settings

```yaml
supervisor:
  host: "localhost"
  port: 8000
  llm_provider: "groq"
  model: "llama-3.1-8b-instant"
  max_iterations: 10
  memory_backend: "in_memory_sqlite"
  rate_limiting:
    min_llm_call_interval: 2.0  # Minimum seconds between LLM calls
    max_requests_per_minute: 20  # Rate limiting for LLM calls
```

## API Reference

### Supervisor Endpoints

- `GET /health` - Health check
- `GET /agents` - List all registered agents
- `POST /agents/register` - Register a new agent
- `DELETE /agents/{agent_id}` - Unregister an agent
- `POST /execute` - Execute a task
- `GET /sessions/{session_id}/history` - Get conversation history
- `POST /sessions/{session_id}/clear` - Clear conversation history

### Agent Endpoints

Each agent exposes these endpoints:

- `GET /health` - Health check
- `GET /info` - Agent information
- `POST /execute` - Execute agent task
- `GET /sessions/{session_id}/history` - Get conversation history
- `POST /sessions/{session_id}/clear` - Clear conversation history

## Creating Custom Agents

### Simple Agent

```python
from agents.langgraph_agent import SimpleAgent
from core.config import Config

class MyAgent(SimpleAgent):
    def __init__(self, config: Config):
        super().__init__(
            config=config,
            agent_id="my_agent",
            name="My Custom Agent",
            description="A custom agent for specific tasks",
            capabilities=["task1", "task2"],
            system_prompt="You are a specialized agent for..."
        )

# Run the agent
if __name__ == "__main__":
    config = Config.load_from_yaml()
    agent = MyAgent(config)
    asyncio.run(agent.run(port=8006))
```

### LangGraph Agent with Tools

```python
from agents.langgraph_agent import LangGraphAgent
from langchain_core.tools import tool

@tool
def my_tool(input_text: str) -> str:
    """A custom tool for the agent."""
    return f"Processed: {input_text}"

class MyLangGraphAgent(LangGraphAgent):
    def __init__(self, config: Config):
        super().__init__(
            config=config,
            agent_id="my_langgraph_agent",
            name="My LangGraph Agent",
            description="A LangGraph agent with custom tools",
            capabilities=["advanced_processing"],
            system_prompt="You are an advanced agent...",
            tools=[my_tool]
        )
```

## Example Usage

### Python Client

```python
import httpx
import asyncio

async def test_supervisor():
    async with httpx.AsyncClient() as client:
        # Test single-agent query
        response = await client.post(
            "http://localhost:8000/execute",
            json={
                "session_id": "test-session",
                "query": "Help me analyze this code and suggest improvements"
            }
        )
        
        result = response.json()
        print(f"Response: {result['response']}")
        print(f"Agents used: {result['agents_used']}")
        print(f"Iterations: {result['iterations']}")
        
        # Test multi-agent query
        response = await client.post(
            "http://localhost:8000/execute",
            json={
                "session_id": "test-session",
                "query": "Research the history of machine learning and write a Python script to demonstrate basic ML concepts"
            }
        )
        
        result = response.json()
        print(f"Multi-agent response: {result['response']}")
        print(f"Agents used: {result['agents_used']}")  # Should show ['research', 'code']
        print(f"Iterations: {result['iterations']}")

asyncio.run(test_supervisor())
```

### JavaScript Client

```javascript
// Single-agent query
const response = await fetch('http://localhost:8000/execute', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    session_id: 'test-session',
    query: 'Help me write a creative story about a robot'
  })
});

const result = await response.json();
console.log('Response:', result.response);
console.log('Agents used:', result.agents_used);

// Multi-agent query
const multiResponse = await fetch('http://localhost:8000/execute', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    session_id: 'test-session',
    query: 'Research quantum computing basics and create a simple Python simulation'
  })
});

const multiResult = await multiResponse.json();
console.log('Multi-agent response:', multiResult.response);
console.log('Agents used:', multiResult.agents_used); // Should show multiple agents
```

## Memory Management

The framework supports multiple memory backends:

### In-Memory Backends (Default)

**SQLite In-Memory** (Default):
- Fast, thread-safe, and requires no setup
- Data is stored in RAM and lost when the process ends
- Perfect for development and testing

**Python Dictionary**:
- Simple Python dictionary-based storage
- Fastest option for small datasets
- Data is stored in RAM and lost when the process ends

### Persistent Backends (Optional)

**SQLite File**:
- Persistent storage in a local SQLite file
- Data survives process restarts
- Good for development and small deployments

**Redis** (Optional):
- High-performance persistent storage
- Requires Redis server installation
- Best for production deployments

```bash
# Install Redis (optional)
brew install redis  # macOS
sudo apt-get install redis-server  # Ubuntu

# Start Redis
redis-server
```

## Development

### Project Structure

```text
agent_orchestrator/
├── core/                    # Core framework components
│   ├── config.py           # Configuration management
│   ├── llm_factory.py      # LLM provider factory
│   └── memory.py           # Memory management
├── supervisor/             # Supervisor agent
│   ├── api.py             # FastAPI server
│   ├── react_supervisor.py # ReAct implementation
│   ├── agent_registry.py   # Agent discovery
│   └── models.py          # Data models
├── agents/                 # Agent implementations
│   ├── base_agent.py      # Base agent class
│   ├── langgraph_agent.py # LangGraph integration
│   └── examples/          # Example agents
├── config.yaml            # Configuration file
├── requirements.txt       # Dependencies
└── README.md             # This file
```

### Adding New LLM Providers

1. Add provider configuration to `config.yaml`
2. Implement provider in `core/llm_factory.py`
3. Add API key environment variable

### Extending Agent Capabilities

1. Create new agent class inheriting from `BaseAgent`
2. Implement the `process_request` method
3. Define agent capabilities and system prompt
4. Register with supervisor via API

## Troubleshooting

### Common Issues

1. **Agent Registration Fails**
   - Ensure supervisor is running first (`python run_supervisor.py`)
   - Check if agent is running and accessible
   - Verify endpoint URL is correct
   - Check network connectivity
   - Run `python test_registration.py` to diagnose registration issues

2. **Multi-Agent Queries Not Working**
   - Check that multiple agents are registered and active
   - Verify the query contains multiple distinct tasks (e.g., "research X and code Y")
   - Review supervisor logs for reasoning steps
   - Run `python test_multi_agent_workflow.py` to test multi-agent functionality

3. **Memory Backend Errors**
   - For in-memory backends: No setup required
   - For Redis: Ensure Redis server is running
   - For SQLite: Check file permissions

4. **LLM Provider Errors**
   - Verify API keys are set correctly
   - Check API quotas and limits
   - Ensure model names are correct
   - Check rate limiting settings in config.yaml

5. **JSON Parsing Errors**
   - The system now uses deterministic reasoning to avoid JSON parsing issues
   - If you see JSON errors, check the supervisor logs for context-aware fallback messages
   - The system will automatically handle LLM response parsing failures

### Logs

Logs are stored in the `logs/` directory:
- `supervisor.log` - Supervisor agent logs
- `agents.log` - Agent logs

## License

This project is licensed under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs
3. Open an issue on GitHub
