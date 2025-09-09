# Quick Start Guide

## 🚀 Get Started in 5 Minutes

A comprehensive multi-agent framework with intelligent task decomposition and enhanced ReAct orchestration.

### 1. Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment file and add your API keys
cp env.example .env
# Edit .env and add your GROQ_API_KEY (or other LLM provider keys)
```

### 2. Start the System
```bash
# Option A: Start everything at once
python start_system.py

# Option B: Start components separately
# Terminal 1: Start Supervisor
python run_supervisor.py

# Terminal 2: Start Agents
python run_agents.py
```

### 3. Test the System
```bash
# Run the demo
python demo.py

# Test multi-agent functionality
python test_multi_agent_workflow.py

# Run comprehensive tests
python test_system.py
```

### 4. Use the API
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

## 🏗️ Architecture Overview

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

### 🧠 Multi-Agent Task Decomposition

The Supervisor automatically detects and decomposes complex queries:

- **Single-Part**: "Write a Python function" → Uses appropriate specialized agent
- **Multi-Part**: "Research Fibonacci origins and code it in Python" → Uses multiple agents:
  1. Research Agent (origins/history)
  2. Code Agent (Python implementation)  
  3. Synthesis (combines responses)

## 🎯 Key Features

- **Enhanced ReAct Supervisor**: Central orchestrator with deterministic reasoning and intelligent multi-agent task decomposition
- **Multi-Agent Task Decomposition**: Automatically breaks down complex queries into specialized agent tasks
- **Microservices**: Each agent runs as independent FastAPI service
- **Unified API**: Consistent REST endpoints for all communications
- **Dynamic Registration**: Agents register/unregister automatically
- **Memory Management**: In-memory conversation memory (no setup required)
- **Configurable LLMs**: Support for Groq, OpenAI, Google Gemini
- **YAML Configuration**: All settings in config.yaml
- **Robust Error Handling**: Intelligent fallback mechanisms and context-aware agent selection

## 📁 Project Structure

```text
agent_orchestrator/
├── core/                    # Core framework
├── supervisor/             # Supervisor agent
├── agents/                 # Agent implementations
│   ├── base_agent.py      # Base agent class
│   ├── langgraph_agent.py # LangGraph integration
│   └── examples/          # Example agents
├── config.yaml            # Configuration
├── requirements.txt       # Dependencies
└── README.md             # Full documentation
```

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
supervisor:
  llm_provider: "groq"  # or "openai", "google"
  model: "llama-3.1-8b-instant"
  max_iterations: 10
  memory_backend: "in_memory_sqlite"  # Default: no setup required

llm_providers:
  groq:
    api_key_env: "GROQ_API_KEY"
    default_model: "llama-3.1-8b-instant"
```

## 🤖 Available Agents

- **General Agent** (Port 8002): General-purpose assistance
- **Code Agent** (Port 8003): Programming and code analysis
- **Research Agent** (Port 8004): Information gathering and research
- **Creative Agent** (Port 8005): Creative writing and content generation

## 📚 API Endpoints

### Supervisor
- `GET /health` - Health check
- `GET /agents` - List registered agents
- `POST /execute` - Execute a task
- `GET /sessions/{id}/history` - Get conversation history

### Agents
- `GET /health` - Health check
- `GET /info` - Agent information
- `POST /execute` - Execute agent task

## 🧪 Example Usage

### Python Client
```python
import httpx
import asyncio

async def test_supervisor():
    async with httpx.AsyncClient() as client:
        # Single-agent query
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
        
        # Multi-agent query
        response = await client.post(
            "http://localhost:8000/execute",
            json={
                "session_id": "test-session",
                "query": "Research machine learning history and write a Python demo"
            }
        )
        result = response.json()
        print(f"Multi-agent response: {result['response']}")
        print(f"Agents used: {result['agents_used']}")  # Should show multiple agents

asyncio.run(test_supervisor())
```

### JavaScript Client
```javascript
// Single-agent query
const response = await fetch('http://localhost:8000/execute', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    session_id: 'test-session',
    query: 'Write a creative story about a robot'
  })
});

const result = await response.json();
console.log('Response:', result.response);
console.log('Agents used:', result.agents_used);

// Multi-agent query
const multiResponse = await fetch('http://localhost:8000/execute', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    session_id: 'test-session',
    query: 'Research AI ethics and create a Python script to demonstrate ethical AI principles'
  })
});

const multiResult = await multiResponse.json();
console.log('Multi-agent response:', multiResult.response);
console.log('Agents used:', multiResult.agents_used); // Should show multiple agents
```

## 🛠️ Creating Custom Agents

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

## 🔍 Troubleshooting

1. **Services won't start**: Check if ports are available
2. **Agent registration fails**: Ensure agents are running and accessible
3. **Multi-agent queries not working**:
   - Check that multiple agents are registered and active
   - Verify the query contains multiple distinct tasks (e.g., "research X and code Y")
   - Run `python test_multi_agent_workflow.py` to test functionality
4. **LLM errors**: Verify API keys in .env file
5. **Memory issues**: Check Redis connection or SQLite permissions
6. **JSON parsing errors**: The system now uses deterministic reasoning to avoid these issues

## 📖 Full Documentation

See `README.md` for complete documentation including:
- Detailed architecture explanation
- Advanced configuration options
- Custom agent development
- API reference
- Troubleshooting guide

## 🎉 You're Ready

The system is now running and ready to handle complex multi-agent tasks. The enhanced Supervisor will automatically:

- **Detect multi-part queries** and decompose them into specialized tasks
- **Route different aspects** to appropriate specialized agents
- **Coordinate responses** from multiple agents into coherent final answers
- **Handle errors gracefully** with intelligent fallback mechanisms

Try complex queries like:
- "Research the history of quantum computing and write a Python simulation"
- "Analyze the pros and cons of different programming languages and create a comparison chart"
- "Write a creative story about AI and research the latest developments in artificial intelligence"
