# Refactored Supervisor System

This document describes the refactored supervisor system that follows proper separation of concerns and implements a clean ReAct framework with LLM-driven decision making.

## Architecture Overview

The supervisor system has been refactored from a monolithic design into a modular architecture with the following components:

### Core Components

1. **ReactSupervisor** - Main orchestrator that coordinates all components
2. **LLMService** - Handles all LLM interactions with caching and rate limiting
3. **QueryAnalyzer** - Analyzes queries using LLM to determine approach
4. **ReActEngine** - Implements the ReAct reasoning loop with LLM-driven decisions
5. **TaskExecutor** - Executes actions determined by the ReAct engine
6. **ErrorHandler** - Centralized error handling with retry logic and circuit breakers

## Component Details

### ReactSupervisor

The main supervisor class that orchestrates the entire process:

```python
class ReactSupervisor:
    def __init__(self, config: Config, agent_registry: AgentRegistry, memory_manager: MemoryManager):
        # Initialize all components
        self.llm_service = LLMService(llm, config)
        self.query_analyzer = QueryAnalyzer(self.llm_service)
        self.react_engine = ReActEngine(self.llm_service, agent_registry)
        self.task_executor = TaskExecutor(agent_registry, self.llm_service)
```

**Key Features:**

- Pure ReAct framework implementation
- LLM-driven decision making (no keyword matching)
- Proper separation of concerns
- Comprehensive error handling

### LLMService

Centralized service for all LLM interactions:

```python
class LLMService:
    async def call_llm(self, messages: List[BaseMessage], call_type: str = "general", timeout: float = 30.0) -> str:
        # Handles caching, rate limiting, and error handling
```

**Features:**

- Response caching to avoid repeated calls
- Rate limiting with configurable intervals
- Retry logic with exponential backoff
- Comprehensive error handling
- Timeout protection

### QueryAnalyzer

LLM-based query analysis to determine the best approach:

```python
class QueryAnalyzer:
    async def analyze_query(self, query: str) -> Dict[str, Any]:
        # Returns: complexity, intent, scope, domain, approach, reasoning, confidence
```

**Analysis Dimensions:**

- **Complexity**: Simple vs Complex
- **Intent**: What the user is trying to achieve
- **Scope**: Single-part vs Multi-part
- **Domain**: Technical, creative, research, general, mixed
- **Approach**: Direct response vs Agent orchestration

### ReActEngine

Implements the ReAct (Reasoning and Acting) framework:

```python
class ReActEngine:
    async def reason_and_act(self, state: SupervisorState, conversation_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Returns: reasoning, action, action_params, is_final
```

**ReAct Process:**

1. **Reason**: Analyze current situation and decide next action
2. **Act**: Execute the chosen action (call_agent or synthesize_response)
3. **Observe**: Record the result for next iteration

### TaskExecutor

Executes actions determined by the ReAct engine:

```python
class TaskExecutor:
    async def execute_action(self, action: str, action_params: Dict[str, Any], state: SupervisorState) -> str:
        # Handles agent calls and synthesis
```

**Supported Actions:**

- `call_agent`: Call a specialized agent
- `synthesize_response`: Combine multiple agent responses

### ErrorHandler

Centralized error handling with advanced features:

```python
class ErrorHandler:
    def handle_error(self, error: Exception, context: str = "", operation: str = "", should_retry: bool = True) -> Dict[str, Any]:
        # Returns comprehensive error information
```

**Features:**

- Error tracking and statistics
- Retry logic with exponential backoff
- Circuit breaker pattern
- Fallback mechanisms
- Comprehensive logging

## Key Improvements

### 1. Eliminated Monolithic Design

**Before:** Single class with 1000+ lines handling everything
**After:** Modular components with single responsibilities

### 2. LLM-Driven Decisions

**Before:** Keyword matching and hardcoded logic
**After:** All decisions made by LLM analysis

### 3. Proper ReAct Implementation

**Before:** Mixed ReAct with separate planning system
**After:** Pure ReAct framework with proper reasoning loop

### 4. Enhanced Error Handling

**Before:** Basic try-catch blocks
**After:** Comprehensive error handling with retry logic and circuit breakers

### 5. Improved Caching and Rate Limiting

**Before:** Basic caching and rate limiting
**After:** Advanced caching with LRU eviction and configurable rate limiting

### 6. Better Type Safety

**Before:** Minimal type hints
**After:** Comprehensive type hints throughout

## Usage Example

```python
# Initialize the supervisor
supervisor = ReactSupervisor(config, agent_registry, memory_manager)

# Execute a task
request = TaskRequest(
    session_id="session_123",
    query="Research the history of AI and write a Python function to calculate Fibonacci numbers",
    context={"user_preferences": "detailed"}
)

response = await supervisor.execute_task(request)
print(response.response)
```

## Configuration

The system can be configured through the config file:

```yaml
supervisor:
  max_iterations: 10
  use_planning: false  # Use pure ReAct
  rate_limiting:
    min_llm_call_interval: 1.0
    max_requests_per_minute: 30
```

## Error Handling

The system includes comprehensive error handling:

- **Retry Logic**: Automatic retries with exponential backoff
- **Circuit Breakers**: Prevent cascading failures
- **Fallback Mechanisms**: Graceful degradation when components fail
- **Error Tracking**: Detailed error statistics and logging

## Performance Optimizations

- **Response Caching**: Avoid repeated LLM calls for similar queries
- **Rate Limiting**: Prevent API rate limit violations
- **Parallel Execution**: Execute independent tasks concurrently
- **Early Termination**: Stop when sufficient information is gathered

## Testing

The refactored system is designed for easy testing:

- Each component can be tested independently
- Mock interfaces for external dependencies
- Comprehensive error scenarios covered
- Performance benchmarks included

## Migration Guide

To migrate from the old system:

1. Update imports to use the new components
2. Remove any direct keyword matching logic
3. Update configuration to use new parameters
4. Test with existing queries to ensure compatibility

## Future Enhancements

- **Streaming Responses**: Support for streaming LLM responses
- **Advanced Caching**: Semantic caching based on query similarity
- **Metrics Collection**: Detailed performance and usage metrics
- **A/B Testing**: Framework for testing different approaches
- **Multi-Modal Support**: Support for images and other media types
