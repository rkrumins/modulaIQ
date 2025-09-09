"""ReAct-based Supervisor agent implementation."""

import json
import asyncio
import time
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from supervisor.models import (
    SupervisorState, ReasoningStep, TaskRequest, TaskResponse, AgentCall,
    ExecutionPlan, ExecutionNode, TaskType, TaskStatus
)
from supervisor.agent_registry import AgentRegistry
from core.llm_factory import LLMFactory
from core.memory import MemoryManager
from core.config import Config


logger = logging.getLogger(__name__)


class ReactSupervisor:
    """ReAct-based Supervisor agent that orchestrates tasks among specialized agents."""
    
    def __init__(self, config: Config, agent_registry: AgentRegistry, memory_manager: MemoryManager):
        self.config = config
        self.agent_registry = agent_registry
        self.memory_manager = memory_manager
        self.llm = LLMFactory.create_supervisor_llm(config)
        
        # Rate limiting for LLM calls
        self.last_llm_call_time = 0
        self.min_llm_call_interval = getattr(config.supervisor, 'rate_limiting', {}).get('min_llm_call_interval', 2.0)  # Increased to 2 seconds
        self.max_requests_per_minute = getattr(config.supervisor, 'rate_limiting', {}).get('max_requests_per_minute', 20)  # Reduced to 20
        
        # Response cache to avoid repeated LLM calls
        self.response_cache: Dict[str, str] = {}
        self.cache_max_size = 100
    
    def _get_cache_key(self, messages: List, call_type: str) -> str:
        """Generate a cache key for LLM calls."""
        content = "".join([msg.content for msg in messages if hasattr(msg, 'content')])
        return hashlib.md5(f"{call_type}:{content}".encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        """Get cached response if available."""
        return self.response_cache.get(cache_key)
    
    def _cache_response(self, cache_key: str, response: str):
        """Cache a response."""
        if len(self.response_cache) >= self.cache_max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.response_cache))
            del self.response_cache[oldest_key]
        self.response_cache[cache_key] = response
        
        # ReAct prompt template
        self.react_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=self._get_system_prompt()),
            HumanMessage(content="{query}")
        ])
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the supervisor."""
        available_agents = self.agent_registry.get_all_agents()
        agent_descriptions = []
        
        for agent in available_agents:
            if agent.status == "active":
                agent_descriptions.append(f"- {agent.agent_id}: {agent.description}")
        
        agents_info = "\n".join(agent_descriptions) if agent_descriptions else "No agents available."
        
        return f"""You are a Supervisor that orchestrates tasks among specialized agents.

Available Agents:
{agents_info}

Your job:
1. Analyze the user query
2. Decide which agent(s) to call
3. Call agents and get responses
4. Synthesize final answer

CRITICAL RULES:
- For queries with "AND" or multiple parts, use DIFFERENT agents for each part
- Examples: "Research X and code Y" → use research agent THEN code agent
- Only set "is_final": true after using multiple agents for multi-part queries

RESPONSE FORMAT (JSON only):
{{
    "reasoning": "Brief explanation",
    "action": "call_agent" or "synthesize_response",
    "action_params": {{"agent_id": "agent_name", "input_data": {{"query": "specific part"}}}},
    "is_final": false
}}

For multi-part queries, call different agents for different parts."""
    
    async def _call_llm_with_rate_limit(self, messages: List, call_type: str) -> Any:
        """Call LLM with rate limiting, caching, and debug logging."""
        # Check cache first
        cache_key = self._get_cache_key(messages, call_type)
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            logger.info(f"Using cached response for {call_type} LLM call")
            # Create a mock response object with the cached content
            class CachedResponse:
                def __init__(self, content):
                    self.content = content
            return CachedResponse(cached_response)
        
        current_time = time.time()
        time_since_last_call = current_time - self.last_llm_call_time
        
        # Rate limiting: ensure minimum interval between calls
        if time_since_last_call < self.min_llm_call_interval:
            sleep_time = self.min_llm_call_interval - time_since_last_call
            logger.info(f"Rate limiting: sleeping for {sleep_time:.2f}s before {call_type} LLM call")
            await asyncio.sleep(sleep_time)
        
        logger.info(f"Making {call_type} LLM call to {self.config.supervisor.llm_provider}")
        start_time = time.time()
        
        try:
            response = await self.llm.ainvoke(messages)
            call_duration = time.time() - start_time
            self.last_llm_call_time = time.time()
            
            # Cache the response
            self._cache_response(cache_key, response.content)
            
            logger.info(f"LLM {call_type} call completed in {call_duration:.2f}s")
            return response
            
        except Exception as e:
            call_duration = time.time() - start_time
            logger.error(f"LLM {call_type} call failed after {call_duration:.2f}s: {str(e)}")
            raise
    

    def _is_simple_query(self, query: str) -> bool:
        """Check if query is simple enough for direct response."""
        query_lower = query.lower()
        
        # Simple math questions
        if any(op in query_lower for op in ["+", "-", "*", "/", "plus", "minus", "times", "divided"]) and len(query.split()) <= 5:
            return True
        
        # Simple factual questions
        simple_patterns = [
            "what is", "who is", "when is", "where is", "how many", "how much"
        ]
        
        if any(pattern in query_lower for pattern in simple_patterns) and len(query.split()) <= 8:
            return True
        
        # Single word questions
        if len(query.split()) <= 3 and query.endswith("?"):
            return True
        
        return False

    def _is_multi_part_query(self, query: str) -> bool:
        """Check if query has multiple distinct parts requiring different agents."""
        query_lower = query.lower()
        
        # Multi-part connectors
        connectors = ["and", "also", "plus", "additionally", "furthermore", "moreover"]
        
        # Check for multiple distinct task types
        task_types = []
        
        # Programming tasks
        if any(keyword in query_lower for keyword in ["write", "function", "code", "program", "python", "javascript"]):
            task_types.append("programming")
        
        # Creative tasks
        if any(keyword in query_lower for keyword in ["poem", "story", "creative", "write a", "narrative"]):
            task_types.append("creative")
        
        # Research tasks
        if any(keyword in query_lower for keyword in ["research", "analyze", "study", "investigate", "origins", "history", "background", "facts", "information"]):
            task_types.append("research")
        
        # If we have multiple task types or connectors, it's likely multi-part
        return len(set(task_types)) > 1 or any(connector in query_lower for connector in connectors)

    async def _create_execution_plan(self, query: str, session_id: str) -> ExecutionPlan:
        """Create a comprehensive execution plan for the query."""
        logger.info(f"Creating execution plan for query: '{query[:100]}...'")
        
        # Get available agents
        available_agents = self.agent_registry.get_all_agents()
        agent_metadata = []
        for agent in available_agents:
            if agent.status == "active":
                capabilities_str = ", ".join(agent.capabilities)
                agent_metadata.append(
                    f"Agent ID: {agent.agent_id}\n"
                    f"Name: {agent.name}\n"
                    f"Description: {agent.description}\n"
                    f"Capabilities: {capabilities_str}\n"
                    f"---\n"
                )
        
        agents_info = "\n".join(agent_metadata)
        
        planning_prompt = f"""Create an execution plan for: "{query}"

Available agents: {agents_info}

Break this into 2-4 tasks max. Use parallel execution when possible.

JSON format:
{{
    "plan_id": "plan_{session_id}",
    "query": "{query}",
    "nodes": {{
        "node_1": {{
            "node_id": "node_1",
            "task_type": "research|code|creative|analysis|synthesis|direct_llm",
            "description": "Brief task description",
            "agent_id": "agent_id_or_null",
            "input_data": {{"query": "task-specific query"}},
            "dependencies": []
        }}
    }},
    "execution_order": [
        ["node_1", "node_2"],
        ["node_3"]
    ]
}}

Rules:
- Max 4 nodes total
- Use parallel execution for independent tasks
- Always end with synthesis node
- Keep descriptions brief

JSON only:"""

        try:
            messages = [HumanMessage(content=planning_prompt)]
            # Add timeout to prevent hanging
            response = await asyncio.wait_for(
                self._call_llm_with_rate_limit(messages, "planning"),
                timeout=30.0  # 30 second timeout
            )
            
            # Parse the JSON response
            plan_data = json.loads(response.content)
            
            # Create ExecutionPlan object with timestamp if not provided
            plan_id = plan_data.get("plan_id", f"plan_{session_id}_{int(time.time())}")
            plan = ExecutionPlan(
                plan_id=plan_id,
                query=plan_data["query"],
                execution_order=plan_data["execution_order"]
            )
            
            # Create ExecutionNode objects
            for node_id, node_data in plan_data["nodes"].items():
                node = ExecutionNode(
                    node_id=node_data["node_id"],
                    task_type=TaskType(node_data["task_type"]),
                    description=node_data["description"],
                    agent_id=node_data.get("agent_id"),
                    input_data=node_data.get("input_data", {}),
                    dependencies=node_data.get("dependencies", [])
                )
                plan.nodes[node_id] = node
            
            plan.total_nodes = len(plan.nodes)
            
            logger.info(f"Created execution plan with {plan.total_nodes} nodes and {len(plan.execution_order)} execution batches")
            return plan
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse planning response: {e}")
            # Fallback to simple plan
            return await self._create_fallback_plan(query, session_id)
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return await self._create_fallback_plan(query, session_id)

    async def _create_fallback_plan(self, query: str, session_id: str) -> ExecutionPlan:
        """Create a fast fallback plan when planning fails."""
        logger.info("Creating fast fallback execution plan")
        
        # Fast heuristic-based planning - only for truly multi-part queries
        query_lower = query.lower()
        
        # Quick detection of task types
        has_research = any(keyword in query_lower for keyword in ["research", "analyze", "study", "investigate", "origins", "history"])
        has_code = any(keyword in query_lower for keyword in ["write", "code", "function", "program", "python", "javascript"])
        has_creative = any(keyword in query_lower for keyword in ["creative", "story", "poem", "write a"])
        
        # Only create plan if we have multiple distinct task types
        task_count = sum([has_research, has_code, has_creative])
        if task_count < 2:
            # Not truly multi-part, let traditional ReAct handle it
            raise Exception("Not a true multi-part query, using traditional ReAct")
        
        nodes = {}
        execution_order = []
        node_counter = 1
        
        # Create nodes for detected task types
        if has_research:
            node_id = f"node_{node_counter}"
            nodes[node_id] = ExecutionNode(
                node_id=node_id,
                task_type=TaskType.RESEARCH,
                description="Research task",
                agent_id="research",
                input_data={"query": query},
                dependencies=[]
            )
            node_counter += 1
        
        if has_code:
            node_id = f"node_{node_counter}"
            nodes[node_id] = ExecutionNode(
                node_id=node_id,
                task_type=TaskType.CODE,
                description="Code task",
                agent_id="code",
                input_data={"query": query},
                dependencies=[]
            )
            node_counter += 1
        
        if has_creative:
            node_id = f"node_{node_counter}"
            nodes[node_id] = ExecutionNode(
                node_id=node_id,
                task_type=TaskType.CREATIVE,
                description="Creative task",
                agent_id="creative",
                input_data={"query": query},
                dependencies=[]
            )
            node_counter += 1
        
        # Add synthesis node
        synthesis_node_id = f"node_{node_counter}"
        nodes[synthesis_node_id] = ExecutionNode(
            node_id=synthesis_node_id,
            task_type=TaskType.SYNTHESIS,
            description="Synthesize results",
            agent_id=None,
            input_data={"query": query},
            dependencies=list(nodes.keys())[:-1]
        )
        
        # Create execution order - parallel for all task nodes, then synthesis
        task_nodes = list(nodes.keys())[:-1]
        execution_order.append(task_nodes)  # Parallel execution
        execution_order.append([synthesis_node_id])  # Synthesis last
        
        plan = ExecutionPlan(
            plan_id=f"fallback_plan_{session_id}_{int(time.time())}",
            query=query,
            nodes=nodes,
            execution_order=execution_order,
            total_nodes=len(nodes)
        )
        
        logger.info(f"Created fast fallback plan with {plan.total_nodes} nodes")
        return plan

    async def _execute_node(self, node: ExecutionNode, session_id: str) -> str:
        """Execute a single node in the execution plan."""
        logger.info(f"Executing node {node.node_id}: {node.description}")
        node.status = TaskStatus.RUNNING
        node.started_at = datetime.now()
        start_time = time.time()
        
        try:
            if node.task_type == TaskType.DIRECT_LLM:
                # Handle direct LLM tasks
                result = await self._execute_direct_llm_task(node, session_id)
            elif node.task_type == TaskType.SYNTHESIS:
                # Handle synthesis tasks - need to pass the plan
                # This will be handled in the _execute_plan method
                result = "Synthesis task - handled separately"
            else:
                # Handle agent tasks
                result = await self._execute_agent_task(node, session_id)
            
            node.result = result
            node.status = TaskStatus.COMPLETED
            node.completed_at = datetime.now()
            node.execution_time = time.time() - start_time
            
            logger.info(f"Node {node.node_id} completed in {node.execution_time:.2f}s")
            return result
            
        except Exception as e:
            node.error = str(e)
            node.status = TaskStatus.FAILED
            node.completed_at = datetime.now()
            node.execution_time = time.time() - start_time
            
            logger.error(f"Node {node.node_id} failed: {e}")
            return f"Error executing {node.description}: {str(e)}"

    async def _execute_direct_llm_task(self, node: ExecutionNode, session_id: str) -> str:
        """Execute a direct LLM task."""
        prompt = f"""
Please provide a comprehensive response to the following task: {node.description}

Query: {node.input_data.get('query', '')}

Provide a clear, well-formatted response that directly addresses the task requirements.
"""
        messages = [HumanMessage(content=prompt)]
        response = await self._call_llm_with_rate_limit(messages, "direct_task")
        return response.content

    async def _execute_agent_task(self, node: ExecutionNode, session_id: str) -> str:
        """Execute a task using a specialized agent."""
        if not node.agent_id:
            raise ValueError(f"No agent specified for node {node.node_id}")
        
        try:
            result = await self.agent_registry.call_agent(
                node.agent_id, 
                node.input_data, 
                session_id
            )
            return result.get('response', str(result))
        except Exception as e:
            raise Exception(f"Agent {node.agent_id} failed: {str(e)}")

    async def _execute_synthesis_task(self, node: ExecutionNode, plan: ExecutionPlan, session_id: str) -> str:
        """Execute a synthesis task to combine results from multiple nodes."""
        # Get results from dependency nodes
        dependency_results = []
        for dep_id in node.dependencies:
            dep_node = plan.nodes[dep_id]
            if dep_node.result:
                dependency_results.append(f"Result from {dep_id}: {dep_node.result}")
        
        if not dependency_results:
            return "No results to synthesize"
        
        synthesis_prompt = f"""
Based on the following results from different tasks, provide a comprehensive final answer to the user's query: "{node.input_data.get('query', '')}"

Task Results:
{chr(10).join(dependency_results)}

Please synthesize these results into a coherent, well-structured response that:
1. Addresses all aspects of the original query
2. Combines information from different sources seamlessly
3. Provides a comprehensive and useful answer
4. Maintains logical flow and context
"""
        messages = [HumanMessage(content=synthesis_prompt)]
        response = await self._call_llm_with_rate_limit(messages, "synthesis")
        return response.content

    async def _execute_plan(self, plan: ExecutionPlan, session_id: str) -> str:
        """Execute the entire execution plan with parallel processing."""
        logger.info(f"Executing plan {plan.plan_id} with {plan.total_nodes} nodes")
        plan.started_at = datetime.now()
        
        try:
            # Execute each batch in the execution order
            for batch_index, batch in enumerate(plan.execution_order):
                logger.info(f"Executing batch {batch_index + 1}/{len(plan.execution_order)}: {batch}")
                
                # Create tasks for parallel execution
                tasks = []
                for node_id in batch:
                    node = plan.nodes[node_id]
                    # Check if dependencies are satisfied
                    if all(plan.nodes[dep_id].status == TaskStatus.COMPLETED for dep_id in node.dependencies):
                        if node.task_type == TaskType.SYNTHESIS:
                            # Handle synthesis tasks specially
                            task = asyncio.create_task(self._execute_synthesis_task(node, plan, session_id))
                        else:
                            task = asyncio.create_task(self._execute_node(node, session_id))
                        tasks.append((node_id, task))
                    else:
                        logger.warning(f"Skipping node {node_id} - dependencies not satisfied")
                        node.status = TaskStatus.SKIPPED
                
                # Execute tasks in parallel
                if tasks:
                    results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
                    
                    # Process results
                    for (node_id, _), result in zip(tasks, results):
                        node = plan.nodes[node_id]
                        if isinstance(result, Exception):
                            node.error = str(result)
                            node.status = TaskStatus.FAILED
                            plan.failed_nodes += 1
                        else:
                            plan.completed_nodes += 1
                
                # Check if we should continue
                if plan.failed_nodes > 0 and plan.failed_nodes >= plan.total_nodes // 2:
                    logger.warning("Too many failed nodes, stopping execution")
                    break
            
            # Get final result from synthesis node or last completed node
            final_result = await self._get_final_result(plan)
            
            plan.completed_at = datetime.now()
            logger.info(f"Plan execution completed. Success: {plan.completed_nodes}/{plan.total_nodes}")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Plan execution failed: {e}")
            return f"Execution failed: {str(e)}"

    async def _get_final_result(self, plan: ExecutionPlan) -> str:
        """Get the final result from the execution plan."""
        # Look for synthesis node first
        synthesis_nodes = [node for node in plan.nodes.values() if node.task_type == TaskType.SYNTHESIS]
        if synthesis_nodes and synthesis_nodes[0].status == TaskStatus.COMPLETED:
            return synthesis_nodes[0].result
        
        # Otherwise, combine results from all completed nodes
        completed_results = []
        for node in plan.nodes.values():
            if node.status == TaskStatus.COMPLETED and node.result:
                completed_results.append(f"{node.description}: {node.result}")
        
        if completed_results:
            if len(completed_results) == 1:
                return completed_results[0]
            else:
                # Combine multiple results
                combined_prompt = f"""
Based on the following results, provide a comprehensive final answer:

Results:
{chr(10).join(completed_results)}

Please combine these into a coherent response.
"""
                messages = [HumanMessage(content=combined_prompt)]
                response = await self._call_llm_with_rate_limit(messages, "final_synthesis")
                return response.content
        
        return "No results available from execution plan"

    async def _handle_simple_query(self, request: TaskRequest, start_time: float) -> TaskResponse:
        """Handle simple queries with direct LLM response."""
        session_id = request.session_id
        query = request.query
        
        # Use direct LLM response for simple queries to avoid agent dependency
        try:
            direct_prompt = f"""
Please provide a direct, well-formatted answer to the user's query: "{query}"

This is a simple question that doesn't require specialized agents. Please provide a clear, helpful, and properly formatted response that:
1. Directly answers the question
2. Is easy to read and understand
3. Provides context when helpful
4. Uses proper formatting (paragraphs, bullet points, etc. as appropriate)

Make sure your response is comprehensive and user-friendly, not just a single word or number.
"""
            messages = [HumanMessage(content=direct_prompt)]
            response = await self._call_llm_with_rate_limit(messages, "simple_query")
            response_text = response.content
            
            total_duration = time.time() - start_time
            logger.info(f"Simple query completed for session {session_id} in {total_duration:.2f}s")
            
            return TaskResponse(
                session_id=session_id,
                response=response_text,
                reasoning=["Simple query detected, using direct LLM response"],
                actions_taken=[{
                    "action": "direct_response",
                    "reasoning": "Simple query detected, using direct LLM response",
                    "observation": f"Direct LLM response: {response_text[:100]}...",
                    "step": 1
                }],
                agents_used=[],
                iterations=1,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error handling simple query: {e}")
            return TaskResponse(
                session_id=session_id,
                response=f"I encountered an error while processing your request: {str(e)}",
                reasoning=["Simple query detected but failed"],
                actions_taken=[],
                agents_used=[],
                iterations=1,
                success=False,
                error=str(e)
            )

    async def _select_agent_with_llm(self, query: str) -> str:
        """Use LLM to select the most appropriate agent based on query analysis and agent metadata."""
        
        available_agents = self.agent_registry.get_all_agents()
        
        # Build comprehensive agent metadata for LLM analysis
        agent_metadata = []
        for agent in available_agents:
            if agent.status == "active":
                capabilities_str = ", ".join(agent.capabilities)
                agent_metadata.append(
                    f"=== AGENT: {agent.agent_id.upper()} ===\n"
                    f"Name: {agent.name}\n"
                    f"Description: {agent.description}\n"
                    f"Capabilities: {capabilities_str}\n"
                    f"Status: {agent.status}\n"
                    f"Endpoint: {agent.endpoint}\n"
                    f"Last Health Check: {agent.last_health_check}\n"
                    f"Registration Time: {agent.registered_at}\n"
                    f"---\n"
                )
        
        agents_info = "\n".join(agent_metadata)
        
        selection_prompt = f"""You are an expert AI agent selector with deep understanding of user intent, context, and agent capabilities. Your task is to analyze the user's query and select the most appropriate agent based on comprehensive analysis.

AVAILABLE AGENTS AND THEIR METADATA:
{agents_info}

USER QUERY: "{query}"

ANALYSIS FRAMEWORK:
Please analyze this query using the following dimensions:

1. **PRIMARY INTENT ANALYSIS**:
   - What is the user's main goal or objective?
   - What type of outcome are they seeking?
   - What domain or field does this query belong to?

2. **QUERY COMPLEXITY & SCOPE**:
   - Is this a simple, single-purpose query or a complex, multi-faceted request?
   - Does it require specialized knowledge or can it be handled generally?
   - Are there multiple distinct components that need different expertise?

3. **AGENT CAPABILITY MATCHING**:
   - Which agent's description and capabilities best align with the user's needs?
   - Consider the agent's expertise area, not just keywords
   - Think about the quality and depth of response each agent can provide

4. **CONTEXT & SENTIMENT ANALYSIS**:
   - What is the tone and context of the query?
   - Is the user looking for factual information, creative content, technical solutions, or general guidance?
   - What level of expertise or specialization is implied?

5. **MULTI-PART QUERY HANDLING**:
   - If the query has multiple components, which agent should handle the FIRST/PRIMARY component?
   - Consider the logical flow and dependencies between parts
   - Prioritize the most critical or foundational aspect

SELECTION CRITERIA:
- Choose the agent whose expertise and capabilities most closely match the user's primary intent
- Consider the agent's description and capabilities holistically, not just keyword matching
- For multi-part queries, select the agent best suited for the most important or foundational part
- Prioritize specialized agents over general ones when the query clearly requires specific expertise
- Consider the user's implied level of technical knowledge and expected response depth

RESPONSE FORMAT:
Respond with ONLY the agent ID (e.g., "code", "research", "creative", or "general") - no other text.

Remember: Your selection should be based on intelligent analysis of the query's intent, context, and the agent's capabilities, not simple keyword matching."""
        
        try:
            messages = [HumanMessage(content=selection_prompt)]
            response = await self._call_llm_with_rate_limit(messages, "agent_selection")
            
            # Extract agent ID from response
            selected_agent = response.content.strip().lower()
            
            # Validate the selected agent exists
            valid_agents = [agent.agent_id for agent in available_agents if agent.status == "active"]
            if selected_agent in valid_agents:
                logger.info(f"LLM selected agent: {selected_agent} for query: '{query[:50]}...'")
                return selected_agent
            else:
                logger.warning(f"LLM selected invalid agent '{selected_agent}', falling back to 'general'")
                return "general"
                
        except Exception as e:
            logger.error(f"LLM agent selection failed: {e}, falling back to 'general'")
            return "general"
    
    async def execute_task(self, request: TaskRequest) -> TaskResponse:
        """Execute a task using the enhanced planning and execution system."""
        session_id = request.session_id
        query = request.query
        max_iterations = request.max_iterations or self.config.supervisor.max_iterations
        
        logger.info(f"Starting task execution for session {session_id}: '{query[:100]}...'")
        start_time = time.time()
        
        # Check for simple queries that can be handled directly
        if self._is_simple_query(query):
            logger.info(f"Detected simple query, using direct response")
            return await self._handle_simple_query(request, start_time)
        
        # Initialize supervisor state
        # Check if planning is enabled (can be disabled for maximum speed)
        use_planning = getattr(self.config.supervisor, 'use_planning', True)
        state = SupervisorState(
            session_id=session_id,
            query=query,
            max_iterations=max_iterations,
            context=request.context or {},
            use_planning=use_planning
        )
        
        # Get conversation context
        conversation_context = self.memory_manager.get_conversation_context(session_id)
        
        try:
            # Use the new planning system only for truly complex multi-part queries
            if state.use_planning and self._is_multi_part_query(query) and len(query.split()) > 15:
                logger.info("Using enhanced planning system for complex multi-part query")
                
                # Try LLM planning first, but fallback quickly if it fails
                try:
                    execution_plan = await self._create_execution_plan(query, session_id)
                    state.execution_plan = execution_plan
                    
                    # Execute the plan
                    final_response = await self._execute_plan(execution_plan, session_id)
                    state.final_response = final_response
                    state.is_complete = True
                except Exception as e:
                    logger.warning(f"Planning system failed, falling back to traditional ReAct: {e}")
                    # Fall through to traditional ReAct system
                    state.use_planning = False
                else:
                    # Record planning steps only if planning succeeded
                    planning_step = ReasoningStep(
                        step=1,
                        reasoning=f"Created execution plan with {execution_plan.total_nodes} nodes",
                        action="planning",
                        observation=f"Plan executed successfully: {execution_plan.completed_nodes}/{execution_plan.total_nodes} nodes completed"
                    )
                    state.reasoning_steps.append(planning_step)
                
            else:
                # Use the old ReAct system for simpler queries
                logger.info("Using traditional ReAct system for simple query")
                
            # ReAct loop with early termination optimization
            while state.current_iteration < state.max_iterations and not state.is_complete:
                state.current_iteration += 1
                logger.info(f"ReAct iteration {state.current_iteration}/{state.max_iterations} for session {session_id}")
                
                # Get reasoning and action from LLM
                reasoning_result = await self._reason_and_act(state, conversation_context)
                
                if reasoning_result.get("is_final", False):
                    final_response = reasoning_result.get("final_response", "")
                    if final_response:
                        state.final_response = final_response
                        state.is_complete = True
                        break
                    else:
                            logger.warning("Reasoning marked as final but provided no final_response, continuing...")
                
                    # Early termination check for synthesis
                if (len(state.agent_calls) >= 1 and 
                    reasoning_result.get("action") == "synthesize_response"):
                        logger.info(f"Early termination: Agent response available, proceeding with synthesis")
                        state.is_complete = True
                        break
                
                # Execute the action
                observation = await self._execute_action(reasoning_result, state)
                
                # Record the reasoning step
                reasoning_step = ReasoningStep(
                    step=state.current_iteration,
                    reasoning=reasoning_result.get("reasoning", ""),
                    action=reasoning_result.get("action", ""),
                    observation=observation
                )
                state.reasoning_steps.append(reasoning_step)
                
                # Check if we should continue or terminate
                if self._is_multi_part_query(state.query):
                    unique_agents = len(set([call.agent_id for call in state.agent_calls]))
                    if unique_agents >= 2:
                        logger.info(f"Multi-part query: Used {unique_agents} different agents, should synthesize")
                        # The next iteration will handle synthesis
                else:
                    if len(state.agent_calls) >= 1:
                        logger.info(f"Single-part query: Used 1 agent, should synthesize")
                        # The next iteration will handle synthesis
            
            # Generate final response if not already complete
            if not state.is_complete:
                state.final_response = await self._generate_final_response(state)
                state.is_complete = True
            
            # Ensure we have a final response
            if not state.final_response:
                logger.warning("No final response generated, using agent response directly")
                if state.agent_calls:
                    # Use the last agent response as final response
                    last_agent_call = state.agent_calls[-1]
                    # Get the response from the reasoning steps
                    last_observation = ""
                    for step in reversed(state.reasoning_steps):
                        if step.action == "call_agent" and last_agent_call.agent_id in step.observation:
                            last_observation = step.observation
                            break
                    state.final_response = last_observation if last_observation else f"Agent {last_agent_call.agent_id} provided a response."
                else:
                    state.final_response = "I was unable to process your request."
            
            # Store the conversation
            self.memory_manager.store_message(
                session_id, "user", query, {"task_request": request.dict()}
            )
            
            # Create a simplified response metadata that's JSON serializable
            response_metadata = {
                "task_response": {
                    "session_id": state.session_id,
                    "query": state.query,
                    "iterations": state.current_iteration,
                    "agents_used": list(set([call.agent_id for call in state.agent_calls])),
                    "success": True,
                    "final_response_length": len(state.final_response) if state.final_response else 0,
                    "execution_plan": {
                        "plan_id": state.execution_plan.plan_id if state.execution_plan else None,
                        "total_nodes": state.execution_plan.total_nodes if state.execution_plan else 0,
                        "completed_nodes": state.execution_plan.completed_nodes if state.execution_plan else 0,
                        "failed_nodes": state.execution_plan.failed_nodes if state.execution_plan else 0
                    } if state.execution_plan else None
                }
            }
            
            self.memory_manager.store_message(
                session_id, "assistant", state.final_response, response_metadata
            )
            
            total_duration = time.time() - start_time
            logger.info(f"Task execution completed for session {session_id} in {total_duration:.2f}s with {state.current_iteration} iterations")
            
            # Collect agents used from both traditional calls and execution plan
            agents_used = list(set([call.agent_id for call in state.agent_calls]))
            if state.execution_plan:
                plan_agents = [node.agent_id for node in state.execution_plan.nodes.values() if node.agent_id]
                agents_used.extend(plan_agents)
                agents_used = list(set(agents_used))
            
            return TaskResponse(
                session_id=session_id,
                response=state.final_response,
                reasoning=[step.reasoning for step in state.reasoning_steps],
                actions_taken=[
                    {
                        "action": step.action,
                        "reasoning": step.reasoning,
                        "observation": step.observation,
                        "step": step.step
                    }
                    for step in state.reasoning_steps
                ],
                agents_used=agents_used,
                iterations=state.current_iteration,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error executing task: {e}")
            return TaskResponse(
                session_id=session_id,
                response=f"I encountered an error while processing your request: {str(e)}",
                reasoning=[step.reasoning for step in state.reasoning_steps],
                actions_taken=[
                    {
                        "action": step.action,
                        "reasoning": step.reasoning,
                        "observation": step.observation,
                        "step": step.step
                    }
                    for step in state.reasoning_steps
                ],
                agents_used=list(set([call.agent_id for call in state.agent_calls])),
                iterations=state.current_iteration,
                success=False,
                error=str(e)
            )
    
    async def _reason_and_act(self, state: SupervisorState, conversation_context: List[Dict]) -> Dict[str, Any]:
        """Get reasoning and action using a more deterministic approach."""
        agents_used = [call.agent_id for call in state.agent_calls]
        is_multi_part = self._is_multi_part_query(state.query)
        
        logger.info(f"Reasoning step - Agents used: {agents_used}, Is multi-part: {is_multi_part}")
        
        # For multi-part queries, use deterministic logic
        if is_multi_part:
            if len(agents_used) == 0:
                # First agent call - select based on query content
                if any(keyword in state.query.lower() for keyword in ["research", "origins", "history", "analyze", "investigate"]):
                    selected_agent = "research"
                    reasoning = "This is a multi-part query. I'll start with the research aspect."
                elif any(keyword in state.query.lower() for keyword in ["code", "python", "function", "program", "write"]):
                    selected_agent = "code"
                    reasoning = "This is a multi-part query. I'll start with the coding aspect."
                else:
                    selected_agent = "research"  # Default to research first
                    reasoning = "This is a multi-part query. I'll start with research."
                
                return {
                    "reasoning": reasoning,
                    "action": "call_agent",
                    "action_params": {"agent_id": selected_agent, "input_data": {"query": state.query}},
                    "is_final": False
                }
            
            elif len(agents_used) == 1:
                # Second agent call - select a different agent
                available_agents = ["code", "research", "creative", "general"]
                remaining_agents = [agent for agent in available_agents if agent not in agents_used]
                
                if remaining_agents:
                    # Prioritize based on query content
                    if "code" in remaining_agents and any(keyword in state.query.lower() for keyword in ["code", "python", "function", "program"]):
                        selected_agent = "code"
                        reasoning = f"I've used {agents_used[0]} agent. Now I need to use the code agent for the programming aspect."
                    elif "research" in remaining_agents and any(keyword in state.query.lower() for keyword in ["research", "origins", "history", "analyze"]):
                        selected_agent = "research"
                        reasoning = f"I've used {agents_used[0]} agent. Now I need to use the research agent for the research aspect."
                    else:
                        selected_agent = remaining_agents[0]
                        reasoning = f"I've used {agents_used[0]} agent. Now I'll use the {selected_agent} agent for the remaining aspect."
                    
                    return {
                        "reasoning": reasoning,
                        "action": "call_agent",
                        "action_params": {"agent_id": selected_agent, "input_data": {"query": state.query}},
                        "is_final": False
                    }
                else:
                    # All agents used, synthesize
                    return {
                        "reasoning": "I have used multiple agents for this multi-part query. Now I'll synthesize the responses.",
                        "action": "synthesize_response",
                        "action_params": {"partial_results": [step.observation for step in state.reasoning_steps if step.action == "call_agent"]},
                        "is_final": True
                    }
            
            else:
                # Multiple agents used, synthesize
                return {
                    "reasoning": "I have used multiple agents for this multi-part query. Now I'll synthesize the responses.",
                    "action": "synthesize_response",
                    "action_params": {"partial_results": [step.observation for step in state.reasoning_steps if step.action == "call_agent"]},
                    "is_final": True
                }
        
        else:
            # Single-part query - use LLM-based selection
            if len(agents_used) == 0:
                selected_agent = await self._select_agent_with_llm(state.query)
                return {
                    "reasoning": f"This is a single-part query. I'll use the {selected_agent} agent.",
                    "action": "call_agent",
                    "action_params": {"agent_id": selected_agent, "input_data": {"query": state.query}},
                    "is_final": False
                }
            else:
                # Single agent used, synthesize
                return {
                    "reasoning": "I have used an agent for this single-part query. Now I'll provide the final response.",
                    "action": "synthesize_response",
                    "action_params": {"partial_results": [step.observation for step in state.reasoning_steps if step.action == "call_agent"]},
                    "is_final": True
                }
    
    async def _execute_action(self, reasoning_result: Dict[str, Any], state: SupervisorState) -> str:
        """Execute the action determined by the LLM."""
        action = reasoning_result.get("action", "")
        action_params = reasoning_result.get("action_params", {})
        
        if action == "call_agent":
            agent_id = action_params.get("agent_id")
            input_data = action_params.get("input_data", {})
            
            if not agent_id:
                return "Error: No agent ID specified"
            
            try:
                # Call the agent with proper session_id
                result = await self.agent_registry.call_agent(agent_id, input_data, state.session_id)
                
                # Record the agent call
                agent_call = AgentCall(
                    agent_id=agent_id,
                    input_data=input_data
                )
                state.agent_calls.append(agent_call)
                
                return f"Agent {agent_id} responded: {result.get('response', str(result))}"
                
            except Exception as e:
                return f"Error calling agent {agent_id}: {str(e)}"
        
        elif action == "synthesize_response":
            partial_results = action_params.get("partial_results", [])
            # This would be handled in the final response generation
            return f"Collected {len(partial_results)} partial results for synthesis"
        
        else:
            return f"Unknown action: {action}"
    
    async def _generate_final_response(self, state: SupervisorState) -> str:
        """Generate the final response based on all collected information."""
        logger.info(f"Generating final response for query: '{state.query}'")
        logger.info(f"Agent calls: {len(state.agent_calls)}, Reasoning steps: {len(state.reasoning_steps)}")
        
        if state.agent_calls:
            # Collect all agent responses from reasoning steps
            agent_responses = []
            for step in state.reasoning_steps:
                if step.action == "call_agent":
                    # Extract the agent response from the observation
                    agent_response = step.observation
                    agent_responses.append(agent_response)
                    logger.info(f"Found agent response: {agent_response}")
            
            if agent_responses:
                # For single agent responses, return directly without synthesis
                if len(agent_responses) == 1 and not self._is_multi_part_query(state.query):
                    logger.info("Single agent response detected, returning directly without synthesis")
                    return agent_responses[0]
                
                # Use LLM to synthesize the final response for multiple agents or multi-part queries
                synthesis_prompt = f"""
Based on the following agent responses and reasoning steps, provide a comprehensive final answer to the user's query: "{state.query}"

Agent Responses:
{chr(10).join(agent_responses)}

Reasoning Steps:
{chr(10).join([f"Step {step.step}: {step.reasoning}" for step in state.reasoning_steps])}

Please provide a clear, comprehensive response that:
1. Addresses all aspects of the user's query
2. Synthesizes information from multiple agents when applicable
3. Provides a coherent, well-structured answer
4. Maintains the context and flow between different parts of the response
"""
                
                messages = [HumanMessage(content=synthesis_prompt)]
                response = await self._call_llm_with_rate_limit(messages, "synthesis")
                return response.content
            else:
                logger.warning("No agent responses found in reasoning steps")
                return "I was unable to process your request with the available agents."
        
        else:
            logger.warning("No agent calls were made")
            # If no agents were called, try to answer directly with LLM
            direct_prompt = f"""
Please provide a direct, well-formatted answer to the user's query: "{state.query}"

This is a simple question that doesn't require specialized agents. Please provide a clear, helpful, and properly formatted response that:
1. Directly answers the question
2. Is easy to read and understand
3. Provides context when helpful
4. Uses proper formatting (paragraphs, bullet points, etc. as appropriate)

Make sure your response is comprehensive and user-friendly, not just a single word or number.
"""
            messages = [HumanMessage(content=direct_prompt)]
            response = await self._call_llm_with_rate_limit(messages, "direct_response")
            return response.content
