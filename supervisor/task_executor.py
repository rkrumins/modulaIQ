"""Task executor for handling agent calls and action execution."""

import time
from typing import Dict, Any, Optional
import logging

from supervisor.models import SupervisorState, AgentCall
from supervisor.agent_registry import AgentRegistry
from supervisor.llm_service import LLMService

logger = logging.getLogger(__name__)


class TaskExecutor:
    """Handles execution of actions determined by the ReAct engine."""
    
    def __init__(self, agent_registry: AgentRegistry, llm_service: LLMService):
        self.agent_registry = agent_registry
        self.llm_service = llm_service
    
    async def execute_action(
        self, 
        action: str, 
        action_params: Dict[str, Any], 
        state: SupervisorState
    ) -> str:
        """
        Execute an action determined by the ReAct engine.
        
        Args:
            action: The action to execute
            action_params: Parameters for the action
            state: Current supervisor state
            
        Returns:
            Observation from the action execution
        """
        logger.info(f"Executing action: {action}")
        
        if action == "call_agent":
            return await self._execute_agent_call(action_params, state)
        elif action == "synthesize_response":
            return await self._execute_synthesis(action_params, state)
        else:
            logger.warning(f"Unknown action: {action}")
            return f"Unknown action: {action}"
    
    async def _execute_agent_call(
        self, 
        action_params: Dict[str, Any], 
        state: SupervisorState
    ) -> str:
        """Execute an agent call."""
        agent_id = action_params.get("agent_id")
        input_data = action_params.get("input_data", {})
        
        if not agent_id:
            return "Error: No agent ID specified"
        
        try:
            logger.info(f"Calling agent: {agent_id}")
            start_time = time.time()
            
            # Call the agent
            result = await self.agent_registry.call_agent(
                agent_id, 
                input_data, 
                state.session_id
            )
            
            call_duration = time.time() - start_time
            logger.info(f"Agent {agent_id} call completed in {call_duration:.2f}s")
            
            # Record the agent call
            agent_call = AgentCall(
                agent_id=agent_id,
                input_data=input_data
            )
            state.agent_calls.append(agent_call)
            
            # Extract response
            response = result.get('response', str(result))
            return f"Agent {agent_id} responded: {response}"
            
        except Exception as e:
            logger.error(f"Error calling agent {agent_id}: {e}")
            return f"Error calling agent {agent_id}: {str(e)}"
    
    async def _execute_synthesis(
        self, 
        action_params: Dict[str, Any], 
        state: SupervisorState
    ) -> str:
        """Execute response synthesis."""
        partial_results = action_params.get("partial_results", [])
        
        # If no partial results provided, collect from reasoning steps
        if not partial_results:
            for step in state.reasoning_steps:
                if step.action == "call_agent":
                    partial_results.append(step.observation)
        
        logger.info(f"Synthesizing {len(partial_results)} partial results")
        
        # Use the ReAct engine's synthesis method
        from supervisor.react_engine import ReActEngine
        react_engine = ReActEngine(self.llm_service, self.agent_registry)
        
        try:
            synthesized_response = await react_engine.synthesize_responses(
                state, partial_results
            )
            return f"Synthesis completed: {synthesized_response}"
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return f"Synthesis failed: {str(e)}"
    
    async def execute_direct_response(
        self, 
        query: str, 
        session_id: str
    ) -> str:
        """
        Execute a direct LLM response for simple queries.
        
        Args:
            query: User query
            session_id: Session identifier
            
        Returns:
            Direct LLM response
        """
        system_prompt = """You are a helpful AI assistant. Provide clear, comprehensive, and well-formatted responses to user queries.

Guidelines:
- Answer directly and completely
- Use proper formatting (paragraphs, bullet points, etc.)
- Provide context when helpful
- Be concise but thorough
- Ensure your response is user-friendly"""

        user_prompt = f"""Please provide a direct, well-formatted answer to this query: "{query}"

Make sure your response is comprehensive and addresses the user's question completely."""

        try:
            return await self.llm_service.call_llm_with_system_prompt(
                system_prompt, user_prompt, "direct_response", timeout=20.0
            )
        except Exception as e:
            logger.error(f"Direct response failed: {e}")
            return f"I encountered an error while processing your request: {str(e)}"
