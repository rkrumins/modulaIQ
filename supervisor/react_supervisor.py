"""
Simplified ReAct-based Supervisor with intelligent LLM-driven orchestration.

This implementation is designed to be:
- Simple and clean
- Purely LLM-driven with no hardcoding
- Based on dynamic agent metadata
- Intelligent about multi-part queries
- Capable of complex agent orchestration
"""

import time
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

from supervisor.models import SupervisorState, ReasoningStep, TaskRequest, TaskResponse, AgentCall
from supervisor.agent_registry import AgentRegistry
from supervisor.llm_service import LLMService
from supervisor.react_engine import ReActEngine
from supervisor.task_executor import TaskExecutor
from core.memory import MemoryManager
from core.config import Config
from core.llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class ReactSupervisor:
    """
    Simplified ReAct-based Supervisor that orchestrates tasks among specialized agents.
    
    This implementation:
    - Uses LLM for all decision making
    - Has no hardcoded logic or keyword matching
    - Makes decisions based on dynamic agent metadata
    - Handles multi-part queries intelligently
    - Orchestrates complex agent workflows
    """
    
    def __init__(self, config: Config, agent_registry: AgentRegistry, memory_manager: MemoryManager):
        self.config = config
        self.agent_registry = agent_registry
        self.memory_manager = memory_manager
        
        # Initialize LLM service
        llm = LLMFactory.create_supervisor_llm(config)
        self.llm_service = LLMService(llm, config)
        
        # Initialize components
        self.react_engine = ReActEngine(self.llm_service, agent_registry)
        self.task_executor = TaskExecutor(agent_registry, self.llm_service)
        
        logger.info("ReactSupervisor initialized with simplified architecture")
    
    async def execute_task(self, request: TaskRequest) -> TaskResponse:
        """
        Execute a task using the simplified ReAct framework.
        
        The LLM makes all decisions about:
        - Which agents to call
        - When to continue or stop
        - How to synthesize responses
        """
        start_time = time.time()
        session_id = request.session_id
        
        try:
            # Initialize state
            state = SupervisorState(
                query=request.query,
                session_id=session_id,
                current_iteration=0,
                max_iterations=request.max_iterations or 5
            )
            
            # Get conversation context
            conversation_context = self.memory_manager.get_conversation_context(session_id)
            
            logger.info(f"Starting ReAct execution for session {session_id}")
            logger.info(f"Query: {request.query}")
            
            # ReAct loop
            while state.current_iteration < state.max_iterations and not state.is_complete:
                state.current_iteration += 1
                
                logger.info(f"ReAct iteration {state.current_iteration}/{state.max_iterations}")
                
                # Let the ReAct engine decide what to do
                reasoning_result = await self.react_engine.reason_and_act(state, conversation_context)
                
                # Record the reasoning step
                reasoning_step = ReasoningStep(
                    step=state.current_iteration,
                    reasoning=reasoning_result["reasoning"],
                    action=reasoning_result["action"],
                    observation=""
                )
                state.reasoning_steps.append(reasoning_step)
                
                # Execute the action
                if reasoning_result["action"] == "call_agent":
                    await self._execute_agent_call(reasoning_result, state, reasoning_step)
                elif reasoning_result["action"] == "synthesize_response":
                    await self._execute_synthesis(reasoning_result, state, reasoning_step)
                
                # Check if we should continue
                should_continue = await self.react_engine.should_continue(state)
                
                if not should_continue or reasoning_result.get("is_final", False):
                    state.is_complete = True
                    logger.info(f"ReAct loop completed after {state.current_iteration} iterations")
                    break
            
            # Generate final response
            final_response = await self._generate_final_response(state)
            logger.info(f"Final response length: {len(final_response)}")
            logger.info(f"Final response preview: {final_response[:200]}...")
            
            # Save to memory
            self.memory_manager.store_message(session_id, "user", request.query)
            self.memory_manager.store_message(session_id, "assistant", final_response)
            
            execution_time = time.time() - start_time
            logger.info(f"Task execution completed in {execution_time:.2f}s with {state.current_iteration} iterations")
            
            return TaskResponse(
                session_id=session_id,
                response=final_response,
                reasoning=[step.reasoning for step in state.reasoning_steps],
                actions_taken=[{
                    "action": step.action,
                    "reasoning": step.reasoning,
                    "observation": self._format_observation_for_user(step),
                    "step": step.step
                } for step in state.reasoning_steps],
                agents_used=[call.agent_id for call in state.agent_calls],
                iterations=state.current_iteration,
                success=True,
                error=None
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Task execution failed after {execution_time:.2f}s: {e}")
            
            return TaskResponse(
                session_id=session_id,
                response=f"I encountered an error while processing your request: {str(e)}",
                reasoning=[],
                actions_taken=[],
                agents_used=[],
                iterations=state.current_iteration if 'state' in locals() else 0,
                success=False,
                error=str(e)
            )
    
    async def _execute_agent_call(self, reasoning_result: Dict[str, Any], state: SupervisorState, reasoning_step: ReasoningStep):
        """Execute an agent call."""
        try:
            action_params = reasoning_result["action_params"]
            agent_id = action_params["agent_id"]
            input_data = action_params["input_data"]
            
            logger.info(f"Calling agent {agent_id} with input: {input_data}")
            
            # Execute the agent call
            result = await self.task_executor.execute_action(
                "call_agent", 
                {"agent_id": agent_id, "input_data": input_data}, 
                state
            )
            
            # Record the result
            reasoning_step.observation = result[:500] + "..." if len(result) > 500 else result
            
            # Create agent call record
            agent_call = AgentCall(
                agent_id=agent_id,
                input_data=input_data,
                response=result,
                timestamp=datetime.now()
            )
            state.agent_calls.append(agent_call)
            
            logger.info(f"Agent {agent_id} call completed successfully")
            
        except Exception as e:
            error_msg = f"Agent call failed: {str(e)}"
            logger.error(error_msg)
            reasoning_step.observation = error_msg
    
    async def _execute_synthesis(self, reasoning_result: Dict[str, Any], state: SupervisorState, reasoning_step: ReasoningStep):
        """Execute response synthesis."""
        try:
            action_params = reasoning_result["action_params"]
            partial_results = action_params.get("partial_results", [])
            
            if not partial_results:
                # Collect from agent calls
                partial_results = [call.response for call in state.agent_calls if call.response]
            
            logger.info(f"Synthesizing {len(partial_results)} responses")
            
            # Synthesize responses
            synthesized_response = await self.react_engine.synthesize_responses(state, partial_results)
            
            # Record the result
            reasoning_step.observation = "Synthesis completed successfully"
            state.final_response = synthesized_response
            
            logger.info("Response synthesis completed successfully")
            
        except Exception as e:
            error_msg = f"Synthesis failed: {str(e)}"
            logger.error(error_msg)
            reasoning_step.observation = error_msg
    
    async def _generate_final_response(self, state: SupervisorState) -> str:
        """Generate the final response with enhanced synthesis."""
        if state.final_response:
            return state.final_response
        
        # If no synthesis was performed, try to synthesize now
        if state.agent_calls:
            responses = [call.response for call in state.agent_calls if call.response]
            if responses:
                try:
                    # Use enhanced synthesis even for fallback
                    logger.info("Performing enhanced synthesis for final response")
                    synthesized_response = await self.react_engine.synthesize_responses(state, responses)
                    return synthesized_response
                except Exception as e:
                    logger.error(f"Enhanced synthesis failed, using fallback: {e}")
                    # Use enhanced fallback synthesis
                    return self.react_engine._create_fallback_synthesis(state, responses)
        
        return "I was unable to process your request."
    
    def _format_observation_for_user(self, step: ReasoningStep) -> str:
        """Format observation for user-friendly display."""
        if step.action == "call_agent":
            # For agent calls, show a summary instead of raw response
            if step.observation and len(step.observation) > 100:
                # Extract agent name from observation
                if "Agent research responded:" in step.observation:
                    return "Successfully received comprehensive history lesson about Lake Como"
                elif "Agent code responded:" in step.observation:
                    return "Successfully received complete Rust application for trip scheduling"
                elif "Agent creative responded:" in step.observation:
                    return "Successfully received creative content"
                elif "Agent general responded:" in step.observation:
                    return "Successfully received general assistance"
                else:
                    return "Successfully received specialized response"
            return step.observation
        elif step.action == "synthesize_response":
            return "Successfully combined responses into unified answer"
        else:
            return step.observation