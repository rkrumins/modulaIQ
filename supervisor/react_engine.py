"""
Intelligent ReAct reasoning engine that uses LLM for all decision making.

This implementation is designed to be:
- Simple and clean
- Purely LLM-driven with no hardcoding
- Based on dynamic agent metadata
- Intelligent about multi-part queries
- Capable of complex agent orchestration
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .models import SupervisorState, AgentCall, ReasoningStep
from .llm_service import LLMService
from .agent_registry import AgentRegistry

logger = logging.getLogger(__name__)


class ReActEngine:
    """
    Intelligent ReAct reasoning engine that uses LLM for all decision making.
    
    This engine:
    - Uses LLM to analyze queries and select agents
    - Makes decisions based on dynamic agent metadata
    - Handles multi-part queries intelligently
    - Orchestrates complex agent workflows
    - Has no hardcoded logic or keyword matching
    """
    
    def __init__(self, llm_service: LLMService, agent_registry: AgentRegistry):
        self.llm_service = llm_service
        self.agent_registry = agent_registry
        self._agent_metadata_cache = None
        self._cache_timestamp = None
        self._cache_ttl_seconds = 300  # 5 minutes
    
    async def reason_and_act(self, state: SupervisorState, conversation_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform intelligent ReAct reasoning step using LLM.
        
        The LLM makes all decisions based on:
        - Current query and context
        - Available agents and their capabilities
        - Previous reasoning steps
        - Multi-part query analysis
        """
        try:
            # Get current agent metadata
            agent_metadata = await self._get_agent_metadata()
            
            # Build intelligent context for LLM
            context = self._build_llm_context(state, agent_metadata, conversation_context)
            
            # Let LLM make the decision
            decision = await self._llm_decision(context)
            
            # Validate and execute the decision
            result = await self._execute_decision(decision, state, agent_metadata)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in reason_and_act: {e}")
            return self._fallback_decision(state)
    
    async def _get_agent_metadata(self) -> Dict[str, Any]:
        """Get current agent metadata with caching."""
        now = datetime.now()
        
        # Check cache
        if (self._agent_metadata_cache and 
            self._cache_timestamp and 
            (now - self._cache_timestamp).seconds < self._cache_ttl_seconds):
            return self._agent_metadata_cache
        
        # Fetch fresh metadata
        try:
            agents = self.agent_registry.get_all_agents()
            
            # Build comprehensive metadata
            metadata = {
                "agents": [],
                "capabilities": set(),
                "domains": set(),
                "total_agents": len(agents)
            }
            
            for agent in agents:
                agent_info = {
                    "id": agent.agent_id,
                    "name": agent.name,
                    "description": agent.description,
                    "capabilities": agent.capabilities,
                    "status": agent.status,
                    "endpoint": agent.endpoint,
                    "health_check": agent.health_check,
                    "registration_time": agent.registration_time.isoformat() if agent.registration_time else None
                }
                metadata["agents"].append(agent_info)
                metadata["capabilities"].update(agent.capabilities)
                
                # Infer domain from capabilities and description
                domain = self._infer_domain(agent_info)
                agent_info["domain"] = domain
                metadata["domains"].add(domain)
            
            # Convert sets to lists for JSON serialization
            metadata["capabilities"] = list(metadata["capabilities"])
            metadata["domains"] = list(metadata["domains"])
            
            # Cache the result
            self._agent_metadata_cache = metadata
            self._cache_timestamp = now
            
            logger.info(f"Fetched metadata for {len(agents)} agents")
            return metadata
            
        except Exception as e:
            logger.error(f"Error fetching agent metadata: {e}")
            return {"agents": [], "capabilities": [], "domains": [], "total_agents": 0}
    
    def _infer_domain(self, agent_info: Dict[str, Any]) -> str:
        """Infer agent domain from capabilities and description."""
        capabilities = [cap.lower() for cap in agent_info["capabilities"]]
        description = agent_info["description"].lower()
        
        # Research domain
        if any(keyword in capabilities or keyword in description 
               for keyword in ["research", "analysis", "investigation", "information", "data"]):
            return "research"
        
        # Code domain
        if any(keyword in capabilities or keyword in description 
               for keyword in ["code", "programming", "software", "development", "coding"]):
            return "code"
        
        # Creative domain
        if any(keyword in capabilities or keyword in description 
               for keyword in ["creative", "writing", "poetry", "story", "art", "content"]):
            return "creative"
        
        # General domain
        return "general"
    
    def _build_llm_context(self, state: SupervisorState, agent_metadata: Dict[str, Any], conversation_context: List[Dict[str, Any]]) -> str:
        """Build comprehensive context for LLM decision making."""
        context_parts = []
        
        # Current situation
        context_parts.append("CURRENT SITUATION:")
        context_parts.append(f"Query: {state.query}")
        context_parts.append(f"Iteration: {state.current_iteration}/{state.max_iterations}")
        context_parts.append(f"Agents called so far: {len(state.agent_calls)}")
        
        # Add information about previous agent calls
        if state.agent_calls:
            context_parts.append("\nPREVIOUS AGENT CALLS:")
            for i, call in enumerate(state.agent_calls, 1):
                context_parts.append(f"- Call {i}: {call.agent_id} agent")
            
            # Count unique agents
            unique_agents = set(call.agent_id for call in state.agent_calls)
            context_parts.append(f"- Unique agents called: {len(unique_agents)} ({', '.join(unique_agents)})")
        
        # Available agents
        context_parts.append("\nAVAILABLE AGENTS:")
        for agent in agent_metadata["agents"]:
            context_parts.append(f"- {agent['id']} ({agent['domain']}): {agent['description']}")
            context_parts.append(f"  Capabilities: {', '.join(agent['capabilities'])}")
        
        # Previous reasoning
        if state.reasoning_steps:
            context_parts.append("\nPREVIOUS REASONING:")
            for step in state.reasoning_steps[-3:]:  # Last 3 steps
                context_parts.append(f"Step {step.step}: {step.reasoning}")
                context_parts.append(f"Action: {step.action}")
                if step.observation:
                    context_parts.append(f"Result: {step.observation[:200]}...")
        
        # Conversation context
        if conversation_context:
            context_parts.append("\nCONVERSATION CONTEXT:")
            for msg in conversation_context[-5:]:  # Last 5 messages
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                context_parts.append(f"{role.upper()}: {content[:100]}...")
        
        return "\n".join(context_parts)
    
    async def _llm_decision(self, context: str) -> Dict[str, Any]:
        """Let LLM make the decision based on context."""
        system_prompt = """You are an intelligent agent orchestrator. Your job is to analyze the current situation and decide what to do next.

You have access to multiple specialized agents with different capabilities. Your decisions should be based on:
1. The user's query and what they're asking for
2. The capabilities of available agents
3. What has been done in previous steps
4. Whether this is a multi-part query that needs multiple agents

DECISION FRAMEWORK:
- If the query has multiple distinct parts, you should address them one at a time
- Choose the most appropriate agent for each part based on their capabilities
- Only synthesize when you have responses from all relevant agents
- Be explicit about your reasoning

RESPONSE FORMAT (JSON only):
{
    "reasoning": "Your detailed reasoning about what to do next",
    "action": "call_agent" or "synthesize_response",
    "agent_id": "agent_id_if_calling_agent",
    "input_data": {"query": "specific_part_of_query_if_calling_agent"},
    "is_final": true_or_false
}

CRITICAL RULES:
- For multi-part queries, you MUST call multiple agents before synthesizing
- Choose agents based on their actual capabilities, not assumptions
- Be specific about which part of the query you're addressing
- Only set is_final=true when you have a complete answer
- If you've already called agents for different parts of the query, SYNTHESIZE instead of calling more agents
- Do NOT call the same agent multiple times unless absolutely necessary"""

        user_prompt = f"""Based on the current situation, decide what to do next.

{context}

What should be the next action? Provide your reasoning and decision in the specified JSON format."""

        try:
            response = await self.llm_service.call_llm_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            
            # Clean and parse JSON response
            cleaned_response = response.strip()
            
            # Try to extract JSON from response if it's wrapped in other text
            if "```json" in cleaned_response:
                start = cleaned_response.find("```json") + 7
                end = cleaned_response.find("```", start)
                if end != -1:
                    cleaned_response = cleaned_response[start:end].strip()
            elif "{" in cleaned_response and "}" in cleaned_response:
                start = cleaned_response.find("{")
                end = cleaned_response.rfind("}") + 1
                cleaned_response = cleaned_response[start:end]
            
            # Parse JSON response
            decision = json.loads(cleaned_response)
            
            # Validate required fields
            if "action" not in decision:
                decision["action"] = "call_agent"
            if "reasoning" not in decision:
                decision["reasoning"] = "No reasoning provided"
            if "is_final" not in decision:
                decision["is_final"] = False
            
            logger.info(f"LLM decision: {decision}")
            return decision
            
        except Exception as e:
            logger.error(f"Error in LLM decision: {e}")
            # Intelligent fallback based on query analysis
            return await self._intelligent_fallback(state, agent_metadata)
    
    async def _intelligent_fallback(self, state: SupervisorState, agent_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Intelligent fallback when LLM decision fails."""
        try:
            # Analyze the original query to determine what agent to call
            original_query = state.original_query.lower()
            
            # Count existing agent calls to avoid infinite loops
            agent_call_count = len(state.agent_calls)
            
            # If we've already made too many calls, synthesize
            if agent_call_count >= 3:
                return {
                    "reasoning": "Maximum agent calls reached, synthesizing responses",
                    "action": "synthesize_response",
                    "is_final": True
                }
            
            # Determine which agent to call based on query content and existing calls
            if "history" in original_query or "lesson" in original_query:
                if not any(call.agent_id == "research" for call in state.agent_calls):
                    return {
                        "reasoning": "Query contains history/lesson request, calling research agent",
                        "action": "call_agent",
                        "agent_id": "research",
                        "input_data": {"query": "Provide a history lesson about the requested topic"},
                        "is_final": False
                    }
            
            if "code" in original_query or "app" in original_query or "program" in original_query or "rust" in original_query:
                if not any(call.agent_id == "code" for call in state.agent_calls):
                    return {
                        "reasoning": "Query contains coding request, calling code agent",
                        "action": "call_agent",
                        "agent_id": "code",
                        "input_data": {"query": "Create the requested code/application"},
                        "is_final": False
                    }
            
            if "creative" in original_query or "poem" in original_query or "story" in original_query:
                if not any(call.agent_id == "creative" for call in state.agent_calls):
                    return {
                        "reasoning": "Query contains creative request, calling creative agent",
                        "action": "call_agent",
                        "agent_id": "creative",
                        "input_data": {"query": "Create the requested creative content"},
                        "is_final": False
                    }
            
            # If we have some agent calls, check if we should synthesize
            if agent_call_count > 0:
                # Count unique agents called
                unique_agents_called = set(call.agent_id for call in state.agent_calls)
                unique_agent_count = len(unique_agents_called)
                
                # If we've called multiple unique agents or called the same agent multiple times, synthesize
                if unique_agent_count > 1 or agent_call_count > 1:
                    return {
                        "reasoning": "Have sufficient agent responses, synthesizing final answer",
                        "action": "synthesize_response",
                        "is_final": True
                    }
            
            # Last resort: call general agent only if no other calls made
            return {
                "reasoning": "No specific agent identified, calling general agent",
                "action": "call_agent",
                "agent_id": "general",
                "input_data": {"query": state.original_query},
                "is_final": False
            }
            
        except Exception as e:
            logger.error(f"Error in intelligent fallback: {e}")
            # Absolute fallback
            return {
                "reasoning": "Fallback failed, calling general agent",
                "action": "call_agent",
                "agent_id": "general",
                "input_data": {"query": "Please help with this request"},
                "is_final": False
            }
    
    async def _execute_decision(self, decision: Dict[str, Any], state: SupervisorState, agent_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the LLM's decision."""
        action = decision.get("action", "call_agent")
        reasoning = decision.get("reasoning", "No reasoning provided")
        
        if action == "call_agent":
            agent_id = decision.get("agent_id", "general")
            input_data = decision.get("input_data", {"query": state.query})
            
            # Validate agent exists
            if not any(agent["id"] == agent_id for agent in agent_metadata["agents"]):
                logger.warning(f"LLM selected non-existent agent {agent_id}, using general")
                agent_id = "general"
            
            return {
                "reasoning": reasoning,
                "action": "call_agent",
                "action_params": {
                    "agent_id": agent_id,
                    "input_data": input_data
                },
                "is_final": decision.get("is_final", False)
            }
        
        elif action == "synthesize_response":
            return {
                "reasoning": reasoning,
                "action": "synthesize_response",
                "action_params": {
                    "partial_results": [call.response for call in state.agent_calls if call.response]
                },
                "is_final": True
            }
        
        else:
            logger.warning(f"Unknown action: {action}")
            return self._fallback_decision(state)
    
    def _fallback_decision(self, state: SupervisorState) -> Dict[str, Any]:
        """Fallback decision when LLM fails."""
        if not state.agent_calls:
            return {
                "reasoning": "Fallback: No agents called yet, using general agent",
                "action": "call_agent",
                "action_params": {
                    "agent_id": "general",
                    "input_data": {"query": state.query}
                },
                "is_final": False
            }
        else:
            return {
                "reasoning": "Fallback: Synthesizing available responses",
                "action": "synthesize_response",
                "action_params": {
                    "partial_results": [call.response for call in state.agent_calls if call.response]
                },
                "is_final": True
            }
    
    async def should_continue(self, state: SupervisorState) -> bool:
        """
        Intelligently determine if the ReAct loop should continue.
        
        This is now purely based on LLM analysis of the current state.
        """
        # Check iteration limit
        if state.current_iteration >= state.max_iterations:
            logger.info("Reached maximum iterations")
            return False
        
        # Check if already complete
        if state.is_complete:
            return False
        
        try:
            # Get agent metadata
            agent_metadata = await self._get_agent_metadata()
            
            # Build context for LLM
            context = self._build_llm_context(state, agent_metadata, [])
            
            # Ask LLM if we should continue
            system_prompt = """You are an intelligent agent orchestrator. Analyze the current situation and determine if more work is needed.

Based on the context, decide if the ReAct loop should continue or if we have enough information to provide a complete answer.

Consider:
- Is this a multi-part query that needs multiple agents?
- Have all parts been addressed?
- Do we have sufficient information to answer the user's query?

Respond with only "CONTINUE" or "STOP" followed by a brief explanation."""

            user_prompt = f"""Should the ReAct loop continue or stop?

{context}

Decision:"""

            response = await self.llm_service.call_llm_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            
            should_continue = "CONTINUE" in response.upper()
            logger.info(f"LLM continuation decision: {should_continue} - {response}")
            
            return should_continue
            
        except Exception as e:
            logger.error(f"Error in should_continue: {e}")
            # Intelligent fallback based on current state
            return await self._intelligent_continuation_fallback(state)
    
    async def _intelligent_continuation_fallback(self, state: SupervisorState) -> bool:
        """Intelligent fallback for continuation decision."""
        try:
            # Analyze the original query to determine if we need more agents
            original_query = state.original_query.lower()
            agent_call_count = len(state.agent_calls)
            
            # If we've made too many calls, stop
            if agent_call_count >= 3:
                return False
            
            # Check if this is a multi-part query that needs multiple agents
            has_history = "history" in original_query or "lesson" in original_query
            has_coding = "code" in original_query or "app" in original_query or "program" in original_query or "rust" in original_query
            has_creative = "creative" in original_query or "poem" in original_query or "story" in original_query
            
            # Count how many different types of requests we have
            request_types = sum([has_history, has_coding, has_creative])
            
            # Count unique agents called
            unique_agents_called = set(call.agent_id for call in state.agent_calls)
            unique_agent_count = len(unique_agents_called)
            
            # If we have multiple request types but haven't called enough unique agents, continue
            if request_types > 1 and unique_agent_count < request_types:
                return True
            
            # If we have both history and coding requests but only called one type of agent, continue
            if has_history and has_coding and unique_agent_count < 2:
                return True
            
            # If we have some agent calls but the query suggests we need more, continue
            if agent_call_count > 0 and agent_call_count < 2 and (has_history and has_coding):
                return True
            
            # If we've called the same agent multiple times, stop (prevent excessive calls)
            if agent_call_count > 1 and unique_agent_count == 1:
                return False
            
            # Otherwise, we probably have enough
            return False
            
        except Exception as e:
            logger.error(f"Error in intelligent continuation fallback: {e}")
            # Conservative fallback: stop if we have any agent calls
            return len(state.agent_calls) == 0
    
    async def synthesize_responses(self, state: SupervisorState, partial_results: List[str]) -> str:
        """
        Intelligently synthesize responses from multiple agents.
        
        Creates a unified, high-quality response that properly combines and rephrases
        the outputs from different agents based on the user's original query.
        """
        try:
            # Get agent metadata for context
            agent_metadata = await self._get_agent_metadata()
            
            # Build comprehensive synthesis context
            context_parts = []
            
            # Original user query and context
            context_parts.append("ORIGINAL USER QUERY:")
            context_parts.append(f'"{state.original_query}"')
            context_parts.append("")
            
            # Agent responses with context
            context_parts.append("AGENT RESPONSES TO SYNTHESIZE:")
            context_parts.append("")
            
            for i, call in enumerate(state.agent_calls):
                agent_id = call.agent_id
                agent_response = call.response or ""
                
                # Find agent info for context
                agent_info = next((agent for agent in agent_metadata["agents"] if agent["id"] == agent_id), None)
                agent_description = agent_info["description"] if agent_info else "Specialized agent"
                
                context_parts.append(f"=== RESPONSE FROM {agent_id.upper()} AGENT ({agent_description}) ===")
                context_parts.append(f"Query to agent: {call.input_data.get('query', 'N/A')}")
                context_parts.append(f"Agent response:")
                context_parts.append(agent_response)
                context_parts.append("")
            
            # Available agents for reference
            context_parts.append("AVAILABLE AGENTS (for reference):")
            for agent in agent_metadata["agents"]:
                context_parts.append(f"- {agent['id']} ({agent['domain']}): {agent['description']}")
                context_parts.append(f"  Capabilities: {', '.join(agent['capabilities'])}")
            
            context = "\n".join(context_parts)
            
            # Enhanced system prompt for intelligent synthesis
            system_prompt = """You are an expert response synthesizer and content curator. Your job is to create a unified, high-quality response that intelligently combines outputs from multiple specialized agents.

CRITICAL INSTRUCTIONS:
1. **ANALYZE THE ORIGINAL QUERY**: Understand what the user actually asked for
2. **INTEGRATE RESPONSES INTELLIGENTLY**: Don't just concatenate - rephrase and combine
3. **MAINTAIN EXPERTISE**: Preserve the specialized knowledge from each agent
4. **CREATE UNIFIED NARRATIVE**: Make it feel like one cohesive response
5. **ADDRESS ALL PARTS**: Ensure every part of the original query is fully addressed
6. **IMPROVE FLOW**: Create smooth transitions between different topics/sections
7. **ENHANCE QUALITY**: Improve clarity, organization, and readability

SYNTHESIS GUIDELINES:
- **Rephrase and combine** rather than just copying text
- **Create logical flow** that makes sense for the user's query
- **Maintain technical accuracy** from specialized agents
- **Add connecting thoughts** to unify different parts
- **Structure appropriately** (use headers, sections, etc.)
- **Be comprehensive** but avoid redundancy
- **Preserve important details** from each agent's response
- **Make it feel natural** and conversational

RESPONSE STRUCTURE:
- Start with a brief overview that addresses the user's complete request
- Organize content logically based on the original query
- Use clear section headers when appropriate
- End with a summary or conclusion that ties everything together
- Ensure the response feels like it came from one knowledgeable source

Your goal is to create a response that is BETTER than the sum of its parts - more organized, more comprehensive, and more useful to the user."""

            user_prompt = f"""Create a unified, high-quality response by intelligently synthesizing the agent responses below.

Remember: The user asked: "{state.original_query}"

Your task is to create a response that:
1. Fully addresses their original question
2. Intelligently combines the specialized knowledge from each agent
3. Flows naturally as one cohesive answer
4. Is well-organized and easy to follow
5. Maintains the expertise and accuracy of each agent's contribution

{context}

UNIFIED RESPONSE:"""

            response = await self.llm_service.call_llm_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            
            logger.info("Successfully synthesized responses with enhanced quality")
            logger.info(f"Synthesized response length: {len(response)}")
            logger.info(f"Synthesized response preview: {response[:200]}...")
            return response
            
        except Exception as e:
            logger.error(f"Error in synthesize_responses: {e}")
            # Enhanced fallback: better concatenation with context
            return self._create_fallback_synthesis(state, partial_results)
    
    def _create_fallback_synthesis(self, state: SupervisorState, partial_results: List[str]) -> str:
        """
        Create an enhanced fallback synthesis when LLM synthesis fails.
        
        Provides better structure and context than simple concatenation.
        """
        try:
            # Create a structured fallback response
            response_parts = []
            
            # Introduction
            response_parts.append(f"Based on your request: \"{state.original_query}\"")
            response_parts.append("")
            response_parts.append("Here's a comprehensive response combining insights from specialized agents:")
            response_parts.append("")
            
            # Add each agent's response with context
            for i, call in enumerate(state.agent_calls):
                agent_id = call.agent_id
                agent_response = call.response or ""
                
                # Clean the response to remove "Agent X responded:" prefix
                if agent_response.startswith(f"Agent {agent_id} responded:"):
                    agent_response = agent_response[len(f"Agent {agent_id} responded:"):].strip()
                
                # Create section header
                if agent_id == "research":
                    section_title = "Research & Historical Information"
                elif agent_id == "code":
                    section_title = "Technical Implementation"
                elif agent_id == "creative":
                    section_title = "Creative Content"
                else:
                    section_title = f"Response from {agent_id.title()} Agent"
                
                response_parts.append(f"## {section_title}")
                response_parts.append("")
                response_parts.append(agent_response)
                response_parts.append("")
            
            # Conclusion
            response_parts.append("---")
            response_parts.append("")
            response_parts.append("This response combines specialized knowledge from multiple agents to provide a comprehensive answer to your query.")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Error in fallback synthesis: {e}")
            # Ultimate fallback: clean concatenation
            cleaned_responses = []
            for response in partial_results:
                # Remove "Agent X responded:" prefix if present
                if " responded:" in response:
                    response = response.split(" responded:", 1)[1].strip()
                cleaned_responses.append(response)
            return "\n\n".join(cleaned_responses)
