"""LangGraph-based agent implementation."""

import asyncio
from typing import Dict, List, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

from agents.base_agent import BaseAgent, AgentRequest, AgentResponse
from core.config import Config


class LangGraphAgent(BaseAgent):
    """LangGraph-based agent with tool support."""
    
    def __init__(self, config: Config, agent_id: str, name: str, description: str, 
                 capabilities: List[str], system_prompt: str, tools: Optional[List] = None):
        super().__init__(config, agent_id, name, description, capabilities)
        self.system_prompt = system_prompt
        self.tools = tools or []
        
        # Create the graph
        self.graph = self._create_graph()
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow."""
        # Define the state
        from typing_extensions import TypedDict
        
        class AgentState(TypedDict):
            messages: List[Any]
            context: Dict[str, Any]
        
        # Create the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", self._agent_node)
        
        if self.tools:
            # Add tool node if tools are available
            tool_node = ToolNode(self.tools)
            workflow.add_node("tools", tool_node)
            
            # Add edges
            workflow.add_edge("agent", "tools")
            workflow.add_edge("tools", "agent")
        else:
            # No tools, just end after agent
            workflow.add_edge("agent", END)
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        return workflow.compile()
    
    async def _agent_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Agent node that processes messages."""
        messages = state.get("messages", [])
        context = state.get("context", {})
        
        # Create prompt with system message
        prompt_messages = [SystemMessage(content=self.system_prompt)]
        prompt_messages.extend(messages)
        
        # Get response from LLM
        response = await self.llm.ainvoke(prompt_messages)
        
        # Add response to messages
        messages.append(response)
        
        return {"messages": messages}
    
    async def process_request(self, request: AgentRequest) -> AgentResponse:
        """Process a request using LangGraph."""
        try:
            # Get conversation context
            context_messages = self.get_conversation_context(request.session_id)
            
            # Prepare messages
            messages = []
            for msg in context_messages:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))
            
            # Add current request
            input_content = request.input_data.get("query", str(request.input_data))
            messages.append(HumanMessage(content=input_content))
            
            # Prepare state
            state = {
                "messages": messages,
                "context": request.context or {}
            }
            
            # Run the graph
            result = await self.graph.ainvoke(state)
            
            # Extract response
            final_messages = result.get("messages", [])
            if final_messages:
                last_message = final_messages[-1]
                if hasattr(last_message, 'content'):
                    response_content = last_message.content
                else:
                    response_content = str(last_message)
            else:
                response_content = "No response generated"
            
            # Store conversation
            self.store_message(request.session_id, "user", input_content)
            self.store_message(request.session_id, "assistant", response_content)
            
            return AgentResponse(
                session_id=request.session_id,
                response=response_content,
                metadata={
                    "agent_id": self.agent_id,
                    "tools_used": len(self.tools),
                    "messages_processed": len(final_messages)
                }
            )
            
        except Exception as e:
            return AgentResponse(
                session_id=request.session_id,
                response=f"Error processing request: {str(e)}",
                success=False,
                error=str(e)
            )


class SimpleAgent(BaseAgent):
    """Simple agent without LangGraph complexity."""
    
    def __init__(self, config: Config, agent_id: str, name: str, description: str, 
                 capabilities: List[str], system_prompt: str):
        super().__init__(config, agent_id, name, description, capabilities)
        self.system_prompt = system_prompt
    
    async def process_request(self, request: AgentRequest) -> AgentResponse:
        """Process a request with a simple LLM call."""
        try:
            # Get conversation context
            context_messages = self.get_conversation_context(request.session_id)
            
            # Prepare messages
            messages = [SystemMessage(content=self.system_prompt)]
            
            for msg in context_messages:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))
            
            # Add current request
            input_content = request.input_data.get("query", str(request.input_data))
            messages.append(HumanMessage(content=input_content))
            
            # Get response from LLM
            response = await self.llm.ainvoke(messages)
            response_content = response.content
            
            # Store conversation
            self.store_message(request.session_id, "user", input_content)
            self.store_message(request.session_id, "assistant", response_content)
            
            return AgentResponse(
                session_id=request.session_id,
                response=response_content,
                metadata={
                    "agent_id": self.agent_id,
                    "model": self.config.supervisor.model
                }
            )
            
        except Exception as e:
            return AgentResponse(
                session_id=request.session_id,
                response=f"Error processing request: {str(e)}",
                success=False,
                error=str(e)
            )
