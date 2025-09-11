"""Query analyzer for LLM-driven query classification and analysis."""

from typing import Dict, Any, List, Optional
import logging

from supervisor.llm_service import LLMService
from supervisor.json_utils import parse_json_response, validate_json_fields

logger = logging.getLogger(__name__)


class QueryAnalyzer:
    """Analyzes queries using LLM to determine complexity and approach."""
    
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
    
    async def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze a query to determine its characteristics and best approach using intelligent LLM analysis.
        
        Args:
            query: User query to analyze
            
        Returns:
            Dictionary with analysis results
        """
        system_prompt = """You are an expert at analyzing user queries to determine their complexity, intent, and best processing approach.

Analyze queries based on:
1. **Complexity**: Simple vs Complex
2. **Intent**: What the user is trying to achieve
3. **Scope**: Single-part vs Multi-part
4. **Domain**: Technical, creative, research, general, etc.
5. **Approach**: Direct response vs Agent orchestration
6. **Agent Requirements**: Which specialized agents would be most appropriate

RESPONSE FORMAT (JSON only):
{
    "complexity": "simple" or "complex",
    "intent": "Brief description of user intent",
    "scope": "single_part" or "multi_part",
    "domain": "technical|creative|research|general|mixed",
    "approach": "direct_response" or "agent_orchestration",
    "suggested_agents": ["agent1", "agent2"],
    "reasoning": "Detailed explanation of your analysis",
    "confidence": 0.0-1.0
}

Guidelines:
- Simple queries: Basic questions, calculations, definitions, single facts
- Complex queries: Multi-step problems, creative tasks, research, analysis
- Single-part: One clear objective or question
- Multi-part: Multiple distinct components requiring different expertise
- Direct response: Can be answered directly without specialized agents
- Agent orchestration: Requires specialized agents or multi-step processing
- Be intelligent about detecting implicit requirements (e.g., "write an app" needs code agent)
- Consider the user's true intent, not just explicit keywords
- Think about what the user is really trying to accomplish"""

        user_prompt = f"""Analyze this query: "{query}"

Provide your analysis in JSON format."""

        try:
            response = await self.llm_service.call_llm_with_system_prompt(
                system_prompt, user_prompt, "query_analysis", timeout=15.0
            )
            
            # Parse JSON response
            analysis = parse_json_response(response, "query_analysis")
            if not analysis:
                return self._fallback_analysis(query)
            
            # Validate required fields
            required_fields = ["complexity", "intent", "scope", "domain", "approach", "reasoning", "confidence"]
            if not validate_json_fields(analysis, required_fields, "query_analysis"):
                return self._fallback_analysis(query)
            
            # Add suggested_agents if not present
            if "suggested_agents" not in analysis:
                analysis["suggested_agents"] = []
            
            # Validate values
            if analysis["complexity"] not in ["simple", "complex"]:
                analysis["complexity"] = "simple"
            
            if analysis["scope"] not in ["single_part", "multi_part"]:
                analysis["scope"] = "single_part"
            
            if analysis["domain"] not in ["technical", "creative", "research", "general", "mixed"]:
                analysis["domain"] = "general"
            
            if analysis["approach"] not in ["direct_response", "agent_orchestration"]:
                analysis["approach"] = "direct_response"
            
            # Ensure confidence is a float between 0 and 1
            try:
                confidence = float(analysis["confidence"])
                analysis["confidence"] = max(0.0, min(1.0, confidence))
            except (ValueError, TypeError):
                analysis["confidence"] = 0.5
            
            logger.info(f"Query analysis: {analysis['complexity']} {analysis['scope']} {analysis['domain']} (confidence: {analysis['confidence']:.2f})")
            return analysis
                
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            return self._fallback_analysis(query)
    
    def _fallback_analysis(self, query: str) -> Dict[str, Any]:
        """Fallback analysis when LLM fails."""
        # Simple heuristic fallback
        query_lower = query.lower()
        
        # Check for simple patterns
        simple_indicators = [
            "what is", "who is", "when is", "where is", "how many", "how much",
            "define", "explain", "tell me about"
        ]
        
        is_simple = any(indicator in query_lower for indicator in simple_indicators)
        is_simple = is_simple and len(query.split()) <= 10
        
        # Check for multi-part indicators
        multi_part_indicators = ["and", "also", "plus", "additionally", "furthermore"]
        is_multi_part = any(indicator in query_lower for indicator in multi_part_indicators)
        
        # Determine domain
        if any(keyword in query_lower for keyword in ["code", "program", "function", "python", "javascript"]):
            domain = "technical"
        elif any(keyword in query_lower for keyword in ["creative", "story", "poem", "write"]):
            domain = "creative"
        elif any(keyword in query_lower for keyword in ["research", "analyze", "study", "investigate"]):
            domain = "research"
        else:
            domain = "general"
        
        return {
            "complexity": "simple" if is_simple else "complex",
            "intent": "User query requiring analysis",
            "scope": "multi_part" if is_multi_part else "single_part",
            "domain": domain,
            "approach": "direct_response" if is_simple else "agent_orchestration",
            "reasoning": "Fallback analysis due to LLM failure",
            "confidence": 0.3
        }
    
    async def is_simple_query(self, query: str) -> bool:
        """
        Determine if a query is simple enough for direct response.
        
        Args:
            query: User query
            
        Returns:
            True if query is simple, False otherwise
        """
        analysis = await self.analyze_query(query)
        return (
            analysis["complexity"] == "simple" and 
            analysis["approach"] == "direct_response" and
            analysis["confidence"] > 0.6
        )
    
    async def is_multi_part_query(self, query: str) -> bool:
        """
        Determine if a query has multiple parts requiring different agents.
        
        Args:
            query: User query
            
        Returns:
            True if query is multi-part, False otherwise
        """
        analysis = await self.analyze_query(query)
        return (
            analysis["scope"] == "multi_part" and 
            analysis["confidence"] > 0.6
        )
    
    async def get_query_components(self, query: str) -> List[Dict[str, str]]:
        """
        Break down a multi-part query into components.
        
        Args:
            query: User query
            
        Returns:
            List of query components with their characteristics
        """
        system_prompt = """You are an expert at breaking down complex, multi-part queries into distinct components.

Your task is to:
1. Identify distinct parts of the query
2. Determine what type of expertise each part requires
3. Suggest which agent would be best for each part

RESPONSE FORMAT (JSON only):
{
    "components": [
        {
            "part": "Description of this part of the query",
            "expertise_needed": "research|code|creative|general",
            "suggested_agent": "agent_id",
            "priority": 1-5
        }
    ],
    "reasoning": "Explanation of how you broke down the query"
}

Guidelines:
- Break down only if the query truly has multiple distinct parts
- Each component should be self-contained
- Prioritize components (1 = highest priority)
- Match expertise to available agents"""

        user_prompt = f"""Break down this query into components: "{query}"

If this is not a multi-part query, return a single component."""

        try:
            response = await self.llm_service.call_llm_with_system_prompt(
                system_prompt, user_prompt, "query_breakdown", timeout=15.0
            )
            
            result = parse_json_response(response, "query_breakdown")
            if not result:
                return [{"part": query, "expertise_needed": "general", "suggested_agent": "general", "priority": 1}]
            
            if "components" not in result:
                logger.error("Missing components field in query breakdown")
                return [{"part": query, "expertise_needed": "general", "suggested_agent": "general", "priority": 1}]
            
            # Validate components
            for component in result["components"]:
                required_fields = ["part", "expertise_needed", "suggested_agent", "priority"]
                if not validate_json_fields(component, required_fields, "query_breakdown_component"):
                    return [{"part": query, "expertise_needed": "general", "suggested_agent": "general", "priority": 1}]
            
            logger.info(f"Query broken down into {len(result['components'])} components")
            return result["components"]
                
        except Exception as e:
            logger.error(f"Query breakdown failed: {e}")
            return [{"part": query, "expertise_needed": "general", "suggested_agent": "general", "priority": 1}]
