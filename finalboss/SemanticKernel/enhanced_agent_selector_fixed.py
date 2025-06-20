#!/usr/bin/env python3
"""
Enhanced Agent Selection System for Azure Foundry Orchestration
"""

import json
import re
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import IntEnum
from datetime import datetime


class TaskComplexity(IntEnum):
    """Task complexity levels for agent selection - orderable by value"""
    SIMPLE = 1              # Basic greetings, simple questions
    MODERATE = 2            # Single-domain questions requiring some expertise
    COMPLEX = 3             # Multi-step or specialized knowledge required
    EXPERT = 4              # Highly specialized or technical


class DomainType(IntEnum):
    """Domain categories for agent specialization"""
    GENERAL = 1
    TECHNICAL = 2
    CAREER = 3
    MICROSOFT365 = 4
    TRANSACTION = 5
    FOUNDRY = 6
    PHILOSOPHY = 7


@dataclass
class RoutingDecision:
    """Detailed routing decision with confidence and reasoning"""
    selected_agent: str
    confidence: float
    reasoning: str
    alternative_agents: List[Tuple[str, float]]
    task_complexity: TaskComplexity
    detected_domains: List[DomainType]


class EnhancedAgentSelector:
    """Enhanced agent selection system with sophisticated routing logic"""
    
    def __init__(self):
        """Initialize the enhanced agent selector"""
        self.routing_history = []
        self.agent_success_rates = {}
    async def select_best_agent_ai(self, user_input: str, available_agents: List[str]) -> RoutingDecision:
        """Select the best agent using AI-powered analysis"""
        
        # Create agent context for AI selection
        agent_context = ""
        for agent_name in available_agents:
            if agent_name in self.agent_definitions:
                agent_def = self.agent_definitions[agent_name]
                agent_context += f"\n**{agent_name}** ({agent_def['name']}):\n"
                agent_context += f"Description: {agent_def['description']}\n"
                agent_context += f"Best for: {agent_def['best_for']}\n"
                agent_context += f"Specialties: {', '.join(agent_def['specialties'])}\n"
        
        # Create AI selection prompt
        selection_prompt = f"""You are an expert agent router. Analyze the user request and select the most appropriate agent.

AVAILABLE AGENTS:
{agent_context}

USER REQUEST: "{user_input}"

ANALYSIS GUIDELINES:
1. Look for specific data patterns:
   - Raw numerical data (decimal numbers in sequences) → TransactionAgent/OrchestratorAgent
   - Boolean sequences (TRUE/FALSE patterns) → TransactionAgent/OrchestratorAgent  
   - Mixed boolean/numerical data → TransactionAgent/OrchestratorAgent
   - Financial transaction keywords → TransactionAgent/OrchestratorAgent

2. Look for domain-specific requests:
   - Career/job/resume topics → CareerAdvisor
   - Microsoft 365/Office/Teams → M365Assistant or GraphAssistant
   - General questions/greetings → GeneralAssistant

3. Consider complexity:
   - Raw data analysis requires specialist agents
   - Simple greetings can use GeneralAssistant
   - Technical tasks need appropriate specialists

Respond with JSON:
{{
  "selected_agent": "AgentName",
  "confidence": 0.0-1.0,
  "reasoning": "Detailed explanation of why this agent was selected",
  "detected_patterns": ["pattern1", "pattern2"],
  "task_complexity": "simple|moderate|complex|expert"
}}

Be very specific about data patterns you detect. If you see numerical sequences or boolean data, route to TransactionAgent/OrchestratorAgent."""

        try:
            # Get AI response
            response = await self.azure_chat_completion.get_chat_message_content_async(
                messages=[{"role": "user", "content": selection_prompt}]
            )
            
            # Parse AI response
            response_text = str(response)
            
            # Extract JSON from response
            import json
            import re
            
            # Look for JSON in the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                ai_decision = json.loads(json_match.group())
                
                # Validate the selected agent is available
                selected_agent = ai_decision.get("selected_agent", "GeneralAssistant")
                if selected_agent not in available_agents:
                    # Fallback logic - prefer TransactionAgent for data patterns
                    if any(pattern in user_input.lower() for pattern in ["false", "true", "data", "analyze"]) and \
                       any(agent in available_agents for agent in ["TransactionAgent", "OrchestratorAgent"]):
                        selected_agent = next(agent for agent in ["TransactionAgent", "OrchestratorAgent"] if agent in available_agents)
                    else:
                        selected_agent = "GeneralAssistant"
                
                confidence = ai_decision.get("confidence", 0.7)
                reasoning = ai_decision.get("reasoning", "AI-powered selection")
                
                # Convert complexity
                complexity_map = {
                    "simple": TaskComplexity.SIMPLE,
                    "moderate": TaskComplexity.MODERATE,
                    "complex": TaskComplexity.COMPLEX,
                    "expert": TaskComplexity.EXPERT
                }
                task_complexity = complexity_map.get(ai_decision.get("task_complexity", "moderate"), TaskComplexity.MODERATE)
                
            else:
                raise ValueError("No valid JSON found in AI response")
                
        except Exception as e:
            print(f"⚠️ AI selection failed: {e}, using fallback logic")
            # Fallback to pattern-based selection for critical cases
            return self._fallback_agent_selection(user_input, available_agents)
        
        # Create routing decision
        decision = RoutingDecision(
            selected_agent=selected_agent,
            confidence=confidence,
            reasoning=f"AI-powered: {reasoning}",
            alternative_agents=[],
            task_complexity=task_complexity,
            detected_domains=[]
        )
        
        # Log decision
        self.routing_history.append({
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "decision": decision,
            "method": "AI-powered"
        })
        
        return decision

    def _fallback_agent_selection(self, user_input: str, available_agents: List[str]) -> RoutingDecision:
        """Fallback agent selection when AI fails"""
        user_lower = user_input.lower()
        
        # Simple pattern-based fallback
        if any(pattern in user_input for pattern in ["FALSE", "TRUE"]) and \
           any(agent in available_agents for agent in ["TransactionAgent", "OrchestratorAgent"]):
            selected_agent = next(agent for agent in ["TransactionAgent", "OrchestratorAgent"] if agent in available_agents)
            confidence = 0.8
            reasoning = "Fallback: Detected boolean data patterns"
        elif "career" in user_lower and "CareerAdvisor" in available_agents:
            selected_agent = "CareerAdvisor"
            confidence = 0.7
            reasoning = "Fallback: Career-related request"
        elif any(word in user_lower for word in ["microsoft", "office", "teams"]) and "M365Assistant" in available_agents:
            selected_agent = "M365Assistant"
            confidence = 0.7
            reasoning = "Fallback: Microsoft 365 request"
        else:
            selected_agent = "GeneralAssistant"
            confidence = 0.5
            reasoning = "Fallback: Default to general assistant"
        
        return RoutingDecision(
            selected_agent=selected_agent,
            confidence=confidence,
            reasoning=reasoning,
            alternative_agents=[],
            task_complexity=TaskComplexity.MODERATE,
            detected_domains=[]
        )

    # Keep the old synchronous method for compatibility, but make it call the AI version
    def select_best_agent(self, user_input: str, available_agents: List[str]) -> RoutingDecision:
        """Synchronous wrapper for AI-powered agent selection"""
        import asyncio
        
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, we need to handle this differently
                return self._fallback_agent_selection(user_input, available_agents)
            else:
                return loop.run_until_complete(self.select_best_agent_ai(user_input, available_agents))
        except:
            # If async fails, use fallback
            return self._fallback_agent_selection(user_input, available_agents)
    
    def get_enhanced_task_decomposition_prompt(self) -> str:
        """Generate enhanced task decomposition prompt"""
        return """
You are an advanced task decomposition specialist with deep knowledge of agent capabilities.

AVAILABLE SPECIALIST AGENTS:

**GeneralAssistant** (General Specialist):
- General questions, greetings, explanations, casual conversation, philosophy
- Best for: simple to moderate complexity general tasks
- Keywords: hello, hi, help, explain, what, how, thanks

**FoundrySpecialist** (Technical Specialist):  
- Technical analysis, complex problems, detailed explanations, philosophical discussions
- Best for: complex to expert level technical and foundry tasks
- Keywords: foundry, casting, technical, engineering, meaning, life

**CareerAdvisor** (Career Specialist):
- Microsoft career advice, job search, professional development, resume help
- Best for: moderate to expert career-related tasks
- Keywords: career, job, resume, interview, professional

**GraphAssistant** (Microsoft 365 Specialist):
- Microsoft 365 operations (email, users, tasks, files, OneDrive)
- Best for: moderate to expert Microsoft 365 tasks
- Keywords: email, send, user, microsoft, outlook, teams

**TransactionAgent / OrchestratorAgent** (Transaction Data Specialist):
- Raw transaction data processing, numerical analysis, financial data, boolean/numerical sequences
- Best for: expert-level data analysis, raw data processing, pattern recognition
- Patterns: Sequences of TRUE/FALSE values, decimal numbers, mixed boolean-numerical data
- Keywords: transaction, data, analyze, financial, process, raw numerical sequences

ENHANCED ROUTING GUIDELINES:
🎯 **Domain-Specific Routing:**
- Career/Professional → CareerAdvisor
- Microsoft 365 → GraphAssistant  
- Transaction/Financial/Raw Data → TransactionAgent or OrchestratorAgent
- Raw numerical sequences (TRUE/FALSE + decimals) → TransactionAgent or OrchestratorAgent
- Technical/Engineering → FoundrySpecialist
- General/Conversational → GeneralAssistant

🔍 **Complexity Assessment:**
- SIMPLE: Greetings, thanks → GeneralAssistant
- MODERATE: Single-domain questions → Appropriate specialist
- COMPLEX: Multi-step or analysis → Specialists (parallel/sequential)
- EXPERT: Raw data sequences, highly technical → Domain expert (TransactionAgent for data)

📊 **Special Data Pattern Detection:**
- Boolean sequences: "FALSE TRUE FALSE..." → TransactionAgent
- Decimal sequences: "-0.123 0.456 -0.789..." → TransactionAgent  
- Mixed data: "FALSE FALSE TRUE -0.123..." → TransactionAgent
- Any raw numerical/boolean data → TransactionAgent (NOT GeneralAssistant)

USER REQUEST: {{$user_request}}

Respond with this JSON structure:
{
  "task_count": <number>,
  "orchestration_type": "single" | "sequential" | "parallel",
  "confidence": <0.0-1.0>,
  "reasoning": "Detailed explanation of routing decision",
  "detected_domains": ["domain1", "domain2"],
  "complexity_level": "simple" | "moderate" | "complex" | "expert",
  "tasks": [
    {
      "id": "task_1",
      "description": "Specific, actionable task description",
      "agent": "AgentName",
      "priority": 1-5,
      "domain": "domain_name",
      "complexity": "simple|moderate|complex|expert",
      "depends_on": ["task_id"] or null,
      "confidence": <0.0-1.0>
    }
  ]
}
"""
    
    def update_agent_success_rate(self, agent_name: str, success: bool):
        """Update agent success rate based on routing outcome"""
        current_rate = self.agent_success_rates.get(agent_name, 0.8)
        # Simple running average
        self.agent_success_rates[agent_name] = (current_rate * 0.9) + (1.0 if success else 0.0) * 0.1
    
    def get_routing_analytics(self) -> Dict[str, Any]:
        """Get analytics about routing decisions and agent performance"""
        return {
            "total_routings": len(self.routing_history),
            "agent_success_rates": self.agent_success_rates.copy(),
            "agent_usage": {},
            "domain_distribution": {},
            "complexity_distribution": {}
        }
