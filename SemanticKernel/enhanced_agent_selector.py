#!/usr/bin/env pyfrom enum import Enum, IntEnum


class TaskComplexity(IntEnum):
    """Task complexity levels for agent selection - orderable by value"""
    SIMPLE = 1              # Basic greetings, simple questions
    MODERATE = 2            # Single-domain questions requiring some expertise
    COMPLEX = 3             # Multi-step or specialized knowledge required
    EXPERT = 4              # Highly specialized or technical
Enhanced Agent Selection System for Azure Foundry Orchestration

This module provides sophisticated agent selection capabilities including:
- Detailed agent capability profiles
- Multi-step reasoning for complex routing decisions
- Confidence scoring and fallback strategies
- Context-aware routing based on conversation history
- Learning from routing success/failure patterns
"""

import json
import re
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class TaskComplexity(Enum):
    """Task complexity levels for agent selection"""
    SIMPLE = "simple"           # Basic greetings, simple questions
    MODERATE = "moderate"       # Single-domain questions requiring some expertise
    COMPLEX = "complex"         # Multi-step or specialized knowledge required
    EXPERT = "expert"          # Highly specialized or technical


class DomainType(Enum):
    """Domain categories for agent specialization"""
    GENERAL = "general"
    TECHNICAL = "technical"
    CAREER = "career"
    MICROSOFT365 = "microsoft365"
    TRANSACTION = "transaction"
    FOUNDRY = "foundry"
    PHILOSOPHY = "philosophy"
    DATA_ANALYSIS = "data_analysis"


@dataclass
class AgentCapability:
    """Detailed capability profile for an agent"""
    domain: DomainType
    skill_level: int  # 1-10 scale
    keywords: List[str]
    examples: List[str]
    max_complexity: TaskComplexity
    confidence_threshold: float = 0.7


@dataclass
class AgentProfile:
    """Comprehensive agent profile with detailed capabilities"""
    name: str
    description: str
    primary_domain: DomainType
    capabilities: List[AgentCapability]
    personality_traits: List[str] = field(default_factory=list)
    input_patterns: List[str] = field(default_factory=list)
    exclusions: List[str] = field(default_factory=list)  # What this agent should NOT handle
    routing_history: List[Dict] = field(default_factory=list)
    success_rate: float = 0.8


@dataclass
class RoutingDecision:
    """Detailed routing decision with confidence and reasoning"""
    selected_agent: str
    confidence: float
    reasoning: str
    alternative_agents: List[Tuple[str, float]]  # (agent_name, confidence)
    task_complexity: TaskComplexity
    detected_domains: List[DomainType]


class EnhancedAgentSelector:
    """
    Enhanced agent selection system with sophisticated routing logic
    """
    
    def __init__(self):
        """Initialize the enhanced agent selector"""
        self.agent_profiles = self._initialize_agent_profiles()
        self.routing_history = []
        self.domain_keywords = self._initialize_domain_keywords()
        
    def _initialize_agent_profiles(self) -> Dict[str, AgentProfile]:
        """Initialize detailed agent profiles"""
        
        profiles = {}
        
        # GeneralAssistant Profile
        profiles["GeneralAssistant"] = AgentProfile(
            name="GeneralAssistant",
            description="Versatile conversational AI specializing in general assistance, explanations, and casual conversation",
            primary_domain=DomainType.GENERAL,
            capabilities=[
                AgentCapability(
                    domain=DomainType.GENERAL,
                    skill_level=9,
                    keywords=["hello", "hi", "thanks", "help", "explain", "what", "how", "why", "general"],
                    examples=[
                        "Hi there! How can I help?",
                        "Can you explain this concept?",
                        "What does this mean?",
                        "Thank you for your help"
                    ],
                    max_complexity=TaskComplexity.MODERATE,
                    confidence_threshold=0.8
                ),
                AgentCapability(
                    domain=DomainType.PHILOSOPHY,
                    skill_level=6,
                    keywords=["meaning", "life", "philosophy", "existence", "purpose", "ethics", "moral"],
                    examples=[
                        "What is the meaning of life?",
                        "What's the purpose of existence?",
                        "Tell me about philosophy"
                    ],
                    max_complexity=TaskComplexity.COMPLEX,
                    confidence_threshold=0.6
                )
            ],
            personality_traits=["friendly", "helpful", "conversational", "adaptive"],
            input_patterns=[
                r"^(hi|hello|hey|greetings)",
                r"(thank you|thanks|appreciate)",
                r"(what|how|why|explain)(?!.*career|email|transaction|foundry)",
                r"(help|assist|support)(?!.*specific domain)"
            ],
            exclusions=["career advice", "email operations", "transaction analysis", "technical foundry"]
        )
        
        # FoundrySpecialist Profile  
        profiles["FoundrySpecialist"] = AgentProfile(
            name="FoundrySpecialist",
            description="Technical expert in foundry operations, metallurgy, manufacturing processes, and advanced engineering analysis",
            primary_domain=DomainType.FOUNDRY,
            capabilities=[
                AgentCapability(
                    domain=DomainType.FOUNDRY,
                    skill_level=10,
                    keywords=["foundry", "casting", "metallurgy", "alloy", "molten", "defect", "porosity", "sand"],
                    examples=[
                        "Analyze casting defects in aluminum",
                        "Optimize foundry process parameters",
                        "Explain sand casting techniques"
                    ],
                    max_complexity=TaskComplexity.EXPERT,
                    confidence_threshold=0.9
                ),
                AgentCapability(
                    domain=DomainType.TECHNICAL,
                    skill_level=9,
                    keywords=["technical", "engineering", "analysis", "optimization", "simulation", "CAD"],
                    examples=[
                        "Technical analysis of manufacturing process",
                        "Engineering optimization problem",
                        "Complex technical explanation"
                    ],
                    max_complexity=TaskComplexity.EXPERT,
                    confidence_threshold=0.8
                ),
                AgentCapability(
                    domain=DomainType.PHILOSOPHY,
                    skill_level=8,
                    keywords=["meaning", "life", "philosophy", "deep", "purpose", "wisdom"],
                    examples=[
                        "Philosophical discussion about engineering ethics",
                        "Deep thoughts on purpose and meaning",
                        "Technical philosophy integration"
                    ],
                    max_complexity=TaskComplexity.COMPLEX,
                    confidence_threshold=0.7
                )
            ],
            personality_traits=["analytical", "detail-oriented", "technical", "thoughtful"],
            input_patterns=[
                r"(foundry|casting|metallurg|alloy)",
                r"(technical.*analysis|engineering.*problem)",
                r"(meaning.*life|philosophy.*technical)"
            ],
            exclusions=["career advice", "email operations", "basic greetings only"]
        )
        
        # CareerAdvisor Profile
        profiles["CareerAdvisor"] = AgentProfile(
            name="CareerAdvisor", 
            description="Microsoft career development specialist providing comprehensive career guidance, job search strategies, and professional development",
            primary_domain=DomainType.CAREER,
            capabilities=[
                AgentCapability(
                    domain=DomainType.CAREER,
                    skill_level=10,
                    keywords=["career", "job", "resume", "interview", "professional", "development", "hiring"],
                    examples=[
                        "Help me improve my resume",
                        "Career advancement strategies",
                        "Job interview preparation tips"
                    ],
                    max_complexity=TaskComplexity.EXPERT,
                    confidence_threshold=0.9
                ),
                AgentCapability(
                    domain=DomainType.GENERAL,
                    skill_level=7,
                    keywords=["advice", "guidance", "help", "support", "mentoring"],
                    examples=[
                        "I need career guidance",
                        "What should I do professionally?",
                        "Help with professional decisions"
                    ],
                    max_complexity=TaskComplexity.COMPLEX,
                    confidence_threshold=0.6
                )
            ],
            personality_traits=["supportive", "motivational", "professional", "insightful"],
            input_patterns=[
                r"(career|job|resume|interview|professional)",
                r"(work.*advice|job.*search|hire|hiring)",
                r"(cv|curriculum.*vitae|portfolio)"
            ],
            exclusions=["technical analysis", "email operations", "transaction data"]
        )
        
        # GraphAssistant Profile
        profiles["GraphAssistant"] = AgentProfile(
            name="GraphAssistant",
            description="Microsoft 365 operations specialist with direct Graph API access for email, users, files, and productivity tasks",
            primary_domain=DomainType.MICROSOFT365,
            capabilities=[
                AgentCapability(
                    domain=DomainType.MICROSOFT365,
                    skill_level=10,
                    keywords=["email", "send", "user", "microsoft", "365", "outlook", "onedrive", "teams"],
                    examples=[
                        "Send email to john@company.com",
                        "Find user Sarah Smith",
                        "Create todo list for project"
                    ],
                    max_complexity=TaskComplexity.EXPERT,
                    confidence_threshold=0.9
                ),
                AgentCapability(
                    domain=DomainType.DATA_ANALYSIS,
                    skill_level=6,
                    keywords=["list", "search", "find", "get", "retrieve", "organize"],
                    examples=[
                        "List all users in organization",
                        "Search for files containing project",
                        "Get my recent emails"
                    ],
                    max_complexity=TaskComplexity.MODERATE,
                    confidence_threshold=0.7
                )
            ],
            personality_traits=["efficient", "organized", "systematic", "reliable"],
            input_patterns=[
                r"(send.*email|email.*to|message.*to)",
                r"(find.*user|search.*user|list.*user)",
                r"(create.*todo|todo.*list|task.*list)",
                r"(onedrive|sharepoint|teams|outlook)"
            ],
            exclusions=["career advice", "foundry operations", "philosophical discussions"]
        )
        
        # Azure AI Agent Profile (Transaction Processing)
        profiles["Azure AI Agent"] = AgentProfile(
            name="Azure AI Agent",
            description="Specialized transaction data analysis agent powered by Azure AI for numerical analysis, financial data processing, and quantitative insights",
            primary_domain=DomainType.TRANSACTION,
            capabilities=[
                AgentCapability(
                    domain=DomainType.TRANSACTION,
                    skill_level=10,
                    keywords=["transaction", "financial", "data", "analysis", "numbers", "calculate", "amount"],
                    examples=[
                        "Analyze transaction patterns",
                        "Process financial data",
                        "Calculate transaction metrics"
                    ],
                    max_complexity=TaskComplexity.EXPERT,
                    confidence_threshold=0.9
                ),
                AgentCapability(
                    domain=DomainType.DATA_ANALYSIS,
                    skill_level=9,
                    keywords=["analyze", "process", "compute", "metrics", "statistics", "report"],
                    examples=[
                        "Generate data analysis report",
                        "Compute statistical metrics",
                        "Process numerical datasets"
                    ],
                    max_complexity=TaskComplexity.EXPERT,
                    confidence_threshold=0.8
                )
            ],
            personality_traits=["analytical", "precise", "data-driven", "quantitative"],
            input_patterns=[
                r"(transaction|financial|money|payment)",
                r"(analyze.*data|data.*analysis|process.*data)",
                r"(calculate|compute|metrics|statistics)"
            ],
            exclusions=["career advice", "email operations", "general conversation"]
        )
        
        return profiles
        
    def _initialize_domain_keywords(self) -> Dict[DomainType, List[str]]:
        """Initialize domain-specific keyword mappings"""
        return {
            DomainType.GENERAL: ["hello", "hi", "help", "explain", "what", "how", "thanks"],
            DomainType.CAREER: ["career", "job", "resume", "interview", "professional", "work"],
            DomainType.MICROSOFT365: ["email", "send", "user", "microsoft", "outlook", "teams"],
            DomainType.TRANSACTION: ["transaction", "financial", "money", "data", "analysis"],
            DomainType.FOUNDRY: ["foundry", "casting", "metallurgy", "technical", "engineering"],
            DomainType.PHILOSOPHY: ["meaning", "life", "philosophy", "purpose", "existence"]
        }
    
    def analyze_user_intent(self, user_input: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Analyze user intent with multi-step reasoning
        
        Args:
            user_input: The user's request
            context: Optional conversation context
            
        Returns:
            Intent analysis with domains, complexity, and keywords
        """
        user_lower = user_input.lower()
        
        # Step 1: Detect domains
        detected_domains = []
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in user_lower)
            if score > 0:
                domain_scores[domain] = score / len(keywords)
                detected_domains.append(domain)
        
        # Step 2: Determine complexity
        complexity_indicators = {
            TaskComplexity.SIMPLE: ["hi", "hello", "thanks", "thank you"],
            TaskComplexity.MODERATE: ["help", "explain", "what", "how"],
            TaskComplexity.COMPLEX: ["analyze", "process", "create", "find"],
            TaskComplexity.EXPERT: ["optimize", "technical", "advanced", "specialized"]
        }
        
        complexity = TaskComplexity.SIMPLE
        for level, indicators in complexity_indicators.items():
            if any(indicator in user_lower for indicator in indicators):
                if level.value > complexity.value:  # Take the highest complexity found
                    complexity = level
        
        # Step 3: Extract key entities
        entities = {
            "email_addresses": re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', user_input),
            "names": re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', user_input),
            "numbers": re.findall(r'\b\d+(?:\.\d+)?\b', user_input)
        }
        
        return {
            "detected_domains": detected_domains,
            "domain_scores": domain_scores,
            "complexity": complexity,
            "entities": entities,
            "input_length": len(user_input),
            "question_words": len(re.findall(r'\b(what|how|why|when|where|who)\b', user_lower))
        }
    
    def calculate_agent_fitness(self, agent_name: str, intent_analysis: Dict[str, Any], user_input: str) -> Tuple[float, str]:
        """
        Calculate how well an agent fits the user request
        
        Args:
            agent_name: Name of the agent to evaluate
            intent_analysis: Output from analyze_user_intent
            user_input: Original user input
            
        Returns:
            (fitness_score, reasoning)
        """
        if agent_name not in self.agent_profiles:
            return 0.0, f"Agent {agent_name} not found in profiles"
        
        profile = self.agent_profiles[agent_name]
        user_lower = user_input.lower()
        
        total_score = 0.0
        reasoning_parts = []
        
        # Domain alignment (40% weight)
        domain_score = 0.0
        for domain in intent_analysis["detected_domains"]:
            for capability in profile.capabilities:
                if capability.domain == domain:
                    domain_score = max(domain_score, capability.skill_level / 10.0)
                    reasoning_parts.append(f"Strong {domain.value} domain match (skill: {capability.skill_level}/10)")
        
        total_score += domain_score * 0.4
        
        # Keyword matching (25% weight)
        keyword_score = 0.0
        matched_keywords = []
        for capability in profile.capabilities:
            for keyword in capability.keywords:
                if keyword in user_lower:
                    keyword_score += 0.1
                    matched_keywords.append(keyword)
        
        keyword_score = min(keyword_score, 1.0)
        total_score += keyword_score * 0.25
        
        if matched_keywords:
            reasoning_parts.append(f"Keyword matches: {', '.join(matched_keywords[:3])}")
        
        # Pattern matching (20% weight)
        pattern_score = 0.0
        for pattern in profile.input_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                pattern_score = 1.0
                reasoning_parts.append(f"Input pattern match: {pattern[:30]}...")
                break
        
        total_score += pattern_score * 0.2
        
        # Complexity handling (10% weight)
        complexity_score = 0.0
        max_complexity = max(cap.max_complexity.value for cap in profile.capabilities)
        if intent_analysis["complexity"].value <= max_complexity:
            complexity_score = 1.0
            reasoning_parts.append(f"Can handle {intent_analysis['complexity'].name.lower()} complexity")
        
        total_score += complexity_score * 0.1
        
        # Exclusion penalty (5% weight)
        exclusion_penalty = 0.0
        for exclusion in profile.exclusions:
            if any(word in user_lower for word in exclusion.split()):
                exclusion_penalty = 0.3
                reasoning_parts.append(f"Exclusion triggered: {exclusion}")
                break
        
        total_score -= exclusion_penalty * 0.05
        
        # Success rate bonus (5% weight)
        total_score += profile.success_rate * 0.05
        
        # Ensure score is between 0 and 1
        total_score = max(0.0, min(1.0, total_score))
        
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "No specific matches found"
        
        return total_score, reasoning
    
    def select_best_agent(self, user_input: str, available_agents: List[str], context: Optional[Dict] = None) -> RoutingDecision:
        """
        Select the best agent using sophisticated multi-step analysis
        
        Args:
            user_input: User's request
            available_agents: List of available agent names
            context: Optional conversation context
            
        Returns:
            RoutingDecision with selected agent and alternatives
        """
        # Step 1: Analyze intent
        intent_analysis = self.analyze_user_intent(user_input, context)
        
        # Step 2: Calculate fitness for each agent
        agent_scores = []
        for agent_name in available_agents:
            if agent_name in self.agent_profiles:
                score, reasoning = self.calculate_agent_fitness(agent_name, intent_analysis, user_input)
                agent_scores.append((agent_name, score, reasoning))
        
        # Step 3: Sort by score
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        
        if not agent_scores:
            # Fallback to GeneralAssistant
            return RoutingDecision(
                selected_agent="GeneralAssistant",
                confidence=0.5,
                reasoning="No agent profiles matched, using fallback",
                alternative_agents=[],
                task_complexity=intent_analysis["complexity"],
                detected_domains=intent_analysis["detected_domains"]
            )
        
        # Step 4: Select best agent
        best_agent, best_score, best_reasoning = agent_scores[0]
        
        # Step 5: Prepare alternatives
        alternatives = [(name, score) for name, score, _ in agent_scores[1:3]]  # Top 2 alternatives
        
        # Step 6: Build routing decision
        decision = RoutingDecision(
            selected_agent=best_agent,
            confidence=best_score,
            reasoning=best_reasoning,
            alternative_agents=alternatives,
            task_complexity=intent_analysis["complexity"],
            detected_domains=intent_analysis["detected_domains"]
        )
        
        # Step 7: Log decision for learning
        self.routing_history.append({
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "decision": decision,
            "intent_analysis": intent_analysis
        })
        
        return decision
    
    def get_enhanced_task_decomposition_prompt(self) -> str:
        """Generate enhanced task decomposition prompt with detailed agent capabilities"""
        
        agent_descriptions = []
        for name, profile in self.agent_profiles.items():
            
            # Build capability summary
            capabilities = []
            for cap in profile.capabilities:
                capabilities.append(f"  • {cap.domain.value.title()}: Level {cap.skill_level}/10 - {', '.join(cap.keywords[:5])}")
            
            agent_desc = f"""
**{name}** ({profile.primary_domain.value.title()} Specialist):
{profile.description}

Core Capabilities:
{chr(10).join(capabilities)}

Personality: {', '.join(profile.personality_traits)}
Best For: {profile.primary_domain.value} tasks up to {max(cap.max_complexity for cap in profile.capabilities).value} complexity
Excludes: {', '.join(profile.exclusions) if profile.exclusions else 'None'}
"""
            agent_descriptions.append(agent_desc)
        
        return f"""
You are an advanced task decomposition specialist with deep knowledge of agent capabilities.

AVAILABLE SPECIALIST AGENTS:
{'='*80}
{chr(10).join(agent_descriptions)}

ENHANCED ROUTING GUIDELINES:
🎯 **Domain-Specific Routing:**
- Career/Professional → CareerAdvisor (resume, job search, interviews, professional development)
- Microsoft 365 → GraphAssistant (email, users, OneDrive, Teams operations)  
- Transaction/Financial → Azure AI Agent (data analysis, numerical processing)
- Technical/Engineering → FoundrySpecialist (foundry, metallurgy, complex analysis)
- General/Conversational → GeneralAssistant (greetings, explanations, philosophy)

🔍 **Complexity Assessment:**
- SIMPLE: Greetings, thanks, basic questions → GeneralAssistant
- MODERATE: Single-domain questions → Appropriate specialist
- COMPLEX: Multi-step or cross-domain → Multiple specialists (parallel/sequential)
- EXPERT: Highly technical or specialized → Domain expert

🎭 **Multi-Task Detection:**
- Independent topics → PARALLEL orchestration
- Dependent tasks → SEQUENTIAL orchestration  
- Single focused topic → SINGLE orchestration

USER REQUEST: {{$user_request}}

Analyze the request and respond with this enhanced JSON structure:
{{
  "task_count": <number>,
  "orchestration_type": "single" | "sequential" | "parallel",
  "confidence": <0.0-1.0>,
  "reasoning": "Detailed explanation of routing decision",
  "detected_domains": ["domain1", "domain2"],
  "complexity_level": "simple" | "moderate" | "complex" | "expert",
  "tasks": [
    {{
      "id": "task_1",
      "description": "Specific, actionable task description",
      "agent": "AgentName",
      "priority": 1-5,
      "domain": "domain_name",
      "complexity": "simple|moderate|complex|expert",
      "depends_on": ["task_id"] or null,
      "confidence": <0.0-1.0>
    }}
  ]
}}

ROUTING EXAMPLES:
• "Hi" → GeneralAssistant (single, simple)
• "Career advice" → CareerAdvisor (single, moderate)  
• "Send email to john@company.com" → GraphAssistant (single, moderate)
• "Career advice AND meaning of life" → CareerAdvisor + FoundrySpecialist (parallel, complex)
• "Find user John AND send him email about project" → GraphAssistant x2 (sequential, moderate)
• "Analyze transaction data AND optimize foundry process" → Azure AI Agent + FoundrySpecialist (parallel, expert)
"""

    def update_agent_success_rate(self, agent_name: str, success: bool):
        """Update agent success rate based on routing outcome"""
        if agent_name in self.agent_profiles:
            profile = self.agent_profiles[agent_name]
            
            # Simple running average update
            history_len = len(profile.routing_history)
            if history_len == 0:
                profile.success_rate = 1.0 if success else 0.0
            else:
                profile.success_rate = (profile.success_rate * history_len + (1.0 if success else 0.0)) / (history_len + 1)
            
            # Log the outcome
            profile.routing_history.append({
                "timestamp": datetime.now().isoformat(),
                "success": success
            })
            
            # Keep only last 100 entries
            if len(profile.routing_history) > 100:
                profile.routing_history = profile.routing_history[-100:]

    def get_routing_analytics(self) -> Dict[str, Any]:
        """Get analytics about routing decisions and agent performance"""
        
        analytics = {
            "total_routings": len(self.routing_history),
            "agent_usage": {},
            "domain_distribution": {},
            "complexity_distribution": {},
            "agent_success_rates": {}
        }
        
        for agent_name, profile in self.agent_profiles.items():
            analytics["agent_success_rates"][agent_name] = profile.success_rate
            analytics["agent_usage"][agent_name] = len(profile.routing_history)
        
        for entry in self.routing_history:
            decision = entry["decision"]
            intent = entry["intent_analysis"]
            
            # Track domain distribution
            for domain in intent["detected_domains"]:
                domain_name = domain.value if hasattr(domain, 'value') else str(domain)
                analytics["domain_distribution"][domain_name] = analytics["domain_distribution"].get(domain_name, 0) + 1
            
            # Track complexity distribution
            complexity = intent["complexity"].value if hasattr(intent["complexity"], 'value') else str(intent["complexity"])
            analytics["complexity_distribution"][complexity] = analytics["complexity_distribution"].get(complexity, 0) + 1
        
        return analytics
