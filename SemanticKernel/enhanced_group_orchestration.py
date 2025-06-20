#!/usr/bin/env python3
"""
Enhanced Azure Foundry Group Chat Orchestration with Human-in-the-Loop
Based on Semantic Kernel Group Chat Orchestration Framework

This implementation demonstrates the User → Manager → Agent A → Manager → Agent B → Manager → ... → User pattern
for intelligent conversation flows with data validation and multi-agent collaboration.

Example workflow:
1. User: "I want to do transaction analysis"
2. Manager: Routes to TransactionAgent
3. TransactionAgent: "I need transaction data to analyze" 
4. Manager: Routes back to user for clarification
5. User: Provides data or more context
6. Manager: Routes to appropriate agent(s) for processing
7. Continue until task is complete
"""

import asyncio
import sys
import os
import re
import traceback
from typing import List, Dict, Any, Optional

# Add compatibility for different Python versions
if sys.version_info >= (3, 12):
    from typing import override  # pragma: no cover
else:
    from typing_extensions import override  # pragma: no cover

from azure.identity import DefaultAzureCredential
from semantic_kernel.agents import Agent, ChatCompletionAgent, GroupChatOrchestration
from semantic_kernel.agents.orchestration.group_chat import BooleanResult, GroupChatManager, MessageResult, StringResult
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.contents import AuthorRole, ChatHistory, ChatMessageContent
from semantic_kernel.functions import KernelArguments
from semantic_kernel.kernel import Kernel
from semantic_kernel.prompt_template import KernelPromptTemplate, PromptTemplateConfig

# Import custom agents from your existing code
from copilot_studio_agent import CopilotAgent
from copilot_studio_agent_thread import CopilotAgentThread
from copilot_studio_channel import CopilotStudioAgentChannel
from copilot_studio_message_content import CopilotMessageContent
from directline_client import DirectLineClient

# Add src folder to Python path for Microsoft Graph integration
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Microsoft Graph imports from src folder
try:
    from graph_agent_plugin import MicrosoftGraphPlugin
    from graph_agent import GraphAgent
    GRAPH_INTEGRATION_AVAILABLE = True
    print("✅ Microsoft Graph integration loaded successfully")
except ImportError as e:
    print(f"⚠️ Microsoft Graph integration not available: {e}")
    GRAPH_INTEGRATION_AVAILABLE = False

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


class HumanInTheLoopGroupChatManager(GroupChatManager):
    """
    Enhanced Group Chat Manager implementing Human-in-the-Loop pattern.
    
    This manager demonstrates the User → Manager → Agent A → Manager → Agent B → Manager → ... → User workflow
    by intelligently determining when to request user input vs continuing agent conversation.
    
    Key Features:
    - Intelligent agent selection based on context and capabilities
    - Automatic data validation and requirement checking
    - Human intervention when agents need additional information
    - Multi-turn conversation flow with proper state management
    """

    # Template-based prompts (inspired by the sample documentation)
    termination_prompt: str = (
        "You are an intelligent conversation mediator analyzing a multi-agent conversation about '{{$topic}}'. "
        "Your role is to determine if the conversation has reached a natural conclusion.\n\n"
        "Consider these factors:\n"
        "1. Has the user's request been adequately addressed with meaningful results?\n"
        "2. Is the conversation becoming repetitive or circular without progress?\n"
        "3. Are agents repeatedly requesting the same information without processing provided data?\n"
        "4. Has sufficient value been provided to the user?\n"
        "5. Would continuing add meaningful value?\n\n"
        "IMPORTANT GUIDANCE:\n"
        "- For simple requests like greetings, terminate after 1-2 exchanges.\n"
        "- For complex requests, ensure the task is complete or substantial progress is made.\n"
        "- DO NOT terminate if the user just provided data that needs to be analyzed.\n"
        "- DO NOT terminate if agents are actively processing user input.\n"
        "- DO terminate if agents are stuck in loops requesting the same data repeatedly.\n"
        "- DO terminate if the conversation has accomplished its stated goal.\n\n"
        "Respond with True to end the conversation, False to continue."
    )

    selection_prompt: str = (
        "You are an intelligent conversation manager routing between specialized agents for '{{$topic}}'.\n\n"
        "Available agents:\n{{$participants}}\n\n"
        "Recent conversation context:\n{{$context}}\n\n"
        "ROUTING RULES:\n"
        "- Route to TransactionAgent for financial data, fraud analysis, numerical processing\n"
        "- Route to GraphAssistant for Microsoft 365, email, Teams operations\n" 
        "- Route to CareerAdvisor for job guidance, career development\n"
        "- Route to GeneralAssistant for explanations, clarifications, general help\n\n"
        "IMPORTANT: If the same agent has responded 2+ times requesting data without progress,\n"
        "route to GeneralAssistant to break the loop and guide the user.\n\n"
        "Select the most appropriate agent name."
    )

    result_filter_prompt: str = (
        "You are summarizing a completed multi-agent conversation about '{{$topic}}'. "
        "The conversation involved a user interacting with specialized AI assistants through an intelligent manager. "
        "Please provide a comprehensive summary that includes:\n"
        "1. What the user requested\n"
        "2. Which agents were involved and their contributions\n"
        "3. Key results or findings\n"
        "4. Any recommended next steps\n"
        "5. Overall outcome of the conversation"
    )

    def __init__(self, service: ChatCompletionClientBase, max_rounds: int = 15, 
                 human_response_function=None, **kwargs):
        """Initialize the Human-in-the-Loop Group Chat Manager"""
        super().__init__(service=service, max_rounds=max_rounds, **kwargs)
        self._chat_service = service
        self._human_response_function = human_response_function
        self._conversation_state = {
            "pending_user_input": False,
            "last_agent_request": None,
            "data_requirements": [],
            "conversation_topic": "multi-agent assistance"
        }
    
    async def _render_prompt(self, prompt: str, arguments: KernelArguments) -> str:
        """Helper to render a prompt with arguments (like the sample)."""
        prompt_template_config = PromptTemplateConfig(template=prompt)
        prompt_template = KernelPromptTemplate(prompt_template_config=prompt_template_config)
        return await prompt_template.render(Kernel(), arguments=arguments)
    
    @override
    async def should_request_user_input(self, chat_history: ChatHistory) -> BooleanResult:
        """
        Determine if user input is needed based on agent responses and conversation state.
        
        This is the core of the human-in-the-loop pattern - the manager decides when
        the conversation should return to the user vs continue between agents.
        """
        if self._human_response_function is None:
            return BooleanResult(
                result=False,
                reason="No human response function configured."
            )
        
        if len(chat_history.messages) == 0:
            return BooleanResult(
                result=False,
                reason="No agents have spoken yet."
            )
        
        # Get the last assistant message
        last_message = None
        for message in reversed(chat_history.messages):
            if message.role == AuthorRole.ASSISTANT:
                last_message = message
                break
        
        if not last_message:
            return BooleanResult(
                result=False,
                reason="No assistant response found."
            )
        
        # Analyze the last agent response to determine if user input is needed
        content = str(last_message.content).lower()
        agent_name = last_message.name or ""
        
        print(f"🔍 Checking if user input needed for {agent_name}: {content[:100]}...")
        
        # Check for explicit requests for user input
        input_request_patterns = [
            "need more information",
            "please provide",
            "could you provide",
            "what data",
            "need transaction data",
            "please upload",
            "missing required",
            "need additional",
            "please specify",
            "could you clarify",
            "need clarification",
            "what type of",
            "please share",
            "require input",
            "i need",
            "can you provide",
            "would you mind",
            "let me know",
            "tell me more",
            "share with me",
            # Enhanced patterns for "not enough data" scenarios
            "not enough data",
            "insufficient data to",
            "need more data to",
            "cannot analyze without",
            "unable to proceed without",
            "missing required data",
            "no data provided",
            "please provide the data",
            "i need the transaction data",
            "require more details to",
            "need additional information to",
            "data appears to be missing",
            "incomplete data set"
        ]
        
        for pattern in input_request_patterns:
            if pattern in content:
                self._conversation_state["pending_user_input"] = True
                self._conversation_state["last_agent_request"] = agent_name
                print(f"✅ User input needed - Agent {agent_name} requested: {pattern}")
                return BooleanResult(
                    result=True,
                    reason=f"Agent {agent_name} explicitly requested user input: {pattern}"
                )
        
        # Check for error conditions that need user intervention
        error_patterns = [
            "error",
            "failed",
            "cannot process",
            "unable to",
            "insufficient data",
            "invalid format",
            "not enough information",
            # Enhanced error patterns for better detection
            "cannot analyze",
            "analysis failed",
            "processing failed",
            "no valid data",
            "data format invalid",
            "unable to parse",
            "cannot proceed",
            "incomplete information",
            "missing critical data",
            "data validation failed"
        ]
        
        for pattern in error_patterns:
            if pattern in content:
                self._conversation_state["pending_user_input"] = True
                self._conversation_state["last_agent_request"] = agent_name
                print(f"⚠️ User input needed - Agent {agent_name} encountered issue: {pattern}")
                return BooleanResult(
                    result=True,
                    reason=f"Agent {agent_name} encountered an issue requiring user intervention"
                )
        
        # Check for incomplete analysis that might benefit from user clarification
        incomplete_patterns = [
            "preliminary analysis",
            "based on limited data",
            "with more information",
            "additional context would help",
            "to provide better analysis"
        ]
        
        for pattern in incomplete_patterns:
            if pattern in content:
                self._conversation_state["pending_user_input"] = True
                self._conversation_state["last_agent_request"] = agent_name
                print(f"📝 User input needed - Agent {agent_name} needs clarification: {pattern}")
                return BooleanResult(
                    result=True,
                    reason=f"Agent {agent_name} indicates analysis could be improved with user input"
                )
        
        # For transaction analysis, check if the agent produced minimal or unclear output
        if "transaction" in agent_name.lower() and len(content.strip()) < 100:
            self._conversation_state["pending_user_input"] = True
            self._conversation_state["last_agent_request"] = agent_name
            print(f"💰 User input needed - Transaction agent provided minimal output")
            return BooleanResult(
                result=True,
                reason="Transaction agent provided minimal output - user clarification may be needed"
            )
        
        # Reset pending state if no input needed
        self._conversation_state["pending_user_input"] = False
        print(f"✅ No user input needed - Agent {agent_name} response appears complete")
        
        return BooleanResult(
            result=False,
            reason="Agent response appears complete - continuing conversation flow"
        )
    
    @override
    async def should_terminate(self, chat_history: ChatHistory) -> BooleanResult:
        """
        Enhanced termination logic using template-based prompts like the sample.
        Uses AI to intelligently determine when conversations should end, with special
        handling for agent-specific patterns like "not enough data".
        """
        # Check parent termination conditions first
        should_terminate = await super().should_terminate(chat_history)
        if should_terminate.result:
            return should_terminate

        if not chat_history.messages:
            return BooleanResult(result=False, reason="No messages yet")

        # Quick termination checks for obvious cases
        user_messages = [msg for msg in chat_history.messages if msg.role == AuthorRole.USER]
        agent_messages = [msg for msg in chat_history.messages if msg.role == AuthorRole.ASSISTANT]
        
        # Handle simple greetings - terminate after first meaningful exchange
        if len(user_messages) == 1 and len(agent_messages) >= 1:
            first_user_message = user_messages[0].content.lower().strip()
            simple_greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
            
            # Check for simple greetings (exact match or very short)
            is_simple_greeting = (
                any(greeting == first_user_message for greeting in simple_greetings) or
                (any(greeting in first_user_message for greeting in simple_greetings) and len(first_user_message) < 20)
            )
            
            if is_simple_greeting:
                return BooleanResult(
                    result=True,
                    reason="Simple greeting exchange completed - conversation should end"
                )

        # CRITICAL FIX: Don't terminate if user just provided data for analysis
        if len(user_messages) >= 2:
            last_user_message = user_messages[-1].content.lower()
            
            # Check if user provided substantial data/content in their last message
            data_indicators = [
                "here's my data", "here is my data", "my transaction data", 
                "csv", "json", "data:", "amount", "date", "transaction", 
                "analysis", "analyze", "fraud", "risk", "financial"
            ]
            
            has_substantial_data = (
                len(last_user_message) > 50 or  # Long message likely contains data
                any(indicator in last_user_message for indicator in data_indicators) or
                any(char.isdigit() for char in last_user_message)  # Contains numbers
            )
            
            if has_substantial_data:
                print(f"📊 User provided substantial data - continuing conversation for analysis")
                return BooleanResult(
                    result=False,
                    reason="User provided substantial data - allowing analysis to proceed"
                )

        # ENHANCED: Agent-aware termination logic for unproductive conversations
        should_terminate_result = self._analyze_agent_productivity(chat_history, user_messages, agent_messages)
        if should_terminate_result.result:
            return should_terminate_result

        # Detect repetitive loops (same agent asking for data repeatedly)
        if len(agent_messages) >= 4:  # Increased threshold to be less aggressive
            recent_agent_contents = [str(msg.content).lower() for msg in agent_messages[-3:]]
            info_request_patterns = ["please provide", "need", "required", "upload", "share", "data"]
            
            info_requests = sum(1 for content in recent_agent_contents 
                              if any(pattern in content for pattern in info_request_patterns))
            
            if info_requests >= 3:  # Require 3 requests instead of 2
                return BooleanResult(
                    result=True,
                    reason="Multiple requests for additional information - ending to avoid loops"
                )

        # Use AI-based termination decision with template prompts (like the sample)
        try:
            # Create context for the AI decision
            context = "\n".join([
                f"{'User' if msg.role == AuthorRole.USER else msg.name or 'Agent'}: {str(msg.content)[:200]}"
                for msg in chat_history.messages[-6:]  # Last 6 messages for context
            ])

            # Create termination analysis history (similar to sample pattern)
            termination_history = ChatHistory()
            termination_history.messages.insert(
                0,
                ChatMessageContent(
                    role=AuthorRole.SYSTEM,
                    content=await self._render_prompt(
                        self.termination_prompt,
                        KernelArguments(topic=self._conversation_state.get("conversation_topic", "assistance"))
                    ),
                ),
            )
            
            # Add recent conversation context
            for msg in chat_history.messages[-4:]:  # Last 4 messages for AI analysis
                termination_history.add_message(msg)
            
            termination_history.add_message(
                ChatMessageContent(role=AuthorRole.USER, content="Determine if the discussion should end.")
            )

            response = await self._chat_service.get_chat_message_content(
                termination_history,
                settings=PromptExecutionSettings(
                    response_format=BooleanResult,
                    max_tokens=150,
                    temperature=0.1
                )
            )

            termination_with_reason = BooleanResult.model_validate_json(response.content)

            print("🔄 *********************")
            print(f"Should terminate: {termination_with_reason.result}")
            print(f"Reason: {termination_with_reason.reason}")
            print("*********************")

            return termination_with_reason

        except Exception as e:
            print(f"⚠️ Error in AI termination decision: {e}")
            # Fallback logic - terminate if conversation seems stuck
            if len(agent_messages) >= 5 and len(user_messages) == 1:
                return BooleanResult(
                    result=True,
                    reason="Fallback termination - too many agent responses without user interaction"
                )
            
            return BooleanResult(
                result=False,
                reason="Continuing due to termination analysis error"
            )
    
    @override
    async def select_next_agent(
        self,
        chat_history: ChatHistory,
        participant_descriptions: Dict[str, str],
    ) -> StringResult:
        """
        Enhanced agent selection using template-based prompts like the sample.
        Implements intelligent routing with loop prevention and proper user input handling.
        """
        if not chat_history.messages:
            # Default to GeneralAssistant for first interaction
            agent_names = list(participant_descriptions.keys())
            default_agent = "GeneralAssistant" if "GeneralAssistant" in agent_names else agent_names[0]
            return StringResult(
                result=default_agent,
                reason="Starting conversation with general assistant"
            )
        
        # Analyze conversation patterns and agent usage
        user_message = ""
        last_selected_agent = None
        agent_selection_count = {}
        recent_context = []
        
        # Track the last few messages to understand the flow
        last_user_message = None
        last_agent_message = None
        
        for message in chat_history.messages:
            if message.role == AuthorRole.USER:
                user_message = str(message.content)
                last_user_message = message
            elif message.role == AuthorRole.ASSISTANT:
                agent_name = message.name or "Assistant"
                content = str(message.content)
                recent_context.append(f"{agent_name}: {content[:150]}...")
                last_selected_agent = agent_name
                last_agent_message = message
                agent_selection_count[agent_name] = agent_selection_count.get(agent_name, 0) + 1

        print(f"🔍 Agent selection - Last agent: {last_selected_agent}, Pending user input: {self._conversation_state.get('pending_user_input', False)}")
        
        # CRITICAL: Handle case where user just provided input after agent requested it
        if (self._conversation_state.get("pending_user_input", False) and 
            last_user_message and last_agent_message and 
            self._conversation_state.get("last_agent_request")):
            
            # Check if the last user message comes after the last agent message (user just responded)
            user_msg_index = chat_history.messages.index(last_user_message)
            agent_msg_index = chat_history.messages.index(last_agent_message)
            
            if user_msg_index > agent_msg_index:
                # User just provided input, route back to the agent that requested it
                requesting_agent = self._conversation_state["last_agent_request"]
                if requesting_agent in participant_descriptions:
                    # Clear the pending state since we're routing back
                    self._conversation_state["pending_user_input"] = False
                    print(f"✅ User provided input - routing back to {requesting_agent}")
                    return StringResult(
                        result=requesting_agent,
                        reason=f"User provided requested input - continuing with {requesting_agent}"
                    )

        # Detect and prevent agent loops (key improvement from sample analysis)
        if (last_selected_agent and 
            agent_selection_count.get(last_selected_agent, 0) >= 2):
            
            last_agent_content = str(chat_history.messages[-1].content).lower()
            is_requesting_data = any(pattern in last_agent_content for pattern in 
                                   ["please provide", "need", "required", "upload", "data"])
            
            if is_requesting_data:
                # Switch to GeneralAssistant to break the loop and guide user
                fallback_agent = "GeneralAssistant" if "GeneralAssistant" in participant_descriptions else list(participant_descriptions.keys())[0]
                if fallback_agent != last_selected_agent:
                    print(f"🔄 Breaking agent loop - {last_selected_agent} repeatedly requesting data")
                    return StringResult(
                        result=fallback_agent,
                        reason=f"Breaking agent loop - {last_selected_agent} repeatedly requesting data, switching to {fallback_agent} for guidance"
                    )

        # Quick routing for specific data patterns (enhanced detection)
        has_numerical_data = any(char.isdigit() or char in ['-', '.', ','] for char in user_message)
        has_boolean_data = 'TRUE' in user_message.upper() or 'FALSE' in user_message.upper()
        data_keywords = ['analyze', 'data', 'transaction', 'fraud', 'risk', 'financial']
        has_data_request = any(keyword in user_message.lower() for keyword in data_keywords)
        
        if (has_numerical_data and has_boolean_data) or (has_data_request and has_numerical_data):
            # Route to transaction agent for data analysis
            transaction_agent = None
            for name in participant_descriptions.keys():
                if 'transaction' in name.lower() or 'orchestrator' in name.lower():
                    transaction_agent = name
                    break
            
            if transaction_agent and agent_selection_count.get(transaction_agent, 0) < 2:
                print(f"💰 Detected transaction/numerical data - routing to {transaction_agent}")
                return StringResult(
                    result=transaction_agent,
                    reason=f"Detected transaction/numerical data - routing to {transaction_agent}"
                )

        # Use AI-based selection with template prompts (like the sample)
        try:
            # Prepare context for AI decision
            context = "\n".join(recent_context[-4:])  # Last 4 agent responses for context
            
            # Create selection history using template pattern
            selection_history = ChatHistory()
            selection_history.messages.insert(
                0,
                ChatMessageContent(
                    role=AuthorRole.SYSTEM,
                    content=await self._render_prompt(
                        self.selection_prompt,
                        KernelArguments(
                            topic=self._conversation_state.get("conversation_topic", "assistance"),
                            participants="\n".join([f"{k}: {v}" for k, v in participant_descriptions.items()]),
                            context=context
                        )
                    ),
                ),
            )
            
            selection_history.add_message(
                ChatMessageContent(role=AuthorRole.USER, content="Select the most appropriate agent for this situation.")
            )

            response = await self._chat_service.get_chat_message_content(
                selection_history,
                settings=PromptExecutionSettings(
                    response_format=StringResult,
                    max_tokens=100,
                    temperature=0.1
                )
            )

            participant_name_with_reason = StringResult.model_validate_json(response.content)

            print("🔄 *********************")
            print(f"Next agent: {participant_name_with_reason.result}")
            print(f"Reason: {participant_name_with_reason.reason}")
            print("*********************")

            # Validate the selected agent exists
            if participant_name_with_reason.result in participant_descriptions:
                return participant_name_with_reason
            else:
                # Try partial matching
                for agent_name in participant_descriptions.keys():
                    if agent_name.lower() in participant_name_with_reason.result.lower():
                        return StringResult(
                            result=agent_name,
                            reason=f"Matched {agent_name} from AI suggestion: {participant_name_with_reason.result}"
                        )
                
                # Fallback to safe choice
                fallback_agent = "GeneralAssistant" if "GeneralAssistant" in participant_descriptions else list(participant_descriptions.keys())[0]
                return StringResult(
                    result=fallback_agent,
                    reason=f"Fallback selection - AI suggested unknown agent: {participant_name_with_reason.result}"
                )

        except Exception as e:
            print(f"⚠️ Error in AI agent selection: {e}")
            # Intelligent fallback based on conversation state
            if user_message and any(keyword in user_message.lower() for keyword in ['transaction', 'analyze', 'data']):
                transaction_agent = next((name for name in participant_descriptions.keys() 
                                        if 'transaction' in name.lower()), None)
                if transaction_agent:
                    return StringResult(result=transaction_agent, reason="Fallback to transaction agent for data analysis")
            
            # Final fallback
            fallback_agent = "GeneralAssistant" if "GeneralAssistant" in participant_descriptions else list(participant_descriptions.keys())[0]
            return StringResult(
                result=fallback_agent,
                reason="Fallback selection due to AI selection error"
            )
        
        # For simple greetings, don't over-analyze
        simple_greetings = ["hi", "hello", "hey", "good morning", "good afternoon"]
        if any(greeting in user_message.lower() for greeting in simple_greetings) and len(conversation_history) <= 2:
            general_agent = "GeneralAssistant" if "GeneralAssistant" in participant_descriptions else list(participant_descriptions.keys())[0]
            return StringResult(
                result=general_agent,
                reason="Simple greeting - routing to general assistant"
            )
        
        # Enhanced context analysis for agent selection
        context_analysis = "\n".join(conversation_history[-6:])  # Last 6 exchanges
        
        # Create sophisticated selection prompt
        selection_prompt = f"""
You are an intelligent conversation manager routing between specialized AI assistants.

AVAILABLE ASSISTANTS:
{self._format_participants_enhanced(participant_descriptions)}

CONVERSATION CONTEXT:
{context_analysis}

CURRENT USER REQUEST: {user_message}

ROUTING RULES (Priority Order):

1. TRANSACTION/FINANCIAL DATA - Route to TransactionAgent if:
   - Request mentions: transaction, financial, fraud, risk, analysis, banking, payment, money
   - User provides data for processing: CSV, JSON, numerical data, financial records
   - Need: data validation, anomaly detection, fraud analysis, risk assessment

2. CAREER/PROFESSIONAL - Route to CareerAdvisor if:
   - Request mentions: career, job, resume, interview, professional development
   - Need: career guidance, job search help, workplace advice

3. MICROSOFT 365/OFFICE - Route to GraphAssistant if:
   - Request mentions: email, Teams, OneDrive, Office, user management, Microsoft 365
   - Need: email operations, file management, user directory, Office automation

4. GENERAL/CLARIFICATION - Route to GeneralAssistant if:
   - General questions, casual conversation, explanations
   - Clarifying agent responses, facilitating between specialists
   - Providing context or bridging between different topics

INTELLIGENT ROUTING CONSIDERATIONS:
- If the user just provided data after an agent requested it, route back to the requesting agent
- If an agent completed analysis but user asks follow-up, route to the same agent
- If conversation needs clarification or explanation, route to GeneralAssistant
- If switching topics, route to the appropriate specialist

Respond with ONLY the exact assistant name.
"""

        # Create analysis history
        analysis_history = ChatHistory()
        analysis_history.add_message(ChatMessageContent(
            role=AuthorRole.SYSTEM,
            content=selection_prompt
        ))
        
        analysis_history.add_message(ChatMessageContent(
            role=AuthorRole.USER,
            content="Which assistant should handle this request?"
        ))

        try:
            # Use AI to select the best agent
            response = await self._chat_service.get_chat_message_content(
                analysis_history,
                settings=PromptExecutionSettings(max_tokens=50, temperature=0.1)
            )
            
            selected_agent = str(response.content).strip()
            
            # Validate selection and provide fallback
            if selected_agent in participant_descriptions:
                return StringResult(
                    result=selected_agent,
                    reason=f"Intelligent routing selected {selected_agent} based on context analysis"
                )
            else:
                # Try partial matching
                for agent_name in participant_descriptions.keys():
                    if agent_name.lower() in selected_agent.lower():
                        return StringResult(
                            result=agent_name,
                            reason=f"Matched {agent_name} from AI suggestion: {selected_agent}"
                        )
                
                # Final fallback
                fallback_agent = list(participant_descriptions.keys())[0]
                return StringResult(
                    result=fallback_agent,
                    reason=f"Fallback to {fallback_agent} - AI selection unclear: {selected_agent}"
                )
                
        except Exception as e:
            print(f"⚠️ Agent selection failed: {e}")
            fallback_agent = list(participant_descriptions.keys())[0]
            return StringResult(
                result=fallback_agent,
                reason=f"Error fallback to {fallback_agent}"
            )
    
    @override
    async def filter_results(self, chat_history: ChatHistory) -> MessageResult:
        """
        Enhanced result filtering using template-based prompts like the sample.
        Provides intelligent conversation summaries and outcomes.
        """
        if not chat_history.messages:
            raise RuntimeError("No messages in chat history")

        try:
            # Create result filtering history using template pattern (like the sample)
            filter_history = ChatHistory()
            filter_history.messages.insert(
                0,
                ChatMessageContent(
                    role=AuthorRole.SYSTEM,
                    content=await self._render_prompt(
                        self.result_filter_prompt,
                        KernelArguments(topic=self._conversation_state.get("conversation_topic", "assistance"))
                    ),
                ),
            )
            
            # Add the conversation history for context
            recent_messages = chat_history.messages[-8:] if len(chat_history.messages) > 8 else chat_history.messages
            for msg in recent_messages:
                filter_history.add_message(msg)
            
            filter_history.add_message(
                ChatMessageContent(role=AuthorRole.USER, content="Please provide a comprehensive summary of this conversation.")
            )

            response = await self._chat_service.get_chat_message_content(
                filter_history,
                settings=PromptExecutionSettings(
                    response_format=StringResult,
                    max_tokens=800,
                    temperature=0.2
                )
            )
            
            string_with_reason = StringResult.model_validate_json(response.content)
            
            return MessageResult(
                result=ChatMessageContent(
                    role=AuthorRole.ASSISTANT,
                    content=string_with_reason.result
                ),
                reason=string_with_reason.reason
            )
            
        except Exception as e:
            print(f"⚠️ Error in result filtering: {e}")
            # Enhanced fallback - create manual summary
            user_messages = [msg for msg in chat_history.messages if msg.role == AuthorRole.USER]
            agent_messages = [msg for msg in chat_history.messages if msg.role == AuthorRole.ASSISTANT]
            
            summary = f"""📋 Conversation Summary:
• User made {len(user_messages)} request(s)
• {len(agent_messages)} agent response(s) provided
• Agents involved: {', '.join(set(msg.name or 'Unknown' for msg in agent_messages))}
"""
            
            if user_messages:
                summary += f"• Main request: {user_messages[0].content[:200]}{'...' if len(user_messages[0].content) > 200 else ''}"
            
            if agent_messages:
                last_agent = agent_messages[-1]
                summary += f"\n• Final response from {last_agent.name or 'Agent'}: {str(last_agent.content)[:200]}{'...' if len(str(last_agent.content)) > 200 else ''}"
            
            return MessageResult(
                result=ChatMessageContent(
                    role=AuthorRole.ASSISTANT,
                    content=summary
                ),
                reason="Manual summary due to AI filtering error"
            )
    
    def _format_participants_enhanced(self, participant_descriptions: Dict[str, str]) -> str:
        """Enhanced participant formatting with capabilities and use cases"""
        formatted = []
        for name, description in participant_descriptions.items():
            # Add capability indicators
            capabilities = ""
            if "transaction" in description.lower() or "financial" in description.lower():
                capabilities = "🔢 Data Processing | 💰 Financial Analysis | ⚠️ Risk Assessment"
            elif "career" in description.lower():
                capabilities = "💼 Career Guidance | 📝 Resume Help | 🎯 Job Search"
            elif "graph" in description.lower() or "microsoft" in description.lower():
                capabilities = "📧 Email Ops | 👥 User Mgmt | 📁 File Ops | ✅ Task Mgmt"
            elif "general" in description.lower():
                capabilities = "💬 Conversation | ❓ Q&A | 🔗 Coordination"
            
            formatted.append(f"• {name}: {description}\n  Capabilities: {capabilities}")
        
        return "\n\n".join(formatted)


async def enhanced_human_response_function(chat_history: ChatHistory) -> ChatMessageContent:
    """
    Enhanced human response function with conversation context and guidance.
    
    This function is called when the manager determines user input is needed,
    implementing the Manager → User part of the workflow.
    """
    print("\n" + "="*80)
    print("🤖 MANAGER: Additional input needed from user")
    print("="*80)
    
    # Analyze what the agent is asking for
    last_agent_message = None
    for message in reversed(chat_history.messages):
        if message.role == AuthorRole.ASSISTANT:
            last_agent_message = message
            break
    
    if last_agent_message:
        agent_name = last_agent_message.name or "Assistant"
        content = str(last_agent_message.content)
        
        print(f"💭 {agent_name} said: {content[:300]}...")
        print("\n🎯 The system is waiting for your response to continue the conversation.")
        
        # Provide context-specific guidance
        if "transaction" in content.lower() or "data" in content.lower():
            print("\n📊 TRANSACTION DATA GUIDANCE:")
            print("• Provide transaction data in CSV format")
            print("• Include fields like: amount, date, merchant, category")
            print("• Or describe what analysis you need")
            print("• Example: 'Analyze this data: [amount, merchant, date]'")
        
        elif "career" in content.lower():
            print("\n💼 CAREER GUIDANCE:")
            print("• Describe your career goals or current situation")
            print("• Ask about specific job search strategies")
            print("• Request resume or interview advice")
        
        elif "email" in content.lower() or "microsoft" in content.lower():
            print("\n📧 MICROSOFT 365 GUIDANCE:")
            print("• Specify email operations (send, search, etc.)")
            print("• Request user management tasks")
            print("• Ask for file or task management help")
        
        else:
            print("\n💡 GENERAL GUIDANCE:")
            print("• Provide the information the agent requested")
            print("• Ask for clarification if you're unsure")
            print("• Give more context about your needs")
    
    print(f"\n🔄 The conversation will continue with {agent_name} after your response.")
    print("\n📝 Your response:")
    user_input = input("You: ").strip()
    
    if not user_input:
        user_input = "Please continue with the available information."
    
    print(f"\n� Processing your response and routing back to {agent_name}...")
    return ChatMessageContent(role=AuthorRole.USER, content=user_input)


def enhanced_agent_response_callback(message: ChatMessageContent) -> None:
    """Enhanced callback with conversation flow indicators"""
    agent_name = message.name or "Agent"
    content = str(message.content) or ""
    
    # Determine agent type and add appropriate icons
    if "Transaction" in agent_name or "Azure" in agent_name:
        icon = "💰"
        agent_type = "Transaction Specialist"
    elif "Career" in agent_name:
        icon = "💼"
        agent_type = "Career Advisor"  
    elif "Graph" in agent_name:
        icon = "📧"
        agent_type = "Microsoft 365 Specialist"
    else:
        icon = "💬"
        agent_type = "General Assistant"
    
    print(f"\n{icon} **{agent_name}** ({agent_type}):")
    print(f"{content}")
    
    # Add enhanced analysis indicators
    if len(content) < 100 and any(char.isdigit() for char in content):
        print("🔍 [Raw output detected - may need interpretation]")
    elif any(keyword in content.lower() for keyword in ["need", "require", "provide", "missing", "please", "could you"]):
        print("⚠️ [Agent requesting additional information - user input will be requested]")
    elif any(keyword in content.lower() for keyword in ["complete", "finished", "summary", "result"]):
        print("✅ [Task completion indicated - conversation may conclude]")
    elif "error" in content.lower() or "failed" in content.lower():
        print("❌ [Error detected - user intervention may be needed]")
    elif len(content) > 500:
        print("📋 [Comprehensive response provided]")
        print("✅ [Task completion indicated]")
    
    print("-" * 80)


class EnhancedAzureFoundryOrchestration:
    """
    Enhanced Azure Foundry Orchestration with Human-in-the-Loop capabilities.
    
    This class demonstrates the complete User → Manager → Agent workflow
    using your existing specialized agents.
    """
    
    def __init__(self, endpoint: str, api_key: str, agent_id: str, model_name: str = "gpt-4o", 
                 bot_secret: str = None):
        """Initialize with your existing configuration"""
        self.endpoint = endpoint
        self.api_key = api_key
        self.agent_id = agent_id
        self.model_name = model_name
        self.bot_secret = bot_secret
        
        # Initialize services
        self.chat_service = self._init_chat_service()
        self.directline_client = None
        if bot_secret:
            self.directline_client = DirectLineClient(
                copilot_agent_secret=bot_secret,
                directline_endpoint="https://europe.directline.botframework.com/v3/directline"
            )
        
        self.agents = []
        print(f"✅ Enhanced Orchestration initialized with human-in-the-loop capabilities")
    
    def _init_chat_service(self) -> AzureChatCompletion:
        """Initialize Azure Chat Completion service using your existing logic"""
        try:
            print("🔄 Initializing Azure Chat Completion service...")
            
            # Use your existing multi-approach initialization
            chat_service = None
            
            try:
                cognitive_services_endpoint = "https://aif-e2edemo.cognitiveservices.azure.com/"
                chat_service = AzureChatCompletion(
                    endpoint=cognitive_services_endpoint,
                    api_key=self.api_key,
                    deployment_name=self.model_name
                )
                print("✅ Azure Chat Completion service initialized")
                
            except Exception as e1:
                print(f"❌ Primary approach failed: {e1}")
                # Add your other fallback approaches here
                raise e1
            
            return chat_service
            
        except Exception as e:
            print(f"❌ Failed to initialize chat service: {e}")
            raise
    
    async def create_specialized_agents(self) -> List[Agent]:
        """Create your specialized agents with enhanced descriptions for the orchestration"""
        
        # 1. General Assistant for coordination and clarification
        general_agent = ChatCompletionAgent(
            name="GeneralAssistant",
            description=(
                "Conversational coordinator and general assistant. Handles greetings, explanations, "
                "clarifications, and facilitates communication between user and specialists. "
                "Best for general questions, providing context, and helping users understand "
                "specialist responses. Acts as a bridge between technical agents and users."
            ),
            instructions=(
                "You are a helpful coordinator that facilitates conversation between users and "
                "specialized agents. For simple greetings like 'hi' or 'hello', respond warmly "
                "and briefly without asking for additional information. For complex requests, "
                "provide clear explanations, help users understand specialist responses, and guide "
                "them toward the right specialist when needed. You're friendly, clear, and focused "
                "on making the overall experience smooth and understandable. Avoid repeatedly asking "
                "for additional information unless the user has posed a specific question or request."
            ),
            service=self.chat_service,
        )
        
        # 2. Transaction Analysis Agent (your specialized Azure Foundry agent)
        transaction_agent = await self._create_transaction_agent()
        
        # 3. Microsoft Graph Agent 
        graph_agent = self._create_graph_agent()
        
        # 4. Career Advisor (if available)
        agents = [general_agent, transaction_agent, graph_agent]
        
        if self.directline_client:
            career_agent = CopilotAgent(
                id="career_advisor",
                name="CareerAdvisor", 
                description=(
                    "Microsoft Career advice specialist providing professional guidance, "
                    "job search strategies, resume optimization, interview preparation, "
                    "and career development advice. Connects to Copilot Studio for "
                    "comprehensive career counseling and workplace guidance."
                ),
                directline_client=self.directline_client,
            )
            agents.append(career_agent)
        
        return agents
    
    async def _create_transaction_agent(self) -> Agent:
        """Create your transaction analysis agent with enhanced workflow support and output formatting"""
        try:
            # Use your existing Azure AI Agent creation logic
            from azure.identity import DefaultAzureCredential
            from semantic_kernel.agents.azure_ai.azure_ai_agent import AzureAIAgent
            from semantic_kernel.agents.azure_ai.azure_ai_agent_settings import AzureAIAgentSettings
            from semantic_kernel.agents import AzureAIAgentThread
            
            print("🔄 Creating enhanced Transaction Analysis Agent...")
            
            ai_agent_settings = AzureAIAgentSettings(
                endpoint=self.endpoint,
                model_deployment_name=self.model_name
            )
            
            credential = DefaultAzureCredential()
            
            client = AzureAIAgent.create_client(
                credential=credential,
                endpoint=ai_agent_settings.endpoint,
                api_version=ai_agent_settings.api_version,
            )
            
            agent_definition = await client.agents.get_agent(self.agent_id)
            
            if hasattr(agent_definition, 'name') and agent_definition.name:
                import re
                agent_definition.name = re.sub(r'[^0-9A-Za-z_-]', '_', agent_definition.name)
            
            azure_ai_agent = AzureAIAgent(
                client=client,
                definition=agent_definition,
            )
            
            # Force the name to be consistent for orchestration
            azure_ai_agent.name = "TransactionAgent"
            
            # Enhanced description for better orchestration routing
            azure_ai_agent.description = (
                "Advanced transaction data analysis specialist with Azure Foundry integration. "
                "Processes financial data, detects fraud patterns, performs risk assessment, "
                "validates transaction authenticity, and analyzes numerical datasets. "
                "Requires structured data input (CSV, JSON, or formatted transaction records) "
                "to perform analysis. Will request specific data format if needed. "
                "Best for: fraud detection, transaction validation, financial analysis, risk scoring."
            )
            
            # Create a wrapper to format raw outputs properly
            class FormattedTransactionAgent:
                def __init__(self, azure_agent):
                    self._azure_agent = azure_agent
                    self.name = azure_agent.name
                    self.description = azure_agent.description
                    self.id = getattr(azure_agent, 'id', 'transaction_agent')
                
                async def invoke(self, *args, **kwargs):
                    """Invoke the agent and format raw outputs"""
                    try:
                        result = await self._azure_agent.invoke(*args, **kwargs)
                        
                        # Check if result is a raw array/list output
                        if hasattr(result, 'content'):
                            content = str(result.content).strip()
                            
                            # Detect raw numerical arrays like [1, 0.0028, 0.9972]
                            if content.startswith('[') and content.endswith(']'):
                                try:
                                    # Try to parse as array
                                    import ast
                                    parsed_array = ast.literal_eval(content)
                                    
                                    if isinstance(parsed_array, list) and len(parsed_array) >= 3:
                                        # Format as proper analysis
                                        formatted_response = self._format_analysis_results(parsed_array)
                                        result.content = formatted_response
                                        
                                except Exception:
                                    # If parsing fails, add explanation
                                    result.content = f"Analysis Result: {content}\n\nThis appears to be a classification result. The values likely represent:\n- Class 0 probability: {content.split(',')[0] if ',' in content else 'N/A'}\n- Class 1 probability: {content.split(',')[1] if ',' in content and len(content.split(',')) > 1 else 'N/A'}\n- Class 2 probability: {content.split(',')[2] if ',' in content and len(content.split(',')) > 2 else 'N/A'}\n\nPlease provide more context about what type of analysis you need."
                        
                        return result
                        
                    except Exception as e:
                        print(f"⚠️ Error in transaction agent invoke: {e}")
                        raise
                
                def _format_analysis_results(self, result_array):
                    """Format raw numerical results into human-readable analysis"""
                    if len(result_array) == 3:
                        # Assume this is a classification result [class_0, class_1, class_2]
                        class_0, class_1, class_2 = result_array
                        
                        # Determine the predicted class
                        max_prob = max(class_0, class_1, class_2)
                        predicted_class = result_array.index(max_prob)
                        
                        analysis = f"""
📊 **Transaction Analysis Results**

🎯 **Classification Result:**
- Predicted Class: {predicted_class}
- Confidence: {max_prob:.1%}

📈 **Probability Distribution:**
- Normal Transaction (Class 0): {class_0:.1%}
- Suspicious Activity (Class 1): {class_1:.1%} 
- Fraud Detected (Class 2): {class_2:.1%}

🔍 **Interpretation:**
"""
                        
                        if predicted_class == 0:
                            analysis += "✅ This transaction appears to be **NORMAL** with low risk indicators."
                        elif predicted_class == 1:
                            analysis += "⚠️ This transaction shows **SUSPICIOUS PATTERNS** - requires further review."
                        elif predicted_class == 2:
                            analysis += "🚨 **POTENTIAL FRAUD DETECTED** - immediate investigation recommended."
                        
                        analysis += f"""

📋 **Recommendation:**
{self._get_recommendation(predicted_class)}

💡 **Next Steps:**
- Review transaction details for anomalies
- Check customer transaction history
- Verify merchant and location information
- Consider additional verification if needed
"""
                        
                        return analysis
                    
                    else:
                        # Generic formatting for other array types
                        return f"Analysis Results: {result_array}\n\nDetailed interpretation requires additional context about the data format and expected outputs."
                
                def _get_recommendation(self, predicted_class):
                    """Get recommendation based on predicted class"""
                    recommendations = {
                        0: "Transaction can proceed normally. Continue standard monitoring.",
                        1: "Flag for manual review. Consider additional verification steps.",
                        2: "Block transaction immediately. Initiate fraud investigation procedures."
                    }
                    return recommendations.get(predicted_class, "Review analysis parameters and reprocess if needed.")
                
                # Delegate other attributes to the underlying agent
                def __getattr__(self, name):
                    return getattr(self._azure_agent, name)
            
            formatted_agent = FormattedTransactionAgent(azure_ai_agent)
            print(f"✅ Enhanced Transaction Agent created: {formatted_agent.name}")
            return formatted_agent
            
        except Exception as e:
            print(f"❌ Azure AI Agent failed, using ChatCompletion fallback: {e}")
            
            return ChatCompletionAgent(
                name="TransactionAgent",
                description=(
                    "Transaction data analysis specialist for financial data processing. "
                    "Analyzes transaction patterns, detects anomalies, performs fraud detection, "
                    "and provides risk assessments. Requires transaction data to be provided "
                    "in structured format (CSV, JSON, or formatted records). Will guide users "
                    "on proper data format if needed."
                ),
                instructions=(
                    "You are a transaction analysis specialist. When users provide transaction data:\n"
                    "1. Analyze the data for patterns and anomalies\n"
                    "2. Provide fraud risk assessment and classification\n"
                    "3. Explain findings in clear, actionable language\n"
                    "4. Include probability scores and confidence levels\n"
                    "5. Provide specific recommendations for next steps\n"
                    "6. Format your response as a structured analysis report\n\n"
                    "If no data is provided, ask for transaction data in proper format.\n"
                    "Always be thorough but clear in your explanations."
                ),
                service=self.chat_service,
            )
    
    def _create_graph_agent(self) -> Agent:
        """Create Microsoft Graph agent with enhanced workflow support"""
        if not GRAPH_INTEGRATION_AVAILABLE:
            return ChatCompletionAgent(
                name="GraphAssistant",
                description="Microsoft 365 assistant (limited capabilities - integration not available)",
                instructions="You provide general guidance about Microsoft 365 but cannot perform actual operations.",
                service=self.chat_service,
            )
        
        try:
            # Create enhanced Graph agent with your existing logic
            graph_kernel = Kernel()
            graph_kernel.add_service(self.chat_service)
            
            graph_plugin = MicrosoftGraphPlugin()
            graph_kernel.add_plugin(
                graph_plugin,
                plugin_name="MicrosoftGraphPlugin"
            )
            
            return ChatCompletionAgent(
                name="GraphAssistant",
                description=(
                    "Microsoft 365 operations specialist with Graph API integration. "
                    "Handles email operations, user management, file operations, task management, "
                    "and Office 365 automation. Can send emails, search directories, manage "
                    "OneDrive files, and coordinate Teams activities. Requires specific operation "
                    "details to execute tasks effectively."
                ),
                instructions=(
                    "You are a Microsoft 365 specialist with access to Graph API functions. "
                    "When users request M365 operations:\n"
                    "1. Identify the specific operation needed (email, user lookup, file management, etc.)\n"
                    "2. If details are missing, ask for clarification (email addresses, file names, etc.)\n"
                    "3. Use appropriate kernel functions to perform operations\n"
                    "4. Provide clear feedback on operation results\n"
                    "5. Suggest related operations that might be helpful\n"
                    "Focus on being helpful and efficient with Microsoft 365 tasks."
                ),
                service=self.chat_service,
                kernel=graph_kernel
            )
            
        except Exception as e:
            print(f"❌ Enhanced Graph agent failed: {e}")
            return ChatCompletionAgent(
                name="GraphAssistant",
                description="Microsoft 365 assistant (fallback mode)",
                instructions="You provide guidance about Microsoft 365 operations but have limited capabilities.",
                service=self.chat_service,
            )
    
    def _analyze_agent_productivity(self, chat_history: ChatHistory, user_messages: list, agent_messages: list) -> BooleanResult:
        """
        Analyze agent responses to detect unproductive patterns, especially when agents
        repeatedly indicate they don't have enough data to proceed.
        
        This is a key enhancement to make termination logic agent-aware.
        """
        if len(agent_messages) < 2:
            return BooleanResult(result=False, reason="Too few agent messages to analyze productivity")
        
        # Track agent responses and patterns
        agent_response_patterns = {}
        recent_agent_messages = agent_messages[-4:]  # Last 4 agent messages
        
        # Patterns that indicate an agent can't proceed without more data
        insufficient_data_patterns = [
            "not enough data",
            "insufficient data", 
            "need more data",
            "cannot analyze without",
            "unable to proceed without",
            "missing required data",
            "no data provided",
            "please provide the data",
            "i need the transaction data",
            "cannot perform analysis",
            "insufficient information",
            "need additional information",
            "require more details",
            "cannot process",
            "no valid data found",
            "data appears to be missing",
            "incomplete data set"
        ]
        
        # Progress indicators that show agents are actually making progress
        progress_patterns = [
            "analysis shows",
            "based on the data",
            "i can see that",
            "the transaction",
            "calculated",
            "determined",
            "identified",
            "found that",
            "analysis indicates",
            "results show",
            "according to",
            "processing completed",
            "successfully analyzed"
        ]
        
        for msg in recent_agent_messages:
            agent_name = msg.name or "Unknown"
            content = str(msg.content).lower()
            
            if agent_name not in agent_response_patterns:
                agent_response_patterns[agent_name] = {
                    "insufficient_data_count": 0,
                    "progress_count": 0,
                    "total_responses": 0
                }
            
            agent_response_patterns[agent_name]["total_responses"] += 1
            
            # Check for insufficient data patterns
            for pattern in insufficient_data_patterns:
                if pattern in content:
                    agent_response_patterns[agent_name]["insufficient_data_count"] += 1
                    print(f"🔍 Agent {agent_name} indicated insufficient data: {pattern}")
                    break
            
            # Check for progress patterns
            for pattern in progress_patterns:
                if pattern in content:
                    agent_response_patterns[agent_name]["progress_count"] += 1
                    print(f"✅ Agent {agent_name} showing progress: {pattern}")
                    break
        
        # Analyze patterns for termination decision
        for agent_name, patterns in agent_response_patterns.items():
            total_responses = patterns["total_responses"]
            insufficient_count = patterns["insufficient_data_count"]
            progress_count = patterns["progress_count"]
            
            print(f"📊 Agent {agent_name} productivity: {insufficient_count} insufficient data responses, {progress_count} progress responses out of {total_responses} total")
            
            # Special handling for TransactionAgent and similar data-processing agents
            if any(keyword in agent_name.lower() for keyword in ["transaction", "data", "analysis", "processor"]):
                # If a data agent has repeatedly said "not enough data" without progress
                if insufficient_count >= 2 and progress_count == 0 and total_responses >= 2:
                    # Check if user has had chances to provide data
                    user_responses_after_first_request = len(user_messages) - 1
                    
                    if user_responses_after_first_request >= 1:
                        # User had chance to provide data but agent still can't proceed
                        print(f"⚠️ {agent_name} repeatedly indicates insufficient data despite user responses")
                        return BooleanResult(
                            result=True,
                            reason=f"{agent_name} repeatedly indicates insufficient data ({insufficient_count} times) despite user having {user_responses_after_first_request} opportunities to provide data. Conversation appears unproductive."
                        )
            
            # General pattern: any agent with high ratio of "insufficient data" responses
            if total_responses >= 3 and insufficient_count >= (total_responses * 0.7):  # 70% or more
                print(f"⚠️ {agent_name} showing high rate of insufficient data responses")
                return BooleanResult(
                    result=True,
                    reason=f"{agent_name} has indicated insufficient data in {insufficient_count} out of {total_responses} responses. Conversation appears stuck."
                )
        
        # Check for overall conversation stagnation
        if len(agent_messages) >= 6:  # Only after sufficient conversation
            recent_insufficient_count = sum(
                1 for msg in recent_agent_messages 
                if any(pattern in str(msg.content).lower() for pattern in insufficient_data_patterns)
            )
            recent_progress_count = sum(
                1 for msg in recent_agent_messages 
                if any(pattern in str(msg.content).lower() for pattern in progress_patterns)
            )
            
            # If most recent responses show no progress and lots of data requests
            if recent_insufficient_count >= 3 and recent_progress_count == 0:
                print(f"⚠️ Conversation showing overall stagnation: {recent_insufficient_count} insufficient data responses, {recent_progress_count} progress responses")
                return BooleanResult(
                    result=True,
                    reason=f"Conversation appears stagnated with {recent_insufficient_count} recent 'insufficient data' responses and no progress indicators. Terminating to prevent endless loops."
                )
        
        return BooleanResult(result=False, reason="Agents showing adequate progress or insufficient data to determine stagnation")
