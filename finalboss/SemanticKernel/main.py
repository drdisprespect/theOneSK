#!/usr/bin/env python3
# Azure Foundry Agent Chat with Group Chat Orchestration

import asyncio
import sys
import os
import re
import traceback
from typing import List, Dict, Any
from azure.identity import DefaultAzureCredential
from semantic_kernel.agents import Agent, ChatCompletionAgent, GroupChatOrchestration
from semantic_kernel.agents.orchestration.group_chat import BooleanResult, GroupChatManager, MessageResult, StringResult
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.agents.azure_ai.azure_ai_agent import AzureAIAgent
from semantic_kernel.agents.azure_ai.azure_ai_agent_settings import AzureAIAgentSettings
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.contents import AuthorRole, ChatHistory, ChatMessageContent
from semantic_kernel.functions import KernelArguments
from semantic_kernel.kernel import Kernel
from semantic_kernel.prompt_template import KernelPromptTemplate, PromptTemplateConfig

# Copilot Studio imports
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
except ImportError as e:
    GRAPH_INTEGRATION_AVAILABLE = False

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


class AzureFoundryOrchestration:
    """
    Enhanced Azure Foundry Agent Chatbot using Group Chat Orchestration Pattern
    
    This implementation combines your specialized Azure Foundry agent with a general
    chat completion agent, orchestrated through a smart group chat manager.
    """

    def __init__(self, endpoint: str, api_key: str, agent_id: str, model_name: str = "gpt-4o", 
                 bot_secret: str = None):
        """
        Initialize the Azure Foundry Orchestration
        
        Args:
            endpoint: Your Azure Foundry project endpoint URL (can be foundry or cognitive services)
            api_key: Your Azure Foundry API key 
            agent_id: Your specific agent ID to retrieve
            model_name: The model deployment name (default: gpt-4o)
            bot_secret: The Copilot Studio bot secret for DirectLine API
        """
        self.endpoint = endpoint
        self.api_key = api_key
        self.agent_id = agent_id
        self.model_name = model_name
        self.bot_secret = bot_secret
        
        # Initialize Azure Chat Completion for the orchestration manager
        self.chat_service = self._init_chat_service()
        
        # Initialize Copilot Studio DirectLine client if bot_secret is provided
        self.directline_client = None
        if bot_secret:
            self.directline_client = DirectLineClient(
                copilot_agent_secret=bot_secret,
                directline_endpoint="https://europe.directline.botframework.com/v3/directline"
            )
        
        # Create agents
        self.agents = []
        self.foundry_agent = None
        self.runtime = None
        
        print(f"✓ Azure Foundry Orchestration initialized")
        if bot_secret:
            print(f"✓ Copilot Studio agent enabled")

    def _init_chat_service(self) -> AzureChatCompletion:
        """Initialize the Azure Chat Completion service"""
        try:
            print("🔄 Initializing Azure Chat Completion service...")
            
            # Try multiple approaches for service initialization
            chat_service = None
            
            # Approach 1: Try with new Cognitive Services endpoint
            try:
                cognitive_services_endpoint = "https://aif-e2edemo.cognitiveservices.azure.com/"
                
                chat_service = AzureChatCompletion(
                    endpoint=cognitive_services_endpoint,
                    api_key=self.api_key,
                    deployment_name=self.model_name
                )
                
            except Exception as e1:
                # Approach 2: Try with original Azure OpenAI endpoint
                try:
                    foundry_openai_endpoint = "https://aif-e2edemo.openai.azure.com/"
                    
                    chat_service = AzureChatCompletion(
                        endpoint=foundry_openai_endpoint,
                        api_key=self.api_key,
                        deployment_name=self.model_name
                    )
                    
                except Exception as e2:
                    print(f"❌ OpenAI endpoint approach failed: {e2}")
                    
                    # Approach 3: Try with DefaultAzureCredential
                    try:
                        print("🔍 Trying Azure credential approach...")
                        credential = DefaultAzureCredential()
                        
                        chat_service = AzureChatCompletion(
                            endpoint=cognitive_services_endpoint,
                            ad_token=credential.get_token("https://cognitiveservices.azure.com/.default").token,
                            deployment_name=self.model_name
                        )
                        print("✅ Azure Chat Completion service initialized with Azure credentials")
                        
                    except Exception as e3:
                        print(f"❌ All approaches failed. Last error: {e3}")
                        raise e3
            
            if chat_service is None:
                raise Exception("Failed to initialize chat service with any approach")
                
            return chat_service
            
        except Exception as e:
            print(f"❌ Failed to initialize chat service: {e}")
            print("⚠️ This might be due to authentication or endpoint configuration issues")
            raise

    def create_general_agent(self) -> ChatCompletionAgent:
        """Create a general conversation agent with detailed description for AI routing"""
        return ChatCompletionAgent(
            name="GeneralAssistant",
            description=(
                "General-purpose AI assistant for casual conversation, clarifications, basic questions, "
                "and topics that don't require specialized expertise. Handles greetings, general inquiries, "
                "simple explanations, and provides broad perspective on various topics. Best for when users "
                "need friendly conversation or basic information."
            ),
            instructions=(
                "You are a helpful, friendly AI assistant. You can engage in general conversation, "
                "answer questions, and provide assistance on a wide range of topics. "
                "When in group discussions, you provide accessible, clear explanations and "
                "help facilitate productive conversations. You complement specialized agents "
                "by offering broader perspective and clarifying complex topics for users."
            ),
            service=self.chat_service,
        )

    async def create_transaction_analysis_agent(self):
        """Create a Transaction Agent using actual Azure Foundry Agent for financial data processing"""
        try:
            # Import required classes for Azure AI Agent integration
            from azure.identity import DefaultAzureCredential
            from semantic_kernel.agents.azure_ai.azure_ai_agent import AzureAIAgent
            from semantic_kernel.agents.azure_ai.azure_ai_agent_settings import AzureAIAgentSettings
            from semantic_kernel.agents import AzureAIAgentThread
            
            print("🔄 Creating Transaction Analysis Agent with direct Azure Foundry integration...")
            
            # Create Azure AI Agent settings for the transaction analysis agent
            ai_agent_settings = AzureAIAgentSettings(
                endpoint=self.endpoint,
                model_deployment_name=self.model_name
            )

            # Initialize with DefaultAzureCredential and get existing agent
            credential = DefaultAzureCredential()
            
            # Create client for the Azure AI Agent
            client = AzureAIAgent.create_client(
                credential=credential,
                endpoint=ai_agent_settings.endpoint,
                api_version=ai_agent_settings.api_version,
            )
            
            # Get the existing agent definition using the agent_id
            agent_definition = await client.agents.get_agent(self.agent_id)
            
            # Fix the agent name to match the required pattern
            if hasattr(agent_definition, 'name') and agent_definition.name:
                import re
                agent_definition.name = re.sub(r'[^0-9A-Za-z_-]', '_', agent_definition.name)
            
            # Create the actual Azure AI Agent
            azure_ai_agent = AzureAIAgent(
                client=client,
                definition=agent_definition,
            )
            
            # Ensure the agent has a description for GroupChatOrchestration
            if not hasattr(azure_ai_agent, 'description') or not azure_ai_agent.description:
                azure_ai_agent.description = (
                    "Specialized transaction data processing and financial analysis agent. "
                    "Processes raw transaction data, performs fraud detection, risk assessment, "
                    "and generates analytical insights from financial records. Expert in "
                    "transaction validation, pattern recognition, and numerical data analysis."
                )
            
            # Store the actual agent name for selection logic
            self.transaction_agent_name = azure_ai_agent.name or "TransactionAgent"
            
            # Create thread for conversation
            azure_ai_thread = AzureAIAgentThread(client=client)
            
            # Store components for access
            self.azure_ai_agent = azure_ai_agent
            self.azure_ai_thread = azure_ai_thread
            self.azure_ai_client = client
            

            
            # Return the actual Azure AI Agent (not a wrapper)
            return azure_ai_agent
            
        except Exception as e:
            # Fallback to standard ChatCompletionAgent
            return ChatCompletionAgent(
                name="TransactionAgent",
                description=(
                    "Specialized transaction data processing and financial analysis agent. "
                    "Processes raw transaction data, financial records, numerical datasets, "
                    "fraud detection analysis, risk assessment, and structured data analysis. "
                    "Excels at transaction validation, pattern recognition in financial data, "
                    "and generating analytical insights from transaction records. Best for "
                    "transaction processing, financial analysis, fraud detection, and "
                    "numerical data interpretation."
                ),
                instructions=(
                    "You are a specialized transaction data processing agent that analyzes raw transaction data "
                    "and provides structured analytical insights. You focus on processing financial records, "
                    "validating transaction data, detecting patterns and anomalies, assessing risk levels, "
                    "and generating comprehensive analytical reports with numerical insights and risk assessments. "
                    "You excel at fraud detection, transaction validation, and financial data analysis."
                ),
                service=self.chat_service,
            )

    def create_copilot_studio_agent(self) -> CopilotAgent:
        """Create a Copilot Studio agent for Microsoft Career advice"""
        if not self.directline_client:
            raise ValueError("DirectLine client not initialized. Bot secret is required.")
        
        return CopilotAgent(
            id="copilot_studio",
            name="CareerAdvisor",
            description=(
                "Microsoft Career advice specialist providing comprehensive career guidance, "
                "job search strategies, resume optimization, interview preparation, "
                "professional development advice, workplace skills assessment, and "
                "Microsoft-specific career paths. Best for career-related questions, "
                "professional development, job search assistance, and workplace guidance."
            ),
            directline_client=self.directline_client,
        )

    def create_microsoft_graph_agent(self) -> ChatCompletionAgent:
        """Create a Microsoft Graph agent for Microsoft 365 operations with kernel functions"""
        if not GRAPH_INTEGRATION_AVAILABLE:
            return ChatCompletionAgent(
                name="GraphAssistant",
                description="Microsoft 365 assistant (fallback mode)",
                instructions=(
                    "You are a helpful assistant for Microsoft 365 operations. "
                    "However, you currently don't have access to the Graph API integration. "
                    "You can provide general guidance about Microsoft 365 services, but cannot perform actual operations."
                ),
                service=self.chat_service,
            )
        
        try:
            print("🔄 Creating Microsoft Graph Agent with kernel functions...")
            
            # Create a kernel for the Graph agent
            graph_kernel = Kernel()
            
            # Add the chat service to the kernel
            graph_kernel.add_service(self.chat_service)
            
            # Create and add the Microsoft Graph plugin with proper initialization
            graph_plugin = MicrosoftGraphPlugin()
            
            # Try to read the plugin description from the Graph plugin module
            try:
                from graph_agent_plugin import graph_plugin_description
                plugin_description = graph_plugin_description
            except ImportError:
                plugin_description = "Microsoft Graph Plugin for Microsoft 365 operations"
            
            graph_kernel.add_plugin(
                graph_plugin,
                plugin_name="MicrosoftGraphPlugin",
                description=plugin_description
            )
            
            # Try to import and set up function calling (optional)
            try:
                from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
                from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
                    AzureChatPromptExecutionSettings,
                )
                
                execution_settings = AzureChatPromptExecutionSettings(tool_choice="auto")
                execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

            except ImportError as e:
                print(f"⚠️ Function calling not available: {e}")
                execution_settings = None
            
            # Enhanced system prompt based on the src/prompts/orchestrator_system_prompt.txt
            enhanced_instructions = (
                "You are a Microsoft Graph assistant with direct access to Microsoft 365 APIs through kernel functions. "
                "You can perform real operations with Microsoft 365 services including:\n\n"
                
                "🧑‍💼 **USER MANAGEMENT OPERATIONS:**\n"
                "- List organization users: 'get_users' (lists all users in directory)\n"
                "- Find specific users: 'find_user_by_name' (search by name or email)\n"
                "- Load user cache: 'load_users_cache' (refresh user directory)\n"
                "- Get user emails: 'get_email_by_name' (get email for user by name)\n"
                "- Set active user: 'set_user_id' (set user context for operations)\n\n"
                
                "📧 **EMAIL OPERATIONS:**\n"
                "- Send emails: 'send_mail' (compose and send emails)\n"
                "- Get inbox: 'get_inbox_messages' (retrieve recent inbox messages)\n"
                "- Search emails: 'search_all_emails' (search across all folders)\n"
                "- Get mail folders: 'get_mail_folders' (list email folder structure)\n\n"
                
                "✅ **TODO & TASK MANAGEMENT:**\n"
                "- Create todo lists: 'create_todo_list' (create new task lists)\n"
                "- Add tasks: 'create_todo_task' (add tasks to lists)\n"
                "- Get all lists: 'get_todo_lists' (retrieve all todo lists)\n"
                "- Get list tasks: 'get_todo_tasks_from_list' (get tasks from specific list)\n\n"
                
                "📁 **ONEDRIVE FILE OPERATIONS:**\n"
                "- Create folders: 'create_folder' (create folders in OneDrive)\n"
                "- Get user drive: 'get_user_drive' (get user's OneDrive information)\n"
                "- Get drive root: 'get_drive_root' (access root folder structure)\n\n"
                
                "🔧 **PLUGIN STATE MANAGEMENT:**\n"
                "- Initialize plugin: 'initialize' (set up Graph API configuration)\n"
                "- Get plugin state: 'get_state' (check plugin initialization status)\n\n"
                
                "💡 **OPERATION GUIDELINES:**\n"
                "• Always use appropriate kernel functions for actual Microsoft 365 operations\n"
                "• Provide clear feedback about operations being performed\n"
                "• Handle errors gracefully and suggest alternatives\n"
                "• When searching users, use fuzzy matching for better results\n"
                "• For email operations, validate recipients before sending\n"
                "• Always confirm successful completion of operations\n\n"
                
                "Example usage patterns:\n"
                "- 'List users' → use get_users function\n"
                "- 'Find John Smith' → use find_user_by_name function\n"
                "- 'Send email to marketing team' → use send_mail function\n"
                "- 'Create project task list' → use create_todo_list function\n"
                "- 'Search emails about budget' → use search_all_emails function"
            )
            
            # Create a ChatCompletionAgent with the kernel
            graph_agent = ChatCompletionAgent(
                name="GraphAssistant",
                description=(
                    "Microsoft 365 and Graph API specialist with 18 kernel functions for email operations, "
                    "user management, file operations, task management, Teams collaboration, and Office 365 services. "
                    "Can send emails, find users, create tasks, manage OneDrive folders, and perform Microsoft 365 "
                    "operations through Graph API. Best for Microsoft 365 tasks, email management, "
                    "user directory operations, and Office productivity automation."
                ),
                instructions=enhanced_instructions,
                service=self.chat_service,
                kernel=graph_kernel
            )
            

            print("   📧 Email: send_mail, get_inbox_messages, search_all_emails, get_mail_folders")
            print("   🧑‍💼 Users: get_users, find_user_by_name, get_email_by_name, load_users_cache")
            print("   ✅ Tasks: create_todo_list, create_todo_task, get_todo_lists, get_todo_tasks_from_list")
            print("   📁 Files: create_folder, get_user_drive, get_drive_root")
            print("   🔧 Management: initialize, get_state, set_user_id")
            return graph_agent
            
        except Exception as e:
            print(f"❌ Failed to create Microsoft Graph agent with kernel: {e}")
            print(f"   Error details: {traceback.format_exc()}")
            print("🔄 Creating fallback agent...")
            
            # Return fallback agent
            return ChatCompletionAgent(
                name="GraphAssistant",
                description=(
                    "Microsoft 365 and Graph API specialist for email operations, user management, "
                    "file operations, task management, Teams collaboration, and Office 365 services. "
                    "Handles sending emails, managing users, creating folders, organizing tasks, "
                    "Teams communication, and OneDrive operations. Best for Microsoft 365 tasks, "
                    "email management, user directory operations, and Office productivity needs."
                ),
                instructions=(
                    "You are a helpful assistant for Microsoft 365 operations. "
                    "You can provide general guidance about Microsoft 365 services, help with "
                    "email management, user operations, file management, and Teams collaboration."
                ),
                service=self.chat_service,
            )

    async def get_agents(self) -> List[Agent]:
        """Create and return the list of agents for orchestration"""
        if not self.agents:
            print("🔄 Creating agent ensemble...")
            
            # Create general assistant
            general_agent = self.create_general_agent()
            
            # Create Transaction Analysis specialist (renamed from Azure AI Agent)
            transaction_agent = await self.create_transaction_analysis_agent()
            
            # Create Microsoft Graph agent
            graph_agent = self.create_microsoft_graph_agent()
            
            # Create list of agents
            self.agents = [general_agent, transaction_agent, graph_agent]
            
            # Add Copilot Studio agent if available
            if self.directline_client:
                copilot_agent = self.create_copilot_studio_agent()
                self.agents.append(copilot_agent)
                print(f"✅ Created {len(self.agents)} agents for orchestration")
                print("   • GeneralAssistant: General conversation and assistance")
                print("   • TransactionAgent: Transaction data processing and financial analysis")
                print("   • CareerAdvisor: Microsoft Career advice from Copilot Studio")
                print("   • GraphAssistant: Microsoft 365 operations and management")
            else:
                print(f"✅ Created {len(self.agents)} agents for orchestration")
                print("   • GeneralAssistant: General conversation and assistance")
                print("   • TransactionAgent: Transaction data processing and financial analysis")
                print("   ⚠️ Copilot Studio agent disabled (no bot secret provided)")
                print("   • GraphAssistant: Microsoft 365 operations and management")
            
        return self.agents


class AIGroupChatManager(GroupChatManager):
    """
    AI-driven Group Chat Manager following Semantic Kernel framework patterns.
    
    This manager lets the framework's AI handle agent selection based on descriptions
    and conversation context, rather than using pattern matching.
    """

    def __init__(self, service: ChatCompletionClientBase, max_rounds: int = 10, human_response_function=None, **kwargs):
        """Initialize the AI group chat manager"""
        super().__init__(service=service, max_rounds=max_rounds, **kwargs)
        self._chat_service = service
        self._human_response_function = human_response_function

    @override
    async def should_request_user_input(self, chat_history: ChatHistory) -> BooleanResult:
        """
        Determine if user input is needed. 
        
        Override this to implement human-in-the-loop scenarios.
        For now, we don't request additional user input during orchestration.
        """
        if self._human_response_function is None:
            return BooleanResult(
                result=False,
                reason="No human response function configured."
            )
        
        # Example: Request user input after certain agent responses
        # You can customize this logic based on your needs
        if len(chat_history.messages) == 0:
            return BooleanResult(
                result=False,
                reason="No agents have spoken yet."
            )
        
        # Don't request user input for now - keep single-turn behavior
        return BooleanResult(
            result=False,
            reason="Single-turn conversation mode - no additional user input needed."
        )

    @override
    async def should_terminate(self, chat_history: ChatHistory) -> BooleanResult:
        """
        Determine if the conversation should end.
        
        Simple termination logic: End after one agent responds to the user's request.
        """
        # Check parent termination conditions first
        should_terminate = await super().should_terminate(chat_history)
        if should_terminate.result:
            return should_terminate

        if not chat_history.messages:
            return BooleanResult(result=False, reason="No messages yet")
        
        # Find the last user message and check if there's an assistant response after it
        last_user_message_idx = -1
        has_assistant_response_after_user = False
        
        for i, message in enumerate(chat_history.messages):
            if message.role == AuthorRole.USER:
                last_user_message_idx = i
                has_assistant_response_after_user = False  # Reset when we find a new user message
            elif message.role == AuthorRole.ASSISTANT and last_user_message_idx >= 0:
                has_assistant_response_after_user = True
                break  # Found assistant response after last user message
        
        if has_assistant_response_after_user:
            return BooleanResult(
                result=True, 
                reason="Agent has responded to user - conversation complete"
            )
        
        return BooleanResult(
            result=False, 
            reason="Waiting for agent response to user request"
        )

    @override
    async def filter_results(self, chat_history: ChatHistory) -> MessageResult:
        """
        Filter and summarize conversation results.
        
        Provides intelligent post-processing for different types of responses,
        including specialized handling for transaction analysis.
        """
        if not chat_history.messages:
            raise RuntimeError("No messages in chat history")

        # Get the last assistant message
        last_assistant_message = None
        for message in reversed(chat_history.messages):
            if message.role == AuthorRole.ASSISTANT:
                last_assistant_message = message
                break

        if not last_assistant_message:
            return MessageResult(
                result=ChatMessageContent(
                    role=AuthorRole.ASSISTANT,
                    content="No assistant response found in conversation."
                ),
                reason="No assistant response to summarize"
            )

        # Check if this is transaction/Azure AI Agent output that needs analysis
        content = str(last_assistant_message.content)
        
        if self._is_raw_output(content):
            try:
                analysis_result = await self._analyze_raw_output(content)
                combined_result = f"Raw Output: {content}\n\n{'='*60}\n\n{analysis_result}"
                
                return MessageResult(
                    result=ChatMessageContent(
                        role=AuthorRole.ASSISTANT,
                        content=combined_result
                    ),
                    reason="Raw output processed and analyzed successfully"
                )
            except Exception as e:
                print(f"⚠️ Raw output analysis failed: {e}")
        
        # For regular responses, return the last assistant message as-is
        return MessageResult(
            result=last_assistant_message,
            reason="Conversation completed successfully"
        )

    def _is_raw_output(self, content: str) -> bool:
        """Check if content appears to be raw numerical output that needs analysis"""
        import re
        
        # Check for array-like patterns [number, number, number]
        array_pattern = r'\[\s*[\d\.\-\s,]+\s*\]'
        if re.search(array_pattern, content.strip()):
            return True
            
        # Check for simple numerical outputs
        simple_number_pattern = r'^\s*[\d\.\-\s,]+\s*$'
        if re.match(simple_number_pattern, content.strip()):
            return True
            
        # Check if the content is very short and primarily numerical
        if len(content.strip()) < 50 and any(char.isdigit() for char in content):
            return True
            
        return False

    async def _analyze_raw_output(self, raw_output: str) -> str:
        """Analyze raw numerical output and provide interpretation"""
        try:
            analysis_history = ChatHistory()
            
            analysis_prompt = (
                "You are analyzing raw output from a specialized Azure AI Agent. "
                "This agent processes transaction data and returns numerical results, typically in array format. "
                "Your role is to interpret these raw numerical outputs:\n\n"
                "🔍 **INTERPRETATION FOCUS:**\n"
                "- Risk scoring and probability analysis\n"
                "- Classification results (fraud/legitimate/suspicious)\n"
                "- Confidence levels and uncertainty measures\n"
                "- Pattern matching scores\n\n"
                "📊 **OUTPUT FORMAT:**\n"
                "=== ANALYSIS INTERPRETATION ===\n"
                "[Interpretation of the numerical values]\n\n"
                "=== RISK ASSESSMENT ===\n"
                "[Risk levels based on the output]\n\n"
                "=== RECOMMENDED ACTIONS ===\n"
                "[Next steps based on the analysis]\n\n"
                "Interpret the following raw output:"
            )
            
            analysis_history.add_message(ChatMessageContent(
                role=AuthorRole.SYSTEM,
                content=analysis_prompt
            ))
            
            analysis_history.add_message(ChatMessageContent(
                role=AuthorRole.USER,
                content=f"Please interpret this raw output:\n\n{raw_output}"
            ))

            response = await self._chat_service.get_chat_message_content(
                analysis_history,
                settings=PromptExecutionSettings(max_tokens=1500, temperature=0.3)
            )
            
            return str(response.content) if response and response.content else "Analysis could not be completed."
            
        except Exception as e:
            print(f"⚠️ Raw output analysis failed: {e}")
            return f"Analysis error: {str(e)}"

    @override
    async def select_next_agent(
        self,
        chat_history: ChatHistory,
        participant_descriptions: Dict[str, str],
    ) -> StringResult:
        """
        Select which agent should respond next using AI analysis.
        
        This method lets the framework's AI analyze the conversation context
        and agent descriptions to make intelligent routing decisions.
        """
        if not chat_history.messages:
            # Default to GeneralAssistant for first interaction
            agent_names = list(participant_descriptions.keys())
            return StringResult(
                result=agent_names[0] if agent_names else "GeneralAssistant",
                reason="No conversation history - starting with first agent"
            )
        
        # Get the latest user message for context
        user_message = ""
        for message in reversed(chat_history.messages):
            if message.role == AuthorRole.USER:
                user_message = str(message.content)
                break
        
        # Create analysis prompt for agent selection
        selection_prompt = (
            "You are an intelligent conversation manager analyzing which AI assistant should respond next.\n\n"
            "Available assistants and their expertise:\n"
            f"{self._format_participants(participant_descriptions)}\n\n"
            "User's request: {user_message}\n\n"
            "SELECTION PRIORITY RULES:\n"
            "1. FOR TRANSACTION DATA: If the request involves transaction data, financial records, "
            "fraud detection, risk assessment, numerical data analysis, or processing financial information, "
            "ALWAYS select TransactionAgent regardless of other content.\n\n"
            "2. FOR CAREER ADVICE: If the request involves career guidance, job search, resume help, "
            "interview preparation, or professional development, select CareerAdvisor.\n\n"
            "3. FOR MICROSOFT 365: If the request involves email operations, user management, "
            "Teams operations, OneDrive, or Office 365 tasks, select GraphAssistant.\n\n"
            "4. FOR GENERAL QUESTIONS: For casual conversation, basic questions, or general topics "
            "that don't fit the above categories, select GeneralAssistant.\n\n"
            "Transaction-related keywords to watch for: transaction, financial, fraud, risk, analysis, "
            "data processing, numerical analysis, banking, payment, money, financial records, "
            "risk assessment, anomaly detection, validation.\n\n"
            "Respond with ONLY the exact assistant name from the list above."
        )
        
        # Create analysis history
        analysis_history = ChatHistory()
        analysis_history.add_message(ChatMessageContent(
            role=AuthorRole.SYSTEM,
            content=selection_prompt
        ))
        
        analysis_history.add_message(ChatMessageContent(
            role=AuthorRole.USER,
            content=f"User's request: {user_message}\n\nWhich assistant should respond?"
        ))

        try:
            # Use AI to select the best agent
            response = await self._chat_service.get_chat_message_content(
                analysis_history,
                settings=PromptExecutionSettings(max_tokens=50, temperature=0.1)
            )
            
            selected_agent = str(response.content).strip()
            
            # Validate the selection
            if selected_agent in participant_descriptions:
                return StringResult(
                    result=selected_agent,
                    reason=f"AI selected {selected_agent} based on user request analysis"
                )
            else:
                # Fallback: try to find partial matches
                for agent_name in participant_descriptions.keys():
                    if agent_name.lower() in selected_agent.lower() or selected_agent.lower() in agent_name.lower():
                        return StringResult(
                            result=agent_name,
                            reason=f"AI selected {agent_name} (partial match for: {selected_agent})"
                        )
                
                # Final fallback to first agent
                agent_names = list(participant_descriptions.keys())
                fallback_agent = agent_names[0] if agent_names else "GeneralAssistant"
                return StringResult(
                    result=fallback_agent,
                    reason=f"AI selection '{selected_agent}' not found, defaulting to {fallback_agent}"
                )
                
        except Exception as e:
            print(f"⚠️ AI agent selection failed: {e}")
            # Default to first agent on error
            agent_names = list(participant_descriptions.keys())
            fallback_agent = agent_names[0] if agent_names else "GeneralAssistant"
            return StringResult(
                result=fallback_agent,
                reason=f"Selection error - defaulting to {fallback_agent}"
            )
    
    def _format_participants(self, participant_descriptions: Dict[str, str]) -> str:
        """Format participant descriptions for the AI selection prompt"""
        formatted = []
        for name, description in participant_descriptions.items():
            formatted.append(f"- {name}: {description}")
        return "\n".join(formatted)


async def human_response_function(chat_history: ChatHistory) -> ChatMessageContent:
    """Function to get user input for human-in-the-loop scenarios"""
    user_input = input("You (follow-up): ")
    return ChatMessageContent(role=AuthorRole.USER, content=user_input)


def agent_response_callback(message: ChatMessageContent) -> None:
    """Callback to store agent responses without displaying intermediate steps"""
    # Store the response but don't print to reduce verbosity
    # The final result will be shown in the conversation summary
    pass


async def interactive_foundry_orchestration():
    """
    Interactive session using Group Chat Orchestration with Azure Foundry Agent
    """
    print("=== Azure AI Multi-Agent Chat ===")
    print("Intelligent agent routing for various tasks\n")

    # Configuration - updated with API key from transaction analysis agent
    endpoint = "https://aif-e2edemo.services.ai.azure.com/api/projects/project-thanos"
    api_key = "D0b8REYc0wXONcnJu7fmj6kyFciM5XTrxjzJmoL1PtAXiqw1GHjXJQQJ99BFACYeBjFXJ3w3AAAAACOGpETv"  # Updated API key from transaction agent
    agent_id = "asst_JE3DIZwUr7MWbb7KCM4OHxV4"
    model_name = "gpt-4o"
    
    # Copilot Studio Bot Secret for Career Advice
    bot_secret = "EIyanQMLVDOeluIdbvfddzUpO2mO14oGy8MKH04lprY08zqu0fqOJQQJ99BFAC4f1cMAArohAAABAZBS2U6n.CXzSOwihZ7dl5h9sI70U5VGr7ydVp75Nfr69psUNlP6KmQneluqoJQQJ99BFAC4f1cMAArohAAABAZBS3VyD"

    if not all([endpoint, api_key, agent_id]):
        print("❌ Error: Missing required configuration")
        return

    try:
        # Initialize orchestration with Copilot Studio bot
        orchestration_system = AzureFoundryOrchestration(
            endpoint=endpoint, 
            api_key=api_key, 
            agent_id=agent_id, 
            model_name=model_name,
            bot_secret=bot_secret
        )
        
        # Create agents
        agents = await orchestration_system.get_agents()
        
        # Create group chat orchestration with optional human-in-the-loop
        group_chat_orchestration = GroupChatOrchestration(
            members=agents,
            manager=AIGroupChatManager(
                service=orchestration_system.chat_service,
                max_rounds=10,
                # Uncomment the next line to enable human-in-the-loop functionality
                # human_response_function=human_response_function,
            ),
            agent_response_callback=agent_response_callback,
        )

        # Create and start runtime
        runtime = InProcessRuntime()
        runtime.start()
        
        print("✅ Orchestration system ready!")
        print("\n=== Chat Session Started ===")
        print("Type 'help' for commands or 'quit' to exit\n")

        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ("quit", "exit", "bye"):
                    break
                    
                if user_input.lower() == "help":
                    print("\n📋 Available Commands:")
                    print("• Type any question or request")
                    print("• [quit/exit/bye] - End session")
                    print("• The system will automatically route to the best agent")
                    print("\n🤖 Agent Routing:")
                    print("• General questions → GeneralAssistant")
                    print("• Transaction data analysis → TransactionAgent (your specialized agent)")
                    print("• Microsoft Career advice → CareerAdvisor (Copilot Studio)")
                    print("• Microsoft 365 operations → GraphAssistant (Graph API)")
                    print("\n💰 Transaction Analysis (TransactionAgent):")
                    print("  • 'Analyze this transaction data: [data]'")
                    print("  • 'Detect fraud in these transactions'")
                    print("  • 'Perform risk assessment on financial records'")
                    print("  • 'Validate transaction patterns'")
                    print("  • 'Process financial data for anomalies'")
                    print("\n📧 Microsoft Graph Operations (GraphAssistant):")
                    print("\n🧑‍💼 USER MANAGEMENT:")
                    print("  • 'List all users in organization'")
                    print("  • 'Find user John Smith'")
                    print("  • 'Search for users named Sarah'")
                    print("  • 'Get email address for Michael Johnson'")
                    print("  • 'Show organization directory'")
                    print("\n📧 EMAIL OPERATIONS:")
                    print("  • 'Send email to marketing@company.com about project update'")
                    print("  • 'Get my latest inbox messages'")
                    print("  • 'Search emails containing quarterly report'")
                    print("  • 'Show my email folders'")
                    print("  • 'Find emails from last week about budget'")
                    print("\n✅ TASK & TODO MANAGEMENT:")
                    print("  • 'Create a todo list called Project Tasks'")
                    print("  • 'Add task: Review quarterly budget'")
                    print("  • 'Show all my todo lists'")
                    print("  • 'List tasks from my Work list'")
                    print("\n📁 ONEDRIVE FILE OPERATIONS:")
                    print("  • 'Create folder called Project Documents'")
                    print("  • 'Get my OneDrive information'")
                    print("  • 'Show my drive root folder'")
                    print("\n🔧 KEYWORDS FOR ROUTING:")
                    print("  User operations: 'users', 'directory', 'find user', 'employee directory'")
                    print("  Email operations: 'email', 'inbox', 'send mail', 'outlook', 'compose'")
                    print("  Task operations: 'todo', 'task', 'checklist', 'task list'")
                    print("  File operations: 'folder', 'onedrive', 'drive', 'documents'")
                    print("  Teams operations: 'teams', 'teams chat', 'microsoft teams'")
                    continue
                    
                if not user_input:
                    continue

                # Invoke orchestration (quietly)
                orchestration_result = await group_chat_orchestration.invoke(
                    task=user_input,
                    runtime=runtime,
                )

                # Get and display final result only
                result = await orchestration_result.get()
                if result:
                    print(f"\n{result}")
                    print("=" * 80)

            except KeyboardInterrupt:
                break
            except Exception as e:
                continue

        # Cleanup
        if hasattr(orchestration_system, 'directline_client') and orchestration_system.directline_client:
            await orchestration_system.directline_client.close()
        
        await runtime.stop_when_idle()
        
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return

if __name__ == "__main__":
    asyncio.run(interactive_foundry_orchestration())
