import asyncio
import copy
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.core_plugins import MathPlugin
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)
from semantic_kernel.contents.function_result_content import FunctionResultContent
from semantic_kernel.filters.prompts.prompt_render_context import PromptRenderContext
from semantic_kernel.contents import ChatMessageContent, TextContent, ImageContent
from semantic_kernel.filters.filter_types import FilterTypes
from semantic_kernel.filters.auto_function_invocation.auto_function_invocation_context import (
    AutoFunctionInvocationContext,
)
from semantic_kernel.functions.function_result import FunctionResult


import time

from typing import List, Optional, Annotated
from typing import TypedDict, Optional
import copy

import sys
sys.path.append("./")
sys.path.append("../")

from utils.openai_utils import *
from utils.openai_data_models import *
from utils.file_utils import *
from utils.text_utils import *

from utils.server_data_models import *
from config import Config
from graph_agent import *
from graph_agent_plugin import *


module_directory = os.path.dirname(os.path.abspath(__file__))
CONFIG = Config()



class Orchestrator():

    def __init__(self, chat_history: ChatHistory = None) -> None:
        super().__init__()

        # 1. Create the kernel with the Lights plugin
        service_id=CONFIG.AZURE_OPENAI_MODEL_DEPLOYMENT_NAME

        if chat_history is not None:
            self.chat_history = chat_history
        else:
            self.chat_history = ChatHistory()

        self.graph_manager = MicrosoftGraphPlugin()
        self.orchestrator_system_prompt = None
        self.plugins = [self.graph_manager]

        self.kernel = Kernel()
        self.kernel.add_service(AzureChatCompletion(
            endpoint=CONFIG.AZURE_OPENAI_ENDPOINT,
            api_key=CONFIG.AZURE_OPENAI_API_KEY,
            deployment_name=CONFIG.AZURE_OPENAI_MODEL_DEPLOYMENT_NAME,

            service_id=service_id
        ))

        self.kernel.add_plugin(MathPlugin(), plugin_name="math")
        
        self.kernel.add_plugin(
            self.graph_manager,
            plugin_name="MicrosoftGraphPlugin",
            description=graph_plugin_description,
        )

        self.chat_completion : AzureChatCompletion = self.kernel.get_service(type=ChatCompletionClientBase)

        # FunctionChoiceBehavior.Auto(filters={"included_plugins": ["math", "time"]})
        self.execution_settings = AzureChatPromptExecutionSettings(tool_choice="auto")
        self.execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

        @self.kernel.filter(FilterTypes.AUTO_FUNCTION_INVOCATION)
        async def auto_function_invocation_filter(context: AutoFunctionInvocationContext, next):
            """A filter that will be called for each function call in the response."""
            print(60*"-")
            print("**Automatic Function Call**")
            print("Plugin:", context.function.plugin_name )
            print(f"Function: {context.function.name}")
            print("Function Arguments", context.arguments)            
            # result = context.function_result
            print(60*"-")
            await next(context)


    async def build_system_prompt(self):
        self.orchestrator_system_prompt = read_file(os.path.join(module_directory, 'prompts/orchestrator_system_prompt.txt'))
        self.list_of_users = await self.graph_manager.get_users()
        self.orchestrator_system_prompt = self.orchestrator_system_prompt.format(users=self.list_of_users, default_user="First user in the users list.")


    async def build_chat_history_from_conversation(self, conversation: List[Dict]) -> ChatHistory:

        if self.orchestrator_system_prompt is None:
            await self.build_system_prompt()

        chat_history = ChatHistory()
        
        chat_history.add_message(
            ChatMessageContent(
                role = AuthorRole.SYSTEM, 
                content = self.orchestrator_system_prompt
            )
        )

        print(conversation)

        for message in conversation:
            chat_history.add_message(
                ChatMessageContent(
                    role = AuthorRole.USER if message['role'] == "user" else AuthorRole.ASSISTANT,
                    content = message['content']
                )
            )
        return chat_history



    async def chat(self, query, conversation=None):
        self.interaction_counter = 0

        # Terminate the loop if the user says "exit"
        if query == "exit":
            return "Goodbye!", self.logged_messages
        

        
        if conversation is None:
            chat_history = self.chat_history
        else:
            chat_history = await self.build_chat_history_from_conversation(conversation)

        chat_history.add_message(
            ChatMessageContent(
                role = AuthorRole.USER, 
                content = query
            )
        )

        result = (await self.chat_completion.get_chat_message_contents(
            chat_history=chat_history,
            settings=self.execution_settings,
            kernel=self.kernel,
            arguments=KernelArguments(),
        ))[0]

        chat_history.add_message(
            ChatMessageContent(
                role = AuthorRole.ASSISTANT, 
                content = str(result)
            )
        )

        # Print the results
        print("Assistant > " + str(result))

       # Share final results
        return str(result)