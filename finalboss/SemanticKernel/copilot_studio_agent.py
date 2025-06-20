# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
import sys
from collections.abc import AsyncIterable
from typing import Any, ClassVar

if sys.version_info >= (3, 12):
    from typing import override  # pragma: no cover
else:
    from typing_extensions import override  # pragma: no cover

from copilot_studio_channel import CopilotStudioAgentChannel
from copilot_studio_agent_thread import CopilotAgentThread
from copilot_studio_message_content import CopilotMessageContent
from directline_client import DirectLineClient

from semantic_kernel.agents import Agent
from semantic_kernel.agents.agent import AgentResponseItem, AgentThread
from semantic_kernel.agents.channels.agent_channel import AgentChannel
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.exceptions.agent_exceptions import AgentInvokeException
from semantic_kernel.utils.telemetry.agent_diagnostics.decorators import (
    trace_agent_get_response,
    trace_agent_invocation,
)

logger: logging.Logger = logging.getLogger(__name__)


class CopilotAgent(Agent):
    """
    An agent that facilitates communication with a Microsoft Copilot Studio bot via the Direct Line API.
    It serializes user inputs into Direct Line payloads, handles asynchronous response polling, and
    transforms bot activities into structured message content.
    Conversation state such as conversation ID and watermark is externally managed by CopilotAgentThread.
    """

    directline_client: DirectLineClient | None = None

    channel_type: ClassVar[type[AgentChannel]] = CopilotStudioAgentChannel

    def __init__(
        self,
        id: str,
        name: str,
        description: str,
        directline_client: DirectLineClient,
    ) -> None:
        """
        Initialize the CopilotAgent.
        """
        super().__init__(id=id, name=name, description=description)
        self.directline_client = directline_client

    @override
    def get_channel_keys(self) -> list[str]:
        """
        Override to return agent ID instead of channel_type for Copilot agents.

        This is particularly important for CopilotAgent because each agent instance
        maintains its own conversation with a unique thread ID in the DirectLine API.
        Without this override, multiple CopilotAgent instances in a group chat would
        share the same channel, causing thread ID conflicts and message routing issues.

        Returns:
            A list containing the agent ID as the unique channel key, ensuring each
            CopilotAgent gets its own dedicated channel and thread.
        """
        return [self.id]

    @trace_agent_get_response
    @override
    async def get_response(
        self,
        *,
        messages: str | ChatMessageContent | list[str | ChatMessageContent],
        thread: AgentThread | None = None,
        **kwargs,
    ) -> AgentResponseItem[CopilotMessageContent]:
        """
        Get a response from the agent on a thread.

        Args:
            messages: The input chat message content either as a string, ChatMessageContent or
                a list of strings or ChatMessageContent.
            thread: The thread to use for the agent.
            kwargs: Additional keyword arguments.

        Returns:
            AgentResponseItem[ChatMessageContent]: The response from the agent.
        """
        thread = await self._ensure_thread_exists_with_messages(
            messages=messages,
            thread=thread,
            construct_thread=lambda: CopilotAgentThread(directline_client=self.directline_client),
            expected_type=CopilotAgentThread,
        )
        assert thread.id is not None  # nosec

        response_items = []
        async for response_item in self.invoke(
            messages=messages,
            thread=thread,
            **kwargs,
        ):
            response_items.append(response_item)

        if not response_items:
            raise AgentInvokeException("No response messages were returned from the agent.")

        return response_items[-1]

    @trace_agent_invocation
    @override
    async def invoke(
        self,
        *,
        messages: str | ChatMessageContent | list[str | ChatMessageContent],
        thread: AgentThread | None = None,
        message_data: dict[str, Any] | None = None,
        **kwargs,
    ) -> AsyncIterable[AgentResponseItem[CopilotMessageContent]]:
        """Invoke the agent on the specified thread.

        Args:
            messages: The input chat message content either as a string, ChatMessageContent or
                a list of strings or ChatMessageContent.
            thread: The thread to use for the agent.
            message_data: Optional dict that will be sent as the "value" field in the payload
                for adaptive card responses.
            kwargs: Additional keyword arguments.

        Yields:
            AgentResponseItem[ChatMessageContent]: The response from the agent.
        """
        logger.debug("Received messages: %s", messages)
        
        # Handle different message formats
        if isinstance(messages, list):
            # Find the original user message (not transfer messages)
            user_message = None
            for msg in messages:
                if isinstance(msg, str):
                    # Skip transfer messages
                    if not msg.startswith("Transferred to") and not msg.startswith("Which agent should"):
                        user_message = msg
                        break
                elif isinstance(msg, ChatMessageContent) and msg.role == AuthorRole.USER:
                    content = str(msg.content) if msg.content else ""
                    # Skip transfer messages
                    if content and not content.startswith("Transferred to") and not content.startswith("Which agent should"):
                        user_message = content
                        break
            
            if user_message is None:
                # If no suitable user message found, look for any user message
                for msg in reversed(messages):
                    if isinstance(msg, str):
                        user_message = msg
                        break
                    elif isinstance(msg, ChatMessageContent) and msg.role == AuthorRole.USER:
                        user_message = str(msg.content) if msg.content else ""
                        break
                        
                if user_message is None:
                    user_message = "Hello, how can I help you today?"
            
            messages = user_message
        elif isinstance(messages, ChatMessageContent):
            messages = str(messages.content) if messages.content else "Hello, how can I help you today?"
        elif not isinstance(messages, str):
            raise AgentInvokeException("Messages must be a string, ChatMessageContent, or list of messages for Copilot Agent.")

        # Ensure we have a non-empty message and skip transfer messages
        if not messages.strip() or messages.startswith("Transferred to"):
            messages = "Hello, how can I help you today?"
            
        # If this is a career advice request, enhance the message for better response
        if any(keyword in messages.lower() for keyword in ['career', 'job', 'resume', 'interview', 'professional']):
            # Check if the message is too generic and needs enhancement
            generic_patterns = [
                'career advice', 'career help', 'career', 'job advice', 'job help',
                'hi can you give me some career advice', 'can you give me career advice',
                'give me career advice', 'i need career advice'
            ]
            
            message_lower = messages.lower().strip()
            if any(pattern in message_lower for pattern in generic_patterns):
                # Enhance generic career requests to be more specific and actionable
                messages = "I'm looking for comprehensive career guidance. Can you provide me with specific advice on job search strategies, resume optimization, interview preparation, networking tips, and professional development opportunities? I'd like actionable steps I can take to advance my career."
                logger.debug(f"Enhanced generic career request to: {messages}")
            elif 'hi' in message_lower and len(message_lower) < 50:
                # Handle casual greetings with career requests
                messages = "Hello! I need career advice and guidance. Can you help me with job search strategies, resume tips, interview preparation, and professional development?"
                logger.debug(f"Enhanced casual career greeting to: {messages}")

        # Ensure DirectLine client is initialized
        if self.directline_client is None:
            raise AgentInvokeException("DirectLine client is not initialized.")

        thread = await self._ensure_thread_exists_with_messages(
            messages=messages,
            thread=thread,
            construct_thread=lambda: CopilotAgentThread(directline_client=self.directline_client),
            expected_type=CopilotAgentThread,
        )
        assert thread.id is not None  # nosec

        normalized_message = (
            ChatMessageContent(role=AuthorRole.USER, content=messages) if isinstance(messages, str) else messages
        )

        payload = self._build_payload(normalized_message, message_data, thread.id)
        logger.debug("Sending payload to DirectLine: %s", payload)
        
        response_data = await self._send_message(payload, thread)
        logger.debug("Received response from DirectLine: %s", response_data)
        
        if response_data is None or "activities" not in response_data:
            raise AgentInvokeException(f"Invalid response from DirectLine Bot.\n{response_data}")

        # Process DirectLine activities and convert them to appropriate message content
        bot_activities = []
        for activity in response_data["activities"]:
            if activity.get("type") != "message" or activity.get("from", {}).get("id") == "user":
                continue
            bot_activities.append(activity)

        logger.debug("Found %d bot activities", len(bot_activities))
        
        if not bot_activities:
            # If no bot activities, yield a default response
            logger.warning("No bot activities found in response, providing default message")
            default_message = CopilotMessageContent(
                role=AuthorRole.ASSISTANT,
                content="I'm here to help with your career questions. Could you please be more specific about what career advice you're looking for?",
                name=self.name
            )
            yield AgentResponseItem(message=default_message, thread=thread)
            return

        for activity in bot_activities:
            # Create a CopilotMessageContent instance from the activity
            message = CopilotMessageContent.from_bot_activity(activity, name=self.name)

            logger.debug("Response message: %s", message.content)

            yield AgentResponseItem(message=message, thread=thread)

    def _build_payload(
        self,
        message: ChatMessageContent,
        message_data: dict[str, Any] | None = None,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        """Build the message payload for the DirectLine Bot.

        Args:
            message: The message content to send.
            message_data: Optional dict that will be sent as the "value" field in the payload
                for adaptive card responses.
            thread_id: The thread ID (conversation ID).

        Returns:
            A dictionary representing the payload to be sent to the DirectLine Bot.
        """
        payload = {
            "type": "message",
            "from": {"id": "user"},
        }

        if message_data and "adaptive_card_response" in message_data:
            payload["value"] = message_data["adaptive_card_response"]
        else:
            payload["text"] = message.content

        payload["conversationId"] = thread_id
        return payload

    async def _send_message(self, payload: dict[str, Any], thread: CopilotAgentThread) -> dict[str, Any] | None:
        """
        Post the payload to the conversation and poll for responses.
        """
        if self.directline_client is None:
            raise AgentInvokeException("DirectLine client is not initialized.")

        # Post the message payload
        await thread.post_message(payload)
        logger.debug("Message posted successfully")

        # Poll for new activities using watermark with increased timeout and better polling
        finished = False
        collected_data = None
        max_attempts = 30  # Increased from implicit limit
        attempt = 0
        
        while not finished and attempt < max_attempts:
            attempt += 1
            data = await thread.get_messages()
            activities = data.get("activities", [])
            
            logger.debug(f"Polling attempt {attempt}: Found {len(activities)} activities")

            # Look for bot messages (more permissive check)
            bot_messages = []
            for activity in activities:
                if activity.get("type") == "message":
                    from_info = activity.get("from", {})
                    # Check for bot responses (id != "user" and not empty)
                    if from_info.get("id") != "user" and from_info.get("id"):
                        bot_messages.append(activity)
                        logger.debug(f"Found bot message: {activity.get('text', '')[:100]}")

            # Also check for completion events
            has_completion_event = any(
                activity.get("type") == "event" and activity.get("name") in ["DynamicPlanFinished", "BotDone"]
                for activity in activities
            )

            # Consider finished if we have bot messages or completion event
            if bot_messages or has_completion_event:
                collected_data = data
                finished = True
                logger.debug(f"Conversation finished: {len(bot_messages)} bot messages, completion event: {has_completion_event}")
                break

            # Wait before next poll
            await asyncio.sleep(2)  # Increased from 1 second

        if not finished:
            logger.warning(f"Polling timed out after {max_attempts} attempts")
            # Return the last data we got, even if incomplete
            collected_data = data if 'data' in locals() else {"activities": []}

        return collected_data

    async def close(self) -> None:
        """
        Clean up resources.
        """
        if self.directline_client:
            await self.directline_client.close()

    @trace_agent_invocation
    @override
    async def invoke_stream(self, *args, **kwargs):
        """
        Stream responses from the agent using the invoke method.
        This method adapts the invoke method to work with GroupChatOrchestration.
        """
        async for response in self.invoke(*args, **kwargs):
            yield response

    async def create_channel(self, thread_id: str | None = None) -> AgentChannel:
        """Create a Copilot Agent channel.

        Args:
            thread_id: The ID of the thread. If None, a new thread will be created.

        Returns:
            An instance of AgentChannel.
        """
        from .copilot_studio_channel import CopilotStudioAgentChannel

        if self.directline_client is None:
            raise AgentInvokeException("DirectLine client is not initialized.")

        thread = CopilotAgentThread(directline_client=self.directline_client, conversation_id=thread_id)

        if thread.id is None:
            await thread.create()

        return CopilotStudioAgentChannel(thread=thread)