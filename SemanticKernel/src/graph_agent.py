import os
import asyncio
import configparser
from rich.console import Console
console = Console()
import re
from pydantic import BaseModel

# Microsoft Graph imports
from azure.identity.aio import ClientSecretCredential
from msgraph import GraphServiceClient
from msgraph.generated.users.item.user_item_request_builder import UserItemRequestBuilder
from msgraph.generated.users.item.mail_folders.item.messages.messages_request_builder import MessagesRequestBuilder
from msgraph.generated.users.item.send_mail.send_mail_post_request_body import SendMailPostRequestBody
from msgraph.generated.models.message import Message
from msgraph.generated.models.item_body import ItemBody
from msgraph.generated.models.body_type import BodyType
from msgraph.generated.models.recipient import Recipient
from msgraph.generated.models.email_address import EmailAddress
from msgraph.generated.users.item.messages.messages_request_builder import MessagesRequestBuilder as UserMessagesRequestBuilder
from msgraph.generated.users.users_request_builder import UsersRequestBuilder
from msgraph.generated.models.chat import Chat
from msgraph.generated.models.chat_type import ChatType
from msgraph.generated.models.conversation_member import ConversationMember
from msgraph.generated.models.aad_user_conversation_member import AadUserConversationMember
from msgraph.generated.models.todo_task_list import TodoTaskList
from msgraph.generated.models.todo_task import TodoTask
from msgraph.generated.models.drive_item import DriveItem
from msgraph.generated.models.folder import Folder
from kiota_abstractions.base_request_configuration import RequestConfiguration
from msgraph.generated.models.team import Team
from msgraph.generated.models.chat_message import ChatMessage
from msgraph.generated.models.chat_message_attachment import ChatMessageAttachment

# Additional imports
from dotenv import load_dotenv
from config import Config
config = Config()



import sys
sys.path.append("./")
sys.path.append("../")


# Import OpenAI utilities for user matching
from utils.openai_utils import call_llm_structured_outputs
from utils.openai_data_models import TextProcessingModelnfo
HAS_OPENAI = True




class GraphAgent:
    """
    A class that encapsulates Microsoft Graph API functionality from the Jupyter notebook.
    """
    def __init__(self, config_path=None):
        """
        Initialize the GraphAgent
        
        Args:
            config_path: Path to the config file. If not provided, will use default config.cfg
        """
        # Initialize console for rich output
        self.console = Console()
        self.client_id = None
        self.tenant_id = None
        self.client_secret = None
        
        # Load environment variables from .env file if it exists
        load_dotenv()
        
        # Load settings from config file
        self.config = configparser.ConfigParser()
        
        if config_path:
            self.config.read([config_path, f"{os.path.splitext(config_path)[0]}.dev.cfg"])
        else:
            self.config.read(['config.cfg', 'config.dev.cfg'])
        
        if 'azure' in self.config:
            self.azure_settings = self.config['azure']
            
            # Get Azure AD settings
            self.client_id = self.azure_settings['clientId']
            self.tenant_id = self.azure_settings['tenantId']
            self.client_secret = self.azure_settings['clientSecret']

        if self.client_id is None:
            self.client_id = config.clientId
            self.tenant_id = config.tenantId
            self.client_secret = config.clientSecret
        else:
            self.client_id = os.get_environ('clientId')
            self.tenant_id = os.get_environ('tenantId')
            self.client_secret = os.get_environ('clientSecret')


        console.print(f"Client ID: {self.client_id}")
        console.print(f"Tenant ID: {self.tenant_id}")
        
        # Set default user ID for operations
        self.user_id = ''
        
        # Store users list
        self.users_cache = []
        self.users_cache_time = None
        
        # Initialize Graph client
        self.client_credential = ClientSecretCredential(self.tenant_id, self.client_id, self.client_secret)
        self.app_client = GraphServiceClient(self.client_credential)
    

    
    async def load_users_cache(self, force_refresh=False, limit=100):
        """
        Load users from Microsoft Graph API and cache them
        
        Args:
            force_refresh: Force refresh the cache
            limit: Maximum number of users to retrieve
            
        Returns:
            List of users
        """
        import datetime
        
        # Check if we need to refresh the cache
        current_time = datetime.datetime.now()
        cache_expiry = datetime.timedelta(hours=1)
        
        if not self.users_cache or force_refresh or (self.users_cache_time and current_time - self.users_cache_time > cache_expiry):
            users = await self.get_users(limit=limit)
            
            # Store users in a more accessible format
            self.users_cache = []
            for user in users.value:
                user_data = {
                    'id': user.id,
                    'displayName': user.display_name,
                    'mail': user.mail if hasattr(user, 'mail') else None
                }
                self.users_cache.append(user_data)
                
            # Update cache time
            self.users_cache_time = current_time
            
        return self.users_cache
    
    async def find_user_by_name(self, name_or_email):
        """
        Find a user by name or email using fuzzy matching
        If user_id is provided, it will be returned (overriding the name matching)
        
        Args:
            name_or_email: Name or email to match
            user_id: Optional user_id to override matching
            
        Returns:
            User ID of the matched user
        """
            
        # Make sure we have users loaded
        await self.load_users_cache()
        
        # First try exact match on email
        if '@' in name_or_email:
            for user in self.users_cache:
                if user['mail'] and user['mail'].lower() == name_or_email.lower():
                    return user['id']
        
        # Try exact match on display name
        for user in self.users_cache:
            if user['displayName'] and user['displayName'].lower() == name_or_email.lower():
                return user['id']
        
        # Try substring match
        matches = []
        for user in self.users_cache:
            name_matched = user['displayName'] and name_or_email.lower() in user['displayName'].lower()
            email_matched = user['mail'] and name_or_email.lower() in user['mail'].lower()
            
            if name_matched or email_matched:
                matches.append(user)
        
        # If we found exactly one match, use it
        if len(matches) == 1:
            return matches[0]['id']
            
        # If we have multiple matches and OpenAI is available, use AI to find the best match
        if len(matches) > 1 and HAS_OPENAI:
            try:
                best_match = await self.match_user_with_llm(name_or_email, matches)
                if best_match:
                    return best_match['id']
            except Exception as e:
                self.console.print(f"[yellow]Error using LLM for user matching: {e}[/yellow]")
        
        # If no match or multiple matches and no OpenAI, use the default user
        if len(matches) == 0:
            self.console.print(f"[yellow]No user found matching '{name_or_email}'. Using default user.[/yellow]")
            return self.users_cache[0]['id']
        elif len(matches) > 1:
            self.console.print(f"[yellow]Multiple users found matching '{name_or_email}'. Using first match: {matches[0]['displayName']}[/yellow]")
            return matches[0]['id']
            
    async def match_user_with_llm(self, query, users_list):
        """
        Use OpenAI to match a user query to the most likely user
        
        Args:
            query: Query string (name or partial name/email)
            users_list: List of user dictionaries to match against
            
        Returns:
            Best matching user or None
        """
        if not HAS_OPENAI:
            return None
            
        # Format the users for the prompt
        users_text = "\n".join([f"ID: {u['id']}, Name: {u['displayName']}, Email: {u['mail']}" for u in users_list])

        class BestMatchingUserId(BaseModel):
            id: str


        prompt = f"""
        I need to find the best matching user from the list below based on the query: "{query}"
        
        USERS LIST:
        {users_text}
        
        Compare the query against each user's name and email. Consider:
        - Exact matches should be prioritized
        - Partial name or email matches are acceptable
        - Names that sound similar phonetically can be a good match
        - Abbreviated names (like "Bob" for "Robert") should be considered
        
        Return only the ID of the best matching user. Return just the ID as a string with no other text.
        If no good match is found, say "NO_MATCH"

        {
            "id": "ID of the best matching user"
        }
        """
        
        try:
            # Use default model settings
            model_info = TextProcessingModelnfo()
            
            # Call the LLM
            result = await call_llm_structured_outputs(prompt, model_info, response_format=BestMatchingUserId)
            
            if result.id == None:
                return None
            
            # Try to match the returned ID to one in our list
            for user in users_list:
                if user['id'] == result.id:
                    return user
            
            # If we didn't find a direct match, just return the first user as fallback
            return users_list[0] if users_list else None
            
        except Exception as e:
            self.console.print(f"[yellow]Error in LLM matching: {e}[/yellow]")
            return None
    

    async def get_email_by_name(self, name):
        """
        Get a user's email address by name
        """
        # Make sure we have users loaded
        await self.load_users_cache()
        recipient = name

        if "@" not in name:
            target_user_id = await self.find_user_by_name(name)
            for user in self.users_cache:
                if user['id'] == target_user_id:
                    recipient = user['mail']
                    break
        elif not "@8kcjwb.onmicrosoft.com" in name:
            target_user_id = await self.find_user_by_name(name.split("@"))
            for user in self.users_cache:
                if user['id'] == target_user_id:
                    recipient = user['mail']
                    break

        return recipient


    async def send_mail(self, subject='Testing Microsoft Graph', body='Hello world!', recipient='samer@8kcjwb.onmicrosoft.com', user_id=None):
        """
        Send an email via Microsoft Graph
        
        Args:
            subject: Email subject
            body: Email body content
            recipient: Recipient email address
            user_id: Optional user ID to send from (will use default if not provided)
        """

        recipient = await self.get_email_by_name(recipient)

        # Create the message
        message = Message()
        message.subject = subject

        message.body = ItemBody()
        message.body.content_type = BodyType.Text
        message.body.content = body

        to_recipient = Recipient()
        to_recipient.email_address = EmailAddress()
        to_recipient.email_address.address = recipient
        message.to_recipients = []
        message.to_recipients.append(to_recipient)

        # Create the SendMailPostRequestBody
        request_body = SendMailPostRequestBody()
        request_body.message = message

        # Send the mail
        await self.app_client.users.by_user_id(user_id).send_mail.post(body=request_body)
        self.console.print(f"Email sent to {recipient}")
    
    async def get_inbox_messages(self, count=5, filter_criteria=None, search_query=None, user_id=None):
        """
        Get inbox messages
        
        Args:
            count: Number of messages to retrieve
            filter_criteria: Filter criteria for messages
            search_query: Search query for messages
            user_id: Optional user ID (will use default if not provided)
        
        Returns:
            The message collection response
        """
        # Find the appropriate user
        # target_user_id = await self.find_user_by_name("", user_id)
        
        if search_query:
            # Use search query to find messages
            query_params = UserMessagesRequestBuilder.MessagesRequestBuilderGetQueryParameters(
                search = f"\"{search_query}\""
            )
        else:
            # Use standard query with ordering and selection
            query_params = MessagesRequestBuilder.MessagesRequestBuilderGetQueryParameters(
                orderby = ["receivedDateTime desc"],
                select = ["id", "subject", "from", "isRead", "bodyPreview", "receivedDateTime"],
                top = count
            )
            
            # Add filter if provided
            if filter_criteria:
                query_params.filter = filter_criteria

        request_configuration = RequestConfiguration(
            query_parameters = query_params
        )

        # Get messages from inbox folder
        result = await self.app_client.users.by_user_id(user_id).mail_folders.by_mail_folder_id('inbox').messages.get(
            request_configuration = request_configuration
        )
        
        return result
        
    async def search_all_emails(self, search_query, count=30, folder_id=None, user_id=None):
        """
        Search for emails across all folders or in a specific folder
        
        Args:
            search_query: The search string to look for in emails
            count: Maximum number of messages to retrieve (default: 10)
            folder_id: Optional specific folder ID to search in (default: None, searches across all folders)
            user_id: Optional user ID (will use default if not provided)
        
        Returns:
            The message collection response containing matching messages
        """
        # Find the appropriate user
        # target_user_id = await self.find_user_by_name("", user_id)
        
        # Set up query parameters for search
        query_params = UserMessagesRequestBuilder.MessagesRequestBuilderGetQueryParameters(
            search = f"\"{search_query}\"",
            top = count,
            select = ["id", "subject", "from", "receivedDateTime", "bodyPreview", "isRead"]
        )
        
        request_configuration = RequestConfiguration(
            query_parameters = query_params
        )
        
        # If a specific folder is provided, search in that folder
        if folder_id:
            result = await self.app_client.users.by_user_id(user_id).mail_folders.by_mail_folder_id(folder_id).messages.get(
                request_configuration = request_configuration
            )
        else:
            # Search across all messages
            result = await self.app_client.users.by_user_id(user_id).messages.get(
                request_configuration = request_configuration
            )
        
        return result
    
    async def get_mail_folders(self, user_id=None):
        """
        Get a list of mail folders for a user
        
        Args:
            user_id: Optional user ID (will use default if not provided)
            
        Returns:
            The collection of mail folders
        """
        # Find the appropriate user
        # target_user_id = await self.find_user_by_name("", user_id)
        
        # Get mail folders
        result = await self.app_client.users.by_user_id(user_id).mail_folders.get()
        return result
    
    
    
    async def create_todo_list(self, display_name="List created from Microsoft Graph Explorer", user_id=None):
        """
        Create a To Do task list
        
        Args:
            display_name: Name for the task list
            user_id: Optional user ID (will use default if not provided)
        
        Returns:
            The created task list
        """
        # Find the appropriate user
        # target_user_id = await self.find_user_by_name("", user_id)
        
        request_body = TodoTaskList(
            display_name = display_name
        )

        result = await self.app_client.users.by_user_id(user_id).todo.lists.post(request_body)
        return result
    
    async def create_todo_task(self, list_id, title="Task created from Microsoft Graph Explorer", user_id=None):
        """
        Create a To Do task
        
        Args:
            list_id: ID of the task list
            title: Title for the task
            user_id: Optional user ID (will use default if not provided)
        
        Returns:
            The created task
        """
        # Find the appropriate user
        # target_user_id = await self.find_user_by_name("", user_id)
        
        request_body = TodoTask(
            title = title
        )

        result = await self.app_client.users.by_user_id(user_id).todo.lists.by_todo_task_list_id(list_id).tasks.post(request_body)
        return result
    
    async def get_todo_tasks_from_list(self, list_id, user_id=None):
        """
        Get all tasks from a To Do list
        
        Args:
            list_id: ID of the task list
            user_id: Optional user ID (will use default if not provided)
        
        Returns:
            The list of todo tasks in a todo list
        """
        result = await self.app_client.users.by_user_id(user_id).todo.lists.by_todo_task_list_id(list_id).tasks.get()
        return result
    
    async def get_todo_lists(self, user_id=None):
        """
        Get all To Do lists
        
        Args:
            user_id: Optional user ID (will use default if not provided)
            
        Returns:
            List of To Do lists
        """
        # Find the appropriate user
        # target_user_id = await self.find_user_by_name("", user_id)
        
        result = await self.app_client.users.by_user_id(user_id).todo.lists.get()
        return result
    
    async def create_folder(self, drive_id, parent_item_id, folder_name="New Folder", user_id=None):
        """
        Create a folder in OneDrive
        
        Args:
            drive_id: ID of the drive
            parent_item_id: ID of the parent item
            folder_name: Name for the folder
            user_id: Optional user ID (will use default if not provided)
        
        Returns:
            The created folder
        """
        # Note: For OneDrive operations, we're using the drives API which doesn't use /me
        request_body = DriveItem(
            name = folder_name,
            folder = Folder()
        )

        result = await self.app_client.drives.by_drive_id(drive_id).items.by_drive_item_id(parent_item_id).children.post(request_body)
        return result
    
    async def get_user_drive(self, user_id=None):
        """
        Get a user's default drive
        
        Args:
            user_id: Optional user ID (will use default if not provided)
            
        Returns:
            The user's default drive
        """
        # Find the appropriate user
        # target_user_id = await self.find_user_by_name("", user_id)
        
        drive = await self.app_client.users.by_user_id(user_id).drive.get()
        return drive
    
    async def get_drive_root(self, drive_id):
        """
        Get the root folder of a drive
        
        Args:
            drive_id: ID of the drive
            
        Returns:
            The root folder of the drive
        """
        root = await self.app_client.drives.by_drive_id(drive_id).root.get()
        return root
  
    
    async def get_users(self, limit=999):
        """
        Get a list of users
        
        Args:
            limit: Maximum number of users to return
            
        Returns:
            Collection of users
        """
        query_params = UsersRequestBuilder.UsersRequestBuilderGetQueryParameters(
            # Only request specific properties
            select = ['displayName', 'id', 'mail'],
            # Get limited results
            top = limit,
            # Sort by display name
            orderby = ['displayName']
        )
        
        request_config = RequestConfiguration(
            query_parameters = query_params
        )

        users = await self.app_client.users.get(request_configuration=request_config)
        return users
    
    def set_user_id(self, user_id):
        """
        Set the default user ID for operations
        
        Args:
            user_id: User ID to use for operations
        """
        self.user_id = user_id


