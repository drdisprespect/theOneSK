"""
Microsoft Graph Agent Plugin for Semantic Kernel

This plugin wraps the GraphAgent class functionality in a Semantic Kernel compatible format,
enabling AI agents to interact with Microsoft Graph API services including email, users, 
teams, chats, OneDrive, and more.
"""

import os
import json
import asyncio
from typing import Annotated, List, Dict, Optional, Union, TypedDict
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from rich.console import Console

import sys
sys.path.append("./")
sys.path.append("../")

# Import the GraphAgent class to wrap
from graph_agent import GraphAgent

# Console for rich text output
console = Console()

class UserInfo(TypedDict):
    """Type definition for user information returned by the plugin"""
    id: str
    displayName: str
    mail: Optional[str]

class GraphAgentState(TypedDict):
    """Type definition for storing the internal state of the Graph Agent Plugin"""
    initialized: bool
    user_id: str
    tenant_id: Optional[str]
    client_id: Optional[str]
    users_cache: List[Dict[str, str]]
    
# Define default state
default_state: GraphAgentState = {
    "initialized": False,
    "user_id": "",
    "tenant_id": None,
    "client_id": None,
    "users_cache": []
}



graph_plugin_description = """
Microsoft Graph Plugin for Semantic Kernel

This plugin provides comprehensive access to Microsoft Graph API functionality, enabling AI agents to interact with Microsoft 365 services and data. The plugin wraps the GraphAgent class functionality in a Semantic Kernel compatible format, allowing for seamless integration with AI workflows.

Key capabilities include:

User Management:
- Retrieve organization user directory
- Search and find users by name or email
- Manage user profiles and information
- Cache user data for efficient lookups

Email Operations:
- Send emails with rich text content
- Read and process inbox messages
- Search emails by subject, body, or sender
- Filter messages based on criteria (read, unread, flagged, etc.)

To Do and Task Management:
- Create task lists
- Add and manage tasks
- Set due dates and priorities
- Track task completion status

The plugin handles authentication and session management automatically, using Azure AD credentials to securely access Microsoft Graph API endpoints. It provides a high-level, function-based interface that abstracts away the complexity of direct API interactions.
"""






class MicrosoftGraphPlugin:
    """
    Microsoft Graph Plugin for Semantic Kernel
    
    This plugin provides access to Microsoft Graph API functionality including:
    - User management and search
    - Email operations (reading messages, sending emails)
    - Teams and chat interactions
    - Todo lists and tasks
    - OneDrive file and folder operations
    
    It wraps the GraphAgent class in a Semantic Kernel compatible format.
    """

    def __init__(self, config_path: str = None, state: GraphAgentState = default_state):
        """
        Initialize the Microsoft Graph Plugin
        
        Args:
            config_path: Optional path to a configuration file with Azure AD settings
            state: Optional initial state dictionary
        """
        self.state = state
        self.agent = GraphAgent(config_path=config_path)
        
        # If the agent initialized properly, update our state
        if self.agent:
            self.state["initialized"] = True
            self.state["user_id"] = self.agent.user_id
            self.state["tenant_id"] = self.agent.tenant_id
            self.state["client_id"] = self.agent.client_id

    @kernel_function(
        name="initialize",
        description="Initializes the Microsoft Graph plugin with the specified configuration."
    )
    async def initialize(self, 
                        config_path: Annotated[str, "Path to the configuration file containing Azure AD credentials"] = None) -> Annotated[bool, "True if initialization was successful"]:
        """
        Initializes the Microsoft Graph plugin with the specified configuration.
        Loads Azure AD credentials from the config file and sets up the Graph client.
        
        Args:
            config_path: Path to the configuration file containing Azure AD credentials
            
        Returns:
            True if initialization was successful
        """
        try:
            # Re-initialize the agent with the new config
            self.agent = GraphAgent(config_path=config_path)
            
            # Update state
            self.state["initialized"] = True
            self.state["user_id"] = self.agent.user_id
            self.state["tenant_id"] = self.agent.tenant_id
            self.state["client_id"] = self.agent.client_id
            
            return True
        except Exception as e:
            console.print(f"[red]Error initializing Microsoft Graph plugin: {str(e)}[/red]")
            return False

    @kernel_function(
        name="get_state",
        description="Gets the current state of the Microsoft Graph plugin."
    )
    async def get_state(self) -> Annotated[Dict, "The current state of the Microsoft Graph plugin including initialization status and user information"]:
        """
        Gets the current state of the Microsoft Graph plugin.
        
        Returns:
            A dictionary containing the current plugin state
        """
        return self.state

    @kernel_function(
        name="set_user_id",
        description="Sets the user ID to use for Microsoft Graph operations."
    )
    async def set_user_id(self, 
                         user_id: Annotated[str, "User ID to use for Microsoft Graph operations. Has the format xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxx."]) -> Annotated[bool, "True if user ID was successfully set"]:
        """
        Sets the user ID to use for Microsoft Graph operations.
        
        Args:
            user_id: User ID to use for Microsoft Graph operations
            
        Returns:
            True if user ID was successfully set
        """
        try:
            self.agent.set_user_id(user_id)
            self.state["user_id"] = user_id
            return True
        except Exception as e:
            console.print(f"[red]Error setting user ID: {str(e)}[/red]")
            return False

    @kernel_function(
        name="get_users",
        description="Gets a list of users from Microsoft Graph."
    )
    async def get_users(self, 
                      limit: Annotated[int, "Maximum number of users to return"] = 999) -> Annotated[List[UserInfo], "A list of users with their ID, display name, and email address"]:
        """
        Gets a list of users from Microsoft Graph.
        
        Args:
            limit: Maximum number of users to return
            
        Returns:
            A list of users with their ID, display name, and email address
        """
        try:
            response = await self.agent.get_users(limit=limit)
            users = []
            for user in response.value:
                user_info = {
                    "id": user.id,
                    "displayName": user.display_name,
                    "mail": user.mail if hasattr(user, "mail") else None
                }
                users.append(user_info)
            
            # Update our user cache
            self.state["users_cache"] = users
            
            return users
        except Exception as e:
            console.print(f"[red]Error getting users: {str(e)}[/red]")
            return []

    @kernel_function(
        name="find_user_by_name",
        description="Finds a user by name or email address. The function get_users() has to be called at least once before so that the users are cached."
    )
    async def find_user_by_name(self, 
                             name_or_email: Annotated[str, "Name or email address of the user to find"]) -> Annotated[Optional[str], "The user ID of the matched user, or None if no match was found"]:
        """
        Finds a user by name or email address.
        Uses fuzzy matching and can optionally use AI to find the best match.
        
        Args:
            name_or_email: Name or email address of the user to find
            user_id: Optional user ID to override matching
            
        Returns:
            The user ID of the matched user, or None if no match was found
        """
        try:
            result = await self.agent.find_user_by_name(name_or_email)
            return result
        except Exception as e:
            console.print(f"[red]Error finding user by name: {str(e)}[/red]")
            return None

    @kernel_function(
        name="load_users_cache",
        description="Loads users from Microsoft Graph API and caches them for faster lookups."
    )
    async def load_users_cache(self, 
                           force_refresh: Annotated[bool, "Force refresh of the cache even if it's not expired"] = False,
                           limit: Annotated[int, "Maximum number of users to retrieve"] = 100) -> Annotated[List[Dict], "The cached users"]:
        """
        Loads users from Microsoft Graph API and caches them for faster lookups.
        The cache expires after 1 hour by default.
        
        Args:
            force_refresh: Force refresh of the cache even if it's not expired
            limit: Maximum number of users to retrieve
            
        Returns:
            The cached users
        """
        try:
            users = await self.agent.load_users_cache(force_refresh=force_refresh, limit=limit)
            
            # Update our state with the cached users
            self.state["users_cache"] = users
            
            return users
        except Exception as e:
            console.print(f"[red]Error loading users cache: {str(e)}[/red]")
            return []
            

    @kernel_function(
        name="get_email_by_name",
        description="Gets the email address of a user by their name or email address."
    )
    async def get_email_by_name(self, 
                                name_or_email: Annotated[str, "Name or email address of the user to find"]) -> Annotated[Optional[str], "The email address of the matched user, or None if no match was found"]:
            """
            Gets the email address of a user by their name or email address.
            
            Args:
                name_or_email: Name or email address of the user to find
            """
            try:
                result = await self.agent.get_email_by_name(name_or_email)
                return result
            except Exception as e:
                console.print(f"[red]Error getting email by name: {str(e)}[/red]")
                return None


    @kernel_function(
        name="send_mail",
        description="Sends an email via Microsoft Graph."
    )
    async def send_mail(self, 
                     recipient: Annotated[str, "Recipient email address. The recipient must be a valid user in the organization which includes '@8kcjwb.onmicrosoft.com'. You **MUST NOT** invent new emails"],
                     subject: Annotated[str, "Email subject"] = "Email from Microsoft Graph Plugin",
                     body: Annotated[str, "Email body content"] = "This email was sent using the Microsoft Graph API.",                     
                     user_id: Annotated[str, "Valid User ID of the sender. Has the format xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxx. You MUST use a valid UserID, do NOT make one up. If not known, use the default user ID."]= None) -> Annotated[bool, "True if email was sent successfully"]:
        """
        Sends an email via Microsoft Graph.
        
        Args:
            subject: Email subject
            body: Email body content
            recipient: Recipient email address
            user_id: Optional user ID to send from (will use default if not provided)
            
        Returns:
            True if email was sent successfully
        """
        try:
            await self.agent.send_mail(subject=subject, body=body, recipient=recipient, user_id=user_id)
            return True
        except Exception as e:
            console.print(f"[red]Error sending email: {str(e)}[/red]")
            return False

    @kernel_function(
        name="get_inbox_messages",
        description="Gets inbox messages from Microsoft Graph."
    )
    async def get_inbox_messages(self, 
                             count: Annotated[int, "Number of messages to retrieve"] = 5,
                             filter_criteria: Annotated[Optional[str], "Filter criteria for messages"] = None,
                             search_query: Annotated[Optional[str], "Search query for messages"] = None,
                             user_id: Annotated[str, "Optional user ID. Has the format xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxx."] = None) -> Annotated[List[Dict], "List of messages with their properties"]:
        """
        Gets inbox messages from Microsoft Graph.
        
        Args:
            count: Number of messages to retrieve
            filter_criteria: Filter criteria for messages
            search_query: Search query for messages
            user_id: Optional user ID (will use default if not provided)
            
        Returns:
            List of messages with their properties
        """
        try:
            results = await self.agent.get_inbox_messages(
                count=count, 
                filter_criteria=filter_criteria, 
                search_query=search_query, 
                user_id=user_id
            )
            
            # Convert to dictionary format
            messages = []
            for msg in results.value:
                message_dict = {
                    "id": msg.id,
                    "subject": msg.subject,
                    "isRead": msg.is_read,
                    "receivedDateTime": str(msg.received_date_time) if hasattr(msg, 'received_date_time') else None,
                    "bodyPreview": msg.body_preview if hasattr(msg, 'body_preview') else None,
                }
                
                # Add sender information if available
                if hasattr(msg, 'from') and hasattr(msg.from_property, 'email_address'):
                    message_dict["from"] = {
                        "name": msg.from_property.email_address.name if hasattr(msg.from_property.email_address, 'name') else None,
                        "address": msg.from_property.email_address.address if hasattr(msg.from_property.email_address, 'address') else None
                    }
                
                messages.append(message_dict)
            
            return messages
        except Exception as e:
            console.print(f"[red]Error getting inbox messages: {str(e)}[/red]")
            return []

    @kernel_function(
        name="search_all_emails",
        description="Searches for emails across all folders or in a specific folder."
    )
    async def search_all_emails(self, 
                            search_query: Annotated[str, "The search string to look for in emails"],
                            count: Annotated[int, "Maximum number of messages to retrieve"] = 10,
                            folder_id: Annotated[Optional[str], "Optional specific folder ID to search in"] = None,
                            user_id: Annotated[str, "User ID (ID string not the name or email). Has the format xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxx."] = None) -> Annotated[List[Dict], "List of matching messages with their properties"]:
        """
        Searches for emails across all folders or in a specific folder.
        The search looks in the subject, body, and other fields of the email.
        
        Args:
            search_query: The search string to look for in emails
            count: Maximum number of messages to retrieve
            folder_id: Optional specific folder ID to search in (if None, searches across all folders)
            user_id: Optional user ID (will use default if not provided)
            
        Returns:
            List of matching messages with their properties
        """
        try:
            results = await self.agent.search_all_emails(
                search_query=search_query,
                count=count,
                folder_id=folder_id,
                user_id=user_id
            )
            
            # Convert to dictionary format
            messages = []
            for msg in results.value:
                message_dict = {
                    "id": msg.id,
                    "subject": msg.subject,
                    "isRead": msg.is_read if hasattr(msg, 'is_read') else None,
                    "receivedDateTime": str(msg.received_date_time) if hasattr(msg, 'received_date_time') else None,
                    "bodyPreview": msg.body_preview if hasattr(msg, 'body_preview') else None,
                }
                
                # Add sender information if available
                if hasattr(msg, 'from') and hasattr(msg.from_property, 'email_address'):
                    message_dict["from"] = {
                        "name": msg.from_property.email_address.name if hasattr(msg.from_property.email_address, 'name') else None,
                        "address": msg.from_property.email_address.address if hasattr(msg.from_property.email_address, 'address') else None
                    }
                
                messages.append(message_dict)
            
            return messages
        except Exception as e:
            console.print(f"[red]Error searching emails: {str(e)}[/red]")
            return []

    @kernel_function(
        name="get_mail_folders",
        description="Gets a list of mail folders for a user."
    )
    async def get_mail_folders(self, 
                          user_id: Annotated[str, "User ID. Has the format xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxx."] = None) -> Annotated[List[Dict], "List of mail folders"]:
        """
        Gets a list of mail folders for a user.
        
        Args:
            user_id: Optional user ID (will use default if not provided)
            
        Returns:
            List of mail folders with their properties
        """
        try:
            results = await self.agent.get_mail_folders(user_id=user_id)
            
            # Convert to dictionary format
            folders = []
            for folder in results.value:
                folder_dict = {
                    "id": folder.id,
                    "displayName": folder.display_name,
                    "parentFolderId": folder.parent_folder_id if hasattr(folder, 'parent_folder_id') else None,
                    "totalItemCount": folder.total_item_count if hasattr(folder, 'total_item_count') else None,
                    "unreadItemCount": folder.unread_item_count if hasattr(folder, 'unread_item_count') else None
                }
                folders.append(folder_dict)
            
            return folders
        except Exception as e:
            console.print(f"[red]Error getting mail folders: {str(e)}[/red]")
            return []

    @kernel_function(
        name="create_todo_list",
        description="Creates a To Do task list."
    )
    async def create_todo_list(self, 
                            display_name: Annotated[str, "Name for the task list"] = "List created by Microsoft Graph Plugin",
                            user_id: Annotated[str, "Optional user ID. Has the format xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxx."] = None) -> Annotated[Dict, "The created task list"]:
        """
        Creates a To Do task list.
        
        Args:
            display_name: Name for the task list
            user_id: Optional user ID (will use default if not provided)
            
        Returns:
            The created task list with its properties
        """
        try:
            todo_list = await self.agent.create_todo_list(display_name=display_name, user_id=user_id)
            
            # Convert to dictionary
            list_dict = {
                "id": todo_list.id,
                "displayName": todo_list.display_name,
                "isShared": todo_list.is_shared if hasattr(todo_list, 'is_shared') else None,
                "isOwner": todo_list.is_owner if hasattr(todo_list, 'is_owner') else None,
                "wellknownListName": todo_list.wellknown_list_name if hasattr(todo_list, 'wellknown_list_name') else None
            }
            
            return list_dict
        except Exception as e:
            console.print(f"[red]Error creating Todo list: {str(e)}[/red]")
            return {"error": str(e)}

    @kernel_function(
        name="create_todo_task",
        description="Creates a To Do task in a specified list. You **MUST** retrieve the ToDo lists first."
    )
    async def create_todo_task(self, 
                            list_id: Annotated[str, "ID of the task list.You **MUST** retrieve the ToDo lists first."],
                            title: Annotated[str, "Title for the task"] = "Task created by Microsoft Graph Plugin",
                            user_id: Annotated[str, "Optional user ID. Has the format xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxx."] = None) -> Annotated[Dict, "The created task"]:
        """
        Creates a To Do task in a specified list.
        
        Args:
            list_id: ID of the task list
            title: Title for the task
            user_id: Optional user ID (will use default if not provided)
            
        Returns:
            The created task with its properties
        """
        try:
            task = await self.agent.create_todo_task(list_id=list_id, title=title, user_id=user_id)
            
            # Convert to dictionary
            task_dict = {
                "id": task.id,
                "title": task.title,
                "status": task.status if hasattr(task, 'status') else None,
                "importance": task.importance if hasattr(task, 'importance') else None,
                "createdDateTime": str(task.created_date_time) if hasattr(task, 'created_date_time') else None,
                "lastModifiedDateTime": str(task.last_modified_date_time) if hasattr(task, 'last_modified_date_time') else None
            }
            
            return task_dict
        except Exception as e:
            console.print(f"[red]Error creating Todo task: {str(e)}[/red]")
            return {"error": str(e)}

    @kernel_function(
        name="get_todo_tasks_from_list",
        description="Gets the list of To Do tasks from a specified list. You **MUST** retrieve the ToDo lists first."
    )
    async def get_todo_tasks_from_list(self, 
                            list_id: Annotated[str, "ID of the task list.You **MUST** retrieve the ToDo lists first."],
                            user_id: Annotated[str, "Optional user ID. Has the format xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxx."] = None) -> Annotated[Dict, "The created task"]:
        """
        Get all tasks from a To Do list
        
        Args:
            list_id: ID of the task list
            user_id: Optional user ID (will use default if not provided)
        
        Returns:
            The list of todo tasks in a todo list
        """
        try:
            tasks = await self.agent.get_todo_tasks_from_list(list_id=list_id, user_id=user_id)

            tasks_list = []
            for task in tasks.value:
                task_dict = {
                    "id": task.id,
                    "title": task.title,
                    "status": task.status if hasattr(task, 'status') else None,
                    "importance": task.importance if hasattr(task, 'importance') else None,
                    "createdDateTime": str(task.created_date_time) if hasattr(task, 'created_date_time') else None,
                    "lastModifiedDateTime": str(task.last_modified_date_time) if hasattr(task, 'last_modified_date_time') else None
                }
                tasks_list.append(task_dict)
            
            return tasks_list
        except Exception as e:
            console.print(f"[red]Error getting Todo tasks: {str(e)}[/red]")
            return {"error": str(e)}
        
    @kernel_function(
        name="get_todo_lists",
        description="Gets all To Do lists for a user."
    )
    async def get_todo_lists(self, 
                          user_id: Annotated[str, "Optional user ID. Has the format xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxx."] = None) -> Annotated[List[Dict], "List of To Do lists"]:
        """
        Gets all To Do lists for a user.
        
        Args:
            user_id: Optional user ID (will use default if not provided)
            
        Returns:
            List of To Do lists with their properties
        """
        try:
            results = await self.agent.get_todo_lists(user_id=user_id)
            
            # Convert to dictionary format
            lists = []
            for lst in results.value:
                list_dict = {
                    "id": lst.id,
                    "displayName": lst.display_name,
                    "isShared": lst.is_shared if hasattr(lst, 'is_shared') else None,
                    "isOwner": lst.is_owner if hasattr(lst, 'is_owner') else None,
                    "wellknownListName": lst.wellknown_list_name if hasattr(lst, 'wellknown_list_name') else None
                }
                lists.append(list_dict)
            
            return lists
        except Exception as e:
            console.print(f"[red]Error getting Todo lists: {str(e)}[/red]")
            return []

    @kernel_function(
        name="create_folder",
        description="Creates a folder in OneDrive."
    )
    async def create_folder(self, 
                         drive_id: Annotated[str, "ID of the drive"],
                         parent_item_id: Annotated[str, "ID of the parent item"],
                         folder_name: Annotated[str, "Name for the folder"] = "Folder created by Microsoft Graph Plugin") -> Annotated[Dict, "The created folder"]:
        """
        Creates a folder in OneDrive.
        
        Args:
            drive_id: ID of the drive
            parent_item_id: ID of the parent item
            folder_name: Name for the folder
            
        Returns:
            The created folder with its properties
        """
        try:
            folder = await self.agent.create_folder(
                drive_id=drive_id,
                parent_item_id=parent_item_id,
                folder_name=folder_name
            )
            
            # Convert to dictionary
            folder_dict = {
                "id": folder.id,
                "name": folder.name,
                "webUrl": folder.web_url if hasattr(folder, 'web_url') else None,
                "createdDateTime": str(folder.created_date_time) if hasattr(folder, 'created_date_time') else None,
                "lastModifiedDateTime": str(folder.last_modified_date_time) if hasattr(folder, 'last_modified_date_time') else None
            }
            
            return folder_dict
        except Exception as e:
            console.print(f"[red]Error creating folder: {str(e)}[/red]")
            return {"error": str(e)}

    @kernel_function(
        name="get_user_drive",
        description="Gets a user's default OneDrive."
    )
    async def get_user_drive(self, 
                          user_id: Annotated[str, "Optional user ID. Has the format xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxx."] = None) -> Annotated[Dict, "The user's OneDrive information"]:
        """
        Gets a user's default OneDrive.
        
        Args:
            user_id: Optional user ID (will use default if not provided)
            
        Returns:
            The user's OneDrive information including ID, name, and web URL
        """
        try:
            drive = await self.agent.get_user_drive(user_id=user_id)
            
            # Convert to dictionary
            drive_dict = {
                "id": drive.id,
                "name": drive.name if hasattr(drive, 'name') else None,
                "driveType": drive.drive_type if hasattr(drive, 'drive_type') else None,
                "webUrl": drive.web_url if hasattr(drive, 'web_url') else None,
                "createdDateTime": str(drive.created_date_time) if hasattr(drive, 'created_date_time') else None,
                "lastModifiedDateTime": str(drive.last_modified_date_time) if hasattr(drive, 'last_modified_date_time') else None
            }
            
            return drive_dict
        except Exception as e:
            console.print(f"[red]Error getting user drive: {str(e)}[/red]")
            return {"error": str(e)}

    @kernel_function(
        name="get_drive_root",
        description="Gets the root folder of a drive."
    )
    async def get_drive_root(self, 
                          drive_id: Annotated[str, "ID of the drive"]) -> Annotated[Dict, "The root folder of the drive"]:
        """
        Gets the root folder of a drive.
        
        Args:
            drive_id: ID of the drive
            
        Returns:
            The root folder of the drive with its properties
        """
        try:
            root = await self.agent.get_drive_root(drive_id=drive_id)
            
            # Convert to dictionary
            root_dict = {
                "id": root.id,
                "name": root.name if hasattr(root, 'name') else None,
                "webUrl": root.web_url if hasattr(root, 'web_url') else None,
                "createdDateTime": str(root.created_date_time) if hasattr(root, 'created_date_time') else None,
                "lastModifiedDateTime": str(root.last_modified_date_time) if hasattr(root, 'last_modified_date_time') else None
            }
            
            return root_dict
        except Exception as e:
            console.print(f"[red]Error getting drive root: {str(e)}[/red]")
            return {"error": str(e)}

  