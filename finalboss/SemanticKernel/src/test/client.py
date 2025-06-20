#!/usr/bin/env python3
"""
Microsoft Graph API Test Client

This script provides a command-line interface to test the Microsoft Graph API server.
It allows you to interact with the server's endpoints for user management, email operations,
todo lists, and other Microsoft Graph functionality.
"""

import argparse
import json
import sys
import requests
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich import box

# Initialize console for rich output
console = Console()

# Default server URL
DEFAULT_SERVER_URL = "http://localhost:8002"

class MicrosoftGraphClient:
    """Client for interacting with the Microsoft Graph API server"""
    
    def __init__(self, server_url: str = DEFAULT_SERVER_URL):
        """
        Initialize the Microsoft Graph client
        
        Args:
            server_url: URL of the Microsoft Graph API server
        """
        self.server_url = server_url
        self.session = requests.Session()
        self.conversation_id = None
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check if the server is healthy
        
        Returns:
            Server health status
        """
        try:
            response = self.session.get(f"{self.server_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            console.print(f"[red]Health check failed: {str(e)}[/red]")
            return {"status": "error", "message": str(e)}
    
    def readiness_check(self) -> Dict[str, Any]:
        """
        Check if the server is ready to handle requests
        
        Returns:
            Server readiness status
        """
        try:
            response = self.session.get(f"{self.server_url}/health/readiness")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            console.print(f"[red]Readiness check failed: {str(e)}[/red]")
            return {"status": "error", "message": str(e)}
    
    def send_chat_message(self, message: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Send a chat message to the Microsoft Graph assistant
        
        Args:
            message: The message to send
            user_id: Optional user ID to identify the sender
        
        Returns:
            The assistant's response
        """
        try:
            payload = {
                "message": message,
                "conversation_id": self.conversation_id
            }
            
            if user_id:
                payload["user_id"] = user_id
            
            response = self.session.post(f"{self.server_url}/chat", json=payload)
            response.raise_for_status()
            
            result = response.json()
            
            # Update the conversation ID for future messages
            self.conversation_id = result.get("conversation_id")
            
            return result
        except requests.exceptions.RequestException as e:
            console.print(f"[red]Error sending chat message: {str(e)}[/red]")
            if hasattr(e, "response") and e.response is not None:
                console.print(f"[red]Response: {e.response.text}[/red]")
            return {"error": str(e)}
    
    def get_conversation(self, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the history of a conversation
        
        Args:
            conversation_id: ID of the conversation to retrieve
        
        Returns:
            Conversation history
        """
        try:
            conv_id = conversation_id or self.conversation_id
            if not conv_id:
                return {"error": "No conversation ID specified"}
            
            response = self.session.get(f"{self.server_url}/conversations/{conv_id}")
            response.raise_for_status()
            
            return response.json()
        except requests.exceptions.RequestException as e:
            console.print(f"[red]Error getting conversation: {str(e)}[/red]")
            return {"error": str(e)}
    
    def list_conversations(self) -> Dict[str, Any]:
        """
        List all conversation IDs
        
        Returns:
            List of conversation IDs
        """
        try:
            response = self.session.get(f"{self.server_url}/conversations")
            response.raise_for_status()
            
            return response.json()
        except requests.exceptions.RequestException as e:
            console.print(f"[red]Error listing conversations: {str(e)}[/red]")
            return {"error": str(e)}
    
    def delete_conversation(self, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Delete a conversation
        
        Args:
            conversation_id: ID of the conversation to delete
        
        Returns:
            Result of the operation
        """
        try:
            conv_id = conversation_id or self.conversation_id
            if not conv_id:
                return {"error": "No conversation ID specified"}
            
            response = self.session.delete(f"{self.server_url}/conversations/{conv_id}")
            response.raise_for_status()
            
            # Clear the conversation ID if we deleted the current conversation
            if conv_id == self.conversation_id:
                self.conversation_id = None
            
            return response.json()
        except requests.exceptions.RequestException as e:
            console.print(f"[red]Error deleting conversation: {str(e)}[/red]")
            return {"error": str(e)}
    
    def clear_conversation(self, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Clear the message history of a conversation
        
        Args:
            conversation_id: ID of the conversation to clear
        
        Returns:
            Result of the operation
        """
        try:
            conv_id = conversation_id or self.conversation_id
            if not conv_id:
                return {"error": "No conversation ID specified"}
            
            response = self.session.post(f"{self.server_url}/conversations/{conv_id}/clear")
            response.raise_for_status()
            
            return response.json()
        except requests.exceptions.RequestException as e:
            console.print(f"[red]Error clearing conversation: {str(e)}[/red]")
            return {"error": str(e)}


def print_conversation(conversation: Dict[str, Any]) -> None:
    """
    Print a conversation in a nice format
    
    Args:
        conversation: Conversation to print
    """
    if "error" in conversation:
        console.print(f"[red]Error: {conversation['error']}[/red]")
        return
    
    if "messages" not in conversation:
        console.print("[yellow]No messages in conversation[/yellow]")
        return
    
    messages = conversation["messages"]
    console.print(f"[bold]Conversation: {conversation['conversation_id']}[/bold] ({len(messages)} messages)")
    console.print("-" * 80)
    
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        timestamp = msg.get("timestamp", "")
        
        if role == "user":
            console.print(Panel(content, title=f"User ({timestamp})", border_style="blue"))
        elif role == "assistant":
            console.print(Panel(content, title=f"Assistant ({timestamp})", border_style="green"))
        else:
            console.print(Panel(content, title=f"{role} ({timestamp})", border_style="yellow"))
        
        console.print("")


def print_conversations_list(conversations: Dict[str, Any]) -> None:
    """
    Print a list of conversations in a table
    
    Args:
        conversations: List of conversations to print
    """
    if "error" in conversations:
        console.print(f"[red]Error: {conversations['error']}[/red]")
        return
    
    if "conversations" not in conversations or not conversations["conversations"]:
        console.print("[yellow]No conversations found[/yellow]")
        return
    
    table = Table(title="Conversations", box=box.ROUNDED)
    table.add_column("ID", style="cyan")
    table.add_column("Messages", style="green")
    table.add_column("Last Updated", style="magenta")
    
    for conv in conversations["conversations"]:
        table.add_row(
            conv["id"],
            str(conv["message_count"]),
            conv["last_updated"] or "N/A"
        )
    
    console.print(table)


def chat_mode(client: MicrosoftGraphClient) -> None:
    """
    Enter interactive chat mode with the Microsoft Graph assistant
    
    Args:
        client: Microsoft Graph client instance
    """
    console.print("[bold blue]Interactive Chat Mode with Microsoft Graph Assistant[/bold blue]")
    console.print("[italic]Type 'exit' or 'quit' to exit, 'help' for commands[/italic]\n")
    
    while True:
        user_input = console.input("[bold blue]You:[/bold blue] ")
        
        # Check for special commands
        if user_input.lower() in ["exit", "quit"]:
            break
        elif user_input.lower() == "help":
            console.print(Panel("""
            [bold]Available Commands:[/bold]
            
            - exit, quit: Exit chat mode
            - help: Show this help message
            - history: Show conversation history
            - clear: Clear conversation history
            - delete: Delete this conversation
            - list: List all conversations
            """, title="Help", border_style="green"))
            continue
        elif user_input.lower() == "history":
            if client.conversation_id:
                conversation = client.get_conversation()
                print_conversation(conversation)
            else:
                console.print("[yellow]No active conversation[/yellow]")
            continue
        elif user_input.lower() == "clear":
            if client.conversation_id:
                result = client.clear_conversation()
                console.print(f"[green]{result.get('message', 'Conversation cleared')}[/green]")
            else:
                console.print("[yellow]No active conversation[/yellow]")
            continue
        elif user_input.lower() == "delete":
            if client.conversation_id:
                result = client.delete_conversation()
                console.print(f"[green]{result.get('message', 'Conversation deleted')}[/green]")
            else:
                console.print("[yellow]No active conversation[/yellow]")
            continue
        elif user_input.lower() == "list":
            conversations = client.list_conversations()
            print_conversations_list(conversations)
            continue
        
        # Send the user's message to the assistant
        response = client.send_chat_message(user_input)
        
        if "error" in response:
            console.print(f"[red]Error: {response['error']}[/red]")
        else:
            console.print(f"\n[bold green]Assistant:[/bold green] {response['response']}\n")


def main() -> None:
    """Main function for the Microsoft Graph API test client"""
    parser = argparse.ArgumentParser(description="Microsoft Graph API Test Client")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Chat parser
    chat_parser = subparsers.add_parser("chat", help="Enter interactive chat mode with the assistant")
    chat_parser.add_argument("--server", default=DEFAULT_SERVER_URL, help=f"Server URL (default: {DEFAULT_SERVER_URL})")
    
    # Message parser
    message_parser = subparsers.add_parser("message", help="Send a single message to the assistant")
    message_parser.add_argument("text", help="Message to send")
    message_parser.add_argument("--conv", help="Conversation ID (optional)")
    message_parser.add_argument("--user", help="User ID (optional)")
    message_parser.add_argument("--server", default=DEFAULT_SERVER_URL, help=f"Server URL (default: {DEFAULT_SERVER_URL})")
    
    # Conversation history parser
    history_parser = subparsers.add_parser("history", help="Show conversation history")
    history_parser.add_argument("conv", help="Conversation ID")
    history_parser.add_argument("--server", default=DEFAULT_SERVER_URL, help=f"Server URL (default: {DEFAULT_SERVER_URL})")
    
    # List conversations parser
    list_parser = subparsers.add_parser("list", help="List all conversations")
    list_parser.add_argument("--server", default=DEFAULT_SERVER_URL, help=f"Server URL (default: {DEFAULT_SERVER_URL})")
    
    # Delete conversation parser
    delete_parser = subparsers.add_parser("delete", help="Delete a conversation")
    delete_parser.add_argument("conv", help="Conversation ID")
    delete_parser.add_argument("--server", default=DEFAULT_SERVER_URL, help=f"Server URL (default: {DEFAULT_SERVER_URL})")
    
    # Clear conversation parser
    clear_parser = subparsers.add_parser("clear", help="Clear a conversation's message history")
    clear_parser.add_argument("conv", help="Conversation ID")
    clear_parser.add_argument("--server", default=DEFAULT_SERVER_URL, help=f"Server URL (default: {DEFAULT_SERVER_URL})")
    
    # Health check parser
    health_parser = subparsers.add_parser("health", help="Check server health")
    health_parser.add_argument("--server", default=DEFAULT_SERVER_URL, help=f"Server URL (default: {DEFAULT_SERVER_URL})")
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no command is provided, show help
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Create client
    client = MicrosoftGraphClient(args.server)
    
    # Execute command
    if args.command == "chat":
        chat_mode(client)
    
    elif args.command == "message":
        if hasattr(args, "conv") and args.conv:
            client.conversation_id = args.conv
        
        response = client.send_chat_message(args.text, args.user if hasattr(args, "user") else None)
        
        if "error" in response:
            console.print(f"[red]Error: {response['error']}[/red]")
        else:
            console.print(f"[bold blue]You:[/bold blue] {args.text}")
            console.print(f"[bold green]Assistant:[/bold green] {response['response']}")
            console.print(f"[grey]Conversation ID: {response['conversation_id']}[/grey]")
    
    elif args.command == "history":
        conversation = client.get_conversation(args.conv)
        print_conversation(conversation)
    
    elif args.command == "list":
        conversations = client.list_conversations()
        print_conversations_list(conversations)
    
    elif args.command == "delete":
        result = client.delete_conversation(args.conv)
        if "error" in result:
            console.print(f"[red]Error: {result['error']}[/red]")
        else:
            console.print(f"[green]{result.get('message', 'Conversation deleted')}[/green]")
    
    elif args.command == "clear":
        result = client.clear_conversation(args.conv)
        if "error" in result:
            console.print(f"[red]Error: {result['error']}[/red]")
        else:
            console.print(f"[green]{result.get('message', 'Conversation cleared')}[/green]")
    
    elif args.command == "health":
        # Check both health and readiness
        health = client.health_check()
        readiness = client.readiness_check()
        
        console.print("[bold]Health Check:[/bold]")
        console.print(health)
        
        console.print("\n[bold]Readiness Check:[/bold]")
        console.print(readiness)


if __name__ == "__main__":
    main()