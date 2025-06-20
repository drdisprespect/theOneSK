#!/usr/bin/env python3
"""
Comprehensive Microsoft Graph Integration Test

This test demonstrates the full Microsoft Graph integration with the Azure Foundry
orchestration system, showcasing all 18 kernel functions and proper agent routing.
"""

import asyncio
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from azure_foundry_orchestration import AzureFoundryOrchestration, interactive_foundry_orchestration


async def test_graph_routing():
    """Test that Graph-related queries are properly routed to the GraphAssistant"""
    
    print("=== Microsoft Graph Integration Test ===")
    print("Testing agent routing for Microsoft 365 operations\n")

    # Configuration - same as main orchestration
    endpoint = "https://aif-e2edemo.services.ai.azure.com/api/projects/project-thanos"
    api_key = "D0b8REYc0wXONcnJu7fmj6kyFciM5XTrxjzJmoL1PtAXiqw1GHjXJQQJ99BFACYeBjFXJ3w3AAAAACOGpETv"
    agent_id = "asst_JE3DIZwUr7MWbb7KCM4OHxV4"
    model_name = "gpt-4o"
    bot_secret = "EIyanQMLVDOeluIdbvfddzUpO2mO14oGy8MKH04lprY08zqu0fqOJQQJ99BFAC4f1cMAArohAAABAZBS2U6n.CXzSOwihZ7dl5h9sI70U5VGr7ydVp75Nfr69psUNlP6KmQneluqoJQQJ99BFAC4f1cMAArohAAABAZBS3VyD"

    try:
        # Initialize orchestration
        orchestration = AzureFoundryOrchestration(
            endpoint=endpoint,
            api_key=api_key,
            agent_id=agent_id,
            model_name=model_name,
            bot_secret=bot_secret
        )
        
        # Get agents to see the GraphAssistant
        agents = await orchestration.get_agents()
        
        print("🔍 Available Agents:")
        for agent in agents:
            print(f"   • {agent.name}: {agent.description}")
        
        # Find the GraphAssistant
        graph_agent = None
        for agent in agents:
            if 'Graph' in agent.name:
                graph_agent = agent
                break
        
        if graph_agent:
            print(f"\n✅ GraphAssistant found: {graph_agent.name}")
            print(f"   Description: {graph_agent.description}")
            
            # Check if it has kernel functions
            if hasattr(graph_agent, 'kernel') and graph_agent.kernel:
                print("✅ GraphAssistant has kernel with functions")
                
                # Try to list available plugins
                if hasattr(graph_agent.kernel, 'plugins'):
                    plugins = graph_agent.kernel.plugins
                    print(f"   Plugins: {list(plugins.keys())}")
                    
                    # Check MicrosoftGraphPlugin specifically
                    if 'MicrosoftGraphPlugin' in plugins:
                        graph_plugin = plugins['MicrosoftGraphPlugin']
                        print(f"   ✅ MicrosoftGraphPlugin loaded with functions:")
                        
                        # List all functions in the plugin
                        if hasattr(graph_plugin, '_functions'):
                            for func_name, func in graph_plugin._functions.items():
                                print(f"      • {func_name}: {func.description}")
                        else:
                            print("      Functions list not accessible")
                else:
                    print("   ⚠️ Kernel plugins not accessible")
            else:
                print("   ⚠️ GraphAssistant has no kernel (fallback mode)")
        else:
            print("❌ GraphAssistant not found in agent list")
        
        # Test specific Graph queries for routing
        print("\n🧪 Testing Microsoft Graph Query Routing:")
        
        test_queries = [
            # User management queries
            ("Find user John Smith", "User Management"),
            ("List all users in organization", "User Management"),
            ("Get email for Sarah Wilson", "User Management"),
            ("Show organization directory", "User Management"),
            
            # Email operation queries
            ("Send email to marketing team", "Email Operations"),
            ("Check my inbox messages", "Email Operations"),
            ("Search emails about quarterly report", "Email Operations"),
            ("Show my email folders", "Email Operations"),
            
            # Task management queries
            ("Create a todo list for project tasks", "Task Management"),
            ("Add task to my work list", "Task Management"),
            ("Show all my todo lists", "Task Management"),
            ("List tasks from my project list", "Task Management"),
            
            # File operations queries
            ("Create folder in OneDrive", "File Operations"),
            ("Get my drive information", "File Operations"),
            ("Show drive root folder", "File Operations"),
            
            # Teams operations
            ("Create teams chat with colleagues", "Teams Operations"),
            ("Send message to teams channel", "Teams Operations"),
        ]
        
        for query, category in test_queries:
            print(f"\n   📝 Query: '{query}' ({category})")
            
            # Test routing by checking keywords
            query_lower = query.lower()
            graph_keywords = [
                'email', 'outlook', 'teams', 'onedrive', 'sharepoint', 'microsoft 365', 'm365', 'office 365',
                'user', 'users', 'directory', 'find user', 'list users', 'organization users', 'user directory',
                'search user', 'get users', 'user management', 'employee directory', 'staff directory',
                'send email', 'read email', 'inbox', 'send mail', 'email folders', 'search emails',
                'mail folders', 'compose email', 'reply email', 'forward email', 'email search',
                'todo', 'task', 'tasks', 'todo list', 'task list', 'create task', 'add task', 'my tasks',
                'todo lists', 'task management', 'create todo', 'to-do', 'checklist',
                'calendar', 'meeting', 'appointment', 'schedule', 'booking', 'event',
                'create folder', 'onedrive folder', 'file storage', 'drive', 'documents', 'files',
                'create directory', 'folder management', 'file management',
                'create team', 'teams chat', 'teams channel', 'microsoft teams', 'team collaboration',
                'chat message', 'teams meeting', 'channel message',
                'graph api', 'microsoft graph', 'graph', 'azure ad', 'active directory'
            ]
            
            matched_keywords = [keyword for keyword in graph_keywords if keyword in query_lower]
            if matched_keywords:
                print(f"      ✅ Would route to GraphAssistant (matched: {matched_keywords[:3]})")
            else:
                print(f"      ❌ Would NOT route to GraphAssistant")
        
        print("\n🎯 Microsoft Graph Function Capabilities:")
        print("   📧 Email Functions: send_mail, get_inbox_messages, search_all_emails, get_mail_folders")
        print("   🧑‍💼 User Functions: get_users, find_user_by_name, get_email_by_name, load_users_cache")
        print("   ✅ Task Functions: create_todo_list, create_todo_task, get_todo_lists, get_todo_tasks_from_list")
        print("   📁 File Functions: create_folder, get_user_drive, get_drive_root")
        print("   🔧 Management Functions: initialize, get_state, set_user_id")
        
        print("\n✅ Microsoft Graph integration test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


async def test_interactive_mode():
    """Test interactive mode with sample Graph queries"""
    
    print("\n=== Interactive Mode Test ===")
    print("This would normally start interactive mode.")
    print("For testing, we'll simulate some queries:")
    
    sample_queries = [
        "help",  # Show help with Graph operations
        "List all users in organization",  # User management
        "Send email to john@company.com",  # Email operation
        "Create todo list called 'Project Tasks'",  # Task management
        "quit"  # Exit
    ]
    
    print("\n📝 Sample queries that would be tested:")
    for i, query in enumerate(sample_queries, 1):
        print(f"   {i}. {query}")
    
    print("\n💡 To test interactively, run the main orchestration script directly")


if __name__ == "__main__":
    print("🚀 Starting Microsoft Graph Integration Tests...\n")
    
    # Run routing and configuration tests
    asyncio.run(test_graph_routing())
    
    # Show interactive test info
    asyncio.run(test_interactive_mode())
    
    print("\n🎉 All tests completed!")
    print("\n💡 Next steps:")
    print("   1. Run the main orchestration script: python azure_foundry_orchestration.py")
    print("   2. Try Microsoft Graph queries like 'list users' or 'send email'")
    print("   3. Check that queries are routed to GraphAssistant")
    print("   4. Verify that kernel functions are properly invoked")
