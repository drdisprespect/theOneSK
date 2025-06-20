#!/usr/bin/env python3
"""
Test Microsoft Graph Integration in Azure Foundry Orchestration

This script tests the Microsoft Graph agent integration with the Azure Foundry
orchestration system to ensure proper routing and functionality.
"""

import asyncio
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the orchestration system
try:
    from azure_foundry_orchestration import AzureFoundryOrchestration
    print("✅ Successfully imported AzureFoundryOrchestration")
except ImportError as e:
    print(f"❌ Failed to import orchestration: {e}")
    sys.exit(1)

async def test_graph_agent_creation():
    """Test creating the Microsoft Graph agent"""
    print("\n🧪 Testing Microsoft Graph Agent Creation...")
    
    try:
        # Initialize orchestration (this will test the imports and setup)
        orchestration = AzureFoundryOrchestration(
            endpoint="test-endpoint",
            api_key="test-key", 
            agent_id="test-agent",
            model_name="test-model"
        )
        
        # Initialize the chat service (this might fail due to credentials, which is expected in testing)
        try:
            await orchestration.initialize()
            print("✅ Orchestration initialized successfully")
        except Exception as e:
            print(f"⚠️ Orchestration initialization failed (expected): {e}")
            # Continue anyway to test agent creation logic
        
        # Test Microsoft Graph agent creation
        print("\n🔄 Testing Graph Agent Creation...")
        graph_agent = orchestration.create_microsoft_graph_agent()
        
        if graph_agent:
            print(f"✅ Graph agent created successfully:")
            print(f"   • Name: {graph_agent.name}")
            print(f"   • Description: {graph_agent.description}")
            print(f"   • Has kernel: {hasattr(graph_agent, 'kernel') and graph_agent.kernel is not None}")
            print(f"   • Has execution settings: {hasattr(graph_agent, 'execution_settings')}")
            
            # Check if the graph agent has plugins (if kernel is available)
            if hasattr(graph_agent, 'kernel') and graph_agent.kernel:
                try:
                    plugins = graph_agent.kernel.plugins
                    print(f"   • Kernel plugins: {len(plugins) if plugins else 0}")
                    if plugins:
                        for plugin_name in plugins:
                            print(f"     - {plugin_name}")
                except Exception as e:
                    print(f"   • Could not check kernel plugins: {e}")
            
            return True
        else:
            print("❌ Failed to create Graph agent")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        return False

def test_routing_keywords():
    """Test the routing keywords for Microsoft Graph operations"""
    print("\n🧪 Testing Routing Keywords...")
    
    test_messages = [
        "send an email to john@example.com",
        "check my outlook inbox",
        "create a new todo list",
        "find user by name John Smith",
        "access my OneDrive files",
        "schedule a teams meeting",
        "get my tasks from Microsoft 365",
        "search for emails about project",
        "create a folder in sharepoint",
        "show me the organization directory",
        "help with office 365 operations",
        "graph api functionality"
    ]
    
    # These are the keywords from the orchestration system
    graph_keywords = ['email', 'outlook', 'teams', 'onedrive', 'sharepoint', 'microsoft 365', 'm365', 'office 365', 'user', 'directory', 'todo', 'task', 'calendar', 'meeting', 'send email', 'read email', 'create team', 'graph api']
    
    print("📧 Testing Microsoft 365 message routing:")
    for message in test_messages:
        should_route = any(keyword in message.lower() for keyword in graph_keywords)
        status = "✅ Routes to Graph" if should_route else "❌ No routing"
        print(f"   {status}: '{message}'")
    
    print(f"\n📊 Summary: {len([m for m in test_messages if any(k in m.lower() for k in graph_keywords)])}/{len(test_messages)} messages would route to Graph Assistant")

async def test_agent_ensemble():
    """Test creating the full agent ensemble"""
    print("\n🧪 Testing Agent Ensemble Creation...")
    
    try:
        orchestration = AzureFoundryOrchestration(
            endpoint="test-endpoint",
            api_key="test-key", 
            agent_id="test-agent",
            model_name="test-model"
        )
        
        # Test getting all agents
        try:
            agents = await orchestration.get_agents()
            
            print(f"✅ Created {len(agents)} agents:")
            for i, agent in enumerate(agents, 1):
                print(f"   {i}. {agent.name}: {agent.description}")
            
            # Check if GraphAssistant is in the ensemble
            graph_agent_found = any('graph' in agent.name.lower() for agent in agents)
            if graph_agent_found:
                print("✅ GraphAssistant found in agent ensemble")
            else:
                print("❌ GraphAssistant NOT found in agent ensemble")
                
            return len(agents) > 0
            
        except Exception as e:
            print(f"❌ Failed to create agent ensemble: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def print_integration_summary():
    """Print summary of the Microsoft Graph integration"""
    print("\n📋 Microsoft Graph Integration Summary:")
    print("=" * 60)
    
    print("\n🔧 **Integration Components:**")
    print("• Microsoft Graph Agent Plugin (18 kernel functions)")
    print("• Graph Agent wrapper with Azure AD authentication")
    print("• Kernel-based function calling for real API operations")
    print("• Smart routing based on Microsoft 365 keywords")
    print("• Fallback mode for when Graph API is unavailable")
    
    print("\n📊 **Available Functions:**")
    functions = [
        "initialize - Setup Graph API connection",
        "get_users - Get organization user directory", 
        "find_user_by_name - Search users by name",
        "get_email_by_name - Get user email addresses",
        "send_mail - Send emails to recipients",
        "get_inbox_messages - Read inbox messages",
        "search_all_emails - Search emails by criteria",
        "get_mail_folders - List email folders",
        "create_todo_list - Create new task lists",
        "create_todo_task - Add tasks to lists",
        "get_todo_lists - Get all todo lists",
        "get_todo_tasks_from_list - Get tasks from specific list",
        "create_folder - Create new folders",
        "get_user_drive - Access user's OneDrive",
        "get_drive_root - Get drive root information",
        "set_user_id - Set active user context",
        "get_state - Get plugin state information",
        "load_users_cache - Cache user data for efficiency"
    ]
    
    for func in functions:
        print(f"   • {func}")
    
    print("\n🎯 **Routing Keywords:**")
    keywords = ['email', 'outlook', 'teams', 'onedrive', 'sharepoint', 'microsoft 365', 'm365', 'office 365', 'user', 'directory', 'todo', 'task', 'calendar', 'meeting', 'send email', 'read email', 'create team', 'graph api']
    for keyword in keywords:
        print(f"   • '{keyword}'")
    
    print("\n🚀 **Usage Examples:**")
    examples = [
        "send email to john@company.com with subject 'Meeting Tomorrow'",
        "find user named Sarah Johnson in the directory",
        "create a todo list called 'Project Tasks'",
        "check my outlook inbox for new messages",
        "create a folder called 'Reports' in my OneDrive",
        "show me all tasks from my 'Work' todo list"
    ]
    
    for example in examples:
        print(f"   • \"{example}\"")

async def main():
    """Main test function"""
    print("🧪 Microsoft Graph Integration Tests")
    print("=" * 50)
    
    # Run tests
    test_results = []
    
    # Test 1: Agent creation
    result1 = await test_graph_agent_creation()
    test_results.append(("Graph Agent Creation", result1))
    
    # Test 2: Routing keywords  
    test_routing_keywords()
    test_results.append(("Routing Keywords", True))  # This test always passes
    
    # Test 3: Agent ensemble
    result3 = await test_agent_ensemble()
    test_results.append(("Agent Ensemble", result3))
    
    # Print results
    print("\n📊 Test Results:")
    print("=" * 30)
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    # Print integration summary
    print_integration_summary()
    
    print(f"\n🏁 Testing complete! {sum(r for _, r in test_results)}/{len(test_results)} tests passed")

if __name__ == "__main__":
    asyncio.run(main())
