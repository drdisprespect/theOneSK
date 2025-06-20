#!/usr/bin/env python3
"""
Test script to verify Copilot Studio integration with the orchestration system
"""

import asyncio
import sys
import os

# Add current directory to path for imports
sys.path.append('.')

async def test_copilot_studio_integration():
    """Test the Copilot Studio agent integration"""
    print("=== Testing Copilot Studio Integration ===\n")
    
    try:
        # Test imports
        print("🔄 Testing imports...")
        from directline_client import DirectLineClient
        from copilot_studio_agent import CopilotAgent
        from copilot_studio_agent_thread import CopilotAgentThread
        from copilot_studio_channel import CopilotStudioAgentChannel
        from copilot_studio_message_content import CopilotMessageContent
        print("✅ All Copilot Studio imports successful")
        
        # Test DirectLine client creation
        print("\n🔄 Testing DirectLine client creation...")
        bot_secret = "C4szkmqcNRF5ilvzhfnOEoyVrPZZuQec1WRm6ZFImRAbaZaUGd4mJQQJ99BFAC77bzfAArohAAABAZBS2wzn.752uIxMCBlmwHTuFssWfCuEcFyGBSb4u9cwN2yuhNCaAnXNyPM51JQQJ99BFAC77bzfAArohAAABAZBS3sPB"
        
        directline_client = DirectLineClient(
            copilot_agent_secret=bot_secret,
            directline_endpoint="https://europe.directline.botframework.com/v3/directline"
        )
        print("✅ DirectLine client created successfully")
        
        # Test CopilotAgent creation
        print("\n🔄 Testing CopilotAgent creation...")
        copilot_agent = CopilotAgent(
            id="copilot_studio",
            name="CareerAdvisor",
            description="Microsoft Career advice specialist from Copilot Studio",
            directline_client=directline_client,
        )
        print("✅ CopilotAgent created successfully")
        print(f"   Agent ID: {copilot_agent.id}")
        print(f"   Agent Name: {copilot_agent.name}")
        print(f"   Description: {copilot_agent.description}")
        
        # Test CopilotAgentThread creation
        print("\n🔄 Testing CopilotAgentThread creation...")
        copilot_thread = CopilotAgentThread(directline_client=directline_client)
        print("✅ CopilotAgentThread created successfully")
        
        # Test a simple career advice question
        print("\n🔄 Testing career advice interaction...")
        test_message = "What skills should I develop for a career at Microsoft?"
        
        print(f"📤 Sending message: '{test_message}'")
        
        # Use the agent to get a response
        async for response in copilot_agent.invoke(
            messages=test_message, 
            thread=copilot_thread
        ):
            print(f"📥 Response received:")
            print(f"   Content: {response.content}")
            if hasattr(response, 'name') and response.name:
                print(f"   From: {response.name}")
            break  # Just get the first response for testing
        
        print("\n✅ Copilot Studio integration test completed successfully!")
        print("🎉 The Career Advisor agent is ready for use in the orchestration system!")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

async def test_orchestration_integration():
    """Test the full orchestration system with Copilot Studio"""
    print("\n=== Testing Full Orchestration Integration ===\n")
    
    try:
        from azure_foundry_orchestration import AzureFoundryOrchestration
        
        # Configuration
        endpoint = "https://aif-e2edemo.services.ai.azure.com/api/projects/project-thanos"
        api_key = "D0b8REYc0wXONcnJu7fmj6kyFciM5XTrxjzJmoL1PtAXiqw1GHjXJQQJ99BFAC77bzfAArohAAABAZBS2wzn"
        agent_id = "asst_JE3DIZwUr7MWbb7KCM4OHxV4"
        model_name = "gpt-4o"
        bot_secret = "C4szkmqcNRF5ilvzhfnOEoyVrPZZuQec1WRm6ZFImRAbaZaUGd4mJQQJ99BFAC77bzfAArohAAABAZBS2wzn.752uIxMCBlmwHTuFssWfCuEcFyGBSb4u9cwN2yuhNCaAnXNyPM51JQQJ99BFAC77bzfAArohAAABAZBS3sPB"
        
        print("🔄 Initializing orchestration system...")
        orchestration_system = AzureFoundryOrchestration(
            endpoint=endpoint, 
            api_key=api_key, 
            agent_id=agent_id, 
            model_name=model_name,
            bot_secret=bot_secret
        )
        
        print("\n🔄 Creating agent ensemble...")
        agents = await orchestration_system.get_agents()
        
        print(f"\n✅ Orchestration system initialized with {len(agents)} agents:")
        for i, agent in enumerate(agents, 1):
            print(f"   {i}. {agent.name}: {agent.description}")
        
        # Verify CareerAdvisor is in the list
        career_agent_found = any('Career' in agent.name for agent in agents)
        if career_agent_found:
            print("\n✅ CareerAdvisor successfully integrated into orchestration!")
        else:
            print("\n❌ CareerAdvisor not found in agent list")
            return False
        
        print("\n🎉 Full orchestration integration test completed successfully!")
        
    except Exception as e:
        print(f"❌ Orchestration integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

async def main():
    """Run all integration tests"""
    print("🚀 Starting Copilot Studio Integration Tests\n")
    
    # Test 1: Basic Copilot Studio integration
    test1_success = await test_copilot_studio_integration()
    
    if test1_success:
        # Test 2: Full orchestration integration
        test2_success = await test_orchestration_integration()
        
        if test1_success and test2_success:
            print("\n" + "="*60)
            print("🎉 ALL TESTS PASSED! 🎉")
            print("Your Copilot Studio Career Advisor is fully integrated!")
            print("="*60)
        else:
            print("\n❌ Some tests failed. Please check the errors above.")
    else:
        print("\n❌ Basic integration test failed. Skipping orchestration test.")

if __name__ == "__main__":
    asyncio.run(main())
