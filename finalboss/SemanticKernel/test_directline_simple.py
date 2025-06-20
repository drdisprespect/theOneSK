#!/usr/bin/env python3

import asyncio
import sys
from directline_client import DirectLineClient

async def test_directline_simple():
    """Simple test of DirectLine client"""
    
    bot_secret = "EIyanQMLVDOeluIdbvfddzUpO2mO14oGy8MKH04lprY08zqu0fqOJQQJ99BFAC4f1cMAArohAAABAZBS2U6n.CXzSOwihZ7dl5h9sI70U5VGr7ydVp75Nfr69psUNlP6KmQneluqoJQQJ99BFAC4f1cMAArohAAABAZBS3VyD"
    
    async with DirectLineClient(
        copilot_agent_secret=bot_secret,
        directline_endpoint="https://europe.directline.botframework.com/v3/directline"
    ) as client:
        
        try:
            print("🔄 Starting conversation...")
            conv_id = await client.start_conversation()
            print(f"✅ Conversation started: {conv_id}")
            
            # Send a specific career advice request
            payload = {
                "type": "message",
                "from": {"id": "user"},
                "text": "Give me 5 career tips for software engineers",
                "conversationId": conv_id
            }
            
            print("🔄 Sending message...")
            response = await client.post_activity(conv_id, payload)
            print(f"✅ Message sent successfully")
            
            # Wait a bit for processing
            await asyncio.sleep(2)
            
            # Get activities
            print("🔄 Getting activities...")
            activities_response = await client.get_activities(conv_id)
            activities = activities_response.get("activities", [])
            
            print(f"📋 Found {len(activities)} activities:")
            for i, activity in enumerate(activities):
                print(f"  Activity {i+1}:")
                print(f"    Type: {activity.get('type', 'unknown')}")
                print(f"    From: {activity.get('from', {}).get('name', 'unknown')}")
                if 'text' in activity:
                    print(f"    Text: {activity['text'][:100]}...")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_directline_simple())
