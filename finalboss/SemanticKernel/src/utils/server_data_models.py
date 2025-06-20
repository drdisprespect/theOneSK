from typing import Dict, List, Optional, Any
from pydantic import BaseModel


# Pydantic models
class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str
    conversation_id: str
    timestamp: str

class Message(BaseModel):
    """Model for a chat message"""
    role: str
    content: str
    timestamp: str

class Conversation(BaseModel):
    """Model for a conversation"""
    id: str
    messages: List[Message]
    created_at: str
    updated_at: str