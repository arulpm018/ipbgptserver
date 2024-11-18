from pydantic import BaseModel
from typing import List

class ThesisTitle(BaseModel):
    title: str
    number: int

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatQuery(BaseModel):
    query: str
    context: str
    chat_history: List[ChatMessage]



