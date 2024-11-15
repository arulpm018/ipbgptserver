from pydantic import BaseModel
from typing import List, Optional

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


class Query(BaseModel):
    question: str

class CombinedQuery(BaseModel):
    query: str
    chat_history: List[ChatMessage]
    context: Optional[str] = None