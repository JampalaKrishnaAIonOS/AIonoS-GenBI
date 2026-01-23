from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ChatRequest(BaseModel):
    question: str
    session_id: str
    conversation_history: Optional[List[Dict[str, str]]] = []

class SourceInfo(BaseModel):
    file_name: str
    sheet_name: str
    rows_used: str
    columns_used: List[str]
    
class TableData(BaseModel):
    headers: List[str]
    rows: List[List[Any]]
    
class ChartData(BaseModel):
    chart_type: str  # bar, line, pie
    data: Dict[str, Any]
    title: str
    
class ChatResponse(BaseModel):
    answer: str
    source: Optional[SourceInfo] = None
    table_data: Optional[TableData] = None
    chart_data: Optional[ChartData] = None
    code_executed: Optional[str] = None
    error: Optional[str] = None