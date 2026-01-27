import os
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
import pandas as pd
from contextlib import asynccontextmanager
import json
import numpy as np
from pathlib import Path

def sanitize_for_json(obj):
    """Recursively convert numpy/pandas types to standard Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(i) for i in obj]
    elif isinstance(obj, (np.int64, np.int32, np.int8, np.integer)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.floating)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.datetime64, pd.Timestamp)):
        return str(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    elif pd.isna(obj):
        return None
    return obj

from models.schemas import ChatRequest, ChatResponse, SourceInfo, TableData, ChartData
from services.excel_indexer import ExcelSchemaIndexer
from services.file_watcher import start_file_watcher
from services.groq_agent import GroqPandasAgent
from services.chart_generator import ChartGenerator
from services.session_manager import SessionManager

# Load environment
load_dotenv()

# Global instances
indexer = None
agent = None
session_manager = SessionManager()
file_observer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global indexer, agent, file_observer
    
    print("🚀 Starting Excel Analytics Chatbot...")
    
    # Initialize indexer
    excel_folder = os.getenv('EXCEL_FOLDER_PATH')
    index_path = os.getenv('FAISS_INDEX_PATH')
    groq_api_key = os.getenv('GROQ_API_KEY')
    
    indexer = ExcelSchemaIndexer(excel_folder, index_path)
    
    # Load existing index or create new
    index_loaded = indexer.load_index()
    
    # Verify index matches files on disk
    needs_reindex = not index_loaded
    if index_loaded:
        current_files = {f.name for f in Path(excel_folder).glob("*.xlsx") if not f.name.startswith('~$')}
        indexed_files = {m['file_name'] for m in indexer.metadata}
        
        if current_files != indexed_files:
            print(f"⚠️ Index out of sync. Folder has {len(current_files)} files, Index has {len(indexed_files)}. Re-indexing...")
            needs_reindex = True
            
    if needs_reindex:
        print("📊 Creating/Updating index...")
        indexer.index_all_sheets()
    
    # Initialize Groq agent
    agent = GroqPandasAgent(groq_api_key)
    
    # Start file watcher
    file_observer = start_file_watcher(excel_folder, indexer)
    
    print("✅ System ready!")
    
    yield
    
    # Shutdown
    if file_observer:
        file_observer.stop()
        file_observer.join()
    print("👋 Shutting down...")

# Create FastAPI app
app = FastAPI(title="Excel Analytics Chatbot", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "Excel Analytics Chatbot is running!"}

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "indexed_sheets": len(indexer.metadata) if indexer else 0
    }

@app.get("/sheets")
async def list_sheets():
    """List all indexed sheets"""
    if not indexer or not indexer.metadata:
        return {"sheets": []}
    
    sheets = [
        {
            "file": m['file_name'],
            "sheet": m['sheet_name'],
            "columns": [c['name'] for c in m['columns']],
            "rows": m['row_count']
        }
        for m in indexer.metadata
    ]
    return {"sheets": sheets}

async def stream_response_generator(question: str, session_id: str, conversation_history: list):
    """Generate streaming response word by word"""
    
    try:
        # Get session
        session = session_manager.get_session(session_id)
        # If the user's question is not data-oriented, reply conversationally instead
        if not agent.is_data_question(question):
            yield json.dumps({"type": "status", "content": "💬 Handling conversational query..."}) + "\n"
            await asyncio.sleep(0.05)
            reply = agent.generate_conversational_reply(question, conversation_history)
            yield json.dumps({"type": "answer_start"}) + "\n"
            for word in (reply or '').split():
                yield json.dumps({"type": "word", "content": word + " "}) + "\n"
                await asyncio.sleep(0.02)
            yield json.dumps({"type": "answer_end"}) + "\n"
            yield json.dumps({"type": "complete"}) + "\n"
            return
        
        # Step 1: Find relevant sheets
        yield json.dumps({"type": "status", "content": "🔍 Searching relevant data..."}) + "\n"
        await asyncio.sleep(0.1)
        
        if any(k in question.lower() for k in ['all', 'across', 'combined', 'overall']):
            relevant_sheets = indexer.metadata
        else:
            relevant_sheets = indexer.search_relevant_sheets(question, top_k=3)
        
        if not relevant_sheets:
            yield json.dumps({"type": "error", "content": "No relevant data found."}) + "\n"
            return
        
        # Build dfs = all relevant sheets
        dfs = {}
        for sheet in relevant_sheets:
            key = f"{sheet['file_name']}::{sheet['sheet_name']}"
            dfs[key] = pd.read_excel(sheet['file_path'], sheet_name=sheet['sheet_name'])

        # Use top match
        best_match = relevant_sheets[0]
        file_path = best_match['file_path']
        sheet_name = best_match['sheet_name']
        file_name = best_match['file_name']
        
        yield json.dumps({
            "type": "status", 
            "content": f"📊 Analyzing {file_name} - {sheet_name}..."
        }) + "\n"
        await asyncio.sleep(0.1)
        
        # Load DataFrame
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # Step 2: Generate pandas code
        yield json.dumps({"type": "status", "content": "🧠 Generating analysis code..."}) + "\n"
        await asyncio.sleep(0.1)
        
        code = agent.generate_pandas_code(
            question, df, file_name, sheet_name, conversation_history
        )
        
        # Step 3: Execute code
        yield json.dumps({"type": "status", "content": "⚡ Executing analysis..."}) + "\n"
        await asyncio.sleep(0.1)
        
        execution_result = agent.execute_code(code, df, dfs, question=question)
        
        if execution_result['type'] == 'error':
            yield json.dumps({
                "type": "error", 
                "content": f"Analysis error: {execution_result['error']}"
            }) + "\n"
            return
        
        # Step 4: Generate natural language response
        answer = agent.generate_natural_response(
            question, execution_result, file_name, sheet_name
        )
        
        # Stream answer while preserving newlines and spaces
        yield json.dumps({"type": "answer_start"}) + "\n"
        
        # Use regex to split by whitespace but KEEP the whitespace tokens
        import re
        tokens = re.split(r'(\s+)', answer)
        for token in tokens:
            if not token: continue
            yield json.dumps({"type": "word", "content": token}) + "\n"
            # Slightly faster stream since we have more tokens now
            await asyncio.sleep(0.01)
        
        yield json.dumps({"type": "answer_end"}) + "\n"
        
        # Step 5: Send source information for ACTUALLY used sheets
        # 1. Always include the best match (since 'df' maps to it) if accessed_keys is empty or code likely used it
        # 2. Include any other keys touched in 'dfs'
        
        accessed_keys = set(execution_result.get('accessed_keys', []))
        best_match_key = f"{best_match['file_name']}::{best_match['sheet_name']}"
        
        # If no specific dfs keys were touched, assume only the primary df was used.
        # If dfs keys WERE touched, we include them. 
        # We always implicitly trust the primary df was context unless proven otherwise, 
        # but to be cleaner: if accessed_keys exists, it means the user logic reached into dfs.
        
        final_source_keys = set()
        final_source_keys.add(best_match_key) # Always safe to show the primary context
        final_source_keys.update(accessed_keys)
        
        for sheet in relevant_sheets:
            key = f"{sheet['file_name']}::{sheet['sheet_name']}"
            
            # ONLY send source if it was actually in the final set of used keys
            if key in final_source_keys and key in dfs:
                sub_df = dfs[key]
                source_info = {
                    "file_name": sheet['file_name'],
                    "sheet_name": sheet['sheet_name'],
                    "rows_used": f"1-{len(sub_df)}",
                    "columns_used": list(sub_df.columns)
                }
                yield json.dumps({"type": "source", "content": source_info}) + "\n"
        
        # Step 6: Send table data if applicable
        if execution_result['type'] in ['dataframe', 'series']:
            data = execution_result['data']
            
            if isinstance(data, pd.Series):
                data = data.reset_index()
                data.columns = ['Category', 'Value']
            elif isinstance(data, pd.DataFrame):
                # If index is not standard numbers (like in .describe()), include it
                if not isinstance(data.index, pd.RangeIndex) or (not data.index.is_numeric() and len(data) > 0):
                    data = data.reset_index()
            
            # Limit to 100 rows for display
            display_df = data.head(100)
            
            table_data = {
                "headers": [str(c) for c in display_df.columns],
                "rows": sanitize_for_json(display_df.values.tolist())
            }
            
            yield json.dumps({"type": "table", "content": table_data}) + "\n"
            
            # Step 7: Generate chart if appropriate
            # FIX 3: Reset plot intent when user says "just / only"
            skip_chart = any(k in question.lower() for k in ['just', 'only', 'table', 'list'])
            
            if not skip_chart and any(keyword in question.lower() for keyword in ['plot', 'chart', 'graph', 'show', 'visualize']):
                # Detect chart type explicitly and pass it to generator for deterministic behavior
                chart_type = ChartGenerator.detect_chart_type(question, data)
                
                # FIX 2: Histogram must use numeric columns only
                chart_data = data
                if chart_type == 'histogram':
                    numeric_cols = data.select_dtypes(include='number').columns
                    if len(numeric_cols) == 0:
                        yield json.dumps({"type": "error", "content": "Histogram requires numeric data."}) + "\n"
                        chart_data = None
                    else:
                        chart_data = data[numeric_cols]
                
                if chart_data is not None:
                    chart = ChartGenerator.generate_chart(
                        chart_data,
                        chart_type=chart_type,
                        title=question
                    )
                    if chart:
                        yield json.dumps({"type": "chart", "content": chart}) + "\n"
        
        # Step 8: Send executed code
        yield json.dumps({"type": "code", "content": code}) + "\n"
        
        # Update session
        session_manager.add_to_history(session_id, "user", question)
        session_manager.add_to_history(session_id, "assistant", answer)
        session_manager.update_context(
            session_id,
            last_used_sheet=(file_name, sheet_name),
            last_code=code
        )
        
        yield json.dumps({"type": "complete"}) + "\n"
        
    except Exception as e:
        yield json.dumps({"type": "error", "content": f"System error: {str(e)}"}) + "\n"

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream chat response word by word"""
    return StreamingResponse(
        stream_response_generator(
            request.question, 
            request.session_id, 
            request.conversation_history
        ),
        media_type="text/event-stream"
    )

@app.post("/reindex")
async def reindex():
    """Manually trigger reindexing"""
    try:
        indexer.index_all_sheets()
        return {"status": "success", "sheets_indexed": len(indexer.metadata)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)