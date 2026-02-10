"""
ULTRA-LIGHTWEIGHT Backend Main - Simplified Architecture (No Pandas/DataFrame overhead)
"""

import os
import asyncio
import logging
import json
import ast
from pathlib import Path
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi import Request
from dotenv import load_dotenv
import re
from contextlib import asynccontextmanager
import langchain

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
langchain.verbose = True
# -----------------------------

# Services
from services.excel_to_sql_sync import ExcelToSQLSync
from services.sql_agent import SQLAgent
from services.file_watcher_sql import start_file_watcher_sql
from services.session_manager import session_manager
from models.schemas import ChatRequest

load_dotenv()

def parse_raw_sql_result(raw):
    """Parse raw observation from agent tool into a list of rows"""
    try:
        if not raw:
            return []
        if isinstance(raw, (list, tuple)):
            return raw
        # If it's a string representation of a list/tuple
        data = ast.literal_eval(str(raw))
        if isinstance(data, (list, tuple)):
            return data
    except Exception as e:
        logger.warning(f"Failed to parse raw SQL result: {e}")
    return []

# Global instances
sql_sync = None
sql_agent = None
file_observer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global sql_sync, sql_agent, file_observer
    
    print("🚀 Starting GenBI Simplified SQL-based Chatbot...")
    
    excel_folder = os.getenv('EXCEL_FOLDER_PATH')
    db_url = os.getenv('DATABASE_URL', 'sqlite:///genbi.db')
    model_name = os.getenv('MODEL_NAME', 'self-hosted')
    llm_api_key = os.getenv('LLM_API_KEY', 'dummy')
    llm_base_url = os.getenv('LLM_BASE_URL')
    
    # Initialize SQL sync
    sql_sync = ExcelToSQLSync(excel_folder, db_url)
    
    print("📊 Syncing Excel files to database...")
    try:
        results = sql_sync.sync_all_excel_files()
        print(f"✅ Synced {len(results['success'])} tables")
    except Exception as e:
        print(f"❌ Initial sync failed: {e}")
    
    # Initialize SQL Agent
    sql_agent = SQLAgent(db_url, llm_api_key, model_name, llm_base_url)
    
    # Start file watcher
    try:
        file_observer = start_file_watcher_sql(excel_folder, sql_sync)
    except Exception as e:
        print(f"⚠️ Failed to start file watcher: {e}")
    
    yield
    
    if file_observer:
        file_observer.stop()
        file_observer.join()
    
    print("🛑 GenBI Chatbot stopped")

app = FastAPI(title="GenBI SQL Chatbot", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"status": "GenBI SQL Chatbot is running!"}

@app.get("/health")
async def health():
    tables = []
    if sql_sync:
        try:
            tables = sql_sync.list_all_tables()
        except:
            pass
    return {
        "status": "healthy",
        "synced_tables": len(tables)
    }

async def stream_response_generator(question: str, session_id: str, conversation_history: List):
    """Stream SQL agent response with raw data parsing and persistence"""
    try:
        logger.info(f"Processing question: {question} (session: {session_id})")
        
        # Status update
        yield json.dumps({"type": "status", "content": "Thinking..."}) + "\n"
        await asyncio.sleep(0.1)
        
        question_lower = question.lower().strip()
        plot_triggers = ['plot', 'plot it', 'show chart', 'visualize', 'visualize it', 'chart it', 'plot the data']
        
        # ✅ BACKEND SHORTCUT FOR "PLOT IT"
        # If it's a plot request and we have a stored table, we can skip the LLM
        if question_lower in plot_triggers:
            last_table = session_manager.get_last_table(session_id)
            if last_table:
                logger.info("🎯 Plot request detected, found persistent table. Informing UI.")
                yield json.dumps({"type": "answer_start"}) + "\n"
                yield json.dumps({"type": "word", "content": "Generating visualization from previous result..."}) + "\n"
                yield json.dumps({"type": "answer_end"}) + "\n"
                yield json.dumps({"type": "table", "content": last_table}) + "\n"
                yield json.dumps({"type": "complete"}) + "\n"
                return

        # Limit Conversation History
        short_history = conversation_history[-4:] if conversation_history else []
        
        # Execute query via SQLAgent (returns raw rows)
        result = sql_agent.query(question, session_id=session_id, conversation_history=short_history)
        
        if not result['success']:
            yield json.dumps({"type": "error", "content": result.get('answer', 'Unknown error')}) + "\n"
            yield json.dumps({"type": "complete"}) + "\n"
            return
        
        answer = result['answer']
        sql_query = result.get('sql_query')
        raw_result = result.get('raw_result')
        extracted_cols = result.get('columns', [])
        
        # Stream answer
        yield json.dumps({"type": "answer_start"}) + "\n"
        tokens = re.split(r'(\s+)', str(answer))
        for token in tokens:
            if token:
                yield json.dumps({"type": "word", "content": token}) + "\n"
                await asyncio.sleep(0.005)
        yield json.dumps({"type": "answer_end"}) + "\n"

        if result.get('raw_result') or raw_result:
            logger.info(f"DEBUG: raw_result is present. Type: {type(raw_result or result.get('raw_result'))}")
        else:
            logger.warning("DEBUG: raw_result is MISSING from agent response")

        # Parse Raw Rows
        rows = parse_raw_sql_result(raw_result)
        logger.info(f"DEBUG: Parsed rows: {len(rows) if rows else 0}")
        
        if rows and isinstance(rows, (list, tuple)) and len(rows) > 0:
            try:
                # Build column names
                first_row = rows[0]
                num_data_cols = len(first_row) if isinstance(first_row, (list, tuple)) else 1
                
                # Use extracted columns if they match the data count, else use generic
                if len(extracted_cols) == num_data_cols:
                    columns = extracted_cols
                else:
                    columns = ["Column_" + str(i+1) for i in range(num_data_cols)]

                # Build Table Payload
                table_rows = []
                for row in rows:
                    if isinstance(row, (list, tuple)):
                        table_rows.append({columns[i]: row[i] for i in range(min(len(row), len(columns)))})
                    else:
                        table_rows.append({columns[0]: row})

                table_data = {
                    "columns": columns,
                    "rows": table_rows
                }

                logger.info(f"📤 Sending table: {len(columns)} columns × {len(table_rows)} rows")
                yield json.dumps({"type": "table", "content": table_data}) + "\n"
                
                # ✅ PERSIST TABLE FOR PLOTTING
                session_manager.set_last_table(session_id, table_data)

                # Update session with last query/data info if needed (minimal)
                if sql_query:
                    session_manager.set_last_sql(session_id, sql_query)
                
            except Exception as table_err:
                logger.error(f"❌ Table parsing failed: {table_err}")
                yield json.dumps({"type": "error", "content": f"Table parsing failed: {str(table_err)}"}) + "\n"
        else:
             logger.info("ℹ️ No rows found to send as table")

        # Update session history
        session_manager.add_to_history(session_id, "user", question)
        session_manager.add_to_history(session_id, "assistant", str(answer))
        session_manager.trim_history(session_id, max_messages=10)
        
        yield json.dumps({"type": "complete"}) + "\n"
        
    except Exception as e:
        logger.error(f"Stream error: {e}", exc_info=True)
        yield json.dumps({"type": "error", "content": str(e)}) + "\n"
        yield json.dumps({"type": "complete"}) + "\n"

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

async def plot_response_generator(table_obj, x_col, y_col, session_id: str):
    """Stream a simple Plotly chart generated from the provided table object without Pandas."""
    try:
        yield json.dumps({"type": "status", "content": "Preparing visualization..."}) + "\n"
        await asyncio.sleep(0.05)

        rows = table_obj.get('rows', [])
        if not rows:
            yield json.dumps({"type": "error", "content": "No data to plot."}) + "\n"
            yield json.dumps({"type": "complete"}) + "\n"
            return

        # Simple insight
        insight = f"### **Plot — {y_col} vs {x_col}**\n\nGenerated from table data.\n"
        
        yield json.dumps({"type": "answer_start"}) + "\n"
        for token in re.split(r'(\s+)', insight):
            if token:
                yield json.dumps({"type": "word", "content": token}) + "\n"
                await asyncio.sleep(0.005)
        yield json.dumps({"type": "answer_end"}) + "\n"

        # Build Plotly object manually
        x_data = [r.get(x_col) for r in rows]
        y_data = []
        for r in rows:
            val = r.get(y_col)
            try:
                y_data.append(float(val) if val is not None else 0)
            except:
                y_data.append(0)

        chart_payload = {
            "type": "chart",
            "content": {
                "type": "chart",
                "data": {
                    "data": [{
                        "x": x_data,
                        "y": y_data,
                        "type": "bar",
                        "marker": {"color": "#4f46e5"}
                    }],
                    "layout": {
                        "title": f"{y_col} by {x_col}",
                        "xaxis": {"title": x_col},
                        "yaxis": {"title": y_col},
                        "template": "plotly_white"
                    }
                }
            }
        }
        
        yield json.dumps(chart_payload) + "\n"
        yield json.dumps({"type": "complete"}) + "\n"
    except Exception as e:
        yield json.dumps({"type": "error", "content": f"Plot error: {str(e)}"}) + "\n"
        yield json.dumps({"type": "complete"}) + "\n"

@app.post("/plot/stream")
async def plot_stream(request: Request):
    """Accepts JSON: { table: { columns: [...], rows: [...] }, x: 'col', y: 'col', session_id: '...'}"""
    body = await request.json()
    return StreamingResponse(
        plot_response_generator(
            body.get('table'), 
            body.get('x'), 
            body.get('y'), 
            body.get('session_id')
        ), 
        media_type="text/event-stream"
    )

@app.post("/reindex")
async def reindex():
    """Manually trigger database re-sync"""
    try:
        results = sql_sync.sync_all_excel_files()
        sql_sync.remove_orphaned_tables()
        return {
            "status": "success",
            "tables_synced": len(results['success']),
            "failed": len(results['failed'])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Using defaults if env not set
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)