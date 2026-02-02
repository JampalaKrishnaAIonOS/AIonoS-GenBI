"""
FIXED Backend Main - Properly handles conversation context and visualization
"""

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
from typing import List

# Services
from services.excel_to_sql_sync import ExcelToSQLSync
from services.sql_agent import SQLAgent
from services.file_watcher_sql import start_file_watcher_sql
from services.chart_generator import ChartGenerator
from services.session_manager import SessionManager
from models.schemas import ChatRequest

load_dotenv()

def sanitize_for_json(obj):
    """Recursively convert numpy/pandas types to standard Python types"""
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
    elif isinstance(obj, (np.datetime64, pd.Timestamp, pd.Period, pd.Interval)):
        return str(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    elif pd.isna(obj):
        return None
    return obj

# Global instances
sql_sync = None
sql_agent = None
session_manager = SessionManager()
file_observer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global sql_sync, sql_agent, file_observer
    
    print("🚀 Starting GenBI SQL-based Chatbot...")
    
    excel_folder = os.getenv('EXCEL_FOLDER_PATH')
    db_url = os.getenv('DATABASE_URL', 'sqlite:///genbi.db')
    groq_api_key = os.getenv('GROQ_API_KEY')
    model_name = os.getenv('MODEL_NAME', 'llama-3.1-70b-versatile')
    
    # Initialize SQL sync
    sql_sync = ExcelToSQLSync(excel_folder, db_url)
    
    print("📊 Syncing Excel files to database...")
    try:
        results = sql_sync.sync_all_excel_files()
        print(f"✅ Synced {len(results['success'])} tables")
    except Exception as e:
        print(f"❌ Initial sync failed: {e}")
    
    # Initialize SQL Agent
    sql_agent = SQLAgent(db_url, groq_api_key, model_name)
    
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

# FIXED STREAM RESPONSE GENERATOR
async def stream_response_generator(question: str, session_id: str, conversation_history: List):
    """Stream SQL agent response with proper table and chart support"""
    try:
        # Get session for context
        session = session_manager.get_session(session_id)
        question_lower = question.lower().strip()
        
        # ✅ HANDLE "PLOT IT" OR "SHOW CHART" FOR PREVIOUS DATA
        is_plot_only = question_lower in ['plot it', 'show chart', 'visualize it', 'plot', 'chart it']
        
        if is_plot_only:
            last_df = session.get('last_dataframe')
            if last_df is None:
                yield json.dumps({"type": "error", "content": "No previous data available to plot. Please ask a data question first."}) + "\n"
                yield json.dumps({"type": "complete"}) + "\n"
                return
            
            yield json.dumps({"type": "status", "content": "Generating visualization for previous data..."}) + "\n"
            await asyncio.sleep(0.1)
            
            # Use last DF for chart
            df = last_df
            answer = "Here is the visualization of the data we just discussed."
            sql_query = None
            result_type = 'dataframe'
        else:
            # Status update
            yield json.dumps({"type": "status", "content": "Generating SQL query..."}) + "\n"
            await asyncio.sleep(0.1)
            
            # ✅ PASS CONVERSATION HISTORY TO SQL AGENT
            result = sql_agent.query(question, conversation_history=conversation_history)
            
            if not result['success']:
                yield json.dumps({"type": "error", "content": result.get('error', 'Unknown error')}) + "\n"
                yield json.dumps({"type": "complete"}) + "\n"
                return # 🔥 REQUIRED: Stop thinking
            
            answer = result['answer']
            sql_query = result['sql_query']
            df = result['data']
            result_type = result['type']
            
            # Store in session for "plot it"
            if df is not None and not df.empty:
                session['last_dataframe'] = df

        # Stream answer with markdown formatting
        yield json.dumps({"type": "answer_start"}) + "\n"
        
        import re
        tokens = re.split(r'(\s+)', answer)
        for token in tokens:
            if token:
                yield json.dumps({"type": "word", "content": token}) + "\n"
                await asyncio.sleep(0.01)
        
        yield json.dumps({"type": "answer_end"}) + "\n"
        
        # Send SQL query for transparency
        if sql_query:
            yield json.dumps({"type": "code", "content": sql_query}) + "\n"
        
        # ✅ SEND TABLE DATA IF DATAFRAME EXISTS
        if result_type == 'dataframe' and df is not None:
            # Prepare table data for frontend
            table_data = {
                "id": f"table_{hash(str(df.columns))}",
                "columns": [str(c) for c in df.columns],
                "rows": sanitize_for_json(df.head(100).to_dict(orient="records"))
            }
            yield json.dumps({"type": "table", "content": table_data}) + "\n"
            
            # ✅ AUTO-GENERATE CHART IF USER ASKS FOR VISUALIZATION
            skip_chart = any(k in question_lower for k in ['just', 'only', 'table', 'list'])
            
            should_chart = is_plot_only or any(kw in question_lower for kw in [
                'plot', 'chart', 'graph', 'visualize', 'show', 'display',
                'trend', 'comparison', 'compare', 'analysis', 'distribution'
            ])
            
            if not skip_chart and should_chart:
                try:
                    chart_type = ChartGenerator.detect_chart_type(question, df)
                    chart = ChartGenerator.generate_chart(
                        df, 
                        chart_type=chart_type, 
                        title=question if not is_plot_only else "Data Visualization"
                    )
                    
                    if chart:
                        # If chart returned an error payload, send it as error
                        if chart.get('type') == 'error':
                             yield json.dumps(chart) + "\n"
                        else:
                             yield json.dumps({"type": "chart", "content": chart}) + "\n"
                except Exception as chart_err:
                    print(f"Chart generation failed: {chart_err}")
                    yield json.dumps({"type": "error", "content": f"Visualization failed: {str(chart_err)}"}) + "\n"
        
        # Update session history
        session_manager.add_to_history(session_id, "user", question)
        session_manager.add_to_history(session_id, "assistant", answer)
        
        yield json.dumps({"type": "complete"}) + "\n"
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_msg = f"System error: {str(e)}"
        yield json.dumps({"type": "error", "content": error_msg}) + "\n"
        yield json.dumps({"type": "complete"}) + "\n"
        return # 🔥 REQUIRED

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
    uvicorn.run(app, host="0.0.0.0", port=8000)