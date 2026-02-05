"""
FIXED Backend Main - Properly handles conversation context and visualization
"""

import os
import asyncio
import logging
import json
import numpy as np
from pathlib import Path
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
import pandas as pd
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
from services.chart_generator import ChartGenerator
from services.session_manager import session_manager
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
    elif isinstance(obj, (np.float64, np.float32, np.floating, float)):
        if not np.isfinite(obj):
            return None
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
file_observer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    global sql_sync, sql_agent, file_observer
    
    print("🚀 Starting GenBI SQL-based Chatbot...")
    
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

# FIXED STREAM RESPONSE GENERATOR
async def stream_response_generator(question: str, session_id: str, conversation_history: List):
    """Stream SQL agent response with proper table and chart support"""
    try:
        logger.info(f"Processing question: {question} (session: {session_id})")
        
        # Get session for context
        session = session_manager.get_session(session_id)
        
        # ✅ Fix 6 — Add Hard Input Truncation (Safety Net)
        if len(question) > 1000:
            logger.warning(f"Truncating long question from {len(question)} to 1000 chars")
            question = question[:1000]
            
        question_lower = question.lower().strip()
        
        # ✅ HANDLE "PLOT IT" OR "SHOW CHART" FOR PREVIOUS DATA
        plot_keywords = ['plot', 'chart', 'visualize', 'graph', 'visual']
        is_plot_request = any(k in question_lower for k in plot_keywords)
        # It's a "plot only" request if it's very short and contains one of these, OR is a specific known phrase
        is_plot_only = is_plot_request and (len(question_lower.split()) <= 4 or question_lower in ['plot it', 'show chart', 'visualize it', 'plot', 'chart it', 'graph it'])
        
        if is_plot_only:
            last_df = session_manager.get_last_dataframe(session_id)
            
            # Fallback to last_result in context if last_dataframe is missing
            if last_df is None or (isinstance(last_df, pd.DataFrame) and last_df.empty):
                session = session_manager.get_session(session_id)
                last_result = session.get("context", {}).get("last_result")
                if isinstance(last_result, pd.DataFrame) and not last_result.empty:
                    last_df = last_result
                    logger.info("Recovered previous dataframe from context['last_result']")

            if last_df is None or (isinstance(last_df, pd.DataFrame) and last_df.empty):
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
            
            # ✅ Fix 1 — Limit Conversation History (MOST IMPORTANT)
            short_history = conversation_history[-4:] if conversation_history else []
            result = sql_agent.query(question, conversation_history=short_history)
            
            if not result['success']:
                yield json.dumps({"type": "error", "content": result.get('error', 'Unknown error')}) + "\n"
                yield json.dumps({"type": "complete"}) + "\n"
                return
            
            answer = result['answer']
            sql_query = result['sql_query']
            df = result['data']
            # HIGH CONFIDENCE LOGGING
            logger.info(f"DEBUG after agent: success={result['success']}, df type={type(df)}, rows={len(df) if df is not None else 'None'}")
            if result.get('success') and df is not None:
                logger.info(f"DEBUG agent result keys: {list(result.keys())}")
                if isinstance(df, pd.DataFrame):
                    logger.info(f"DEBUG first few rows sample:\n{df.head(2).to_string()}")
            
        # ALWAYS store dataframe if we have one
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            logger.info(f"DEBUG: Setting last_dataframe with {len(df)} rows")
            session_manager.set_last_dataframe(session_id, df)
            # Also sync to context for fallback recovery
            session_manager.update_context(session_id, last_result=df)

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
        if df is not None:
            # Prepare table data for frontend
            table_data = {
                "columns": [str(c) for c in df.columns],
                "rows": sanitize_for_json(df.head(100).to_dict(orient="records"))
            }
            yield json.dumps({"type": "table", "content": table_data}) + "\n"
            
            # ✅ AUTO-GENERATE CHART ONLY IF USER EXPLICITLY ASKS FOR VISUALIZATION
            # As per user feedback, visualization should only come when mentioned (plot, chart, etc.)
            should_chart = is_plot_request or any(kw in question_lower for kw in ['plot', 'chart', 'graph', 'visualize', 'visualization'])
            
            # Allow skipping if explicitly told not to
            skip_chart = any(k in question_lower for k in ['no chart', 'no plot', 'without chart', 'only table', 'just table'])
            if skip_chart:
                should_chart = False
            
            logger.info(f"DEBUG visualization decision: should_chart={should_chart}, is_plot_request={is_plot_request}, df_rows={len(df) if df is not None else 'None'}")
            
            if not skip_chart and should_chart and df is not None:
                try:
                    # 3️⃣ Send DataFrame directly to ChartGenerator
                    logger.info("Generating chart using ChartGenerator...")
                    chart = ChartGenerator.generate_chart(df, title=question)
                    if chart and chart.get('type') == 'chart':
                        yield json.dumps({"type": "chart", "content": chart}) + "\n"
                except Exception as chart_err:
                    logger.error(f"Visualization pipeline failed: {chart_err}")
                    if is_plot_request:
                        yield json.dumps({"type": "error", "content": f"Visualization failed: {str(chart_err)}"}) + "\n"
        
        # Update session history
        session_manager.add_to_history(session_id, "user", question)
        session_manager.add_to_history(session_id, "assistant", answer)
        
        # ✅ Fix 7 — Clear Session Context When It Grows
        session_manager.trim_history(session_id, max_messages=10)
        
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
    host = os.getenv("HOST")
    port = int(os.getenv("PORT"))
    uvicorn.run(app, host=host, port=port)