from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
from typing import Dict, Any, List
import logging
import langchain
import re
import ast

# Enable global LangChain logging
langchain.verbose = True

logger = logging.getLogger(__name__)


from langchain_core.callbacks.base import BaseCallbackHandler

class SQLCaptureCallback(BaseCallbackHandler):
    def __init__(self):
        self.sql_query = None
        self.raw_result = None

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs: Any) -> Any:
        # Capture SQL if the tool looks like a SQL query tool
        name = serialized.get("name", "").lower()
        if "sql" in name:
            # Input might be a string or a dict
            if "SELECT" in input_str.upper():
                self.sql_query = input_str.replace('```sql', '').replace('```', '').strip()

    def on_tool_end(self, output: Any, **kwargs: Any) -> Any:
        # Capture the output if it looks like data
        if (isinstance(output, (list, tuple)) and len(output) > 0) or \
           (isinstance(output, str) and output.strip().startswith('[')):
            self.raw_result = output

class SQLAgent:
    def __init__(self, db_url: str, api_key: str, model_name: str, api_base: str):
        """Initialize SQL Agent with Self-Hosted LLM"""
        self.db = SQLDatabase.from_uri(db_url)
        self.api_key = api_key
        self.model_name = model_name
        self.api_base = api_base
        
        logger.info(f"🚀 Initializing SQL Agent with self-hosted LLM at {api_base}")
        
        # Initialize Primary LLM (Self-hosted/Local)
        self.llm = ChatOpenAI(
            openai_api_key=self.api_key,
            model_name=self.model_name,
            base_url=self.api_base,
            temperature=0.0,
            streaming=False
        )
        
        # Create Executor
        self.agent_executor = self._create_executor(self.llm)
        
        logger.info(f"✅ SQL Agent initialized with {len(self.db.get_usable_table_names())} tables (Self-hosted only)")

    def _create_executor(self, llm):
        prefix = """You are a senior power plant data analyst.

Your responsibilities:
- Always use SQL tools for any question that requires data.
- After executing SQL, ALWAYS return results only from sql_db_query tool. 
- Never answer from memory when data exists in the database.
- Use only existing tables and exact column names from the database schema.
- For totals, always use SUM.
- For counts, always use COUNT(DISTINCT).
- Never return raw rows for KPI questions.
- Prefer aggregated queries over raw row-level queries.
- Return minimal rows required to answer the question.
- Use LIMIT for top/bottom or listing queries.
- Use DISTINCT when listing names or categories.
- Ensure queries are valid for SQLite.

CRITICAL RULES:
- Table data is displayed separately in the UI. NEVER repeat raw data, numbers, or names in your text response.
- Provide INSIGHTS, analysis, trends, and comparisons ONLY.
- Speak in executive-level summaries. For example:
  * "CENTRAL COAL FIELDS LTD dominates the top rankings (4 out of 10 positions)."
  * "Northern Coal fields show strong consistency across the middle ranks."
  * "Private sector sidings are showing competitive performance."
- NEVER list out data like "1. Name, 2. Name".
- All tabular data must come from SQL execution; never describe data in text if it can be a table.
- ALWAYS use numbered lists (1. Insight) ONLY for analysis points, never for raw data.
- NEVER use comma-separated sentences for raw data records.
- For questions about 'top sidings', 'merit order', 'siding wise merit order', or 'top companies', YOU MUST USE SQL tool.
- NEVER return lists of data in natural language text alone; the table will be handled by the UI.
- If data is requested, it must come from a database query.
"""
        return create_sql_agent(
            llm=llm,
            toolkit=SQLDatabaseToolkit(db=self.db, llm=llm),
            verbose=True,
            agent_type="zero-shot-react-description",
            handle_parsing_errors=True,
            return_intermediate_steps=True,
            prefix=prefix
        )

    def query(self, question: str, session_id: str = None, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Execute natural language query and return raw tool results for backend parsing
        """
        try:
            logger.info(f"🔍 Agent executing query for: {question}")
            
            # 1. Initialize our spy callback
            capture_callback = SQLCaptureCallback()
            
            # BUILD CONTEXT FROM HISTORY
            context = ""
            if conversation_history and len(conversation_history) > 1:
                context = "\n\nPrevious conversation context:\n"
                for msg in conversation_history[-2:]:  # Last 1 exchange
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    context += f"{role}: {content}\n"
            
            # Modify input to include context
            enhanced_question = f"{context}\n\nCurrent question: {question}" if context else question
            input_data = {"input": enhanced_question}

            # 2. Execute with agent + CAPTURE CALLBACK
            result = self.agent_executor.invoke(
                input_data, 
                config={"callbacks": [capture_callback]}
            )
            
            answer = result.get("output", "")
            
            # 3. Use Captured Data as Primary Source
            sql_query = capture_callback.sql_query
            raw_result = capture_callback.raw_result

            # 4. Fallback: Parse intermediate_steps if captures are empty
            if not raw_result:
                for step in result.get("intermediate_steps", []):
                    try:
                        action, observation = step
                        # Check Tool Type & Observation Type
                        if isinstance(observation, (list, tuple)) and len(observation) > 0:
                            raw_result = observation
                            break
                        elif isinstance(observation, str) and observation.strip().startswith('['):
                            try:
                                parsed = ast.literal_eval(observation)
                                if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
                                    raw_result = parsed
                                    break
                            except: pass
                    except: continue

            # Final check/logger
            if raw_result:
                logger.info(f"✅ DATA CAPTURED: {len(str(raw_result))} chars")
            else:
                logger.error("❌ NO DATA CAPTURED - Even with callback")

            # Extract column names from SQL if possible
            columns = []
            if sql_query:
                try:
                    match = re.search(r'SELECT\s+(.+?)\s+FROM', sql_query, re.IGNORECASE | re.DOTALL)
                    if match:
                        col_text = match.group(1)
                        columns = [c.strip().split(' AS ')[-1].split('.')[-1].strip('"`[] ') for c in col_text.split(',')]
                except Exception as e:
                    logger.warning(f"Failed to extract column names from SQL: {e}")

            return {
                "answer": answer,
                "sql_query": sql_query,
                "raw_result": raw_result,
                "columns": columns,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"❌ Agent query failed: {e}")
            return {
                "answer": f"Error: {str(e)}",
                "sql_query": None,
                "raw_result": None,
                "columns": [],
                "success": False
            }
    
    def list_tables(self) -> list:
        """List all available tables"""
        return self.db.get_usable_table_names()
    
    def get_table_schema(self, table_name: str) -> str:
        """Get schema information for a specific table"""
        try:
            return self.db.get_table_info([table_name])
        except:
            return None