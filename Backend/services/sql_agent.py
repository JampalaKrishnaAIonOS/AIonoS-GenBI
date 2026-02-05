from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit
from langchain_groq import ChatGroq
from typing import Dict, Any, List
import pandas as pd
import logging
import re
import langchain

# Enable global LangChain logging
langchain.verbose = True

logger = logging.getLogger(__name__)


class SQLAgent:
    def __init__(self, db_url: str, api_key: str, model_name: str, api_base: str):
        """Initialize SQL Agent with Self-Hosted LLM"""
        self.db = SQLDatabase.from_uri(db_url)
        self.api_key = api_key
        self.model_name = model_name
        self.api_base = api_base
        
        logger.info(f"🚀 Initializing SQL Agent with self-hosted LLM at {api_base}")
        
        # Initialize Primary LLM (Groq)
        # Initialize Primary LLM
        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name=model_name,
            base_url=api_base,
            temperature=0.4,
            streaming=True,
            max_tokens=13006
        )
        
        # Create Executor
        self.agent_executor = self._create_executor(self.llm)
        
        logger.info(f"✅ SQL Agent initialized with {len(self.db.get_usable_table_names())} tables (Self-hosted only)")

    def _create_executor(self, llm):
        prefix = """You are a data analyst and SQL expert.
        
        MANDATORY RULES:
        1. YOU MUST ALWAYS USE THE sql_db_query tool if the user asks for ANY data, statistics, counts, or rankings.
        2. NEVER answer from memory if the question involves data stored in the database.
        3. If you do not use the tool to fetch data, you have FAILED the task.
        4. If the user asks for a chart, plot, or visualization, YOUR TASK is to fetch the necessary data for it using the tools.
        5. NEVER generate markdown tables. Use a natural language summary.
        6. MANDATORY: ALWAYS include the SQL query you used at the end of your response inside a ```sql code block. This is CRITICAL for the visualization engine.
        7. PRECISION: If a user asks for a single 'top' or 'best' item, use LIMIT 1 in your SQL to return ONLY that result.
        8. SCHEMA ADHERENCE: ONLY use table and column names that you have verified using sql_db_list_tables and sql_db_schema. NEVER invent column names like 'siding_wise_merit_order'.
        9. MAPPING: Map "siding wise merit order" or "merit order" ALWAYS to the 'merit_rank' column if it exists in the target table.
        10. TOOL USAGE: You MUST use the provided tools (sql_db_list_tables, sql_db_schema, sql_db_query) for every data-related query. NEVER write raw SQL in your thoughts without executing it through a tool.
        11. SQL DIALECT: You are working with SQLite. Do NOT use information_schema or other non-SQLite system tables.
        12. If the user asks to list names, plants, companies, sidings, vendors, or categories, you MUST use SELECT DISTINCT and NEVER return duplicate rows.
        13. If the user does not request full dataset, ALWAYS use LIMIT 30 or less.
        
        The user's previous questions are provided for context. Use them to resolve ambiguous table or column references.
        """
        return create_sql_agent(
            llm=llm,
            toolkit=SQLDatabaseToolkit(db=self.db, llm=llm),
            verbose=False,
            agent_type="openai-tools",
            handle_parsing_errors=True,
            return_intermediate_steps=False,
            prefix=prefix
        )

    
    def _get_table_context(self) -> str:
        """Get available tables with sample data"""
        tables = self.db.get_usable_table_names()
        context = f"Available tables: {', '.join(tables)}\n\n"
        
        # Get schema for all tables
        for table in tables:
            try:
                schema_info = self.db.get_table_info([table])
                context += f"{schema_info}\n\n"
            except:
                pass
        
        return context
    
    def _generate_sql_query(self, question: str, conversation_history: List[Dict] = None) -> str:
        """Generate SQL query using LLM"""
        
        # Build conversation context
        history_text = ""
        if conversation_history and len(conversation_history) > 0:
            history_text = "\n\nPrevious conversation:\n"
            for msg in conversation_history[-2:]:  # Last 1 exchange
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                history_text += f"{role.capitalize()}: {content}\n"
        
        system_prompt = f"""You are a SQL expert. Generate ONLY a SQL query to answer the user's question.

Available tables: {', '.join(self.db.get_usable_table_names())}

CRITICAL RULES:
1. ONLY use tables that exist: {', '.join(self.db.get_usable_table_names())}
2. Check column names carefully - use EXACT column names from the schema above.
3. NEVER invent or hallucinate column names. If a column isn't in the schema, you cannot use it.
4. Return ONLY the SQL query, no explanations.
5. Use proper SQL syntax for the database.
6. For aggregations, use appropriate GROUP BY clauses.
7. Case-insensitive WHERE clauses: use LOWER(column) = LOWER('value').
8. SQLite DOES NOT support PERCENTILE_CONT. To calculate median (50th percentile) for a column 'X' in table 'T', use:
   SELECT X FROM T ORDER BY X LIMIT 1 OFFSET (SELECT COUNT(*) / 2 FROM T)
9. ALWAYS use this LIMIT/OFFSET pattern for medians, never PERCENTILE_CONT.
10. FOR SINGULAR QUESTIONS (e.g., "Which is...", "Top one...", "The best..."), ALWAYS use LIMIT 1 to return only the most relevant result.
11. BUSINESS LOGIC: "siding wise merit order" = 'merit_rank'. Use 'merit_rank' whenever merit order is mentioned.
12. FUZZY MATCHING: For plant names (like 'Dadri', 'Barh', etc.), use LOWER(column) LIKE '%value%' in the WHERE clause if an exact match fails or for better coverage.
13. SQLite LIMITATION: Do NOT use information_schema. If you need to list tables, use SELECT name FROM sqlite_master WHERE type='table'.

{history_text}

Current question: {question}

SQL Query:"""

        try:
            response = self.llm.invoke(system_prompt)
            sql_query = response.content.strip()
            
            # Clean the query
            sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
            
            logger.info(f"Generated SQL: {sql_query}")
            return sql_query
            
        except Exception as e:
            logger.error(f"Failed to generate SQL: {e}")
            return None
    
    def _enforce_limits(self, sql: str) -> str:
        """Apply hard safety limits to the SQL query"""
        lower = sql.lower()

        # Add DISTINCT for plant_name listing
        if "select plant_name" in lower and "distinct" not in lower:
            sql = sql.replace("SELECT Plant_Name", "SELECT DISTINCT Plant_Name")
            sql = sql.replace("select plant_name", "select DISTINCT plant_name")

        # Add LIMIT if missing
        if "limit" not in lower:
            sql = sql.rstrip(";").strip() + " LIMIT 30"

        return sql

    def _execute_sql_to_dataframe(self, sql_query: str, question: str = None) -> pd.DataFrame:
        """Execute SQL and return DataFrame"""
        try:
            # 2️⃣ Block non-aggregated SQL for charts
            if question:
                chart_words = ["chart", "plot", "pie", "graph", "visual"]
                if any(w in question.lower() for w in chart_words):
                    if "group by" not in sql_query.lower():
                        logger.error("🛑 Blocked non-aggregated chart query")
                        raise ValueError("Chart queries must use aggregated SQL with GROUP BY.")

            # 🔥 Fix #2 — Hard Safety Net After SQL Generation
            sql_query = self._enforce_limits(sql_query)

            # Sanitize for SQLite before execution
            sql_query = self._sanitize_sql_for_sqlite(sql_query)
            df = pd.read_sql(sql_query, self.db._engine)
            
            # 🔥 Fix #3 — Block Giant Result Sets
            if len(df) > 5000:
                logger.error(f"🛑 Query returned too many rows: {len(df)}")
                raise ValueError("Query returned too many rows. Please use aggregation or filtering.")
                
            logger.info(f"Query returned {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
            raise

    def _sanitize_sql_for_sqlite(self, sql: str) -> str:
        """Sanitize SQL query for SQLite dialect and block dangerous system table access"""
        lower_sql = sql.lower().strip()
        
        # 1. Map common generic system table queries to SQLite equivalents
        if "information_schema.tables" in lower_sql or "information_schema.columns" in lower_sql:
            logger.warning("📍 Intercepted information_schema query, mapping to sqlite_master")
            return "SELECT name FROM sqlite_master WHERE type='table';"
            
        # 2. Block forbidden system tables that will crash SQLite
        forbidden = [
            "information_schema",
            "pg_catalog",
            "pg_tables",
            "sys.tables",
            "master..sys"
        ]
        
        for k in forbidden:
            if k in lower_sql:
                logger.error(f"🚫 Blocking unsupported SQL dialect: {k}")
                raise ValueError(f"Unsupported system table for SQLite: {k}")
                
        return sql
    
    def _generate_natural_response(self, question: str, df: pd.DataFrame, sql_query: str) -> str:
        """Generate natural language response with proper markdown formatting"""
        
        # Create summary of the data
        data_summary = f"Query returned {len(df)} rows with {len(df.columns)} columns.\n\n"
        
        if len(df) > 0:
            data_summary += "**Sample data:**\n"
            data_summary += df.head(10).to_string(index=False)
        
        system_prompt = """You are a data analyst explaining query results to business users.

FORMATTING RULES:
1. Use **bold** for important numbers, names, and key findings
2. Use bullet points for lists
3. Use proper paragraph breaks (double newline)
4. Start with a direct answer to the question
5. Be concise but informative
6. DO NOT use excessive headers (###) unless really needed
7. Format numbers with commas for readability
8. SINGULAR ANSWERS: If the question asks for one specific item (e.g., "Which company..."), provide that item as the primary answer and avoid listing other secondary results from the data summary unless they are vital for context.

Example good response:
The total cost for Barh plant is **₹417,663,677**. This is based on the optimized allocation across all coal companies.

Example bad response:
### Analysis Results
The analysis shows that...
"""

        user_prompt = f"""Question: {question}

Data Summary:
{data_summary}

Provide a clear, well-formatted answer using the data above. Be direct and use markdown formatting properly."""

        try:
            response = self.llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            # Fallback to simple summary
            if len(df) == 1 and len(df.columns) == 1:
                value = df.iloc[0, 0]
                return f"The result is **{value:,}** (if applicable)." if isinstance(value, (int, float)) else f"The result is **{value}**."
            else:
                return f"Found **{len(df)} results** matching your query."
    
    def query(self, question: str, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Execute natural language query using LangChain SQL Agent
        """
        try:
            logger.info(f"🔍 Agent executing query for: {question}")
            
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
            
            # Prepare the input
            input_data = {"input": enhanced_question}
            
            # Execute with agent (this will trigger the verbose logs)
            # We use return_intermediate_steps to extract SQL and results
            result = self.agent_executor.invoke(input_data)
            
            # HIGH CONFIDENCE LOGGING
            logger.info(f"DEBUG: agent output type: {type(result.get('output'))}")
            logger.info(f"DEBUG: intermediate_steps count: {len(result.get('intermediate_steps', []))}")

            answer = result.get("output", "")
            
            # Try to extract the SQL query and dataset from intermediate steps
            sql_query = None
            df = None
            
            # The agent might have performed multiple steps. We want the ones related to querying.
            for step in result.get("intermediate_steps", []):
                # Handle both tuple/list and AgentStep objects
                if isinstance(step, (list, tuple)) and len(step) >= 2:
                    action, observation = step[0], step[1]
                elif hasattr(step, 'action') and hasattr(step, 'observation'):
                    action, observation = step.action, step.observation
                else:
                    continue
                
                # Check for SQL calling tools
                tool_name = None
                tool_input = None
                
                if hasattr(action, 'tool'):
                    tool_name = action.tool
                    tool_input = action.tool_input
                elif isinstance(action, dict):
                    tool_name = action.get('tool')
                    tool_input = action.get('tool_input')

                # Support various tool names used by SQL toolkits
                sql_tool_names = ["sql_db_query", "query_sql_db", "sql_db_query_checker", "execute_sql"]
                is_sql_tool = tool_name and any(name in tool_name.lower() for name in sql_tool_names)
                
                if is_sql_tool:
                    # Extract query
                    current_query = None
                    if isinstance(tool_input, dict):
                        current_query = tool_input.get('query') or tool_input.get('sql')
                    else:
                        current_query = str(tool_input)
                    
                    if current_query:
                        # Basic cleanup
                        sql_query = current_query.replace('```sql', '').replace('```', '').strip()
                        # If it's a dict-looking string, try to extract 'query'
                        if sql_query.startswith('{') and '"query":' in sql_query:
                            try:
                                import json
                                sql_query = json.loads(sql_query).get('query', sql_query)
                            except:
                                pass
                                
                        logger.info(f"📍 Found SQL query: {sql_query[:100]}...")
                        
                        # Execute to get DataFrame
                        try:
                            temp_df = self._execute_sql_to_dataframe(sql_query, question=question)
                            if temp_df is not None and not temp_df.empty:
                                df = temp_df
                                logger.info(f"✅ Successfully extracted DataFrame with {len(df)} rows")
                        except Exception as e:
                            logger.error(f"❌ Failed to execute extracted SQL: {e}")
            
            # Fallback if no intermediate steps captured the query
            if not sql_query:
                # If intermediate_steps failed, try extracting SQL from output text
                if answer:
                    import re
                    # Look for SQL SELECT statements in the output text or code blocks
                    # Handle both markdown code blocks and raw text
                    sql_match = re.search(r'```sql\s+(.*?)\s+```', answer, re.IGNORECASE | re.S)
                    if not sql_match:
                        # More permissive regex for SELECT statements
                        sql_match = re.search(r'(SELECT\s+.+?)(?:;|\n\s*\n|$)', answer, re.IGNORECASE | re.S)
                    
                    if sql_match:
                        extracted_sql = sql_match.group(1).strip()
                        # Basic validation to ensure it looks like a query
                        if 'FROM' in extracted_sql.upper():
                            sql_query = extracted_sql
                            logger.info(f"📍 Extracted SQL from output text fallback: {sql_query[:100]}...")
                            try:
                                temp_df = self._execute_sql_to_dataframe(sql_query, question=question)
                                if temp_df is not None and not temp_df.empty:
                                    logger.info(f"✅ Fallback execution successful: {len(temp_df)} rows")
                                    # ✅ RETURN IMMEDIATELY WITH DATA
                                    return {
                                        'answer': answer,
                                        'sql_query': sql_query,
                                        'data': temp_df,
                                        'type': 'dataframe',
                                        'success': True
                                    }
                            except Exception as e:
                                logger.error(f"❌ Fallback execution failed: {e}")

            if not sql_query:
                # If agent didn't use the tool but gave an answer, it might be conversational
                return {
                    'answer': answer,
                    'sql_query': None,
                    'data': None,
                    'type': 'text',
                    'success': True
                }

            # ✅ GENERATE CLEAN NATURAL RESPONSE IF DATA EXISTS
            # This prevents the LLM from hallucinating/formatting its own markdown tables
            if df is not None and not df.empty:
                try:
                    logger.info("Generating clean summary Answer from DataFrame...")
                    answer = self._generate_natural_response(question, df, sql_query)
                except Exception as e:
                    logger.error(f"Natural response generation failed: {e}")
                    # Keep existing answer if generation fails

            return {
                'answer': answer,
                'sql_query': sql_query,
                'data': df if (df is not None and not df.empty) else None,
                'type': 'dataframe' if (df is not None and not df.empty) else 'text',
                'success': True
            }
            
        except Exception as e:
            logger.error(f"❌ Agent query failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'answer': f"I encountered an error while processing your request: {str(e)}",
                'sql_query': None,
                'data': None,
                'type': 'error',
                'success': False,
                'error': str(e)
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