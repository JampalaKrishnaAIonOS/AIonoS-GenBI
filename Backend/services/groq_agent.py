import os
import pandas as pd
from groq import Groq
from typing import Dict, Any, List, Optional
import json
import re
import traceback
import ast

METRIC_INTENTS = {
    "cost": ["cost", "amount", "value", "price", "rs", "usd", "savings", "expenditure"],
    "quantity": ["qty", "quantity", "volume", "count", "units", "tonnage", "allocated"],
    "energy": ["energy", "kcal", "mw", "power", "gwh", "mu", "generation"],
    "percentage": ["percent", "%", "ratio", "share", "allocation", "contribution", "pct"],
    "rank": ["rank", "merit", "position", "order", "priority"]
}

def infer_best_numeric_column(df: pd.DataFrame, intent_keywords: list) -> Optional[str]:
    """Dynamically infer the best numeric column based on intent keywords and heuristics"""
    candidates = []

    # Get only numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    
    for col in numeric_df.columns:
        score = 0
        col_name = str(col).lower()

        # 1. Name Similarity (highest weight)
        for kw in intent_keywords:
            if kw in col_name:
                score += 5  # Increased weight for keyword match

        # Get data sample for heuristics
        series = df[col].dropna()
        if series.empty:
            continue

        # 2. Heuristics (generic, domain-free)
        # Positive values are more likely to be metrics of interest
        if (series >= 0).all():
            score += 1
            
        # Non-zero values are better
        if series.nunique() > 1:
            score += 1
            
        # Large values often indicate totals/costs
        if series.mean() > 100:
            score += 1

        candidates.append((col, score))

    if not candidates:
        return None

    # Sort by score descending, then by original column order as tie-breaker
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Only return if the best score is significantly positive
    if candidates[0][1] > 0:
        return candidates[0][0]
    
    return None

class GroqPandasAgent:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        # Use model from environment when provided (keeps default for older setups)
        self.model = os.getenv('MODEL_NAME', 'moonshotai/kimi-k2-instruct')
        
    def create_system_prompt(self, df: pd.DataFrame, file_name: str, sheet_name: str) -> str:
        """Create a detailed system prompt with DataFrame context"""
        
        # Get DataFrame info
        columns_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            sample_vals = df[col].dropna().head(3).tolist()
            columns_info.append(f"  - {col} ({dtype}): {sample_vals}")
        
        COLUMN_ALIASES = {
            'coal cost': ['coal_cost', 'total_coal_cost'],
            'freight cost': ['freight_cost', 'total_freight_cost'],
            'cost per 1000 kcal': ['cost_per_1000_kcal', 'cost_1000_kcal'],
            'energy': ['total_energy', 'energy_sent_out', 'energy_generated']
        }
        
        prompt = f"""You are a data analyst assistant working with a pandas DataFrame.

**Current Data Context:**
- File: {file_name}
- Sheet: {sheet_name}
- Shape: {df.shape[0]} rows × {df.shape[1]} columns
- DataFrame variable name: `df`

**Columns:**
{chr(10).join(columns_info)}

**Your Task:**
1. Analyze the user's question carefully
2. Write ONLY the pandas code needed to answer it
3. Use ONLY columns that exist in the DataFrame
4. Return code that produces a result (number, DataFrame, Series, etc.)

**Rules:**
- Use `df` as the DataFrame variable
- Write clean, efficient pandas code
- For aggregations: df.groupby().agg()
- For filtering: df[df['column'] condition]
- For sorting: df.sort_values()
- Always handle missing values: .dropna() or .fillna()
- Return the final result, don't just print it
- IMPORTANT FOR CHARTS: NEVER use `df.plot()`, `matplotlib`, or attempt to render figures in the generated code.
    If the user requests a visualization, return a `DataFrame` or `Series` that contains the data to plot.
    The backend will call `ChartGenerator.generate_chart(...)` to create a Plotly figure JSON from that data.
    Do NOT return Plotly JSON or call plotting libraries from the generated code.
- When the user explicitly specifies a chart type (for example: "pie", "line", "bar"), preserve that intent exactly.
    Do NOT reinterpret, generalize, or override the requested chart type in the generated code.

- CRITICAL FILE RULE:
    Never hardcode file paths or filenames.
    Always access data via `df` (single sheet) or `dfs` (multi-sheet dictionary).
    DO NOT call pd.read_excel() inside generated code.

- CRITICAL COLUMN RULE:
    Before using any column name, verify it exists in df.columns (or each DataFrame in dfs).
    If a column does not exist, choose the closest matching column from the schema.
    If user mentions a metric, map it to the closest matching column using these known aliases if applicable:
    {json.dumps(COLUMN_ALIASES, indent=4)}
    NEVER assume column names.

- CRITICAL: NEITHER `df.append()` NOR `series.append()` exist in modern pandas (2.0+). 
    They will cause an IMMEDIATE crash.
    To combine results, ALWAYS use `pd.concat([item1, item2])`. 
    Example: To join a row to a DataFrame: `pd.concat([df, row_as_df])`.
    Example: To join two series: `pd.concat([s1, s2])`.
    NEVER use `.append()`.
- LIBRARIES: You have access to `df` (DataFrame), `pd` (pandas), `np` (numpy), and `stats` (scipy.stats).
- CRITICAL: DO NOT WRITE ANY `import` STATEMENTS. The necessary libraries are already provided in the execution environment. Any code containing `import` will be rejected.
- SYNTAX WARNING: `df.columns` and `df.index` are properties, NOT functions. Do not write `df.columns()`. Use `df.columns`.
- To calculate correlation significance: Use `stats.pearsonr(df['col1'], df['col2'])`.
- To use numpy: Use `np.mean()`, etc.
- For broad requests like "summarize the data" or "give me insights":
    Return a comprehensive overview (e.g., `df.describe(include='all')`, or just `df` if you want the backend to generate a detailed textual summary from a sample).
    The goal is to provide enough data for a high-quality textual report.

**Output Format:**
Return ONLY Python code, no explanations, no markdown, no ```python``` tags.
Just raw executable code.

**Example:**
User: "What's the total sales?"
You: df['sales'].sum()

User: "Show top 5 products by revenue"
You: df.groupby('product')['revenue'].sum().sort_values(ascending=False).head(5)

HYBRID DATA ACCESS RULES (CRITICAL):

You have access to TWO variables:

1. df  
- Represents the SINGLE best-matching sheet.
- Use this for questions about a specific plant, file, or sheet.

2. dfs  
- A dictionary of ALL relevant sheets.
- Key: "filename::sheetname"
- Value: pandas DataFrame

YOU MUST CHOOSE CORRECTLY:

- If the user asks about:
  "this sheet", "Barh", "Kudgi", "Dadri", "this plant"
  → Use df

- If the user asks about:
  "all sheets", "combined", "across plants", "overall", "total across files"
  → Use dfs

- If the user asks about specific rankings ("top 5", "highest", "best rank"):
  → Use `df.sort_values(by='col', ascending=False).head(5)`
  → If across multiple plants, use `pd.concat(dfs.values())` first and then sort.

EXAMPLES:

Single-sheet:
df['total_cost'].sum()

Multi-sheet Total (across and sum totals of all files):
sum(d['total_cost'].sum() for d in dfs.values())

Top N across multiple files:
combined = pd.concat(dfs.values())
combined.sort_values('allocation', ascending=False).head(5)['total_cost'].sum()
"""
        return prompt

    def is_data_question(self, question: str) -> bool:
        """Heuristic to decide if the user's question is asking for data analysis."""
        q = (question or '').lower()
        keywords = [
            'analyz', 'summar', 'plot', 'chart', 'graph', 'visualiz', 
            'mean', 'median', 'average', 'avg',
            'sum', 'total', 'count', 'percentage', 'percent', 'split',
            'compare', 'correl', 'regress', 'trend', 'calculate', 'efficien',
            'show', 'top', 'bottom', 'groupby', 'aggregate', 'filter', 'select', 'rows', 'columns'
        ]
        return any(kw in q for kw in keywords)

    def generate_conversational_reply(self, question: str, conversation_history: List[Dict] = None) -> str:
        """Return a conversational reply from the LLM (no code)."""
        # Build a light system prompt that asks the model to reply conversationally
        system_msg = (
            "You are a helpful GenBI assistant. Answer the user's question conversationally. "
            "If the user is asking about the data (e.g., 'what can you do?', 'what data is this?'), "
            "explain your capabilities: you can summarize sheets, generate charts, and perform deep analysis. "
            "Do NOT generate any code or python. Keep the reply concise, professional and helpful."
        )

        messages = [{"role": "system", "content": system_msg}]
        if conversation_history:
            for msg in conversation_history[-5:]:
                messages.append(msg)

        messages.append({"role": "user", "content": question})

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.5,
                max_tokens=200
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return "I can chat about your data and answer questions — if you'd like analysis, ask me to analyze or plot a sheet."
    
    def generate_pandas_code(self, question: str, df: pd.DataFrame, 
                            file_name: str, sheet_name: str, 
                            conversation_history: List[Dict] = None) -> str:
        """Generate pandas code using Groq LLM"""
        
        system_prompt = self.create_system_prompt(df, file_name, sheet_name)
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history for context
        if conversation_history:
            for msg in conversation_history[-5:]:  # Last 5 messages
                messages.append(msg)
        
        messages.append({"role": "user", "content": question})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=500
            )
            
            code = response.choices[0].message.content.strip()
            
            # Clean up code
            code = code.replace('```python', '').replace('```', '').strip()
            
            return code
            
        except Exception as e:
            raise Exception(f"Groq API Error: {str(e)}")
    
    def execute_code(self, code: str, df: pd.DataFrame, dfs: Dict[str, pd.DataFrame] = None, question: str = "") -> Dict[str, Any]:
        """Safely execute pandas code and return results"""
        code = (code or '').strip()
        question_lower = (question or "").lower()

        # Hard guard and auto-fix for deprecated .append() usage
        if ".append(" in code:
            # Try to auto-fix common simple .append(other) -> pd.concat([self, other])
            fixed_code = self._fix_deprecated_code(code)
            if fixed_code != code:
                print(f"🔧 Auto-fixed deprecated .append() usage:\nOriginal: {code}\nFixed: {fixed_code}")
                code = fixed_code
            else:
                return {
                    'type': 'error',
                    'error': "Generated code uses deprecated pandas `.append()`. In pandas 2.0+, use `pd.concat()` instead.",
                    'code': code
                }

        # Disallow import statements from generated code — model should use provided `df` and `pd`.
        try:
            parsed_ast = ast.parse(code)
            import_nodes = [n for n in parsed_ast.body if isinstance(n, (ast.Import, ast.ImportFrom))]
            if import_nodes:
                imports = []
                for n in import_nodes:
                    if isinstance(n, ast.Import):
                        for alias in n.names:
                            imports.append(alias.name)
                    else:
                        module = n.module or ''
                        for alias in n.names:
                            imports.append(f"{module}.{alias.name}")

                return {
                    'type': 'error',
                    'error': 'Import statements are not allowed in generated code. Use the provided `df` and `pd` (pandas) objects instead.',
                    'found_imports': imports,
                    'code': code
                }
        except Exception:
            # If AST parsing fails, fall through and let the existing error handling capture it.
            pass

        # Create execution environments
        import numpy as np
        try:
            from scipy import stats
        except ImportError:
            stats = None
            
        # Define safe built-ins
        safe_builtins = {
            'abs': abs, 'all': all, 'any': any, 'ascii': ascii, 'bin': bin, 'bool': bool,
            'bytearray': bytearray, 'bytes': bytes, 'callable': callable, 'chr': chr,
            'complex': complex, 'delattr': delattr, 'dict': dict, 'divmod': divmod,
            'enumerate': enumerate, 'filter': filter, 'float': float, 'format': format,
            'frozenset': frozenset, 'getattr': getattr, 'hasattr': hasattr, 'hash': hash,
            'hex': hex, 'id': id, 'int': int, 'isinstance': isinstance, 'issubclass': issubclass,
            'iter': iter, 'len': len, 'list': list, 'map': map, 'max': max, 'min': min,
            'next': next, 'object': object, 'oct': oct, 'ord': ord, 'pow': pow,
            'print': print, 'property': property, 'range': range, 'repr': repr,
            'reversed': reversed, 'round': round, 'set': set, 'setattr': setattr,
            'slice': slice, 'sorted': sorted, 'str': str, 'sum': sum, 'super': super,
            'tuple': tuple, 'type': type, 'vars': vars, 'zip': zip
        }
        
        safe_globals = {"__builtins__": safe_builtins}
        
        # 1. Access Tracking Wrapper
        class AccessTrackingDict(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.accessed_keys = set()
            def __getitem__(self, key):
                self.accessed_keys.add(key)
                return super().__getitem__(key)
            def get(self, key, default=None):
                self.accessed_keys.add(key)
                return super().get(key, default)
            def values(self):
                self.accessed_keys.update(self.keys())
                return super().values()
            def items(self):
                self.accessed_keys.update(self.keys())
                return super().items()

        # Wrap dfs to track usage
        tracked_dfs = AccessTrackingDict(dfs) if dfs else None
        
        local_vars = {
            'df': df,
            'dfs': tracked_dfs, 
            'pd': pd, 
            'np': np, 
            'stats': stats
        }

        # Log the code for debugging
        print("--- Executing generated code ---")
        print(code)
        print("--- End generated code ---")

        # GENBI OVERRIDE: Intent-driven execution for simple aggregations and comparisons
        # If the user asks for a 'total', 'sum', 'overall', OR 'compare'/'scatter', we calculate it directly from the schema.
        # FIX: Exclude queries that involve ranking (top, bottom, best, etc.) as they require row-level logic
        ranking_keywords = ['top', 'bottom', 'best', 'worst', 'highest', 'lowest', 'rank', 'limit', 'head', 'tail']
        is_ranking = any(k in question_lower for k in ranking_keywords)
        is_agg = any(k in question_lower for k in ['total', 'sum', 'overall', 'combined']) and not is_ranking
        is_comp = any(k in question_lower for k in ['compare', 'scatter', 'vs', 'plot']) and len(dfs or {}) > 1

        if is_agg or is_comp:
            check_dict = dfs if dfs else {'df': df}
            accessed_keys = list(check_dict.keys())

            if is_comp:
                # Resolve TWO metrics for comparison (e.g., cost vs energy)
                found_intents = []
                for intent, keywords in METRIC_INTENTS.items():
                    if intent in question_lower:
                        found_intents.append((intent, keywords))
                
                if len(found_intents) >= 2:
                    rows = []
                    intent_x, keywords_x = found_intents[0]
                    intent_y, keywords_y = found_intents[1]

                    for name, d in check_dict.items():
                        col_x = infer_best_numeric_column(d, keywords_x)
                        col_y = infer_best_numeric_column(d, keywords_y)

                        if col_x and col_y:
                            # Extract and normalize
                            tmp = d[[col_x, col_y]].copy()
                            tmp.columns = [f"{intent_x}", f"{intent_y}"]
                            tmp['source'] = name
                            rows.append(tmp)
                    
                    if rows:
                        combined_df = pd.concat(rows, ignore_index=True)
                        print(f"⚡ GENBI OVERRIDE: Executed direct '{intent_x}' vs '{intent_y}' comparison.")
                        return {
                            'type': 'dataframe',
                            'data': combined_df,
                            'rows_returned': len(combined_df),
                            'accessed_keys': accessed_keys
                        }

            # Fallback to single metric aggregation if not comparison or comparison failed
            for intent, keywords in METRIC_INTENTS.items():
                if intent in question_lower:
                    totals = {}
                    for name, d in check_dict.items():
                        col = infer_best_numeric_column(d, keywords)
                        if not col:
                            return {
                                'type': 'error',
                                'error': f"Could not infer a '{intent}' metric from schema of {name}. Available columns: {list(d.columns)}"
                            }
                        totals[name] = d[col].sum()

                    result_df = pd.DataFrame.from_dict(
                        totals, orient='index', columns=[f'total_{intent}']
                    )
                    
                    if any(k in question_lower for k in ['average', 'mean', 'avg']):
                        # If user asked for average across plants
                        avg_val = result_df[f'total_{intent}'].mean()
                        print(f"⚡ GENBI OVERRIDE: Executed direct '{intent}' average across plants.")
                        return {
                            'type': 'scalar',
                            'data': f"Average {intent} across plants: {avg_val:.2f}",
                            'accessed_keys': accessed_keys
                        }

                    print(f"⚡ GENBI OVERRIDE: Executed direct '{intent}' aggregation.")
                    return {
                        'type': 'dataframe',
                        'data': result_df,
                        'rows_returned': len(result_df),
                        'accessed_keys': accessed_keys
                    }

        # 1.6 Dataset Normalization (CRITICAL for Ranking/Complex Queries)
        # LLM writes code based on 'df' (the best-match schema). 
        # If 'dfs' contains other files with different naming (e.g. Merit files vs Optimized),
        # we must normalize their columns to match 'df' so pd.concat() and sorting work.
        if dfs and len(dfs) > 1:
            for intent, keywords in METRIC_INTENTS.items():
                # Check if the intent or any of its keywords appear in the question
                if intent in question_lower or any(kw in question_lower for kw in keywords):
                    # The LLM is likely to use the column name from the 'best match' (df)
                    target_col = infer_best_numeric_column(df, keywords)
                    if target_col:
                        for name, d in dfs.items():
                            inferred = infer_best_numeric_column(d, keywords)
                            if inferred and inferred != target_col:
                                # Rename the column in the tracked DataFrame
                                d.rename(columns={inferred: target_col}, inplace=True)
                                print(f"🔄 Schema Alignment: Normalized {name} column '{inferred}' -> '{target_col}'")

        # 2. Column existence enforcement (CRITICAL GUARDRAIL)
        try:
            # Pattern matches ['column_name'] or ["column_name"]
            # Correctly handles multi-column selections like [['col1', 'col2']]
            requested_cols = (
                re.findall(r"\['([^']+)'\]", code) +
                re.findall(r'\["([^"]+)"\]', code)
            )
            check_dict = dfs if dfs else {'df': df}
            for name, d in check_dict.items():
                for col in requested_cols:
                    if col not in d.columns:
                        return {
                            'type': 'error',
                            'error': f"Column '{col}' not found. Available columns in {name}: {list(d.columns)}",
                            'code': code
                        }
        except Exception:
            pass

        # First try eval (works when model returns a single expression)
        try:
            result = eval(code, safe_globals, local_vars)
        except SyntaxError as se:
            # If eval fails due to syntax, try exec (for multi-line statements)
            try:
                exec(code, safe_globals, local_vars)

                # Prefer explicit 'result' variable if author provided it
                if 'result' in local_vars:
                    result = local_vars['result']
                else:
                    # Pick last non-df/pd variable as best-effort result
                    candidates = [v for k, v in local_vars.items() if k not in ('df', 'pd', 'dfs', 'np', 'stats')]
                    result = candidates[-1] if candidates else None

            except SyntaxError as se2:
                return {
                    'type': 'error',
                    'error': f"SyntaxError in generated code: {se2.msg} (line {se2.lineno})",
                    'traceback': traceback.format_exc(),
                    'code': code
                }
            except Exception as e:
                return {
                    'type': 'error',
                    'error': f"Execution error: {str(e)}",
                    'traceback': traceback.format_exc(),
                    'code': code
                }
        except Exception as e:
            return {
                'type': 'error',
                'error': f"Evaluation error: {str(e)}",
                'traceback': traceback.format_exc(),
                'code': code
            }

        # Retrieve accessed keys
        accessed_keys = list(tracked_dfs.accessed_keys) if tracked_dfs else []

        # Process result based on type
        try:
            if isinstance(result, pd.DataFrame):
                return {
                    'type': 'dataframe',
                    'data': result,
                    'rows_returned': len(result),
                    'accessed_keys': accessed_keys
                }
            elif isinstance(result, pd.Series):
                return {
                    'type': 'series',
                    'data': result,
                    'length': len(result),
                    'accessed_keys': accessed_keys
                }
            elif isinstance(result, (int, float, str)):
                return {
                    'type': 'scalar',
                    'data': result,
                    'accessed_keys': accessed_keys
                }
            elif result is None:
                return {
                    'type': 'error',
                    'error': "No calculable result produced (result is None). Ensure the analysis returns a value, Series, or DataFrame.",
                    'code': code
                }
            else:
                return {
                    'type': 'error',
                    'error': f"No calculable result produced. Result type '{type(result).__name__}' is not recognized for reporting.",
                    'code': code
                }
        except Exception as e:
            return {
                'type': 'error',
                'error': f"Result processing error: {str(e)}",
                'traceback': traceback.format_exc(),
                'code': code
            }
    
    def generate_natural_response(self, question: str, execution_result: Dict, 
                                 file_name: str, sheet_name: str) -> str:
        """Generate natural language response from execution result"""
        
        if execution_result['type'] == 'error':
            return f"I encountered an error: {execution_result['error']}"
        
        # Create context for response generation
        result_summary = self._summarize_result(execution_result)
        
        # Determine if it's a comprehensive summary request
        is_summary_req = any(kw in question.lower() for kw in ['summar', 'insight', 'overview', 'details', 'analyze'])

        # Aggressive persona for the LLM
        system_msg = """You are a Senior BI Lead specializing in executive reporting. 
Your goal is to provide a visually spaced, professional report.

CRITICAL FORMATTING RULES:
1. NEVER bold an entire paragraph. Only bold short metrics, values, and names.
2. ALWAYS use TWO newlines (\n\n) between every heading, list item, and paragraph.
3. Every section MUST start with a '### ' header.
4. If you use lists, put a blank line between each bullet point.
5. NO generic intros. Start with '### Data Summary'.
"""
        
        if is_summary_req:
            prompt = f"""Write a PROFESSIONAL BUSINESS REPORT for the following query: "{question}"

DATASET SUMMARY:
{result_summary}

### STRICT REPORTING STRUCTURE:

### 1. Executive Summary
Provide a clear 1-2 paragraph overview of the global metrics.
(Ensure there is a blank line above and below this heading)

### 2. Key Data Insights
- **Metric/Finding Title**: Detailed explanation of why this matters.
- **Metric/Finding Title**: Another distinct insight.
(Ensure there is a blank line above and below this heading)

### 3. Final Conclusion
Sum up the overall situation and suggest next steps.
(Ensure there is a blank line above and below this heading)

FORMATTING RULES:
- EVERY heading MUST start on a brand new line.
- NEVER put a heading at the end of a paragraph.
- Use DOUBLE NEWLINES between every paragraph and heading.
- Use **bold** ONLY for numbers, dates, and entity names.
- NEVER bold entire sentences.
"""
        else:
            prompt = f"""Provide a crisp, professional answer to: "{question}"
Data Context: {result_summary}

Formatting:
- Use **bold** ONLY for numbers and names.
- Use bullet points if listing multiple facts.
- Use double newlines for separation."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Lower temperature for more consistent formatting
                max_tokens=1200 if is_summary_req else 600
            )
            
            return response.choices[0].message.content.strip()
            
        except:
            return result_summary
    
    def _summarize_result(self, execution_result: Dict) -> str:
        """Create a text summary of execution result for the LLM to process"""
        
        res_type = execution_result['type']
        data = execution_result.get('data')

        if res_type == 'scalar':
            return f"The result of the analysis is the single value: {data}"
        
        elif res_type == 'series':
            # Convert series to a more descriptive format
            return f"The analysis produced the following Series (index and values):\n{data.to_string()}"
        
        elif res_type == 'dataframe':
            # Provide info about the structure and the first few rows for context
            shape = data.shape
            cols = list(data.columns)
            info = f"The analysis produced a DataFrame with {shape[0]} rows and {shape[1]} columns.\n"
            info += f"Columns: {', '.join(cols)}\n\n"
            
            if shape[0] > 0:
                info += "Sample Data (Top 10 rows):\n"
                info += data.head(10).to_string()
                
                # If there are many rows, also give some stats
                if shape[0] > 10:
                    info += f"\n\n(Total {shape[0]} rows)"
                    # Optionally add numeric description if relevant
                    numeric_cols = data.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        info += "\n\nNumeric Columns Summary:\n"
                        info += data[numeric_cols].describe().to_string()
            else:
                info += "The resulting DataFrame is empty."
            
            return info
        
        else:
            return f"Data result: {str(data)}"

    def _fix_deprecated_code(self, code: str) -> str:
        """Attempt to fix deprecated pandas patterns like .append() -> pd.concat()"""
        import re
        
        # We use re.DOTALL to match across newlines if the expression is split.
        # We also allow for optional whitespace/newline before .append
        
        def concat_replacer(match):
            obj = match.group(1).strip()
            args = match.group(2).strip()
            
            # Basic handling for keyword arguments like ignore_index
            if ',' in args:
                # Split at first comma to separate 'other' from kwargs
                # We try to be careful about commas inside parentheses (like pd.Series({...}))
                # but for a best-effort fixer, a simple split on the first comma is a start.
                # A better way is to find the first comma at the top level of the args.
                
                # Simple heuristic: if the first part has unbalanced parens, it's not the split point.
                comma_idx = -1
                paren_balance = 0
                for i, char in enumerate(args):
                    if char == '(': paren_balance += 1
                    elif char == ')': paren_balance -= 1
                    elif char == ',' and paren_balance == 0:
                        comma_idx = i
                        break
                
                if comma_idx != -1:
                    other = args[:comma_idx].strip()
                    kwargs = args[comma_idx+1:].strip()
                    return f"pd.concat([{obj}, {other}], {kwargs})"
            
            return f"pd.concat([{obj}, {args}])"

        # Regex pattern: (expression)\s*\.append\((argument)\)
        # We use [^]* or similar to match everything including newlines for the object.
        # But let's stick to a robust pattern.
        pattern = r"(.*?)\s*\.append\((.*)\)"
        fixed = re.sub(pattern, concat_replacer, code, flags=re.DOTALL)
        
        return fixed
