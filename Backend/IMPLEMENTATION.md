# GenBI Backend: AI-Powered Excel Analytics

## Project Overview
GenBI is a sophisticated business intelligence platform that allows users to interact with Excel data using natural language. The backend is built with FastAPI and leverages Large Language Models (LLMs) via Groq to translate user queries into executable Python/Pandas code, providing real-time data analysis, visualization, and insights.

---

## Core Architecture & Implementations

### 1. Semantic Data Indexing (`ExcelSchemaIndexer`)
- **Technology**: FAISS (Facebook AI Similarity Search) & Sentence Transformers.
- **Implementation**: 
    - Automatically scans Excel files and extracts metadata, column types, and sample data.
    - Converts sheet schemas into high-dimensional embeddings using `all-MiniLM-L6-v2`.
    - Enables extremely fast retrieval of relevant sheets based on the semantic meaning of user questions.

### 2. Intelligent Agent (`GroqPandasAgent`)
- **Technology**: Groq LPU™ Inference Engine.
- **Implementation**:
    - **Code Generation**: Transforms natural language into precise Pandas code.
    - **Safe Execution**: Executes generated code in a controlled environment with pre-loaded libraries (`pandas`, `numpy`, `scipy.stats`).
    - **Context Awareness**: Maintains conversation history to handle follow-up questions predictably.
    - **Statistical Power**: Pre-configures `scipy.stats` to support advanced analytical queries like correlation and significance testing without requiring risky `import` statements.

### 3. Dynamic Visualization (`ChartGenerator`)
- **Technology**: Plotly.
- **Implementation**:
    - Automatically detects the best chart type (Bar, Line, Pie, etc.) based on the data structure and user intent.
    - Generates interactive Plotly JSON objects that are rendered seamlessly in the frontend.

### 4. Real-time Synchronization (`FileWatcher`)
- **Technology**: Watchdog.
- **Implementation**:
    - Monitors the `excel_data/` directory for any file changes (additions, deletions, or modifications).
    - Triggers automatic re-indexing to ensure the chatbot always has access to the most current data without requiring a restart.

### 5. Secure Session Management (`SessionManager`)
- **Implementation**:
    - Manages isolated state for multiple users, tracking conversation history and last-accessed data context to provide a personalized experience.

---

## Tech Stack
- **API Framework**: FastAPI
- **Data Processing**: Pandas, NumPy
- **Statistics**: SciPy
- **AI/ML**: Groq, FAISS, Sentence-Transformers
- **Visualization**: Plotly
- **Environment**: Python 3.11+

---

## Implementation Rules (for Developers)
- **Zero-Import Policy**: The LLM agent is forbidden from writing `import` statements. All necessary tools (`df`, `pd`, `np`, `stats`) are injected into the execution context.
- **Sanitization**: All output data is strictly sanitized to ensure JSON compatibility, converting complex NumPy types to standard Python primitives before streaming.
- **Streaming**: Responses are streamed via `Server-Sent Events (SSE)` to provide a snappy, real-time "typing" experience in the UI.
