# GenBI Backend System

This is the core engine of the GenBI platform, responsible for transforming natural language questions into data-driven insights from Excel files.

## 🌟 Key Features

- **Natural Language to Pandas**: Advanced NLP and reasoning using Groq's high-speed inference.
- **Semantic Data Search**: Vector-based search (FAISS) to instantly find the correct sheet for any question.
- **Interactive Visualizations**: Automatic Plotly chart generation based on analysis results.
- **Auto-Indexing**: Real-time detection of data changes using a file watcher.
- **Advanced Statistics**: Built-in support for correlation, significance testing, and trend analysis.

## 🏗️ Technical Architecture

The backend is structured as a modular FastAPI service:

- `main.py`: The API gateway and streaming coordinator.
- `services/groq_agent.py`: The brain that generates and executes analysis code.
- `services/excel_indexer.py`: Handles semantic vectorization and sheet retrieval.
- `services/chart_generator.py`: Logic for intelligent chart type detection and Plotly JSON generation.
- `services/file_watcher.py`: Background monitor for the data folder.
- `models/`: Pydantic schemas for structured communication with the frontend.

## 🚀 Setup & Execution

### 1. Requirements
Ensure you have Python 3.11+ installed.

### 2. Installation
Install the verified dependencies:
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in this directory:
```env
GROQ_API_KEY=your_groq_api_key
EXCEL_FOLDER_PATH=./excel_data
FAISS_INDEX_PATH=./faiss_index
MODEL_NAME=moonshotai/kimi-k2-instruct
```

### 4. Running the Server
Start the backend with:
```bash
python main.py
```
The server will start at `http://localhost:8000`.

## 📁 Data Management
Place any `.xlsx` files you want to analyze into the `excel_data/` folder. The system will detect them and start indexing automatically.

## 📘 More Information
For deep technical details on how the semantic search or the AI agent works, please refer to [IMPLEMENTATION.md](./IMPLEMENTATION.md).

---
© 2026 AIonOS GenBI Team
