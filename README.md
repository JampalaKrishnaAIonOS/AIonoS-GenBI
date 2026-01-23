# GenBI - Conversational Business Intelligence

Welcome to **GenBI**, a premium analytics platform built by **AIonOS**. GenBI allows you to chat with your Excel spreadsheets as if you were talking to a Data Scientist.

---

## Project Structure

### 📁 [Backend](./Backend)
Python-based FastAPI server handling data processing, AI reasoning, and indexing.
- **AI Agent**: Uses Groq to generate dynamic Pandas code.
- **Search**: FAISS-powered semantic search to find the right data sheets.
- **Real-time**: Automatic indexing of local Excel files.

### 📁 [frontend](./frontend)
React-based web application with a premium, Indigo Airways-inspired aesthetic.
- **Auth**: Secure login with AIonOS branding.
- **UI**: Clean 50/50 split design and compact chat interface.
- **Visuals**: Interactive charts and professional Markdown reports.

---

## Getting Started

### 1. Prerequisites
- Python 3.11+
- Node.js 18+
- Groq API Key

### 2. Backend Setup
1. Navigate to directory: `cd Backend`
2. Install dependencies: `pip install -r requirements.txt`
3. Configure environment: Create a `.env` file with:
   ```env
   GROQ_API_KEY=your_key_here
   EXCEL_FOLDER_PATH=./excel_data
   FAISS_INDEX_PATH=./faiss_index
   ```
4. Place your Excel files in `Backend/excel_data/`
5. Run server: `python main.py`

### 3. Frontend Setup
1. Navigate to directory: `cd frontend`
2. Install dependencies: `npm install`
3. Run app: `npm start`

---

## Default Credentials
- **Email**: `Admin@aionos.ai`
- **Password**: `Admin@123`

---
Copyright © 2026 AIonOS
