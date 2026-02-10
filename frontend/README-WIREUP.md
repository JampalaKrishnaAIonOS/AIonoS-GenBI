# Frontend ↔ Backend Wireup

Quick steps to run frontend + backend and debug streaming events.

1) Backend

- From `Backend/` create a virtualenv and install requirements:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

2) Frontend

- From `frontend/` install and start:

```bash
cd frontend
npm install
# Copy .env.example to .env or set REACT_APP_API_URL
# Windows PowerShell:
copy .env.example .env
npm start
```

3) Test flow

- Open the frontend in the browser (usually http://localhost:3000).
- Ask a data question like: `Show me the top 10 companies by revenue and plot it`.

4) If you don't see a table/chart:

- Toggle the raw stream debugger (use the small toggle in header if present) or open browser DevTools console.
- Look for events of type `table` and `chart`. Example event shapes:
  - `{ type: 'table', content: { columns: [...], rows: [...] } }`
  - `{ type: 'chart', content: { type: 'chart', data: {...} } }`

5) Debug checklist

- Backend returned `table` events? If yes, frontend should display table. If not, check backend logs for SQL results and that `session_manager.set_last_dataframe` is called.
- Backend returned `chart` event? Frontend expects a Plotly figure under `chart.data.data` or a `chart.data` array.
- If streaming stops early, check network tab for `text/event-stream` response and any HTTP error.

6) Next actions I can take for you

- Enable an always-on developer panel showing raw incoming events and timeline (I added a simple raw-events viewer already).
- Force the backend to always send a compact Markdown `answer` that follows the required format (bold headings, short insights).
- Add unit tests / a small harness to simulate stream events for frontend testing.

If you want, I can run through the checklist with you or implement the next debug step automatically.
