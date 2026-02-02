
import os
from dotenv import load_dotenv
from services.sql_agent import SQLAgent

load_dotenv()
db_url = os.getenv('DATABASE_URL')
groq_api_key = os.getenv('GROQ_API_KEY')
model_name = os.getenv('MODEL_NAME')

print(f"Testing SQL Agent with DB: {db_url}")
agent = SQLAgent(db_url, groq_api_key, model_name)

# Use a query that definitely requires a SQL query
question = "List the distinct plant names in the coal energy cost table"
print(f"Question: {question}")

try:
    # We can't easily access the internal agent executor's response here 
    # unless our query method returns it. Our query method returns a simplified dict.
    # Let's modify sql_agent.py temporarily to return the full result for debugging.
    result = agent.query(question)
    print("\n--- Result ---")
    print(f"Answer: {result['answer']}")
    print(f"SQL Query: {result['sql_query']}")
    print(f"Result Type: {result['type']}")
    if result['data'] is not None:
        print(f"Data Sample:\n{result['data'].head()}")
except Exception as e:
    print(f"Error: {e}")
