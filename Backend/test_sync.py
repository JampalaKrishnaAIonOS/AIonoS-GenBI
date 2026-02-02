
import os
import traceback
from dotenv import load_dotenv
from services.excel_to_sql_sync import ExcelToSQLSync

load_dotenv()
excel_folder = os.getenv('EXCEL_FOLDER_PATH')
db_url = os.getenv('DATABASE_URL')

print(f"Excel folder: {excel_folder}")
print(f"DB URL: {db_url}")

try:
    sync = ExcelToSQLSync(excel_folder, db_url)
    results = sync.sync_all_excel_files()
    print(f"Synced {len(results['success'])} tables")
    print(f"Failed: {len(results['failed'])} tables")
    print("Tables List:", sync.list_all_tables())
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    traceback.print_exc()
