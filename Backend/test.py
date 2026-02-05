import sqlite3

db_path = r'C:\Users\J RISHI KRISHNA\Downloads\AIonoS-GenBI\Backend\genbi.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT * FROM merged_rack_optimization_oct25_results_3plants_sheet1 ORDER BY siding_wise_merit_order;")
rows = cursor.fetchall()

for row in rows[:5]:  # Print first 5 rows as a sample
    print(row)


conn.close()
