"""
STEP 1 of 3: Build a local SQLite database of public holidays.
==================================================================
This replaces the unreliable external date.nager.at API with your
OWN local, fast, always-available data source.

RUN THIS ONCE:
    python create_holiday_db.py

It creates holidays.db in the current folder. Re-running this script
is safe -- it drops and recreates the table each time, so you can
freely edit the HOLIDAYS list below and re-run to refresh the data.
"""

import sqlite3

DB_FILE = "holidays.db"

# Sample data -- expand this with your real company/country calendar
# as needed. Format: (date, country_code, name, local_name)
HOLIDAYS = [
    ("2026-01-01", "IN", "New Year's Day", "New Year's Day"),
    ("2026-01-26", "IN", "Republic Day", "Republic Day"),
    ("2026-03-04", "IN", "Holi", "Holi"),
    ("2026-08-15", "IN", "Independence Day", "Independence Day"),
    ("2026-10-02", "IN", "Gandhi Jayanti", "Gandhi Jayanti"),
    ("2026-10-20", "IN", "Diwali", "Deepavali"),
    ("2026-12-25", "IN", "Christmas Day", "Christmas Day"),

    ("2026-01-01", "US", "New Year's Day", "New Year's Day"),
    ("2026-01-19", "US", "Martin Luther King Jr. Day", "Martin Luther King Jr. Day"),
    ("2026-07-04", "US", "Independence Day", "Independence Day"),
    ("2026-11-26", "US", "Thanksgiving Day", "Thanksgiving Day"),
    ("2026-12-25", "US", "Christmas Day", "Christmas Day"),

    ("2026-01-01", "GB", "New Year's Day", "New Year's Day"),
    ("2026-04-03", "GB", "Good Friday", "Good Friday"),
    ("2026-05-25", "GB", "Spring Bank Holiday", "Spring Bank Holiday"),
    ("2026-12-25", "GB", "Christmas Day", "Christmas Day"),
    ("2026-12-26", "GB", "Boxing Day", "Boxing Day"),
]

connection = sqlite3.connect(DB_FILE)
cursor = connection.cursor()

cursor.execute("DROP TABLE IF EXISTS holidays")
cursor.execute("""
    CREATE TABLE holidays (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        country_code TEXT NOT NULL,
        name TEXT NOT NULL,
        local_name TEXT NOT NULL
    )
""")

cursor.executemany(
    "INSERT INTO holidays (date, country_code, name, local_name) VALUES (?, ?, ?, ?)",
    HOLIDAYS
)

connection.commit()

print(f"Created {DB_FILE} with {cursor.rowcount if cursor.rowcount != -1 else len(HOLIDAYS)} rows inserted.")
print("Sample check:")
cursor.execute("SELECT * FROM holidays WHERE country_code = 'IN' LIMIT 3")
for row in cursor.fetchall():
    print(" ", row)

connection.close()
