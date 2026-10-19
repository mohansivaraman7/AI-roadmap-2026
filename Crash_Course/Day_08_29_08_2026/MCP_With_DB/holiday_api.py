"""
STEP 2 of 3: A simple local API in front of holidays.db
==========================================================
This is a genuine REST API -- not MCP yet, just a plain HTTP service,
the same kind any web/mobile app would call. This is the layer that
MCP will wrap in step 3.

SETUP:
    pip install fastapi uvicorn --break-system-packages

RUN (make sure holidays.db already exists -- run create_holiday_db.py first):
    uvicorn holiday_api:app --reload --port 8000

TEST IT DIRECTLY IN YOUR BROWSER (no MCP involved at all here):
    http://127.0.0.1:8000/holidays?date=2026-01-26&country_code=IN
    http://127.0.0.1:8000/docs   <- FastAPI's auto-generated interactive API docs
"""

import sqlite3
from fastapi import FastAPI, HTTPException

DB_FILE = "holidays.db"

app = FastAPI(title="Holiday API")


def get_connection():
    # check_same_thread=False because FastAPI can call this from different
    # threads -- fine for this simple read-mostly demo database.
    return sqlite3.connect(DB_FILE, check_same_thread=False)


@app.get("/holidays")
def get_holiday(date: str, country_code: str = "IN"):
    """
    Check whether a given date is a public holiday.
    Query params:
      date          e.g. 2026-01-26 (YYYY-MM-DD)
      country_code  e.g. IN, US, GB
    """
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        "SELECT name, local_name FROM holidays WHERE date = ? AND country_code = ?",
        (date, country_code.upper())
    )
    row = cursor.fetchone()
    connection.close()

    if row is None:
        return {
            "date": date,
            "country_code": country_code.upper(),
            "is_holiday": False,
            "name": None
        }

    name, local_name = row
    return {
        "date": date,
        "country_code": country_code.upper(),
        "is_holiday": True,
        "name": name,
        "local_name": local_name
    }


@app.get("/holidays/{country_code}/{year}")
def list_holidays(country_code: str, year: int):
    """List all holidays for a given country and year."""
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        "SELECT date, name, local_name FROM holidays WHERE country_code = ? AND date LIKE ?",
        (country_code.upper(), f"{year}-%")
    )
    rows = cursor.fetchall()
    connection.close()

    if not rows:
        raise HTTPException(status_code=404, detail="No holidays found for that country/year")

    return [{"date": d, "name": n, "local_name": ln} for d, n, ln in rows]
