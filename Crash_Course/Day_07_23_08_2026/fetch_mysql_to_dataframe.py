"""
Fetch chocolate sales data from MySQL into a pandas DataFrame
ready to feed into an ML model.

Schema assumed (matches your Excel file):
  Shipments(ShipmentID, SPID, PID, GID, Shipdate, Amount, Boxes, Order_Status)
  Dimension_Data(Product, Category, Cost_per_box, PID,
                  Geo, Region, GID,
                  Sales_person, Team, Picture, SPID)

Install dependencies first:
    pip install mysql-connector-python pandas sqlalchemy --break-system-packages
"""

import pandas as pd
from sqlalchemy import create_engine

# ---------------------------------------------------------
# 1. DATABASE CONNECTION SETTINGS  -- edit these for your setup
# ---------------------------------------------------------
DB_USER = "root"
DB_PASSWORD = "your_password"
DB_HOST = "localhost"
DB_PORT = 3306
DB_NAME = "chocolate_sales"

# SQLAlchemy connection string for MySQL
connection_string = f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(connection_string)


# ---------------------------------------------------------
# 2. QUERY -- joins Shipments with Dimension_Data
#    so the DataFrame has product/region/sales-person info,
#    not just raw IDs. This is usually what a model wants.
# ---------------------------------------------------------
QUERY = """
    SELECT
        s.ShipmentID,
        s.Shipdate,
        s.Amount,
        s.Boxes,
        s.Order_Status,
        d.Product,
        d.Category,
        d.Cost_per_box,
        d.Geo,
        d.Region,
        d.Sales_person,
        d.Team
    FROM Shipments s
    JOIN Dimension_Data d
        ON s.PID = d.PID
       AND s.GID = d.GID
       AND s.SPID = d.SPID
"""


def fetch_dataframe() -> pd.DataFrame:
    """Runs the query and returns a pandas DataFrame."""
    df = pd.read_sql(QUERY, con=engine)
    return df


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning + feature engineering commonly needed
    before feeding data into a model.
    """
    # Ensure correct dtypes
    df["Shipdate"] = pd.to_datetime(df["Shipdate"])
    df["Amount"] = df["Amount"].astype(float)
    df["Boxes"] = df["Boxes"].astype(int)
    df["Cost_per_box"] = df["Cost_per_box"].astype(float)

    # Drop rows with missing critical values
    df = df.dropna(subset=["Amount", "Boxes", "Product"])

    # Feature engineering examples
    df["Total_Cost"] = df["Cost_per_box"] * df["Boxes"]
    df["Profit"] = df["Amount"] - df["Total_Cost"]
    df["Profit_Margin"] = df["Profit"] / df["Amount"]
    df["Ship_Month"] = df["Shipdate"].dt.month
    df["Ship_Year"] = df["Shipdate"].dt.year
    df["Ship_Weekday"] = df["Shipdate"].dt.day_name()

    # One-hot encode categorical columns commonly used as model input
    df_encoded = pd.get_dummies(
        df,
        columns=["Category", "Region", "Order_Status"],
        drop_first=True
    )

    return df_encoded


def main():
    print("Connecting to MySQL and fetching data...")
    raw_df = fetch_dataframe()
    print(f"Fetched {len(raw_df)} rows.")

    print("Cleaning and engineering features...")
    model_df = clean_and_engineer(raw_df)

    print("Final DataFrame shape:", model_df.shape)
    print(model_df.head())

    # Save to CSV as a checkpoint (optional)
    model_df.to_csv("chocolate_sales_model_input.csv", index=False)
    print("Saved to chocolate_sales_model_input.csv")

    return model_df


if __name__ == "__main__":
    model_df = main()

    # ---------------------------------------------------------
    # Example: this DataFrame is now ready for a model, e.g.
    # ---------------------------------------------------------
    # from sklearn.model_selection import train_test_split
    #
    # X = model_df.drop(columns=["Amount", "ShipmentID", "Shipdate",
    #                             "Product", "Sales_person", "Team", "Geo"])
    # y = model_df["Amount"]
    #
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=42
    # )
