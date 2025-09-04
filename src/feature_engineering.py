import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path

# --- Configuration: Define file paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "creditcard.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "featured_transactions.csv"
SQL_QUERY_PATH = BASE_DIR / "sql" / "feature_engineering.sql"
DB_PATH = BASE_DIR / "transactions.db"

def engineer_features():
    """
    Orchestrates the entire feature engineering process.
    """
    print("Starting feature engineering process...")

    # --- 1. Load Raw Data ---
    try:
        df = pd.read_csv(RAW_DATA_PATH)
        print(f" Raw data loaded successfully from {RAW_DATA_PATH}")
    except FileNotFoundError:
        print(f" Error: Raw data file not found at {RAW_DATA_PATH}.")
        print("Please download the dataset from Kaggle and place it in the 'data/raw/' directory.")
        return

    # --- 2. Simulate user_id ---
    num_users = 1000
    df['user_id'] = np.random.randint(1, num_users + 1, df.shape[0])
    print(f"Simulated 'user_id' for {num_users} hypothetical users.")

    # --- 3. Create and Populate SQLite Database ---
    conn = None  # Initialize connection to None
    try:
        conn = sqlite3.connect(DB_PATH)
        # Use the dataframe's index as a primary key named 'index' in the SQL table.
        df.to_sql('transactions', conn, if_exists='replace', index=True, index_label='index')
        print(f" SQLite database created and populated at {DB_PATH}")
    except Exception as e:
        print(f" Error during database operation: {e}")
        return
    finally:
        if conn:
            conn.close()

    # --- 4. Execute Feature Engineering SQL ---
    conn = None
    try:
        with open(SQL_QUERY_PATH, 'r') as f:
            sql_query = f.read()
        
        conn = sqlite3.connect(DB_PATH)
        print("Executing SQL query for feature generation...")
        featured_df = pd.read_sql_query(sql_query, conn)
        print(" Feature engineering complete.")

    except FileNotFoundError:
        print(f" Error: SQL query file not found at {SQL_QUERY_PATH}.")
        return
    except Exception as e:
        print(f" Error executing SQL query: {e}")
        return
    finally:
        if conn:
            conn.close()
            # Clean up the database file after use
            DB_PATH.unlink()
            print(" Cleaned up database file.")


    # --- 5. Clean and Save Processed Data ---
    # The SQL query might produce NaNs for the first transaction of a user. Filling with 0s for consistency. 
    featured_df.fillna(0, inplace=True)

    # Ensure the output directory exists
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    featured_df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f" Processed data with new features saved to {PROCESSED_DATA_PATH}")

if __name__ == '__main__':
    engineer_features()