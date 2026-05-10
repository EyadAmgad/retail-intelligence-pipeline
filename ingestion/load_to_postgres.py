import os
import pandas as pd
from sqlalchemy import create_engine, text
from db_config import DATABASE_URI

def get_db_engine():
    """Create and return a SQLAlchemy engine."""
    return create_engine(DATABASE_URI)

def load_processed_data(processed_dir, engine):
    """Iterate over processed Parquet files and load them into PostgreSQL."""
    if not os.path.exists(processed_dir):
        print(f"Warning: Directory {processed_dir} does not exist or is empty.")
        return

    # Look for .parquet files in the directory
    parquet_files = [f for f in os.listdir(processed_dir) if f.endswith(".parquet")]
    if not parquet_files:
        print("No Parquet files found in the processed data directory.")
        return

    # Schema is typically retail based on your init.sql
    schema_name = "retail"

    for filename in parquet_files:
        file_path = os.path.join(processed_dir, filename)
        # Use the filename (without extension) as the table name
        table_name = os.path.splitext(filename)[0]
        
        print(f"Loading {file_path} into table '{schema_name}.{table_name}'...")
        try:
            # Check row count and clear if necessary
            with engine.connect() as conn:
                try:
                    count = conn.execute(text(f"SELECT COUNT(*) FROM {schema_name}.{table_name}")).scalar()
                    if count > 0:
                        print(f"Table '{schema_name}.{table_name}' has {count} rows. Clearing table to 0 rows...")
                        conn.execute(text(f"TRUNCATE TABLE {schema_name}.{table_name} CASCADE"))
                        conn.commit()
                    else:
                        print(f"Table '{schema_name}.{table_name}' already has 0 rows.")
                except Exception:
                    # Table might not exist yet; rollback and proceed
                    conn.rollback()

            # Read the parquet file into a pandas DataFrame
            df = pd.read_parquet(file_path)
            
            # Load into PostgreSQL using append to preserve init.sql constraints
            df.to_sql(table_name, engine, schema=schema_name, if_exists='append', index=False)
            print(f"Successfully loaded {len(df)} rows into '{schema_name}.{table_name}'.")
        except Exception as e:
            print(f"Error loading '{table_name}': {e}")

if __name__ == "__main__":
    # Base path assuming this script is in `ingestion/`
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
    
    print(f"Connecting to database at {DATABASE_URI.split('@')[1]}...")
    engine = get_db_engine()
    
    print("Starting data ingestion process...")
    load_processed_data(PROCESSED_DATA_DIR, engine)
    print("Ingestion complete.")
