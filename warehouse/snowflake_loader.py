import os
import pandas as pd
from sqlalchemy import create_engine
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# PostgreSQL connection details
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_HOST = os.getenv("POSTGRES_HOST")
DB_PORT = os.getenv("POSTGRES_PORT")
DB_NAME = os.getenv("POSTGRES_DB")

# Snowflake connection details
SF_USER = os.getenv("SF_USER")
SF_PASSWORD = os.getenv("SF_PASSWORD")
SF_ACCOUNT = os.getenv("SF_ACCOUNT")
SF_WAREHOUSE = os.getenv("SF_WAREHOUSE")
SF_DATABASE = os.getenv("SF_DATABASE")
SF_SCHEMA = os.getenv("SF_SCHEMA")
SF_ROLE = os.getenv("SF_ROLE", "ACCOUNTADMIN")

def get_postgres_engine():
    uri = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(uri)

def get_snowflake_conn():
    return snowflake.connector.connect(
        user=SF_USER,
        password=SF_PASSWORD,
        account=SF_ACCOUNT,
        warehouse=SF_WAREHOUSE,
        role=SF_ROLE
    )

def load_table_to_snowflake(table_name, pg_engine, sf_conn):
    print(f"Reading '{table_name}' from PostgreSQL...")
    # Read the table from postgres
    df = pd.read_sql_table(table_name, con=pg_engine, schema="retail")
    
    # Uppercase column names as Snowflake expects uppercase by default
    df.columns = [col.upper() for col in df.columns]
    
    # Convert datetime columns to avoid timezone issues or unsupported types
    for col in df.select_dtypes(include=['datetime64[ns, UTC]', 'datetime64[ns]']).columns:
        df[col] = df[col].dt.tz_localize(None)
        
    # Explicitly convert full_date to python date objects so Snowflake casts it correctly to DATE
    if 'FULL_DATE' in df.columns:
        df['FULL_DATE'] = pd.to_datetime(df['FULL_DATE']).dt.date
    
    print(f"Writing '{table_name.upper()}' to Snowflake...")
    # Write to snowflake
    success, nchunks, nrows, _ = write_pandas(
        conn=sf_conn,
        df=df,
        table_name=table_name.upper(),
        auto_create_table=False,
        quote_identifiers=False,
        overwrite=False
    )
    
    if success:
        print(f"Successfully loaded {nrows} rows to {table_name.upper()} in Snowflake.")
    else:
        print(f"Failed to load {table_name.upper()} to Snowflake.")

if __name__ == "__main__":
    tables_to_load = [
        "dim_date",
        "dim_customers",
        "dim_products",
        "dim_sellers",
        "fact_orders"
    ]
    
    print("Connecting to PostgreSQL and Snowflake...")
    pg_engine = get_postgres_engine()
    sf_conn = get_snowflake_conn()
    
    try:
        if SF_DATABASE:
            sf_conn.cursor().execute(f"CREATE DATABASE IF NOT EXISTS {SF_DATABASE}")
            sf_conn.cursor().execute(f"USE DATABASE {SF_DATABASE}")
            
        schema_name = SF_SCHEMA.split('.')[-1] if SF_SCHEMA and '.' in SF_SCHEMA else SF_SCHEMA
        if schema_name:
            sf_conn.cursor().execute(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
            sf_conn.cursor().execute(f"USE SCHEMA {schema_name}")

        # Execute DDL first to create tables in the correct order
        print("Executing DDL to create/replace tables...")
        for table in tables_to_load:
            ddl_file = os.path.join(os.path.dirname(__file__), "ddl", f"{table}.sql")
            if os.path.exists(ddl_file):
                with open(ddl_file, 'r', encoding='utf-8') as f:
                    ddl_sql = f.read()
                print(f"Creating table '{table.upper()}' from {ddl_file} ...")
                sf_conn.cursor().execute(ddl_sql)
            else:
                print(f"Warning: DDL file not found for {table.upper()} at {ddl_file}")
        
        for table in tables_to_load:
            try:
                load_table_to_snowflake(table, pg_engine, sf_conn)
            except Exception as e:
                print(f"Error processing table {table}: {str(e)}")
                
    finally:
        sf_conn.close()
        print("Snowflake connection closed.")
