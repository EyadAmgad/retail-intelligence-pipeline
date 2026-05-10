import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

from pathlib import Path

def preprocess_data():
    # Define the path to the CSV files using pathlib
    raw_data_dir = Path(__file__).resolve().parent.parent / 'data' / 'raw'
    
    csv_customer = raw_data_dir / 'olist_customers_dataset.csv'
    csv_geolocation = raw_data_dir / 'olist_geolocation_dataset.csv'
    csv_order_items = raw_data_dir / 'olist_order_items_dataset.csv'
    csv_order_payments = raw_data_dir / 'olist_order_payments_dataset.csv'
    csv_order_reviews = raw_data_dir / 'olist_order_reviews_dataset.csv'
    csv_products = raw_data_dir / 'olist_products_dataset.csv'
    csv_sellers = raw_data_dir / 'olist_sellers_dataset.csv'
    csv_product_categories = raw_data_dir / 'product_category_name_translation.csv'

    # Load the CSV files into DataFrames
    df_customer = pd.read_csv(csv_customer)
    df_geolocation = pd.read_csv(csv_geolocation)
    df_order_items = pd.read_csv(csv_order_items)
    df_order_payments = pd.read_csv(csv_order_payments)
    df_order_reviews = pd.read_csv(csv_order_reviews)
    df_products = pd.read_csv(csv_products)
    df_sellers = pd.read_csv(csv_sellers)
    df_product_categories = pd.read_csv(csv_product_categories)

    df_order_items.drop(columns=["shipping_limit_date"], inplace=True)
    df_order_payments.drop(columns=["payment_sequential" , "payment_installments"], inplace=True)
    df_order_reviews.drop(columns=["review_id", "review_answer_timestamp"], inplace=True)
    df_products.drop(columns=["product_name_lenght", "product_description_lenght", "product_photos_qty", "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"], inplace=True)
    df_sellers.drop(columns=["seller_zip_code_prefix"], inplace=True)

    
    print(df_customer.columns)
    print(df_geolocation.columns)
    print(df_order_items.columns)
    print(df_order_payments.columns)
    print(df_order_reviews.columns)
    print(df_products.columns)
    print(df_sellers.columns)
    print(df_product_categories.columns)
# Display the column names of each DataFrame 
# Construct path relative to the script's location
    

    

preprocess_data()

