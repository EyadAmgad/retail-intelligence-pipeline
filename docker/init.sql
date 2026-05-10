-- init.sql
-- Initialize schemas and create tables for processed data

CREATE SCHEMA IF NOT EXISTS retail;

-- 1. Create Dimension Tables

CREATE TABLE IF NOT EXISTS retail.dim_customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    customer_unique_id VARCHAR(50) NOT NULL,
    customer_zip_code_prefix VARCHAR(50),
    customer_city VARCHAR(255),
    customer_state VARCHAR(50),
    customer_state_valid BOOLEAN,
    customer_lat FLOAT,
    customer_lng FLOAT,
    has_geo BOOLEAN
);

CREATE TABLE IF NOT EXISTS retail.dim_products (
    product_id VARCHAR(50) PRIMARY KEY,
    product_category_name VARCHAR(255),
    product_category_name_english VARCHAR(255),
    product_name_length FLOAT,
    product_description_length FLOAT,
    product_photos_qty FLOAT,
    product_weight_g FLOAT,
    product_length_cm FLOAT,
    product_height_cm FLOAT,
    product_width_cm FLOAT,
    product_volume_cm3 FLOAT,
    size_category VARCHAR(50),
    has_missing_dimensions BOOLEAN
);

CREATE TABLE IF NOT EXISTS retail.dim_sellers (
    seller_id VARCHAR(50) PRIMARY KEY,
    seller_zip_code_prefix VARCHAR(50),
    seller_city VARCHAR(255),
    seller_state VARCHAR(50),
    seller_lat FLOAT,
    seller_lng FLOAT,
    has_geo BOOLEAN
);

CREATE TABLE IF NOT EXISTS retail.dim_date (
    date_id INT PRIMARY KEY,
    full_date TIMESTAMP,
    year INT,
    month INT,
    quarter INT,
    day_of_week INT,
    is_weekend BOOLEAN,
    year_month VARCHAR(20),
    month_name VARCHAR(50),
    day_of_month INT,
    week_of_year INT,
    year_quarter VARCHAR(20),
    day_name VARCHAR(50)
);

-- 2. Create Fact Tables

CREATE TABLE IF NOT EXISTS retail.fact_orders (
    order_id VARCHAR(50),
    order_item_id INT,
    customer_id VARCHAR(50),
    product_id VARCHAR(50),
    seller_id VARCHAR(50),
    date_id INT,
    order_status VARCHAR(50),
    order_purchase_timestamp TIMESTAMP,
    order_approved_at TIMESTAMP,
    order_delivered_carrier_date TIMESTAMP,
    order_delivered_customer_date TIMESTAMP,
    order_estimated_delivery_date TIMESTAMP,
    shipping_limit_date TIMESTAMP,
    price FLOAT,
    freight_value FLOAT,
    review_score FLOAT,
    order_dayofweek INT,
    order_dayofweek_name VARCHAR(50),
    n_payment_rows FLOAT,
    is_split_payment BOOLEAN,
    n_items INT,
    delivery_days_actual FLOAT,
    total_payment_value FLOAT,
    n_unique_sellers INT,
    is_weekend BOOLEAN,
    n_payment_types FLOAT,
    order_hour INT,
    has_comment_message BOOLEAN,
    order_year INT,
    primary_payment_type VARCHAR(50),
    order_total_freight FLOAT,
    max_installments FLOAT,
    is_price_outlier BOOLEAN,
    n_unique_products INT,
    order_total_value FLOAT,
    installment_bin VARCHAR(50),
    total_item_value FLOAT,
    sentiment_bucket VARCHAR(50),
    order_quarter INT,
    is_late BOOLEAN,
    order_total_price FLOAT,
    review_response_time_hours FLOAT,
    delay_days FLOAT,
    approval_time_hours FLOAT,
    delivery_days_estimated FLOAT,
    has_price_outlier BOOLEAN,
    order_month INT,
    PRIMARY KEY (order_id, order_item_id)
);
