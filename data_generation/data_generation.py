import psycopg2
import random
from faker import Faker
from datetime import datetime, timedelta

# Initialize Faker
fake = Faker()


def create_tables(conn):
    """Creates tables for storing synthetic data with more complexity."""
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id SERIAL PRIMARY KEY,
        name VARCHAR(100),
        email VARCHAR(100),
        join_date TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS products (
        product_id SERIAL PRIMARY KEY,
        name VARCHAR(100),
        price FLOAT
    );

    CREATE TABLE IF NOT EXISTS transactions (
        transaction_id SERIAL PRIMARY KEY,
        user_id INTEGER REFERENCES users(user_id),
        product_id INTEGER REFERENCES products(product_id),
        quantity INTEGER,
        transaction_date TIMESTAMP
    );
    """)
    conn.commit()
    cursor.close()


def generate_users(conn, num_users=100):
    """Generates synthetic user data."""
    cursor = conn.cursor()
    for _ in range(num_users):
        name = fake.name()
        email = fake.email()
        join_date = fake.date_time_between(start_date="-2y", end_date="now")

        cursor.execute("""
        INSERT INTO users (name, email, join_date)
        VALUES (%s, %s, %s)
        """, (name, email, join_date))
    conn.commit()
    cursor.close()


def generate_products(conn, num_products=50):
    """Generates synthetic product data."""
    cursor = conn.cursor()
    for _ in range(num_products):
        name = fake.word(ext_word_list=['Laptop', 'Phone', 'Tablet', 'Monitor', 'Mouse', 'Keyboard', 'Camera'])
        price = random.uniform(20, 1500)

        cursor.execute("""
        INSERT INTO products (name, price)
        VALUES (%s, %s)
        """, (name, price))
    conn.commit()
    cursor.close()


def generate_transactions(conn, num_transactions=500):
    """Generates synthetic transaction data."""
    cursor = conn.cursor()
    for _ in range(num_transactions):
        user_id = random.randint(1, 100)  # Assuming 100 users
        product_id = random.randint(1, 50)  # Assuming 50 products
        quantity = random.randint(1, 5)
        transaction_date = fake.date_time_between(start_date="-2y", end_date="now")

        cursor.execute("""
        INSERT INTO transactions (user_id, product_id, quantity, transaction_date)
        VALUES (%s, %s, %s, %s)
        """, (user_id, product_id, quantity, transaction_date))
    conn.commit()
    cursor.close()


def main():
    # Database connection parameters
    db_params = {
        'dbname': 'your_dbname',
        'user': 'your_username',
        'password': 'your_password',
        'host': 'localhost'
    }

    try:
        # Connect to your PostgreSQL database
        conn = psycopg2.connect(**db_params)

        # Create tables
        create_tables(conn)

        # Generate and insert data
        generate_users(conn, 100)  # Generate 100 users
        generate_products(conn, 50)  # Generate 50 products
        generate_transactions(conn, 500)  # Generate 500 transactions

        print("Data generation completed successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()


if __name__ == "__main__":
    main()
