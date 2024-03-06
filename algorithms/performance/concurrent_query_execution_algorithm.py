import asyncio
import asyncpg


async def run_query(conn, query):
    """
    Execute a single query on the database.

    Parameters:
    - conn: The database connection object.
    - query: The SQL query to execute.
    """
    try:
        start_time = asyncio.get_event_loop().time()
        result = await conn.fetch(query)
        end_time = asyncio.get_event_loop().time()
        print(f"Query completed in {end_time - start_time:.2f} seconds.")
        return result
    except Exception as e:
        print(f"Query failed: {e}")
        return None


async def concurrent_query_execution(queries):
    """
    Execute multiple queries concurrently on the database.

    Parameters:
    - queries: A list of SQL query strings to execute.
    """
    # Database connection parameters
    conn_info = {
        'user': 'your_username',
        'password': 'your_password',
        'database': 'your_database',
        'host': 'localhost'
    }

    # Establish a connection to the database
    conn = await asyncpg.connect(**conn_info)

    # Schedule concurrent execution of queries
    tasks = [run_query(conn, query) for query in queries]
    results = await asyncio.gather(*tasks)

    # Close the database connection
    await conn.close()

    return results


# List of queries to execute concurrently
queries = [
    "SELECT sleep(2);",  # Simulate a query taking 2 seconds
    "SELECT sleep(1);",  # Simulate a query taking 1 second
    "SELECT sleep(3);"  # Simulate a query taking 3 seconds
]

# Run the concurrent query execution
loop = asyncio.get_event_loop()
loop.run_until_complete(concurrent_query_execution(queries))
