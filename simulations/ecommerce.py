import psycopg2
from psycopg2.extras import DictCursor


def fetch_user_activities(user_id):
    """Fetch recent activities of a user."""
    with psycopg2.connect("dbname=ecommerce user=dbuser password=dbpass") as conn:
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute("""
                SELECT product_id, action_type FROM user_activity
                WHERE user_id = %s
                ORDER BY timestamp DESC
                LIMIT 100;
            """, (user_id,))
            activities = cur.fetchall()
    return activities


def update_recommendation_probability(user_id, product_id, updated_probability):
    """Update the recommendation probability for a user-product pair."""
    with psycopg2.connect("dbname=ecommerce user=dbuser password=dbpass") as conn:
        with conn.cursor() as cur:
            # This is a simplified example; in a real application, you would likely
            # need to insert or update a more complex structure.
            cur.execute("""
                UPDATE recommendations
                SET probability = %s
                WHERE user_id = %s AND product_id = %s;
            """, (updated_probability, user_id, product_id))
            conn.commit()


def bayesian_update_for_user(user_id):
    """Perform Bayesian update based on user activities."""
    activities = fetch_user_activities(user_id)
    # For each product, calculate the updated probability based on actions.
    # This is simplified; in reality, you would compute this based on the actions' nature.
    for activity in activities:
        product_id = activity['product_id']
        action_type = activity['action_type']
        # Assume we have a function to calculate updated_probability based on the action type.
        updated_probability = calculate_updated_probability(action_type)
        update_recommendation_probability(user_id, product_id, updated_probability)


def calculate_updated_probability(action_type):
    # Placeholder for the actual Bayesian update calculation based on action type.
    # The actual implementation would involve calculating the posterior probability
    # based on prior (existing recommendation probability) and likelihood (impact of the action type).
    return 0.75  # Example fixed value for demonstration purposes.

# Example usage:
# bayesian_update_for_user('some_user_id')
