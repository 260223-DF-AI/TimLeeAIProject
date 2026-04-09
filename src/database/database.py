"""Handles database creation, connection, writing, and reading.

You'll need to manually do this before database functions will work:
1. Create a JSON file in src called "keys.json"
2. add the following content, replacing the second password with 
    your local postgres password:
{
    "postgres": {
        "password" : "password"
    }
}
"""

# main imports
import psycopg2
import json

# project imports
from src.utils import logger

# constants
DB_NAME = "driver_image_classification"

logger = logger.setup_logger(__name__, "debug", console=False)


def connect_db(db: str = DB_NAME) -> tuple[psycopg2.extensions.connection, psycopg2.extensions.cursor]:
    """Do not call externally, is automatically called by other functions.
    Sets up psycopg2 connection and returns it."""

    logger.debug("Attempting db connection...")

    # Retrieve the user's local db password
    try:
        logger.debug("Retrieving user password...")
        with open("src/keys.json") as f:
            keys = json.load(f)
            user_password = keys["postgres"]["password"]
    except Exception as e:
        logger.error(f"Error type {type(e)}: {e}")
    else:
        logger.debug("Successfully retrieved user password.")

    # Set up/connect with psycopg2
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database=f"{db}",
            user="postgres",
            password=user_password
        )
        conn.autocommit = True
        cursor = conn.cursor()
    except Exception as e:
        logger.error(f"Error type {type(e)}: {e}")
    else:
        logger.debug("Successfully established psycopg2 connection.")
    
    return conn, cursor
    

def create_db(db: str = DB_NAME) -> None:
    """Creates the database."""
    logger.debug("Attempting db creation...")

    conn, cursor = connect_db()
    # Create the database
    try:
        cursor.execute(f"CREATE DATABASE {db};")
    except psycopg2.errors.DuplicateDatabase as e:
        logger.warning(f"Database {db} already exists.")
    except Exception as e:
        logger.error(f"Error type {type(e)}: {e}")
    else:
        logger.info(f"Successfully created database {db}.")
    
    conn.close()
    logger.debug(f"Finished attempting to create database {db}.")


def delete_db(db: str = DB_NAME) -> None:
    """Deletes the database."""

    logger.debug("Attempting db deletion...")

    conn, cursor = connect_db()
    # Delete the database
    try:
        cursor.execute(f"DROP DATABASE {db};")
    except psycopg2.errors.ObjectInUse as e:
        logger.error(f"Couldn't delete database since it's currently in use. Manually close the connection on your end and try again.")
    except Exception as e:
        logger.error(f"Error type {type(e)}: {e}")
    else:
        logger.info(f"Successfully deleted database {db}.")
    
    conn.close()
    logger.debug(f"Finished attempting to delete database {db}.")

def create_tables(db: str = DB_NAME) -> None:
    """Creates the tables from schema.sql file, more info about what tables/columns represent
    can be found in planning/about_database.md"""
    
    logger.debug("Attempting table creation...")

    conn, cursor = connect_db()
    # Create the tables
    try:
        cursor.execute(open("src/database/schema.sql", "r").read())
    except Exception as e:
        logger.error(f"Error type {type(e)}: {e}")
    else:
        logger.info("Successfully created tables.")
    
    conn.close()
    logger.debug("Finished attempting to create tables.")
    

# for testing purposes
if __name__ == "__main__":
    create_db(db="postgres")
    create_tables()
    # delete_db()
