"""Handles database creation, connection, writing, and reading. Designed to be called 
from process_database.py, not on its own.
"""

# main imports
import sqlalchemy as sa
import pandas as pd
import os
from dotenv import load_dotenv
from pathlib import Path

# project imports
from src.utils import logger
from src.paths import SRC_ROOT

# constants
DB_NAME = "driver_image_classification"

logger = logger.setup_logger(__name__, "debug")

def get_connection_string(db: str = DB_NAME) -> str:
    try:
        load_dotenv(SRC_ROOT / ".env")
        logger.debug("Retrieving connection string...")
        cs = f"{os.getenv("CS")}/{db}"
        logger.debug(f"CS: {cs}")
    except KeyError as e:
        logger.error("Couldn't find your connection string, make sure to follow the instructions at the top of the file.")
    except Exception as e:
        logger.error(f"Error type {type(e)}: {e}")
    else:
        logger.debug("Successfully retrieved connection string.")
        return cs


def get_engine(db: str = DB_NAME) -> sa.Engine:
    """Do not call externally, is automatically called by other functions.
    Sets up psycopg2 connection and returns it."""

    logger.debug("Attempting db connection...")

    # connect
    try:
        cs = get_connection_string(db)
        engine = sa.create_engine(cs)
    except Exception as e:
        logger.error(f"Error type {type(e)}: {e}")
    else:
        logger.debug("Successfully established sqlalchemy connection.")
        return engine
    

def create_db(db: str = DB_NAME) -> None:
    """Creates the database."""

    logger.debug("Attempting db creation...")

    # Create the database
    try:
        default_engine = get_engine("postgres")

        with default_engine.connect() as conn:
            conn.execution_options(isolation_level="AUTOCOMMIT")
            conn.execute(sa.text(f"CREATE DATABASE {db}"))
    except sa.exc.ProgrammingError as e:
        logger.info(f"Database {db} already exists. Skipping.")
    except Exception as e:
        logger.error(f"Error type {type(e)}: {e}")
    else:
        logger.info(f"Successfully created database {db}.")
    finally:
        default_engine.dispose()

    logger.debug(f"Finished attempting to create database {db}.")


def delete_db(db: str = DB_NAME) -> None:
    """Deletes the database."""

    logger.debug("Attempting to clear db...")
    
    # Delete the database
    try:
        default_engine = get_engine("postgres")
        with default_engine.connect() as conn:
            conn.execution_options(isolation_level="AUTOCOMMIT")
            conn.execute(sa.text(f"DROP DATABASE IF EXISTS {db}"))
    except sa.exc.OperationalError:
        logger.error("Database is currently in use, close the connection on your end and try again.")
    except Exception as e:
        logger.error(f"Error type {type(e)}: {e}")
    else:
        logger.info(f"Successfully deleted database {db}.")
    finally:
        default_engine.dispose()
    logger.debug(f"Finished attempting to delete database {db}.")

def create_tables(db: str = DB_NAME) -> None:
    """Creates the tables from schema.sql file, more info about what tables/columns represent
    can be found in planning/about_database.md"""
    
    logger.debug("Attempting table creation...")

    # Create the tables
    try:
        engine = get_engine(db)
        schema = SRC_ROOT / "database/schema.sql"

        with engine.connect() as conn:
            conn.execution_options(isolation_level="AUTOCOMMIT")
            conn.execute(sa.text(schema.read_text()))
    except Exception as e:
        logger.error(f"Error type {type(e)}: {e}")
    else:
        logger.info("Successfully created tables.")
    finally:
        conn.close()
        engine.dispose()
    logger.debug("Finished attempting to create tables.")


# for testing purposes
if __name__ == "__main__":
    create_db()
    create_tables()
    #delete_db()
    
