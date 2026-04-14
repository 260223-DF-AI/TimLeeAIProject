"""Processes all the data & adds it to the dataset using functions in database_core.py.
Combines core db logic & dataset processing.

You'll need to manually download the dataset from kaggle, follow these directions:
1. Visit https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/data
2. Authenticate with phone number if needed
3. Click "Download All" from the right hand side, all the way down
4. Put the dataset folder in src/data. create src/data if it doesn't exist

The script here will do the unzipping/processing for you so just drop the zip file in the data folder.

You will also need to add a .env file with your local postgres information:
1. Ensure you have a .env file in src
2. Write the connection string to it, replace password with your password:
CS = "postgresql://postgres:PASSWORD@localhost:5432/driver_image_classification"

After following both of these sets of instructions, run process_database.py once to have everything
set up for model training. It will take several minutes.
"""

# main imports
import zipfile 
import os
import shutil
import pandas as pd
import sqlalchemy as sa

# project imports
from src.utils import logger
from src.paths import DATA_ROOT
from src.database import database_core as db_core

# path constants
IMGS = DATA_ROOT / "imgs"
TRAIN = DATA_ROOT / "imgs/train"
VAL = DATA_ROOT / "imgs/val"
TEST = DATA_ROOT / "imgs/test"
UNLABELED = DATA_ROOT / "imgs/unlabeled"
FUTURE_TRAIN = DATA_ROOT / "imgs/future_train"

logger = logger.setup_logger(__name__, "debug")

def unzip_dataset() -> None:
    """Checks that data exists & unzips it."""

    logger.debug("Unzipping files, this will take about two minutes.")
    try:
        with zipfile.ZipFile(DATA_ROOT / "state-farm-distracted-driver-detection.zip", "r") as zip_ref:
            zip_ref.extractall(DATA_ROOT)
    except FileNotFoundError as e:
        logger.error("Couldn't find dataset zip file. Make sure the zip file is in src/data and " \
        "that it is named 'state-farm-distracted-driver-detection.zip'")
    except Exception as e:
        logger.error(f"Error type {type(e)}: {e}")
    else:
        logger.info("Successfully unzipped files.")

    logger.debug("Finished dataset processing.")

def clear_folders_to_reprocess():
    """Mostly for testing. Clears out all the folders for reprocessing."""

    logger.debug("Clearing out folders for reprocessing, this will take about 30 seconds.")

    try:
        shutil.rmtree(IMGS)
        os.remove(DATA_ROOT / "driver_imgs_list.csv")
        os.remove(DATA_ROOT / "sample_submission.csv")
    except Exception as e:
        logger.error(f"Error type {type(e)}: {e}")
    else:
        logger.info("Successfully cleared out folders for reprocessing.")
    

def process_images():
    """Processes the dataset with these steps:
    1. Rename existing data/imgs/test folder to data/imgs/unlabeled
    2. Creates necessary folders
    3. Partition existing train data into train/val/test
        a. along the way, condenses c1-c4 into c1
        b. also adds images to database
    """

    logger.debug("Processing dataset...")

    # rename old "test" folder
    try:
        if os.path.exists(UNLABELED):
            raise FileExistsError
        os.rename(TEST, UNLABELED)
    except FileExistsError:
        logger.info("Unlabeled folder already exists. Skipping.")
    except FileNotFoundError as e:
        logger.error("Couldn't find old test folder. Did you unzip the dataset first?")
    except Exception as e:
        logger.error(f"Error type {type(e)}: {e}")
    else:
        logger.info("Successfully renamed old test folder to unlabeled.")

    # verify train exists, create new val & test folders/subfolders
    try:
        if not os.path.exists(TRAIN):
            raise FileNotFoundError

        if os.path.exists(VAL) or os.path.exists(TEST) or os.path.exists(FUTURE_TRAIN):
            raise FileExistsError

        os.makedirs(VAL)
        os.makedirs(TEST)
        os.makedirs(FUTURE_TRAIN)

        for i in range (0, 7):
            os.makedirs(f"{VAL}/c{i}")
            os.makedirs(f"{TEST}/c{i}")
            os.makedirs(f"{FUTURE_TRAIN}/c{i}")

    except FileExistsError:
        logger.info("Val/test folders already exist. Assuming everything has already been partitioned & skipping rest of function.")
        return None
    except FileNotFoundError:
        logger.error("Couldn't find train folder. Did you unzip the dataset first?")
    except Exception as e:
        logger.error(f"Error type {type(e)}: {e}")
    else:
        logger.info("Successfully partitioned train data.")

    # create map for consolidating c1-c4 in the original database to one category, c1
    category_map = {"c0": "c0", "c1": "c1",  "c2": "c1",
        "c3": "c1", "c4": "c1", "c5": "c2", "c6": "c3",
        "c7": "c4", "c8": "c5", "c9": "c6"}
    
    # partition train & create df with image info
    try:
        # initialize df from csv
        image_df = pd.read_csv(DATA_ROOT / "driver_imgs_list.csv")
        # rename df columns to schema columns
        image_df = image_df.rename(columns={"img": "image_name", 
                                            "classname": "image_class", 
                                            "subject": "driver_id"})

        counter = 0
        # for each subfolder in train
        for category in os.listdir(TRAIN):
            for image in os.listdir(f"{TRAIN}/{category}"):
                # map new categories, consolidating the old c1-c4
                new_category = category_map[category]

                # Validation partition
                if counter % 9 == 0:
                    # move 1/9 to val
                    os.rename(f"{TRAIN}/{category}/{image}", f"{VAL}/{new_category}/{image}")

                    # configure df
                    image_df.loc[image_df["image_name"] == image, "partition_loc"] = "val"

                # Test partition
                elif counter % 9 == 1:
                    # move 1/9 to test
                    os.rename(f"{TRAIN}/{category}/{image}", f"{TEST}/{new_category}/{image}")

                    # configure df
                    image_df.loc[image_df["image_name"] == image, "partition_loc"] = "test"
                    
                # correctly categorize the rest & separate them to a new dir to prevent
                # accidentally going over them again
                else:
                    os.rename(f"{TRAIN}/{category}/{image}", f"{FUTURE_TRAIN}/{new_category}/{image}")

                    # configure df
                    image_df.loc[image_df["image_name"] == image, "partition_loc"] = "train"
                counter += 1
            
    except Exception as e:
        logger.error(f"Error type {type(e)}: {e}, line {e.__traceback__.tb_lineno}")
    else: 
        logger.info("Successfully partitioned train data.")

    # add image_df to db
    try:
        engine = db_core.get_engine()

        with engine.connect() as conn:
            conn.execution_options(isolation_level="AUTOCOMMIT")
            image_df.to_sql("images", con=conn, if_exists="append", index=False)

    except Exception as e:
        logger.error(f"Error type {type(e)}: {e}")
    else:
        logger.info("Successfully added image info to database.")
    finally:
        engine.dispose()

    # clean up/rename
    try:
        # now that exising train should contain only empty folders, remove them & train
        shutil.rmtree(TRAIN)

        # rename future_train to train
        os.rename(FUTURE_TRAIN, TRAIN)
    except Exception as e:
        logger.error(f"Error type {type(e)}: {e}")
    else:
        logger.debug("Successfully cleaned up train data.")

    logger.debug("Finished processing dataset.")

def add_drivers():
    logger.debug("Adding drivers...")

    # currently just adds driver ids to the table and no other information
    driver_id_list = ['p002', 'p012', 'p014', 'p015', 'p016', 'p021', 'p022', 'p024', 'p026',
 'p035', 'p039', 'p041', 'p042', 'p045', 'p047', 'p049', 'p050', 'p051',
 'p052', 'p056', 'p061', 'p064', 'p066', 'p072', 'p075', 'p081']
    
    try:
        engine = db_core.get_engine()

        with engine.connect() as conn:
            conn.execution_options(isolation_level="AUTOCOMMIT")
            for driver_id in driver_id_list:
                conn.execute(sa.text(f"INSERT INTO drivers (driver_id) VALUES ('{driver_id}')"))
    except Exception as e:
        logger.error(f"Error type {type(e)}: {e}")
    else:
        logger.info("Successfully added drivers to database.")
    finally:
        engine.dispose()

    logger.debug("Finished adding drivers.")

def log_http_request():
    """Logs an http request as it is made."""
    logger.debug("Logging http request...")
    try:
        engine = db_core.get_engine()

        with engine.connect() as conn:
            conn.execution_options(isolation_level="AUTOCOMMIT")
            conn.execute(sa.text(f"INSERT INTO http_requests (time_received) VALUES (now())"))
    except Exception as e:
        logger.error(f"Error type {type(e)}: {e}")
    else:
        logger.info("Successfully logged http request.")
    finally:
        engine.dispose()
    logger.debug("Finished logging http request.")

def log_cv_result(http_id: int, image_file_name, cv_result):
    """Logs cv result.
    @param http_id: id of the http request that triggered this input
    @param image_file_name: name of the image that was sent to the model
    @param cv_result: result of the cv, formatted as "cA: 00.00, cB: 00.00, cC: 00.00"
    """
    logger.debug("Logging cv result...")
    try:
        engine = db_core.get_engine()

        with engine.connect() as conn:
            conn.execution_options(isolation_level="AUTOCOMMIT")
            image_id = conn.execute(sa.text(f"SELECT image_id FROM images WHERE image_name = '{image_file_name}'")).fetchall()[0][0]
            conn.execute(sa.text(f"INSERT INTO cv_results (http_id, image_id, cv_result) VALUES({http_id}, {image_id}, '{cv_result}')"))
    except Exception as e:
        logger.error(f"Error type {type(e)}: {e}")
    else:
        logger.info("Successfully logged cv result.")
    finally:
        engine.dispose()
    logger.debug("Finished logging cv result.")
    

# for testing purposes
if __name__ == "__main__":
    #clear_folders_to_reprocess()
    #unzip_dataset()
    #db_core.delete_db()
    #db_core.create_db()
    #db_core.create_tables()
    #add_drivers()
    #process_images()
    #log_http_request()
    log_cv_result(1, "img_44733.jpg", "c0: 00.00, c1: 00.00, c2: 00.00")