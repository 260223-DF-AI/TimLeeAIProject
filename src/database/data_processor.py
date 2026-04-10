"""Processes all the data.

You'll need to manually download the dataset from kaggle, follow these directions:
1. Visit https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/data
2. Authenticate with phone number if needed
3. Click "Download All" from the right hand side, all the way down
4. Put the dataset folder in src/data. create src/data if it doesn't exist

The script here will do the unzipping/processing for you so just drop the zip file in here.
"""

# main imports
import zipfile 
import os
import shutil

# project imports
from src.utils import logger
from src.paths import DATA_ROOT

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
    

def process_dataset():
    """Processes the dataset with these steps:
    1. Rename existing data/imgs/test folder to data/imgs/unlabeled
    2. Partition existing train data into train/val/test.
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

        os.makedirs(VAL)
        os.makedirs(TEST)
        os.makedirs(FUTURE_TRAIN)

        for i in range (0, 7):
            os.makedirs(f"{VAL}/c{i}")
            os.makedirs(f"{TEST}/c{i}")
            os.makedirs(f"{FUTURE_TRAIN}/c{i}")

    except FileExistsError:
        logger.info("Val/test folders already exist. Skipping.")
    except FileNotFoundError:
        logger.error("Couldn't find train folder. Did you unzip the dataset first?")
    except Exception as e:
        logger.error(f"Error type {type(e)}: {e}")
    else:
        logger.info("Successfully partitioned train data.")

    # create map for consolidating c1-c4 in the original database to one category, c1
    category_map = {
        "c0": "c0",
        "c1": "c1", 
        "c2": "c1",
        "c3": "c1",
        "c4": "c1",
        "c5": "c2",
        "c6": "c3",
        "c7": "c4",
        "c8": "c5",
        "c9": "c6"
    }

    # take 1 of every 9 files for test/val
    try:
        counter = 0
        # for each subfolder in train
        for category in os.listdir(TRAIN):
            for image in os.listdir(f"{TRAIN}/{category}"):
                # map new categories, consolidating the old c1-c4
                new_category = category_map[category]

                # move 1/9 to val
                if counter % 9 == 0:
                    os.rename(f"{TRAIN}/{category}/{image}", f"{VAL}/{new_category}/{image}")
                # move 1/9 to test
                elif counter % 9 == 1:
                    os.rename(f"{TRAIN}/{category}/{image}", f"{TEST}/{new_category}/{image}")
                # correctly categorize the rest & separate them to a new dir to prevent
                # accidentally going over them again
                else:
                    os.rename(f"{TRAIN}/{category}/{image}", f"{FUTURE_TRAIN}/{new_category}/{image}")
                counter += 1
            
    except Exception as e:
        logger.error(f"Error type {type(e)}: {e}, line {e.__traceback__.tb_lineno}")
    else: 
        logger.info("Successfully partitioned train data.")

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

# for testing purposes
if __name__ == "__main__":
    clear_folders_to_reprocess()
    unzip_dataset()
    process_dataset()