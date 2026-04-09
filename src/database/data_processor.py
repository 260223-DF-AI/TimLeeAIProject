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

# project imports
from src.utils import logger
from src.paths import DATA_ROOT

logger = logger.setup_logger(__name__, "debug")

def unzip_dataset() -> None:
    """Checks that data exists & unzips it."""

    logger.debug("Unzipping files, this will take about a minute.")
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

def process_dataset():
    """Processes the dataset with these steps:
    1. Rename existing data/imgs/test folder to data/imgs/unlabeled
    2. Partition existing train data into train/val/test.
    
    """

    logger.debug("Processing dataset...")

    # rename old "test" folder
    try:
        if os.path.exists(DATA_ROOT / "imgs/unlabeled"):
            raise FileExistsError
        os.rename(DATA_ROOT / "imgs/test", DATA_ROOT / "imgs/unlabeled")
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
        train = DATA_ROOT / "imgs/train"
        if not os.path.exists(train):
            raise FileNotFoundError
        
        val = DATA_ROOT / "imgs/val"
        test = DATA_ROOT / "imgs/test"
        os.makedirs(val)
        os.makedirs(test)

        for i in range (0, 7):
            os.makedirs(f"{val}/c{i}")
            os.makedirs(f"{test}/c{i}")

    except FileNotFoundError:
        logger.error("Couldn't find train folder. Did you unzip the dataset first?")
    except Exception as e:
        logger.error(f"Error type {type(e)}: {e}")
    else:
        logger.info("Successfully partitioned train data.")

    logger.debug("Finished processing dataset.")

    # take 1 of every 9 files for test/val
    try:
        # for each subfolder in train
        pass
    except Exception as e:
        logger.error(f"Error type {type(e)}: {e}")
    else: 
        logger.info("Successfully partitioned train data.")

    logger.debug("Finished processing dataset.")

# for testing purposes
if __name__ == "__main__":
    #unzip_dataset()
    process_dataset()