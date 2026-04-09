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
from src.paths import DATA_ROOT

# project imports
from src.utils import logger


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
        logger.debug("Successfully unzipped files.")

    logger.debug("Finished dataset processing.")

def process_dataset():
    pass


# for testing purposes
if __name__ == "__main__":
    #unzip_dataset()
    process_dataset()