import os
import boto3
from dotenv import load_dotenv

# load .env
load_dotenv()

# create s3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    #aws_session_token=os.getenv("AWS_SESSION_TOKEN"),  # ok if None
    region_name=os.getenv("AWS_DEFAULT_REGION")
)

response = s3.list_objects_v2(Bucket="driver-photo-bucket-554448410167-us-east-1-an")

for obj in response.get("Contents", []):
    print(obj["Key"])


# bucket_name = "driver-photo-bucket-554448410167-us-east-1-an"
# key = "driver-photo-bucket-554448410167-us-east-1-an/blackberry1.jpg"   # full path inside bucket
# local_file = "downloaded_image.png"

# # download
# s3.download_file(bucket_name, key, local_file)

# print(f"Downloaded {key} → {local_file}")
