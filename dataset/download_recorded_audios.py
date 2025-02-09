import boto3
import os
from dotenv import load_dotenv

load_dotenv()

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET"),
)

bucket_name = "finetunian"
output_dir = "dataset/audios"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix="dataset/audios/")

    for obj in response.get("Contents", []):
        key = obj["Key"]
        if key.endswith(".wav"):
            output_path = os.path.join(output_dir, os.path.basename(key))
            if os.path.exists(output_path):
                print(f"File {output_path} already exists, skipping download.")
                continue
            s3.download_file(bucket_name, key, output_path)
            print(f"Downloaded {key} to {output_path}")
