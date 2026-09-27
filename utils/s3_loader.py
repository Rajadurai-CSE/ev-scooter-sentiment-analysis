import boto3
import os
import pandas as pd
import json

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_ACCESS_KEY"),
    region_name="ap-south-1"
)

print("Bucket name",os.environ.get("BUCKET_NAME"))

def load_s3_csv(key:str):
    response = s3.get_object(
        Bucket = os.environ.get("BUCKET_NAME"),
        Key = key
    )
    return pd.read_csv(response["Body"])

def load_s3_json(key:str):
    response = s3.get_object(
        Bucket = os.environ.get("BUCKET_NAME"),
        Key = key
    )
    return json.loads(response["Body"].read())
    
