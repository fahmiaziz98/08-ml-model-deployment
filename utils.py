import os
import boto3
import streamlit as st
from pathlib import Path

# credentials aws
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
bucket_name = os.getenv("BUCKET_NAME")


s3 = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key
)


def download_model_from_s3(local_path: Path, s3_prefix: str):
    if os.path.exists(local_path) and os.listdir(local_path):
        st.toast(f"✅ Model {local_path} Available!", icon="🎉")
        return

    os.makedirs(local_path, exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")

    for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        if "Contents" in result:
            for key in result["Contents"]:
                s3_key = key["Key"]
                local_file = os.path.join(local_path, os.path.relpath(s3_key, s3_prefix))
               
                os.makedirs(os.path.dirname(local_file), exist_ok=True)
                s3.download_file(bucket_name, s3_key, local_file)










