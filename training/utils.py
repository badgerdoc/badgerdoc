import os
from pathlib import Path
import logging
from typing import Union
from smart_open import open

import boto3
import botocore
from decouple import config

AWS_S3_ENDPOINT = config("AWS_S3_ENDPOINT",
                         default="http://localhost:4566")
AWS_ACCESS_KEY_ID = config("AWS_ACCESS_KEY_ID", default="")
AWS_SECRET_ACCESS_KEY = config("AWS_SECRET_ACCESS_KEY", default="")
AWS_REGION = config("AWS_REGION", default=None)
AWS_S3_SSE_TYPE = config("AWS_S3_SSE_TYPE", default=None)

LOGGER = logging.getLogger(__file__)


def get_transport_params():
    boto_config = botocore.config.Config(retries={'mode': 'standard'})

    transport_params = {
        'resource_kwargs': {
            'config': boto_config,
            'endpoint_url': AWS_S3_ENDPOINT,
            'region_name': AWS_REGION
        }
    }

    if AWS_SECRET_ACCESS_KEY:
        session = boto3.Session(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        transport_params.update({
            'session': session
        })

    if AWS_S3_SSE_TYPE:
        transport_params.update({
            'multipart_upload_kwargs': {
                'ServerSideEncryption': AWS_S3_SSE_TYPE
            }
        })

    return transport_params


def get_local_model_filepath(local_work_dir: str) -> str:
    local_work_dir = Path(local_work_dir)
    pth_files = list(local_work_dir.glob("epoch_*.pth"))
    if len(pth_files) > 0:
        file_path = sorted(pth_files, key=os.path.getmtime, reverse=True)[0]
        LOGGER.debug(f"model local file path: {file_path}")
        return str(file_path.absolute())
    else:
        raise FileNotFoundError


def upload_model(model_file_path: Union[str, Path], model_out: str):
    LOGGER.debug(f"model file path: {model_out}")

    tp = get_transport_params() if model_out.startswith('s3://') else {}
    with open(model_file_path, 'rb') as model_file:
        with open(model_out, 'wb', transport_params=tp) as remote_model_file:
            remote_model_file.write(model_file.read())


def download_model_from_path(model_file_path, output_path):
    tp = (get_transport_params() if model_file_path.startswith("s3://") else {})

    with open(model_file_path, 'rb', transport_params=tp) as remote_model_file:
        with open(output_path, 'wb') \
                as model_file:
            model_file.write(remote_model_file.read())


def upload_dir_to_s3(local_directory, bucket, destination):
    s3 = boto3.client(
        's3',
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        endpoint_url=AWS_S3_ENDPOINT,
    )
    for root, dirs, files in os.walk(local_directory):
        for filename in files:
            local_path = os.path.join(root, filename)
            relative_path = os.path.relpath(local_path, local_directory)
            s3_path = os.path.join(destination, relative_path)
            LOGGER.debug("Uploading %s..." % s3_path)
            s3.upload_file(local_path, bucket, s3_path)


def download_s3_folder(bucket_name, s3_folder, local_dir=None):
    """
    Download the contents of a folder directory
    Args:
        bucket_name: the name of the s3 bucket
        s3_folder: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system
    """
    s3 = boto3.client(
        's3',
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        endpoint_url=AWS_S3_ENDPOINT,
    )
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = obj.key if local_dir is None \
            else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, target)
