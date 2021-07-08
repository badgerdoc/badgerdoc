import os
from pathlib import Path
import logging
from typing import Union

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
