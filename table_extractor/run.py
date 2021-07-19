import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict
from smart_open import open
import boto3
import botocore
import click
from decouple import config
from openpyxl.reader.excel import SUPPORTED_FORMATS

from excel_extractor.excel_extractor import run_excel_job
from table_extractor.cascade_rcnn_service.inference import (
    CascadeRCNNInferenceService,
)
from table_extractor.pipeline.pipeline import PageProcessor, pdf_preprocess
from table_extractor.visualization.table_visualizer import TableVisualizer

EXCEL_EXTRA = ('.xls', '.csv')

LOGGER = logging.getLogger(__name__)

LOGGING_FORMAT = "[%(asctime)s] - [%(name)s] - [%(levelname)s] - %(message)s"

CASCADE_CONFIG_PATH = (
    Path(os.environ.get("CASCADE_CONFIG_PATH"))
    if os.environ.get("CASCADE_CONFIG_PATH")
    else Path(__file__).parent.parent.joinpath(
        "configs/config_3_cls_w18.py"
    )
)
CASCADE_MODEL_PATH = (
    Path(os.environ.get("CASCADE_MODEL_PATH"))
    if os.environ.get("CASCADE_MODEL_PATH")
    else Path(__file__).parent.parent.joinpath("models/3_cls_w18_e30.pth")
)
PADDLE_MODEL_DIR = (
    Path(os.environ.get("PADDLE_MODEL_DIR"))
    if os.environ.get("PADDLE_MODEL_DIR")
    else Path(__file__).parent.parent.joinpath(
        "models/ch_ppocr_mobile_v2.0_det_infer"
    )
)
PADDLE_MODEL_CLS = (
    Path(os.environ.get("PADDLE_MODEL_CLS"))
    if os.environ.get("PADDLE_MODEL_CLS")
    else Path(__file__).parent.parent.joinpath(
        "models/ch_ppocr_mobile_v2.0_cls_infer"
    )
)
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


def download_model_from_path(model_file_path, output_path):
    tp = (get_transport_params() if model_file_path.startswith("s3://") else {})

    with open(model_file_path, 'rb', transport_params=tp) as remote_model_file:
        with open(str(output_path.absolute()), 'wb') \
                as model_file:
            model_file.write(remote_model_file.read())


def save_document(document: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path.absolute()), "w") as f:
        f.write(json.dumps(document, indent=4))


def configure_logging():
    formatter = logging.Formatter(LOGGING_FORMAT)
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(
        str(
            Path(__file__)
            .parent.parent.joinpath("python_logging.log")
            .absolute()
        )
    )
    file_handler.setFormatter(formatter)
    logging.root.setLevel(logging.DEBUG)
    logging.root.addHandler(console_handler)
    logging.root.addHandler(file_handler)


@click.group()
def run_pipeline():
    pass


def run_pipeline_sequentially(
    pdf_path: Path, output_dir: Path,  model_path: str, should_visualize: bool, paddle_on: bool
):
    if model_path:
        inference_model = Path("/tmp/source.pth")
        download_model_from_path(model_path, inference_model)
    else:
        inference_model = CASCADE_MODEL_PATH
    LOGGER.info(
        "Initializing CascadeMaskRCNN with config: %s and model: %s",
        CASCADE_CONFIG_PATH,
        inference_model,
    )
    cascade_rcnn_detector = CascadeRCNNInferenceService(
        CASCADE_CONFIG_PATH, inference_model, should_visualize
    )

    LOGGER.info("Visualizer should_visualize set to: %s", should_visualize)
    visualizer = TableVisualizer(should_visualize)
    page_processor = PageProcessor(
        cascade_rcnn_detector, visualizer, paddle_on
    )
    images_path, poppler_pages = pdf_preprocess(pdf_path, output_dir)
    pages = page_processor.process_pages(images_path, poppler_pages)
    document = {"doc_name": str(pdf_path.name), "pages": pages}

    return document


def run_sequentially_and_save(pdf_path, output_path, model_path, verbose, paddle_on):
    save_document(
        run_pipeline_sequentially(
            Path(pdf_path), Path(output_path), model_path, verbose, paddle_on
        ),
        Path(output_path) / Path(pdf_path).name / "document.json",
    )


@run_pipeline.command()
@click.argument("pdf_path")
@click.argument("output_path")
@click.option("--model_path", type=str, default=None)
@click.option("--verbose", type=bool)
@click.option("--paddle_on", type=bool)
def run_sequentially(pdf_path, output_path, model_path, verbose, paddle_on):
    run_sequentially_and_save(pdf_path, output_path, model_path, verbose, paddle_on)


@run_pipeline.command()
@click.argument("input_path")
@click.argument("output_path")
@click.option("--model_path", type=str, default=None)
@click.option("--verbose", type=bool)
@click.option("--paddle_on", type=bool)
def run(input_path, output_path, model_path, verbose, paddle_on):
    extension = os.path.splitext(input_path)[-1].lower()
    if extension == ".pdf":
        run_sequentially_and_save(input_path, output_path, model_path, verbose, paddle_on)
    elif extension in SUPPORTED_FORMATS or extension in EXCEL_EXTRA:
        run_excel_job(input_path, output_path)
    else:
        raise ValueError("Not supported file format")


if __name__ == "__main__":
    configure_logging()
    run_pipeline()
