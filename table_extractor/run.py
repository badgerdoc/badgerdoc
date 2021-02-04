import json
import logging
import sys
import os
from pathlib import Path
from typing import Dict

import click

from table_extractor.cascade_rcnn_service.inference import CascadeRCNNInferenceService
from table_extractor.paddle_service.text_detector import PaddleSwitchWrapper
from table_extractor.pipeline.pipeline import PageProcessor, pdf_preprocess
from table_extractor.visualization.table_visualizer import TableVisualizer

LOGGER = logging.getLogger(__name__)

LOGGING_FORMAT = "[%(asctime)s] - [%(name)s] - [%(levelname)s] - %(message)s"

CASCADE_CONFIG_PATH = Path(os.environ.get("CASCADE_CONFIG_PATH")) if os.environ.get("CASCADE_CONFIG_PATH") \
    else Path(__file__).parent.parent.joinpath("models/cascadetabnet_config.py")
CASCADE_MODEL_PATH = Path(os.environ.get("CASCADE_MODEL_PATH")) if os.environ.get("CASCADE_MODEL_PATH") \
    else Path(__file__).parent.parent.joinpath("models/epoch_41_acc_94_mmd_v2.pth")
PADDLE_MODEL_DIR = Path(os.environ.get("PADDLE_MODEL_DIR")) if os.environ.get("PADDLE_MODEL_DIR") \
    else Path(__file__).parent.parent.joinpath("models/ch_ppocr_mobile_v2.0_det_infer")
PADDLE_MODEL_CLS = Path(os.environ.get("PADDLE_MODEL_CLS")) if os.environ.get("PADDLE_MODEL_CLS") \
    else Path(__file__).parent.parent.joinpath("models/ch_ppocr_mobile_v2.0_cls_infer")


def save_document(document: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path.absolute()), 'w') as f:
        f.write(json.dumps(document, indent=4))


def configure_logging():
    formatter = logging.Formatter(LOGGING_FORMAT)
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(str(Path(__file__).parent.parent.joinpath('python_logging.log').absolute()))
    file_handler.setFormatter(formatter)
    logging.root.setLevel(logging.DEBUG)
    logging.root.addHandler(console_handler)
    logging.root.addHandler(file_handler)


@click.group()
def run_pipeline():
    pass


def run_pipeline_sequentially(pdf_path: Path, output_dir: Path, should_visualize: bool, paddle_on: bool):
    LOGGER.info("Initializing CascadeMaskRCNN with config: %s and model: %s", CASCADE_CONFIG_PATH, CASCADE_MODEL_PATH)
    cascade_rcnn_detector = CascadeRCNNInferenceService(CASCADE_CONFIG_PATH, CASCADE_MODEL_PATH, should_visualize)

    LOGGER.info("Initializing Paddle with model_dir: %s and model_cls: %s, paddle mode: %s",
                PADDLE_MODEL_DIR, PADDLE_MODEL_CLS, paddle_on)
    paddle_detector = PaddleSwitchWrapper(PADDLE_MODEL_DIR, PADDLE_MODEL_CLS, paddle_on)
    LOGGER.info("Visualizer should_visualize set to: %s", should_visualize)
    visualizer = TableVisualizer(should_visualize)
    page_processor = PageProcessor(
        cascade_rcnn_detector,
        paddle_detector,
        visualizer,
        paddle_on
    )
    images_path, poppler_pages = pdf_preprocess(pdf_path, output_dir)
    pages = page_processor.process_pages(images_path, poppler_pages)
    document = {
        'doc_name': str(pdf_path.name),
        'pages': pages
    }

    return document


def run_sequentially_and_save(pdf_path, output_path, verbose, paddle_on):
    save_document(run_pipeline_sequentially(Path(pdf_path), Path(output_path), verbose, paddle_on),
                  Path(output_path) / Path(pdf_path).name / 'document.json')


@run_pipeline.command()
@click.argument('pdf_path')
@click.argument('output_path')
@click.option('--verbose', type=bool)
@click.option('--paddle_on', type=bool)
def run_sequentially(pdf_path, output_path, verbose, paddle_on):
    run_sequentially_and_save(pdf_path, output_path, verbose, paddle_on)


if __name__ == '__main__':
    configure_logging()
    run_pipeline()