from pathlib import Path

import pytest

from table_extractor.bordered_service.models import Image


@pytest.fixture()
def demo_images_dir():
    return Path('images')


@pytest.fixture()
def demo_image_path(demo_images_dir):
    return demo_images_dir / 'demo.png'


@pytest.fixture()
def demo_image(demo_image_path):
    return Image(path=demo_image_path)
