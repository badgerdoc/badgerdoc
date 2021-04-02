import os
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis_fspaths import fspaths

from table_extractor.bordered_service.bordered_tables_detection import (
    detect_bordered_tables_on_image,
    detect_tables_on_page,
)
from table_extractor.common.utils import has_image_extension


@given(fspaths(allow_pathlike=True))
def test_has_image_extension_property(path):
    p = Path(str(path))
    assert isinstance(has_image_extension(p), bool)


@pytest.mark.parametrize(
    'filename, result',
    [
        ('test.png', True),
        ('test.PNG', True),
        ('test.jpg', True),
        ('test.JPG', True),
        ('test.jpeg', True),
        ('test.JPEG', True),
        ('test.bmp', True),
        ('test.BMP', True),
    ],
)
def test_has_image_extension_result(filename, result):
    assert has_image_extension(Path(filename)) == result


@pytest.mark.parametrize('draw', [True, False])
def test_detect_bordered_tables_on_image(draw, demo_image):
    detect_bordered_tables_on_image(
        demo_image, draw, res_path=Path('./test_results')
    )

    if draw:
        assert os.path.exists(Path('test_results') / demo_image.path.name)
        os.remove(Path('test_results') / demo_image.path.name)


@pytest.mark.parametrize('draw', [True, False])
def test_detect_tables_on_page(draw, demo_image_path):
    result = detect_tables_on_page(
        demo_image_path, draw, res_path=Path('./test_results')
    )
    assert result

    if draw:
        assert os.path.exists(Path('test_results') / result.path.name)
        os.remove(Path('test_results') / result.path.name)
