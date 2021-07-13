import json
import os
import sys
import uuid
from copy import deepcopy

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import click
import logging

import cv2
import numpy as np

from random import shuffle

from table_extractor.bordered_service.models import Page
from table_extractor.model.table import StructuredTableHeadered, TextField, CellLinked, BorderBox
from table_extractor.pdf_service.pdf_to_image import convert_pdf_to_images
from table_extractor.poppler_service.poppler_text_extractor import extract_text, PopplerPage, \
    poppler_text_field_to_text_field
from table_extractor.tesseract_service.tesseract_extractor import TextExtractor
from table_extractor.visualization.table_visualizer import TableVisualizer
from training.utils import download_s3_folder, upload_dir_to_s3

LOGGING_FORMAT = "[%(asctime)s] - [%(name)s] - [%(levelname)s] - %(message)s"

LOGGER = logging.getLogger(__file__)
VISUALIZER = TableVisualizer(True)


@dataclass
class Category:
    id: int
    name: str
    color: str
    metadata: Dict[str, str] = field(default_factory=dict)
    keypoint_colors: List[str] = field(default_factory=list)
    supercategory: str = ''


TABLE = Category(1, 'table', '#ef703f')
CELL = Category(2, 'Cell', '#38fb5c')
HEADER = Category(3, 'header', '#e17282')
CATEGORIES = [asdict(c) for c in [TABLE, CELL, HEADER]]


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


def group_annotations_by_image_id(annotations):
    annotations_group = {}
    for ann in annotations:
        if ann['image_id'] in annotations_group:
            annotations_group[ann['image_id']].append(ann)
        else:
            annotations_group[ann['image_id']] = [ann]

    return annotations_group


def image_transform(image_meta, image_path, prefix, id_prefix, image_new_path):
    new_file_name = prefix + image_meta['file_name']
    os.system(f"cp {image_path / image_meta['file_name']} {image_new_path / new_file_name}")
    return {
        'file_name': new_file_name,
        'id': str(id_prefix) + str(image_meta['id']),
        'width': image_meta['width'],
        'height': image_meta['height']
    }


def transform_ann(image_id, id_prefix, ann):
    new_ann = deepcopy(ann)
    new_ann['image_id'] = image_id
    new_ann['id'] = int(str(id_prefix) + str(ann['id']))
    return new_ann


def merge_dataset(output_dir, *datasets):
    cats = datasets[0]['set']['categories']
    if any(cats != dataset['set']['categories'] for dataset in datasets):
        raise ValueError('categories are not equal')

    images = []
    annotations = []
    for idx, dataset in enumerate(datasets):
        name = dataset['name']
        coco = dataset['set']
        image_dir = dataset['image_dir']
        image_new_dir = output_dir / 'images'
        image_new_dir.mkdir(parents=True, exist_ok=True)

        ann_gr = group_annotations_by_image_id(coco['annotations'])
        for img_meta in coco['images']:
            new_image = image_transform(img_meta, image_dir, name, idx, image_new_dir)
            images.append(new_image)
            old_ann = ann_gr[img_meta['id']] if img_meta['id'] in ann_gr else []
            new_ann = [transform_ann(new_image['id'], idx, ann) for ann in old_ann]
            annotations.extend(new_ann)

    return {
        'images': images,
        'annotations': annotations,
        'categories': cats
    }


def save_split(split, coco, images_dir, out_dir):
    imgs_dir = out_dir / split / 'images'
    imgs_dir.mkdir(exist_ok=True, parents=True)
    for img in coco['images']:
        os.system(f"cp {str((images_dir / img['file_name']).absolute())} {str((imgs_dir / img['file_name']).absolute())}")
    with open(str((imgs_dir.parent / f"{split}.json").absolute()), 'w') as f:
        f.write(json.dumps(coco))


def test_train_val_split(coco, images_dir, out_dir):
    img_ids = [img['id'] for img in coco['images']]
    shuffle(img_ids)

    img_ids = img_ids[:int(len(img_ids) * 0.33)]

    test_s = int(len(img_ids) * 0.2)
    val_s = int(len(img_ids) * 0.1)

    test_ids = img_ids[:test_s]
    val_ids = img_ids[test_s:val_s+test_s]

    img_ann = group_annotations_by_image_id(coco['annotations'])
    test_ann = []
    val_ann = []
    train_ann = []
    test_imgs = []
    val_imgs = []
    train_imgs = []

    for img_meta in coco['images']:
        if img_meta['id'] in test_ids:
            test_imgs.append(img_meta)
            test_ann.extend(img_ann[img_meta['id']] if img_meta['id'] in img_ann else [])
        elif img_meta['id'] in val_ids:
            val_imgs.append(img_meta)
            val_ann.extend(img_ann[img_meta['id']] if img_meta['id'] in img_ann else [])
        elif img_meta['id'] in img_ids:
            train_imgs.append(img_meta)
            train_ann.extend(img_ann[img_meta['id']] if img_meta['id'] in img_ann else [])
        else:
            continue

    test_coco = {
        'images': test_imgs,
        'annotations': test_ann,
        'categories': deepcopy(coco['categories'])
    }
    val_coco = {
        'images': val_imgs,
        'annotations': val_ann,
        'categories': deepcopy(coco['categories'])
    }
    train_coco = {
        'images': train_imgs,
        'annotations': train_ann,
        'categories': deepcopy(coco['categories'])
    }

    sets = [
        ('test', test_coco),
        ('val', val_coco),
        ('train', train_coco),
    ]

    for name, c in sets:
        save_split(name, c, images_dir, out_dir)


def extract_pages(data: Dict[str, Any]) -> List[Page]:
    pages = []
    for page_raw in data['pages']:
        pages.append(Page.from_dict(page_raw))
    return pages


def match_source_data(source: Path) -> List[List[Path]]:
    pdfs = list((source / 'pdfs').glob('*.pdf'))
    json_source = list((source / 'json').glob('*.json'))
    names = {str(pdf.name).replace(".pdf", ""): [pdf] for pdf in pdfs}
    for json_s in json_source:
        name = str(json_s.name).replace(".json", "")
        if name in names:
            names[name].append(json_s)
        else:
            names[name] = [json_s]
    matched = []
    not_matched = []
    for name, paths in names.items():
        if len(paths) == 2:
            matched.append(paths)
        else:
            not_matched.extend(paths)
    if not_matched:
        LOGGER.warning("Following paths not matched: %s", ",".join([str(p) for p in not_matched]))
    if not matched:
        LOGGER.error("No data for preprocess found in dir %s", str(source.absolute()))
        raise ValueError("No data for preprocess found")
    return matched


def scale_poppler_result(
        img, output_path, poppler_page, image_path
):
    scale = img.shape[0] / poppler_page.bbox.height
    text_fields = [
        poppler_text_field_to_text_field(text_field, scale)
        for text_field in poppler_page.text_fields
    ]
    if text_fields:
        VISUALIZER.draw_object_and_save(
            img,
            text_fields,
            Path(f"{output_path}/poppler_text/{image_path.name}"),
        )
    return text_fields


def match_table_text(table: StructuredTableHeadered, text_boxes: List[TextField]):
    in_table: List[TextField] = []
    out_of_table: List[TextField] = []
    for t_box in text_boxes:
        if t_box.bbox.box_is_inside_box(table.bbox):
            in_table.append(t_box)
        else:
            out_of_table.append(t_box)
    return in_table, out_of_table


def match_cells_text(cells: List[CellLinked], text_box: TextField):
    for cell in cells:
        if text_box.bbox.box_is_inside_box(cell):
            cell.text_boxes.append(text_box)
            return True
    return False


def match_cells_text_fields(
        cells: List[CellLinked], text_boxes: List[TextField]
) -> Tuple[int, List[TextField]]:
    count = 0
    for cell in cells:
        cell.text_boxes = []
    not_matched = []
    for t_box in text_boxes:
        if match_cells_text(cells, t_box):
            count += 1
        else:
            not_matched.append(t_box)

    return count, not_matched


def flatten(list_fo_lists: List[List]) -> List:
    res = []
    for l in list_fo_lists:
        for element in l:
            res.append(element)
    return res


def match_cells_table(text_fields: List[TextField], table: StructuredTableHeadered):
    _ = match_cells_text_fields(table.all_cells, text_fields)


def actualize_text(tables: List[StructuredTableHeadered], image_path: Path, img_shape: Tuple[int]):
    # ToDo: Check if tesseract needed
    with TextExtractor(str(image_path.absolute())) as te:
        for table in tables:
            for cell in table.all_cells:
                if not cell.text_boxes or any(
                        [not text_box.text for text_box in cell.text_boxes]
                ):
                    top_left_x = max(0, cell.top_left_x + 4)
                    top_left_y = max(0, cell.top_left_y + 4)
                    bottom_right_x = min(img_shape[1], cell.bottom_right_x - 4)
                    bottom_right_y = min(img_shape[0], cell.bottom_right_y - 4)
                    if bottom_right_x - top_left_x > 10 and bottom_right_y - top_left_y > 10:
                        text, _, regions = te.extract_region(
                            top_left_x, top_left_y, bottom_right_x - top_left_x, bottom_right_y - top_left_y
                        )
                        if regions:
                            if len(regions) > 1:
                                regions = [reg for _, reg in regions]
                                x1 = int(min([reg['x'] for reg in regions]))
                                y1 = int(min([reg['y'] for reg in regions]))
                                x2 = int(max([reg['x'] + reg['w'] for reg in regions]))
                                y2 = int(max([reg['y'] + reg['h'] for reg in regions]))
                            else:
                                x1 = int(regions[0][1]['x'])
                                y1 = int(regions[0][1]['y'])
                                x2 = int(regions[0][1]['x'] + regions[0][1]['w'])
                                y2 = int(regions[0][1]['y'] + regions[0][1]['h'])
                            cell.text_boxes.append(TextField(bbox=BorderBox(
                                top_left_x=cell.top_left_x + 4 + x1,
                                top_left_y=cell.top_left_y + 4 + y1,
                                bottom_right_x=cell.top_left_x + 4 + x2,
                                bottom_right_y=cell.top_left_y + 4 + y2
                            ), text='tesseracted'))    # For now text is useless


def actualize_cell(cell: CellLinked):
    if cell.text_boxes and len(cell.text_boxes) > 1:
        return (
            max(cell.top_left_x, min([tb.bbox.top_left_x for tb in cell.text_boxes])),
            max(cell.top_left_y, min([tb.bbox.top_left_y for tb in cell.text_boxes])),
            min(cell.bottom_right_x, max([tb.bbox.bottom_right_x for tb in cell.text_boxes])),
            min(cell.bottom_right_y, max([tb.bbox.bottom_right_y for tb in cell.text_boxes])),
        )
    if cell.text_boxes and len(cell.text_boxes) == 1:
        return (
            max(cell.top_left_x, cell.text_boxes[0].bbox.top_left_x),
            max(cell.top_left_y, cell.text_boxes[0].bbox.top_left_y),
            min(cell.bottom_right_x, cell.text_boxes[0].bbox.bottom_right_x),
            min(cell.bottom_right_y, cell.text_boxes[0].bbox.bottom_right_y),
        )
    return []


def extract_annotations(tables: List[StructuredTableHeadered], img_id: int):
    instances = []
    for table in tables:
        x1, y1, x2, y2 = table.bbox.box
        segm_box = [x1, y1, x2, y1, x2, y2, x1, y2]
        instances.append(
            {
                'id': int(str(uuid.uuid4().int)[:6]),
                'image_id': img_id,
                'category_id': 1,
                'bbox': [x1, y1, x2 - x1, y2 - y1],
                'segmentation': [segm_box],
                'area': (x2 - x1) * (y2 - y1),
                'score': 0.99,
                "iscrowd": False,
                "isbbox": True,
                "color": '#e17282',
                "keypoints": [],
                "metadata": {},
            }
        )
        for cell in table.all_cells:
            coords = actualize_cell(cell)
            if coords:    # skip empty to avoid false-positive
                x1, y1, x2, y2 = coords
                segm_box = [x1, y1, x2, y1, x2, y2, x1, y2]
                instances.append(
                    {
                        'id': int(str(uuid.uuid4().int)[:6]),
                        'image_id': img_id,
                        'category_id': 2,
                        'bbox': [x1, y1, x2 - x1, y2 - y1],
                        'segmentation': [segm_box],
                        'area': (x2 - x1) * (y2 - y1),
                        'score': 0.99,
                        "iscrowd": False,
                        "isbbox": True,
                        "color": '#ef703f',
                        "keypoints": [],
                        "metadata": {},
                    }
                )
        if table.header_cols:
            x1 = min([cell.top_left_x for cell in flatten(table.header_cols)])
            y1 = min([cell.top_left_y for cell in flatten(table.header_cols)])
            x2 = max([cell.bottom_right_x for cell in flatten(table.header_cols)])
            y2 = max([cell.bottom_right_y for cell in flatten(table.header_cols)])
            segm_box = [x1, y1, x2, y1, x2, y2, x1, y2]
            instances.append(
                {
                    'id': int(str(uuid.uuid4().int)[:6]),
                    'image_id': img_id,
                    'category_id': 3,
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'segmentation': [segm_box],
                    'area': (x2 - x1) * (y2 - y1),
                    'score': 0.99,
                    "iscrowd": False,
                    "isbbox": True,
                    "color": '#38f8fb',
                    "keypoints": [],
                    "metadata": {},
                }
            )
        if table.header_rows:
            x1 = min([cell.top_left_x for cell in flatten(table.header_rows)])
            y1 = min([cell.top_left_y for cell in flatten(table.header_rows)])
            x2 = max([cell.bottom_right_x for cell in flatten(table.header_rows)])
            y2 = max([cell.bottom_right_y for cell in flatten(table.header_rows)])
            segm_box = [x1, y1, x2, y1, x2, y2, x1, y2]
            instances.append(
                {
                    'id': int(str(uuid.uuid4().int)[:6]),
                    'image_id': img_id,
                    'category_id': 3,
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'segmentation': [segm_box],
                    'area': (x2 - x1) * (y2 - y1),
                    'score': 0.99,
                    "iscrowd": False,
                    "isbbox": True,
                    "color": '#38f8fb',
                    "keypoints": [],
                    "metadata": {},
                }
            )
    return instances


def _draw_rectangle(
        color: Tuple[int, int, int],
        thickness: int,
        img: np.ndarray,
        bbox: List,
):
    cv2.rectangle(
        img,
        (int(bbox[0]), int(bbox[1])),
        (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
        color,
        thickness,
    )


def process_page(poppler_page: PopplerPage, page: Page, image_path: Path):
    img = cv2.imread(str(image_path.absolute()))
    text_fields = scale_poppler_result(img, image_path.parent.parent, poppler_page, image_path)
    text_fields_to_match = text_fields
    for table in page.tables:
        in_table, text_fields_to_match = match_table_text(
            table, text_fields_to_match
        )
        match_cells_table(in_table, table)
    actualize_text(page.tables, image_path, img.shape)
    image_id = int(str(uuid.uuid4().int)[:6])
    annotations = extract_annotations(page.tables, image_id)
    img_meta = {
        'file_name': str(image_path.name),
        'id': image_id,
        'width': img.shape[1],
        'height': img.shape[0]
    }

    for ann in annotations:
        if ann['category_id'] == 1:
            _draw_rectangle((0, 255, 0), 3, img, BorderBox(ann['bbox'][0], ann['bbox'][1], ann['bbox'][2], ann['bbox'][3]))
        elif ann['category_id'] == 2:
            _draw_rectangle((255, 0, 0), 2, img, BorderBox(ann['bbox'][0], ann['bbox'][1], ann['bbox'][2], ann['bbox'][3]))
        else:
            _draw_rectangle((0, 0, 255), 2, img, BorderBox(ann['bbox'][0], ann['bbox'][1], ann['bbox'][2], ann['bbox'][3]))
    draw_ann_path = image_path.parent.parent / 'draw_ann'
    draw_ann_path.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str((draw_ann_path / f"{image_path.name}").absolute()), img)
    return annotations, img_meta


def process_pdf(pdf_path: Path, json_path: Path, pdfs_preprocess_dir: Path):
    convert_pdf_to_images(pdf_path, pdfs_preprocess_dir / pdf_path.name, already_incl=True)
    poppler_pages = extract_text(pdf_path)
    with open(str(json_path.absolute()), 'r') as f:
        data = json.loads(f.read())
    markup_pages = extract_pages(data)
    #ToDo: scale pages according to images
    annotations = []
    img_metas = []
    for page in markup_pages:
        poppler_page = poppler_pages[str(page.page_num)]
        image_path = pdfs_preprocess_dir / pdf_path.name / 'images' / f"{page.page_num}.png"
        page_ann, img_meta = process_page(poppler_page, page, image_path)
        annotations.extend(page_ann)
        img_metas.append(img_meta)
    coco = {
        'annotations': annotations,
        'images': img_metas,
        'categories': CATEGORIES
    }
    with open(str((pdfs_preprocess_dir / pdf_path.name / "coco.json").absolute()), "w") as f:
        f.write(json.dumps(coco))

    return {
        'name': str(pdf_path.name),
        'set': coco,
        'image_dir': pdfs_preprocess_dir / pdf_path.name / 'images'
    }


@click.command()
@click.option("--working_dir", type=str)
@click.option("--source", type=str)
@click.option("--s3_bucket", type=str)
@click.option("--s3_source_folder", type=str)
@click.option("--s3_target_folder", type=str)
@click.option("--verbose", type=bool)
def preprocess(working_dir: str, source: str, s3_bucket, s3_source_folder, s3_target_folder, verbose: bool):
    VISUALIZER = TableVisualizer(verbose)

    if s3_bucket:
        if not s3_source_folder or not s3_target_folder:
            raise ValueError("s3_source_folder and s3_target_folder should be both provided if s3_bucket is defined")
        if not source:
            source = '/tmp/s3_source_download'
        download_s3_folder(s3_bucket, s3_source_folder, source)
    LOGGER.info(f"Resolved source folder {source}")

    if not working_dir:
        working_dir = '/tmp/preprocess_working'
    LOGGER.info(f"Resolved working directory {working_dir}")
    working_dir_path = Path(working_dir)

    matched = match_source_data(Path(source))

    pdfs_preprocess_dir = working_dir_path / 'pdfs'
    datasets = []
    for pair in matched:
        datasets.append(process_pdf(pair[0], pair[1], pdfs_preprocess_dir))
    full_dataset = working_dir_path / 'full'
    full_dataset.mkdir(parents=True, exist_ok=True)
    full_coco = merge_dataset(full_dataset, *datasets)

    with open(str((full_dataset / 'coco.json').absolute()), 'w') as f:
        f.write(json.dumps(full_coco))

    test_train_val_dir = working_dir_path / 'ttv'
    test_train_val_dir.mkdir(exist_ok=True, parents=True)
    test_train_val_split(full_coco, full_dataset / 'images', test_train_val_dir)

    if s3_bucket:
        upload_dir_to_s3(str(working_dir_path), s3_bucket, s3_target_folder)


if __name__ == "__main__":
    configure_logging()
    preprocess()
