from typing import List, Tuple, Dict
import json
import uuid
from dataclasses import dataclass, asdict,field
from pathlib import Path
from mmdet.apis import init_detector, inference_detector
import numpy as np
import cv2

from bordered_tables.models import TextField, InferenceTable, inference_result_to_boxes, BorderBox
from pdf_reader.pdf_reader import poppler_text_field_to_text_field, extract_text
from utils import extract_boxes_from_result

label2color = {
    0: '#e17282',
    1: '#ef703f',
    2: '#38fb5c',
    3: '#38f8fb',
}


category_to_id = {
    'Bordered': 0,
    'Cell': 1,
    'Borderless': 2,
}


@dataclass
class AnnotatedBBox(BorderBox):
    source: str

    @classmethod
    def from_bbox(cls, bbox: BorderBox, source: str):
        return cls(
            top_left_x=bbox.top_left_x,
            top_left_y=bbox.top_left_y,
            bottom_right_x=bbox.bottom_right_x,
            bottom_right_y=bbox.bottom_right_y,
            source=source
        )


@dataclass
class Category:
    id: int
    name: str
    color: str
    metadata: Dict[str, str] = field(default_factory=dict)
    keypoint_colors: List[str] = field(default_factory=list)
    supercategory: str = ''


@dataclass
class ImageCOCO:
    id: int
    file_name: str
    width: int
    height: int


def extract_boxes_from_inference_result(img_id, inference_result: List[InferenceTable], not_matched: List[AnnotatedBBox]):
    instances = []
    for table in inference_result:
        x1, y1, x2, y2 = table.bbox.box
        segm_box = [x1, y1, x2, y1, x2, y2, x1, y2]
        instances.append(
            {
                'id': int(str(uuid.uuid4().int)[:6]),
                'image_id': img_id,
                'category_id': category_to_id[table.label],
                'bbox': table.bbox.box,
                'segmentation': [segm_box],
                'area': (x2 - x1) * (y2 - y1),
                'score': 0.99,
                "iscrowd": False,
                "isbbox": True,
                "color": label2color[category_to_id[table.label]],
                "keypoints": [],
                "metadata": {},
            }
        )
        for t_field in table.tags:
            x1, y1, x2, y2 = t_field.box
            segm_box = [x1, y1, x2, y1, x2, y2, x1, y2]
            instances.append(
                {
                    'id': int(str(uuid.uuid4().int)[:6]),
                    'image_id': img_id,
                    'category_id': 1,
                    'bbox': t_field.box,
                    'segmentation': [segm_box],
                    'area': (x2 - x1) * (y2 - y1),
                    'score': 0.99,
                    "iscrowd": False,
                    "isbbox": True,
                    "color": label2color[1 if t_field.source == 'model' else 3],
                    "keypoints": [],
                    "metadata": {},
                }
            )
        for t_field in not_matched:
            x1, y1, x2, y2 = t_field.box
            segm_box = [x1, y1, x2, y1, x2, y2, x1, y2]
            instances.append(
                {
                    'id': int(str(uuid.uuid4().int)[:6]),
                    'image_id': img_id,
                    'category_id': 1,
                    'bbox': t_field.box,
                    'segmentation': [segm_box],
                    'area': (x2 - x1) * (y2 - y1),
                    'score': 0.99,
                    "iscrowd": False,
                    "isbbox": True,
                    "color": label2color[1 if t_field.source == 'model' else 3],
                    "keypoints": [],
                    "metadata": {},
                }
            )
    return instances


def extract_boxes_from_result_2(img_id, result, score_thr=0.3):
    bboxes_res, segm_result = result
    bboxes = np.vstack(bboxes_res)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bboxes_res)
    ]
    labels = np.concatenate(labels)

    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    assert bboxes.shape[1] == 5
    scores = bboxes[:, -1]
    inds = scores > score_thr
    bboxes = bboxes[inds, :]
    labels = labels[inds]

    instances = []
    for bbox, label in zip(bboxes, labels):
        bbox_int = [int(i) for i in bbox.astype(np.int32)[:4]]
        x1, y1, x2, y2 = bbox_int
        segm_box = [x1, y1, x2, y1, x2, y2, x1, y2]
        instances.append(
            {
                'id': int(str(uuid.uuid4().int)[:6]),
                'image_id': img_id,
                'category_id': int(label),
                'bbox': bbox_int,
                'segmentation': [segm_box],
                'area': (x2 - x1) * (y2 - y1),
                'score': float(bbox[-1]),
                "iscrowd": False,
                "isbbox": True,
                "color": label2color[int(label)],
                "keypoints": [],
                "metadata": {},
            }
        )
    return instances


def batch(pdf_path, img_dir, out_json, model, config, threshold):
    model = init_detector(config, model, device='cpu')
    bordered = Category(0, 'Bordered', '#ef703f')
    cell = Category(1, 'Cell', '#38fb5c')
    borderless = Category(2, 'Borderless', '#e17282')
    tables = {
        'images': [],
        'annotations': [],
        'categories': [asdict(c) for c in [cell, borderless, bordered]],
    }
    poppler_text = extract_poppler_text(Path(pdf_path), Path(img_dir))
    for idx, img in enumerate(list(Path(img_dir).iterdir())):
        print(f'Processing image {img}')
        img_obj = cv2.imread(str(img))
        h, w = img_obj.shape[:2]
        result = inference_detector(model, img)
        inference_tables, _ = inference_result_to_boxes(
            extract_boxes_from_result(result, ('Bordered', 'Cell', 'Borderless'), score_thr=threshold))
        not_matced = match_text_and_inference(poppler_text[str(img.name).replace(".png", '')], inference_tables)

        annotations = extract_boxes_from_inference_result(idx, inference_tables, not_matced)

        tables['images'].append(asdict(ImageCOCO(idx, str(img).rpartition('/')[-1], w, h)))
        tables['annotations'] += annotations
    with open(out_json, 'w') as f:
        json.dump(tables, f)


def extract_poppler_text(pdf_path: Path, images_path: Path):
        pages = extract_text(pdf_path)
        pages_dict = {}
        for page_num, poppler_page in pages.items():
            page_image = cv2.imread(str(images_path.absolute()) + f"/{page_num}.png")
            scale = page_image.shape[0] / poppler_page.bbox.height
            text_fields = [poppler_text_field_to_text_field(text_field, scale) for text_field in poppler_page.text_fields]
            pages_dict[page_num] = text_fields
        return pages_dict


def merge_closest_text_fields(text_fields: List[TextField]):
    merged_fields: List[TextField] = []
    curr_field: TextField = None
    for text_field in sorted(text_fields, key=lambda x: (x.bbox.top_left_y, x.bbox.top_left_x)):
        if not curr_field:
            curr_field = text_field
        if curr_field:
            if text_field.bbox.top_left_x - curr_field.bbox.bottom_right_x < 20 and text_field.bbox.top_left_x - curr_field.bbox.bottom_right_x > -20:
                curr_field = TextField(
                    bbox=curr_field.bbox.merge(text_field.bbox),
                    text=curr_field.text + " " + text_field.text
                )
            else:
                merged_fields.append(curr_field)
                curr_field = text_field
    if curr_field:
        merged_fields.append(curr_field)

    return merged_fields


def match_table_text(table: InferenceTable, text_boxes: List[TextField]):
    in_table: List[TextField] = []
    out_of_table: List[TextField] = []
    for t_box in text_boxes:
        if t_box.bbox.box_is_inside_another(table.bbox):
            in_table.append(t_box)
        else:
            out_of_table.append(t_box)
    return in_table, out_of_table


def match_bbox(border_boxes: List[BorderBox], text_box: TextField):
    for bbox in border_boxes:
        if text_box and text_box.bbox.box_is_inside_another(bbox, threshold=0.6):
            return True
    return False


def match_cells_text_fields(border_boxes: List[BorderBox], text_boxes: List[TextField], source='poppler'):
    result_boxes: List[AnnotatedBBox] = []
    result_boxes.extend([AnnotatedBBox.from_bbox(bbox, 'model') for bbox in border_boxes])
    for t_box in text_boxes:
        if t_box and not match_bbox(border_boxes, t_box):
            result_boxes.append(
                AnnotatedBBox.from_bbox(t_box.bbox, source)
            )

    return result_boxes


def match_text_and_inference(poppler_text: List[TextField], inference_result: List[InferenceTable]):
    text_fields_to_match = poppler_text
    for table in inference_result:
        in_table, text_fields_to_match = match_table_text(table, text_fields_to_match)
        in_table = merge_closest_text_fields(in_table)
        bboxes = match_cells_text_fields(table.tags, in_table)
        table.tags = bboxes
    not_matched = [AnnotatedBBox.from_bbox(t_f.bbox, 'poppler') for t_f in text_fields_to_match]
    return []


if __name__ == '__main__':
    model = 'models/epoch_36_mmd_v2.pth'
    config = 'models/cascadetabnet_config.py'
    batch('/home/ilia/Downloads/24.pdf',
          '/home/ilia/tess_test/24.pdf/images',
          'test.json',
          model,
          config,
          0.8)
