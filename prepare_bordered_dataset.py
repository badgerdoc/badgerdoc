from typing import List, Tuple, Dict
import json
import uuid
from dataclasses import dataclass, asdict,field
from pathlib import Path
from mmdet.apis import init_detector, inference_detector
import numpy as np
import cv2
import os

from bordered_tables.models import TextField, InferenceTable, inference_result_to_boxes, BorderBox
from pdf_reader.pdf_reader import poppler_text_field_to_text_field, extract_text
from utils import extract_boxes_from_result
from pipeline import convert_pdf_to_images
from bordered_tables.bordered_tables_detection import detect_tables_on_page
from prepare_dataset import AnnotatedBBox, label2color, category_to_id, Category, ImageCOCO
from random import randint


def extract_poppler_text(pdf_path: Path, images_path: Path):
    pages = extract_text(pdf_path.absolute())
    pages_dict = {}
    for page_num, poppler_page in pages.items():
        page_image = cv2.imread(str(images_path.absolute()) + f"/{page_num}.png")
        scale = page_image.shape[0] / poppler_page.bbox.height
        text_fields = [poppler_text_field_to_text_field(text_field, scale) for
                       text_field in poppler_page.text_fields]
        pages_dict[page_num] = text_fields
    return pages_dict


def draw_boxes(img, boxes: List[BorderBox], color=(0, 0, 0), thickness=2):
    img = img.copy()
    for bbox in boxes:
        x1, y1, x2, y2 = bbox.box
        cv2.rectangle(
            img,
            pt1=(int(x1), int(y1)),
            pt2=(int(x2), int(y2)),
            color=color,
            thickness=thickness,
        )
    return img


def extract_boxes_from_inference_result(img_id, bboxes: List[AnnotatedBBox], category_id: int):
    result = []
    for bbox in bboxes:
        x1, y1, x2, y2 = [v + randint(-3, 7) for v in bbox.box]  # add some variance
        segm_box = [x1, y1, x2, y1, x2, y2, x1, y2]
        result.append(
            {
                'id': int(str(uuid.uuid4().int)[:6]),
                'image_id': img_id,
                'category_id': category_id,
                'bbox': bbox.box,
                'segmentation': [segm_box],
                'area': (x2 - x1) * (y2 - y1),
                'score': 0.99,
                "iscrowd": False,
                "isbbox": True,
                "color": label2color[1 if bbox.source == 'model' else 3],
                "keypoints": [],
                "metadata": {},
            }
        )
    return result


if __name__ == '__main__':
    for i in range(1, 6):
        pdf_file = f'multiline_cells_{i}.pdf'
        out_json = pdf_file.replace('.pdf', '.json')
        pdf_path = Path('/home/egor/bordered/multiline_cells') / pdf_file
        out_path = Path('/home/egor/bordered_dataset/') / pdf_file.replace('.pdf', '')
        img_dir = out_path / pdf_file / 'images'
        detections_dir = out_path / pdf_file / 'detected'
        os.makedirs(str(detections_dir), exist_ok=True)
        out_json_path = out_path / pdf_file / out_json
        images_path = convert_pdf_to_images(pdf_path, out_path)
        poppler_text = extract_poppler_text(Path(pdf_path), Path(img_dir))
        bordered = Category(0, 'Bordered', '#ef703f')
        cell = Category(1, 'Cell', '#38fb5c')
        borderless = Category(2, 'Borderless', '#e17282')
        tables = {
            'images': [],
            'annotations': [],
            'categories': [asdict(c) for c in [cell, borderless, bordered]],
        }
        for img_path in list(Path(img_dir).iterdir()):
            page_num = img_path.name.replace('.png', '')
            detection_img = detect_tables_on_page(img_path, None)
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            ppl_boxes = [text.bbox for text in poppler_text[page_num]]
            table_bboxes = sorted([cell for cell in detection_img.objs], key=lambda t: t.height * t.width)
            table_perimeter = [AnnotatedBBox.from_bbox(table_bboxes.pop(-1), 'model')] if table_bboxes else []
            merged_ppl_boxes = []
            for t_box in table_bboxes:
                group = []
                for p_box in ppl_boxes:
                    if p_box.box_is_inside_another(t_box, 0.7):
                        group.append(p_box)
                if group:
                    x1 = min([v.top_left_x for v in group])
                    y1 = min([v.top_left_y for v in group])
                    x2 = max([v.bottom_right_x for v in group])
                    y2 = max([v.bottom_right_y for v in group])
                    merged_ppl_boxes.append(AnnotatedBBox.from_bbox(BorderBox(x1, y1, x2, y2), 'poppler'))

            img = draw_boxes(img, merged_ppl_boxes, (255, 0, 0))
            img = draw_boxes(img, table_perimeter, (255, 0, 255), 3)
            cv2.imwrite(str(detections_dir / f'{page_num}.png'), img)
            result = extract_boxes_from_inference_result(str(page_num), merged_ppl_boxes, category_to_id['Cell'])
            result += extract_boxes_from_inference_result(str(page_num), table_perimeter, category_to_id['Bordered'])

            tables['images'].append(asdict(ImageCOCO(str(page_num), img_path.name, w, h)))
            tables['annotations'] += result
        with open(str(out_json_path), 'w') as f:
            json.dump(tables, f)