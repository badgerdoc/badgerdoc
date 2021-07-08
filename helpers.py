import os
import shutil
import uuid
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
from pdf2image import convert_from_path
from pikepdf import Pdf
import json

import datetime

from random import shuffle

from copy import deepcopy

from mmdetection.mmdet.apis import inference_detector
from table_extractor.bordered_service.models import TABLE_TAGS
from table_extractor.cascade_rcnn_service.inference import CLASS_NAMES
from table_extractor.cascade_rcnn_service.utils import extract_boxes_from_result
from table_extractor.model.table import BorderBox
from sklearn.cluster import DBSCAN

others_path = Path('/home/ilia/Downloads/OneDrive_1_4-14-2021/United_Healthcare/')

out_path = Path('/home/ilia/cigna/united')

out_path.mkdir(parents=True, exist_ok=True)

for path in others_path.glob('*.pdf'):
    try:
        input_pdf = Pdf.open((str(path.absolute())))
        for n, page in enumerate(input_pdf.pages):
            new_output = Pdf.new()
            filename = path.name.replace(".pdf", "") + f"_{n + 1}.pdf"
            filepath = out_path / filename
            new_output.pages.append(page)
            new_output.save(str(filepath.absolute()))
    except Exception as e:
        print(str(path) + ' ' + str(e))



pdfs = Path('/home/ilia/cigna/sort_2')
out = Path('/home/ilia/cigna/')
for dir in pdfs.iterdir():
    out_pdf = Pdf.new()
    for path in dir.glob('*.pdf'):
        input_pdf = Pdf.open((str(path.absolute())))
        out_pdf.pages.extend(input_pdf.pages)
    filepath = out / f"{str(dir.name)}.pdf"
    out_pdf.save(str(filepath))


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
            old_ann = ann_gr[img_meta['id']]
            new_ann = [transform_ann(new_image['id'], idx, ann) for ann in old_ann]
            annotations.extend(new_ann)

    return {
        'images': images,
        'annotations': annotations,
        'categories': cats
    }


def filter_annotations_by_category(dataset, ids):
    annotations = []
    for ann in dataset['annotations']:
        if ann['category_id'] in ids:
            annotations.append(deepcopy(ann))
    image_ids = {ann['image_id'] for ann in annotations}
    images = []
    for img in dataset['images']:
        if img['id'] in image_ids:
            images.append(deepcopy(img))
    cats = []
    for cat in dataset['categories']:
        if cat['id'] in ids:
            cats.append(deepcopy(cat))
    return {
        'images': images,
        'annotations': annotations,
        'categories': cats
    }


def remap_categories(dataset, cat_mapping):
    annotations = []
    for ann in dataset['annotations']:
        if ann['category_id'] in cat_mapping:
            ann = deepcopy(ann)
            ann['category_id'] = cat_mapping[ann['category_id']]
            annotations.append(ann)
    cats = []
    for cat in dataset['categories']:
        if cat['id'] in cat_mapping:
            cat = deepcopy(cat)
            cat['id'] = cat_mapping[cat['id']]
            cats.append(cat)
    return {
        'images': deepcopy(dataset['images']),
        'annotations': annotations,
        'categories': cats
    }


def images_from_path(images_path):
    images = []
    for image in images_path.glob('*.png'):
        img = cv2.imread(str(image.absolute()))
        images.append(
            {
                'file_name': str(image.name),
                'id': int(str(uuid.uuid4().int)[:6]),
                'width': img.shape[1],
                'height': img.shape[0]
            }
        )
    return images


def copy_by_mapping(cats, images, annotations, mapping):
    copy_ann = []
    copy_imgs = []
    ann_mapping = group_annotations_by_image_id(annotations)
    for img_name, copy_to in mapping.items():
        image = images[img_name]
        if image['id'] in ann_mapping:
            anns = ann_mapping[image['id']]
            for c_img_name in copy_to:
                c_img = images[c_img_name]
                annt = deepcopy(anns)
                for ann in annt:
                    ann['image_id'] = c_img['id']
                copy_ann.extend(annt)
                copy_imgs.append(c_img)
    return {
        'images': copy_imgs,
        'annotations': copy_ann,
        'categories': deepcopy(cats)
    }


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
    # x1 = max(0, int(bbox[0]))
    # y1 = max(0, int(bbox[1]))
    # if x1 >= img.shape[1] or y1 >= img.shape[0]:
    #     return
    # x2 = max(min(int(bbox[2] + bbox[0]), img.shape[1] - 1), x1 + 1)
    # y2 = max(min(int(bbox[3] + bbox[1]), img.shape[0] - 1), y1 + 1)
    # sub = img[y1:y2, x1:x2]
    #
    # black = np.zeros_like(sub)
    # black[0:black.shape[0], 0:black.shape[1]] = color
    #
    # blend = cv2.addWeighted(sub, 0.75, black, 0.25, 0)
    # img[y1:y2, x1:x2] = blend


def rescale_annotation(annotation, w_scale, h_scale):
    r_ann = deepcopy(annotation)
    r_ann['bbox'] = [
        r_ann['bbox'][0] * w_scale,
        r_ann['bbox'][1] * h_scale,
        r_ann['bbox'][2] * w_scale,
        r_ann['bbox'][3] * h_scale,
        ]
    r_ann['segmentation'] = [
        [
            r_ann['bbox'][0],
            r_ann['bbox'][1],
            r_ann['bbox'][0] + r_ann['bbox'][2],
            r_ann['bbox'][1],
            r_ann['bbox'][0] + r_ann['bbox'][2],
            r_ann['bbox'][1] + r_ann['bbox'][3],
            r_ann['bbox'][0],
            r_ann['bbox'][1] + r_ann['bbox'][3],
            ]
    ]
    r_ann['area'] = int(r_ann['bbox'][2] * r_ann['bbox'][3])
    return r_ann


def resize_image_ann(image_meta, annotations, source_path, target_path, out_path):
    image_meta = deepcopy(image_meta)
    source = cv2.imread(str(source_path.absolute()))
    target = cv2.imread(str(target_path.absolute()))
    w_scale = target.shape[1] / source.shape[1]
    h_scale = target.shape[0] / source.shape[0]
    r_ann = [rescale_annotation(ann, w_scale, h_scale) for ann in annotations]
    image_meta['width'] = target.shape[1]
    image_meta['height'] = target.shape[0]
    image_meta['file_name'] = str(target_path.name)

    out_path.mkdir(exist_ok=True, parents=True)
    os.system(f'cp {str(target_path.absolute())} {str((out_path / image_meta["file_name"]).absolute())}')
    for ann in r_ann:
        _draw_rectangle(
            (255, 0, 0),
            1,
            target,
            ann['bbox']
        )
    (out_path / 'annotated').mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str((out_path / 'annotated' / image_meta["file_name"]).absolute()), target)
    return r_ann, image_meta


def resize_annotations(coco, source_prefix, repl, source_path, target_path, out_path):
    ann_by_img_id = group_annotations_by_image_id(coco['annotations'])
    images = []
    annotations = []
    for img_meta in coco['images']:
        annt = ann_by_img_id[img_meta['id']]
        s_img_path = source_path / img_meta['file_name']
        t_img_path = target_path / img_meta['file_name'].replace(source_prefix, repl)
        r_ann, r_img_meta = resize_image_ann(img_meta, annt, s_img_path, t_img_path, out_path)
        images.append(r_img_meta)
        annotations.extend(r_ann)
    return {
        'images': images,
        'annotations': annotations,
        'categories': deepcopy(coco['categories'])
    }


def convert_pdf_to_images(
        pdf_file: Path, out_dir: Path, dpi: int = 90
):
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = convert_from_path(
        pdf_file,
        dpi=dpi,
        output_folder=str(out_dir.absolute()),
        paths_only=True,
        fmt="png",
    )
    for i, page in enumerate(pages):
        shutil.move(page, out_dir.absolute() / f"{i}.png")


def group_by_image_prefix(coco_ann):
    prefix_img = {}
    for img_meta in coco_ann['images']:
        if img_meta['file_name'][:10] in prefix_img:
            prefix_img[img_meta['file_name'][:10]].append(img_meta)
        else:
            prefix_img[img_meta['file_name'][:10]] = [img_meta]

    img_ann = group_annotations_by_image_id(coco_ann['annotations'])

    splits = []
    for prefix, img_metas in prefix_img.items():
        img_annt = []
        for img_meta in img_metas:
            img_annt.extend(img_ann[img_meta['id']])
        splits.append((prefix,
                       {
                           'images': img_metas,
                           'annotations': img_annt,
                           'categories': deepcopy(coco_ann['categories'])
                       },
                       Path('/home/ilia/cigna/merge/images'),
                       Path('/home/ilia/cigna/img_90') / prefix,
                       Path('/home/ilia/cigna/merge/images_90')))
    return splits


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
            if img_meta['id'] in img_ann:
                test_ann.extend(img_ann[img_meta['id']])
        elif img_meta['id'] in val_ids:
            val_imgs.append(img_meta)
            if img_meta['id'] in img_ann:
                val_ann.extend(img_ann[img_meta['id']])
        elif img_meta['id'] in img_ids:
            train_imgs.append(img_meta)
            if img_meta['id'] in img_ann:
                train_ann.extend(img_ann[img_meta['id']])
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


def copy_group(prefix, coco, imgs_dir, num_copies=2):
    ann_img = group_annotations_by_image_id(coco['annotations'])

    to_copy = [img_meta for img_meta in coco['images'] if img_meta['file_name'].startswith(prefix)]

    for img_meta in to_copy:
        for i in range(num_copies):
            new_img_meta = deepcopy(img_meta)
            new_img_meta['id'] = str(new_img_meta['id']) + str(i)
            new_img_meta['file_name'] = str(i) + '_' + new_img_meta['file_name']
            annotations = ann_img[img_meta['id']]
            new_ann = deepcopy(annotations)
            for ann in new_ann:
                ann['image_id'] = new_img_meta['id']
                ann['id'] = str(ann['id']) + str(i)
            coco['images'].append(new_img_meta)
            coco['annotations'].extend(new_ann)
            os.system(f"cp {str((imgs_dir / img_meta['file_name']).absolute())} {str((imgs_dir / new_img_meta['file_name']).absolute())}")


def remove_group(prefix, coco):
    ann_img = group_annotations_by_image_id(coco['annotations'])

    to_copy = [img_meta for img_meta in coco['images'] if not img_meta['file_name'].startswith(prefix)]

    new_meta = []
    new_annt = []
    for img_meta in to_copy:
        new_img_meta = deepcopy(img_meta)
        new_meta.append(new_img_meta)
        annotations = ann_img[img_meta['id']]
        new_annt.extend(deepcopy(annotations))
    coco['images'] = new_meta
    coco['annotations'] = new_annt


label2color = {
    "Bordered": '#e17282',
    "Cell": '#ef703f',
    "Borderless": '#38fb5c',
    "table_header": '#38f8fb',
}

label_to_cat = {
    "Bordered": 0,
    "Cell": 1,
    "Borderless": 2,
    "table_header": 3,
}


def bboxes_to_annotations(bboxes, img_id):
    instances = []
    for bbox_l in bboxes:
        x1, y1, x2, y2 = bbox_l['bbox']
        segm_box = [x1, y1, x2, y1, x2, y2, x1, y2]
        instances.append(
            {
                'id': int(str(uuid.uuid4().int)[:6]),
                'image_id': img_id,
                'category_id': label_to_cat[bbox_l['label']],
                'bbox': [x1, y1, x2 - x1, y2 - y1],
                'segmentation': [segm_box],
                'area': (x2 - x1) * (y2 - y1),
                'score': 1.0,
                "iscrowd": False,
                "isbbox": True,
                "color": label2color[bbox_l['label']],
                "keypoints": [],
                "metadata": {},
            }
        )
    return instances


def inference_to_coco(img_source=Path('/home/ilia/cigna/img_90/2_tbls/'),
                      model=None,
                      img_out_path=Path('/home/ilia/cigna/2_tbls_inf'),
                      categories=None):
    print(datetime.datetime.now().strftime('%H:%M:%S'))
    annotations = []
    img_metas = []

    for idx, img in enumerate(sorted(img_source.glob('*.png'))):
        result = inference_detector(model, str(img.absolute()))
        boxes = extract_boxes_from_result(
            result, CLASS_NAMES, score_thr=0.3
        )
        img_arr = cv2.imread(str(img.absolute()))
        img_metas.append({
            'file_name': str(img.name),
            'id': idx,
            'width': img_arr.shape[1],
            'height': img_arr.shape[0]
        })
        annotations.extend(
            bboxes_to_annotations(boxes, idx)
        )
        for bbox in boxes:
            if bbox['label'] in TABLE_TAGS:
                _draw_rectangle((0, 255, 0), 1, img_arr, BorderBox(bbox['bbox'][0], bbox['bbox'][1], bbox['bbox'][2], bbox['bbox'][3]))
            elif bbox['label'] == 'Cell':
                _draw_rectangle((255, 0, 0), 1, img_arr, BorderBox(bbox['bbox'][0], bbox['bbox'][1], bbox['bbox'][2], bbox['bbox'][3]))
            else:
                _draw_rectangle((0, 0, 255), 1, img_arr, BorderBox(bbox['bbox'][0], bbox['bbox'][1], bbox['bbox'][2], bbox['bbox'][3]))
        out = img_out_path / 'images' / img.name
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out.absolute()), img_arr)
    print(datetime.datetime.now().strftime('%H:%M:%S'))
    with open(str((img_out_path / 'coco.json').absolute()), 'w') as f:
        f.write(json.dumps({
            'images': img_metas,
            'annotations': annotations,
            'categories': categories
        }))


def xywh_to_xyxy(annotation):
    ann = deepcopy(annotation)
    ann['bbox'] = [
        ann['bbox'][0],
        ann['bbox'][1],
        ann['bbox'][0] + ann['bbox'][2],
        ann['bbox'][1] + ann['bbox'][3]
    ]
    return ann


def xyxy_to_xywh(annotation):
    ann = deepcopy(annotation)
    ann['bbox'] = [
        ann['bbox'][0],
        ann['bbox'][1],
        ann['bbox'][2] - ann['bbox'][0],
        ann['bbox'][3] - ann['bbox'][1]
    ]
    return ann


def is_intersect(bb1, bb2, threshold=0.5) -> bool:
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])
    if x_right < x_left or y_bottom < y_top:
        intersection_area = 0.0
    else:
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    return any(
        (intersection_area / bb) > threshold for bb in (bb1_area, bb2_area)
    )


def in_segment(segment, value):
    top, bottom = segment
    return top <= value <= bottom


def border_in_zone(zone_bbox, cell_bbox):
    x1, y1, x2, y2 = zone_bbox
    cx1, cy1, cx2, cy2 = cell_bbox
    if in_segment((x1, x2), cx1) and in_segment((y1, y2), cy1) and in_segment((y1, y2), cy2):
        return 'left'
    if in_segment((x1, x2), cx2) and in_segment((y1, y2), cy1) and in_segment((y1, y2), cy2):
        return 'right'
    return None


class Holder:
    def __init__(self, value):
        self.value = value


def bbox_to_holders(ann):
    ann['bbox'] = [
        Holder(val) for val in ann['bbox']
    ]


def holders_to_bbox(ann):
    ann['bbox'] = [
        val.value for val in ann['bbox']
    ]


def cluster(holders):
    if len(holders) < 3:
        return
    coords = [holder.value for holder in holders]
    np_coords = np.array(coords).reshape(-1, 1)
    clust = DBSCAN(eps=5, min_samples=1).fit(np_coords)
    lines = {}
    for holder, label in zip(holders, clust.labels_):
        if label in lines:
            lines[label].append(holder)
        else:
            lines[label] = [holder]
    for lines_v in lines.values():
        avg = sum([line.value for line in lines_v]) / len(lines_v)
        for holder in lines_v:
            holder.value = avg
    return


def align(annotations):
    tables = [ann for ann in annotations if ann['category_id'] in (1, 3)]
    cells = [ann for ann in annotations if ann['category_id'] == 2]
    zones = [ann for ann in annotations if ann['category_id'] == 14]
    heades = [ann for ann in annotations if ann['category_id'] == 12]

    tables_arr = []
    for table in tables:
        table_cells = []
        for cell in cells:
            if is_intersect(table['bbox'], cell['bbox']):
                table_cells.append(cell)
        table_zones = []
        for zone in zones:
            if is_intersect(table['bbox'], zone['bbox']):
                table_zones.append(zone)
        tables_arr.append(
            {
                "table": table,
                "zones": zones,
                "cells": table_cells
            }
        )

    for t in tables_arr:
        for zone in t['zones']:
            lefts = []
            rights = []
            for cell in t['cells']:
                targ_border = border_in_zone(zone['bbox'], cell['bbox'])
                if targ_border and targ_border == 'left':
                    lefts.append(cell)
                if targ_border and targ_border == 'right':
                    rights.append(cell)
            x = [l['bbox'][0] for l in lefts]
            x.extend([r['bbox'][2] for r in rights])
            if x:
                avg_x = sum(x) / len(x)
                for cell in lefts:
                    cell['bbox'][0] = avg_x + 1
                for cell in rights:
                    cell['bbox'][2] = avg_x - 1
        for cell in t['cells']:
            if cell['bbox'][0] <= t['table']['bbox'][0] + 30:
                cell['bbox'][0] = t['table']['bbox'][0]
            if cell['bbox'][2] >= t['table']['bbox'][2] - 30:
                cell['bbox'][2] = t['table']['bbox'][2]
    for t in tables_arr:
        holders = []
        for cell in t['cells']:
            bbox_to_holders(cell)
            holders.extend([cell['bbox'][1], cell['bbox'][3]])
        cluster(holders)
        for cell in t['cells']:
            holders_to_bbox(cell)

    return tables + cells + heades
