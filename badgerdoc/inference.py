import click
import os
import json
import time
from pathlib import Path

from .bordered_tables.models import inference_result_to_boxes
from .utils import extract_boxes_from_result, has_image_extension, load_predictions, convert_to_xywh
from .bordered_tables.bordered_tables_detection import detect_images, BorderBox, Cell
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import cv2
from badgerdoc.paddle_detector.text_detector import get_text_detector, dt_to_bboxes

DEFAULT_THRESHOLD = 0.3


@click.group()
def inference():
    pass


@inference.command()
@click.argument('img_path')
@click.argument('model')
@click.argument('config')
@click.option('--threshold', default=DEFAULT_THRESHOLD)
def show(img_path, model, config, threshold):
    model = init_detector(config, model, device='cpu')
    result = inference_detector(model, img_path)
    show_result_pyplot(model, img_path, result, score_thr=threshold)


def batch_(img_dir, out_dir, model, config, threshold, limit):
    model_name = Path(model).name.rstrip('.pth')
    model = init_detector(config, model, device='cpu')
    tables = {}
    for img in list(Path(img_dir).iterdir())[:limit]:
        if not has_image_extension(img):
            print(f'Not image {img}')
            continue
        print(f'Processing image {img}')
        result = inference_detector(model, img)
        boxes = extract_boxes_from_result(result, ('Borderless', 'Cell', 'Bordered'), score_thr=threshold)
        tables[img.name] = boxes
    os.makedirs(out_dir, exist_ok=True)
    with open(f'{out_dir}/{model_name}_{int(time.time())}.json', 'w') as f:
        json.dump(tables, f)


def inference_batch(img_dir, out_dir, model, config, threshold, limit, paddler_dir):
    # TODO: model to singleton
    model_name = Path(model).name.rstrip('.pth')
    model = init_detector(config, model, device='cpu')
    tables = {}
    for img in list(Path(img_dir).iterdir())[:limit]:
        if not has_image_extension(img):
            print(f'Not image {img}')
            continue
        print(f'Processing image {img}')
        result = inference_detector(model, img)
        inf_tables, not_matched = inference_result_to_boxes(
            extract_boxes_from_result(result, ('Bordered', 'Cell', 'Borderless'), score_thr=threshold))
        text_detector = get_text_detector(base_dir=paddler_dir)
        img_array = cv2.imread(str(img))
        for table in inf_tables:
            x1, y1, x2, y2 = table.bbox
            dt_boxes, elapse = text_detector(img_array[y1:y2, x1:x2])
            bboxes = dt_to_bboxes(dt_boxes)
            bboxes = [Cell(b[0]+x1, b[1]+y1, b[2]+x1, b[3]+y1) for b in bboxes]
            table.paddler = bboxes
        tables[img.name] = inf_tables
    os.makedirs(out_dir, exist_ok=True)
    # with open(f'{out_dir}/{model_name}_{int(time.time())}.json', 'w') as f:
    #     json.dump(tables, f)
    return tables


def extract_tables_bordered():
    pass

@inference.command()
@click.argument('img_dir')
@click.argument('out_dir')
@click.argument('model')
@click.argument('config')
@click.option('--threshold', default=DEFAULT_THRESHOLD)
@click.option('--limit', default=None, type=int, help='Specify the amount of files to process.')
def batch(img_dir, out_dir, model, config, threshold, limit):
    batch_(img_dir, out_dir, model, config, threshold, limit)


def draw_(img_dir, out_dir, json_file, threshold=0.0001):
    predictions = load_predictions(json_file)
    exec_time = json_file.rpartition('_')[-1].rstrip('.json')
    pfd_pgs_sizes = []
    for img_name, result in predictions.items():
        img_path = f'{img_dir}/{img_name}'
        img = cv2.imread(img_path)
        pfd_pgs_sizes.append(img.shape[:2])
    #     new_img = img.copy()
    #     img_obj = Image(Path(img_path))
    #     for obj in result:
    #         obj_label = obj['label']
    #         if obj_label == 'Bordered':
    #             table_bbox = obj['bbox']
    #             table_img = crop_img_to_bbox(img, table_bbox)
    #             boxes = recognize_bordered_table(table_img, threshold)
    # new_img = draw_boxes(new_img, boxes, table_bbox[:2], stroke=3)
    # new_img = draw_boxes(new_img, [convert_to_xywh(table_bbox)], color=(255, 0, 0), stroke=4)
    # detect_bordered_tables_on_image(img_obj)
    detection_json = detect_images(Path(img_dir), pfd_pgs_sizes)
    subdir = f'{out_dir}/{json_file.rpartition("_")[-1].rstrip(".json")}'
    os.makedirs(subdir, exist_ok=True)
    with open(f'{subdir}/detections.json', 'w') as f:
        json.dump(detection_json, f)

        # out_subdir = f'{out_dir}/{exec_time}'
        # os.makedirs(out_subdir, exist_ok=True)
        # cv2.imwrite(f'{out_subdir}/{img_name}', new_img)


@inference.command()
@click.argument('img_dir')
@click.argument('out_dir')
@click.argument('json_file')
def draw(img_dir, out_dir, json_file, threshold=0.0001):
    draw_(img_dir, out_dir, json_file, threshold)


if __name__ == '__main__':
    inference()
