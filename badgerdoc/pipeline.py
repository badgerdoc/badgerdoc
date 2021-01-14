import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import click
import cv2
import numpy
from tesserocr import PSM

from .bordered_tables.bordered_tables_detection import detect_tables_on_page
from .bordered_tables.models import (
    InferenceTable, Cell, TableHeadered, BorderBox, Row, Table, TextField, Page
)
from .inference import batch_, draw_, DEFAULT_THRESHOLD, inference_batch
from .pdf_reader.pdf_reader import (
    convert_pdf_to_images, extract_text_to_json, extract_text, poppler_text_field_to_text_field,
)
from .semi_bordered import parse_semi_bordered
from .tesseract.extractor import TextExtractor
from .tesseract.tesseract_manager import ocr_pages_in_path
from .text_cells_matcher.text_cells_matcher import match_table_text, match_cells_text_fields
from .utils import has_image_extension

logger = logging.getLogger(__name__)


def configure_logging():
    console_handler = logging.StreamHandler(stream=sys.stdout)
    logging.root.setLevel(8)
    logging.root.addHandler(console_handler)


@click.group()
def run_pipeline():
    pass


def box_to_cell(box, table_origin) -> Cell:
    y0, x0 = table_origin
    x, y, w, h = box
    return Cell(x + x0, y + y0, x + x0 + w, y + y0 + h)


def find_tables_in_boxes(cells: List[Cell], x_max) -> List[Row]:
    h_lines = {}

    for box in sorted(cells, key=lambda x: (x.top_left_x, x.top_left_y)):
        h_line_key = box[1]
        if h_line_key not in h_lines:
            row = Row(
                bbox=BorderBox(box[0], box[1], x_max, box[3]),
                table_id=1,
            )
            row.add(box)
            h_lines[h_line_key] = row
        else:
            h_lines[h_line_key].add(box)

    return list(h_lines.values())


# TODO: move to algorithm
def semi_bordered(page_img, inference_table: InferenceTable):
    top_left_x = inference_table.bbox.top_left_x
    top_left_y = inference_table.bbox.top_left_y
    bottom_right_x = inference_table.bbox.bottom_right_x
    bottom_right_y = inference_table.bbox.bottom_right_y
    table_image = page_img[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    table_origin = (top_left_y, top_left_x)  # (y1, x1)
    # TODO: rewrite try ... catch
    try:
        boxes, header_box = parse_semi_bordered(table_image)
    except Exception as e:
        logger.warning(str(e))
        return None
    header_cell = None
    if header_box:
        header_cell = box_to_cell(header_box, table_origin)
    cells = [box_to_cell(box, table_origin) for box in boxes]

    in_rows = []
    in_header = []
    for cell in cells:
        if header_cell and cell.box_is_inside_another(header_cell):
            in_header.append(cell)
        else:
            in_rows.append(cell)
    if not in_rows and in_header:
        in_rows = in_header
        in_header = []
    rows = find_tables_in_boxes(in_rows, bottom_right_x)

    if not boxes:
        return None
    # TODO: find also cols
    table = TableHeadered(bbox=inference_table.bbox, table_id=0, cols=[], rows=rows, header=in_header)

    return table, header_cell


def text_to_cell(text_field: TextField):
    return Cell(
        top_left_x=text_field.bbox.top_left_x,
        top_left_y=text_field.bbox.top_left_y,
        bottom_right_x=text_field.bbox.bottom_right_x,
        bottom_right_y=text_field.bbox.bottom_right_y,
    )


def merge_closest_text_fields(text_fields: List[TextField]):
    merged_fields: List[TextField] = []
    curr_field: TextField = None
    for text_field in sorted(text_fields, key=lambda x: (x.bbox.top_left_y, x.bbox.top_left_x)):
        if not curr_field:
            curr_field = text_field
        if curr_field:
            if text_field.bbox.top_left_x - curr_field.bbox.bottom_right_x < 20 \
                    and text_field.bbox.top_left_x - curr_field.bbox.bottom_right_x > -20:
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


def cmp_to_key(mycmp):
    """Convert a cmp= function into a key= function"""

    class K:
        def __init__(self, obj, *args):
            self.obj = obj

        def __lt__(self, other):
            return mycmp(self.obj, other.obj) < 0

        def __gt__(self, other):
            return mycmp(self.obj, other.obj) > 0

        def __eq__(self, other):
            return mycmp(self.obj, other.obj) == 0

        def __le__(self, other):
            return mycmp(self.obj, other.obj) <= 0

        def __ge__(self, other):
            return mycmp(self.obj, other.obj) >= 0

        def __ne__(self, other):
            return mycmp(self.obj, other.obj) != 0

    return K


def extract_table_from_inference(inf_table: InferenceTable, header_box: Optional[BorderBox], not_matched_text):
    merged_t_fields = merge_closest_text_fields(not_matched_text)
    inf_table.tags.extend([text_to_cell(text_field) for text_field in merged_t_fields])

    def compare(cell_1: Cell, cell_2: Cell):
        if cell_2.bottom_right_y - cell_1.top_left_y > 0 and \
                cell_1.bottom_right_y - cell_2.top_left_y > 0 and \
                min(cell_2.bottom_right_y - cell_1.top_left_y, cell_1.bottom_right_y - cell_2.top_left_y) \
                / min(cell_2.height, cell_1.height) > 0.2:
            return cell_1.top_left_x - cell_2.top_left_x
        else:
            return cell_1.top_left_y - cell_2.top_left_y

    in_header = []
    in_rows = []
    for cell in inf_table.tags:
        if header_box and cell.box_is_inside_another(header_box, 0.7):
            in_header.append(cell)
        else:
            in_rows.append(cell)

    if not in_rows:
        in_rows = in_header
        in_header = []

    rows = []
    current_top_y = 0
    current_bottom_y = 0
    current_row = Row(
        bbox=None,
        table_id=1,
    )
    for cell in sorted(in_rows, key=lambda x: (x.top_left_x + x.top_left_y * 5000)):
        if current_bottom_y - cell.top_left_y > 0 and \
                cell.bottom_right_y - current_top_y > 0 and \
                min(cell.bottom_right_y - current_top_y, current_bottom_y - cell.top_left_y) \
                / min(cell.height, current_bottom_y - current_top_y) > 0.2:
            current_row.add(cell)
        else:
            if current_row.objs:
                rows.append(current_row)
            current_row = Row(
                bbox=cell,
                table_id=1,
            )
            current_row.add(cell)
            current_top_y = cell.top_left_y
            current_bottom_y = cell.bottom_right_y
    if current_row.objs:
        rows.append(current_row)
    if not in_header and rows and len(rows) > 1:
        in_header = rows[0].objs
        rows = rows[1:]

    return TableHeadered(bbox=inf_table.bbox, table_id=0, cols=[], rows=rows, header=in_header)


def block_to_json(block):
    if isinstance(block, TextField):
        return {
            "type": "text",
            "text": block.text,
            "bbox": block.bbox.box
        }
    elif isinstance(block, Table):
        return table_to_json(block)
    return {}


def cnt_ciphers(cells: List[Cell]):
    count = 0
    for cell in cells:
        for char in "".join([tb.text for tb in cell.text_boxes]):
            if char in '0123456789':
                count += 1
    return count


def actualize_header(table: TableHeadered):
    if table.header:
        count_ciphers = cnt_ciphers(table.header)
        header_candidates = []
        current_ciphers = count_ciphers
        for row in table.rows:
            count = cnt_ciphers(row.objs)
            if count / 5 > current_ciphers:
                break
            else:
                header_candidates.append(row.objs)

        if len(header_candidates) < len(table.rows):
            for cand in header_candidates:
                table.header.extend(cand)
            table.rows = table.rows[len(header_candidates):]
    elif len(table.rows) > 1:
        count_ciphers = cnt_ciphers(table.rows[0].objs)
        header_candidates = [table.rows[0].objs]
        current_ciphers = count_ciphers
        for row in table.rows[1:]:
            count = cnt_ciphers(row.objs)
            if count / 5 > current_ciphers:
                break
            else:
                header_candidates.append(row.objs)

        if len(header_candidates) < len(table.rows):
            for cand in header_candidates:
                table.header.extend(cand)
            table.rows = table.rows[len(header_candidates):]


def table_to_json(table: Table):
    json_t = {
        'type': 'table',
        'bbox': table.bbox.box
    }
    rows = []
    for row in table.rows:
        cells = []
        for cell in row.objs:
            text = " ".join([text_box.text for text_box in
                             sorted(cell.text_boxes, key=lambda x: (x.bbox.top_left_y, x.bbox.top_left_x))])
            cells.append({
                "bbox": cell.box,
                "text": text
            })
        rows.append({
            'cells': cells
        })
    json_t['rows'] = rows

    if hasattr(table, 'header') and table.header:
        h_cells = sorted(table.header, key=lambda x: (x.top_left_x, x.top_left_y))
        top_y = max([h_c.top_left_y for h_c in h_cells])
        bottom_y = min([h_c.top_left_y for h_c in h_cells])
        top = []
        bottom = []
        other = []
        for cell in h_cells:
            if top_y - 10 < cell.top_left_y < top_y + 10:
                top.append(cell)
            elif bottom_y - 10 < cell.top_left_y < bottom_y + 10:
                bottom.append(cell)
            else:
                other.append(cell)
        head = []
        if top and bottom:
            chunk = len(top) // len(bottom)
            if len(top) % len(bottom) > 0:
                chunk += 1
            for i, t in enumerate(bottom):
                end = (i + 1) * chunk
                if end >= len(top):
                    end = len(top)
                top_ch = top[i * chunk:end]
                t_text = " ".join([text_box.text for text_box in
                                   sorted(t.text_boxes, key=lambda x: (x.bbox.top_left_y, x.bbox.top_left_x))])
                for ce in top_ch:
                    text = t_text + "->" + " ".join([text_box.text for text_box in sorted(ce.text_boxes, key=lambda x: (
                        x.bbox.top_left_y, x.bbox.top_left_x))])
                    if text:
                        head.append({
                            "bbox": ce.box,
                            "text": text
                        })
        for ce in other:
            text = " ".join([text_box.text for text_box in
                             sorted(ce.text_boxes, key=lambda x: (x.bbox.top_left_y, x.bbox.top_left_x))])
            if head:
                head.append({
                    "bbox": ce.box,
                    "text": text
                })

        json_t['header'] = head
    return json_t


def draw_text_boxes(img: numpy.ndarray, text_fields: List[TextField]):
    for text_field in text_fields:
        cv2.rectangle(img,
                      (text_field.bbox[0], text_field.bbox[1]),
                      (text_field.bbox[2], text_field.bbox[3]),
                      (0, 255, 0),
                      3)


def draw_inference(img: numpy.ndarray, inference_result: List[InferenceTable]):
    for inference_table in inference_result:
        cv2.rectangle(img,
                      (inference_table.bbox[0], inference_table.bbox[1]),
                      (inference_table.bbox[2], inference_table.bbox[3]),
                      (255, 0, 0),
                      3)
        cv2.putText(img,
                    f"{inference_table.label}: {inference_table.confidence}",
                    (inference_table.bbox[0], inference_table.bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 0, 0),
                    2)
        for box in inference_table.tags:
            cv2.rectangle(img,
                          (box.box[0], box.box[1]),
                          (box.box[2], box.box[3]),
                          (255, 0, 0),
                          3)


def draw_table(img: numpy.ndarray, table: Table):
    for row in table.rows:
        for obj in row.objs:
            if obj.text_boxes:
                cv2.rectangle(img,
                              (int(obj.box[0]), int(obj.box[1])),
                              (int(obj.box[2]), int(obj.box[3])),
                              (0, 0, 255),
                              5)
            else:
                cv2.rectangle(img,
                              (int(obj.box[0]), int(obj.box[1])),
                              (int(obj.box[2]), int(obj.box[3])),
                              (0, 128, 128),
                              5)
    if hasattr(table, 'header') and table.header:
        for obj in table.header:
            cv2.rectangle(img,
                          (int(obj.box[0]), int(obj.box[1])),
                          (int(obj.box[2]), int(obj.box[3])),
                          (0, 128, 255),
                          5)


@run_pipeline.command()
@click.argument('pdf_path')
@click.argument('output_path')
def full(pdf_path, output_path):
    logger.info("Started processing of file %s", pdf_path)
    input_pdf = Path(pdf_path)
    out_path = Path(output_path)
    images_path = convert_pdf_to_images(input_pdf, out_path)
    poppler_pages = extract_text(input_pdf)
    inference_result = inference_batch(str(images_path.absolute()), str(out_path.joinpath(input_pdf.name).absolute()),
                                       'models/epoch_41_acc_94_mmd_v2.pth', 'models/cascadetabnet_config.py',
                                       DEFAULT_THRESHOLD, None)
    # TODO: Serialize to Json method
    doc = {
        'document': {
            'path': pdf_path,
            'pages': []
        }
    }
    pages = []
    for img in list(images_path.iterdir()):
        if not has_image_extension(img):
            print(f'Not image {img}')
            continue
        logger.info("Extracting tables: %s", str(img.absolute()))
        # TODO: ImageService
        cv_image = cv2.imread(str(img.absolute()))
        page_image = cv_image.copy()
        draw_inference(cv_image, inference_result[img.name])

        # TODO: move to poppler service
        poppler_page = poppler_pages[img.name.split(".")[0]]
        scale = page_image.shape[0] / poppler_page.bbox.height
        text_fields = [poppler_text_field_to_text_field(text_field, scale) for text_field in poppler_page.text_fields]
        draw_text_boxes(cv_image, text_fields)
        Path(f"{output_path}/{str(input_pdf.name)}/tagg/").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(f"{output_path}/{str(input_pdf.name)}/tagg/{img.name}", cv_image)

        json_img = {
            "image_path": str(img.absolute()),
            "page_num": poppler_page.page_num,
            "blocks": []
        }
        doc['document']['pages'].append(json_img)

        has_bordered = any([i_tab.label == 'Bordered' for i_tab in inference_result[img.name]])

        page = Page()
        pages.append(page)
        text_fields_to_match = text_fields

        inf_tables = []
        for inf_table in inference_result[img.name]:
            if inf_table.paddler:
                logger.info("In paddler")
                in_inf_table = [TextField(bbox=cell, text='') for cell in inf_table.paddler]
            else:
                in_inf_table, text_fields_to_match = match_table_text(inf_table, text_fields_to_match)
            mask_rcnn_count_matches, not_matched = match_cells_text_fields(inf_table.tags, in_inf_table)

            if inf_table.label == 'Borderless':
                res = semi_bordered(page_image, inf_table)
                if res:
                    semi_border, header_cell = res
                    for row in semi_border.rows:
                        for cell in row.objs:
                            cell.top_left_x = cell.top_left_x - 5
                            cell.top_left_y = cell.top_left_y - 5
                            cell.bottom_right_x = cell.bottom_right_x + 5
                            cell.bottom_right_y = cell.bottom_right_y + 5
                    cells_count = 0
                    for row in semi_border.rows:
                        cells_count += len(row.objs)
                    if semi_border.header:
                        cells_count += len(semi_border.header)

                    semi_border_score = match_cells_table(in_inf_table, semi_border)
                    if semi_border_score >= mask_rcnn_count_matches and cells_count > len(inf_table.tags):
                        inf_tables.append((semi_border_score, semi_border))
                    else:
                        inf_tables.append((mask_rcnn_count_matches,
                                           extract_table_from_inference(inf_table, header_cell, not_matched)))
                else:
                    inf_tables.append(
                        (mask_rcnn_count_matches, extract_table_from_inference(inf_table, None, not_matched)))
            else:
                inf_tables.append((mask_rcnn_count_matches, extract_table_from_inference(inf_table, None, not_matched)))

        if has_bordered or any(score < 5 for score, _ in inf_tables):
            image = detect_tables_on_page(img, inference_result[img.name], draw=True)
            if image.tables:
                bordered_tables = []
                for table in image.tables:
                    rows = sorted(table.rows, key=lambda x: x.objs[0].top_left_y)
                    bordered_tables.append(TableHeadered(bbox=table.bbox,
                                                         table_id=0,
                                                         cols=table.cols,
                                                         rows=rows[1:],
                                                         header=rows[0].objs))
                text_fields_to_match = text_fields
                for bordered_table in bordered_tables:
                    matched = False
                    for score, inf_table in inf_tables:
                        if inf_table.bbox.box_is_inside_another(bordered_table.bbox):
                            in_table, text_fields_to_match = match_table_text(bordered_table, text_fields_to_match)
                            bordered_score = match_cells_table(in_table, bordered_table)

                            cells_count = 0
                            for row in bordered_table.rows:
                                cells_count += len(row.objs)
                            if bordered_table.header:
                                cells_count += len(bordered_table.header)

                            inf_cells_count = 0
                            for row in inf_table.rows:
                                inf_cells_count += len(row.objs)
                            if inf_table.header:
                                inf_cells_count += len(inf_table.header)

                            if bordered_score > score or bordered_score == score and cells_count >= inf_cells_count:
                                page.tables.append(bordered_table)
                            else:
                                page.tables.append(inf_table)
                            inf_tables.remove((score, inf_table))
                            matched = True
                            break
                    if not matched:
                        in_table, text_fields_to_match = match_table_text(bordered_table, text_fields_to_match)
                        _ = match_cells_table(in_table, bordered_table)
                        page.tables.append(bordered_table)
                if inf_tables:
                    page.tables.extend([inf_table for _, inf_table in inf_tables])
        else:
            page.tables.extend([tab for _, tab in inf_tables])

        count_object = 0
        for tab in page.tables:
            for row in tab.rows:
                count_object += len(row.objs)

        text_fields_to_match = text_fields
        count_in_table = 0
        for table in page.tables:
            in_table, text_fields_to_match = match_table_text(table, text_fields_to_match)
            count_in_table += len(in_table)
            table_obj = [row.objs for row in table.rows]
            # FIXME: dirty hacks should be fixed
            objs = []
            if hasattr(table, 'header') and table.header:
                objs.extend(table.header)
            for ls in table_obj:
                objs.extend(ls)
            _ = match_cells_text_fields(objs, in_table)
        if count_in_table < count_object // 10:
            with TextExtractor(str(img.absolute()), seg_mode=PSM.SPARSE_TEXT) as extractor:
                for table in page.tables:
                    for row in table.rows:
                        for cell in row.objs:
                            text, _ = extractor.extract(
                                cell.top_left_x, cell.top_left_y,
                                cell.width, cell.height
                            )
                            if text:
                                cell.text_boxes = []
                                cell.text_boxes.append(TextField(cell, text))

        for tbl in page.tables:
            actualize_header(tbl)

        with TextExtractor(str(img.absolute()), seg_mode=PSM.SPARSE_TEXT) as extractor:
            text_borders = [1]
            for table in page.tables:
                _, y, _, y2 = table.bbox.box
                text_borders.extend([y, y2])
            text_borders.append(page_image.shape[0])
            text_candidate_boxes: List[BorderBox] = []
            for i in range(len(text_borders) // 2):
                if text_borders[i * 2 + 1] - text_borders[i * 2] > 3:
                    text_candidate_boxes.append(
                        BorderBox(
                            top_left_x=1,
                            top_left_y=text_borders[i * 2],
                            bottom_right_x=page_image.shape[1],
                            bottom_right_y=text_borders[i * 2 + 1],
                        )
                    )
            for box in text_candidate_boxes:
                text, _ = extractor.extract(
                    box.top_left_x, box.top_left_y,
                    box.width, box.height
                )
                if text:
                    page.text.append(TextField(box, text))

        blocks = []
        blocks.extend(page.tables)
        blocks.extend(page.text)

        for block in sorted(blocks, key=lambda b: b.bbox.top_left_y):
            json_img['blocks'].append(block_to_json(block))

        result = page_image.copy()
        for r_table in page.tables:
            draw_table(result, r_table)

        for idx, block in enumerate(sorted(blocks, key=lambda b: b.bbox.top_left_y)):
            if isinstance(block, Table):
                cells = []
                for row in block.rows:
                    for cell in row.objs:
                        cells.append({
                            "pos": [
                                cell.top_left_x,
                                cell.bottom_right_x,
                                cell.top_left_y,
                                cell.bottom_right_y
                            ],
                            "text": " ".join([text_box.text for text_box in sorted(cell.text_boxes, key=lambda x: (x.bbox.top_left_y, x.bbox.top_left_x))])
                        })
                if hasattr(block, 'header') and block.header:
                    for cell in block.header:
                        cells.append({
                            "pos": [
                                cell.top_left_x,
                                cell.bottom_right_x,
                                cell.top_left_y,
                                cell.bottom_right_y
                            ],
                            "text": " ".join([text_box.text for text_box in sorted(cell.text_boxes, key=lambda x: (x.bbox.top_left_y, x.bbox.top_left_x))])
                        })
                Path(f"{output_path}/{str(input_pdf.name)}/{img.name.split('.')[0]}/").mkdir(parents=True, exist_ok=True)
                with open(f"{output_path}/{str(input_pdf.name)}/{img.name.split('.')[0]}/{idx}_json_cells.json", "w") as f:
                    json.dump({"chunks": cells}, f, indent=4)

        Path(f"{output_path}/{str(input_pdf.name)}/tables/").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(f"{output_path}/{str(input_pdf.name)}/tables/{img.name}", result)

    Path(f"{output_path}/{str(input_pdf.name)}/").mkdir(parents=True, exist_ok=True)
    with open(f"{output_path}/{str(input_pdf.name)}/json_out.json", "w") as f:
        json.dump(doc, f, indent=4)


def match_cells_table(in_inf_table, headered_table):
    table_obj = [row.objs for row in headered_table.rows]
    # FIXME: dirty hacks should be fixed
    objs = []
    if hasattr(headered_table, 'header') and headered_table.header:
        objs.extend(headered_table.header)
    for ls in table_obj:
        objs.extend(ls)
    semi_border_score, not_matched = match_cells_text_fields(objs, in_inf_table)
    return semi_border_score


if __name__ == '__main__':
    configure_logging()
    run_pipeline()
    # full('/home/ilia/Downloads/sp_3.pdf', '/home/ilia/test_header_alg/')
