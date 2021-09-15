import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
from tesserocr import PSM

from table_extractor.bordered_service.bordered_tables_detection import (
    detect_tables_on_page,
)
from table_extractor.bordered_service.models import InferenceTable, Page
from table_extractor.borderless_service.semi_bordered import semi_bordered
from table_extractor.cascade_rcnn_service.inference import (
    CascadeRCNNInferenceService,
)
from table_extractor.headers.header_utils import HeaderChecker
from table_extractor.inference_table_service.constuct_table_from_inference import (
    construct_table_from_cells,
    find_grid_table,
    reconstruct_table_from_grid,
)
from table_extractor.model.table import (
    BorderBox,
    Cell,
    CellLinked,
    StructuredTable,
    StructuredTableHeadered,
    Table,
    TextField,
)
from table_extractor.pdf_service.pdf_to_image import convert_pdf_to_images
from table_extractor.poppler_service.poppler_text_extractor import (
    PopplerPage,
    extract_text,
    poppler_text_field_to_text_field,
)
from table_extractor.tesseract_service.tesseract_extractor import TextExtractor
from table_extractor.text_cells_matcher.text_cells_matcher import (
    match_cells_text_fields,
    match_table_text,
)
from table_extractor.visualization.table_visualizer import TableVisualizer

logger = logging.getLogger(__name__)


def cnt_ciphers(cells: List[Cell]):
    count = 0
    all_chars_count = 0
    for cell in cells:
        sentence = "".join([tb.text for tb in cell.text_boxes])
        for char in sentence:
            if char in "0123456789":
                count += 1
        all_chars_count += len(sentence)
    return count / all_chars_count if all_chars_count else 0.0


def actualize_header(table: StructuredTable):
    table_rows = table.rows
    count_ciphers = cnt_ciphers(table_rows[0])
    header_candidates = [table_rows[0]]
    current_ciphers = count_ciphers
    for row in table_rows[1:]:
        count = cnt_ciphers(row)
        if count > current_ciphers:
            break
        else:
            header_candidates.append(row)

    if len(header_candidates) < len(table_rows):
        return StructuredTableHeadered.from_structured_and_rows(
            table, header_candidates
        )
    return StructuredTableHeadered(
        bbox=table.bbox, cells=table.cells, header=[]
    )


def merge_text_fields(
    paddle_t_b: List[TextField], poppler_t_b: List[TextField]
) -> List[TextField]:
    not_matched = []
    merged_t_b = []
    for pop_t_b in poppler_t_b:
        merged = False
        for pad_t_b in paddle_t_b:
            if pop_t_b.bbox.box_is_inside_another(
                pad_t_b.bbox, threshold=0.00
            ):
                merged_t_b.append(
                    TextField(
                        bbox=pad_t_b.bbox.merge(pop_t_b.bbox),
                        text=pop_t_b.text,
                    )
                )
                merged = True
        if not merged:
            not_matched.append(pop_t_b)

    for pad_t_b in paddle_t_b:
        exists = False
        for mer_t_b in merged_t_b:
            if mer_t_b.bbox.box_is_inside_another(pad_t_b.bbox, threshold=0.0):
                exists = True
        if not exists:
            not_matched.append(pad_t_b)

    merged_t_b.extend(not_matched)

    return merged_t_b


def text_to_cell(text_field: TextField):
    return Cell(
        top_left_x=text_field.bbox.top_left_x,
        top_left_y=text_field.bbox.top_left_y,
        bottom_right_x=text_field.bbox.bottom_right_x,
        bottom_right_y=text_field.bbox.bottom_right_y,
        text_boxes=[text_field],
    )


def merge_closest_text_fields(text_fields: List[TextField]):
    merged_fields: List[TextField] = []
    curr_field: TextField = None
    for text_field in sorted(
        text_fields, key=lambda x: (x.bbox.top_left_y, x.bbox.top_left_x)
    ):
        if not curr_field:
            curr_field = text_field
            continue
        if curr_field:
            if (
                20
                > text_field.bbox.top_left_x - curr_field.bbox.bottom_right_x
                > -20
                and curr_field.bbox.top_left_y - 10
                < text_field.bbox.top_left_y
                < curr_field.bbox.top_left_y + 10
            ):
                curr_field = TextField(
                    bbox=curr_field.bbox.merge(text_field.bbox),
                    text=curr_field.text + " " + text_field.text,
                )
            else:
                merged_fields.append(curr_field)
                curr_field = text_field
    if curr_field:
        merged_fields.append(curr_field)

    return merged_fields


def pdf_preprocess(
    pdf_path: Path, output_path: Path
) -> Tuple[Path, Dict[str, PopplerPage], Path]:
    images_path = convert_pdf_to_images(pdf_path, output_path)
    images_path_400 = convert_pdf_to_images(pdf_path,
                                            output_path / pdf_path.name / 'images_400',
                                            already_incl=True,
                                            dpi=400)
    poppler_pages = extract_text(pdf_path)
    return images_path, poppler_pages, images_path_400


def actualize_text(table: StructuredTable, image_path_400: Path, img_shape):
    img_400 = cv2.imread(str(image_path_400.absolute()))
    x_scale = img_400.shape[1] / img_shape[1]
    y_scale = img_400.shape[0] / img_shape[0]
    cells = table.all_cells if isinstance(table, StructuredTableHeadered) else table.cells
    with TextExtractor(str(image_path_400.absolute())) as te:
        for cell in cells:
            if not cell.text_boxes or any(
                [not text_box.text for text_box in cell.text_boxes]
            ):
                top_left_x = max(0, cell.top_left_x)
                top_left_y = max(0, cell.top_left_y)
                bottom_right_x = min(img_shape[1], cell.bottom_right_x)
                bottom_right_y = min(img_shape[0], cell.bottom_right_y)
                text, _ = te.extract(
                    int(top_left_x * x_scale) + 4,
                    int(top_left_y * y_scale) + 4,
                    int((bottom_right_x - top_left_x) * x_scale) - 4,
                    int((bottom_right_y - top_left_y) * y_scale) - 4
                )
                cell.text_boxes.append(TextField(bbox=cell, text=text))


def semi_border_to_struct(
    semi_border: Table, image_shape: Tuple[int, int]
) -> StructuredTable:
    cells = []
    for row in semi_border.rows:
        cells.extend(row.objs)
    semi_border.bbox.top_left_y = semi_border.bbox.top_left_y - 20
    semi_border.bbox.top_left_x = semi_border.bbox.top_left_x - 20
    semi_border.bbox.bottom_right_y = semi_border.bbox.bottom_right_y + 20
    semi_border.bbox.bottom_right_x = semi_border.bbox.bottom_right_x + 20
    structured_table = construct_table_from_cells(
        semi_border.bbox, cells, image_shape
    )
    return structured_table


def bordered_to_struct(bordered_table: Table) -> StructuredTable:
    v_lines = []
    for col in bordered_table.cols:
        v_lines.extend([col.bbox.top_left_x, col.bbox.bottom_right_x])
    v_lines = sorted(v_lines)
    v_lines_merged = [v_lines[0], v_lines[-1]]
    for i in range(0, len(v_lines[1:-1]) // 2):
        v_lines_merged.append((v_lines[2 * i] + v_lines[2 * i + 1]) // 2)
    v_lines = sorted(list(set(v_lines_merged)))

    h_lines = []
    for row in bordered_table.rows:
        h_lines.extend([row.bbox.top_left_y, row.bbox.bottom_right_y])
    h_lines = sorted(h_lines)
    h_lines_merged = [h_lines[0], h_lines[-1]]
    for i in range(0, len(h_lines[1:-1]) // 2):
        h_lines_merged.append((h_lines[2 * i] + h_lines[2 * i + 1]) // 2)
    h_lines = sorted(list(set(h_lines_merged)))
    cells = []
    for row in bordered_table.rows:
        cells.extend(row.objs)
    grid = find_grid_table(h_lines, v_lines)
    table, _ = reconstruct_table_from_grid(grid, cells)
    return table


def cell_to_dict(cell: CellLinked):
    return {
        "row": cell.row,
        "column": cell.col,
        "rowspan": cell.row_span,
        "colspan": cell.col_span,
        "bbox": {
            "left": cell.top_left_x,
            "top": cell.top_left_y,
            "height": cell.height,
            "width": cell.width,
        },
        "text": " ".join(
            [
                field.text
                for field in sorted(
                    cell.text_boxes,
                    key=lambda x: (x.bbox.top_left_y, x.bbox.top_left_x),
                )
            ]
        ),
    }


def table_to_dict(table: StructuredTableHeadered):
    header = []
    for row in table.header:
        for cell in row:
            header.append(cell)
    return {
        "type": "table",
        "bbox": {
            "left": table.bbox.top_left_x,
            "top": table.bbox.top_left_y,
            "height": table.bbox.height,
            "width": table.bbox.width,
        },
        "header": [cell_to_dict(cell) for cell in header],
        "cells": [cell_to_dict(cell) for cell in table.cells],
    }


def text_to_dict(text: TextField):
    return {
        "type": "text_block",
        "bbox": {
            "left": text.bbox.top_left_x,
            "top": text.bbox.top_left_y,
            "height": text.bbox.height,
            "width": text.bbox.width,
        },
        "text": text.text,
    }


def block_to_dict(block: Union[TextField, StructuredTableHeadered]):
    if isinstance(block, StructuredTableHeadered):
        return table_to_dict(block)
    if isinstance(block, TextField):
        return text_to_dict(block)
    raise TypeError(f"Incorrect type provided: {type(block)}")


def page_to_dict(page: Page):
    blocks = page.blocks
    return {
        "page_num": page.page_num,
        "bbox": {
            "left": page.bbox.top_left_x,
            "top": page.bbox.top_left_y,
            "height": page.bbox.height,
            "width": page.bbox.width,
        },
        "blocks": [block_to_dict(block) for block in blocks],
    }


def save_page(page_dict: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path.absolute()), "w") as f:
        f.write(json.dumps(page_dict, indent=4))


def softmax(array: Tuple[float]) -> List[float]:
    x = np.array(array)
    e_x = np.exp(x - np.max(x))
    return (e_x / e_x.sum()).tolist()


def rematch_text(tables: List[StructuredTable], text_fields: List[TextField], image_path_400, img_shape):
    text_fields_to_match = text_fields
    for table in tables:
        in_table, text_fields_to_match = match_table_text(
            table, text_fields_to_match
        )
        _ = match_cells_text_fields(table.cells, in_table, threshold=0.5)
        actualize_text(table, image_path_400, img_shape)


class PageProcessor:
    def __init__(
        self,
        inference_service: CascadeRCNNInferenceService,
        visualizer: TableVisualizer,
        paddle_on=True,
    ):
        self.inference_service = inference_service
        self.visualizer = visualizer
        self.paddle_on = paddle_on
        self.header_checker = HeaderChecker()

    def cell_in_inf_header(
        self, cell: CellLinked, inf_headers: List[Cell]
    ) -> float:
        confidences = [0.0]
        for header in inf_headers:
            if cell.box_is_inside_another(header):
                confidences.append(header.confidence)
        return max(confidences)

    def analyse(self, series: List[CellLinked], inf_headers: List[Cell]):
        # Check if series is header
        headers = []
        first_line = False
        for cell in series:
            inf_header_score = self.cell_in_inf_header(cell, inf_headers)
            header_score, cell_score = softmax(
                self.header_checker.get_cell_score(cell)
            )
            header_score, cell_score = softmax(
                (header_score + inf_header_score, cell_score)
            )

            if header_score > cell_score:
                headers.append(cell)
            if cell.col == 0 and cell.row == 0:
                first_line = True

        if first_line:
            empty_cells_num = self._count_empty_cells(series)
            return len(headers) > (len(series) - empty_cells_num) / 2
        # return len(headers) > (len(series) / 5) if len(series) > thresh else len(headers) > (len(series) / 2)
        return len(headers) > (len(series) / 2)

    def create_header(
        self,
        series: List[List[CellLinked]],
        inf_headers: List[Cell],
        header_limit: int,
    ):
        """
        Search for headers based on cells contents
        @param series: cols or rows of the table
        @param table:
        @param header_limit:
        """
        header_candidates = []
        last_header = None
        for idx, line in enumerate(series[:header_limit]):
            if self.analyse(line, inf_headers):
                header_candidates.append((idx, True, line))
                last_header = idx
            else:
                header_candidates.append((idx, False, line))

        if last_header is not None:
            header = [
                line
                for idx, is_header, line in header_candidates[
                    : last_header + 1
                ]
            ]
        else:
            header = []

        if len(header) > 0.75 * len(series):
            with open("cases75.txt", "a") as f:
                f.write(str(series) + "\n")
            header = []

        return header

    @staticmethod
    def _count_empty_cells(series: List[CellLinked]):
        return len([True for cell in series if cell.is_empty()])

    def extract_table_from_inference(
        self,
        img,
        inf_table: InferenceTable,
        not_matched_text: List[TextField],
        image_shape: Tuple[int, int],
        image_path: Path,
    ) -> StructuredTable:
        merged_t_f = merge_closest_text_fields(not_matched_text)

        for cell in inf_table.tags:
            if cell.text_boxes:
                cell.top_left_x = max(cell.top_left_x, min(
                    [text_box.bbox.top_left_x for text_box in cell.text_boxes]
                ))
                cell.top_left_y = max(cell.top_left_y, min(
                    [text_box.bbox.top_left_y for text_box in cell.text_boxes]
                ))
                cell.bottom_right_x = min(cell.bottom_right_x, max(
                    [
                        text_box.bbox.bottom_right_x
                        for text_box in cell.text_boxes
                    ]
                ))
                cell.bottom_right_y = min(cell.bottom_right_y, max(
                    [
                        text_box.bbox.bottom_right_y
                        for text_box in cell.text_boxes
                    ]
                ))

        if inf_table.tags:
            inf_table.bbox.top_left_y = min(inf_table.bbox.top_left_y,
                                            min([cell.top_left_y for cell in inf_table.tags]),)
            inf_table.bbox.top_left_x = min(inf_table.bbox.top_left_x,
                                            min([cell.top_left_x for cell in inf_table.tags]),)
            inf_table.bbox.bottom_right_y = max(inf_table.bbox.bottom_right_y,
                                                max([cell.bottom_right_y for cell in inf_table.tags]),)
            inf_table.bbox.bottom_right_x = max(inf_table.bbox.bottom_right_x,
                                                max([cell.bottom_right_x for cell in inf_table.tags]),)

        self.visualizer.draw_object_and_save(
            img,
            [inf_table],
            image_path.parent.parent
            / "modified_cells"
            / f"{str(image_path.name).replace('.png', '')}_"
            f"{inf_table.bbox.top_left_x}_{inf_table.bbox.top_left_y}.png",
        )
        return construct_table_from_cells(
            inf_table.bbox, inf_table.tags, image_shape
        )

    def _scale_poppler_result(
        self, img, output_path, poppler_page, image_path
    ):
        scale = img.shape[0] / poppler_page.bbox.height
        text_fields = [
            poppler_text_field_to_text_field(text_field, scale)
            for text_field in poppler_page.text_fields
        ]
        if text_fields:
            self.visualizer.draw_object_and_save(
                img,
                text_fields,
                Path(f"{output_path}/poppler_text/{image_path.name}"),
            )
        return text_fields

    def process_pages(
        self, images_path: Path, poppler_pages: Dict[str, PopplerPage], images_path_400: Path
    ) -> List:
        pages = []
        for image_path, image_path_400 in zip(sorted(images_path.glob("*.png")), sorted(images_path_400.glob("*.png"))):
            try:
                pages.append(
                    self.process_page(
                        image_path,
                        images_path.parent,
                        poppler_pages[image_path.name.split(".")[0]],
                        image_path_400
                    )
                )
            except Exception as e:
                # ToDo: Rewrite, needed to not to fail pipeline for now in sequential mode
                logger.warning(str(e))
                raise e
        return pages

    def process_page(
        self, image_path: Path, output_path: Path, poppler_page, image_path_400: Path
    ) -> Dict[str, Any]:
        img = cv2.imread(str(image_path.absolute()))
        page = Page(
            page_num=int(image_path.name.split(".")[0]),
            bbox=BorderBox(
                top_left_x=0,
                top_left_y=0,
                bottom_right_x=img.shape[1],
                bottom_right_y=img.shape[0],
            ),
        )
        text_fields = self._scale_poppler_result(
            img, output_path, poppler_page, image_path
        )

        logger.info("Start inference")
        inference_tables, headers = self.inference_service.inference_image(
            image_path
        )
        logger.info("End inference")
        self.visualizer.draw_object_and_save(
            img,
            inference_tables,
            Path(f"{output_path}/inference_result/{image_path.name}"),
            headers=headers,
        )

        if inference_tables:
            logger.info("Start bordered")
            image = detect_tables_on_page(
                image_path, draw=self.visualizer.should_visualize
            )
            logger.info("End bordered")
            text_fields_to_match = text_fields.copy()
            bordered_tables = []
            if image.tables:
                for bordered_table in image.tables:
                    in_table, text_fields_to_match = match_table_text(
                        bordered_table, text_fields_to_match
                    )
                    _ = match_cells_table(in_table, bordered_table)
                    bordered_tables.append(
                        semi_border_to_struct(bordered_table, img.shape)
                    )

            inf_tables_to_detect = []
            for inf_table in inference_tables:
                matched = False
                if image.tables:
                    for bordered_table in bordered_tables:
                        if (
                            inf_table.bbox.box_is_inside_another(
                                bordered_table.bbox, 0.8
                            )
                            and len(bordered_table.cells)
                            > len(inf_table.tags) * 0.5
                        ):
                            matched = True
                            page.tables.append(bordered_table)
                if not matched:
                    inf_tables_to_detect.append(inf_table)

            semi_bordered_tables = []
            for inf_table in inf_tables_to_detect:
                in_inf_table, text_fields_to_match = match_table_text(
                    inf_table, text_fields_to_match
                )

                mask_rcnn_count_matches, not_matched = match_cells_text_fields(
                    inf_table.tags, in_inf_table
                )

                struct = self.extract_table_from_inference(
                    img, inf_table, not_matched, img.shape, image_path
                )
                if struct:
                    page.tables.append(struct)

            for table in page.tables:
                actualize_text(table, image_path_400, img.shape[:2])

            # TODO: Headers should be created only once
            cell_header_scores = []
            for table in page.tables:
                cell_header_scores.extend(
                    self.header_checker.get_cell_scores(table.cells)
                )

            self.visualizer.draw_object_and_save(
                img,
                cell_header_scores,
                output_path / "cells_header" / f"{page.page_num}.png",
            )

            rematch_text(page.tables, text_fields, image_path_400, img.shape[:2])

            tables_with_header = []
            for table in page.tables:
                header_rows = self.create_header(table.rows, headers, 5)
                table_with_header = (
                    StructuredTableHeadered.from_structured_and_rows(
                        table, header_rows
                    )
                )
                header_cols = self.create_header(table.cols, headers, 1)
                # TODO: Cells should be actualized only once
                table_with_header.actualize_header_with_cols(header_cols)
                tables_with_header.append(table_with_header)
            page.tables = tables_with_header

            self.visualizer.draw_object_and_save(
                img,
                semi_bordered_tables,
                output_path.joinpath("semi_bordered_tables").joinpath(
                    image_path.name
                ),
            )
            self.visualizer.draw_object_and_save(
                img,
                page.tables,
                output_path.joinpath("tables").joinpath(image_path.name),
            )
        logger.info("Start text extraction")
        img_400 = cv2.imread(str(image_path_400.absolute()))
        x_scale = img_400.shape[1] / img.shape[1]
        y_scale = img_400.shape[0] / img.shape[0]
        with TextExtractor(
            str(image_path_400.absolute()), seg_mode=PSM.SPARSE_TEXT
        ) as extractor:
            text_borders = [1]
            for table in page.tables:
                _, y, _, y2 = table.bbox.box
                text_borders.extend([int(y * y_scale), int(y2 * y_scale)])
            text_borders.append(img_400.shape[0])
            text_candidate_boxes: List[BorderBox] = []
            for i in range(len(text_borders) // 2):
                if text_borders[i * 2 + 1] - text_borders[i * 2] > 3:
                    text_candidate_boxes.append(
                        BorderBox(
                            top_left_x=1,
                            top_left_y=text_borders[i * 2],
                            bottom_right_x=img_400.shape[1],
                            bottom_right_y=text_borders[i * 2 + 1],
                        )
                    )
            for box in text_candidate_boxes:
                text, _ = extractor.extract(
                    box.top_left_x, box.top_left_y, box.width, box.height
                )
                bbox = BorderBox(
                    int(box.top_left_x / x_scale),
                    int(box.top_left_y / y_scale),
                    int(box.bottom_right_x / x_scale),
                    int(box.bottom_right_y / y_scale)
                )
                if text:
                    page.text.append(TextField(bbox, text))
        logger.info("End text extraction")
        page_dict = page_to_dict(page)
        if self.visualizer.should_visualize:
            save_page(
                page_dict, output_path / "pages" / f"{page.page_num}.json"
            )

        return page_dict


def match_cells_table(text_fields: List[TextField], table: Table) -> int:
    table_cells = [row.objs for row in table.rows]
    cells = []
    for cell in table_cells:
        cells.extend(cell)
    score, not_matched = match_cells_text_fields(cells, text_fields)
    return score
