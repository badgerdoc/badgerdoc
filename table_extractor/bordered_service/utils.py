import cv2

from .models import Image

# BGR color scheme
COL_COLOR = (0, 255, 0)
ROW_COLOR = (0, 0, 255)
TABLE_COLOR = (255, 0, 0)


def draw_cols_and_rows(image: Image):
    mask = cv2.imread(str(image.path.absolute()))
    if image.tables is None:
        return
    for table in image.tables:
        for col in table.cols:
            cv2.rectangle(
                mask,
                (col.bbox[0], col.bbox[1]),
                (col.bbox[2], col.bbox[3]),
                COL_COLOR,
                2,
            )

        for row in table.rows:
            cv2.rectangle(
                mask,
                (row.bbox[0], row.bbox[1]),
                (row.bbox[2], row.bbox[3]),
                ROW_COLOR,
                2,
            )
        cv2.rectangle(
            mask,
            (table.bbox[0], table.bbox[1]),
            (table.bbox[2], table.bbox[3]),
            TABLE_COLOR,
            2,
        )
    res_path = image.path.parent.parent / "detected_structure"
    res_path.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str((res_path / image.path.name).absolute()), mask)
