import logging
import shutil
from pathlib import Path

from pdf2image import convert_from_path

logger = logging.getLogger(__name__)
DPI = 400


# ToDo: Implement also with pagination
def convert_pdf_to_images(
    pdf_file: Path, out_dir: Path, already_incl: bool = False
) -> Path:
    logger.info("Start pdf to png conversion for %s", str(pdf_file.name))
    out_dir = (
        out_dir.joinpath(Path(f"{pdf_file.name}/images/"))
        if not already_incl
        else out_dir / "images"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = convert_from_path(
        pdf_file,
        dpi=DPI,
        output_folder=str(out_dir.absolute()),
        paths_only=True,
        fmt="png",
    )
    for i, page in enumerate(pages):
        shutil.move(page, out_dir.absolute() / f"{i}.png")
    logger.info("Done  pdf to png conversion for %s", str(pdf_file.name))
    return out_dir
