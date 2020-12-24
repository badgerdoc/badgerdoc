import logging
import subprocess
from pathlib import Path
from multiprocessing.pool import ThreadPool

logger = logging.getLogger(__name__)
TESSERACT_COMMAND_TEMPLATE = "tesseract {input_path} {output_path} -l {lang} {format}"


def execute_shell(command):
    def __log_lines(pipe, logging_level):
        res = []
        for line in iter(pipe.readline, b''):
            res.append(line)
            logger.log(logging_level, line.rstrip())
        return b''.join(res).rstrip()

    pool = ThreadPool(processes=2)
    pipes = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    err = pool.apply_async(__log_lines, args=(pipes.stderr, logging.getLevelName("ERROR")))
    out = pool.apply_async(__log_lines, args=(pipes.stdout, logging.getLevelName("INFO")))

    err.wait()
    out.wait()
    pipes.communicate()

    if pipes.returncode != 0:
        raise RuntimeError(err.get())

    return out.get(), err.get()


def ocr_page(page_path: Path, output_path: Path, lang='eng', format='hocr'):
    command = ["tesseract", str(page_path.absolute()), str(output_path.absolute()), "-l", lang, format]
    logger.info("Started command %s", " ".join(command))

    execute_shell(command)


def ocr_pages_in_path(input_dir: Path, output_path: Path):
    logger.info("Started processing path: %s", input_dir.absolute())
    out_ocr = output_path.joinpath('ocr')
    out_ocr.mkdir(exist_ok=True)
    for file in input_dir.glob("*.png"):
        ocr_page(file, out_ocr.joinpath(file.name))
    logger.info("Finished processing path^ %s", input_dir.absolute())

