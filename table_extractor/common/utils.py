from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

IMG_EXTENSIONS = ("png", "jpg", "jpeg", "bmp")


def has_image_extension(path: Path, allowed_extensions=IMG_EXTENSIONS) -> bool:
    logger.debug(f'Checking if {path} is an image...')

    if not path:
        return False

    ext = path.suffix[1:]
    return ext.lower() in allowed_extensions
