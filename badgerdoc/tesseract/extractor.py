from typing import Tuple

from tesserocr import PyTessBaseAPI, PSM


class TextExtractor:
    def __init__(self, image_path, seg_mode=PSM.SPARSE_TEXT):
        self.api = PyTessBaseAPI()
        self.api.SetPageSegMode(seg_mode)
        self.api.SetImageFile(image_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _extract(self) -> Tuple:
        text = self.api.GetUTF8Text()
        conf = self.api.MeanTextConf()
        return text, conf

    def _extract_from_rect(self, x, y, w, h) -> Tuple:
        self.api.SetRectangle(x, y, w, h)
        return self._extract()

    #TODO: Add support of zero values
    def extract(self, x=None, y=None, w=None, h=None) -> Tuple:
        if all([x, y, w, h]):
            return self._extract_from_rect(x, y, w, h)
        else:
            return self._extract()

    def close(self):
        self.api.End()
