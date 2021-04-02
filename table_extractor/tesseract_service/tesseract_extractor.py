from typing import Tuple

from tesserocr import PSM, PyTessBaseAPI


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

    def extract(self, x=None, y=None, w=None, h=None) -> Tuple:
        if all([e is not None for e in [x, y, w, h]]):
            return self._extract_from_rect(x, y, w, h)
        else:
            return self._extract()

    def extract_region(self, x, y, w, h) -> Tuple:
        self.api.SetRectangle(x, y, w, h)
        text = self.api.GetUTF8Text()
        conf = self.api.MeanTextConf()
        regions = self.api.GetRegions()
        return text, conf, regions

    def close(self):
        self.api.End()
