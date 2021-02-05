import json
import string
from typing import Tuple, List
from pathlib import Path

from table_extractor.model.table import CellLinked


class HeaderChecker:
    def __init__(self, cell_dictionary_path: Path = Path(__file__).parent.parent.parent.joinpath("models/cell.json"),
                 header_dictionary_path: Path = Path(__file__).parent.parent.parent.joinpath("models/header.json")):
        self.cell_path = cell_dictionary_path
        self.header_path = header_dictionary_path

        with open(self.cell_path, 'r') as f:
            self.cell_words = json.load(f)

        with open(self.header_path, 'r') as f:
            self.header_words = json.load(f)

    def check(self, text: str) -> Tuple:
        header_probability = 0
        cell_probability = 0

        for word in text.lower().split():
            word = word.translate(str.maketrans('', '', string.punctuation)).strip()
            if word in self.header_words.keys():
                header_probability += self.header_words[word]['probability']
            if word in self.cell_words.keys():
                cell_probability += self.cell_words[word]['probability']

        return round(header_probability, 4), round(cell_probability, 4)

    def get_cell_scores(self, cells: List[CellLinked]):
        score_cells = []
        for cell in cells:
            header_score, cell_score = self.get_cell_score(cell)
            score_cells.append((cell, header_score, cell_score))
        return score_cells

    def get_cell_score(self, cell: CellLinked):
        sentence = " ".join([text_field.text for text_field in cell.text_boxes])
        header_score, non_header_score = self.check(sentence)
        return header_score, non_header_score
