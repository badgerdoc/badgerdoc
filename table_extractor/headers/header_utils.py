import json
import os
import string
from pathlib import Path
from typing import List, Tuple

import nltk
import numpy as np
import regex
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer

from table_extractor.model.table import CellLinked

stemmer = WordNetLemmatizer()
stop_list = set(stopwords.words("english"))
word_list = set(words.words())
SPECIAL_CHARACTERS_REGEX = regex.compile("[%=]+")
NUMBER_REGEX = regex.compile("([0-9]+\\.[0-9]+)")
CELL_DICT = os.environ.get(
    'CELL_DICT',
    Path(__file__).parent.parent.parent.joinpath("language/cells_exc.json")
)
HEADER_DICT = os.environ.get(
    'HEADER_DICT',
    Path(__file__).parent.parent.parent.joinpath("language/headers_exc.json")
)


def softmax(array: Tuple[float]) -> List[float]:
    x = np.array(array)
    e_x = np.exp(x - np.max(x))
    return (e_x / e_x.sum()).tolist()


class HeaderChecker:
    def __init__(
        self,
        cell_dictionary_path: Path = Path(
            __file__
        ).parent.parent.parent.joinpath("language/cells.json"),
        header_dictionary_path: Path = Path(
            __file__
        ).parent.parent.parent.joinpath("language/headers.json"),
    ):
        self.cell_path = cell_dictionary_path
        self.header_path = header_dictionary_path

        with open(self.cell_path, "r") as f:
            cell_dict = json.load(f)
            self.cell_words = cell_dict["data"]
            self.cell_number_probability = cell_dict["number_probability"]

        with open(self.header_path, "r") as f:
            header_dict = json.load(f)
            self.header_words = header_dict["data"]
            self.header_number_probability = header_dict["number_probability"]

    def check(self, text: str) -> Tuple:
        header_probability = 0
        cell_probability = 0

        # TODO: check word_tokenize
        for word in nltk.word_tokenize(text.lower()):
            word = self._translate(word)
            word = stemmer.lemmatize(word)

            spc_list = SPECIAL_CHARACTERS_REGEX.findall(word)
            if spc_list:
                header_word_prob, cell_word_prob = (
                    sum(i) for i in zip(self._check_punctuation(word))
                )
                header_probability += header_word_prob
                cell_probability += cell_word_prob

            header_word_prob, cell_word_prob = self._count_probability(word)
            header_probability += header_word_prob
            cell_probability += cell_word_prob

        header_probability, cell_probability = softmax(
            (header_probability, cell_probability)
        )
        if NUMBER_REGEX.findall(text):
            weight = len("".join(NUMBER_REGEX.findall(text))) / len(text)
            h_n_prob, c_n_prob = softmax(
                (self.header_number_probability, self.cell_number_probability)
            )
            header_probability += h_n_prob * weight
            cell_probability += c_n_prob * weight

        return softmax(
            (round(header_probability, 6), round(cell_probability, 6))
        )

    def get_cell_scores(self, cells: List[CellLinked]):
        score_cells = []
        for cell in cells:
            header_score, cell_score = self.get_cell_score(cell)
            score_cells.append((cell, header_score, cell_score))
        return score_cells

    def get_cell_score(self, cell: CellLinked):
        sentence = " ".join(
            [text_field.text for text_field in cell.text_boxes]
        )
        # Sets high probability to cell if it is number

        # Check if string is a number
        try:
            float(sentence)
            return self.header_number_probability, self.cell_number_probability
        except ValueError:
            return self.check(sentence)

    def _count_probability(self, word):
        header_probability = 0
        cell_probability = 0

        if word in stop_list:
            return 0, 0
        if word in self.header_words.keys():
            header_probability += self.header_words[word]["probability"]
        if word in self.cell_words.keys():
            cell_probability += self.cell_words[word]["probability"]
        return header_probability, cell_probability

    def _check_punctuation(self, spc_list):
        return (self._count_probability(i) for i in spc_list)

    @staticmethod
    def _translate(word) -> str:
        return word.translate(
            str.maketrans("", "", string.punctuation)
        ).strip()
