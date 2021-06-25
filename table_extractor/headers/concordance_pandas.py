import json
import os
import string
from pathlib import Path

import pandas as pd
import regex
from nltk import word_tokenize
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer

reg = regex.compile("[\p{S}\p{P}]+")
save_path = os.path.dirname(os.path.abspath(__file__))

stemmer = WordNetLemmatizer()
word_list = set(words.words())
stoplist = set(stopwords.words("english"))


def read_from_file(file: Path):
    with open(str(file.absolute()), "r") as f:
        return f.readlines()


def translate(word) -> str:
    return word.translate(str.maketrans("", "", string.punctuation)).strip()


def lemmatize_or_split(word: str) -> str or list:
    """
    Return lemmatized word if word is in words dictionary
    Else
    If word has punctuation letters, return punctuation letters and word
    @param word:
    @return:
    """

    reg.match(string.punctuation).group(0)
    try:
        punctuation = reg.match(word)
    except Exception as e:
        print(word)

    if word not in word_list and punctuation:
        lst = list(punctuation.group(0))
        lst.append(stemmer.lemmatize(translate(word)))
        return lst
    return stemmer.lemmatize(word)


def count_probability(df: pd.DataFrame) -> pd.DataFrame:
    # Create DataFrame consisting of words
    word_sequence = (
        df["words"].map(lambda x: word_tokenize(x.lower().strip())).explode()
    )

    # Remove NaN
    word_sequence = word_sequence[word_sequence.astype(bool)]
    # Lemmatize
    word_sequence = word_sequence.apply(lemmatize_or_split).explode()
    final_df = word_sequence.to_frame()
    # Calculate word's count and probability
    final_df["counts"] = final_df["words"].map(
        final_df["words"].value_counts()
    )
    final_df["probability"] = final_df["words"].map(
        final_df["words"].value_counts(normalize=True)
    )

    final_df = final_df[~final_df["words"].isin(stoplist)]

    return final_df


def count_number_probability(df):
    # TODO: float and int probability
    numbers = df[df["words"].str.isdigit()]
    return df["probability"].mean() * (len(numbers) / len(df))


def process_words(name, sequence):
    sequence = [cell.replace("\n", " ").replace("  ", " ") for cell in sequence]
    df = pd.DataFrame(sequence, columns=["words"])
    df = count_probability(df)
    number_probability = count_number_probability(df)
    df.set_index("words", inplace=True)
    df = df[~df.index.duplicated()]
    df.rename_axis(index={"words": f"{name}"}, inplace=True)
    main_dict = df.to_dict(orient="index")
    main_dict = {"data": main_dict, "number_probability": number_probability}
    return main_dict


def run_lemmatizer(name, text_path: Path, out_path: Path):
    """
    @param name: name of the cell type
    """

    sequence = read_from_file(
        text_path
    )
    main_dict = process_words(name, sequence)
    with open(out_path, "w") as f:
        f.write(json.dumps(main_dict))
