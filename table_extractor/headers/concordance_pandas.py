import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import words, stopwords
from nltk import word_tokenize
import pandas as pd
from datetime import datetime
import pathlib
import os
import json
import regex

reg = regex.compile("[\p{S}\p{P}]+")
save_path = os.path.dirname(os.path.abspath(__file__))

stemmer = WordNetLemmatizer()
word_list = set(words.words())
stoplist = set(stopwords.words('english'))
# stoplist.union({'(', ')'})


def read_from_file(file):
    with open(file, 'r') as f:
        cells = f.readlines()
    return [cell.replace('\n', ' ').replace('  ', ' ') for cell in cells]


def translate(word) -> str:
    return word.translate(str.maketrans('', '', string.punctuation)).strip()


def lemmatize_or_split(word: str) -> str or list:
    """
    Return lemmatized word if word is in words dictionary
    Else
    If word has punctuation letters, return punctuation letters and word
    @param word:
    @return:
    """

    reg.match(string.punctuation).group(0)
    punctuation = reg.match(word)

    if word not in word_list and punctuation:
        lst = list(punctuation.group(0))
        lst.append(stemmer.lemmatize(translate(word)))
        return lst
    return stemmer.lemmatize(word)

    # if word in word_list:
    #     return word
    # else:
    #     lst = list(word)
    #     lst.append(translate(word))
    #     return lst


def count_probability(df: pd.DataFrame) -> pd.DataFrame:
    # Create DataFrame consisting of words
    word_sequence = df['words'].map(
        lambda x: word_tokenize(x.lower().strip())).explode()

    # Remove NaN
    word_sequence = word_sequence[word_sequence.astype(bool)]
    # Lemmatize
    word_sequence = word_sequence.apply(lemmatize_or_split).explode()
    final_df = word_sequence.to_frame()
    # Calculate word's count and probability
    final_df['counts'] = final_df['words']\
        .map(final_df['words'].value_counts())
    final_df['probability'] = final_df['words']\
        .map(final_df['words'].value_counts(normalize=True))

    # final_df.reset_index(drop=True, inplace=True)
    # Drop stop list words
    final_df = final_df[~final_df['words'].isin(stoplist)]

    return final_df


def count_number_probability(df):
    # TODO: float and int probability
    numbers = df[df['words'].str.isdigit()]
    return df['probability'].mean() * (len(numbers) / len(df))


def run_lemmatizer(name):
    """
    @param name: name of the cell type
    """

    sequence = read_from_file(f'{name}.txt')
    df = pd.DataFrame(sequence, columns=['words'])
    df = count_probability(df)
    number_probability = count_number_probability(df)
    df.set_index('words', inplace=True)

    df = df[~df.index.duplicated()]
    df.rename_axis(index={'words': f'{name}'}, inplace=True)
    main_dict = df.to_dict(orient='index')
    main_dict = {'data': main_dict, 'number_probability': number_probability}
    with open(f'{save_path}/{name}.json', 'w') as f:
        f.write(json.dumps(main_dict))
    return df


if __name__ == '__main__':
    start = datetime.now()
    dfh = run_lemmatizer('headers')
    # Drop digits in headers
    # dfh = dfh[~dfh['words'].str.isdigit()]
    dfc = run_lemmatizer('cells')
    # TODO: number probability

    print(datetime.now() - start)
