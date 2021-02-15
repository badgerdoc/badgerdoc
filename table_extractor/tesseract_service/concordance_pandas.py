import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
import pandas as pd
from datetime import datetime

save_path = '../model'
stemmer = WordNetLemmatizer()
stoplist = stopwords.words('english')


def read_from_file(file):
    with open(file, 'r') as f:
        cells = f.readlines()
    return [cell.replace('\n', ' ').replace('  ', ' ') for cell in cells]


def translate(word) -> str:
    return word.translate(str.maketrans('', '', string.punctuation))


def count_probability(df: pd.DataFrame) -> pd.DataFrame:
    # Create DataFrame consisting of words
    word_sequence = df['words'].map(
        lambda x: word_tokenize(x.lower().strip())).explode()

    # Remove NaN
    word_sequence = word_sequence[word_sequence.astype(bool)]
    # Lemmatize
    word_sequence = word_sequence.apply(lambda x: stemmer.lemmatize(x))

    final_df = word_sequence.to_frame()
    # Calculate word's count and probability
    final_df['counts'] = final_df['words']\
        .map(final_df['words'].value_counts())
    final_df['probability'] = final_df['words']\
        .map(final_df['words'].value_counts(normalize=True))

    # final_df.reset_index(drop=True, inplace=True)
    # Drop stop list words
    final_df = final_df[~final_df['words'].isin(stoplist)]
    final_df.set_index('words', inplace=True)

    # TODO: change count logic
    final_df = final_df[~final_df.index.duplicated()]
    return final_df


def run_lemmatizer(name):
    """
    @param name: name of the cell type
    """

    sequence = read_from_file(f'{name}.txt')
    df = pd.DataFrame(sequence, columns=['words'])
    df = count_probability(df)
    df.rename_axis(index={'words': f'{name}'}, inplace=True)
    df.to_json(f'{save_path}/{name}.json', orient='index')
    return df


start = datetime.now()
dfh = run_lemmatizer('headers')
# Drop digits in headers
# dfh = dfh[~dfh['words'].str.isdigit()]
dfc = run_lemmatizer('cells')

print(datetime.now() - start)
