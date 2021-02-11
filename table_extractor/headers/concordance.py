import string
import re
import json
import statistics
from operator import itemgetter
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


def main():
    with open('cells.txt', 'r') as f:
        cells = f.readlines()
    
    with open('headers.txt', 'r') as f:
        headers = f.readlines()

    cells = [cell.replace('\n', ' ').replace('  ', ' ') for cell in cells]
    headers = [header.replace('\n', ' ').replace('  ', ' ') for header in headers]

    header_words = count_words(headers)
    cell_words = count_words(cells)

    with open('headers.json', 'w') as f:
        json.dump(header_words, f)

    with open('cells.json', 'w') as f:
        json.dump(cell_words, f)

def count_words(sentences):
    stemmer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    dictionary = list()
    concordance = dict()
    for sentence in sentences:
        for word in sentence.lower().split(' '):
            word = word.translate(str.maketrans('', '', string.punctuation)).strip()
            if word:
                try:
                    word = int(word)
                    dictionary.append('#')
                except:
                    word = re.sub("\d+", "#", word)
                    dictionary.append(word)
    dictionary = [stemmer.lemmatize(word) for word in dictionary]
    for word in dictionary:
        counts = dictionary.count(word)
        probability = counts / len(dictionary)
        if word not in stop_words and word not in concordance.keys():
            concordance[word] = {'counts': counts, 'probability': probability}
    return concordance


if __name__ == '__main__':
    main()
