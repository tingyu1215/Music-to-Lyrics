from gensim.test.utils import datapath
from gensim import utils
import gensim.downloader as dataset_list
import gensim.models

class MyCorpus:
    """An iterator that yields sentences (lists of str)."""

    def __iter__(self):
        corpus_path = datapath('lee_background.cor')
        for line in open(corpus_path):
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(line)


sentences = MyCorpus()
model = gensim.models.Word2Vec(sentences=sentences)

print(model.wv['king'])

print(type(dataset_list.load('glove-twitter-25')))

for index, word in enumerate(model.index_to_key):
    if index == 10:
        break
    print(f"word #{index}/{len(wv.index_to_key)} is {word}")