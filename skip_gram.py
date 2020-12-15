# need a pretrain model
# maybe the code is on tf1
# https://radimrehurek.com/gensim/models/word2vec.html
#import tensorflow as tf ;
from gensim.models import Word2Vec ;
from gensim.models import KeyedVectors ;
from gensim.models import Phrases ;
import gensim.downloader as dataset_list
import numpy ;
from gensim import utils
import os ;


#pretrained_model = dataset_list.load('glove-twitter-25')
#not an full model
# just an 2d array, so wv[] is allowed
# cosine similarity
#print(pretrained_vectors.most_similar('minecraft')) ;



#forward_text = numpy.load('') ;

class MyCorpus:
    """An iterator that yields sentences (lists of str)."""

    def __iter__(self):
        corpus_fileplace = os.getcwd()+'\\Sentence_and_Word_Parsing' ;
        for file_name in os.listdir(corpus_fileplace) :
            #print(file_name) ;
            if file_name == 'desktop.ini' : break ;
            files = numpy.load(corpus_fileplace+'\\'+file_name,allow_pickle=True) ;
            #print(files[0][2])
            combine = [] ;
            for senetece in files[0][2] :
                #print(senetece)
                for word_len in senetece :
                    #print(type(word_list))
                    #print(word_list)
                    combine += word_len[0]
                    combine += " " ;
                print(combine)
                #yield " ".join(combine) ;
                yield combine ;

model = Word2Vec(sentences=MyCorpus() ,size= 100, window= 5 ,min_count= 1, workers= 8)
# it can access iterator !!!
print("I'm fine") ;
'''
for sentence in iter(MyCorpus()) :
    combine = [] ;
    print(sentence) ;
    for word_list in sentence :
        combine += word_list ;
    model.build_vocab(" ".join(combine),update=False) ;
    model.train(" ".join(combine)) ;
'''
'''
model.save("w2v.model")
model = Word2Vec.load("w2v.model") ;
'''

'''
model.train([[test,foo,bar]], total_examples= 1, epochs= 1) # put our dataset there
#The trained word vectors are stored in a KeyedVectors instance
#that can make it faster (smaller)
vector = model.wv["computer"] ;
word_vectors = model.mv ;
print(model.mv) ;
word_vectors.save("w2v.wordvectors",mmap="r") ;
vector = wv["computer"] ;
'''
