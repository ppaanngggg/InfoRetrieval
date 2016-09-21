import os
import cPickle
from gensim.utils import tokenize
from bs4 import BeautifulSoup
import math
import spacy


def html2txt():
    nlp = spacy.load('en')
    for filename in os.listdir('./Reuters/'):
        if filename.endswith('.html'):
            soup = BeautifulSoup(open('./Reuters/' + filename), 'html.parser')
            text = soup.get_text()
            word_list = []
            for token in nlp(text):
                if token.is_alpha:
                    word_list.append(str(token.lemma_))
            text = ' '.join(word_list)
            f = open('./Plain/' + filename.split('.')[0] + '.txt', 'w')
            f.write(text)
            f.close()


def get_dicts():
    word_dict = {}
    doc_dict = {}
    for filename in os.listdir('./Plain/'):
        if filename.endswith('.txt'):
            doc = int(filename.split('.')[0])
            f = open('./Plain/' + filename)
            text = f.read().strip()
            doc_dict[doc] = {}
            for word in text.split(' '):
                if word not in word_dict.keys():
                    word_dict[word] = set()
                word_dict[word].add(doc)
                if word not in doc_dict[doc].keys():
                    doc_dict[doc][word] = 0
                doc_dict[doc][word] += 1
    cPickle.dump(word_dict, open('./Database/word_dict', 'w'))
    cPickle.dump(doc_dict, open('./Database/doc_dict', 'w'))


def get_tf_idf():
    word_dict = cPickle.load(open('./Database/word_dict'))
    doc_dict = cPickle.load(open('./Database/doc_dict'))
    N = len(doc_dict)
    wf_dict = {}
    idf_dict = {}
    norm_dict = {}

    for word in word_dict.keys():
        idf_dict[word] = math.log10(N / len(word_dict[word]))

    for doc in doc_dict.keys():
        wf_dict[doc] = {}
        norm_dict[doc] = 0.
        for word in doc_dict[doc].keys():
            wf_dict[doc][word] = \
                (1 + math.log10(doc_dict[doc][word])) * idf_dict[word]
            norm_dict[doc] += wf_dict[doc][word] ** 2
        norm_dict[doc] = math.sqrt(norm_dict[doc])
    cPickle.dump(wf_dict, open('./Database/wf_dict', 'w'))
    cPickle.dump(idf_dict, open('./Database/idf_dict', 'w'))
    cPickle.dump(norm_dict, open('./Database/norm_dict', 'w'))

if __name__ == '__main__':
    # html2txt()
    # get_dicts()
    get_tf_idf()
