import spacy
import cPickle
import math
from bs4 import BeautifulSoup
from nltk.corpus import wordnet

doc_dict = cPickle.load(open('./Database/doc_dict'))
idf_dict = cPickle.load(open('./Database/idf_dict'))
norm_dict = cPickle.load(open('./Database/norm_dict'))
wf_dict = cPickle.load(open('./Database/wf_dict'))
word_dict = cPickle.load(open('./Database/word_dict'))

nlp = spacy.load('en')

while 1:
    text = raw_input("Please input a query:")
    top_k = raw_input("Please input K:")
    top_k = int(top_k)
    # get the query and remove word less than threshold
    query = []
    for word in nlp(unicode(text)):
        word_str = str(word.lemma_)
        if word.is_alpha and word_str in word_dict.keys() and idf_dict[word_str] > 0.4:
            query.append(word_str)
    if len(query) == 0:
        for word in nlp(unicode(text)):
            word_str = str(word.lemma_)
            if word.is_alpha and word_str in word_dict.keys():
                query.append(word_str)
    new_query = []
    for word in query:
        word_set = set()
        word_set.add((word,))
        for s in wordnet.synsets(word):
            for w in s.lemma_names():
                tmp = []
                w = str(w).lower()
                for ww in w.split('_'):
                    if ww in word_dict.keys() and idf_dict[ww] > 0.4:
                        tmp.append(ww)
                if len(tmp):
                    word_set.add(tuple(tmp))
        new_query.append(word_set)
    old_query = query
    query = new_query
    print query

    query_doc_set = []
    for word_set in query:
        doc_set = set()
        for word_tuple in word_set:
            tmp = word_dict[word_tuple[0]]
            for s in word_tuple[1:]:
                tmp = tmp.intersection(word_dict[s])
            doc_set = doc_set.union(tmp)
        query_doc_set.append(doc_set)

    # get doc set
    query_doc_dict = {}
    for doc_set in query_doc_set:
        for doc_id in doc_set:
            if doc_id not in query_doc_dict.keys():
                query_doc_dict[doc_id] = 1
            else:
                query_doc_dict[doc_id] += 1
    inv_query_doc_dict = {}
    for doc_id in query_doc_dict.keys():
        freq = query_doc_dict[doc_id]
        if freq not in inv_query_doc_dict.keys():
            inv_query_doc_dict[freq] = [doc_id]
        else:
            inv_query_doc_dict[freq].append(doc_id)

    # get query_wf
    # 1. count word
    query_wf = {}
    for word in old_query:
        if word in query_wf.keys():
            query_wf[word] += 1
        else:
            query_wf[word] = 1
    # 2. get tf * idf, and norm
    query_norm = 0.
    for word in query_wf.keys():
        query_wf[word] = (1 + math.log10(query_wf[word])) * idf_dict[word]
        query_norm += query_wf[word] ** 2
    query_norm = math.sqrt(query_norm)

    # score of cosine, build ret
    query_ret = []
    for freq in reversed(inv_query_doc_dict.keys()):
        doc_score_list = []
        for doc in inv_query_doc_dict[freq]:
            doc_wf, doc_norm = wf_dict[doc], norm_dict[doc]
            score = 0.
            for word in query_wf.keys():
                if word in doc_wf.keys():
                    score += query_wf[word] * doc_wf[word]
            score /= query_norm * doc_norm
            doc_score_list.append((doc, score))
        sorted_doc_score_list = sorted(
            doc_score_list, key=lambda x: x[1], reverse=True)
        for t in sorted_doc_score_list:
            query_ret.append(t[0])
            if len(query_ret) >= top_k:
                break
        if len(query_ret) >= top_k:
            break

    # show docs of topK
    for doc_id in query_ret:
        print '###### DOC_ID:', doc_id, '######'
        doc_filename = str(doc_id) + '.html'
        soup = BeautifulSoup(open('./Reuters/' + doc_filename), 'html.parser')
        print soup.get_text()
        if raw_input() == 'q':
            break
