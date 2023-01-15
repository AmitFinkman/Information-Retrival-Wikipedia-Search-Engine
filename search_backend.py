import re
import numpy as np
# from google.cloud import storage
import pickle
import nltk
import builtins
from nltk.corpus import stopwords
import json
from contextlib import closing
from nltk.stem import *
from collections import defaultdict, Counter


nltk.download('stopwords')

# %cd -q /home/dataproc
# !ls inverted_index_gcp.py
from inverted_index_gcp import InvertedIndex, MultiFileReader

import math
from google.cloud import storage

client = storage.Client()
bucket = client.get_bucket('all_indexes_semi_final')

################################## load index #####################################################

# load title Inverted index
title_index_nostem = pickle.loads(bucket.get_blob('title_index_nostem.pkl').download_as_string())
title_index_nostem.bucket_name = "title_bucket_without_stem"

# load title Inverted index with stemming
title_index_stem = pickle.loads(bucket.get_blob('title_index.pkl').download_as_string())
title_index_stem.bucket_name = "title_bucket_with_stem"

# load title Inverted index with stemming and ngram
title_index_bigram = pickle.loads(bucket.get_blob('title_bigram_index.pkl').download_as_string())
title_index_bigram.bucket_name = "title_bucket_ngramm"

# load body Inverted index without stemming
body_index_nostem = pickle.loads(bucket.get_blob('body_index_nostem.pkl').download_as_string())
body_index_nostem.bucket_name = "body_bucket_nostem2"

# load body Inverted index without stemming
body_index_stem = pickle.loads(bucket.get_blob('body_index_withstem.pkl').download_as_string())
body_index_stem.bucket_name = "body_bucket_with_stem2"

# load id to title dictionary
id2title = pickle.loads(bucket.get_blob('id2title.pkl').download_as_string())

# load anchor Inverted index
index_anchor = pickle.loads(bucket.get_blob('anchor_index.pkl').download_as_string())
index_anchor.bucket_name = "anchor_bucket_amit"

# load page rank and page views
page_rank = pickle.loads(bucket.get_blob('page_rank.pickle').download_as_string())
page_views = pickle.loads(bucket.get_blob('pageviews-202108-user.pkl').download_as_string())

################################## end load #####################################################

TUPLE_SIZE = 6
stemmer = PorterStemmer()


def get_tokens(text, stem=False):
    """
    function that returns tokens of query, stemming is option (depends on task)
    input: text: query (a string)
           stem: boolean value if the tokenization is with or without stemming
    output: list of tokens
    """
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    if stem:
        tokens = [stemmer.stem(term) for term in tokens if term not in all_stopwords]
    else:
        tokens = [term for term in tokens if term not in all_stopwords]
    return tokens


def get_tokens_ngrams(text):
    """
    function that returns tokens of query with n-gram, for the main search
    input: text: query (a string)
           stem: boolean value if the tokenization is with or without stemming
    output: list of tokens with n-gram
    """
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    tokens = [stemmer.stem(term) for term in tokens if term not in all_stopwords]
    ngrams_tokens = [token[0] + " " + token[1] for token in list(nltk.bigrams(tokens))]
    return ngrams_tokens


# remove stopwords
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became", "best"]
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
all_stopwords = english_stopwords.union(corpus_stopwords)


def get_candidate_documents(q, index):
    """
    function that returns relevant docs for the query with its term frequencies
    input: query_to_search: list of tokens
           stem: boolean value if the tokenization is with or without stemming
           index: inverted index
    output: set of relevant docs, dict of candidates as the follow:
                                  :key: term
                                  value: dict of relevant docs as the follow:
                                        :key:  doc_id
                                        value: term frequency

    """
    # create a set because we want the relevant docs without duplicates
    # create a dict will contain all words and relevant docs
    candidates_d, candidates = {}, set()
    for w in np.unique(q):
        # cheaks if word in inverted
        df = index.df
        if w in df:
            p = index.get_posting_list_term(w, 2000000)
            only_doc = [x[0] for x in p]
            candidates_d.update({w: dict(p)})
            candidates.update(only_doc)
    return candidates, candidates_d


def get_top_n(sim_dict, N=300):
    """
    function from homework, sorts the best docs by their values.
    input: sim_dict: dict as follows:
            key: doc_id
            value: score of doc
    output: best N docs

    """
    return sorted([(doc_id, score) for doc_id, score in sim_dict.items()], key=lambda x: x[1], reverse=True)[:N]


def get_num_of_match_title_anchor(q, term_dict):
    """
    helper function for search_for_title_and_anchor.
    function that calculates unique words from the query that appears in a document.
    since the calculation for search title and search anchor is the same, we combined them into one function.
    input: query: list of tokens
            term dict: dict as follows:
                key: term from the query
                value: dict of relevant docs as the follow:
                                        :key:  doc_id
                                        value: term frequency
    output: dict as follows:
                        :key: doc_id
                        value: score based on how many unique words appears in each document

    """
    dict_to_return = {}
    for w in term_dict.keys():
        for doc in term_dict[w].keys():
            dict_to_return[doc] = dict_to_return.get(doc, 0) + 1
    return dict_to_return


def search_for_title_and_anchor(query, index):
    """
    helper function for search title and search anchor.
    function that returns the best docs for title and anchor as defined in the project.
    the docs are sorted by score.
    input: query: list of tokens
           index: inverted index
    output: list of tuples as follows: [(doc_id,title)...], rr is a dictionary that will be helpful for the main search.
                                                            :key: doc_id
                                                            value: score
    """
    rel_docs, candidates_dict = get_candidate_documents(query, index)
    rr = get_num_of_match_title_anchor(query, candidates_dict)
    id_score = get_top_n(rr, N=100000000)
    res = [i[0] for i in id_score]
    return [(j, id2title[j]) for j in res if j in id2title], rr


def search_title_backend(Q):
    """
    calculates the best documents by their title, as defined in the project.
    the helper functions above contain all the calculations.
    input: Q: query as a string
    output: list of tuples as follows: [(doc_id, title)...]
    """
    Q = get_tokens(Q)
    return search_for_title_and_anchor(Q, title_index_nostem)[0]


def search_anchor_backend(Q):
    """
    calculates the best documents by their anchor, as defined in the project.
    the helper functions above contain all the calculations.
    input: Q: query as a string
    output: list of tuples as follows: [(doc_id, title)...]
    """
    Q = get_tokens(Q)
    return search_for_title_and_anchor(Q, index_anchor)[0]


def get_page_rank(wiki_ids):
    """
    function that returns the page rank for each doc in given docs_ids list.
    input: wiki_ids: list of integers id etc. [1,3,5,6]
    output: a list of corresponds page rank according to the input we received.
    """
    return [page_rank[i] if i in page_rank else 0 for i in wiki_ids if i in wiki_ids]


def get_page_views(wiki_ids):
    """
    function that returns the page views for each doc in given docs_ids list.
    input: wiki_ids: list of integers id etc. [1,3,5,6]
    output: a list of corresponds page views according to the input we received.
    """
    return [page_views[i] if i in page_views else 0 for i in wiki_ids if i in wiki_ids]


def tf_idf_calculation_and_cosine(query_tokens, index, docs, candidates_dict, tfidf_scores):
    """
    function that calculates tf-idf and cosine similarity for search body
    input: query_tokens: list of tokens
           index: inverted index
           candidate dict: dict as follows:
                            :key: term
                            value: dict as follows:
                                    :key: doc_id
                                    value: term frequency
    output: a dict as follows:
                        :key: doc_id
                        value: score of doc_id based on tf-idf and cosine similarity.
    """
    # get size of query
    n = len(query_tokens)
    query_lst = np.ones(n)
    answer = defaultdict(int)
    for doc in docs:
        tf_ids_score = np.empty(n)
        for i, w in enumerate(query_tokens):
            if (w in index.df) and (doc in candidates_dict[w]):
                tf_term = (candidates_dict[w][doc] / index.DL[doc])
                idf_term = math.log2(6348911 / index.df[w])
                tf_ids_score[i] = tf_term * idf_term

            else:
                tf_ids_score[i] = 0

        inner_product = np.dot(tf_ids_score, query_lst)
        size_q = np.linalg.norm(query_lst)
        answer[doc] = inner_product / (size_q * index.DS[doc])

    return answer


def search_body_for_all(index, query):
    """
    helper function for search body.
    function that returns the best docs based on the body, as defined in the project.
    the docs are sorted by score.
    input: query: list of tokens
           index: inverted index
    output: list of tuples as follows: [(doc_id,title)...], rr is a dictionary that will be helpful for the main search.
                                                            :key: doc_id
                                                            value: score
    """
    rel_docs, candidates_dict = get_candidate_documents(query, index)
    similarity = tf_idf_calculation_and_cosine(query, index, rel_docs, candidates_dict, index.DS)
    top_n = get_top_n(similarity, 100)
    res = [i[0] for i in top_n]
    return [(j, id2title[j]) for j in res], similarity


def search_body_backend(query):
    """
    calculates the best documents by their body, as defined in the project.
    the helper functions above contain all the calculations.
    input: Q: query as a string
    output: list of tuples as follows: [(doc_id, title)...]
    """
    query = get_tokens(query)
    return search_body_for_all(body_index_nostem, query)[0]


# we chose to use in class BM25_from_index from the assignments for the final search.
class BM25_from_index:

    def __init__(self, index, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.index = index
        self.N = len(index.DL)
        self.AVGDL = builtins.sum(index.DL.values()) / self.N

    def calc_idf(self, list_of_tokens):
        idf = {}
        for term in list_of_tokens:
            if term in self.index.df:
                n_ti = self.index.df[term]
                idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
            else:
                pass
        return idf

    def _score(self, query, doc_id, candidate_dict):
        score = 0.0
        doc_len = self.index.DL[doc_id]
        for term in query:
            if term in self.index.df:
                term_frequencies = candidate_dict[term]
                if doc_id in term_frequencies:
                    freq = term_frequencies[doc_id]
                    numerator = self.idf[term] * freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)
                    score += (numerator / denominator)
        return score

    def search(self, q, N=10000):
        # q is tokens lst
        candidates, candidates_dict = get_candidate_documents(q, self.index)
        self.idf = self.calc_idf(q)
        temp_dict = {k: self._score(q, k, candidates_dict) for k in candidates}
        # highest_N_score is list of tuples where (docID, score)
        highest_n_score = get_top_n(temp_dict, N)
        return highest_n_score


def find_minmax(lst):
    """
    returns the minimum and the maximum in a given list.
    this function is a helper function for min max scaler that will be used in the main search.
    input: lst: list of floats
    output: min and max number from the list.
    """
    if len(lst) == 0: return 0, 1
    min_v, max_v = np.min(lst), np.max(lst)
    if max_v == 0.0:
        max_v = 1
    return min_v, max_v


def search_extanded(q, idx_title_stem, idx_title_ngram, idx_body_stem):
    """
    the main search in our project.
    this function calculates the best documents with the highest scores by their body, title, title with ngram, page rank and page views.
    input: q: query as a string.
           idx_title_stem: inverted index on title with stemming.
           idx_title_ngram: inverted index on title with stemming and as bigrmas.
           idx_body_stem: inverted index on body with stemming.
    output: list of tuples as follows: [(doc_id,title)...].
    The list of the documents that received the highest scores based on various metrics while using weights.
    """
    # tokens to search
    query_tokens = get_tokens(q, True)

    # search with bm25
    bm25 = BM25_from_index(idx_body_stem)
    # what in v is the pageid and its bm25 score
    bm25_cand = bm25.search(query_tokens, 5000)
    # search on body without n_gram

    cadndiate_scores = {}

    bm25_lst_scores = np.zeros(len(bm25_cand))
    top_n_titles_scores = np.zeros(len(bm25_cand))
    top_n_titles_scores_ngram = np.zeros(len(bm25_cand))
    page_rank_scores = np.zeros(len(bm25_cand))
    page_views_scores = np.zeros(len(bm25_cand))

    for i, c in enumerate(bm25_cand):
        bm25_lst_scores[i] = c[1]
        cadndiate_scores[c[0]] = [c[1], 0, 0, 0, 0]

    # search on title without n_gram
    similarity = search_for_title_and_anchor(query_tokens, idx_title_stem)[1]
    top_n_title = get_top_n(similarity, 5000)

    for i, c in enumerate(top_n_title):
        if c[0] in cadndiate_scores:
            top_n_titles_scores[i] = c[1]
            cadndiate_scores[c[0]][1] = c[1]

    # tokens to search with n gram
    query_tokens_ngram = get_tokens_ngrams(q)

    # search on title with n_gram
    search_title_for_all = search_for_title_and_anchor(query_tokens_ngram, idx_title_ngram)[1]
    top_n_title_ngram = get_top_n(search_title_for_all, 5000)
    for i, c in enumerate(top_n_title_ngram):
        if c[0] in cadndiate_scores:
            top_n_titles_scores_ngram[i] = c[1]
            cadndiate_scores[c[0]][2] = c[1]

    for i, c in enumerate(cadndiate_scores):
        if c in page_rank:
            page_rank_scores[i] = page_rank[c]
            cadndiate_scores[c][3] = page_rank[c]
        if c in page_views:
            page_views_scores[i] = page_views[c]
            cadndiate_scores[c][4] = page_views[c]

    # find min max values for normalize
    minimum_bm25, maximum_bm25 = find_minmax(bm25_lst_scores)
    minimum_title, maximum_title = find_minmax(top_n_titles_scores)
    minimum_title_ngram, maximum_title_ngram = find_minmax(top_n_titles_scores_ngram)
    minimum_prank, maximum_prank = find_minmax(page_rank_scores)
    minimum_pv, maximum_pv = find_minmax(page_views_scores)

    for k in cadndiate_scores:
        cc = 0
        cadndiate_scores[k][0] = (cadndiate_scores[k][0] - minimum_bm25) / (maximum_bm25 - minimum_bm25) * 5
        cadndiate_scores[k][1] = (cadndiate_scores[k][1] - minimum_title) / (maximum_title - minimum_title) * 1
        cadndiate_scores[k][2] = (cadndiate_scores[k][2] - minimum_title_ngram) / (
                    maximum_title_ngram - minimum_title_ngram) * 1
        cadndiate_scores[k][3] = (cadndiate_scores[k][3] - minimum_prank) / (maximum_prank - minimum_prank) * 3
        cadndiate_scores[k][4] = (cadndiate_scores[k][4] - minimum_pv) / (maximum_pv - minimum_pv) * 3

        # get aggregated score for each document
        for i in cadndiate_scores[k]:
            cc += i
        cadndiate_scores[k] = cc

    result = get_top_n(cadndiate_scores, 20)
    res = [i[0] for i in result]
    res = [(j, id2title[j]) for j in res]

    return res


def final_search(q):
    """
    helper function that only calls to the final search.
    """
    return search_extanded(q, title_index_stem, title_index_bigram, body_index_stem)










