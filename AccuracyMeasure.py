from sklearn.feature_extraction.text import TfidfVectorizer
import scipy


class Measurer:

    def __init__(self):
        pass

    def VectorizeCorpus(self, corpus: list) -> scipy.sparse.csr.csr_matrix:
        """
        Save corpus' tf-idf frequencies relative to other documents in the corpus in
        a matrix and return the matrix
        :param corpus:
        :return:
        """
        vect = TfidfVectorizer()
        tfidf_matrix = vect.fit_transform(corpus)
        return tfidf_matrix
