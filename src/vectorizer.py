from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse

class TextVectorizer:
    def __init__(self, max_features=10000, min_df=2, max_df=0.9):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True
        )

    def fit_transform(self, corpus):
        """
        Fits vectorizer and transforms corpus.
        Args:
            corpus (list of str): List of processed sentences.
        Returns:
            scipy.sparse.csr_matrix: TF-IDF matrix (Sentences x Terms).
            list of str: Feature names.
        """
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        return tfidf_matrix, self.vectorizer.get_feature_names_out()

    def transform(self, corpus):
        return self.vectorizer.transform(corpus)

