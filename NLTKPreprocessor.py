import string

from nltk import pos_tag
from nltk import sent_tokenize
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords as sw
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import ItalianStemmer
from sklearn.base import BaseEstimator, TransformerMixin


class NLTKPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, stopwords=None, punct=None,
                 lower=True, strip=True, Language='English'):
        self.lower = lower
        self.strip = strip
        self.punct = punct or set(string.punctuation)
        if Language == 'English':
            self.stopwords = stopwords or set(sw.words('english'))
            self.stemmer = PorterStemmer()
        elif Language == 'Italian':
            self.stopwords = stopwords or set(sw.words('italian'))
            self.stemmer = ItalianStemmer()

    def fit(self, X, y=None):
        return self

    def inverse_transform(self, X):
        return [" ".join(doc) for doc in X]

    def transform(self, X):

        return [
            list(self.tokenize(doc)) for doc in X
        ]

    def tokenize(self, document):
        # Break the document into sentences
        for sent in sent_tokenize(document):
            # Break the sentence into part of speech tagged tokens
            for token, tag in pos_tag(wordpunct_tokenize(sent)):
                # Apply preprocessing to the token
                token = token.lower() if self.lower else token
                token = token.strip() if self.strip else token
                token = token.strip('_') if self.strip else token
                token = token.strip('*') if self.strip else token

                # If stopword, ignore token and continue
                if token in self.stopwords:
                    continue

                # If punctuation, ignore token and continue
                if all(char in self.punct for char in token):
                    continue

                # Lemmatize the token and yield
                lemma = self.stemmer.stem(token)
                yield lemma

def identity(arg):
    """
    Simple identity function works as a passthrough.
    """
    return arg