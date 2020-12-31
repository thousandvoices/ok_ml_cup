import numpy as np
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer  # noqa: F401
from textvec.vectorizers import TforVectorizer  # noqa: F401
from sklearn.linear_model import LogisticRegression  # noqa: F401
from scipy.sparse import hstack


class LogisticClassifier:
    def __init__(self, model_path):
        with open(self.vectorizer_path(model_path), 'rb') as f:
            self.vectorizers = pickle.load(f)

        with open(self.classifier_path(model_path), 'rb') as f:
            self.classifiers = pickle.load(f)

    @staticmethod
    def _predict_single(classifier, data):
        return classifier.predict_proba(data)[:, 1]

    def predict(self, data):
        vectorized_data = [vectorizer.transform(data) for vectorizer in self.vectorizers]
        class_probabilities = [
            self._predict_single(
                classifier,
                hstack([
                    transformer.transform(item)
                    for item, transformer in zip(vectorized_data, transformers)
                ])
            )
            for transformers, classifier in self.classifiers
        ]

        return np.stack(class_probabilities, axis=1)

    @staticmethod
    def vectorizer_path(path):
        return Path(path) / 'vectorizer.pkl'

    @staticmethod
    def classifier_path(path):
        return Path(path) / 'classifier.pkl'
