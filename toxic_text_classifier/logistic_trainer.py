import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack, vstack
from textvec.vectorizers import TforVectorizer

from .inference.logistic_classifier import LogisticClassifier
from .trainer import Trainer


class LogisticTrainer(Trainer):
    def __init__(self, C, use_textvec, distill, labels):
        super().__init__(labels)

        char_vectorizer = CountVectorizer(
            strip_accents='unicode',
            analyzer='char',
            ngram_range=(1, 6),
            binary=True,
            min_df=2
        )
        word_vectorizer = CountVectorizer(
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{1,}',
            ngram_range=(1, 1),
            min_df=2
        )
        self.vectorizers = [char_vectorizer, word_vectorizer]

        self.C = C
        self.use_textvec = set(use_textvec)
        self.distill = distill

    def _fit_single(self, vectorized_data, target, use_textvec):
        if use_textvec:
            transformer_class = TforVectorizer
        else:
            transformer_class = TfidfTransformer

        transformers = [
            transformer_class(sublinear_tf=True).fit(item, target > 0.5)
            for item in vectorized_data
        ]
        data = hstack([
            transformer.transform(item)
            for item, transformer in zip(vectorized_data, transformers)
        ])

        if self.distill:
            data = vstack([data, data])
            sample_weight = np.concatenate([1 - target, target], axis=0)
            target = np.concatenate([
                np.zeros_like(target, dtype=np.int32),
                np.ones_like(target, dtype=np.int32)], axis=0)
        else:
            sample_weight = None

        classifier = LogisticRegression(solver='lbfgs', C=self.C)
        classifier.fit(data, target, sample_weight=sample_weight)

        return transformers, classifier

    def fit(self, data, target, eval_set=None):
        vectorized_data = [vectorizer.fit_transform(data) for vectorizer in self.vectorizers]

        self.classifiers = []
        for target_idx in range(target.shape[-1]):
            current_target = target[:, target_idx]
            self.classifiers.append(self._fit_single(
                vectorized_data, current_target, target_idx in self.use_textvec))

    @staticmethod
    def _inference_class_name(export_type):
        return 'logistic'

    def _save_impl(self, path, export_type=None):
        with open(LogisticClassifier.vectorizer_path(path), 'wb') as f:
            pickle.dump(self.vectorizers, f)

        with open(LogisticClassifier.classifier_path(path), 'wb') as f:
            pickle.dump(self.classifiers, f)
