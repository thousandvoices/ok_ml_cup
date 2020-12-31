from abc import ABC, abstractmethod

from .inference.classifier import Classifier


class Trainer(ABC):
    def __init__(self, labels):
        self.labels = labels

    @abstractmethod
    def fit(self, data, target, eval_set=None):
        pass

    def save_classifier(self, path, export_type=None):
        Classifier.save_metadata(
            self._inference_class_name(export_type),
            self.labels,
            path
        )
        self._save_impl(path, export_type)

    @staticmethod
    @abstractmethod
    def _inference_class_name(export_type):
        pass

    @abstractmethod
    def _save_impl(self, path, export_type):
        pass
