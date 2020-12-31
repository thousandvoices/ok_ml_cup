import json
from pathlib import Path
import importlib

from .model_cache import ModelCache


CLASSIFIER_IMPORTS = {
    'bert': '.bert_cpu_classifier.BertCpuClassifier',
    'bert_gpu': '.bert_gpu_classifier.BertGpuClassifier',
    'logistic': '.logistic_classifier.LogisticClassifier'
}

cached_imports = {}


def _import_class(full_class_name):
    module_name, class_name = full_class_name.rsplit('.', maxsplit=1)
    package = __name__.rsplit('.', maxsplit=1)[0]
    module = importlib.import_module(module_name, package)
    return getattr(module, class_name)


def _load_class(classifier_name):
    cached_import = cached_imports.get(classifier_name)
    if cached_import is not None:
        return cached_import
    else:
        result = _import_class(CLASSIFIER_IMPORTS[classifier_name])
        cached_imports[classifier_name] = result
        return result


class Classifier:
    MODEL_CACHE = ModelCache(Path.home() / '.toxic_text_classifier')

    def __init__(self, impl, labels):
        self.impl = impl
        self.labels = labels

    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        raw_probabilities = self.impl.predict(texts)
        return {label: raw_probabilities[:, idx] for idx, label in enumerate(self.labels)}

    @staticmethod
    def _metadata_path(path):
        return Path(path) / 'metadata.json'

    @classmethod
    def save_metadata(cls, class_name, labels, path):
        metadata = {'class': class_name, 'labels': labels}
        Path(path).mkdir(parents=True, exist_ok=True)
        with open(cls._metadata_path(path), 'w') as f:
            json.dump(metadata, f, indent=4)

    @classmethod
    def load(cls, path):
        cached_path = cls.MODEL_CACHE.cached_path(path)

        with open(cls._metadata_path(cached_path)) as f:
            config = json.load(f)

        return cls(
            _load_class(config['class'])(str(cached_path)),
            config['labels']
        )
