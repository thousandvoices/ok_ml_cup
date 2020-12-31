import unittest
from tempfile import TemporaryDirectory
import numpy as np
from sklearn.metrics import average_precision_score

from toxic_text_classifier.logistic_trainer import LogisticTrainer
from toxic_text_classifier.bert_trainer import BertTrainer
from toxic_text_classifier.inference.classifier import Classifier


NUM_TEXTS = 500


class ClassifiersTest(unittest.TestCase):
    def _test_trainer(self, trainer, export_type):
        texts = ['lorem ipsum', 'dolor sit amet'] * NUM_TEXTS
        targets = np.int32([text == texts[0] for text in texts])[:, None]
        targets = np.tile(targets, (1, len(trainer.labels)))
        trainer.fit(texts, targets, eval_set=(texts, targets))

        with TemporaryDirectory() as temp_dir:
            trainer.save_classifier(temp_dir, export_type)
            classifier = Classifier.load(temp_dir)
            train_predictions = classifier.predict(texts)
            stacked_predictions = np.stack([
                train_predictions[label] for label in trainer.labels
            ], axis=1)
            max_difference = np.max(np.abs(stacked_predictions - targets))
            self.assertLess(max_difference, 0.3)

            single_prediction = classifier.predict(texts[0])
            for label in single_prediction.keys():
                difference = np.abs(train_predictions[label][0] - single_prediction[label][0])
                self.assertLess(difference, 1e-3)

    def test_logistic(self):
        trainer = LogisticTrainer(
            C=1, use_textvec=[], distill=False, labels=['test', 'test2']
        )
        self._test_trainer(trainer, None)

    def test_logistic_distillation(self):
        trainer = LogisticTrainer(
            C=1, use_textvec=[], distill=True, labels=['test']
        )
        self._test_trainer(trainer, None)

    def test_bert(self):
        trainer = BertTrainer(
            'DeepPavlov/rubert-base-cased-conversational',
            3,
            1,
            False,
            [average_precision_score],
            labels=['1', '2', '3']
        )
        self._test_trainer(trainer, None)

    def test_bert_cpu(self):
        trainer = BertTrainer(
            'DeepPavlov/rubert-base-cased-conversational',
            3,
            1,
            False,
            [average_precision_score],
            labels=['1', '2']
        )
        self._test_trainer(trainer, 'cpu')
