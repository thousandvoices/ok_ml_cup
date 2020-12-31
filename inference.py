import argparse
import random
import itertools
from collections import defaultdict
import pandas as pd
import numpy as np

from toxic_text_classifier.inference.classifier import Classifier
from toxic_text_classifier.utils import ExtendAction


def average_predictions(texts, filenames):
    results = defaultdict(list)
    for path in filenames:
        classifier = Classifier.load(path)
        classifier_result = classifier.predict(texts)
        for label, scores in classifier_result.items():
            results[label].append(scores)

    return {label: np.mean(scores, axis=0) for label, scores in results.items()}


def augment_sub(text):
    subs = {
        'х': 'x',
        'у': 'y',
        'е': 'e',
        'а': 'a',
        'о': 'o',
        'с': 'c',
        'к': 'k',
        'р': 'p',
        'и': 'u'
    }
    sub = random.sample(list(subs.items()), 1)[0]
    return text.replace(*sub)


def augment(texts):
    random.seed(42)
    added_texts = random.sample(list(texts), k=len(texts))
    return list(itertools.chain.from_iterable([
        texts,
        [x.strip() + ' ' + y for x, y in zip(texts, added_texts)],
        [text[:random.randrange(len(text) // 3, len(text))] for text in texts],
        [augment_sub(text) for text in texts]
    ]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.register('action', 'extend', ExtendAction)
    parser.add_argument('--data-path', dest='data_path', required=True)
    parser.add_argument('--output-path', dest='output_path', required=True)
    parser.add_argument('--filter-path', dest='filter_path')
    parser.add_argument(
        '--fast-classifiers', dest='fast_classifiers', nargs='+', action='extend', default=[])
    parser.add_argument(
        '--slow-classifiers', dest='slow_classifiers', nargs='*', action='extend', default=[])
    parser.add_argument('--write-text', dest='write_text', action='store_true')
    parser.add_argument('--augment', dest='augment', action='store_true')

    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    texts = list(df['text'].astype('str'))
    if args.augment:
        texts = augment(texts)

    result = average_predictions(texts, args.fast_classifiers)

    if len(args.slow_classifiers) > 0:
        def make_uncertain_mask(predictions, threshold):
            return (predictions > threshold) & (predictions < 1 - threshold)
        mask = make_uncertain_mask(result['obscenity'], 0.04)
        mask = mask | make_uncertain_mask(result['threat'], 0.04)

        slow_classifier_texts = np.array(texts)[mask]
        slow_classifier_result = average_predictions(
            slow_classifier_texts, args.slow_classifiers)
        for label in result.keys():
            result[label][mask] = 0.2 * result[label][mask] + 0.8 * slow_classifier_result[label]

    def write_row(f, row):
        f.write(','.join(row) + '\n')

    with open(args.output_path, 'w') as f:
        labels = ['normal', 'insult', 'obscenity', 'threat']
        columns = labels[:]
        if args.write_text:
            columns += ['text']
        write_row(f, columns)

        predictions = np.stack([result[label] for label in labels], axis=1)

        for text, prediction in zip(texts, predictions):
            formatted_prediction = [str(value) for value in prediction]
            if args.write_text:
                formatted_prediction += ['"' + text.strip() + '"']
            write_row(f, formatted_prediction)
