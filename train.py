import pandas as pd
import argparse
from sklearn.metrics import average_precision_score

from toxic_text_classifier.utils import ExtendAction
from toxic_text_classifier.bert_trainer import BertTrainer
from toxic_text_classifier.logistic_trainer import LogisticTrainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.register('action', 'extend', ExtendAction)
    parser.add_argument('--classifier', dest='classifier', required=True)
    parser.add_argument('--save-path', dest='save_path')
    parser.add_argument('--train', dest='train', nargs='+', action='extend')
    parser.add_argument('--validation', dest='validation', nargs='*', action='extend')
    parser.add_argument('--export-type', dest='export_type')
    parser.add_argument('--layers', dest='layers', type=int)
    parser.add_argument('--epochs', dest='epochs', type=int)
    parser.add_argument('--distill', dest='distill', action='store_true')
    args = parser.parse_args()

    label_columns = ['normal', 'insult', 'obscenity', 'threat']

    train_df = pd.concat([pd.read_csv(filename) for filename in args.train], axis=0)
    train_df = train_df.reset_index()
    if args.validation is not None:
        val_df = pd.concat([pd.read_csv(filename) for filename in args.validation], axis=0)
        val_df = val_df.reset_index()
        eval_set = (val_df['text'], val_df[label_columns].values)
    else:
        eval_set = None

    if args.classifier == 'logistic':
        trainer = LogisticTrainer(
            C=8, use_textvec=[0, 1], distill=args.distill, labels=label_columns)
    else:
        trainer = BertTrainer(
            args.classifier,
            args.layers,
            args.epochs,
            args.distill,
            [average_precision_score],
            label_columns
        )

    trainer.fit(
        train_df['text'],
        train_df[label_columns].values,
        eval_set=eval_set
    )
    if args.save_path is not None:
        trainer.save_classifier(args.save_path, args.export_type)
