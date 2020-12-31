import pandas as pd
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split


def parse(lines, label_columns):
    rows = []
    for line in lines:
        labels, text = line.split(' ', maxsplit=1)
        labels = labels.split(',')
        row_id = '0'
        labels = set(labels)
        numeric_labels = [int(label in labels) for label in label_columns]
        rows.append((row_id, text, *numeric_labels))

    return list(zip(*rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train-path', dest='input_path', required=True)
    parser.add_argument('--output-dir', dest='output_dir', required=True)
    args = parser.parse_args()

    label_columns = ['__label__NORMAL', '__label__INSULT', '__label__OBSCENITY', '__label__THREAT']

    with open(args.input_path) as f:
        lines = f.readlines()
    data = parse(lines, label_columns)
    fixed_labels = [label.lower().split('_')[-1] for label in label_columns]
    columns = ['id', 'text'] + fixed_labels

    df = pd.DataFrame.from_dict({column: item for column, item in zip(columns, data)})
    train_df, val_df = train_test_split(df, test_size=20000, random_state=1337)

    output_dir = Path(args.output_dir)
    train_df.to_csv(output_dir / 'train.csv', index=False)
    val_df.to_csv(output_dir / 'validation.csv', index=False)

    train_df['id'].to_csv(output_dir / 'train_ids.txt', index=False, header=False)
    val_df['id'].to_csv(output_dir / 'val_ids.txt', index=False, header=False)
