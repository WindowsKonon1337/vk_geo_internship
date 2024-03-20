#!/usr/bin/python3.10

import argparse
import pickle

import pandas as pd
from utils import check_upload_dirs, check_dump_dirs

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('--extended_test_path', type=str)
arg_parser.add_argument('--model_path', type=str)
arg_parser.add_argument('--submission_path', type=str)

if __name__ == '__main__':

    args = arg_parser.parse_args()

    check_upload_dirs(
        args.extended_test_path,
        args.model_path
    )

    check_dump_dirs(
        args.submission_path
    )

    df_test_extended = pd.read_csv(args.extended_test_path)

    X = df_test_extended.drop(columns=['id']).to_numpy()

    model = pickle.load(open(args.model_path, 'rb'))

    pred = model.predict(X)

    df_test_extended['score'] = pd.Series(pred)

    df_test_extended[['id', 'score']].to_csv(args.submission_path, index=False)
