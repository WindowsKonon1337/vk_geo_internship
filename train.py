#!/usr/bin/python3.10

import argparse
import pickle

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from geopy.distance import geodesic
from sklearn.metrics import make_scorer, mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split
from utils import check_dump_dirs, check_upload_dirs

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('--train_csv_path', type=str)
arg_parser.add_argument('--test_csv_path', type=str)
arg_parser.add_argument('--features_csv_path', type=str)
arg_parser.add_argument('--dump_extended_train_path', type=str)
arg_parser.add_argument('--dump_extended_test_path', type=str)
arg_parser.add_argument('--dump_model_path', type=str)

def set_feature_dataframe(df_features):
    inner_df = df_features
    def find_nearest_point(row):
        distances = inner_df.apply(
            lambda x: geodesic((row['lat'], row['lon']), (x['lat'], x['lon'])).meters, axis=1
        )
        nearest_index = distances.idxmin()
        return inner_df.loc[nearest_index]
    
    return find_nearest_point


if __name__ == '__main__':

    args = arg_parser.parse_args()

    check_upload_dirs(
        args.train_csv_path,
        args.test_csv_path,
        args.features_csv_path
    )

    check_dump_dirs(
        args.dump_extended_train_path,
        args.dump_extended_test_path,
        args.dump_model_path
    )

    df_train = pd.read_csv(args.train_csv_path)
    df_test = pd.read_csv(args.test_csv_path)
    df_features = pd.read_csv(args.features_csv_path)

    nearest_func = set_feature_dataframe(df_features)

    nearest_train_features = df_train.apply(nearest_func, axis=1)
    nearest_test_features = df_test.apply(nearest_func, axis=1)

    df_train_extended = pd.concat(
        (
            df_train,
            nearest_train_features[nearest_train_features.columns[2:]]
        ), axis=1
    )
    
    df_test_extended = pd.concat(
        (
            df_test,
            nearest_test_features[nearest_test_features.columns[2:]]
        ), axis=1
    )

    df_train_extended.to_csv(args.dump_extended_train_path)
    df_test_extended.to_csv(args.dump_extended_test_path)

    X = df_train_extended.drop(columns=['score', 'id'])
    y = df_train_extended.score
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1337, shuffle=True)

    from catboost import CatBoostRegressor

    reg = CatBoostRegressor()
    params = {'iterations': [500],
            'depth': [4, 5, 6],
            'loss_function': ['MAE'],
            'l2_leaf_reg': np.logspace(-20, -19, 3),
            'leaf_estimation_iterations': [10],
            'logging_level':['Silent'],
            'random_seed': [42]
            }
    scorer = make_scorer(mean_absolute_error)
    reg = GridSearchCV(estimator=reg, param_grid=params, scoring=scorer, cv=5)

    reg.fit(X_train, y_train)

    pickle.dump(
        reg.best_estimator_, open(args.dump_model_path, 'wb')
    )
