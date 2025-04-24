import argparse
import logging
import os
import sys
import time
from pyexpat import features
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.impute import KNNImputer, SimpleImputer

from classifiers.perceptron_unary_classifier import PerceptronUnaryClassifier
from entities.metrics import Metrics
from entities.train_params import TrainParams


def get_logger(use_file: bool, logs_path: str) -> logging.Logger:
    if use_file:
        logging.basicConfig(filename=logs_path, filemode="a", level=logging.INFO, format="%(message)s")
    else:
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    return logging.getLogger()


def init_metrics(split_names: List[str], metric_names: List[str]) -> Metrics:
    metrics = Metrics()

    for split in split_names:
        for metric in metric_names:
            metrics.add(split=split, metric=metric)

    return metrics


def fill_nans(df: pd.DataFrame, rows: int) -> pd.DataFrame:
    features = df.columns[:-1]

    for i in range(len(df) - rows, len(df)):
        num_missing = np.random.randint(1, len(features))
        missing_indices = np.random.choice(features, num_missing, replace=False)
        df.loc[i, missing_indices] = np.nan

    return df


def imput_data(df: pd.DataFrame, imputer_call: Callable, classes: int) -> pd.DataFrame:
    feature_columns = df.columns[:-1]

    for label in range(classes):
        imputer = imputer_call()

        label_mask = df["label"] == label
        features = df.loc[label_mask, feature_columns].values
        df.loc[label_mask, feature_columns] = imputer.fit_transform(features)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="dataset name", type=str, default="clusters")
    parser.add_argument("--missing-part", help="part of missed rows", type=float, default=0.5)
    parser.add_argument("--dimension", help="dimension of input space", type=int, default=2)
    parser.add_argument("--classes", help="number of classes", type=int, default=4)
    parser.add_argument("--beta", help="trust threshold", type=float, default=0.5)
    parser.add_argument("--plot-period", help="period of model plot", type=int, default=50)
    parser.add_argument("--log-period", help="period of metrics print", type=int, default=10)
    parser.add_argument("--save-period", help="period of classifiers save", type=int, default=50)
    parser.add_argument("--plot-metrics-period", help="period of metrics plot", type=int, default=50)
    parser.add_argument("--loops", help="number of experiment loops", type=int, default=50)
    parser.add_argument("--epoch-steps", help="number of epoch steps", type=int, default=100)
    parser.add_argument("--learning-rate", help="training learning rate", type=float, default=0.01)
    parser.add_argument("--batch-size", help="training batch size", type=int, default=16)
    parser.add_argument("--device", help="training torch device", type=str, default="cuda")
    parser.add_argument("-f", help="log to file instead of stdout", action="store_true")
    args = parser.parse_args()

    sizes = [20, 20]
    device = torch.device(args.device)

    train_params = TrainParams(
        batch_size=args.batch_size,
        epochs=10,
        learning_rate=args.learning_rate
    )

    dataset_dir = os.path.join("datasets", args.dataset)
    experiment_dir = os.path.join("experiments", "missing_data", args.dataset, f"missing {args.missing_part:.0%}")
    os.makedirs(experiment_dir, exist_ok=True)

    logger = get_logger(args.f, os.path.join(experiment_dir, "log.txt"))
    logger.info(f"- Dataset: {args.dataset} (dimension: {args.dimension}, classes: {args.classes})")
    logger.info(f"- Missing part: {args.missing_part}")
    logger.info(f"- Model: {args.dimension}-{sizes}-1 (beta: {args.beta}, device: {args.device})")
    logger.info(f"- Training: {train_params.epochs} epochs (batch size: {train_params.batch_size}, learning rate: {train_params.learning_rate})\n")

    if not os.path.exists(os.path.join(dataset_dir, "train.csv")):
        logger.error("Train dataset does not exists")
        return

    if not os.path.exists(os.path.join(dataset_dir, "test.csv")):
        logger.error("Test dataset does not exists")
        return

    train_df = pd.read_csv(os.path.join(dataset_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(dataset_dir, "test.csv"))

    missing_rows = int(args.missing_part * len(train_df))
    missing_df = fill_nans(df=train_df.copy(), rows=missing_rows)

    logger.info(f" Train total: {len(train_df)}")
    logger.info(f"  Test total: {len(test_df)}")
    logger.info(f"Missing rows: {missing_rows} ({missing_rows / len(train_df):.1%})")
    logger.info(train_df.describe())

    mask = missing_df.isna().any(axis=1)
    complete_df = missing_df[~mask].copy()

    imputers = {
        "mean": lambda: SimpleImputer(strategy="mean"),
        "most frequent": lambda: SimpleImputer(strategy="most_frequent"),
        "knn 1": lambda: KNNImputer(n_neighbors=1),
        "knn 3": lambda: KNNImputer(n_neighbors=3),
        "knn 7": lambda: KNNImputer(n_neighbors=7)
    }

    plot_metrics = ["refuse", "accuracy", "accuracy_full"]
    log_metrics = ["refuse", "accuracy", "accuracy_full", "f1_macro_full", "f1_micro_full", "fit_time"]
    plot_splits = ["full", "complete", "reproduce", "mean", "knn 1", "knn 3", "knn 7", "most frequent"]
    metrics = init_metrics(
        split_names=plot_splits,
        metric_names=[
            "fit_time",
            "refuse",
            "accuracy", "accuracy_full",
            "f1_macro", "f1_macro_full",
            "f1_micro", "f1_micro_full",
            "precision_micro",
            "precision_macro",
            "recall_micro",
            "recall_macro"
        ]
    )

    train_dfs = {
        "full": train_df,
        "complete": complete_df,
        "reproduce": complete_df.copy()
    }

    for strategy, imputer_call in imputers.items():
        train_dfs[strategy] = imput_data(df=missing_df.copy(), imputer_call=imputer_call, classes=args.classes)

    for loop in range(args.loops):
        classifiers = {}
        init_classifier = PerceptronUnaryClassifier(dimension=args.dimension, classes=args.classes, sizes=sizes, beta=args.beta, device=device)

        for key in plot_splits:
            classifiers[key] = PerceptronUnaryClassifier(dimension=args.dimension, classes=args.classes, sizes=sizes, beta=args.beta, device=device)

            for label in range(args.classes):
                classifiers[key].models[label].load_state_dict(init_classifier.models[0].state_dict())

            classifiers[key].init_compacts(train_df=train_dfs[key])

        for epoch_step in range(args.epoch_steps):
            step = train_params.epochs * (epoch_step + 1)

            classifiers_dir = os.path.join(experiment_dir, "classifiers", f"{loop + 1}")
            plots_dir = os.path.join(experiment_dir, "plots", f"{loop + 1}")
            os.makedirs(classifiers_dir, exist_ok=True)
            os.makedirs(plots_dir, exist_ok=True)

            if step > 50:
                train_dfs["reproduce"] = classifiers["reproduce"].fill_nans(missing_df)

            if step % args.log_period == 0:
                logger.info(f"\nLoop {loop + 1} ({step} epochs)")
                logger.info(metrics.get_header(split_name="strategy", metric_names=log_metrics))

            for key in plot_splits:
                start_time = time.time()
                classifiers[key].fit(train_df=train_dfs[key], train_params=train_params)
                end_time = time.time()
                metrics.set_values(split=key, metrics=classifiers[key].evaluate(df=test_df), step=step)
                metrics.set_value(split=key, metric="fit_time", step=step, value=end_time - start_time)

                if step % args.log_period == 0:
                    logger.info(metrics.get_metrics(split=key, metric_names=log_metrics, step=step))

                if step % args.plot_period == 0:
                    classifiers[key].plot(train_df=train_dfs[key], test_df=test_df, path=os.path.join(plots_dir, f"{key}_{step}.png"), v_min=-2, v_max=2)

                if step % args.save_period == 0:
                    classifiers[key].save(os.path.join(classifiers_dir, f"{key}_{step}.zip"))

            if step % args.plot_metrics_period == 0:
                metrics.plot(path=os.path.join(experiment_dir, "metrics.png"), split_names=plot_splits, metric_names=plot_metrics, period=args.plot_metrics_period)
                metrics.save(path=os.path.join(experiment_dir, "metrics.json"))


if __name__ == '__main__':
    main()
