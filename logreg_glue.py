import pandas as pd
import scipy.sparse
import os
import wandb

from argparse import ArgumentParser
from datasets import load_dataset
from pprint import pprint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def get_args():
    parser = ArgumentParser(description="Train logistic regression on a GLUE task")
    parser.add_argument("--task", type=str, help="Any GLUE binary classification task")
    parser.add_argument("--seed", type=int, help="Random seed used for training")
    parser.add_argument("--log_file", type=str, default="results.csv", help="Filename for storing results")
    parser.add_argument("--use_wandb", action="store_true", help="Whether to enable wandb logging")
    parser.add_argument("--wandb_entity", type=str, default=None, 
                        help="Entity parameter for initializing wandb logging")
    parser.add_argument("--wandb_project", type=str, default=None, 
                        help="Project parameter for initializing wandb logging")

    return parser.parse_args()


# Log metrics into pandas DataFrame
def log_metrics(metrics, filename, task, seed):
    metrics_df = pd.DataFrame({
        "model": ["logistic_regression"],
        "task": [task],
        "seed": [seed],
        "train_accuracy": [metrics["train/accuracy"]],
        "test_accuracy": [metrics["test/accuracy"]],
        "train_f1": [metrics["train/f1"]],
        "test_f1": [metrics["test/f1"]]
    })

    if os.path.isfile(filename):
        metrics_df = pd.concat([pd.read_csv(filename), metrics_df])
    
    metrics_df.to_csv(filename, index=False)


def compute_metrics(y_true, y_pred, prefix):
    return {
        f"{prefix}/accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}/precision": precision_score(y_true, y_pred),
        f"{prefix}/recall": recall_score(y_true, y_pred),
        f"{prefix}/f1": f1_score(y_true, y_pred),
    }


def main():
    args = get_args()

    # Load and preprocess dataset using TF-IDF vectorization
    dataset = load_dataset("glue", args.task)
    if args.task == "rte":
        vectorizer1 = TfidfVectorizer().fit(dataset["train"]["sentence1"])
        X_train1 = vectorizer1.transform(dataset["train"]["sentence1"])
        X_test1 = vectorizer1.transform(dataset["validation"]["sentence1"])
        vectorizer2 = TfidfVectorizer().fit(dataset["train"]["sentence2"])
        X_train2 = vectorizer2.transform(dataset["train"]["sentence2"])
        X_test2 = vectorizer2.transform(dataset["validation"]["sentence2"])
        X_train = scipy.sparse.hstack((X_train1, X_train2))
        X_test = scipy.sparse.hstack((X_test1, X_test2))
    else:
        vectorizer = TfidfVectorizer().fit(dataset["train"]["sentence"])
        X_train = vectorizer.transform(dataset["train"]["sentence"])
        X_test = vectorizer.transform(dataset["validation"]["sentence"])

    # Train logistic regression and get predictions
    model = LogisticRegression(max_iter=1000, random_state=args.seed).fit(X_train, dataset["train"]["label"])
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Compute metrics
    metrics = compute_metrics(dataset["train"]["label"], y_train_pred, "train")
    metrics.update(compute_metrics(dataset["validation"]["label"], y_test_pred, "test"))
    log_metrics(metrics, args.log_file, args.task, args.seed)

    # Log metrics to wandb if needed or just print
    if args.use_wandb:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            tags=["logreg", args.task],
        )
        wandb.log(metrics)
        run.finish()
    else:
        print("Final metrics:")
        pprint(metrics)


if __name__ == "__main__":
    main()
