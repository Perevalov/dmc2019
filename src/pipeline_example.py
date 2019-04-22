from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import uuid

import mlflow
import mlflow.sklearn

import sys
sys.path.append('..')
from utils.ScoreFunction import score_function
random_seed = 42
np.random.seed(random_seed)
mlflow.set_tracking_uri("mlflow-storage")

if __name__ == "__main__":
    
    data = pd.read_csv("../data/train.csv",sep='|')
    X = np.array(data.drop(["fraud"],axis=1))
    y = np.array(data["fraud"])
    
    skf = StratifiedKFold(n_splits=5)
    scores, f1_scores = [],[]
    for train_index, test_index in skf.split(X, y):
        clf = LogisticRegression(random_state=random_seed)
        clf.fit(X[train_index], y[train_index])
        y_pred = clf.predict(X[test_index])
        tn, fp, fn, tp = confusion_matrix(y[test_index], y_pred).ravel()
        scores.append(score_function(tp,fp,fn,tn))
        f1_scores.append(f1_score(y[test_index], y_pred))
    
    dataset_id = str(uuid.uuid4())
    data.to_csv("../data/mlflow-datasets/{0}.csv".format(dataset_id))
    
    with mlflow.start_run():
        mlflow.log_param("dataset_id", dataset_id)
        
        mean_score = np.array(scores).mean()
        mlflow.log_metric("mean_score", mean_score)

        mean_f1 = np.array(f1_scores).mean()
        mlflow.log_metric("mean_f1", mean_score)

        clf = LogisticRegression(random_state=random_seed)
        clf.fit(X, y)
        mlflow.sklearn.log_model(clf, "model")