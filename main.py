from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from utils.model_tuning_utils import ModelTunerBestParams, score_function
from utils.model_tuning_utils_v2 import ModelStacker
import pandas as pd
import numpy as np
import warnings
import sys

sc = StandardScaler()
sys.path.append('..')
warnings.filterwarnings("ignore")
mt = ModelTunerBestParams()
ms = ModelStacker(LGBMClassifier, 5)

params_cat = {'depth': [1, 3, 5, 7],
              'iterations': [250, 500, 750, 1000],
              'learning_rate': [0.03, 0.001, 0.01, 0.1],
              'l2_leaf_reg': [1, 5, 10],
              'border_count': [2, 5, 10, 20, 50],
              'thread_count': 4,
              'silent': True}

params_lgb = {'max_depth': [1, 3, 5, -1],
              'boosting': ['gbdt', 'dart', 'goss'],
              'num_leaves': [20, 31, 40, 50],
              'num_iterations': [100, 400, 600, 1000, 1500],
              'learning_rate': [0.03, 0.001, 0.01, 0.1]}

params_xg = {'max_depth': [3, 5],
             'n_estimators': [100, 500, 700, 900],
             'learning_rate': [0.03, 0.001, 0.01, 0.1],
             'min_child_weight': [1, 2, 3, 4]}

params_ada = {'learning_rate': [0.03, 0.001, 0.01, 0.1],
              'n_estimators': [100, 300, 500, 700, 900, 1000],
              'algorithm': ['SAMME', 'SAMME.R']}

params_log_reg = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}


def most_common(lst):
    return max(set(lst), key=lst.count)


def prepare_data():
    data = pd.read_csv("data/train.csv", sep='|')
    test = pd.read_csv("data/test.csv", sep='|')
    X = data.drop(["fraud"], axis=1)
    y = data["fraud"]

    X['totalScanned'] = X['scannedLineItemsPerSecond'] * X['totalScanTimeInSeconds']
    X['avgTimePerScan'] = 1 / X['scannedLineItemsPerSecond']
    X['avgValuePerScan'] = X['avgTimePerScan'] * X['valuePerSecond']
    X['withoutRegisPerPosition'] = X['scansWithoutRegistration'] * X['totalScanned']
    X['quantityModsPerPosition'] = X['quantityModifications'] / X['totalScanned']

    X['grandTotalCat'] = pd.cut(X['grandTotal'], 10, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    X['totalScanTimeInSecondsCat'] = pd.cut(X['totalScanTimeInSeconds'], 2, labels=[1, 2])
    X['lineItemVoidsPerPositionCat'] = pd.cut(X['lineItemVoidsPerPosition'], 10, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    X['avgTimePerScan'] = pd.cut(X['avgTimePerScan'], 4, labels=[1, 2, 3, 4])

    pca = PCA(n_components=8)
    pca.fit(data.drop(["fraud"], axis=1).append(test, ignore_index=True))

    pca_ = pca.transform(data.drop(["fraud"], axis=1))

    X = np.array(X)
    X = pd.DataFrame(np.concatenate((X, pca_), axis=1))
    y = pd.DataFrame(np.array(y))
    for column in X.columns:
        X[column] = X[column].astype('float64')
    return X, y


def get_best_params_of_models():
    best_params, best_scores = {}, {}
    best_params_xg, best_score_xg = mt.xgboost_param_tune(params_xg, X, y)
    print('XgBoost ready. Best score = {0}'.format(best_score_xg))
    best_params_cat, best_score_cat = mt.cat_param_tune(params_cat, X, y)
    print('CatBoost ready. Best score = {0}'.format(best_score_cat))
    best_params_lgb, best_score_lgb = mt.lgb_param_tune(params_lgb, X, y)
    print('LightGbm ready. Best score = {0}'.format(best_score_lgb))
    best_params_ada, best_score_ada = mt.ada_param_tune(params_ada, X, y)
    print('AdaBoost ready. Best score = {0}'.format(best_score_ada))
    best_params_log_reg, best_score_log_reg = mt.log_reg_param_tune(params_log_reg, X, y)
    print('LogisticRegression ready. Best score = {0}'.format(best_score_log_reg))
    best_params['cat'], best_params['lgb'], best_params['xg'], best_params['ada'], best_params['log_reg'] = \
        best_params_cat, best_params_lgb, best_params_xg, best_params_ada, best_params_log_reg
    best_scores['cat'], best_scores['lgb'], best_scores['xg'], best_scores['ada'], best_scores['log_reg'] = \
        best_score_cat, best_score_lgb, best_score_xg, best_score_ada, best_score_log_reg
    return best_params, best_scores


def model_testing(best_params):
    skf = StratifiedKFold(n_splits=5)
    accuracy, score, f1 = [], [], []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        clf = LogisticRegression(**best_params['log_reg'])
        clf.fit(X_train, y_train)
        y_pred_log_reg = np.array(clf.predict(X_test))

        clf = XGBClassifier(**best_params['xg'])
        clf.fit(X_train, y_train)
        y_pred_xg = np.array(clf.predict(X_test))

        clf = AdaBoostClassifier(**best_params['ada'])
        clf.fit(X_train, y_train)
        y_pred_ada = np.array(clf.predict(X_test))

        clf = CatBoostClassifier(**best_params['cat'])
        clf.fit(X_train, y_train)
        y_pred_cat = np.array(clf.predict(X_test))

        clf = LGBMClassifier(**best_params['lgb'])
        clf.fit(X_train, y_train)
        y_pred_lgb = np.array(clf.predict(X_test))

        y_pred = []
        for i in range(len(y_pred_cat)):
            temp_prediction = [float(y_pred_log_reg[i]), float(y_pred_xg[i]), float(y_pred_lgb[i]),
                               float(y_pred_ada[i]), float(y_pred_cat[i])]
            y_pred.append(most_common(temp_prediction))
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        accuracy.append(accuracy_score(y_test, y_pred))
        score.append(score_function(tp, fp, fn, tn))
        f1.append(f1_score(y_test, y_pred))

    print('Mean Score = {0}'.format(np.mean(score)))
    print('Mean Accuracy = {0}'.format(np.mean(accuracy)))
    print('Mean F1_score = {0}'.format(np.mean(f1)))


def model_testing_v2(best_params):
    skf = StratifiedKFold(n_splits=5)
    accuracy, score, f1 = [], [], []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        y_pred, y_pred_temp, y_pred_temp2, temp_prediction = [], [], [], []
        for i in range(len(best_params)):
            clf = ms.classifier(**best_params[i][1])
            clf.fit(X_train, y_train)
            y_pred_temp = np.array(clf.predict(X_test))
            y_pred_temp2.append(y_pred_temp)

        y_pred = []
        for i in range(len(y_pred_temp)):
            for j in range(len(y_pred_temp2)):
                temp_prediction.append(y_pred_temp2[j][i])
            y_pred.append(most_common(temp_prediction))
            temp_prediction = []
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        accuracy.append(accuracy_score(y_test, y_pred))
        score.append(score_function(tp, fp, fn, tn))
        f1.append(f1_score(y_test, y_pred))

    print('Mean Score = {0}'.format(np.mean(score)))
    print('Mean Accuracy = {0}'.format(np.mean(accuracy)))
    print('Mean F1_score = {0}'.format(np.mean(f1)))


X, y = prepare_data()
#best_params, best_scores = get_best_params_of_models()
#model_testing(best_params)
best_params = ms.param_tune(params_lgb, X, y)
model_testing_v2(best_params)