from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from utils.model_tuning_utils import ModelTunerBestParams, score_function, profit_scorer
from utils.model_tuning_utils_v2 import ModelStacker
import pandas as pd
import numpy as np
import warnings
import time
import sys

pd.options.mode.chained_assignment = None

selected_features = [
    "trustLevel",
    "trustLevel quantityModifications",
    "trustLevel withoutRegisPerPosition",
    "totalScanTimeInSeconds quantityModifications",
    "lineItemVoids^2",
    "lineItemVoids valuePerSecond",
    "lineItemVoids totalScanned",
    "scansWithoutRegistration avgValuePerScan",
    "valuePerSecond avgValuePerScan withoutRegisPerPosition",
    "lineItemVoidsPerPosition totalScanTimeInSecondsStdNorm^2",
    "lineItemVoidsPerPosition totalScanTimeInSecondsStdNorm totalScanned",
    "lineItemVoidsPerPosition totalScanTimeInSecondsStdNorm quantityModsPerPosition"
]

sc = StandardScaler()
sys.path.append('..')
warnings.filterwarnings("ignore")
mt = ModelTunerBestParams()
ms = ModelStacker(XGBClassifier, 5)

params_cat = {'depth': [1, 3, 5, 7],
              'iterations': [250, 500, 750, 1000],
              'learning_rate': [0.03, 0.001, 0.01, 0.1],
              'l2_leaf_reg': [1, 5, 10],
              'border_count': [2, 5, 10, 20, 50],
              'thread_count': 4,
              'silent': True}

params_lgb = {'max_depth': [1, 3, 5, -1],
              'boosting_type': ['gbdt', 'dart', 'goss'],
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
    test_new = test.copy()

    THRESHOLDS = {'scannedLineItemsPerSecond': 1, 'valuePerSecond': 1, 'lineItemVoidsPerPosition': 1}

    for x in list(THRESHOLDS.keys()):
        X[x] = X[x].clip(lower=X[x].quantile(0.01), upper=X[x].quantile(0.99))
        test_new[x] = test_new[x].clip(lower=test_new[x].quantile(0.01), upper=test_new[x].quantile(0.99))

    X['totalScanTimeInSecondsStdNorm'] = (X['totalScanTimeInSeconds'] - X['totalScanTimeInSeconds'].mean()) / X[
        'totalScanTimeInSeconds'].std()

    X['totalScanned'] = X['scannedLineItemsPerSecond'] * X['totalScanTimeInSeconds']
    X['avgTimePerScan'] = 1 / X['scannedLineItemsPerSecond']
    X['avgValuePerScan'] = X['avgTimePerScan'] * X['valuePerSecond']
    X['withoutRegisPerPosition'] = X['scansWithoutRegistration'] * X['totalScanned']
    X['quantityModsPerPosition'] = X['quantityModifications'] / X['totalScanned']

    test_new['totalScanTimeInSecondsStdNorm'] = (test_new['totalScanTimeInSeconds'] - test_new[
        'totalScanTimeInSeconds'].mean()) / test_new['totalScanTimeInSeconds'].std()

    test_new['totalScanned'] = test_new['scannedLineItemsPerSecond'] * test_new['totalScanTimeInSeconds']
    test_new['avgTimePerScan'] = 1 / test_new['scannedLineItemsPerSecond']
    test_new['avgValuePerScan'] = test_new['avgTimePerScan'] * test_new['valuePerSecond']
    test_new['withoutRegisPerPosition'] = test_new['scansWithoutRegistration'] * test_new['totalScanned']
    test_new['quantityModsPerPosition'] = test_new['quantityModifications'] / test_new['totalScanned']

    for column in data.columns:
        data[column] = data[column].astype('float64')

    for column in test.columns:
        test[column] = test[column].astype('float64')

    pca = PCA(n_components=8)
    pca.fit(data.drop(["fraud"], axis=1).append(test, ignore_index=True))

    pca_ = pca.transform(data.drop(["fraud"], axis=1))

    X = np.array(X)
    X = pd.DataFrame(np.concatenate((X, pca_), axis=1))
    y = pd.DataFrame(np.array(y))
    for column in X.columns:
        X[column] = X[column].astype('float64')

    pca_ = pca.transform(test)

    test_new = np.array(test_new)
    test_new = pd.DataFrame(np.concatenate((test_new, pca_), axis=1))
    for column in test_new.columns:
        test_new[column] = test_new[column].astype('float64')

    return X, y, test_new


def prepare_data_v2():
    data = pd.read_csv("data/train.csv", sep='|')
    X_test = pd.read_csv("data/test.csv", sep='|')
    X = data.drop(["fraud"], axis=1)
    y = data["fraud"]

    THRESHOLDS = {'scannedLineItemsPerSecond': 1, 'valuePerSecond': 1, 'lineItemVoidsPerPosition': 1}

    for x in list(THRESHOLDS.keys()):
        X[x] = X[x].clip(lower=X[x].quantile(0.01), upper=X[x].quantile(0.99))
        X_test[x] = X_test[x].clip(lower=X_test[x].quantile(0.01), upper=X_test[x].quantile(0.99))

    X['totalScanTimeInSecondsStdNorm'] = (X['totalScanTimeInSeconds'] - X['totalScanTimeInSeconds'].mean()) / X[
        'totalScanTimeInSeconds'].std()

    X['totalScanned'] = X['scannedLineItemsPerSecond'] * X['totalScanTimeInSeconds']
    X['avgTimePerScan'] = 1 / X['scannedLineItemsPerSecond']
    X['avgValuePerScan'] = X['avgTimePerScan'] * X['valuePerSecond']
    X['withoutRegisPerPosition'] = X['scansWithoutRegistration'] * X['totalScanned']
    X['quantityModsPerPosition'] = X['quantityModifications'] / X['totalScanned']

    X_test['totalScanTimeInSecondsStdNorm'] = (X_test['totalScanTimeInSeconds'] - X_test[
        'totalScanTimeInSeconds'].mean()) / X_test['totalScanTimeInSeconds'].std()

    X_test['totalScanned'] = X_test['scannedLineItemsPerSecond'] * X_test['totalScanTimeInSeconds']
    X_test['avgTimePerScan'] = 1 / X_test['scannedLineItemsPerSecond']
    X_test['avgValuePerScan'] = X_test['avgTimePerScan'] * X_test['valuePerSecond']
    X_test['withoutRegisPerPosition'] = X_test['scansWithoutRegistration'] * X_test['totalScanned']
    X_test['quantityModsPerPosition'] = X_test['quantityModifications'] / X_test['totalScanned']

    cols = list(X.columns) + [1, 2]

    # generate features and rescale
    polyFeatures = PolynomialFeatures(3, interaction_only=False)
    polyFeatures.fit(X.append(X_test, ignore_index=True))

    X_poly = polyFeatures.transform(X)
    X_test_poly = polyFeatures.transform(X_test)

    # remove the first var because it is the constant term
    X_poly = X_poly[:, 1:]
    X_test_poly = X_test_poly[:, 1:]

    features = polyFeatures.get_feature_names(input_features=X.columns)[1:]

    X_poly = pd.DataFrame(X_poly, columns=features)
    X_test_poly = pd.DataFrame(X_test_poly, columns=features)

    X_tmp = X.copy()
    X_test_tmp = X_test.copy()

    for f in selected_features:
        X_tmp = pd.concat([X_tmp, pd.Series(X_poly[f])], axis=1)
        X_test_tmp = pd.concat([X_test_tmp, pd.Series(X_test_poly[f])], axis=1)

    X = pd.DataFrame(np.array(X_tmp))
    y = pd.DataFrame(np.array(y))
    test = pd.DataFrame(np.array(X_test_tmp))

    return X, y, test


def get_best_params_of_models():
    best_params, best_scores = {}, {}
    start = time.time()
    best_params_cat, best_score_cat = mt.cat_param_tune(params_cat, X, y)
    print('CatBoost ready. Best score = {0}. Time = {1}'.format(best_score_cat, time.time() - start))
    start = time.time()
    best_params_xg, best_score_xg = mt.xgboost_param_tune(params_xg, X, y)
    print('XgBoost ready. Best score = {0}. Time = {1}'.format(best_score_xg, time.time() - start))
    start = time.time()
    best_params_lgb, best_score_lgb = mt.lgb_param_tune(params_lgb, X, y)
    print('LightGbm ready. Best score = {0}. Time = {1}'.format(best_score_lgb, time.time() - start))
    start = time.time()
    best_params_ada, best_score_ada = mt.ada_param_tune(params_ada, X, y)
    print('AdaBoost ready. Best score = {0}. Time = {1}'.format(best_score_ada, time.time() - start))
    start = time.time()
    best_params_log_reg, best_score_log_reg = mt.log_reg_param_tune(params_log_reg, X, y)
    print('LogisticRegression ready. Best score = {0}. Time = {1}'.format(best_score_log_reg, time.time() - start))
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


def final_predict(best_params, X, y, test):
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

    clf = LogisticRegression(**best_params['log_reg'])
    clf.fit(X, y)
    y_pred_log_reg = np.array(clf.predict(test))

    clf = XGBClassifier(**best_params['xg'])
    clf.fit(X, y)
    y_pred_xg = np.array(clf.predict(test))

    clf = AdaBoostClassifier(**best_params['ada'])
    clf.fit(X, y)
    y_pred_ada = np.array(clf.predict(test))

    clf = CatBoostClassifier(**best_params['cat'])
    clf.fit(X, y)
    y_pred_cat = np.array(clf.predict(test))

    clf = LGBMClassifier(**best_params['lgb'])
    clf.fit(X, y)
    y_pred_lgb = np.array(clf.predict(test))

    y_pred = []
    for i in range(len(y_pred_cat)):
        temp_prediction = [float(y_pred_log_reg[i]), float(y_pred_xg[i]), float(y_pred_lgb[i]),
                           float(y_pred_ada[i]), float(y_pred_cat[i])]
        y_pred.append(most_common(temp_prediction))
    pd.DataFrame(list(map(int, y_pred)), columns=['fraud']).to_csv('prediction.csv', index=False)


X, y, test = prepare_data()
best_params, best_scores = get_best_params_of_models()
# model_testing(best_params)
final_predict(best_params, X, y, test)
# best_params = ms.param_tune(params_xg, X, y)
# model_testing_v2(best_params)
