from sortedcontainers import SortedList
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from itertools import product, chain
import numpy as np
import copy
import collections


def score_function(TP, FP, FN, TN):
    return TP * 5 + FP * (-25) + FN * (-5) + TN * 0


def profit_scorer(y, y_pred):
    profit_matrix = {(0, 0): 0, (0, 1): -5, (1, 0): -25, (1, 1): 5}
    return sum(profit_matrix[(pred, actual)] for pred, actual in zip(y_pred, y))


class paramsearch:
    def __init__(self, pdict):
        self.pdict = {}

        for a, b in pdict.items():
            if isinstance(b, collections.Sequence) and not isinstance(b, str):
                self.pdict[a] = b
            else:
                self.pdict[a] = [b]

        self.results = SortedList()

    def grid_search(self, keys=None):

        if keys == None:
            keylist = self.pdict.keys()
        else:
            keylist = keys

        listoflists = []
        for key in keylist: listoflists.append([(key, i) for i in self.pdict[key]])
        for p in product(*listoflists):

            if len(self.results) > 0:
                template = self.results[-1][1]
            else:
                template = {a: b[0] for a, b in self.pdict.items()}

            if self.equaldict(dict(p), template): continue

            yield self.overwritedict(dict(p), template)

    def equaldict(self, a, b):
        for key in a.keys():
            if a[key] != b[key]: return False
        return True

    def overwritedict(self, new, old):
        old = copy.deepcopy(old)
        for key in new.keys(): old[key] = new[key]
        return old

    def register_result(self, result, params):
        self.results.add((result + np.random.randn() * 1e-10, params))

    def bestscore(self):
        return self.results[-1][0]

    def bestparam(self):
        return self.results[-1][1]


class ModelTunerBestParams:
    def crossvaltest_xg(self, params, X, y):
        skf = StratifiedKFold(n_splits=5)
        accuracy, score, f1 = [], [], []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            clf = XGBClassifier(**params)
            clf.fit(X_train, y_train)

            y_pred = np.array(clf.predict(X_test))
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

            accuracy.append(accuracy_score(y_test, y_pred))
            score.append(score_function(tp, fp, fn, tn))
            f1.append(f1_score(y_test, y_pred))

        return np.mean(score)

    def xgboost_param_tune(self, params, X, y):
        ps = paramsearch(params)
        for prms in chain(ps.grid_search(['n_estimators', 'learning_rate']),
                          ps.grid_search(['max_depth', 'min_child_weight'])):
            res = self.crossvaltest_xg(prms, X, y)
            ps.register_result(res, prms)
        return ps.bestparam(), ps.bestscore()

    def crossvaltest_cat(self, params, X, y):
        skf = StratifiedKFold(n_splits=5)
        accuracy, score, f1 = [], [], []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            clf = CatBoostClassifier(**params)
            clf.fit(X_train, y_train)

            y_pred = np.array(clf.predict(X_test))
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

            accuracy.append(accuracy_score(y_test, y_pred))
            score.append(score_function(tp, fp, fn, tn))
            f1.append(f1_score(y_test, y_pred))

        return np.mean(score)

    def cat_param_tune(self, params, X, y):
        ps = paramsearch(params)
        for prms in chain(ps.grid_search(['border_count']),
                          ps.grid_search(['l2_leaf_reg']),
                          ps.grid_search(['iterations', 'learning_rate']),
                          ps.grid_search(['depth'])):
            res = self.crossvaltest_cat(prms, X, y)
            ps.register_result(res, prms)
        return ps.bestparam(), ps.bestscore()

    def crossvaltest_ada(self, params, X, y):
        skf = StratifiedKFold(n_splits=5)
        accuracy, score, f1 = [], [], []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            clf = AdaBoostClassifier(**params)
            clf.fit(X_train, y_train)

            y_pred = np.array(clf.predict(X_test))
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

            accuracy.append(accuracy_score(y_test, y_pred))
            score.append(score_function(tp, fp, fn, tn))
            f1.append(f1_score(y_test, y_pred))

        return np.mean(score)

    def ada_param_tune(self, params, X, y):
        ps = paramsearch(params)
        for prms in chain(ps.grid_search(['n_estimators', 'learning_rate', 'algorithm'])):
            res = self.crossvaltest_ada(prms, X, y)
            ps.register_result(res, prms)
        return ps.bestparam(), ps.bestscore()

    def crossvaltest_lgb(self, params, X, y):
        skf = StratifiedKFold(n_splits=5)
        accuracy, score, f1 = [], [], []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            clf = LGBMClassifier(**params)
            clf.fit(X_train, y_train)

            y_pred = np.array(clf.predict(X_test))
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

            accuracy.append(accuracy_score(y_test, y_pred))
            score.append(score_function(tp, fp, fn, tn))
            f1.append(f1_score(y_test, y_pred))

        return np.mean(score)

    def lgb_param_tune(self, params, X, y):
        ps = paramsearch(params)
        for prms in chain(ps.grid_search(['num_iterations', 'learning_rate']),
                          ps.grid_search(['max_depth', 'num_leaves']),
                          ps.grid_search(['boosting'])):
            res = self.crossvaltest_lgb(prms, X, y)
            ps.register_result(res, prms)
        return ps.bestparam(), ps.bestscore()

    def crossvaltest_log_reg(self, params, X, y):
        skf = StratifiedKFold(n_splits=5)
        accuracy, score, f1 = [], [], []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            clf = LogisticRegression(**params)
            clf.fit(X_train, y_train)

            y_pred = np.array(clf.predict(X_test))
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

            accuracy.append(accuracy_score(y_test, y_pred))
            score.append(score_function(tp, fp, fn, tn))
            f1.append(f1_score(y_test, y_pred))

        return np.mean(score)

    def log_reg_param_tune(self, params, X, y):
        ps = paramsearch(params)
        for prms in chain(ps.grid_search(['C'])):
            res = self.crossvaltest_log_reg(prms, X, y)
            ps.register_result(res, prms)
        return ps.bestparam(), ps.bestscore()
