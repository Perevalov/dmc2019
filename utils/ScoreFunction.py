from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold


def score_function(TP,FP,FN,TN):
    return TP*5 + FP*(-25) + FN*(-5) + TN*0

def other_score_function(X,y,model):
    """
    В найденных источниках значение данной функции равно 29.989
    """
    cv = StratifiedKFold(n_splits=10, random_state=42)
    def profit_scorer(y, y_pred):
        profit_matrix = {(0,0): 0, (0,1): -5, (1,0): -25, (1,1): 5}
        return sum(profit_matrix[(pred, actual)] for pred, actual in zip(y_pred, y))

    profit_scoring = make_scorer(profit_scorer, greater_is_better=True)
    score = cross_validate(model, X, y=y, cv=10, scoring=profit_scoring)['test_score'].mean()
    check = True if score > 29.989 else False
    
    return score, check