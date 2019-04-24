from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, recall_score
import sys
sys.path.append('..')
from utils.ScoreFunction import score_function

def compute_score_f1_recall(y_true,y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    score_func = score_function(tp,fp,fn,tn)
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return score,f1,recall

def compare_results(m1 = (y1_true,y1_pred),m2 = (y2_true,y2_pred)):
    """
    Возвращает True если первые результаты лучше вторых
    """
    score_1, f1_1, recall_1 = compute_score_f1_recall(y1_true,y1_pred)
    score_2, f1_2, recall_2 = compute_score_f1_recall(y2_true,y2_pred)
    
    if score_1 >= score_2 and f1_1 >= f1_2 and recall_1 >= recall_2:
        return True
    else:
        return False
    