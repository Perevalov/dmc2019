"""
Посмотреть как считается Confusion Matrix в Sklearn
"""

def score_function(TP,FP,FN,TN):
    return TP*5 + FP*(-25) + FN*(-5) + TN*0