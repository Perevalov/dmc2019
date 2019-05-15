#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
import numpy as np
import random
from sklearn.linear_model import LogisticRegression as LR

# see https://sklearn-template.readthedocs.io/en/latest/user_guide.html
# for documentation about how to write a custom model






# takes a sklearn model and hard-predicts all instances with trust level > 2 as non fraud
# note: assumes trust level is the first feature in the feature list!
class TrustHard(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        #sklearn model object
        self.model = model
        
          
    def fit(self,X,y):
        self.classes_ = np.unique(y)
        
        # only train on instances with trust==1 or 2
        # note dataset might be normalized 
        # ASSUME trust is the first variable
        uniq  = np.unique(X[:,0])
        val2 = uniq[1] 
        print(val2)
        # this is what you expect from train set
        assert(val2 == -0.8202756459239328 )
        idx =  (X[:,0]<= val2)
        low_trust = X[idx,:]
        y_l = y[idx]
        self.model.fit(low_trust,y_l)
        return self

    def predict(self,X):
        preds = self.model.predict(X)
        uniq  = np.unique(X[:,0])
        val2 = uniq[1]
        assert(val2 == -0.8202756459239328 )
        idx =  (X[:,0]>val2)
        preds[idx] = 0
        return preds

    def predict_proba(self,X):
        pass
    


# this model takes any classifier model and a threshold as input
# the predict of this model is the predict of the input model but the
# prediction threshold is not 0.5 instead it is the custom threshold
class CustomModelWithThreshold(BaseEstimator, ClassifierMixin):
    def __init__(self, model, threshold):
        #sklearn model object
        self.model = model
        self.threshold = threshold
          
    def fit(self,X,y):
        self.classes_ = np.unique(y)
        self.model.fit(X,y)
        return self

    def predict(self,X):
        preds = self.model.predict_proba(X)
        preds[preds>self.threshold] = int(1)
        preds[preds<= self.threshold] = int(0)
        return preds[:,1]

    def predict_proba(self,X):
        return self.model.predict_proba(X)
    
    
# perceptron
# learns with the pocket variant of the perceptron learning algorithm 
# the learning process here is adapted to the DMC cost matrix    
class PerceptronLearner(BaseEstimator, ClassifierMixin):
    def __init__(self, epochs=250): 
        self.epochs = epochs
        self.coef_ = None
        
    def predict(self, X0):
         # add intercept (bias)
        X = np.zeros((X0.shape[0], X0.shape[1]+1))
        X[:,0] = 1
        X[:,1:] = X0
        
        # calc scores
        sc = np.dot(X,self.coef_)
        sc[sc>=0]=1
        sc[sc<0]=0
        return np.int64(sc)


    def predict_proba(self,X0):
        # there are actually no probabs so use the predicions
        pr = np.zeros((X0.shape[0],2))
        pr[:,1] = self.predict(X0)
        pr[:,0] = 1 - pr[:,1]
        return pr
    
    
    # implementation of Pocket algorithm for Perceptron learning
    def fit(self,X0,y):
        
        self.classes_ = np.unique(y)
        
        ### adapted Pocket algorithm
      
        # add intercept (bias)
        X = np.zeros((X0.shape[0], X0.shape[1]+1))
        X[:,0] = 1
        X[:,1:] = X0
        N, D = X.shape
         # change here to initialize weights in a different way
        w = np.zeros(D)
       
        
        y_true = y.copy()
        y_true[y_true==0] = -1
        epochcounter = 0
            
        profit_matrix = {(-1,-1): 0, (-1,1): -5, (1,-1): -25, (1,1): 5}
        profits = []
        weight_sequence = []
        
        # convert to list to prevent key error in cross val
        y_true = list(y_true)
        
        # note: we keep current profit over epochs
        # and only reset profit when we hit an error
        current_profit = 0
        print('Start pocket algorithm with {} epochs'.format(self.epochs))
        while(epochcounter<self.epochs):
            if epochcounter % 50 == 0:
                print('Epoch: {}'.format(epochcounter))
            epochcounter +=1
            instcounter = 0
            while(instcounter <= N):
                # draw instance randomly
                idx = random.choice(range(0,N,1))
                instcounter += 1
                ip = sum(X[idx]*w)
                # make prediction for the current instance
                if ip>=0:
                    y_hat_i = 1
                else:
                    y_hat_i = -1
                y_true_i = y_true[idx]
                current_profit = current_profit + profit_matrix[(y_hat_i,y_true_i)]
                
                # update weights in a perceptron learning fashion
                # different error types are not weighted (yet?)
                if (y_hat_i != y_true_i):
                     profits.append(current_profit)
                     current_profit = 0
                     weight_sequence.append(w)
                     w = w + y_true_i * X[idx]    
            
        # set best weights
        self.coef_ = weight_sequence[profits.index(max(profits))]
        return self
        

class OutlierRemover(BaseEstimator, ClassifierMixin):
    def __init__(self, model, thrs=(-0.1,0.2)):
        #sklearn model object
        self.model = model
        self.thrs = thrs
          
    def fit(self,X,y):
        self.classes_ = np.unique(y)
        #indx to filter out
                
        X = X.copy()
        y = y.copy()
        X['fraud'] = np.float64(y)
        keep1 = -1*((X['valuePerSecond']>self.thrs[0]) & (X['fraud']==1.0)) + 1
        keep2 = -1*((X['scannedLineItemsPerSecond']>self.thrs[1]) & (X['fraud']==1.0)) +1

        X = X[keep1.astype('bool')]
        X = X[keep2.astype('bool')]
        y = X.pop('fraud')
        self.model.fit(X,y)
        return self

    def predict(self,X):
        preds = self.model.predict(X)
        return preds

    #def predict_proba(self,X):
        #return self.model.predict_proba(X)
        



# takes a list of models and weights and predicts according to a voting scheme
class VoteClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, modellist=[], weights=[]):
       # assert(len(modellist)==len(weights))
        self.set_params(modellist=modellist, weights=weights)

    def fit(self, X, y):
        self.classes_ = np.unique(y)   
        modellist = self.get_params()['modellist']
        for model in modellist:
            model.fit(X,y)
        return self

    def predict(self, X):
        preds = np.zeros(len(X))
        idx = 0
        weights = self.get_params()['weights']
        for model in self.get_params()['modellist']:
            preds = preds + weights[idx] * model.predict(X)
            idx += 1
        preds = preds / sum(weights)
        preds[preds>0.5] = 1
        preds[preds<=0.5] = 0
        return preds
        
            
      
     
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    