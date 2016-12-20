#!/usr/bin/python

import os
import numpy as np
import pandas as pd
import gzpickle
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import f1_score

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.layers.core import Dropout

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

# from keras.optimizers import Adam
# from keras.optimizers import SGD
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers.core import Dropout
# from keras.wrappers.scikit_learn import KerasRegressor
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
# from sklearn.pipeline import Pipeline


#### NON - ENSEMBLE LEARNERS #####################################################################
class MultiTargetLogisticRegression(object):
    def __init__(self, **kvargs):
        self.kvargs = kvargs

    def fit(self, X, y):
        self.num_targets = y.shape[1]
        self.models = [ LogisticRegression(**self.kvargs).fit(X, y[:,tgtidx]) for tgtidx in xrange(0, self.num_targets) ]
        return self

    def predict(self, X):
        # Allocate space for predicted target dataframe
        ypred = np.zeros((X.shape[0], self.num_targets), dtype=int)
        for tgtidx in xrange(0, self.num_targets):
            ypred[:, tgtidx] = self.models[tgtidx].predict(X)
        return ypred

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean([ f1_score(y[:, tgtidx], y_pred[:, tgtidx]) for tgtidx in xrange(0, self.num_targets) ])
    

class BenchMarkModel(object):
    """ Accepts SingleStockDataSet object as input and
        trains the benchmark model on train set and
        evaluate on test set. """
    def __init__(self, dset):
        self.dset = dset
        self.trainX, _, self.testX, self.trainY, _, self.testY = self.dset.get_train_val_test_sets(0.9, 0.0, 0.1)
        self.num_targets = self.trainY.shape[1]
        self.predictions = None
        self.classifier = MultiTargetLogisticRegression(C=1)

    def fit(self):
        self.classifier.fit(self.trainX, self.trainY)
        #print 'BenchMarkModel : train score = %f' % np.mean(self.classifier.score(self.trainX, self.trainY))

    def predict(self):
        self.predictions = self.classifier.predict(self.testX)
        return self.predictions

    def score(self):
        if self.predictions is None:
            print "predict() needs to be called first"
            return 0.0
        return np.mean([ f1_score(self.testY[:,tgtidx], self.predictions[:, tgtidx]) for tgtidx in xrange(0, self.num_targets) ])

    def evaluate(self):
        """ fits the model, predicts the targets and returns evaluation score """
        self.fit()
        self.predict()
        return self.score()

class MultiTargetSVC(object):
    def __init__(self, C, gamma='auto'):
        self.C = C
        self.gamma = gamma

    def fit(self, X, y):
        self.num_targets = y.shape[1]
        self.models = [ SVC(C=self.C, gamma=self.gamma).fit(X, y[:,tgtidx]) for tgtidx in xrange(0, self.num_targets) ]
        return self

    def predict(self, X):
        # Allocate space for predicted target dataframe
        ypred = np.zeros((X.shape[0], self.num_targets), dtype=int)
        for tgtidx in xrange(0, self.num_targets):
            ypred[:, tgtidx] = self.models[tgtidx].predict(X)
        return ypred

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean([ f1_score(y[:, tgtidx], y_pred[:, tgtidx]) for tgtidx in xrange(0, self.num_targets) ])

class SVCModel(object):
    """ Accepts SingleStockDataSet object as input and
        trains the benchmark model on train set and
        evaluate on test set. """
    def __init__(self, dset):
        self.dset = dset
        self.trainX, self.valX, self.testX, self.trainY, self.valY, self.testY = self.dset.get_train_val_test_sets(0.8, 0.1, 0.1)
        self.num_targets = self.trainY.shape[1]
        self.predictions = None
        self.parmfile = '../data/pkl/parms/%s_SVCModel.pkl.gz' % self.dset.ticker

    def fit(self):
        Gset = 2.0**np.arange(-5., 6., 1.)
        Cset = 2.0**np.arange(-5., 6., 1.)
        bestscore = -9999
        bestG = 1.0
        bestC = 1.0
        #lst = []
        if not os.path.exists(self.parmfile):
            for C in Cset:
                for G in Gset:
                    tmpmodel = MultiTargetSVC(C=C, gamma=G)
                    tmpmodel.fit(self.trainX, self.trainY)
                    scr = tmpmodel.score(self.valX, self.valY)
                    if scr > bestscore:
                        bestscore = scr
                        bestG = G
                        bestC = C
            gzpickle.save([bestC, bestG], self.parmfile)
        else:
            print 'Reading parameters from pklfile = %s' % self.parmfile
            bestC, bestG = gzpickle.load( self.parmfile )
        print 'Best C = %f, G = %f' % (bestC, bestG)
        self.trainX, self.valX, self.testX, self.trainY, self.valY, self.testY = self.dset.get_train_val_test_sets(0.9, 0, 0.1)
        self.classifier = MultiTargetSVC(C=bestC, gamma=bestG).fit(self.trainX, self.trainY)

    def predict(self):
        self.predictions = self.classifier.predict(self.testX)
        return self.predictions

    def score(self):
        if self.predictions is None:
            print "predict() needs to be called first"
            return 0.0
        return np.mean([ f1_score(self.testY[:,tgtidx], self.predictions[:, tgtidx]) for tgtidx in xrange(0, self.num_targets) ])

    def evaluate(self):
        """ fits the model, predicts the targets and returns evaluation score """
        self.fit()
        self.predict()
        return self.score()

class KNNClassificationModel(object):
    """ Accepts SingleStockDataSet object as input and
        trains the benchmark model on train set and
        evaluate on test set. """
    def __init__(self, dset):
        self.dset = dset
        self.trainX, self.valX, self.testX, self.trainY, self.valY, self.testY = self.dset.get_train_val_test_sets(0.8, 0.1, 0.1)
        self.predictions = None
        self.parmfile = '../data/pkl/parms/%s_KNNClassificationModel.pkl.gz' % self.dset.ticker
        self.num_targets = self.trainY.shape[1]

    def fit(self):
        kset = list(xrange(2, 20))
        bestscore = -9999
        bestk = 1.0
        if not os.path.exists(self.parmfile):
            for k in kset:
                tmpmodel = KNeighborsClassifier(n_neighbors=k)
                tmpmodel.fit(self.trainX, self.trainY)
                ypred = tmpmodel.predict(self.valX)
                scr = np.mean([ f1_score(self.valY[:,tgtidx], ypred[:, tgtidx]) for tgtidx in xrange(0, self.num_targets) ])
                if scr > bestscore:
                    bestscore = scr
                    bestk = k
            gzpickle.save(bestk, self.parmfile)
        else:
            print 'Reading parameters from pklfile = %s' % self.parmfile
            bestk = gzpickle.load( self.parmfile )
        print 'Best k = %f' % bestk
        self.trainX, self.valX, self.testX, self.trainY, self.valY, self.testY = self.dset.get_train_val_test_sets(0.9, 0, 0.1)
        self.classifier = KNeighborsClassifier(n_neighbors=bestk).fit(self.trainX, self.trainY)

    def predict(self):
        self.predictions = self.classifier.predict(self.testX)
        return self.predictions

    def score(self):
        if self.predictions is None:
            print "predict() needs to be called first"
            return 0.0
        return np.mean([ f1_score(self.testY[:,tgtidx], self.predictions[:, tgtidx]) for tgtidx in xrange(0, self.num_targets) ])

    def evaluate(self):
        """ fits the model, predicts the targets and returns evaluation score """
        self.fit()
        self.predict()
        return self.score()

class DTClassificationModel(object):
    """ Accepts SingleStockDataSet object as input and
        trains the benchmark model on train set and
        evaluate on test set. """
    def __init__(self, dset):
        self.dset = dset
        self.trainX, self.valX, self.testX, self.trainY, self.valY, self.testY = self.dset.get_train_val_test_sets(0.8, 0.1, 0.1)
        self.predictions = None
        self.parmfile = '../data/pkl/parms/%s_DTClassificationModel.pkl.gz' % self.dset.ticker
        self.num_targets = self.trainY.shape[1]

    def fit(self):
        mdset = [5, 7, 9, 10, 12, 15, 20, 25, 30, 35, 40, 45]
        bestscore = -9999
        bestmd = 5.0
        if not os.path.exists(self.parmfile):
            for md in mdset:
                tmpmodel = DecisionTreeClassifier(max_depth=md, random_state=16)
                tmpmodel.fit(self.trainX, self.trainY)
                ypred = tmpmodel.predict(self.valX)
                scr = np.mean([ f1_score(self.valY[:,tgtidx], ypred[:, tgtidx]) for tgtidx in xrange(0, self.num_targets) ])
                if scr > bestscore:
                    bestscore = scr
                    bestmd = md
            gzpickle.save(bestmd, self.parmfile)
        else:
            print 'Reading parameters from pklfile = %s' % self.parmfile
            bestmd = gzpickle.load( self.parmfile )
        print 'Best md = %f' % bestmd
        self.trainX, self.valX, self.testX, self.trainY, self.valY, self.testY = self.dset.get_train_val_test_sets(0.9, 0, 0.1)
        self.classifier = DecisionTreeClassifier(criterion='gini', max_depth=bestmd, random_state=16).fit(self.trainX, self.trainY)

    def predict(self):
        self.predictions = self.classifier.predict(self.testX)
        return self.predictions

    def score(self):
        if self.predictions is None:
            print "predict() needs to be called first"
            return 0.0
        return np.mean([ f1_score(self.testY[:,tgtidx], self.predictions[:, tgtidx]) for tgtidx in xrange(0, self.num_targets) ])

    def evaluate(self):
        """ fits the model, predicts the targets and returns evaluation score """
        self.fit()
        self.predict()
        return self.score()

class MultiTargetGNB(object):
    def __init__(self):
        pass

    def fit(self, X, y):
        self.num_targets = y.shape[1]
        self.models = [ GaussianNB().fit(X, y[:,tgtidx]) for tgtidx in xrange(0, self.num_targets) ]
        return self

    def predict(self, X):
        # Allocate space for predicted target dataframe
        ypred = np.zeros((X.shape[0], self.num_targets), dtype=int)
        for tgtidx in xrange(0, self.num_targets):
            ypred[:, tgtidx] = self.models[tgtidx].predict(X)
        return ypred

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean([ f1_score(y[:, tgtidx], y_pred[:, tgtidx]) for tgtidx in xrange(0, self.num_targets) ])

class MultiTargetEnsembleClassifier(object):
    """ Ensemble Classifier of sklearn does not support multiple targets,
        so this class wraps ensemble learner to support multiple targets"""
    def __init__(self, ensemble_learner_class, base_estimator=None, random_state=16, **kvargs):
        self.base_estimator = base_estimator
        self.random_state = random_state
        self.ensemble_learner_class = ensemble_learner_class
        self.kvargs = kvargs

    def fit(self, X, y):
        self.num_targets = y.shape[1]
        self.models = [ self.ensemble_learner_class(base_estimator=self.base_estimator, random_state=self.random_state,
                                              **self.kvargs).fit(X, y[:,tgtidx]) for tgtidx in xrange(0, self.num_targets) ]
        return self

    def predict(self, X):
        # Allocate space for predicted target dataframe
        y = np.zeros((X.shape[0], self.num_targets))
        for tgtidx in xrange(0, self.num_targets):
            y[:, tgtidx] = self.models[tgtidx].predict(X)
        return y

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean([ f1_score(y[:, tgtidx], y_pred[:, tgtidx]) for tgtidx in xrange(0, self.num_targets) ])

class RFClassifierModel(object):
    """ Accepts SingleStockDataSet object as input and
        trains the benchmark model on train set and
        evaluate on test set. """
    def __init__(self, dset):
        self.dset = dset
        self.trainX, self.valX, self.testX, self.trainY, self.valY, self.testY = self.dset.get_train_val_test_sets(0.9, 0, 0.1)
        self.predictions = None
        self.num_targets = self.trainY.shape[1]

    def fit(self):
        self.classifier = MultiTargetEnsembleClassifier(AdaBoostClassifier, base_estimator=DecisionTreeClassifier(max_depth=3), n_estimators=50, learning_rate=1.5, random_state=None).fit(self.trainX, self.trainY)

    def predict(self):
        self.predictions = self.classifier.predict(self.testX)
        return self.predictions

    def score(self):
        if self.predictions is None:
            print "predict() needs to be called first"
            return 0.0
        scr = np.mean([ f1_score(self.testY[:,tgtidx], self.predictions[:, tgtidx]) for tgtidx in xrange(0, self.num_targets) ])
        print scr
        return scr

    def evaluate(self):
        """ fits the model, predicts the targets and returns evaluation score """
        self.fit()
        self.predict()
        return self.score()

################################################################################################################################
#################################################################################################################################

class NeuralNetworkClassificationBaselineModel(object):
    """ Accepts SingleStockDataSet object as input and
        trains the benchmark model on train set and
        evaluate on test set. """
    def __init__(self, dset, random_state=16):
        self.dset = dset
        self.random_state = random_state
        self.trainX, self.valX, self.testX, self.trainY, self.valY, self.testY = self.dset.get_train_val_test_sets(0.8, 0.1, 0.1)
        self.predictions = None
        self.lr = 0.01
        self.num_targets = self.trainY.shape[1]
        #self.build_regressor()

    def build_nn_arch(self):
        input_dim = self.trainX.shape[1]
        num_classes = 6
        model = Sequential()
        model.add(Dense(10, input_dim=input_dim, init='normal', activation='relu'))
        #model.add(Dropout(0.1))
        #model.add(Dense(5, init='normal', activation='relu'))
        model.add(Dense(num_classes, init='normal', activation='sigmoid'))
        #sgd = SGD(lr=0.00009)
        adam = Adam(lr=self.lr)
        # Compile model
	model.compile(loss='categorical_crossentropy', optimizer=adam)
        return model

    def build_classifier(self):
        # fix random seed for reproducibility
        np.random.seed(self.random_state)
        # evaluate model with standardized dataset
        self.classifiers = [ KerasClassifier(build_fn=self.build_nn_arch, nb_epoch=50, batch_size=5, verbose=0) for idx in xrange(self.num_targets) ]

    def fit(self):
        bestscr = -9999
        bestlr = 0.001
        lst = []
        lrset = [ 0.3, 0.03, 0.003, 0.0003 ]
        for lr in lrset:
            self.lr = lr
            self.build_classifier()
            ypred = np.zeros((self.valX.shape[0], self.num_targets), dtype=int)
            for tgtidx in xrange(self.num_targets):
                self.classifiers[tgtidx].fit(self.trainX, self.trainY[:, tgtidx])
                ypred[:,tgtidx] = self.classifiers[tgtidx].predict(self.valX)
            scr = np.mean([ f1_score(self.valY[:,tgtidx], ypred[:, tgtidx]) for tgtidx in xrange(0, self.num_targets) ])
            if scr > bestscr:
                bestscr = scr
                bestlr = lr
        print 'best lr = %f, best scr = %f' % (bestlr, bestscr)
        self.trainX, self.valX, self.testX, self.trainY, self.valY, self.testY = self.dset.get_train_val_test_sets(0.9, 0, 0.1)
        self.lr = bestlr
        self.build_classifier()
        for tgtidx in xrange(self.num_targets):
            self.classifiers[tgtidx].fit(self.trainX, self.trainY[:, tgtidx])

    def predict(self):
        self.predictions = np.zeros((self.testX.shape[0], self.num_targets), dtype=int)
        for tgtidx in xrange(self.num_targets):
            self.predictions[:,tgtidx] = self.classifiers[tgtidx].predict(self.testX)
        return self.predictions

    def score(self):
        if self.predictions is None:
            print "predict() needs to be called first"
            return 0.0
        return np.mean([ f1_score(self.testY[:,tgtidx], self.predictions[:, tgtidx]) for tgtidx in xrange(0, self.num_targets) ])

    def evaluate(self):
        """ fits the model, predicts the targets and returns evaluation score """
        self.fit()
        self.predict()
        return self.score()

#ticker2nnparms = { 'AAPL' : { 'lr' : 0.01, 'nb_epoch' : 120, 'batch_size' : 5} }

class NeuralNetworkClassificationModelFast(object):
    """ Accepts SingleStockDataSet object as input and
        trains the benchmark model on train set and
        evaluate on test set. """
    def __init__(self, dset, random_state=16):
        self.dset = dset
        self.random_state = random_state
        self.trainX, self.valX, self.testX, self.trainY, self.valY, self.testY = self.dset.get_train_val_test_sets(0.8, 0.1, 0.1)
        self.predictions = None
        self.num_targets = self.trainY.shape[1]
        #self.build_regressor()

    def build_nn_arch(self):
        input_dim = self.trainX.shape[1]
        num_classes = 6
        model = Sequential()
        model.add(Dense(80, input_dim=input_dim, init='normal', activation='relu'))
        #model.add(Dropout(0.1))
        model.add(Dense(40, init='normal', activation='sigmoid'))
        #model.add(Dense(10, init='normal', activation='relu'))
        model.add(Dense(num_classes, init='normal', activation='sigmoid'))
        sgd = SGD(lr=0.4)
        #adam = Adam(lr=0.001)
        # Compile model
	model.compile(loss='binary_crossentropy', optimizer=sgd)
        return model

    def build_classifier(self):
        # fix random seed for reproducibility
        np.random.seed(self.random_state)
        # evaluate model with standardized dataset
        self.classifier = KerasClassifier(build_fn=self.build_nn_arch, nb_epoch=100, batch_size=8, verbose=0)

    def fit(self):
        self.build_classifier()
        self.classifier.fit(self.trainX, self.trainY[:, 1])

    def predict(self):
        self.predictions = self.classifier.predict(self.testX)
        return self.predictions

    def score(self):
        scr = f1_score(self.testY[:, 1], self.predictions)
        print 'f1 score = %f' % scr
        return scr

    def evaluate(self):
        """ fits the model, predicts the targets and returns evaluation score """
        self.fit()
        self.predict()
        return self.score()

    def to_categorical(self, y):
        # convert integers to dummy variables (i.e. one hot encoded)
        return np_utils.to_categorical(y)
