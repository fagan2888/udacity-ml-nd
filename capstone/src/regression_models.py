#!/usr/bin/python

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.core import Dropout
from keras.wrappers.scikit_learn import KerasRegressor
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
# from sklearn.pipeline import Pipeline


#### NON - ENSEMBLE LEARNERS #####################################################################
class BenchMarkModel(object):
    """ Accepts SingleStockDataSet object as input and
        trains the benchmark model on train set and
        evaluate on test set. """
    def __init__(self, dset):
        self.dset = dset
        self.trainX, _, self.testX, self.trainY, _, self.testY = self.dset.get_train_val_test_sets(0.9, 0.0, 0.1)
        self.predictions = None
        self.regressor = LinearRegression(normalize=True)

    def fit(self):
        self.regressor.fit(self.trainX, self.trainY)
        #print 'BenchMarkModel : train score = %f' % np.mean(self.regressor.score(self.trainX, self.trainY))

    def predict(self):
        self.predictions = self.regressor.predict(self.testX)
        return self.predictions

    def score(self):
        if self.predictions is None:
            print "predict() needs to be called first"
            return 0.0
        return r2_score(self.testY, self.predictions, multioutput='uniform_average')

    def evaluate(self):
        """ fits the model, predicts the targets and returns evaluation score """
        self.fit()
        self.predict()
        return self.score()

class TimeWeightedLinearRegression(object):
    """ Accepts SingleStockDataSet object as input and
        trains the benchmark model on train set and
        evaluate on test set. """
    def __init__(self, dset):
        self.dset = dset
        self.trainX, _, self.testX, self.trainY, _, self.testY = self.dset.get_train_val_test_sets(0.9, 0.0, 0.1)
        self.predictions = None
        self.regressor = LinearRegression(normalize=True)

    def fit(self):
        m = self.trainX.shape[0]
        weights = np.zeros(m)
        numslots = 50
        slotsize = m/numslots
        wt = 100.0
        for slotidx in xrange(numslots-1, -1, -1):
            endidx = m + ((slotidx - numslots + 1)*slotsize)
            startidx = endidx - slotsize
            if slotidx == 0:
                startidx = 0
            weights[startidx:endidx] = wt
            wt = wt*0.7
        self.regressor.fit(self.trainX, self.trainY, weights)
        #print 'BenchMarkModel : train score = %f' % np.mean(self.regressor.score(self.trainX, self.trainY))

    def predict(self):
        self.predictions = self.regressor.predict(self.testX)
        return self.predictions

    def score(self):
        if self.predictions is None:
            print "predict() needs to be called first"
            return 0.0
        return r2_score(self.testY, self.predictions, multioutput='uniform_average')

    def evaluate(self):
        """ fits the model, predicts the targets and returns evaluation score """
        self.fit()
        self.predict()
        return self.score()
    
class RidgeRegressionModel(object):
    """ Accepts SingleStockDataSet object as input and
        trains the benchmark model on train set and
        evaluate on test set. """
    def __init__(self, dset):
        self.dset = dset
        self.trainX, self.valX, self.testX, self.trainY, self.valY, self.testY = self.dset.get_train_val_test_sets(0.8, 0.1, 0.1)
        self.predictions = None

    def fit(self):
        alphas = [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
        #alphas = [1.0]
        bestscore = -9999
        bestalpha = 1.0
        lst = []
        for alpha in alphas:
            tmpmodel = Ridge(alpha=alpha)
            tmpmodel.fit(self.trainX, self.trainY)
            scr = tmpmodel.score(self.valX, self.valY)
            #scrtrain = tmpmodel.score(self.trainX, self.trainY)
            #print 'alpha = %f : train score = %f, validation score = %f' % (alpha, scrtrain, scr)
            lst.append((alpha, scr))
            if scr > bestscore:
                bestscore = scr
                bestalpha = alpha
        print lst
        print 'Best alpha = %f' % bestalpha
        self.trainX, self.valX, self.testX, self.trainY, self.valY, self.testY = self.dset.get_train_val_test_sets(0.9, 0, 0.1)
        self.regressor = Ridge(alpha=bestalpha).fit(self.trainX, self.trainY)

    def predict(self):
        self.predictions = self.regressor.predict(self.testX)
        return self.predictions

    def score(self):
        if self.predictions is None:
            print "predict() needs to be called first"
            return 0.0
        return r2_score(self.testY, self.predictions, multioutput='uniform_average')

    def evaluate(self):
        """ fits the model, predicts the targets and returns evaluation score """
        self.fit()
        self.predict()
        return self.score()

class MultiTargetSVR(object):
    def __init__(self, C, gamma='auto'):
        self.C = C
        self.gamma = gamma

    def fit(self, X, y):
        self.num_targets = y.shape[1]
        self.models = [ SVR(C=self.C, gamma=self.gamma).fit(X, y[:,tgtidx]) for tgtidx in xrange(0, self.num_targets) ]
        return self

    def predict(self, X):
        # Allocate space for predicted target dataframe
        ypred = np.zeros((X.shape[0], self.num_targets))
        for tgtidx in xrange(0, self.num_targets):
            ypred[:, tgtidx] = self.models[tgtidx].predict(X)
        return ypred

    def score(self, X, y):
        return r2_score(y, self.predict(X), multioutput='uniform_average')    

class SVRModel(object):
    """ Accepts SingleStockDataSet object as input and
        trains the benchmark model on train set and
        evaluate on test set. """
    def __init__(self, dset):
        self.dset = dset
        self.trainX, self.valX, self.testX, self.trainY, self.valY, self.testY = self.dset.get_train_val_test_sets(0.8, 0.1, 0.1)
        self.predictions = None
        self.gamma = 0.1

    def fit(self):
        bs = np.arange(5., 11., 1.)
        Cset = 2.0**bs
        bestscore = -9999
        bestC = 1.0
        lst = []
        for C in Cset:
            tmpmodel = MultiTargetSVR(C=C)
            tmpmodel.fit(self.trainX, self.trainY)
            scr = tmpmodel.score(self.valX, self.valY)
            #scrtrain = tmpmodel.score(self.trainX, self.trainY)
            #print 'alpha = %f : train score = %f, validation score = %f' % (alpha, scrtrain, scr)
            lst.append((C, scr))
            if scr > bestscore:
                bestscore = scr
                bestC = C
        print lst
        print 'Best C = %f' % bestC
        self.trainX, self.valX, self.testX, self.trainY, self.valY, self.testY = self.dset.get_train_val_test_sets(0.9, 0, 0.1)
        self.regressor = MultiTargetSVR(C=bestC).fit(self.trainX, self.trainY)

    def predict(self):
        self.predictions = self.regressor.predict(self.testX)
        return self.predictions

    def score(self):
        if self.predictions is None:
            print "predict() needs to be called first"
            return 0.0
        return r2_score(self.testY, self.predictions, multioutput='uniform_average')

    def evaluate(self):
        """ fits the model, predicts the targets and returns evaluation score """
        self.fit()
        self.predict()
        return self.score()

class KNNRegressionModel(object):
    """ Accepts SingleStockDataSet object as input and
        trains the benchmark model on train set and
        evaluate on test set. """
    def __init__(self, dset):
        self.dset = dset
        self.trainX, self.valX, self.testX, self.trainY, self.valY, self.testY = self.dset.get_train_val_test_sets(0.8, 0.1, 0.1)
        self.predictions = None

    def fit(self):
        kset = list(xrange(2, 20))
        #alphas = [1.0]
        lst = []
        bestscore = -9999
        bestk = 1.0
        for k in kset:
            tmpmodel = KNeighborsRegressor(n_neighbors=k)
            tmpmodel.fit(self.trainX, self.trainY)
            scr = tmpmodel.score(self.valX, self.valY)
            #scrtrain = tmpmodel.score(self.trainX, self.trainY)
            lst.append((k, scr))
            if scr > bestscore:
                bestscore = scr
                bestk = k
        print lst
        print 'Best k = %f' % bestk
        self.trainX, self.valX, self.testX, self.trainY, self.valY, self.testY = self.dset.get_train_val_test_sets(0.9, 0, 0.1)
        self.regressor = KNeighborsRegressor(n_neighbors=bestk).fit(self.trainX, self.trainY)

    def predict(self):
        self.predictions = self.regressor.predict(self.testX)
        return self.predictions

    def score(self):
        if self.predictions is None:
            print "predict() needs to be called first"
            return 0.0
        return r2_score(self.testY, self.predictions, multioutput='uniform_average')

    def evaluate(self):
        """ fits the model, predicts the targets and returns evaluation score """
        self.fit()
        self.predict()
        return self.score()
    
####################################################################################################################

class MultiTargetEnsembleRegressor(object):
    """ Ensemble Regressor of sklearn does not support multiple targets,
        so this class wraps AdaBoostRegressor to support multiple targets"""
    def __init__(self, ensemble_learner_class, base_estimator=None, random_state=16, **kvargs):
        self.base_estimator = base_estimator
        self.random_state = random_state
        self.ensemble_learner_class = ensemble_learner_class
        self.kvargs = kvargs

    def fit(self, X, y):
        self.num_targets = y.shape[1]
        self.models = [ self.ensemble_learner_class(base_estimator=self.base_estimator, random_state=self.random_state,
                                              **self.kvargs).fit(X, y[:,tgtidx]) for tgtidx in xrange(0, self.num_targets) ]
    def predict(self, X):
        # Allocate space for predicted target dataframe
        y = np.zeros((X.shape[0], self.num_targets))
        for tgtidx in xrange(0, self.num_targets):
            y[:, tgtidx] = self.models[tgtidx].predict(X)
        return y

    def score(self, X, y):
        return r2_score(y, self.predict(X), multioutput='uniform_average')

class TunableMultiTargetEnsembleRegressor(object):
    """ Accepts SingleStockDataSet object and base regression class as input and
        trains the AdaBoost regressor with the base model
        on train set, tune paramters in validation set and evaluate on test set. """
    def __init__(self, dset, ensemble_learner_class, base_estimator_class, random_state=16):
        self.dset = dset
        self.random_state = random_state
        self.base_estimator_class = base_estimator_class
        self.ensemble_learner_class = ensemble_learner_class
        self.trainX, self.valX, self.testX, self.trainY, self.valY, self.testY = self.dset.get_train_val_test_sets(0.8, 0.1, 0.1)
        self.predictions = None
        self.regressor = None

    def fit_tune(self, baseparms, adaboostparms):
        """ Accepts set of values for a single base regression model parameter and
            set of values for a single AdaBoost parameter """
        self.bestbaseparm = None
        self.bestabparm  = None
        self.bestvalidscore = -9999
        abparmname = adaboostparms[0].keys()[0]
        sortedabparm = sorted([abparm[abparmname] for abparm in adaboostparms])
        bparmname = self.base_estimator_class.__name__
        sortedbparm = [bparmname]
        if len(baseparms) > 0:
            bparmname = baseparms[0].keys()[0]
            sortedbparm = sorted([bparm[bparmname] for bparm in baseparms])
        self.validscores = pd.DataFrame(index=sortedabparm, columns=sortedbparm)
        self.trainscores = pd.DataFrame(index=sortedabparm, columns=sortedbparm)
        self.validscores.index.name = abparmname
        self.trainscores.index.name = abparmname
        #print self.validscores
        if len(baseparms) > 0:
            for parmbase in baseparms:
                for parmab in adaboostparms:
                    tmpmodel = MultiTargetEnsembleRegressor(self.ensemble_learner_class, self.base_estimator_class(**parmbase), random_state=self.random_state, **parmab)
                    tmpmodel.fit(self.trainX, self.trainY)
                    trainscore = tmpmodel.score(self.trainX, self.trainY)
                    validscore = tmpmodel.score(self.valX, self.valY)
                    self.trainscores.loc[parmab[abparmname], parmbase[bparmname]] = trainscore
                    self.validscores.loc[parmab[abparmname], parmbase[bparmname]] = validscore
                    if validscore > self.bestvalidscore:
                        self.bestbaseparm = parmbase
                        self.bestabparm = parmab
                        self.bestvalidscore = validscore
        else:
            for parmab in adaboostparms:
                #print parmab[abparmname], bparmname
                tmpmodel = MultiTargetEnsembleRegressor(self.ensemble_learner_class, self.base_estimator_class(), random_state=self.random_state, **parmab)
                tmpmodel.fit(self.trainX, self.trainY)
                trainscore = tmpmodel.score(self.trainX, self.trainY)
                validscore = tmpmodel.score(self.valX, self.valY)
                #print trainscore, validscore
                self.trainscores.loc[parmab[abparmname], bparmname] = trainscore
                self.validscores.loc[parmab[abparmname], bparmname] = validscore
                if validscore > self.bestvalidscore:
                    self.bestabparm = parmab
                    self.bestvalidscore = validscore
        #print 'Best valid score = %f' % self.bestvalidscore
        #print self.trainscores
        #print self.validscores
        # The following makes self.validscores.plot() and self.trainscores.plot() show correct labels
        if len(baseparms) > 0:
            self.validscores.columns = ['%s=%f' % (bparmname, val) for val in self.validscores.columns ]
            self.trainscores.columns = ['%s=%f' % (bparmname, val) for val in self.trainscores.columns ]
        # Combine train set with validation set and fit the model with best parameters.
        self.trainX, self.valX, self.testX, self.trainY, self.valY, self.testY = self.dset.get_train_val_test_sets(0.9, 0.0, 0.1)
        if len(baseparms) > 0:
            self.regressor = MultiTargetEnsembleRegressor(self.ensemble_learner_class, self.base_estimator_class(**self.bestbaseparm), random_state=self.random_state, **self.bestabparm)
        else:
            self.regressor = MultiTargetEnsembleRegressor(self.ensemble_learner_class, self.base_estimator_class(), random_state=self.random_state, **self.bestabparm)
        self.regressor.fit(self.trainX, self.trainY)

    def predict(self):
        if self.regressor is None:
            print "TunableMultiTargetEnsembleRegressor : fit_tune() needs to be called first"
            return None
        self.predictions = self.regressor.predict(self.testX)
        return self.predictions

    def score(self):
        if self.predictions is None:
            print "TunableMultiTargetEnsembleRegressor : predict() needs to be called first"
            return 0.0
        return r2_score(self.testY, self.predictions, multioutput='uniform_average')

    def evaluate(self, baseparms, adaboostparms):
        """ fits, tunes the model, predicts the targets and returns evaluation score """
        self.fit_tune(baseparms, adaboostparms)
        self.predict()
        return self.score()

class BoostedLinearRegression(object):
    """ Accepts SingleStockDataSet object and parmeter settings and uses TunableMultiTargetAdaBoostRegressor.
        Uses Linear Regression as base algorithm
        Evaluates in test dataset """
    def __init__(self, dset, random_state=16):
        self.dset = dset
        self.predictions = None
        self.regressor = TunableMultiTargetEnsembleRegressor(dset, AdaBoostRegressor, LinearRegression, random_state)

    def fit(self):
        bparms = []
        abparms = [ {'n_estimators' : n_estimators} for n_estimators in xrange(1, 15) ]
        self.regressor.fit_tune(bparms, abparms)
        self.testY = self.regressor.testY
        #print 'BoostedLinearRegression : best params : %s' % str((self.regressor.bestbaseparm, self.regressor.bestabparm))

    def predict(self):
        self.regressor.predict()
        self.predictions = self.regressor.predictions
        self.trainscores = self.regressor.trainscores
        self.validscores = self.regressor.validscores
        return self.predictions

    def score(self):
        if self.predictions is None:
            print "BoostedLinearRegression : predict() needs to be called first"
            return 0.0
        return self.regressor.score()

    def evaluate(self):
        """ fits the model, predicts the targets and returns evaluation score """
        self.fit()
        self.predict()
        return self.score()

class BoostedRidgeRegression(object):
    """ Accepts SingleStockDataSet object and parmeter settings and uses TunableMultiTargetAdaBoostRegressor.
        Uses Ridge Regression as base algorithm
        Evaluates in test dataset """
    def __init__(self, dset, random_state=16):
        self.dset = dset
        self.predictions = None
        self.regressor = TunableMultiTargetAdaBoostRegressor(dset, AdaBoostRegressor, Ridge, random_state)

    def fit(self):
        bparms = [ {'alpha' : alpha} for alpha in np.arange(0.1, 1.1, 0.1) ]
        abparms = [ {'n_estimators' : n_estimators} for n_estimators in xrange(1, 15) ]
        #abparms = [ {'n_estimators' : n_estimators} for n_estimators in xrange(4, 10) ]
        self.regressor.fit_tune(bparms, abparms)
        #print 'BoostedRidgeRegression : best params : %s' % str((self.regressor.bestbaseparm, self.regressor.bestabparm))
        self.testY = self.regressor.testY

    def predict(self):
        self.regressor.predict()
        self.predictions = self.regressor.predictions
        self.trainscores = self.regressor.trainscores
        self.validscores = self.regressor.validscores
        return self.predictions

    def score(self):
        if self.predictions is None:
            print "BoostedLinearRegression : predict() needs to be called first"
            return 0.0
        return self.regressor.score()

    def evaluate(self):
        """ fits the model, predicts the targets and returns evaluation score """
        self.fit()
        self.predict()
        return self.score()

class BaggedLinearRegressionModel(object):
    """ Accepts SingleStockDataSet object as input and
        trains the benchmark model on train set and
        evaluate on test set. """
    def __init__(self, dset):
        self.dset = dset
        self.trainX, _, self.testX, self.trainY, _, self.testY = self.dset.get_train_val_test_sets(0.9, 0.0, 0.1)
        self.predictions = None
        self.regressor = MultiTargetEnsembleRegressor(BaggingRegressor, LinearRegression(), n_estimators=40, max_samples=0.65, bootstrap=True)

    def fit(self):
        self.regressor.fit(self.trainX, self.trainY)
        #print 'BenchMarkModel : train score = %f' % np.mean(self.regressor.score(self.trainX, self.trainY))

    def predict(self):
        self.predictions = self.regressor.predict(self.testX)
        return self.predictions

    def score(self):
        if self.predictions is None:
            print "predict() needs to be called first"
            return 0.0
        return r2_score(self.testY, self.predictions, multioutput='uniform_average')

    def evaluate(self):
        """ fits the model, predicts the targets and returns evaluation score """
        self.fit()
        self.predict()
        return self.score()

class NeuralNetworkRegressionBaselineModel(object):
    """ Accepts SingleStockDataSet object as input and
        trains the benchmark model on train set and
        evaluate on test set. """
    def __init__(self, dset, random_state=16):
        self.dset = dset
        self.random_state = random_state
        self.trainX, self.valX, self.testX, self.trainY, self.valY, self.testY = self.dset.get_train_val_test_sets(0.8, 0.1, 0.1)
        self.predictions = None
        self.lr = 0.01
        #self.build_regressor()

    def build_nn_arch(self):
        input_dim = self.trainX.shape[1]
        num_targets = self.trainY.shape[1]
        model = Sequential()
        model.add(Dense(50, input_dim=input_dim, init='normal', activation='relu'))
        #model.add(Dropout(0.1))
        model.add(Dense(5, init='normal', activation='relu'))
        model.add(Dense(num_targets, init='normal'))
        #sgd = SGD(lr=0.00009)
        adam = Adam(lr=self.lr)
        # Compile model
	model.compile(loss='mean_squared_error', optimizer=adam)
        return model

    def build_regressor(self):
        # fix random seed for reproducibility
        np.random.seed(self.random_state)
        # evaluate model with standardized dataset
        self.regressor = KerasRegressor(build_fn=self.build_nn_arch, nb_epoch=50, batch_size=5, verbose=0)

    def fit(self):
        bestscr = -9999
        bestlr = 0.001
        lst = []
        lrset = [ 0.3, 0.03, 0.003, 0.0003 ]
        for lr in lrset:
            self.lr = lr
            self.build_regressor()
            self.regressor.fit(self.trainX, self.trainY)
            y_pred = self.regressor.predict(self.valX)
            scr = r2_score(self.valY, y_pred, multioutput='uniform_average')
            lst.append((lr, scr))
            if scr > bestscr:
                bestscr = scr
                bestlr = lr
        print lst
        print 'best lr = %f' % bestlr
        self.trainX, self.valX, self.testX, self.trainY, self.valY, self.testY = self.dset.get_train_val_test_sets(0.9, 0, 0.1)
        self.lr = bestlr
        self.build_regressor()
        self.regressor.fit(self.trainX, self.trainY)
        #print 'NeuralNetworkRegressionBaselineModel : train score = %f' % np.mean(self.regressor.score(self.trainX, self.trainY))

    def predict(self):
        self.predictions = self.regressor.predict(self.testX)
        return self.predictions

    def score(self):
        if self.predictions is None:
            print "predict() needs to be called first"
            return 0.0
        return r2_score(self.testY, self.predictions, multioutput='uniform_average')

    def evaluate(self):
        """ fits the model, predicts the targets and returns evaluation score """
        self.fit()
        self.predict()
        return self.score()

class RandomForestModel(object):
    """ Accepts SingleStockDataSet object as input and
        trains the benchmark model on train set and
        evaluate on test set. """
    def __init__(self, dset):
        self.dset = dset
        self.trainX, _, self.testX, self.trainY, _, self.testY = self.dset.get_train_val_test_sets(0.9, 0.0, 0.1)
        self.predictions = None
        self.regressor = RandomForestRegressor(n_estimators=10, max_depth=30, min_samples_split=20, random_state=16)

    def fit(self):
        self.regressor.fit(self.trainX, self.trainY)
        #print 'BenchMarkModel : train score = %f' % np.mean(self.regressor.score(self.trainX, self.trainY))

    def predict(self):
        self.predictions = self.regressor.predict(self.testX)
        return self.predictions

    def score(self):
        if self.predictions is None:
            print "predict() needs to be called first"
            return 0.0
        return r2_score(self.testY, self.predictions, multioutput='uniform_average')

    def evaluate(self):
        """ fits the model, predicts the targets and returns evaluation score """
        self.fit()
        self.predict()
        return self.score()
