#!/usr/bin/python

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

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
