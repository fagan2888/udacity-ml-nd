#!/usr/bin/python

from models import BenchMarkModel
from genfeatures import SingleStockDataSet
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import datetime


class ModelEvaluator(object):
    """ Takes a model and many single stock datasets as input and calculates
        generalization performance of the model with visualization """
    def __init__(self, modelclass, dsets):
        self.modellst = []
        for dset in dsets:
            self.modellst.append(modelclass(dset))

    def evaluate(self):
        self.scores = []
        for mdl in self.modellst:
            print 'Evaluating model for %s...' % mdl.dset.ticker
            self.scores.append(mdl.evaluate())
        self.meanscore = np.mean(self.scores)
        return self.meanscore

    def visualize_predictions(self):
        fig, axs = plt.subplots(len(self.modellst), 1, figsize=(10,10))
        singleplot = False
        if len(self.modellst) == 1:
            singleplot = True
        idx = 0
        for mdl in self.modellst:
            if singleplot:
                ax = axs
            else:
                ax = axs[idx]
            ticker = mdl.dset.ticker
            testY  = mdl.testY['ac_tp1']
            predY  = mdl.predictions[:,0]
            dates  = date2num([datetime.datetime.strptime(dt, '%Y-%m-%d') for dt in testY.index])
            ax.plot_date(dates, testY, fmt='-')
            ax.plot_date(dates, predY, fmt='-')
            ax.legend(['Actual', 'Predicted'])
            ax.set_title('%s stock T+1 adjusted-close price predictions' % ticker)
            idx += 1
        plt.subplots_adjust(hspace=0.6)

def run(tickers):
    badtickers = {'NEE' : True, 'WRK' : True, 'SE' : True, 'WU' : True, 'WYN' : True}
    dsets = []
    for ticker in tickers:
        if ticker in badtickers:
            continue
        dsets.append(SingleStockDataSet(ticker))
    me = ModelEvaluator(BenchMarkModel, dsets)
    meanscore = me.evaluate()
    print '**** Average R^2 score = %f' % meanscore
    me.visualize_predictions()
    plt.show()

if __name__ == '__main__':
    run(['AAPL', 'HIG', 'MRK', 'PLD', 'XOM'])
