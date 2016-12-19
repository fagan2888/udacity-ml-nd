#!/usr/bin/python

from models import BenchMarkModel
from models import SVCModel
from models import KNNClassificationModel
from models import DTClassificationModel
from models import RFClassifierModel
from models import NeuralNetworkClassificationBaselineModel
from models import NeuralNetworkClassificationModelFast

from genfeatures import SingleStockDataSet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import datetime
from matplotlib.ticker import FormatStrFormatter

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
            print 'Evaluating model %s for %s...' % (type(mdl).__name__, mdl.dset.ticker)
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
            testY  = mdl.testY[:,0]
            predY  = mdl.predictions[:,0]
            dates  = date2num([datetime.datetime.strptime(dt, '%Y-%m-%d') for dt in testY.index])
            ax.plot_date(dates, testY, fmt='-')
            ax.plot_date(dates, predY, fmt='-')
            ax.legend(['Actual', 'Predicted'])
            ax.set_title('%s stock T+1 day return predictions' % ticker)
            idx += 1
        plt.suptitle(type(self.modellst[0]).__name__, fontsize=16)
        plt.subplots_adjust(hspace=0.6)

    def show_model_complexity_curve(self):
        if not hasattr(self.modellst[0], 'trainscores'):
            return
        fig, axs = plt.subplots(len(self.modellst)*2, 1, figsize=(10,10))
        singleplot = False
        if len(self.modellst) == 1:
            singleplot = True
        idx = 0
        for mdl in self.modellst:
            ax1 = axs[idx]
            ax2 = axs[idx+1]
            ticker = mdl.dset.ticker
            mdl.trainscores.plot.line(ax=ax1)
            ax1.set_title('%s - Training' % ticker)
            ax1.set_ylabel('R^2 score')
            ax1.yaxis.set_major_formatter(FormatStrFormatter('%.6f'))
            mdl.validscores.plot.line(ax=ax2)
            ax2.set_title('%s - Validation' % ticker)
            ax2.set_ylabel('R^2 score')
            ax2.yaxis.set_major_formatter(FormatStrFormatter('%.6f'))
            idx += 2
        plt.suptitle(type(self.modellst[0]).__name__, fontsize=16)
        plt.subplots_adjust(hspace=0.6)

def run(tickers, finddimensionality=False):
    badtickers = {'NEE' : True, 'WRK' : True, 'SE' : True, 'WU' : True, 'WYN' : True}
    dsets = []
    Xs    = []
    for ticker in tickers:
        if ticker in badtickers:
            continue
        dset = SingleStockDataSet(ticker, compute_pca=True, logprices=False)
        #print 'ticker = %s : explained var ratios :' % ticker
        #print dset.pca.explained_variance_ratio_
        Xs.append(dset.X)
        #dset.X = dset.X[:,0:2]
        dsets.append(dset)
    if finddimensionality:
        modelclasses = [BenchMarkModel, RidgeRegressionModel, KNNRegressionModel]
        dimslist = list(xrange(1, 26))
        perfdim = pd.DataFrame(index=dimslist, columns=[cls.__name__ for cls in modelclasses])
        perfdim.index.name = 'Number of input dimensions'
        for dims in dimslist:
            for idx in xrange(0, len(dsets)):
                dsets[idx].X = Xs[idx][:,0:dims]
            for mc in modelclasses:
                me = ModelEvaluator(mc, dsets)
                meanscore = me.evaluate()
                perfdim.loc[dims, mc.__name__] = meanscore
        perfdim.plot.line()
        plt.suptitle('Test-set Performances of tuned models with input dimensionality', fontsize=14)
    modelclasses = [BenchMarkModel, DTClassificationModel]
    for idx in xrange(0, len(dsets)):
        dsets[idx].X = Xs[idx][:,0:5]
    for mc in modelclasses:
        me = ModelEvaluator(mc, dsets)
        meanscore = me.evaluate()
        print '**** Average F1 score for %s = %f' % (mc.__name__, meanscore)
        #me.visualize_predictions()
        #me.show_model_complexity_curve()
    plt.show()

if __name__ == '__main__':
    #run(['AAPL', 'HIG', 'MRK', 'PLD', 'XOM'])
    run(['AAPL'])
