#!/usr/bin/python

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from genfeatures import GenStockFeatures, SingleStockDataSet

class StockTimeSeriesExplorer(object):
    """ Reads a single stock's time-series and visualize its statistics """
    def __init__(self, ticker, window=10):
        self.ticker = ticker
        self.window = window
        self.datafile = "../data/" + self.ticker + ".csv"
        self.fileloaded = False
        if not os.path.exists(self.datafile):
            print "Error : Cannot find data file %s" % self.datafile
            return
        self.df = pd.read_csv(self.datafile)
        # order the rows from oldest to newest
        self.df.set_index(self.df['Date'], inplace=True)
        self.df = self.df.reindex(index=self.df.index[::-1])
        self.fileloaded = True
        self.hasfeatures = False
        self.features = []

    def print_basic_stats(self):
        if not self.fileloaded:
            return
        if not self.hasfeatures:
            return
        print "First 5 days data:"
        print self.df.head()
        print "Stats:"
        des = self.df[self.features + ['Adj Close']].describe()
        des.to_csv('../report/' + self.ticker + "_desc.csv", float_format='%.2f')
        print des

    def generate_features(self):
        if ( not self.fileloaded ) or self.hasfeatures:
            return
        self.features.extend(('SMA', 'RSD', 'BB', 'MOM', 'LogVolume'))
        self.df = GenStockFeatures().generate_features(dataframe=self.df, feature_names=self.features, windowsize=self.window)
        self.hasfeatures = True

    def show_histograms(self):
        if not self.fileloaded:
            return
        df1 = self.df.ix[:,self.features + ['Adj Close']]
        axs = df1.hist(color='k', alpha=0.5, bins=100, figsize=(12, 12))
        fig = axs[0][0].get_figure()
        fig.text(0.07, 0.5, 'Number of trading days', va='center', rotation='vertical')
        plt.suptitle('Histograms of all features of %s stock' % self.ticker, fontsize=16)
        plt.savefig('../report/plots/histograms.png')

    def show_timeseries_plots(self):
        if not self.fileloaded:
            return
        dftmp = self.df.ix[0:300,self.features + ['Adj Close']]
        dftmp = normalize(dftmp)
        print dftmp.head()
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10,10))
        #print axes
        ax = dftmp[['Adj Close', 'SMA', 'RSD']].plot(ax = axes[0], title='Adjusted close/SMA/RSD of ' + self.ticker)
        ax.grid(True)
        ax = dftmp[['Adj Close', 'BB', 'MOM']].plot(ax = axes[1], title='Adjusted close/BBscore/MomentumScore of ' + self.ticker)
        ax.grid(True)
        ax = dftmp[['Adj Close', 'LogVolume']].plot(ax = axes[2], title='Adjusted close/Log-Volume of ' + self.ticker)
        ax.grid(True)
        plt.subplots_adjust(hspace=0.6)
        plt.savefig('../report/plots/allplots.png')
        
    def show_feature_correlation_with_target(self):
        dset = SingleStockDataSet('AAPL')
        basefeatures = [ ('ac', 'Adj Close'),
                         ('sma', 'SMA'),
                         ('rsd', 'RSD'),
                         ('bb', 'BBScore'),
                         ('mom', 'MScore'),
                         ('vol', 'LogVolume')]
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(12, 8))
        for tgtshift in xrange(1, 6):
            targetcol = 'ac_tp%d' % (tgtshift)
            heading = 'Adj Close of T+%d' % (tgtshift)
            corrdf = pd.DataFrame(index=[basef[1] for basef in basefeatures],
                                  columns=['T-0', 'T-1', 'T-2', 'T-3', 'T-4'])
            for basef in basefeatures:
                for tshift in xrange(0, 5):
                    featname = '%s_tm%d' % (basef[0], tshift)
                    colname = 'T-%d' % (tshift)
                    corrdf.loc[basef[1], colname] = np.corrcoef(dset.X[[featname]].values.astype(np.float64),
                                                                dset.Y[[targetcol]].values.astype(np.float64), rowvar=0)[0][1]
            print corrdf
            subax = corrdf.plot.barh(ax=axes[tgtshift-1])
            subax.set_title(heading, fontsize=10)
            subax.legend(fontsize='xx-small')
            subax.grid(True)
            if tgtshift > 1:
                subax.set_yticklabels(['' , '', '', '', '', ''])
            labels = subax.get_xticklabels()
            for label in labels:
                label.set_rotation(90)
                label.set_fontsize(10)
            #plt.subplots_adjust(hspace=0.6)
        fig.text(0.5, 0.03, 'Pearson correlation coefficient', ha='center', va='bottom', rotation='horizontal')
        plt.suptitle('Correlation coefficients of targets with features of %s stock' % self.ticker, fontsize=16)
        plt.savefig('../report/plots/corrcoeffs.png')

    def run(self):
        self.generate_features()
        #self.print_basic_stats()
        #self.show_timeseries_plots()
        #self.show_histograms()
        self.show_feature_correlation_with_target()

def normalize(df):
    return (df - df.mean())/df.std()

if __name__ == '__main__':
    StockTimeSeriesExplorer(ticker='AAPL').run()
    plt.show()
        
        
