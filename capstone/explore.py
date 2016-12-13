#!/usr/bin/python

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class PriceTimeSeries(object):
    """ Reads a single stock's time-series and visualize its statistics """
    def __init__(self, ticker, window=5):
        self.ticker = ticker
        self.window = window
        self.datafile = "data/" + self.ticker + ".csv"
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
        self.features = ['Volume']

    def print_basic_stats(self):
        if not self.fileloaded:
            return
        print "First 5 days data:"
        print self.df.head()
        print "Stats:"
        print self.df.describe()

    def generate_features(self):
        if ( not self.fileloaded ) or self.hasfeatures:
            return
        self.df['SMA'] = self.df['Adj Close'].rolling(window=self.window, center=False).mean()
        self.df['RSD'] = self.df['Adj Close'].rolling(window=self.window, center=False).std()
        self.df['BB']  = (self.df['Adj Close'] - self.df['SMA']) / ( 2*self.df['RSD'] )
        self.df['MOM'] = self.df['Adj Close']/self.df['Adj Close'].shift(self.window)
        self.df['Volume'] = np.log(self.df['Volume'])
        # remove the oldest three rows due to NaN in moving stats columns
        #self.df = self.df.drop(['2006-01-03', '2006-01-04', '2006-01-05'])
        self.features.extend(('SMA', 'RSD', 'BB', 'MOM'))
        self.hasfeatures = True

    def show_histograms(self):
        if not self.fileloaded:
            return
        df1 = self.df.ix[:,self.features + ['Adj Close']]
        axs = df1.hist(color='k', alpha=0.5, bins=100, figsize=(12, 12))

    def show_timeseries_plots(self):
        dftmp = self.df.ix[0:300,self.features + ['Adj Close']]
        dftmp = normalize(dftmp)
        print dftmp.head()
        fig, axes = plt.subplots(nrows=3, ncols=1)
        #print axes
        ax = dftmp[['Adj Close', 'SMA', 'RSD']].plot(ax = axes[0], title='Adjusted close/SMA/RSD')
        ax.grid(True)
        ax = dftmp[['Adj Close', 'BB', 'MOM']].plot(ax = axes[1], title='Adjusted close/BBscore/MomentumScore')
        ax.grid(True)
        ax = dftmp[['Adj Close', 'Volume']].plot(ax = axes[2], title='Adjusted close/Log-Volume')
        ax.grid(True)

    def run(self):
        self.print_basic_stats()
        self.generate_features()
        self.show_timeseries_plots()
        self.show_histograms()

def normalize(df):
    return (df - df.mean())/df.std()

if __name__ == '__main__':
    PriceTimeSeries(ticker='GOOG').run()
    plt.show()
        
        
