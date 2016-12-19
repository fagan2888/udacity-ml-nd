#!/usr/bin/python

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import os
import gzpickle

class GenStockFeatures(object):
    """ Create basic features into price-volume dataframe of a stock """
    def __init__(self):
        return

    def generate_features(self, dataframe, feature_names, windowsize=5):
        featset = { feat: True for feat in feature_names }
        if ('BB' in featset):
            featset['SMA'] = True
            featset['RSD'] = True
            if not 'SMA' in featset:
                feature_names.append('SMA')
            elif not 'RSD' in featset:
                feature_names.append('RSD')
        if 'SMA' in featset:
            dataframe['SMA'] = dataframe['Adj Close'].rolling(window=windowsize, center=False).mean()
        if 'RSD' in featset:
            dataframe['RSD'] = dataframe['Adj Close'].rolling(window=windowsize, center=False).std()
        if 'BB' in featset:
            dataframe['BB']  = (dataframe['Adj Close'] - dataframe['SMA']) / ( 2*dataframe['RSD'] )
        if 'MOM' in featset:
            dataframe['MOM'] = dataframe['Adj Close']/dataframe['Adj Close'].shift(windowsize).values
        if 'LogVolume' in featset:
            dataframe['LogVolume'] = np.log(dataframe['Volume'])
        # remove the oldest windowsize rows due to NaN in moving stats columns
        if ('SMA' in featset) or ('RSD' in featset) or ('MOM' in featset):
            dataframe = dataframe.iloc[windowsize:,:]
        return dataframe

class SingleStockDataSet(object):
    """ Loads a single stock's raw price-volume file and generates
        a dataset of the form { (x1, y1), (x2, y2),... } ready for
        application of a regressor
        window is the num days of look back to calculate rolling measures
        m is the num days look back for prediction of p future days"""
    def __init__(self, ticker, window=10, m=5, p=5, normalize=True, compute_pca=False, dropcolnames=True, logprices=False):
        badtickers = {'NEE' : True, 'WRK' : True, 'SE' : True, 'WU' : True, 'WYN' : True}
        if ticker in badtickers:
            print 'Cannot do ticker %s' % ticker
            return
        self.ticker = ticker
        self.window = window
        self.m = m
        self.p = p
        self.datafile = "../data/" + self.ticker + ".csv"
        self.pklfile = "../data/pkl/" + self.ticker + ".pkl.gz"
        self.fileloaded = False
        if not os.path.exists(self.datafile):
            print "Error : Cannot find data file %s" % self.datafile
            return
        self.formed_dataset = False
        if not os.path.exists(self.pklfile):
            self.__genXY__()
        else:
            print 'Reading from pklfile = %s' % self.pklfile
            self.X, self.Y = gzpickle.load( self.pklfile )
            self.formed_dataset = True
        if logprices:
            pricescols = ['ac_tm%d' % idx for idx in xrange(0, self.m) ]
            self.X[pricescols] = np.log(self.X[pricescols].astype('float64'))
        if normalize:
            self.X = ( (self.X - self.X.mean()) / self.X.std() )
        if dropcolnames:
            self.X = self.X.values
            self.Y = self.Y.values
        if compute_pca:
            self.pca = PCA().fit(self.X)
            self.X = self.pca.transform(self.X)            

    def __genXY__(self):
        self.df = pd.read_csv(self.datafile)
        # order the rows from oldest to newest
        self.df.set_index(self.df['Date'], inplace=True)
        self.df = self.df.reindex(index=self.df.index[::-1])

        self.features = ['Adj Close', 'SMA', 'RSD', 'BB', 'MOM', 'LogVolume']
        column_bases = ['ac', 'sma', 'rsd', 'bb', 'mom', 'vol']
        # Generate col names for features : ex: ac_tm3 means Adjusted close at T-3
        self.column_names = [ base + "_tm" + str(suff) for suff in xrange(0, self.m) for base in column_bases ]
        self.df = GenStockFeatures().generate_features(self.df, self.features, windowsize=self.window)
        self.hasfeatures = True
        self.S = self.df.shape[0]
        if self.S - self.m + 1 - self.p < 2000:
            print 'Size of the raw dataset too low, not generating dataset'
            return
        self.X = pd.DataFrame(index=self.df.index[self.m-1:self.S-self.p], columns=self.column_names)
        # To store targets : Adjusted close at T+1, Adjusted close at T+2, ... Adjusted close at T+5
        self.Y = pd.DataFrame(index=self.df.index[self.m-1:self.S-self.p], columns=('ac_tp1', 'ac_tp2', 'ac_tp3', 'ac_tp4', 'ac_tp5'))
        for idx in xrange(self.m-1, self.S-self.p):
            newidx = idx - self.m + 1
            row = [ self.df.loc[self.df.index[subidx], feat] for subidx in xrange(idx,idx-self.m, -1) for feat in self.features ]
            self.X.iloc[newidx] = row
            #self.Y.iloc[newidx] = self.df['Adj Close'][idx+1:idx+self.p+1].values
            curr_ac = self.X.iloc[newidx]['ac_tm0']
            self.Y.iloc[newidx] = self.convert_to_classes(self.df['Adj Close'][idx+1:idx+self.p+1].values, curr_ac)
            # For target = (P_tpx / P_tm0) -1
            #curr_ac = self.X.iloc[newidx]['ac_tm0']
            #self.Y.iloc[newidx] = ( self.df['Adj Close'][idx+1:idx+self.p+1].values / curr_ac ) - 1
            #self.Y.iloc[newidx] = [ (self.df.loc[self.df.index[subidx], 'Adj Close']/curr_ac) - 1 for subidx in xrange(idx+1, idx+self.p+1) ]
        self.formed_dataset = True
        # release df
        self.df = None
        if not os.path.exists('../data/pkl'):
            os.makedirs('../data/pkl')
        gzpickle.save([self.X, self.Y], self.pklfile)
        print 'Wrote X and Y to %s' % self.pklfile

    def get_ac_ulimits(self):
        ac = self.df['Adj Close'].values
        acset = set(ac)
        ac = sorted(list(acset))
        numclasses = 10
        binsize = len(ac)/(numclasses-1)
        ulimits = []
        for clsidx in xrange(0, numclasses-1):
            endidx = (clsidx + 1)*binsize - 1
            ulimits.append(ac[endidx])
        return ulimits

    def convert_to_classes(self, ylist, curr_ac):
        """ for positive returns : class 0
            for negative returns : class 1
        """
        outlist = []
        for y in ylist:
            returnpct = ((y/float(curr_ac))-1.0)*100
            classidx = 0
            if returnpct > 0:
                classidx = 1
            outlist.append(classidx)
        return outlist

    def get_train_val_test_sets(self, train_pct=0.8, val_pct=0.1, test_pct=0.1):
        """ returns (Xtrain, Xval, Xtest, Ytrain, Yval, Ytest)"""
        if not self.formed_dataset:
            print "Error : The dataset was not formed !! Cannot continue !!"
            return
        if train_pct < 0.0 or val_pct < 0.0 or test_pct < 0.0:
            print "Error : parameters must be positive"
            return
        if train_pct + val_pct + test_pct > 1.0:
            print "Error : train_pct + val_pct + test_pct > 1.0"
            return
        if train_pct == 0.0 or test_pct == 0.0:
            print "Error : None of train_pct or test_pct can be zero"
            return
        sz = self.X.shape[0]
        train_end = int(round(train_pct*sz) - 1)
        val_end = train_end+1
        if val_pct != 0.0:
            val_end = int(train_end + 1 + round(val_pct*sz) - 1)
        if val_end == train_end+1:
            return self.X[0:train_end+1,:], None, self.X[train_end+1:,:], self.Y[0:train_end+1,:], None, self.Y[train_end+1:,:]
        return self.X[0:train_end+1, :], self.X[train_end+1:val_end+1, :], self.X[val_end+1:, :], \
            self.Y[0:train_end+1, :], self.Y[train_end+1:val_end+1, :], self.Y[val_end+1:, :]

