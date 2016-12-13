#!/usr/bin/python

import numpy as np

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
