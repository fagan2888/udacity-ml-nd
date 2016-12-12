#!/usr/bin/python

import os
import logging

class SP500(object):
    """ Class to fetch list of stocks tickers in S&P500 that got first added
        to the index after first_add_date and then fetch daily price/volume
        data for each of them """

    def __init__(self, startdate='2006-01-01', enddate='2016-12-10', first_add_date='2006-01-01', datadir='data'):
        self.tickerlist_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        self.startdate      = startdate
        self.enddate        = enddate
        self.first_add_date = first_add_date
        self.tickers        = []
        self.datadir        = datadir
        if not os.path.exists('log'):
            os.makedirs('log')
        logging.basicConfig(filename='log/fetchdata.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    def get_ticker_list(self):
        self.tickers = []
        logging.info('About to get tickers')
        # scrap wikipedia page to get tickers
        logging.info('Finished getting tickers')

    def dump_all_tickers(self):
        logging.info('About to download data for %d tickers', len(self.tickers))
        for ticker in self.tickers:
            self.dump_ticker_data(ticker)
        logging.info('Finished downloading data for all tickers')

    def dump_ticker_data(self, ticker):
        logging.info('About to download data for ticker = %s', ticker)
        # TODO
        logging.info('Finshed downloading data for ticker = %s', ticker)

    def run(self):
        self.get_ticker_list()
        self.dump_all_tickers()

if __name__ == '__main__':
    SP500().run()
