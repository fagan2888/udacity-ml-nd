#!/usr/bin/python

import os
import logging
from bs4 import BeautifulSoup
import urllib2
import datetime
import time
import shutil

class SP500(object):
    """ Class to fetch list of stocks tickers in S&P500 that got first added
        to the index before first_add_date and then fetch daily price/volume
        data for each of them """

    def __init__(self, startdate='2006-01-01', enddate='2016-12-10', first_add_date='2006-01-01', datadir='data'):
        self.tickerlist_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        self.startdate      = datetime.datetime.strptime(startdate, "%Y-%m-%d")
        self.enddate        = datetime.datetime.strptime(enddate,   "%Y-%m-%d")
        self.first_add_date = first_add_date
        self.tickers        = []
        self.datadir        = datadir
        if not os.path.exists('log'):
            os.makedirs('log')
        if not os.path.exists(self.datadir):
            os.makedirs(self.datadir)
        logging.basicConfig(filename='log/fetchdata.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    def get_ticker_list(self):
        self.tickers = []
        logging.info('About to get tickers')
        # scrap wikipedia page to get tickers
        page = self.datadir + "/sp500.html"
        if os.path.exists(page):
            logging.info('Found local sp500 html file : %s, using it', page)
            page = open(page, 'r').read()
        else:
            logging.info('Did not find local sp500 html file : %s, so fetching from wikipedia', page)
            localfile = open(page, 'w')
            page = urllib2.urlopen(self.tickerlist_url).read()
            localfile.write(page)
            localfile.close()
        soup = BeautifulSoup(page, 'lxml')
        # there are two tables and we want the first one.
        tbl = soup.find_all('table')[0]
        # take all rows except the header
        rows = tbl.find_all('tr')[1:]
        for row in rows:
            cols = row.find_all('td')
            if cols[6].text <= self.first_add_date:
                self.tickers.append(cols[0].text)
        logging.info('Got %d tickers : %s', len(self.tickers), str(self.tickers))
        logging.info('Finished getting tickers')

    def dump_all_tickers(self):
        logging.info('About to download data for %d tickers', len(self.tickers))
        for ticker in self.tickers:
            self.dump_ticker_data(ticker)
        logging.info('Finished downloading data for all tickers')

    def dump_ticker_data(self, ticker):
        logging.info('About to download data for ticker = %s', ticker)
        #yfin url for ticker
        urldict = {'ticker' : ticker,
                   'a' : self.startdate.month - 1,
                   'b' : self.startdate.day,
                   'c' : self.startdate.year,
                   'd' : self.enddate.month - 1,
                   'e' : self.enddate.day,
                   'f' : self.enddate.year}
        yfinurl = 'http://chart.finance.yahoo.com/table.csv?s=%(ticker)s&a=%(a)d&b=%(b)d&c=%(c)d&d=%(d)d&e=%(e)d&f=%(f)d&g=d&ignore=.csv' % urldict
        datafile = self.datadir + "/" + ticker + ".csv"
        if os.path.exists(datafile):
            logging.info('Found %s, so not downloading for ticker = %s', datafile, ticker)
            return
        logging.info('Fetching data from url : %s', yfinurl)
        shutil.copyfileobj(urllib2.urlopen(yfinurl), open(datafile, 'w'))
        #open(datafile, 'w').write( urllib2.urlopen(yfinurl).read() )
        logging.info('Finshed downloading data for ticker = %s', ticker)
        time.sleep(3)

    def run(self):
        self.get_ticker_list()
        self.dump_all_tickers()

if __name__ == '__main__':
    SP500().run()
