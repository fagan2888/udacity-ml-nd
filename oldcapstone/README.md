

## Datasets and inputs

The project needs a list of companies(tickers) in S&P 500 index that were first added before
2006-01-01. This information can be programmatically obtained from [https://en.wikipedia.org/wiki/List_of_S%26P_500_companies](here).
Now that the list of tickers are retrieved, for each ticker in the list, the daily adjusted closing prices and volume for
the date range [2006-01-01 to 2016-12-10] can be obtained programmatically in csv format
using the URL template:

```
http://chart.finance.yahoo.com/table.csv?s=<TICKER>&a=0&b=1&c=2006&d=11&e=10&f=2016&g=d&ignore=.csv
```

For each ticker there would be 2755 rows as there are 2755 business days in the selected
date range. As per the problem statement, the features required are SMA, RSD, BBscore,
Mscore and Volume of which Volume can be used directly from the downloaded data.
The rest of the features can be calculated from adjusted closing price present in the
downloaded data files.