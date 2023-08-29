import datetime
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import pytz

pd.set_option('display.float_format', lambda x: '%.5f' % x)

def getNextFriday(today=datetime.date.today()):
    # returns upcoming friday in 'yyyy-mm-dd' format
    friday = today + datetime.timedelta((4-today.weekday()) % 7)
    return friday


def getLastFriday(today=datetime.date.today()):
    # returns last friday in 'yyyy-mm-dd' format
    friday = today - datetime.timedelta((today.weekday() - 4) % 7)
    return friday


def yearsuntil(date, close=True):
    # number of years until a given date, if close then date at 3pm 
    ye, m, d = date.split('-')
    ye, m, d = int(ye), int(m), int(d)
    delta = datetime.datetime(ye, m, d) - datetime.datetime.now()
    years = delta.total_seconds()/(24*3600*365)
    if close:
        years = years + (21/24)/365
    return years


def getStockData(tickers, date):
    # downloads stockprice
    if type(tickers) is not list:
        tickers = [tickers]
    if len(tickers) > 0:
        if len(tickers) == 1:  # dumb hack. yf.download() returns bad data struct if tickers has one element
            tickers.append('J')
    tomorr = str(datetime.datetime.today().date() + datetime.timedelta(days=1))
    sdatanew = yf.download(tickers, start=date, end=tomorr,
                           interval='1d', show_errors=True)
    return sdatanew

def getDataYahooFinance(tickers, edates, verbose=True):
    ''' 
    >takes tickers (list) and exp. dates (list or str ('yyyy-mm-dd') 
    or int for the first int expdates)
    >returns dataframe of options info with cols.: 
    'contractSymbol', 'lastTradeDate', 'strike', 'lastPrice', 'bid', 'ask',
    'volume', 'impliedVolatility', 'corp', 'expdate', 'utc', 'ticker', 'sprice'
    '''

    if type(edates) == list:
        expdates = edates
    elif type(edates) == str:
        expdates = [edates]
    elif type(edates) == int:
        expdates = range(edates)

    data = []
    if datetime.datetime.today().weekday() > 4:
        date = str(getLastFriday())
    else:
        date = str(datetime.datetime.today().date())
    sdatas = getStockData(tickers, date)
    for ticker in tickers:
        tdata = []
        tick = yf.Ticker(ticker)
        sprice = sdatas['Close'][ticker].iloc[0]
        if verbose:
            print('ticker:', ticker)
        optdates = tick.options
        tlen = len(optdates)
        for i, expdate in enumerate(expdates):
            dltimeutc = str(round(datetime.datetime.now().astimezone(pytz.UTC).timestamp()))
            if verbose:
                print('expdate:', i)

            if type(expdates) == list: # expdates should be list of 'yyyy-mm-dd'
                if expdate not in optdates:
                    if verbose:
                        print('downloadoptdata: expdate not available for', ticker)
                    continue
            else: # expdates is range
                if expdate <= tlen - 1:
                    expdate = optdates[expdate]
                else:
                    if verbose:
                        print('%d expdates requested. only %d available for %s' % (
                            edates, tlen, ticker))
                    break

            opt = tick.option_chain(date=expdate)
            optc = opt.calls
            optp = opt.puts
            optc['corp'] = 'c'
            optp['corp'] = 'p'
            try:
                optdata = pd.concat([optc, optp])
            except ValueError:
                if verbose:
                    print('downloadoptdata: valerr1')
                continue

            optdata['expdate'] = expdate
            optdata['utc'] = dltimeutc
            optdata['sprice'] = sprice
            optdata = optdata.drop(columns=['percentChange', 'currency', 'change', 'contractSize', 'inTheMoney'])
            tdata.append(optdata)
        try:
            optdata = pd.concat(tdata)
        except ValueError:
            if verbose:
                print('downloadoptdata: valerr2')
            optdata = pd.DataFrame(tdata)
        optdata['ticker'] = ticker
        data.append(optdata)
    try:
        optdata = pd.concat(data)
    except ValueError:
        if verbose:
            print('downloadoptdata: valerr3')
        optdata = 'bad'
    return optdata


def getCDF(s,k,t,r,sig):
    # d1 and d2 from Black-Scholes formula, ref. Willmot p120
    d1 = (np.log(s/k) + (r + sig**2/2)*t) / (sig*np.sqrt(t))
    d2 = d1 - sig * np.sqrt(t)
    return norm.cdf(d2)


def addCDF(data):
    data['cdf'] = data.apply(lambda x: getCDF(x['sprice'], 
                                        x['strike'],
                                        yearsuntil(x['expdate']),
                                        .05, #interest rate, really shouldn't matter too much at short time intervals
                                        x['impliedVolatility'],), axis=1)
    return(data)


def BuCS(strike1,strike2,price1,price2):
    # bull call spread, strike1<strike2
    cost = -price1 + price2
    maxpay = -strike1 + strike2
    return cost, maxpay


def BePS(strike1,strike2,price1,price2):
    # bear put spread, strike1<strike2
    cost = price1 - price2
    maxpay = -strike1 + strike2
    return cost, maxpay


def addSpreads(data):
    # takes df, adds cost, payoff, odds for bucs and beps for adjacent stock options
    data['contract1'] = data['contractSymbol']
    data['contract2'] = data['contractSymbol'].shift(periods=-1)

    datac = data.loc[(data['corp'] == 'c')]
    strike1 = datac['strike']
    strike2 = datac['strike'].shift(periods=-1)
    price1 = datac['ask']
    price2 = datac['bid'].shift(periods=-1)
    cost, maxpay = BuCS(strike1,strike2,price1,price2)
    datac['cost'] = cost
    datac['maxpay'] = maxpay
    datac['odds'] = datac['cdf'].shift(periods=-1)
    datac['badodds'] = datac['cdf']-datac['cdf'].shift(periods=-1)
    datac['eval'] = datac['maxpay']*datac['odds']+datac['cost']

    datap = data.loc[(data['corp'] == 'p')]
    strike1 = datap['strike']
    strike2 = datap['strike'].shift(periods=-1)
    price1 = datap['bid']
    price2 = datap['ask'].shift(periods=-1)
    cost, maxpay = BePS(strike1,strike2,price1,price2)
    datap['cost'] = cost
    datap['maxpay'] = maxpay
    datap['odds'] = 1-datap['cdf']
    datap['badodds'] = datap['cdf']-datap['cdf'].shift(periods=-1)
    datap['eval'] = datap['maxpay']*datap['odds']+datap['cost']

    data = pd.concat([datac,datap])
    return data


def sortCons(data,odds,n=10,boweight=.2,evalweight=.2,minvol=200,minoi=200):
    # creates score for each pair of options, maximizing closeness to odds and eval, minimizing badodds
    data = data[data['volume']>=minvol]
    data = data[data['volume'].shift(periods=-1)>=minvol]
    data = data[data['openInterest']>=minoi]
    data = data[data['openInterest'].shift(periods=-1)>=minoi]

    data['score'] = ((data['odds'] - odds).abs() + boweight*data['badodds'] - evalweight*data['eval'])
    data = data.sort_values('score')
    data = data.dropna()
    return data.head(n)


def front(tickers, expdate, odds):
    data = getDataYahooFinance(tickers,expdate)
    data = addCDF(data)
    data = addSpreads(data)
    data = sortCons(data,odds)
    return data