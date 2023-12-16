import requests
from datetime import datetime
import pandas as pd
from typing import List, Dict
import yfinance as yf
import pandas as pd
import numpy as np
import os
import json
from sklearn.cluster import KMeans
from scipy import optimize 

class PriceHistory():

    def __init__(self, symbols: List[str], api_key: str) -> None:
        self._api_url = 'https://www.alphavantage.co/query'
        self._symbols = symbols
        self._api_key = api_key
        self.price_data_frame = self._build_data_frames()

    def _build_data_frames(self) -> pd.DataFrame:
        all_data = []

        for symbol in self._symbols:
            all_data += self._grab_prices(symbol)
        price_data_frame = pd.DataFrame(data=all_data)
        price_data_frame['date'] = pd.to_datetime(price_data_frame['date'])
        return price_data_frame

    def _grab_prices(self, symbol: str) -> List[Dict]:
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'apikey': self._api_key,
        }

        response = requests.get(self._api_url, params=params)

        if response.status_code == 200:
            data = response.json()
            if 'Time Series (Daily)' in data:
                time_series = data['Time Series (Daily)']
                historical_data = []
                for date_str, values in time_series.items():
                    date = datetime.strptime(date_str, '%Y-%m-%d')
                    row = {
                        'date': date,
                        'symbol': symbol,
                        'open': float(values['1. open']),
                        'high': float(values['2. high']),
                        'low': float(values['3. low']),
                        'close': float(values['4. close']),
                        'volume': int(values['5. volume']),
                    }
                    historical_data.append(row)
                return historical_data

        return []

def StockReturnsComputing(StockPrice):
    import numpy as np

    # Convertir le DataFrame Pandas en un array NumPy
    StockPriceArray = StockPrice.values

    # Récupérer le nombre de lignes et de colonnes
    Rows, Columns = StockPriceArray.shape

    # Initialiser le tableau de rendements
    StockReturn = np.zeros([Rows-1, Columns])

    # Calculer les rendements
    for j in range(Columns):        # j: Assets
        for i in range(Rows-1):     # i: Daily Prices
            StockReturn[i, j] = ((StockPriceArray[i+1, j] - StockPriceArray[i, j]) / StockPriceArray[i, j]) * 100

    return StockReturn

#function obtains maximal return portfolio using linear programming

def MaximizeReturns(MeanReturns, PortfolioSize):
    
    #dependencies
    from scipy.optimize import linprog
    
    c = (np.multiply(-1, MeanReturns))
    A = np.ones([PortfolioSize,1]).T
    b=[1]
    res = linprog(c, A_ub = A, b_ub = b, bounds = (0,1), method = 'simplex') 
    
    return res
#function obtains minimal risk portfolio 

def MinimizeRisk(CovarReturns, PortfolioSize):
    
    def  f(x, CovarReturns):
        func = np.matmul(np.matmul(x, CovarReturns), x.T) 
        return func

    def constraintEq(x):
        A=np.ones(x.shape)
        b=1
        constraintVal = np.matmul(A,x.T)-b 
        return constraintVal
    
    xinit=np.repeat(0.1, PortfolioSize)
    cons = ({'type': 'eq', 'fun':constraintEq})
    lb = 0
    ub = 1
    bnds = tuple([(lb,ub) for x in xinit])
    
    opt = optimize.minimize (f, x0 = xinit, args = (CovarReturns),  bounds = bnds, \
                             constraints = cons, tol = 10**-3)
    
    return opt

#function obtains Minimal risk and Maximum return portfolios

def MinimizeRiskConstr(MeanReturns, CovarReturns, PortfolioSize, R):
    
    def  f(x,CovarReturns):
         
        func = np.matmul(np.matmul(x,CovarReturns ), x.T)
        return func

    def constraintEq(x):
        AEq=np.ones(x.shape)
        bEq=1
        EqconstraintVal = np.matmul(AEq,x.T)-bEq 
        return EqconstraintVal
    
    def constraintIneq(x, MeanReturns, R):
        AIneq = np.array(MeanReturns)
        bIneq = R
        IneqconstraintVal = np.matmul(AIneq,x.T) - bIneq
        return IneqconstraintVal
    

    xinit=np.repeat(0.1, PortfolioSize)
    cons = ({'type': 'eq', 'fun':constraintEq},
            {'type':'ineq', 'fun':constraintIneq, 'args':(MeanReturns,R) })
    lb = 0
    ub = 1
    bnds = tuple([(lb,ub) for x in xinit])

    opt = optimize.minimize (f, args = (CovarReturns), method ='trust-constr',  \
                        x0 = xinit,   bounds = bnds, constraints = cons, tol = 10**-3)
    
    return  opt