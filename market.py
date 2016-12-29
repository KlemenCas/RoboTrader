import commons
import pandas as pd
import numpy as np
import datetime as dt

        
class stock_market(object):
    data_sp500=pd.DataFrame()
    index_composition=dict()
    portfolio=dict()
    portfolio_log=dict()
    startdates=dict()
    transaction_price=0

    def __init__(self,dba,initial_budget,firstDate,earliest=False,portfolioValue=dict()):
        self.dba=dba
        self.transaction_price=10
        self.initialize_index()
        self.data_sp500=pd.read_hdf(commons.data_path+'WIKI_SP500.h5','table')
        self.data_sp500=self.data_sp500.fillna(method='backfill')
        startdate=dict()

        for k,v in commons.getIndexCodes().items():
            if earliest:
                startdate[v[-8:]]=min(self.index_composition[v[-8:]].index)
            else:
                startdate[v[-8:]]=commons.date_index_external[firstDate]
                
        for k,v in commons.getIndexCodes().items():
            index_t=v[-8:]
            if startdate[index_t]<=min(self.index_composition[index_t].index):
                startdate[index_t]=min(self.index_composition[index_t].index)
            
            if startdate[index_t] not in self.index_composition[index_t].index:
                while startdate[index_t] not in self.index_composition[index_t].index:
                    startdate[index_t]=startdate[index_t]+dt.timedelta(days=1)   
            if portfolioValue==dict():
                self.initialize_portfolio(k,commons.date_index_internal[startdate[index_t]],initial_budget)
            else:
                self.initialize_portfolio(k,commons.date_index_internal[startdate[index_t]],portfolioValue[index_t])

            self.startdates[index_t]=startdate[index_t]
    
    def initialize_index(self):
        for k,v in commons.getIndexCodes().items():
            index_t=v[-8:]    
            self.index_composition[index_t]=pd.read_hdf(commons.data_path+'PCT_'+index_t+'.h5','table',mode='r')
                        

    def get_index_portfolio(self,index_t,dix):
        portfolio=dict()
        for t in commons.getHistSp500Composition(commons.date_index_external[dix])[index_t]:
            if dix>=commons.date_index_internal[commons.data_sp500_1st_date[t]]: 
                portfolio[t]=self.index_composition[index_t].ix[commons.date_index_external[dix],t]
        return portfolio
        
    def index_portfolio_value(self,idx_external,dix):  
        index_t=commons.getIndexCodes()[idx_external][-8:]
        value=0
        for ticker,volume in self.portfolio[index_t].items():
            value+=volume*self.get_closing_price(ticker,dix)
        return value
            
        
    def order_executable(self,ticker,dix,price,action):
        if (action==commons.action_code['buy'] and price>=float(self.data_sp500.ix[commons.date_index_external[dix],[ticker+'_Low']].values)) or\
           (action==commons.action_code['sell'] and price<=float(self.data_sp500.ix[commons.date_index_external[dix],[ticker+'_High']].values)):
            return True
        else:
            return False
            
    def get_closing_price(self,ticker,dix):
        price = self.data_sp500.ix[commons.date_index_external[dix],[ticker+'_Close']]
        if np.isnan(price.values):
            try:
                price = self.data_sp500.ix[commons.date_index_external[dix],[commons.alternative_symbol[ticker]+'_Close']]
                price=float(price.values)
            except KeyError:
                price=0
                print 'Ticker: ',ticker, 'Date: ', commons.date_index_external[dix],' set to zero.'
        else:
            price=float(price.values)
        return price
        
    def get_opening_price(self,ticker,dix):
        price = self.data_sp500.ix[commons.date_index_external[dix],[ticker+'_Open']]
        if np.isnan(price.values):
            try:
                price = self.data_sp500.ix[commons.date_index_external[dix],[commons.alternative_symbol[ticker]+'_Open']]
                price=float(price.values)
            except KeyError:
                price=0
        else:
            price=float(price.values)
        return price        

    def get_low_price(self,ticker,dix):
        price = self.data_sp500.ix[commons.date_index_external[dix],[ticker+'_Low']]
        if np.isnan(price.values):
            try:
                price = self.data_sp500.ix[commons.date_index_external[dix],[commons.alternative_symbol[ticker]+'_Low']]
                price=float(price.values)
            except KeyError:
                price=0
        else:
            price=float(price.values)
        return price   

    def get_high_price(self,ticker,dix):
        price = self.data_sp500.ix[commons.date_index_external[dix],[ticker+'_High']]
        if np.isnan(price.values):
            try:
                price = self.data_sp500.ix[commons.date_index_external[dix],[commons.alternative_symbol[ticker]+'_High']]
                price=float(price.values)
            except KeyError:
                price=0
        else:
            price=float(price.values)
        return price   
        
    def align_index_portfolio(self,dix):
        for k,v in commons.getIndexCodes().items():
            #sell everything and repurchase with the current index composition
            index_t=v[-8:]
            cash=0
            for ticker,volume in self.portfolio[index_t].items():
                cash+=volume*self.get_closing_price(ticker,dix)
            self.initialize_portfolio(k,dix,cash)
                
    def initialize_portfolio(self,idx_external,dix,budget):
        index_t=commons.getIndexCodes()[idx_external][-8:]       
        self.portfolio[index_t]=dict()
        self.portfolio[index_t]=self.get_index_portfolio(index_t,dix)
        portfolio=dict()
        for ticker,pct in self.portfolio[index_t].items():
            portfolio[ticker]=pct*budget/self.get_closing_price(ticker,dix)
        self.portfolio[index_t]=portfolio
        try:
            a=self.portfolio_log[index_t]
        except KeyError:
            self.portfolio_log[index_t]=pd.DataFrame()
        
    def log_portfolio(self,dix,sim_uuid):
        for index_t,portfolio in self.portfolio.items():
            for ticker,volume in portfolio.items():
#                print 'Portfolio log; ticker:',ticker,'dix:',dix,'volume=',volume
                self.dba.i_log.row['sim_uuid']=sim_uuid
                self.dba.i_log.row['index_t']=index_t
                self.dba.i_log.row['dix']=dix
                self.dba.i_log.row['ticker']=ticker
                self.dba.i_log.row['volume']=volume
                self.dba.i_log.row.append()
                self.dba.i_log.flush()

                    
    def get_min_startdate(self):
        startdate=dt.datetime.today()
        for i,d in self.startdates.items():
            if startdate>d:
                startdate=d
        return startdate