import commons
import pandas as pd   
import tables 

        
class investments(object):
    sim_uuid=''
    portfolio=dict()
    cash=dict()
    market=None
    portfolio_value=dict()
    portfolio_log=dict()

    def __init__(self,initial_budget,market,init_dix,dba,sim_uuid,portfolio=dict(),cash=dict()): 
        self.dba=dba
        self.sim_uuid=sim_uuid
        self.market=market
        if portfolio==dict():
            self.initialize_cash(initial_budget)
            self.initialize_portfolio(init_dix)   
        else:
            self.refreshPortfolio(portfolio)
            self.refreshCash(cash)
            
    def refreshPortfolio(self,portfolio):
        self.portfolio=portfolio
        
    def refreshCash(self,cash):
        self.cash=cash

    def initialize_cash(self,initial_budget):
        for k,v in commons.getIndexCodes().items():
            index_t=v[-8:]
            self.cash[index_t]=initial_budget
        

    def initialize_portfolio(self,dix):
        for k,v in commons.getIndexCodes().items():
            index_t=v[-8:]    
            portfolio=dict()        
            self.portfolio[index_t]=self.market.get_index_portfolio(index_t,dix)
            for ticker,pct in self.portfolio[index_t].items():
                portfolio[ticker]=int(pct*self.cash[index_t]/self.market.get_closing_price(ticker,dix))
            for k,v in portfolio.items():
                self.cash[index_t]-=float(v*self.market.get_closing_price(k,dix))
            self.portfolio[index_t]=portfolio
            self.portfolio_log[index_t]=pd.DataFrame()
            
            
    def align_buying_list(self,buying_list,dix):
        delta=dict()
        cash_and_valuation=dict()
        aligned_order_list=dict()
        aligned_order_list_clean=dict()
        remaining_amount=dict(self.cash.items())
        
#1st calculate total value - portfolio + cash
        for k,v in commons.getIndexCodes().items():
            index_t=v[-8:]            
            cash_and_valuation[index_t]=0
            for ticker,volume in self.portfolio[index_t].items():
                cash_and_valuation[index_t]+=volume*self.market.get_closing_price(ticker,dix)
        for k,v in self.cash.items():
            cash_and_valuation[k]+=v

#get delta to target portfolio
        for ticker,price in buying_list.items():
            try:
                test=self.portfolio[index_t][ticker]
                delta=self.market.get_index_portfolio(commons.sp500_ticker[ticker],dix)[ticker]*cash_and_valuation[index_t]-\
                        self.portfolio[index_t][ticker]*self.market.get_closing_price(ticker,dix)
            except KeyError:
                delta=self.market.get_index_portfolio(commons.sp500_ticker[ticker],dix)[ticker]*cash_and_valuation[index_t]
            index_t=commons.getIndexCodes()[commons.sp500_ticker[ticker]][-8:]
            if delta>remaining_amount[index_t]:
                delta=remaining_amount[index_t]
            if delta>0. and remaining_amount[index_t]>0:
                aligned_order_list[ticker]=int(delta/price)
                remaining_amount[index_t]=remaining_amount[index_t]-aligned_order_list[ticker]*price
#                print 'Buying List Alignment-Delta; Order List: ',aligned_order_list,' Remaining Amount: ',remaining_amount

#and spend what's left equally on the buying list         
        cum_weight=dict()                   
        for k,v in commons.sp500_composition.items():
            index_t=commons.getIndexCodes()[k][-8:]
            cum_weight[index_t]=0
            for ticker,price in buying_list.items():
                if ticker in v:
                    cum_weight[index_t]+=self.market.get_index_portfolio(commons.sp500_ticker[ticker],dix)[ticker]

        for ticker,price in buying_list.items():                                           
            index_t=commons.getIndexCodes()[commons.sp500_ticker[ticker]][-8:]
            try:
                aligned_order_list[ticker]=aligned_order_list[ticker]+int((self.market.get_index_portfolio(commons.sp500_ticker[ticker],dix)[ticker]/cum_weight[index_t])*\
                                       remaining_amount[index_t]/price)
            except KeyError:
                aligned_order_list[ticker]=0
        for ticker,volume in aligned_order_list.items():
            if volume>0:
                aligned_order_list_clean[ticker]=volume
                self.dba.log_recommendation(self.sim_uuid,dix,self.dba.ti_ticker_ids[ticker],9,volume)
        return aligned_order_list_clean

#check buying list
    def check_buying_list(self,buying_list,dix):
        for k,v in commons.getIndexCodes().items():
            portfolio=self.market.get_index_portfolio(k,dix)
            for ticker in buying_list:
                if commons.sp500_ticker[ticker]==k:
                    found=''
                    for x,y in portfolio.items():
                        if ticker==x:
                            found='X'
                    if found=='':
                        print 'Missing: ',ticker

    def logTransaction(self,simUuid,dix,ticker,price,volume,close,cashBefore,cashAfter,_12dd,tx):
        self.dba.t_log.row['sim_uuid']=self.sim_uuid
        self.dba.t_log.row['dix']=dix-1
        self.dba.t_log.row['ticker']=ticker
        self.dba.t_log.row['price']=price
        self.dba.t_log.row['volume']=volume
        self.dba.t_log.row['close']=self.market.get_closing_price(ticker,dix)
        self.dba.t_log.row['cash_before']=cashBefore     
        self.dba.t_log.row['cash_after']=cashAfter
        self.dba.t_log.row['12dd']=_12dd
        self.dba.t_log.row['tx']=tx
        self.dba.t_log.row.append()
        self.dba.t_log.flush()
                                
#execute order
    def execute_order(self,ticker,volume,dix,price,action,last_close,_12dd,checkExecutable=True):
        try:
            index_t=commons.getHistSp500Ticker(commons.date_index_external[dix])[ticker]
        except KeyError:
            index_t=commons.getHistSp500Ticker(commons.date_index_external[dix-1])[ticker]
        cash_before=self.cash[index_t]
        try:
            a=self.portfolio[index_t][ticker]
        except KeyError:
            self.portfolio[index_t][ticker]=0

        if (self.market.order_executable(ticker,dix,price,action)) or not checkExecutable:
            if action==commons.action_code['buy']:
                if self.cash[index_t]<(price*volume+self.market.transaction_price):
                    volume=int((self.cash[index_t]/(price*volume+self.market.transaction_price))*volume)
                self.portfolio[index_t][ticker]+=volume
                self.cash[index_t]-=price*volume-self.market.transaction_price
                tx='buy'
            if action==commons.action_code['sell']:
                self.portfolio[index_t][ticker]-=volume
                self.cash[index_t]+=price*volume-self.market.transaction_price
                tx='sell'

        else:
            if action==commons.action_code['buy']:
                tx='canc_buy'
            elif action==commons.action_code['sell']:
                tx='canc_sell'
            
        self.logTransaction(self.sim_uuid,dix-1,ticker,price,volume,self.market.get_closing_price(ticker,dix),\
                            cash_before,self.cash[index_t],_12dd,tx)

        return self.getReward(ticker,dix,last_close,action)

    def getReward(self,ticker,dix,last_close,action):
        if (self.market.get_closing_price(ticker,dix)-last_close)>0 and action==commons.action_code['buy']:
            reward=200
        if (self.market.get_closing_price(ticker,dix)-last_close)>0 and action==commons.action_code['sell']:
            reward=-200
        if (self.market.get_closing_price(ticker,dix)-last_close)==0 and action==commons.action_code['buy']:
            reward=100
        if (self.market.get_closing_price(ticker,dix)-last_close)==0 and action==commons.action_code['sell']:
            reward=100
        if (self.market.get_closing_price(ticker,dix)-last_close)<0 and action==commons.action_code['buy']:
            reward=-200
        if (self.market.get_closing_price(ticker,dix)-last_close)<0 and action==commons.action_code['sell']:
            reward=+200
        return reward
        
    def get_portfolio_value(self,idx,dix):
        value=0
        for k,v in self.portfolio[idx].items():
            value+=v*self.market.get_closing_price(k,dix)
        return value
            
    def log_portfolio(self,dix,sim_uuid):
        for index_t,portfolio in self.portfolio.items():
            for ticker,volume in portfolio.items():
                self.dba.p_log.row['sim_uuid']=sim_uuid
                self.dba.p_log.row['index_t']=index_t
                self.dba.p_log.row['dix']=dix
                self.dba.p_log.row['ticker']=ticker
                self.dba.p_log.row['volume']=volume
                self.dba.p_log.row.append()
                self.dba.p_log.flush()
            
                
    def get_index_alignment(self,budget,recommendations,dix):
#% of the index
        pct=dict()
        index=dict()
        basket=dict()
        sp500_ticker=commons.getHistSp500Ticker(commons.date_index_external[dix])
        for idx_external,v in commons.getIndexCodes().items():
            index[v[-8:]]=self.market.get_index_portfolio(v[-8:],dix)
        for t,p in recommendations.items():
            try:
                pct[sp500_ticker[t]]+=index[sp500_ticker[t]][t]
            except KeyError:
                pct[sp500_ticker[t]]=index[sp500_ticker[t]][t]
        for t,p in recommendations.items():
            index_t=sp500_ticker[t]
            basket[t]=int(index[sp500_ticker[t]][t]/pct[sp500_ticker[t]]*budget[index_t]/self.market.get_closing_price(t,dix))
        return basket
        
    def get_portfolio_alignment(self,budget,recommendations,dix):
        sp500_ticker=commons.getHistSp500Ticker(commons.date_index_external[dix])
        aligned_portfolio=self.get_index_alignment(budget,recommendations,dix)
        
        orderBook=dict()
        for t,p in recommendations.items():
            orderBook[t]=aligned_portfolio[t]-self.portfolio[sp500_ticker[t]][t]
        return orderBook
                
                
            
                
            
        
                
            
        