import precommons_updates
import commons
import mydata_get_delta
import csv
import datetime as dt
import time
import uuid
from market import stock_market
from database import db
from forecasts import forecast
from portfolio import investments


def readPortfolio():
    globals portfolio,portfolioValue,cash    
    with open(daily_path+'currentPortfolio.csv','r') as csvfile:
        csvreader=csv.reader(csvfile, delimiter=',')
        lI=0
        for k,v in commons.getIndexCodes().items():
            index_t=v[-8:]
            portfolio[index_t]=dict()
        for row in csvreader:
            if lI>0:
                if dt.datetime.strptime(row[0],'%m/%d/%Y')==commons.idx_today:
                    if row[2]=='PORTFOLIOCASH':
                        cash[row[1]]=int(row[3])
                    elif row[2]=='PORTFOLIOVALUE':
                        portfolioValue[row[1]]=int(row[3])
                    else:
                        portfolio[row[1]][row[2]]=int(row[3])
                else:
                    raise 'currentPortfolioWrongDate'
            iI+=1
    csvfile.close()

def setRefDates():
    globals refDix,refDate,tradeDate,tradeDix
    refDix=commons.date_index_internal[commons.getClosestDate(commons.idx_today)]
    refDate=commons.date_index_external[refDix]                                   
    tradeDate=commons.getNextTradeDay(refDate)
    tradeDix=commons.date_index_internal[tradeDate]

def trainIfNeeded():
    globals f,refDix,train_uuid,m,dba
    if refDix%20==0:
        start=time.time()
        train_uuid=uuid.uuid1().hex
        #train_uuid='b988552ec64f11e69128c82a142bddcf'
        print 'Retraining the models. Date:',refDate,'training guid:',train_uuid
    
        newTraining=cl_trainSection(refDix-1,train_uuid,scenario,True)
        newTraining.train()
        end=time.time()
        print 'Training took',end-start,'seconds.'
    else:
        maxDix=0
        for row in dba.t_train_h:
            if row['enddix']>maxDix:
                maxDix=row['enddix']
                train_uuid=row['train_uuid']
    f=forecast(m,train_uuid)

def initializePortfolioAndMarket():
    globals m,p,dba,sim_uuid
    #initialize portfolio & market
    sim_uuid=uuid.uuid1().hex
    dba=db(sim_uuid,'r+')
    
    m=stock_market(dba,0,refDix,False,portfolioValue)
    p=investments(0,m,refDix,dba,sim_uuid,portfolio,cash)

#log recommendation
def logRecommendation(tradeDate,symbol,tradeTx,tradeVol,tradePrice,tradeDateCopy,_12dd):
    with open(commons.daily_path+'planedTrades.csv','r+') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(list([tradeDate,symbol,tradeTx,tradeVol,tradePrice,tradeDateCopy,_12dd]))
    csvfile.close()

#daily run
def playForADay(idx_external,sim_uuid,train_uuid,scenario):
    global f,refDix,refDate,tradeDix,tradeDate
    temperature=1.5
    state=dict()
    proposed_action=dict()
    order_entry=dict()
    order_untrained=dict()
    reward=dict()
    action=dict()
    dba.db_main.flush()
    index_t=commons.getIndexCodes()[idx_external][-8:]
    sp500_ticker=commons.getHistSp500Ticker(commons.date_index_external[refDix])

    for ticker in commons.getHistSp500Composition(commons.date_index_external[refDix])[index_t]:
        reward[ticker]=9999
        if f.trained(train_uuid,ticker):
            state[ticker]=f.get_forecast_state(ticker,refDix)

            proposed_action[ticker]=dba.get_softmax_action(ticker,state[ticker],temperature,scenario)
    #sell
            if proposed_action[ticker]==commons.action_code['sell']:               
                vol=p.portfolio[index_t][ticker]
                forecastPrice=f.get_order_price(ticker,state[ticker],refDix,proposed_action[ticker],\
                                m.get_closing_price(ticker,refDix))
                x=p.execute_order(ticker,vol,dix,forecast_price,proposed_action[ticker],\
                                  m.get_closing_price(ticker,dix),state[ticker]['12dd_Close'],False)                
                logRecommendation(tradeDate,ticker,'sell',vol,forecastPrice,tradeDate,state[ticker]['12dd_Close'])

    #buy, but only after everythin has been sold        
            if proposed_action[ticker]==commons.action_code['buy']:
                order_entry[ticker]=0

        else: #for the tickers that are not trained yet align with the index
            action[ticker]=f.getActionUntrained(p,index_t,ticker,refDix)
            if action[ticker][0]==commons.action_code['buy']:
                order_untrained[ticker]=0
            elif action[ticker][0]==commons.action_code['sell']:
                x=p.execute_order(ticker,action[ticker][1],tradeDix,m.get_closing_price(ticker,refDix),\
                                  commons.action_code['sell'],m.get_closing_price(ticker,refDix),0,False)                
                logRecommendation(tradeDate,ticker,'sellOpen',action[ticker][1],0,tradeDate,0)                
            
    #allocate for alignment
    for ticker,opening_price in order_untrained.items():
        x=p.execute_order(ticker,action[ticker][1],tradeDix,m.get_closing_price(ticker,refDix),\
                          commons.action_code['buy'],m.get_closing_price(ticker,refDix),0,False)                        
        logRecommendation(tradeDate,ticker,'buyOpen',action[ticker][1],0,tradeDate,0)                

    budget=dict()
    for k,v in commons.getIndexCodes().items():
        index_t=v[-8:]
        budget[index_t]=p.cash[index_t]
    for ticker,price in order_entry.items():
        index_t=sp500_ticker[ticker]
        budget[index_t]+=p.portfolio[index_t][ticker]*m.get_closing_price(ticker,refDix)
                
    #order book; realign the portfolio to the index according to buying recommendations            
    orderBook=p.get_portfolio_alignment(budget,order_entry,refDix)
    for ticker,volume in orderBook.items():
        if volume<0: #selling what we have too much of
            forecast_price=f.get_order_price(ticker,state[ticker],refDix,commons.action_code['sell'],\
                                             m.get_closing_price(ticker,refDix))
            x=p.execute_order(ticker,0-volume,tradeDix,forecast_price,commons.action_code['sell'],\
                              m.get_closing_price(ticker,refDix),state[ticker]['12dd_Close'],False)
            logRecommendation(tradeDate,ticker,'sell',volume,forecast_price,tradeDate,state['ticker']['12dd_Close'])                

            
    for ticker,volume in orderBook.items():
        if volume>0:
            if commons.data_sp500_1st_date[ticker]<=refDate:
                forecast_price=f.get_order_price(ticker,state[ticker],refDix,commons.action_code['buy'],\
                                                 m.get_closing_price(ticker,refDix))
                x=p.execute_order(ticker,volume,tradeDix,forecast_price,commons.action_code['buy'],\
                                m.get_closing_price(ticker,refDix),state[ticker]['12dd_Close'],False)
                logRecommendation(tradeDate,ticker,'buy',volume,forecast_price,tradeDate,state['ticker']['12dd_Close'])                
            

#EXECUTE
portfolio=dict()
portfolioValue=dict()
cash=dict()
scenario='best'
readPortfolio()
setRefDates()
initializePortfolioAndMarket()
trainIfNeeded()

for k,v in commons.getIndexCodes().items():
    playForADay(k,sim_uuid,train_uuid,scenario)
    