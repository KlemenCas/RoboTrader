import commons
from market import stock_market
from database import db
from forecasts import forecast
from portfolio import investments
import uuid
from trainSection import cl_trainSection 

def play_for_a_day(idx_external,dix,alpha,gamma,sim_uuid,train_uuid):
    global f
    temperature=1.5
    state=dict()
    proposed_action=dict()
    order_entry=dict()
    order_random=dict()
    reward=dict()
    dba.db_main.flush()
    random_action=dict()
    index_t=commons.getIndexCodes()[idx_external][-8:]
    sp500_ticker=commons.getHistSp500Ticker(commons.date_index_external[dix])

    for ticker in commons.getHistSp500Composition(commons.date_index_external[dix])[index_t]:
        reward[ticker]=9999
        if commons.data_sp500_1st_date[ticker]<=commons.date_index_external[dix]:
            try:
                state[ticker]=new_state[ticker]
            except KeyError:
                state[ticker]=f.get_forecast_state(dba.t_stats,ticker,dix)
            proposed_action[ticker]=dba.get_softmax_action(ticker,state[ticker],temperature)
            max_q_action=dba.get_max_q(ticker,state[ticker])
            if max_q_action[2]==False and max_q_action[1]==proposed_action[ticker]:
                random_action[ticker]=False
            else:
                random_action[ticker]=True
##compare max_axction and proposed action and if different then set volume to 1
            next_dix=dix+1
    #sell
            if proposed_action[ticker]==commons.action_code['sell']:               
                try:
                    if random_action[ticker]==False:
                        vol=p.portfolio[index_t][ticker]
                    else:
                        vol=1
                    opening_price=m.get_opening_price(ticker,next_dix)
                    forecast_price=f.get_order_price(dba,ticker,state[ticker],dix,proposed_action[ticker],m.get_closing_price(ticker,dix))

                    if opening_price>(forecast_price*.997):
                        forecast_price=opening_price*.997
                    reward[ticker]=p.execute_order(ticker,vol,next_dix,forecast_price,proposed_action[ticker],m.get_closing_price(ticker,dix))
                    dba.log_recommendation(sim_uuid,dix,ticker,commons.action_code['sell'],vol)    

                except KeyError:
                    p.portfolio[index_t][ticker]=0
                    print 'New Ticker: ', ticker
    
    #buy, but only after everythin has been sold        
            if proposed_action[ticker]==commons.action_code['buy']:
                if random_action[ticker]==False:
                    order_entry[ticker]=m.get_opening_price(ticker,next_dix)
                else:
                    order_random[ticker]=m.get_opening_price(ticker,next_dix)
                dba.log_recommendation(sim_uuid,dix,ticker,commons.action_code['buy']) 
                
    #allocate money for the randoms
    for ticker,opening_price in order_random.items():
        forecast_price=f.get_order_price(dba,ticker,state[ticker],dix,commons.action_code['buy'],m.get_closing_price(ticker,dix))
        if order_random[ticker]*1.01<forecast_price:
            forecast_price=order_random[ticker]*1.01
        reward[ticker]=p.execute_order(ticker,1,next_dix,forecast_price,commons.action_code['buy'],m.get_closing_price(ticker,dix))
    
    budget=dict()
    for k,v in commons.getIndexCodes().items():
        index_t=v[-8:]
        budget[index_t]=p.cash[index_t]
    for ticker,price in order_entry.items():
        index_t=sp500_ticker[ticker]
        budget[index_t]+=p.portfolio[index_t][ticker]*m.get_closing_price(ticker,dix)
                
    #order book; realign the portfolio to the index according to buying recommendations            
    orderBook=p.get_portfolio_alignment(budget,order_entry,dix)
    for ticker,volume in orderBook.items():
        if volume<0: #selling what we have too much of
            forecast_price=f.get_order_price(dba,ticker,state[ticker],dix,commons.action_code['sell'],m.get_closing_price(ticker,dix))
            if order_entry[ticker]>(forecast_price*.997):
                forecast_price=order_entry[ticker]*.997
            a=p.execute_order(ticker,0-volume,next_dix,forecast_price,commons.action_code['sell'],m.get_closing_price(ticker,dix))
            
    for ticker,volume in orderBook.items():
        if volume>0:
            if commons.data_sp500_1st_date[ticker]<=commons.date_index_external[dix]:
                forecast_price=f.get_order_price(dba,ticker,state[ticker],dix,commons.action_code['buy'],m.get_closing_price(ticker,dix))
                if order_entry[ticker]<forecast_price:
                    forecast_price=order_entry[ticker]
                reward[ticker]=p.execute_order(ticker,volume,next_dix,forecast_price,commons.action_code['buy'],m.get_closing_price(ticker,dix))
                dba.log_recommendation(sim_uuid,dix,ticker,commons.action_code['buy'],volume)
            
#on the way to the next q
    dix+=1
    for ticker in commons.getHistSp500Composition(commons.date_index_external[dix])[index_t]:
        if commons.data_sp500_1st_date[ticker]<=commons.date_index_external[dix]:
            new_state[ticker]=f.get_forecast_state(dba.t_stats,ticker,dix)
            try:
                if reward[ticker]!=9999:
                    newQ=dba.get_reward(ticker,state[ticker],proposed_action[ticker])+\
                            alpha*(reward[ticker]+gamma*dba.get_max_q(ticker,new_state[ticker])[0]-dba.get_reward(ticker,state[ticker],proposed_action[ticker]))
                    dba.update_q_table(ticker,state[ticker],proposed_action[ticker],newQ,dix)
            except KeyError:
                p.portfolio[index_t][ticker]=0
                print 'Date:',dix,' New Ticker:',ticker

        
def getMaxSimrun(dba):
    maxsim=0
    for row in dba.s_log:
        if row['simrun']>maxsim:
            maxsim=row['simrun']
    return maxsim+1
    
gamma=.8
alpha=.5

for simrun in range(1,10):
    sim_uuid=uuid.uuid1().hex
    dba=db(sim_uuid,'r+')
    
    initial_budget=100000
    m=stock_market(dba,initial_budget,True)

    new_state=dict()
    p=investments(initial_budget,m,11084,dba,sim_uuid)

    dba.new_simulation(sim_uuid,commons.date_index_internal[commons.max_date['WIKI_SP500']],11084,commons.date_index_internal[commons.max_date['WIKI_SP500']])

    runningyear=0
    for dix in range(11624,commons.date_index_internal[commons.max_date['WIKI_SP500']]):
        if (dix-4)%10==0:
            newTraining=cl_trainSection(dix)
            f=forecast(m,newTraining)
            train_uuid=newTraining.train()
            
        if commons.date_index_external[dix].year!=runningyear:
            maxsim=getMaxSimrun(dba)
            print 'Simulation for year:',commons.date_index_external[dix].year,'started. Simulation:',maxsim
        runningyear=commons.date_index_external[dix].year

        if dix!=11624:
            m.align_index_portfolio(dix)
            
        p.log_portfolio(dix,sim_uuid)
        m.log_portfolio(dix,sim_uuid)
        
        for k,v in commons.getIndexCodes().items():
            index_t=v[-8:]   
            print 'Date: ',commons.date_index_external[dix],' Index: ',v[-8:],'Portfolio: ',int(p.get_portfolio_value(v[-8:],dix)),\
                    ' Cash: ',int(p.cash[v[-8:]]),' Total: ',int(p.get_portfolio_value(v[-8:],dix))+int(p.cash[v[-8:]]),\
                    ' Index: ',int(m.index_portfolio_value(k,dix))                             
            dba.s_log.row['gamma']=gamma
            dba.s_log.row['alpha']=alpha
            dba.s_log.row['simrun']=maxsim
            dba.s_log.row['sim_uuid']=sim_uuid
            dba.s_log.row['dix']=dix
            dba.s_log.row['index']=v[-8:]
            dba.s_log.row['p_value']=int(p.get_portfolio_value(v[-8:],dix))
            dba.s_log.row['cash']=int(p.cash[v[-8:]])
            dba.s_log.row['i_value']=int(m.index_portfolio_value(k,dix))
            dba.s_log.row.append()
            dba.s_log.flush()
            
            play_for_a_day(k,dix, alpha, gamma,sim_uuid,train_uuid)
                    