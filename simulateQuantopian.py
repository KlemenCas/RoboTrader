import commons
from market import stock_market
from database import db
from forecasts import forecast
from portfolio import investments
import uuid
from trainSection import cl_trainSection 
import time

def play_for_a_day(idx_external,dix,alpha,gamma,sim_uuid,train_uuid,scenario):
    global f,m
    temperature=1.5
    state=dict()
    proposed_action=dict()
    reward=dict()
    dba.db_main.flush()
    index_t=commons.getIndexCodes()[idx_external][-8:]

    for ticker in commons.getHistSp500Composition(commons.date_index_external[dix])[index_t]:
        reward[ticker]=9999
        if f.trained(train_uuid,ticker):
            state[ticker]=f.get_forecast_state(ticker,dix)

            proposed_action[ticker]=dba.get_softmax_action(ticker,state[ticker],temperature,scenario)
            price=f.get_order_price(ticker,state[ticker],dix,proposed_action[ticker],m.get_closing_price(ticker,dix))
            dba.log_recommendation(sim_uuid,dix,ticker,proposed_action[ticker],price,state[ticker]['12dd_Close'],m.get_index_portfolio(index_t,dix)[ticker])
        else:
            dba.log_recommendation(sim_uuid,dix,ticker,9,0,0,m.get_index_portfolio(index_t,dix)[ticker])
            
        
def getMaxSimrun(dba):
    maxsim=0
    for row in dba.s_log:
        if row['simrun']>maxsim:
            maxsim=row['simrun']
    return maxsim+1
    
gamma=.8
alpha=.5

sim_uuid=uuid.uuid1().hex
dba=db(sim_uuid,'r+')
scenario='best'
offset=1500

initial_budget=100000
firstDate=13254
m=stock_market(dba,initial_budget,firstDate,False)

new_state=dict()
p=investments(initial_budget,m,firstDate,dba,sim_uuid)

dba.new_simulation(sim_uuid,commons.date_index_internal[commons.max_date['WIKI_SP500']],firstDate,commons.date_index_internal[commons.max_date['WIKI_SP500']])

runningyear=0
for dix in range(firstDate,commons.date_index_internal[commons.max_date['WIKI_SP500']]):
    print 'date;',commons.date_index_external[dix]
    if (dix-14)%20==0:
        start=time.time()
        train_uuid=uuid.uuid1().hex
        #train_uuid='2bf75d91cdb411e68bbbc82a142bddcf'
        print 'Retraining the models. Date:',commons.date_index_external[dix],'training guid:',train_uuid

        newTraining=cl_trainSection(dix-1,train_uuid,scenario,offset,True)
        newTraining.train()
        f=forecast(m,train_uuid)
        end=time.time()
        print 'Training took',end-start,'seconds.'
        
        
        
    if commons.date_index_external[dix].year!=runningyear:
        maxsim=getMaxSimrun(dba)
        print 'Simulation for year:',commons.date_index_external[dix].year,'started. Simulation:',maxsim
    runningyear=commons.date_index_external[dix].year

    p.log_portfolio(dix,sim_uuid)
    
    for k,v in commons.getIndexCodes().items():
        index_t=v[-8:]   
        
        play_for_a_day(k,dix, alpha, gamma,sim_uuid,train_uuid,scenario)
                