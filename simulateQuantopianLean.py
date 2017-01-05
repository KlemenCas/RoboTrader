refreshData=False #True if new data from quandl needed
simulation=True #False if forecast for tomorrow
offsetTraining=700 #take last x days for the training
minTraining=200 #new ticker below this time will be traded according to index %
firstRun=True
firstDix=13254 #1/1/2014
dailyRun=False #set to True, if forecast for tomorrow



import time
start=time.time()
if refreshData:
    import precommons_updates
    end=time.time()
    print 'precommons updates done'
    print 'runtime:',end-start
    import mydata_get_delta
    print 'data refreshed'
import reload_stats
print 'stats reloaded'


import commons
from database import db
from runQlib import cl_runQ
import uuid
from trainSection import cl_trainSection 

def play_for_a_day(idx_external,dix,sim_uuid,minTraining,offsetTraining):
    global rQ
    state=dict()
    proposed_action=dict()
    dba.db_main.flush()
    index_t=commons.getIndexCodes()[idx_external][-8:]

    for ticker in commons.getHistSp500Composition(commons.date_index_external[dix])[index_t]:
        if rQ.trained(ticker):
            state[ticker]=rQ.get_forecast_state(ticker,dix)
            proposed_action[ticker]=rQ.get_best_action(state[ticker])
            price=rQ.get_order_price(ticker,state[ticker],dix,proposed_action[ticker])
            dba.log_recommendation(minTraining,offsetTraining,sim_uuid,dix,ticker,proposed_action[ticker],price,state[ticker]['12dd_Close'],rQ.get_index_portfolio(index_t,dix)[ticker],rQ.getModelAccuracy(ticker,'1dd_Close'))
            
        else:
            dba.log_recommendation(minTraining,offsetTraining,sim_uuid,dix,ticker,9,[0,0,0],0,rQ.get_index_portfolio(index_t,dix)[ticker],0)
            
        
sim_uuid=uuid.uuid1().hex
dba=db(sim_uuid,'r+',simulation)

#maxDix=commons.date_index_internal[commons.max_date['WIKI_SP500']]+1 #12/13/16
maxDix=13521 #12/31/14
#maxDix=13786 #12/31/15

for dix in range(firstDix,commons.date_index_internal[commons.max_date['WIKI_SP500']]+1):
    if dailyRun:
        refDate=commons.getClosestDate(commons.idx_today)
        refDix=commons.date_index_internal[refDate]
    else:
        refDate=commons.date_index_external[dix]
        refDix=dix
    
    
    print 'date;',commons.date_index_external[refDix]
    if refDix%20==0 or firstRun:
        start=time.time()
        train_uuid=uuid.uuid1().hex
        print 'Retraining the models. Date:',commons.date_index_external[refDix],'training guid:',train_uuid
    
        newTraining=cl_trainSection(minTraining,refDix-1,train_uuid,offsetTraining,True)
        newTraining.train()
        dba.new_training(train_uuid,refDix)
        end=time.time()
        print 'Training took',end-start,'seconds.'
        firstRun=False
        
    dba.db_main.close()
    dba=db(sim_uuid,'r+',simulation)
    rQ=cl_runQ(dba)
        
    for k,v in commons.getIndexCodes().items():
        index_t=v[-8:]       
        play_for_a_day(k,refDix, sim_uuid,minTraining,offsetTraining)
        
    if dailyRun:
        break
        
#extract data
import extract_stats
            