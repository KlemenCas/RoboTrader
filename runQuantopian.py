import time
start=time.time()
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

import datetime as dt

def play_for_a_day(idx_external,dix,sim_uuid,train_uuid):
    global rQ
    state=dict()
    proposed_action=dict()
    dba.db_main.flush()
    index_t=commons.getIndexCodes()[idx_external][-8:]

    for ticker in commons.getHistSp500Composition(commons.date_index_external[dix])[index_t]:
        if rQ.trained(train_uuid,ticker):
            state[ticker]=rQ.get_forecast_state(ticker,dix)
            proposed_action[ticker]=rQ.get_best_action(state[ticker])
            price=rQ.get_order_price(ticker,state[ticker],dix,proposed_action[ticker])
            dba.log_recommendation(sim_uuid,dix,ticker,proposed_action[ticker],price,state[ticker]['12dd_Close'],rQ.get_index_portfolio(index_t,dix)[ticker])
            
        else:
            dba.log_recommendation(sim_uuid,dix,ticker,9,0,0,rQ.get_index_portfolio(index_t,dix)[ticker])
            
        
sim_uuid=uuid.uuid1().hex
dba=db(sim_uuid,'r+')
offset=1500
firstRun=True

refDate=commons.getClosestDate(commons.idx_today)
refDix=commons.date_index_internal[refDate]


print 'date;',commons.date_index_external[refDix]
if refDix%20==0 or firstRun:
    start=time.time()
    train_uuid=uuid.uuid1().hex
    print 'Retraining the models. Date:',commons.date_index_external[refDix],'training guid:',train_uuid

    newTraining=cl_trainSection(refDix-1,train_uuid,offset,True)
    newTraining.train()
    dba.new_training(train_uuid,refDix)
    end=time.time()
    print 'Training took',end-start,'seconds.'
    

dba=db(sim_uuid,'r+')
rQ=cl_runQ(dba)
    
for k,v in commons.getIndexCodes().items():
    index_t=v[-8:]       
    play_for_a_day(k,refDix, sim_uuid,train_uuid)
    
#extract data
import extract_stats
            