#file with all stock prices is too big for github. it's being reduced to the demo scenario here.

import pandas as pd
import commons

#stock prices
data_sp500=pd.read_hdf(commons.local_path+'data/WIKI_SP500.h5','table')

c=list()
for symbol in commons.sp500_composition['Telecommunications Services']:
    c.append(symbol+'_Ex-Dividend')
    c.append(symbol+'_Split Ratio')    
    c.append(symbol+'_Open')    
    c.append(symbol+'_High')    
    c.append(symbol+'_Low')    
    c.append(symbol+'_Close')    
    c.append(symbol+'_Volume')   

data_sp500_reduced=data_sp500[c]
data_sp500_reduced.to_hdf(commons.local_path+'data_for_github/WIKI_SP500.h5','table',mode='w')

#alpha and beta
anb=pd.read_hdf(commons.local_path+'data/anb.h5','table')

c=list()
for symbol in commons.sp500_composition['Telecommunications Services']:
    c.append('B_'+symbol+'_Open')
    c.append('B_'+symbol+'_Close')    
    c.append('B_'+symbol+'_High')    
    c.append('B_'+symbol+'_Low')    
    c.append('A_'+symbol+'_Open')
    c.append('A_'+symbol+'_Close')    
    c.append('A_'+symbol+'_High')    
    c.append('A_'+symbol+'_Low')    

anb_reduced=anb[c]
anb_reduced.to_hdf(commons.local_path+'data_for_github/anb.h5','table',mode='w')