import commons
import csv
import pandas as pd
import datetime as dt
import numpy as np

sp500_ticker=dict()
sp500_composition=dict()
#load sp500 composition        
with open(commons.local_path+'data/WIKI.csv','r') as csvfile:
    csvreader=csv.reader(csvfile, delimiter=',')
    for row in csvreader:
        if row[0]=='BF_B':
            row[0]='BFB'
        if row[0]=='DISCK':
            row[0]='DISCA'
        if row[0]=='BRK_B':
            row[0]='BRKB'
        if row[0]!='GOOG' and row[0]!='FOXA' and row[0]!='UA_C' and row[0]!='NWS':
            sp500_ticker[row[0]]=row[1]
    csvfile.close()

for k,v in sp500_ticker.items():
    a=sp500_composition.get(v,'nok')
    if a=='nok':
        sp500_composition[v]=list([k])
    else:
        b=sp500_composition[v]
        b.append(k)
        sp500_composition[v]=b

sp500_new=dict()
sp500_dropped=dict()

with open(commons.local_path+'backup/SP500TickerNew.csv','r') as csvfile:
    csvreader=csv.reader(csvfile, delimiter=',')
    i=1
    for row in csvreader:
        if i>1:
            sp500_new[row[0]]=row[2]
            sp500_ticker[row[0]]=row[1]
        i+=1
    csvfile.close()

with open(commons.local_path+'backup/SP500TickerDropped.csv','r') as csvfile:
    csvreader=csv.reader(csvfile, delimiter=',')
    i=1
    for row in csvreader:
        if i>1:
            sp500_dropped[row[0]]=row[2]
            sp500_ticker[row[0]]=row[1]
        i+=1
    csvfile.close()

#store ticker
with open(commons.local_path+'data/sp500_ticker.csv', 'w') as f:  # Just use 'w' mode in 3.x
    w = csv.DictWriter(f, sp500_ticker.keys())
    w.writeheader()
    w.writerow(sp500_ticker)
    f.close()

sp500_comp=commons.read_dataframe(commons.local_path+'data/sp500_composition.h5')
sp500=commons.read_dataframe(commons.local_path+'data/wiki_sp500.h5')
#get 1 for all that are in the index today
np1=np.ones((len(pd.date_range(dt.date(2000,1,1),dt.date.today())),len(sp500_ticker)))
columns=list()
for k,i in sp500_ticker.items():
    columns.append(k)
sp500_comp=pd.DataFrame(np1,index=pd.date_range(dt.date(2000,1,1),dt.date.today()),columns=columns)
    
#cut of at the introduction
for k,i in sp500_new.items():
    for d in pd.date_range(dt.date(2000,1,1),i):
        sp500_comp.ix[d,k]=0

          
#cut off at dropping
for k,i in sp500_dropped.items():
    startdate=dt.datetime.strptime(sp500_dropped[k], "%m/%d/%Y")
    for d in pd.date_range(startdate+dt.timedelta(days=1),dt.date.today()):
        sp500_comp.ix[d,k]=0
        
#drop obsolete
for d in sp500_comp.index:
    if d not in sp500.index:
        sp500_comp=sp500_comp.drop(d,axis=0)

sp500_comp.to_hdf(commons.local_path+'data/sp500_composition.h5','table')


