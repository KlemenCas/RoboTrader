import commons
from database import db
import quandl as Quandl
import csv
import pandas as pd

Quandl.ApiConfig.api_key='QWQMKc-NYsoYgtfadPZs'

dba=db('','r+')
t_log=dba.t_log

subList=list()
for t in commons.getHistSp500TickerList(1,1,False):
    subList.append('WIKI/'+t)

df1=Quandl.get(subList,start_date=commons.min_date,end_date=commons.max_date['WIKI_SP500'])

columns=list([])
for x in df1.columns:
    x=str(x).replace('WIKI/','')
    x=str(x).replace(' - ', '_')
    columns.append(x)
df1.columns=columns

dfIndex=list()
for dix in range(commons.date_index_internal[commons.min_date],commons.date_index_internal[commons.max_date['WIKI_SP500']]):
   dfIndex.append(commons.date_index_external[dix]) 
df2=pd.DataFrame(index=dfIndex)
df1=df1.join(df2,how='outer')
df1=df1.sort_index()
df1=df1.fillna(method='ffill')
df1=df1.fillna(method='backfill')    

            
toRecollect=list()

with open(commons.data_path+'backtest.csv','w') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',')
    for row in t_log:
        xrow=list()
        xrow.append(commons.date_index_external[row['dix']])
        xrow.append(row['ticker'])
        xrow.append(row['tx'])
        xrow.append(row['volume'])
        try:
            xrow.append(row['price']*df1.ix[commons.date_index_external[row['dix']],row['ticker']+'_Close']/df1.ix[commons.date_index_external[row['dix']],row['ticker']+'_Adj. Close'])
        except KeyError:
            xrow.append(row['price'])
        csvwriter.writerow(xrow)

print toRecollect