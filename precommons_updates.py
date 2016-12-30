import commons
import datetime as dt
import csv
import urllib2
from bs4 import BeautifulSoup
import tables
import pandas as pd
import quandl as Quandl
import numpy as np
import time

Quandl.ApiConfig.api_key='QWQMKc-NYsoYgtfadPZs'

def cleanTicker(ticker):
    if ticker=='AV':
        return 'AV1'        
    elif ticker=='UST':
        return 'UST1'    
    elif ticker!='GOOG' and ticker!='FOXA' and ticker!='UA_C' and ticker!='NWS' and ticker!='UAA':
        return ticker.replace('.','_')
    else:
        return False

def loadSp500Composition():
    global sp500_composition
    global sp500_ticker
    
    #load sp500 composition      
    hdr = {'User-Agent': 'Mozilla/5.0'}
    site = 'http://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    req = urllib2.Request(site, headers=hdr)
    page = urllib2.urlopen(req)
    soup = BeautifulSoup(page,"lxml")
    
    table = soup.find('table', {'class': 'wikitable sortable'})
    sector_tickers = dict()
    for row in table.findAll('tr'):
        col = row.findAll('td')
        if len(col) > 0:
            sector = str(col[3].string.strip())
            ticker = str(col[0].string.strip()).replace('.','_')
            ticker = str(col[0].string.strip()).replace('-','_')
            if sector not in sector_tickers:
                sector_tickers[sector] = list()
            if cleanTicker(ticker)!=False:
                sector_tickers[sector].append(cleanTicker(ticker))
    sp500_composition=sector_tickers
    sp500_ticker=dict()
    for k,v in sp500_composition.items():
        for i in v:
            sp500_ticker[i]=k
    del k,v
    
    with open(commons.data_path+'WIKI.csv','w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        for k,v in sp500_ticker.items():
            csvwriter.writerow([k,v])
        csvfile.close()
    print 'SP500 composition recorded'   

def initSp500Changes():
    global sp500Changes
    #load sp500 changes
    db_main = tables.open_file(commons.stats_path+'sp500Changes.h5', 'w')
    try:
        db_main.remove_node('/','sp500Changes')
    except tables.exceptions.NoSuchNodeError:
        a=1
        
    desc={'ticker':tables.StringCol(10),
          'dix':tables.IntCol(),
          'sector':tables.StringCol(8),
          'change':tables.StringCol(10)}
    sp500Changes=db_main.create_table('/','sp500Changes',desc)
    sp500Changes.cols.ticker.create_index()
    sp500Changes.cols.dix.create_index(kind='full')    

def loadSp500Data():
    global data_sp500
    global maxDixPreWikiRefresh
    data_sp500=commons.read_dataframe(commons.data_path+'WIKI_SP500.h5')
    maxDixPreWikiRefresh=dt.datetime.strptime('01/03/2006','%m/%d/%Y')

def refreshWikiSp500(ticker,changes=False):
    global date_index_internal, date_index_external
    global sp500Changes
    global sp500_ticker
    global data_sp500
    global maxDixPreWikiRefresh
    global changeItemsClean
    global regularItemsClean
    localItems=dict()

    if ticker=='all':
        if changes:
            newChanges=sp500Changes.read_where('(dix>'+str(date_index_internal[commons.max_date['WIKI_SP500']])+')')
    #        newChanges=sp500Changes.read_where('(dix>13782)')
    
            #has the sp500 composition changed?
            if any(newChanges):
                for row in newChanges:
                    item=dict()
                    if row['change']=='new':
                        item['startdate']=date_index_external[row['dix']]
                        item['enddate']=max_date=max(data_sp500.index)
                        item['sector']=commons.sp500_index[row['sector']][-8:]
                    elif row['change']=='dropped':
                        try:
                            x=getattr(data_sp500,row['ticker']+'_Open')
                            y=x[x>0]
                            item['startdate']=max(y.index)
                        except AttributeError:
                            item['startdate']=commons.min_date
                        item['sector']=commons.sp500_index[row['sector']][-8:]
                        item['enddate']=date_index_external[row['dix']]
                    if item['startdate']<=item['enddate']:
                        localItems['WIKI/'+row['ticker']]=item
                        changeItemsClean[k]=item
    
        else: #current sp500 members
            max_date=commons.max_date['WIKI_SP500']
            maxDixPreWikiRefresh=commons.max_date['WIKI_SP500']
        
            if max_date!=commons.idx_today:
                #collect unknown days and update data_SP500
                for k,v in sp500_ticker.items():
                    item = dict()
                    item['startdate']=max_date+dt.timedelta(days=1)
                    item['enddate']=commons.idx_today                    
                    item['sector']=commons.sp500_index[v][-8:]
                    localItems['WIKI/'+k]=item
                    regularItemsClean[k]=item            
        
        if any(localItems): #merge all with the same date range into 1 dict entry
            items=dict()
            for t,d in localItems.items():
                key=(d['startdate'],d['enddate'])
                try:
                    items[key].append(t)
                except KeyError:
                    items[key]=list()
                    items[key].append(t)
            
            #but only collect max 50 at a time
            df1=pd.DataFrame()
            for dates,tickerlist in items.items():
                subList=list()
                for i in range(0,len(tickerlist)):
                    subList.append(tickerlist[i])
                    if i%50==0 or i==(len(tickerlist)-1):
                        df0=pd.DataFrame()
    #                    print subList
                        df0=Quandl.get(subList,start_date=dates[0],end_date=dates[1])
                        df1=df1.join(df0,how='outer')
                        subList=list()
    
            columns=list([])
            for x in df1.columns:
                x=str(x).replace('WIKI/','')
                x=str(x).replace(' - ', '_')
                columns.append(x)
            df1.columns=columns
    
            full_refresh_items=list([])
            for i in df1.index:
                for c in df1.columns:
                    if 'Adj.' in c:
                        target_c=c.replace('Adj. ','')
                        data_sp500.ix[i,target_c]=df1.ix[i,c]
                    if 'Split Ratio'in c and df1.ix[i,c]!=1.:
                            full_refresh_items.append('WIKI/'+str(c).strip('_Split Ratio'))
            print 'Missing SP500 data days retrieved'
            
            #recollect data where there was a split
            if len(full_refresh_items)!=0:
                df0=pd.DataFrame()
                df0=Quandl.get(full_refresh_items,start_date=commons.min_date,end_date=commons.max_date['WIKI_SP500'])
                columns=list()
                for x in df0.columns:
                    x=str(x).replace(' - ', '_').strip('WIKI').strip('/')
                    columns.append(x)
                df0.columns=columns
                for i in df0.index:
                    for c in df0.columns:
                        if 'Adj.' in c:
                            target_c=c.replace('Adj. ','')                        
                            data_sp500.ix[i,target_c]=df0.ix[i,c]
            print 'Data with split recollected'
    else:
        df0=pd.DataFrame()
        
        df0=Quandl.get(['WIKI/'+ticker],start_date=commons.min_date,end_date=commons.idx_today)
        columns=list()
        for x in df0.columns:
            x=str(x).replace(' - ', '_').strip('WIKI').strip('/')
            columns.append(x)
        df0.columns=columns
        for i in df0.index:
            for c in df0.columns:
                if 'Adj.' in c:
                    target_c=c.replace('Adj. ','')                        
                    data_sp500.ix[i,target_c]=df0.ix[i,c]
        print 'Ticker: ', ticker,' loaded.'
        #update storage    
    data_sp500=data_sp500.sort_index()
    data_sp500=data_sp500.fillna(method='ffill')
    data_sp500=data_sp500.fillna(method='backfill')
    data_sp500.to_hdf(commons.data_path+'WIKI_SP500.h5','table',mode='w')
    print 'sp500 data refreshed'


def updateDateIndex():
    global date_index_internal, date_index_external
    global data_sp500
    
    #maxI=max(date_index_external)
    maxI=0
    
    for d in data_sp500.index.sort_values():
        try:
            a=date_index_internal[d]
        except KeyError:
            maxI+=1
            date_index_internal[d]=maxI
            date_index_external[maxI]=d

    with open(commons.data_path+'dix.csv','w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        for i,d in date_index_external.items():
            csvwriter.writerow([i,d])
    csvfile.close()       

#    try:
#        with open(commons.data_path+'dix.csv','r') as csvfile:
#            csvreader = csv.reader(csvfile, delimiter=',')
#            for row in csvreader:
#                date_index_internal[dt.datetime.strptime(row[1],'%Y-%m-%d %H:%M:%S')]=int(row[0])
#                date_index_external[int(row[0])]=dt.datetime.strptime(row[1],'%Y-%m-%d %H:%M:%S')
#        csvfile.close()     
#    except IOError:
#        a=1
    

def loadSp500Changes():
    global date_index_internal, date_index_external
    global sp500Changes
    
    with open(commons.backup_path+'sp500_changes.csv','r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            ticker=cleanTicker(row[0])
            if ticker!=False:
                if cleanTicker(row[0])!=False:
                    sp500Changes.row['ticker']=cleanTicker(row[0])
                    try:
                        sp500Changes.row['dix']=date_index_internal[dt.datetime.strptime(row[2], '%m/%d/%Y')]
                    except KeyError:
                        sp500Changes.row['dix']=date_index_internal[dt.datetime.strptime(row[2], '%m/%d/%Y')+dt.timedelta(days=1)]
                    sp500Changes.row['sector']=commons.sp500_index[row[1]][-8:]
                    sp500Changes.row['change']=row[3]
                    sp500Changes.row.append()
                    sp500Changes.flush()
    csvfile.close()

def updateSp500Matrix():
    global date_index_internal, date_index_external
    global regularItemsClean
    global changeItemsClean
    global sp500CompMatrix
    global data_sp500
    items=dict(regularItemsClean,**changeItemsClean)

    sp500CompMatrix=commons.read_dataframe(commons.data_path+'SP500_COMP.h5')
    maxIndex=max(sp500CompMatrix.index)
    for ticker,dates in items.items():
        if dates['startdate']>maxIndex:
            dates['startdate']=maxIndex
        if dates['enddate']>max(data_sp500.index):
            dates['enddate']=max(data_sp500.index)
        for dix in range(date_index_internal[dates['startdate']],date_index_internal[dates['enddate']]+1):
                sp500CompMatrix.ix[date_index_external[dix],ticker]=1
    sp500CompMatrix=sp500CompMatrix.fillna(0)
    sp500CompMatrix.to_hdf(commons.data_path+'SP500_COMP.h5','table',mode='w')  
    
    for sector,index in commons.sp500_index.items():
        index_t=index[-8:]
        sp500IndexMatrix=commons.read_dataframe(commons.data_path+'HIST_'+index_t+'.h5')
        maxIndex=max(sp500IndexMatrix.index)
        for ticker,dates in items.items():
            if dates['sector']==index_t:
                if dates['startdate']>maxIndex:
                    dates['startdate']=maxIndex
                if dates['enddate']>max(data_sp500.index):
                    dates['enddate']=max(data_sp500.index)
                for dix in range(date_index_internal[dates['startdate']],date_index_internal[dates['enddate']]+1):
                        sp500IndexMatrix.ix[date_index_external[dix],ticker]=1
        sp500IndexMatrix=sp500IndexMatrix.fillna(0)
        sp500IndexMatrix.to_hdf(commons.data_path+'HIST_'+index_t+'.h5','table',mode='w')          
        
#only needed 1x at the introduction of the timeseries for index composition        
def loadHistoricalSp500(self):
    global date_index_internal, date_index_external
    data_sp500=commons.read_dataframe(commons.data_path+'WIKI_SP500.h5')
    dropped=list()
    with open(commons.local_path+'backup/SP500TickerDropped.csv','r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            dropped.append(row[0])
    csvfile.close()       
    
    columns=list()
    
    max_date=max(data_sp500.index)
    l_i=0
    for t in dropped:
        items=list([])
        exampleColumn=t+'_Open'
        if exampleColumn not in data_sp500.columns:
            items.append('WIKI/'+t)
            
        if any(items):
            df=pd.DataFrame([])
            df=Quandl.get(items,start_date=commons.min_date,end_date=dt.date.today())
            columns=list([])
            for x in df.columns:
                x=str(x).replace(' - ', '_').strip('WIKI').strip('/')
                columns.append(x)
            df.columns=columns
        
            for i in df.index:
                for c in df.columns:
                    if 'Adj.' in c:
                        target_c=c.replace('Adj. ','')
                        data_sp500.ix[i,target_c]=df.ix[i,c]
            print items, 'retrieved'
            #update storage    

            if l_i==50:
                data_sp500=data_sp500.sort_index()
                data_sp500.to_hdf(commons.data_path+'WIKI_SP500.h5','table',mode='w')
                l_i=0
            else:
                l_i+=1
    print 'sp500 data refreshed'

def initSp500Matrix(writeFile=True):
    #set for those that were in the index all the time
    global sp500_ticker
    global date_index_internal, date_index_external
    fDict=dict()
    pList=list()
    for ticker, sector in sp500_ticker.items():
        changes=sp500Changes.read_where('(ticker=='+"'"+ticker+"')")
        if not any(changes):
            dates=dict()
            pList=list()
            dates['startdate']=commons.min_date
            dates['enddate']=max(data_sp500.index)
            dates['sector']=commons.sp500_index[sector][-8:]
            pList.append(dates)
            fDict[ticker]=pList
    del ticker, sector
    
    #and those with changes
    for row in sp500Changes.read_sorted('dix'):
        if row['change']=='new':
            dates=dict()
            dates['startdate']=date_index_external[row['dix']]
            dates['enddate']=max(data_sp500.index)
            dates['sector']=row['sector']
            try:
                fDict[row['ticker']].append(dates)
            except KeyError:
                dates=dict()
                dates['startdate']=date_index_external[row['dix']]
                dates['enddate']=max(data_sp500.index)   
                dates['sector']=row['sector']
                pList=list()
                pList.append(dates)
                fDict[row['ticker']]=pList

        if row['change']=='dropped':
            try:
                fDict[row['ticker']].sort()
                for dates in fDict[row['ticker']]:
#                    print 'ticker: ',row['ticker'],' startdate ',dates['startdate'],' enddate ', date_index_external[row['dix']]
                    if dates['startdate']<date_index_external[row['dix']] and\
                        dates['enddate']>date_index_external[row['dix']]:
                        dates['enddate']=date_index_external[row['dix']]
                        break   
            except KeyError:
                dates=dict()
                dates['enddate']=date_index_external[row['dix']]
                dates['startdate']=commons.min_date
                dates['sector']=row['sector']
                if dates['startdate']<dates['enddate']:
                    pList=list([dates])
                    fDict[row['ticker']]=pList
    
    if (writeFile):
        df1=pd.DataFrame([])   
        idx=list()
        for i in range(date_index_internal[commons.min_date],date_index_internal[max(data_sp500.index)]+1):
            idx.append(date_index_external[i])
        
        dfSector=dict()
        for ticker,dates in fDict.items():
            i=1
            tickerSeries=dict()
            for date in dates:
                ts1=np.zeros(date_index_internal[date['startdate']]-date_index_internal[commons.min_date])
                ts2=np.ones(date_index_internal[date['enddate']]-date_index_internal[date['startdate']]+1)
                ts3=np.zeros(date_index_internal[max(data_sp500.index)]-date_index_internal[date['enddate']])
                if len(ts1)>0:
                    tickerSeries[i]=np.append(ts1,ts2)
                else:
                    tickerSeries[i]=ts2
                if len(ts3)>0:
                    tickerSeries[i]=np.append(tickerSeries[i],ts3)
                i+=1
                sector=date['sector']
                
            sumSeries=np.zeros(date_index_internal[max(data_sp500.index)]-date_index_internal[commons.min_date]+1)
            for i,singleSeries in tickerSeries.items():
                sumSeries+=singleSeries
            df=pd.DataFrame(data=sumSeries.reshape(-1,1),index=idx,columns=[ticker])
            df1=df1.join(df,how='outer')
            
            try:
                a=dfSector[sector]
            except KeyError:
                dfSector[sector]=pd.DataFrame()
            dfSector[sector]=dfSector[sector].join(df,how='outer')

        for k,df in dfSector.items():
            df.to_hdf(commons.data_path+'HIST_'+k+'.h5','table',mode='w')
                
        df1.to_hdf(commons.data_path+'SP500_COMP.h5','table',mode='w')

            
    with open(commons.data_path+'SP500_COMP_dates.csv','w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        for ticker,dates in fDict.items():
            for date in dates:
                if date['enddate']>max(data_sp500.index):
                   date['enddate']=max(data_sp500.index) 
                csvwriter.writerow([ticker,date_index_internal[date['startdate']],date_index_internal[date['enddate']]])
    csvfile.close()       
    
def initForeignSource():
    global date_index_internal, date_index_external
    global data_sp500
    if 'FMCC' not in data_sp500.columns or 'FNMA' not in data_sp500.columns:
        df=pd.DataFrame([])
        df=Quandl.get(['YAHOO/FMCC','YAHOO/FNMA','YAHOO/PA_NYX','YAHOO/KATE','YAHOO/PTC','YAHOO/WFT'],start_date=commons.min_date,end_date=dt.date.today())
        
        columns=list([])
        for x in df.columns:
            x=str(x).replace(' - ', '_')
            x=x.replace('YAHOO/PA_','')
            x=x.replace('YAHOO/','')
            columns.append(x)
        df.columns=columns
    
        for i in df.index:
            for c in df.columns:
                if 'Adjusted' in c:
                    data_sp500.ix[i,c[0:4]+'_Close']=df.ix[i,c]
                else:
                    data_sp500.ix[i,c]=df.ix[i,c]

    if 'EKDKQ' not in data_sp500.columns:
        df=pd.DataFrame([])
        df=Quandl.get('GOOG/PINK_EKDKQ',start_date=commons.min_date,end_date=dt.date.today())
        
        columns=list([])
        for x in df.columns:
            x=str(x).replace('GOOG/PINK_','')
            columns.append('EKDKQ_'+x)
        df.columns=columns
    
        for i in df.index:
            for c in df.columns:
                data_sp500.ix[i,c]=df.ix[i,c]
                
    data_sp500.to_hdf(commons.data_path+'WIKI_SP500.h5','table',mode='w')                

def getCloseFromSP1():
    global data_sp500
    for c in commons.read_dataframe(commons.data_path+'SP500_COMP.h5').columns:    
        cClose=c+'_Close'
        if cClose not in data_sp500.columns:
            print c            
            df=pd.DataFrame()
            df=Quandl.get('SF1/'+commons.getSP1Ticker(c)+'_PRICE',start_date=commons.min_date,end_date=commons.max_date['WIKI_SP500'])
            for i in df.index:
                data_sp500.ix[i,c+'_Close']=df.ix[i,'Value']
                data_sp500.ix[i,c+'_Open']=df.ix[i,'Value']
                data_sp500.ix[i,c+'_Low']=df.ix[i,'Value']
                data_sp500.ix[i,c+'_High']=df.ix[i,'Value']
    data_sp500=data_sp500.fillna(method='ffill')
    data_sp500=data_sp500.fillna(method='backfill')
    data_sp500.to_hdf(commons.data_path+'WIKI_SP500.h5','table',mode='w')

def loadBSC():
    global date_index_internal, date_index_external
    data_sp500=commons.read_dataframe(commons.data_path+'WIKI_SP500.h5')
    data_sp500_marketcap=commons.read_dataframe(commons.data_path+'MARKETCAP.h5')
    dix=[11628,11586,11522,11459,11395,11334,11271,11208,11145,11083,11020]
    price=[2,80,120,145,150,162,150,142,140,115,110]
    cap=[240000000, 9830000000,11880000000,13970000000,17790000000,19050000000,18060000000,16250000000,16580000000,15520000000,12510000000]
    for i in range(0,len(dix)):
        data_sp500.ix[date_index_external[dix[i]],'BSC_Open']=price[i]
        data_sp500.ix[date_index_external[dix[i]],'BSC_Close']=price[i]
        data_sp500.ix[date_index_external[dix[i]],'BSC_Low']=price[i]
        data_sp500.ix[date_index_external[dix[i]],'BSC_High']=price[i]
        data_sp500_marketcap.ix[date_index_external[dix[i]],'BSC']=cap[i]
    data_sp500=data_sp500.fillna(method='backfill')
    data_sp500_marketcap.fillna(method='ffill')
    data_sp500_marketcap.fillna(method='backfill')
    data_sp500.to_hdf(commons.data_path+'WIKI_SP500.h5','table',mode='w')
    data_sp500_marketcap.to_hdf(commons.data_path+'MARKETCAP.h5','table',mode='w')
    

date_index_internal=dict()
date_index_external=dict()
#execute
start = time.time()
regularItemsClean=dict()
changeItemsClean=dict()

loadSp500Composition()
print 'composition read'

initSp500Changes()
print 'sp500 changes initialized'

loadSp500Data()
print 'sp500 data loaded'

#initForeignSource() loads some tickers from yahoo and google. only needed 1x

getCloseFromSP1() #for some tickers there is no data in wiki, but there is closing price in SP1
print 'close from SP1 retrieved.'

#loadBSC() loads Bear Stearns. Only needed 1x
refreshWikiSp500('all') #update w/o changes
print 'sp500 data refreshed'

updateDateIndex()
print 'date index updated'

######refreshWikiSp500('EOP') #single maintenance call, if onlt 1 ticker is missing

loadSp500Changes()
print 'sp500 changes loaded' #load of changes in SP500 index

initSp500Matrix(False) #write the SP500_changes_dates.xls, to be used in various places for start/enddate calculation

initSp500Matrix() #only needed 1x in the beginning
print 'sp500 matrix initialized'

refreshWikiSp500('all',True) #update of WIKI changes
print 'sp500 index change data refreshed'

#updateSp500Matrix()
#print 'sp500 matrix updated'

end = time.time()
print 'Execution time in secs:',end - start