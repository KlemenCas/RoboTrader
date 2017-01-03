import pandas as pd
import datetime as dt
import csv
import platform

def init():
    global demo_scenario,sp500CompMatrix,refresh_data
    
    demo_scenario=True
    refresh_data=False
    if demo_scenario:
        print 'Preparing data. Limited (demo) scope! Please wait.'     
    else:
        print 'Preparing data. Please wait.'
    sp500CompMatrix=read_dataframe(data_path+'SP500_COMP.h5') 
    

def getSP1Ticker(ticker):
    if ticker=='ANR':
        return 'ANRZ'
    elif ticker=='UA_C':
        return 'UA'
    elif ticker=='BRK_B':
        return 'BRKB'
    elif ticker=='BTU':
        return 'BTUUK'
    elif ticker=='IACI':
        return 'IAC'
    elif ticker=='DTV':
        return 'DTV2'
    elif ticker=='BF_B':
        return 'BFB'        
    elif ticker=='DISCK':
        return 'DISCA'
    elif ticker=='WB':
        return 'WB1'
    elif ticker=='MIL':
        return 'MIL1'
    elif ticker=='Q':
        return 'Q1'
    elif ticker=='AT':
        return 'AT1'
    else:
        return ticker

def getWIKITicker(ticker):
    if ticker=='ANRZ':
        return 'ANR'
    elif ticker=='UA':
        return 'UA_C'
    elif ticker=='BRKB':
        return 'BRK_B'
    elif ticker=='BTUUK':
        return 'BTU'
    elif ticker=='IAC':
        return 'IACI'
    elif ticker=='DTV2':
        return 'DTV'
    elif ticker=='BFB':
        return 'BF_B'        
    elif ticker=='DISCA':
        return 'DISCC'
    elif ticker=='WB1':
        return 'WB'
    elif ticker=='Q1':
        return 'Q'
    elif ticker=='AT1':
        return 'AT'
    else:
        return ticker    

def setPath():
    global data_path,local_path,stats_path,model_path,backup_path,analytics_path
    if platform.node()=='DESKTOP-5HHG5ET':
        backup_path='C:/Users/kleme/OneDrive/HF_Trading/RoboTrader/backup/'
        data_path  ='C:/Users/kleme/OneDrive/HF_Trading/RoboTrader/data/'
        model_path ='C:/Users/kleme/OneDrive/HF_Trading/RoboTrader/model/'
        stats_path ='C:/Users/kleme/OneDrive/HF_Trading/RoboTrader/stats/'
        daily_path ='C:/Users/kleme/OneDrive/HF_Trading/RoboTrader/daily/'
        analytics_path ='C:/Users/kleme/Google Drive/'
    elif 'Klemens' in platform.node():
        backup_path='/Users/kncas/OneDrive/HF_Trading/RoboTrader/backup/'
        data_path  ='/Users/kncas/OneDrive/HF_Trading/RoboTrader/data/'
        model_path ='/Users/kncas/OneDrive/HF_Trading/RoboTrader/model/'
        stats_path ='/Users/kncas/OneDrive/HF_Trading/RoboTrader/stats/'
        daily_path ='/Users/kncas/OneDrive/HF_Trading/RoboTrader/daily/'
        analytics_path='/Users/kncas/Google Drive/mac/'
    local_path='./'
    
#read datafrom from disc
def read_dataframe(file):
    try:
        return pd.read_hdf(file,'table')
    except IOError:
        return pd.DataFrame()
        
#init index
def initCodes():
    global sp500_index,alternative_symbol,y_labels,y_dd_labels,y_chr_clr_labels,action_code,kpi,qkc
    
    sp500_index['Telecommunications Services']='GOOG/NYSEARCA_VOX'
    sp500_index['Consumer Discretionary']='GOOG/NYSEARCA_VCR'
    sp500_index['Consumer Staples']='GOOG/NYSEARCA_VDC'
    sp500_index['Energy']='GOOG/NYSEARCA_VDE'
    sp500_index['Financials']='GOOG/NYSEARCA_VFH'
    sp500_index['Health Care']='GOOG/NYSEARCA_VHT'
    sp500_index['Industrials']='GOOG/NYSEARCA_VIS'
    sp500_index['Information Technology']='GOOG/NYSEARCA_VGT'
    sp500_index['Materials']='GOOG/NYSEARCA_VAW'
    sp500_index['Utilities']='GOOG/NYSEARCA_VPU'
    sp500_index['Real Estate']='GOOG/NYSEARCA_RWR'

    #alternative symbols, due to mismatch between the quandl databases
    alternative_symbol=dict()
    alternative_symbol['BF_B']='BFB'
    alternative_symbol['BFB']='BF_B'
    alternative_symbol['DISCK']='DISCA'
    alternative_symbol['DISCA']='DISCK'
    alternative_symbol['BRK_B']='BRKB'
    alternative_symbol['BRKB']='BRK_B'

    #y labels
    y_labels=list(['1dd_Close','5dd_Close','20dd_Close','12dd_Close','clr_cluster_0','clr_cluster_1','clr_cluster_2','clr_cluster_3',\
                   'clr_cluster_4','chr_cluster_0','chr_cluster_1','chr_cluster_2','chr_cluster_3','chr_cluster_4'])
    y_dd_labels=list(['1dd_Close','5dd_Close','20dd_Close','12dd_Close'])
    y_chr_clr_labels=list(['clr_cluster_0','clr_cluster_1','clr_cluster_2','clr_cluster_3',\
                           'clr_cluster_4','chr_cluster_0','chr_cluster_1','chr_cluster_2','chr_cluster_3','chr_cluster_4'])

    #action codes for trading
    action_code=dict()
    action_code['buy']=1
    action_code['sell']=-1
    action_code['hold']=0
    kpi=dict()
    kpi['clr']=0
    kpi['chr']=1

    #codes for q_key
    qkc=dict()
    qkc['1dd_Close']=1
    qkc['5dd_Close']=2
    qkc['20dd_Close']=3
    qkc['clr_cluster_0']=4
    qkc['clr_cluster_1']=5
    qkc['clr_cluster_2']=6
    qkc['clr_cluster_3']=7
    qkc['clr_cluster_4']=8
    qkc['chr_cluster_0']=9
    qkc['chr_cluster_1']=10
    qkc['chr_cluster_2']=11
    qkc['chr_cluster_3']=12
    qkc['chr_cluster_4']=13
    qkc['12dd_Close']=14
       
def loadSp500Composition():
    global currentSp500Constituents,demo_scenario
    
    #load sp500 composition    
    currentSp500Constituents=dict()  
    with open(data_path+'WIKI.csv','r') as csvfile:
        csvreader=csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            if demo_scenario==False or row[1]=='Telecommunications Services':
                currentSp500Constituents[row[0]]=row[1]
    csvfile.close()


def initDates():
    global anb_min_date,data_sp500_1st_date,date_index_internal,date_index_external,idx_today,max_date,min_date,data_path
    
    #load dix
    with open(data_path+'dix.csv','r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            date_index_internal[dt.datetime.strptime(row[1],'%Y-%m-%d %H:%M:%S')]=int(row[0])
            date_index_external[int(row[0])]=dt.datetime.strptime(row[1],'%Y-%m-%d %H:%M:%S')
    csvfile.close()        
    
    #load 1st dates - from when onwards are stock prices available on Quandl
    try:
        with open(data_path+'sp500_1st_date.csv','r') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            for row in csvreader:
                data_sp500_1st_date[row[0]]=dt.datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S')
            csvfile.close()            
    except IOError:
        print 'no 1st dates available'         

    #set max dates        
    idx_today=dt.datetime.strptime(str(dt.date.today())+' 00:00:00','%Y-%m-%d %H:%M:%S')
    source=list(['FUND_SP500','SHORT_SP500','SENT_SP500','MARKETCAP','WIKI_SP500','ANB','SECTOR_SP500'])
    for s in source:
        try:
            a=pd.read_hdf(data_path+''+s+'.h5','table')
            max_date[s]=max(a.index)
        except IOError:
            if s!='SECTOR_SP500':
                max_date[s]=dt.datetime.strptime('01/03/2006','%m/%d/%Y')
            else:
                max_date[s]=getClosestDate(dt.datetime.strptime('01/02/2005','%m/%d/%Y'))
    min_date=dt.datetime.strptime('01/03/2006','%m/%d/%Y')
    anb_min_date=getClosestDate(dt.datetime.strptime('01/02/2005','%m/%d/%Y'))
        
#assemble columns for forecasting
def Xy_columns(Xy_all,mode,feature_only=False):
    modes=list(['Open','Close','Low','High'])
    modes.remove(mode) 
    
    select_columns=list([])            
    for c in Xy_all.columns:
        if mode in str(c):
            select_columns.append(c)
        else:
            m_found=False
            for m in modes:
                if m in str(c):
                    m_found=True
            if m_found==False:
                select_columns.append(c)
    if feature_only==True:
        select_columns.remove('_chr')
        select_columns.remove('_clr')
        select_columns.remove('1dr_Close')
        select_columns.remove('5dr_Close')
        select_columns.remove('20dr_Close')        
        for c in y_labels:
            try:
                select_columns.remove(c)
            except ValueError:
                print 'not in the list:', c
        
    return select_columns
    
def getSectorForDropped(ticker,date):
    global local_path
    global data_path
    global date_index_internal
    global sp500SectorAssignments
    
    i=0
    for index_t,tickers in sp500SectorAssignments.items():
       if ticker in tickers:
           i+=1
           retSector=index_t
    if i>1:
        for index_t,tickers in sp500SectorAssignments.items():
            sp500IndexMatrix=read_dataframe(data_path+'HIST_'+index_t+'.h5')
            if sp500IndexMatrix.ix[date,ticker]==1:
                retSector=index_t
    else:
        return retSector  

    
def getHistSp500Ticker(date):
    global sp500CompMatrix
    global demo_scenario
    global sp500_index
    global currentSp500Constituents
    
    if demo_scenario:
        matrix=read_dataframe(data_path+'HIST_ARCA_VOX.h5').ix[date,:] #Telco
    else:
        matrix=sp500CompMatrix.ix[date,:]
    ticker=dict()
    for t in matrix[matrix!=0].index:
        try:
            ticker[t]=sp500_index[currentSp500Constituents[t]][-8:]
        except KeyError:
            ticker[t]=getSectorForDropped(t,date)
    return ticker
    
def getHistSp500TickerList(startdix,enddix,limited=True):
    global min_date,demo_scenario
    retList=list()
    if limited:
        if enddix==date_index_internal[max_date['WIKI_SP500']] and startdix==date_index_internal[min_date]:
            if demo_scenario:
                for c in read_dataframe(data_path+'HIST_ARCA_VOX.h5').columns:
                    retList.append(c)
            else:
                for c in read_dataframe(data_path+'SP500_COMP.h5').columns:
                    retList.append(c)
        else:
            for dix in range(startdix,enddix+1):
                tDict=dict()
                tDict=getHistSp500Ticker(date_index_external[dix])
                for t,sector in tDict.items():
                    if t not in retList:
                        retList.append(t)
    else:
        if demo_scenario:
            for c in read_dataframe(data_path+'HIST_ARCA_VOX.h5').columns:
                retList.append(c)
        else:
            for c in read_dataframe(data_path+'SP500_COMP.h5').columns:
                retList.append(c)
    return retList
    
def getHistSp500Composition(date):
    ticker=getHistSp500Ticker(date)
    comp=dict()
    for k,v in ticker.items():
        try:
            comp[v].append(k)
        except KeyError:
            comp[v]=list([k])
    return comp 

def loadHistSp500Index():
    global sp500_index,sp500SectorAssignments,data_path,sp500SectorAssignmentsTicker,demo_scenario
    
    if demo_scenario:
        sp500IndexMatrix=read_dataframe(data_path+'HIST_ARCA_VOX.h5')
        sp500SectorAssignments['ARCA_VOX']=sp500IndexMatrix.columns
        for c in sp500IndexMatrix.columns:
            sp500SectorAssignmentsTicker[c]='ARCA_VOX'
    else:
        for k,index in sp500_index.items():
            index_t=index[-8:]
            sp500IndexMatrix=read_dataframe(data_path+'HIST_'+index_t+'.h5')
            sp500SectorAssignments[index_t]=sp500IndexMatrix.columns
            for c in sp500IndexMatrix.columns:
                sp500SectorAssignmentsTicker[c]=index_t
    del sp500IndexMatrix
        
def getClosestDate(date):
    try:
        dix=date_index_internal[date]
        retDate=date
    except KeyError:
        retDate=getClosestDate(date+dt.timedelta(days=-1))
    return retDate

def getSp500CompositionAll():
    global sp500_index,data_path,demo_scenario
    comp=dict()
    if demo_scenario:
        a=read_dataframe(data_path+'HIST_ARCA_VOX.h5')
        comp['Telecommunications Services']=a.columns
    else:        
        for index,index_t in commons.sp500_index.items():
            a=read_dataframe(data_path+'HIST_'+index_t[-8:]+'.h5')
            comp[index]=a.columns
    return comp

def memberInDemoIndex(ticker):
    return (ticker in read_dataframe(data_path+'HIST_ARCA_VOX.h5').columns)
    
def loadSp500CompDates():
    global sp500CompDates,date_index_external,demo_scenario
    
    with open(data_path+'SP500_COMP_dates.csv','r') as csvfile:
        csvreader=csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            try:
                sp500CompDates[row[0]].append((date_index_external[int(row[1])],date_index_external[int(row[2])]))
            except KeyError:
                sp500CompDates[row[0]]=list()
                sp500CompDates[row[0]].append((date_index_external[int(row[1])],date_index_external[int(row[2])]))
    csvfile.close()
    
    if demo_scenario:
        for ticker,dates in sp500CompDates.items():
            if not memberInDemoIndex(ticker):
                del sp500CompDates[ticker]
                
def getIndexCodes():
    global demo_scenario,sp500_index
    retDict=dict()
    if demo_scenario:
        retDict['Telecommunications Services']='GOOG/NYSEARCA_VOX'
    else:
        retDict=sp500_index
    return retDict

def getNextTradeDay(date):
    weekday=date.weekday()
    if weekday==5:
        return date+dt.timedelta(days=2)
    elif weekday==4:
        return date+dt.timedelta(days=3)
    else:
        return date+dt.timedelta(days=1)


sp500CompDates=dict()    
date_index_internal=dict()
date_index_external=dict()
data_sp500_1st_date=dict()
max_date=dict()
sp500_index=dict()
sp500SectorAssignments=dict()
sp500SectorAssignmentsTicker=dict()
idx_today=None

setPath()    
init()
initCodes()
loadSp500Composition()
initDates()
loadHistSp500Index()
loadSp500CompDates()
