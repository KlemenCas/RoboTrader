import commons
import pandas as pd
import numpy as np
import quandl as Quandl
import csv
import datetime as dt
from sklearn.preprocessing import MinMaxScaler

class mydata(object):   
    Quandl.ApiConfig.api_key=''      
    df1st_date={}
    last_date=commons.max_date['WIKI_SP500']
    reload_baseline=False
    refresh_data=True
    demo_scenario=True
    df12dd=pd.DataFrame()
    df1dr=pd.DataFrame()
    df5dr=pd.DataFrame()
    df20dr=pd.DataFrame()
    df5dm=pd.DataFrame()
    dfSectorAnB=pd.DataFrame()
    df30dsma=pd.DataFrame()
    df30dmx=pd.DataFrame()
    df30dmn=pd.DataFrame()    
    dfanb=pd.DataFrame()
    dfBbands=pd.DataFrame()
    dfclr=pd.DataFrame()
    dfchr=pd.DataFrame()
    dfny_ss=pd.DataFrame()
    dfns_ss=pd.DataFrame()
    df1er=pd.DataFrame()
    df2er=pd.DataFrame()
    df5er=pd.DataFrame()   
    dfSectorSliced=dict()
    anbDict=dict()

        
        
    def __init__(self,refresh_data=False,reload_baseline=False,demo_scenario=True,quandlkey=''):    
        Quandl.ApiConfig.api_key=quandlkey
        self.demo_scenario=demo_scenario
        self.refresh_data=refresh_data
        self.df1st_date=commons.data_sp500_1st_date
        self.dfWikiSp500=commons.read_dataframe(commons.data_path+'WIKI_SP500.h5')
        self.dfSector=commons.read_dataframe(commons.data_path+'SECTOR_SP500.h5')
        self.dfSentiment=commons.read_dataframe(commons.data_path+'SENT_SP500.h5')
        self.dfFundamentals=commons.read_dataframe(commons.data_path+'FUND_SP500.h5')    
        self.dfShortSell=commons.read_dataframe(commons.data_path+'SHORT_SP500.h5')
        self.dfMarketcap=commons.read_dataframe(commons.data_path+'MARKETCAP.h5')  
        self.dfanb=commons.read_dataframe(commons.data_path+'anb.h5')  
        self.dfLastCalloff=commons.read_dataframe(commons.data_path+'last_calloff.h5')
        #select columns with low, high, open and close
        if commons.demo_scenario:
            column_selection=list()
            for ticker in commons.getHistSp500TickerList(1,1,False):
                column_selection.append(ticker+'_Open')
                column_selection.append(ticker+'_Close')
                column_selection.append(ticker+'_Low')
                column_selection.append(ticker+'_High')
        else:
            column_selection=list([])
            for c in self.dfWikiSp500.columns:
                if c[-4:] in ['Open', 'lose', 'High', '_Low']:
                    column_selection.append(c)
        self.dfPrices=self.dfWikiSp500.ix[:,column_selection]
         
#this is the initial update from the bulk download from quandl. needs to be done only 1x        
    def process_quandl_csv(self):
        #upload and process local csv
        wiki=pd.read_csv(commons.data_path+'WIKI.csv',header=None, index_col=['Ticker','Date'],\
                         names=['Ticker','Date','Open','High','Low','Close','Volume','Ex-Dividend','Split Ratio',\
                         'Adj. Open','Adj. High','Adj. Low','Adj. Close', 'Adj. Volume'], usecols=['Ticker','Date',\
                         'Ex-Dividend','Split Ratio','Adj. Open','Adj. High','Adj. Low','Adj. Close', 'Adj. Volume'], parse_dates=[1])
        wiki.to_hdf(commons.data_path+'WIKI.h5','table',mode='w')
        print 'hdf5 created from WIKI csv'
        #extract sp500 data
        wiki_sp500=pd.DataFrame()
        for k,v in commons.sp500SectorAssignmentsTicker.items():
            values=wiki.ix[k].values
            index=wiki.ix[k].index
            columns=str(k)+'_'+wiki.ix[k].columns
            df=pd.DataFrame(data=values,index=index,columns=columns)
            wiki_sp500=wiki_sp500.join(df,how='outer')
            self.df1st_date[k]=min(wiki.ix[k].index)
        wiki_sp500.dropna(axis=0,how='all',inplace=True)
        wiki_sp500.to_hdf(commons.data_path+'WIKI_SP500.h5','table',mode='w')                    
        print 'WIKI based SP500 ticker data stored'



#load index prices
    def getIndexData(self):
        self.dfSector=commons.read_dataframe(commons.data_path+'SECTOR_SP500.h5')   

        if self.refresh_data==True:
            enddate=commons.max_date['WIKI_SP500']
            startdate=commons.max_date['SECTOR_SP500']

            for k,v in commons.sp500_index.items():
                df=pd.DataFrame()
                df=Quandl.get(v,start_date=startdate,end_date=enddate)
                df.columns=[str(v[-8:])+'_Open',str(v[-8:])+'_High',str(v[-8:])+'_Low',str(v[-8:])+'_Close',str(v[-8:])+'_Volume']
                for c in df.columns:
                    if 'Volume' not in c:
                        if c not in self.dfSector.columns:
                            self.dfSector=self.dfSector.join(getattr(df,c),how='outer')
                        else:
                            for i in df.index:
                                self.dfSector.ix[i,c]=df.ix[i,c]
            
            self.dfSector=self.processResults(df,self.dfSector,'SECTOR_SP500')
            print 'Index prices retrieved and stored'
        else:
            print 'Local index data loaded.'



#load fundamentals            
    def getFundamentals(self):
        if self.refresh_data==True:
            startdate=commons.max_date['FUND_SP500']
            enddate=commons.max_date['WIKI_SP500']

            fundamentals=list(['_PB_ARQ','_EPSDIL_ARQ'])
            items=list([])
            for fund in fundamentals:
                for ticker in commons.getHistSp500TickerList(commons.date_index_internal[commons.max_date['FUND_SP500']],commons.date_index_internal[commons.max_date['WIKI_SP500']]):
                    items.append('SF1/'+str(ticker)+str(fund))

            for i in range(0,len(items)):
                if i%100==5 or i==(len(items)-1):
                    if i==5:
                        itemList=items[0:i]
                    else:
                        itemList=items[i-100:i]
            
                    df=pd.DataFrame([])
                    df=Quandl.get(itemList,start_date=startdate,end_date=enddate)
                
                    columns=list([])
                    for x in df.columns:
                        x=str(x).replace(' - ', '_')
                        x=x.replace('SF1/','')
                        x=x.replace('_Value','')
                        columns.append(x)
                    df.columns=columns
                
                    self.dfFundamentals=self.processResults(df,self.dfFundamentals,'FUND_SP500','ffill')                

            self.dfLastCalloff.ix[dt.datetime.today(),'fundamentals']=1
            self.dfLastCalloff.to_hdf(commons.data_path+'last_calloff.h5','table',mode='w')             
            print 'Fundamentals data refreshed'


#load short selling    
    def getShortSell(self):
        if self.refresh_data==True:        
            startdate=commons.max_date['SHORT_SP500']
            enddate=commons.max_date['WIKI_SP500']

            short_sell=list(['FINRA/FNSQ_','FINRA/FNYX_'])
            for s in short_sell:
                items=list()
                for ticker in commons.getHistSp500TickerList(commons.date_index_internal[startdate],commons.date_index_internal[enddate]):
                    items.append(s+ticker)
            
                df=pd.DataFrame()
                df=Quandl.get(items,start_date=startdate,end_date=enddate)
                columns=list([])
                for x in df.columns:
                    x=str(x).replace(' - ', '_')
                    x=x.replace('FINRA/','')
                    columns.append(x)
                df.columns=columns

                self.dfShortSell=self.processResults(df,self.dfShortSell,'SHORT_SP500')
            print 'Short sell data refreshed'

#load sentiment                
    def getSentiment(self):
        if self.refresh_data==True:                  
            startdate=commons.max_date['SENT_SP500']
            enddate=commons.max_date['WIKI_SP500']
                
            df=pd.DataFrame([])
            df=Quandl.get('AAII/AAII_SENTIMENT',start_date=startdate,end_date=enddate)
            self.dfSentiment=self.processResults(df,self.dfSentiment,'SENT_SP500')
            print 'Sentiment data refreshed'


#this is not really needed as we have no historical records to index composition. If we had them,
#this method would deliver the deltas            


    def logSp500Changes(self):
        maxDate=max(self.dfWikiSp500.index)
        with open(commons.local_path+'backup/sp500_changes.csv','r') as csvfile:
            csvreader=csv.reader(csvfile, delimiter=',')
            i=0
            dChanges=dict()
            for ticker,change in self.sp500_change.items():
                newChange=True
                for row in csvreader:
                    if row[0]==ticker and row[1]==maxDate and row[2]==change:
                        newChange=False
                if newChange:
                    xrow=list()
                    xrow.append(ticker)
                    xrow.append(maxDate)
                    xrow.append(change)
                    dChanges[i]=xrow
                    i+=1
        csvfile.close()
                
        with open(commons.local_path+'backup/sp500_changes.csv','r+') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            for i,row in dChanges.items():
                csvwriter.writerow(row)
        csvfile.close()
        print 'SP500 changes recorded.'        

#NaN treatment
    def sp500fillna(self):
        self.dfWikiSp500=self.dfWikiSp500.sort_index()
        self.dfFundamentals=self.dfFundamentals.sort_index()
        self.dfShortSell=self.dfShortSell.sort_index()
        self.dfSentiment=self.dfSentiment.sort_index()
        self.dfSector=self.dfSector.sort_index()
        self.dfLastCalloff=self.dfLastCalloff.sort_index()
        self.dfanb=self.dfanb.sort_index()

        self.dfWikiSp500=self.fillUpIndex(self.dfWikiSp500)
        self.dfFundamentals=self.fillUpIndex(self.dfFundamentals)
        self.dfShortSell=self.fillUpIndex(self.dfShortSell)
        self.dfSentiment=self.fillUpIndex(self.dfSentiment)
        self.dfSector=self.fillUpIndex(self.dfSector)
        self.dfLastCalloff=self.fillUpIndex(self.dfLastCalloff)   
        self.dfanb=self.fillUpIndex(self.dfanb)
        
        print 'FillNa on source data performed.'
        

#calculate features and labels
    def calcIndicators(self):     
        self.dfFundamentals=self.checkDemoData('dfFundamentals')
        self.dfShortSell=self.checkDemoData('dfShortSell')
        self.dfSentiment=self.checkDemoData('dfSentiment')
        self.dfSector=self.checkDemoData('dfSector')
        self.dfanb=self.checkDemoData('dfanb')
        
        #momentum
        self.df5dm=(self.dfPrices-self.dfPrices.shift(5))/self.dfPrices.shift(5)
        self.df2dm=(self.dfPrices-self.dfPrices.shift(2))/self.dfPrices.shift(2)
        self.df1dm=(self.dfPrices-self.dfPrices.shift(1))/self.dfPrices.shift(1)
        print 'Momentum calculated.'
        
        #delta to expected return
        for sector,tickerList in commons.getSp500CompositionAll().items():
            index_t=commons.sp500_index[sector][-8:]
            self.df1er=self.df1dm-((self.dfSectorSliced[index_t]-self.dfSectorSliced[index_t].shift(1))/self.dfSectorSliced[index_t].shift(1)*self.anbDict['B_']+self.anbDict['A_'])
            self.df2er=self.df2dm-((self.dfSectorSliced[index_t]-self.dfSectorSliced[index_t].shift(2))/self.dfSectorSliced[index_t].shift(2)*self.anbDict['B_']+self.anbDict['A_'])
            self.df5er=self.df5dm-((self.dfSectorSliced[index_t]-self.dfSectorSliced[index_t].shift(5))/self.dfSectorSliced[index_t].shift(5)*self.anbDict['B_']+self.anbDict['A_'])
        print 'Delta to expected return.'
        
        #sma 30 days
        self.df30dsma=self.dfPrices/pd.DataFrame.rolling(self.dfPrices,30).mean()
        #comp to max and min 30 days
        self.df30dmx=self.dfPrices/pd.DataFrame.rolling(self.dfPrices,30).max()
        self.df30dmn=self.dfPrices/pd.DataFrame.rolling(self.dfPrices,30).min()
        #vola week
        self.df5dv=1-pd.DataFrame.rolling(self.dfPrices,30).min()/pd.DataFrame.rolling(self.dfPrices,30).max()
        #bollinger bands
        stock_rm_df=pd.DataFrame.rolling(self.dfPrices,200).mean()
        self.dfbbands=(self.dfPrices-stock_rm_df)/(2*self.dfPrices.std(axis=0))     
        print 'min, max, sma, vola and bbbands calculated.'
        #returns for labels        
        self.df1dr=(self.dfPrices.shift(-1)/self.dfPrices-1).round(2)*100
        self.df5dr=(self.dfPrices.shift(-5)/self.dfPrices-1).round(2)*100
        self.df20dr=(self.dfPrices.shift(-20)/self.dfPrices-1).round(2)*100
        #directional labels
        self.df1dd=(self.dfPrices.shift(-1)/self.dfPrices-1)*100
        self.df12dd=(self.dfPrices.shift(-2)/self.dfPrices.shift(-1)-1)*100
        self.df5dd=(self.dfPrices.shift(-5)/self.dfPrices-1)*100
        self.df20dd=(self.dfPrices.shift(-20)/self.dfPrices-1)*100
        
        #close to low and close to high
        for k,v in commons.sp500SectorAssignmentsTicker.items():        
            #close to low and close to high
            df1=pd.DataFrame(self.dfPrices.ix[:,str(k)+'_Low'].shift(-1)/self.dfPrices.ix[:,str(k)+'_Close']-1).round(2)*100
            df1.columns=list([str(k)+'_clr'])
            self.dfclr=self.dfclr.join(df1,how='outer')
            df1=pd.DataFrame(self.dfPrices.ix[:,str(k)+'_High'].shift(-1)/self.dfPrices.ix[:,str(k)+'_Close']-1).round(2)*100
            df1.columns=list([str(k)+'_chr'])
            self.dfchr=self.dfchr.join(df1,how='outer')
            
            #short %
            try:
                df1=pd.DataFrame(self.dfShortSell.ix[:,'FNSQ_'+str(k)+'_ShortVolume']/self.dfShortSell.ix[:,'FNSQ_'+str(k)+'_TotalVolume'])*10
                df1.columns=list([str(k)+'_ns_ss'])
                self.dfns_ss=self.dfns_ss.join(df1,how='outer')
                df1=pd.DataFrame(self.dfShortSell.ix[:,'FNYX_'+str(k)+'_ShortVolume']/self.dfShortSell.ix[:,'FNYX_'+str(k)+'_TotalVolume'])*10
                df1.columns=list([str(k)+'_ny_ss'])
                self.dfny_ss=self.dfny_ss.join(df1,how='outer')
            except KeyError:
                print 'Short Sell data for: ',k,' missing.'
                df1=pd.DataFrame(columns=[str(k)+'_ns_ss'])
                self.dfns_ss=self.dfns_ss.join(df1,how='outer')
                df1=pd.DataFrame(columns=[str(k)+'_ny_ss'])
                self.dfny_ss=self.dfns_ss.join(df1,how='outer')
                
            

        print 'Labels calculated.'
        #alpha & beta

        #drop outliers; top and bottom 1%    
        a=list(['1dm','2dm','5dm','30dsma','30dmx','30dmn','5dv','bbands','clr','chr','ny_ss','ns_ss','1er','2er','5er'])
        for x in a:
            setattr(self,'df'+str(x),self.drop_outliers(getattr(self,'df'+str(x))))
                
        
        #fill, minmax and direction
        a=list(['1dm','2dm','5dm','30dsma','30dmx','30dmn','5dv','bbands','1dd','12dd','5dd','20dd','clr','chr','ny_ss','ns_ss','1er','2er','5er'])
        for x in a:
            setattr(self,'df'+str(x),self.fillUpIndex(getattr(self,'df'+str(x))))
            
        a=list(['1dm','2dm','5dm','30dsma','30dmx','30dmn','5dv','bbands'])
        for x in a:
            try:
                setattr(self,'df'+str(x),self.minmaxscale('df'+str(x)))
            except ValueError:
                print 'MinMax value error in df'+str(x)
                getattr(self,'df'+str(x)).to_hdf(commons.data_path+'df'+str(x)+'.h5','table',mode='w')
                

        a=list(['1dd','5dd','20dd','12dd'])
        for x in a: 
            setattr(self,'df'+str(x),self.p_direction(getattr(self,'df'+str(x))))
            setattr(self,'df'+str(x),self.n_direction(getattr(self,'df'+str(x))))
            
        l_i=505
        for k,v in commons.sp500SectorAssignmentsTicker.items():     
            Xy_all=self.assemble_xy(k)
            Xy_all.to_hdf(commons.data_path+'Xy_all_'+str(k),'table',mode='w')
            l_i-=1
            print 'Xy_all to '+str(k)+' assembled. '+str(l_i)+' to go.'
            
        #self.drop_obsolete_anb()#only needed 1x due to obsolete index.

        
    def sliceIndex(self):
        iList=list(['_Open','_Close','_Low','_High'])
        for sector,tickerList in commons.getSp500CompositionAll().items():
            index_t=commons.sp500_index[sector][-8:]
            df=pd.DataFrame()
            cList=list()
            for t in tickerList:
                for i in iList:
                    df.ix[:,t+i]=self.dfSector.ix[:,index_t+i]
            self.dfSectorSliced[index_t]=df
        
#individual alpha and beta, compared to the industry
    def calcRollingSectorBeta(self,startdate,ticker):
        stock_column=list([ticker+'_Open',ticker+'_High',ticker+'_Low',ticker+'_Close'])
        stock_df=self.dfPrices.ix[commons.anb_min_date:,stock_column]
        stock_df.columns=list(['Open','High','Low','Close'])
        sector_column=list([commons.sp500SectorAssignmentsTicker[ticker]+'_Open',commons.sp500SectorAssignmentsTicker[ticker]+'_High',commons.sp500SectorAssignmentsTicker[ticker]+'_Low',commons.sp500SectorAssignmentsTicker[ticker]+'_Close'])
        sector_df=self.dfSector.ix[commons.anb_min_date:,sector_column]
        sector_df.columns=list(['Open','High','Low','Close'])
        #resample
        dfsm=pd.DataFrame()
        for c in stock_df.columns:
            dfs=pd.DataFrame({'Stock_'+str(c):stock_df[c],'Sector_'+str(c):sector_df[c]},index=stock_df.index)
            dfsm = dfsm.join(dfs,how='outer')
        # compute returns
        dfsm[['Sector_Open_Return','Stock_Open_Return','Sector_High_Return','Stock_High_Return','Sector_Low_Return',\
            'Stock_Low_Return','Sector_Close_Return','Stock_Close_Return']] = dfsm/dfsm.shift(1)-1
        dfsm = dfsm.dropna()
        anb=pd.DataFrame()
        for c in sector_df.columns:
            stockRolling=dfsm['Stock_'+str(c)+'_Return'].rolling(window=200)
            sectorRolling=dfsm['Sector_'+str(c)+'_Return'].rolling(window=200)
            covmat=stockRolling.cov(sectorRolling)
            df1=covmat/stockRolling.var()
            df1=pd.DataFrame(df1)
            df1.columns=['B_'+str(ticker)+'_'+c]
            anb=anb.join(df1,how='outer')
            df2=stockRolling.mean()-covmat/stockRolling.var()*sectorRolling.mean()
            df2=pd.DataFrame(df2)
            df2.columns=['A_'+str(ticker)+'_'+c]
            anb=anb.join(df2,how='outer')
        return anb        

    def sliceSectorBetas(self):
        i1List=list(['A_','B_'])
        i2List=list(['_Open','_Close','_Low','_High'])
        for i1 in i1List:
            tList=list()
            xList=list()            
            for i2 in i2List:
                for ticker,sector in commons.sp500SectorAssignmentsTicker.items():
                    xList.append(i1+ticker+i2)
                    tList.append(ticker+i2)
            self.anbDict[i1]=self.dfanb.ix[:,xList]
            self.anbDict[i1].columns=tList
            self.anbDict[i1].to_hdf(commons.data_path+i1+'anb.h5','table',mode='w')

        #alphas and betas, calls self.calc_sector_beta()
    def calcSectorBetas(self):
        self.dfanb=pd.DataFrame() #reset anb
        #calc by ticker
        cList=list()
        for ticker,sector in commons.sp500SectorAssignmentsTicker.items():
            startdate=commons.getClosestDate(commons.data_sp500_1st_date[ticker]+dt.timedelta(days=-365))
            anb=self.calcRollingSectorBeta(startdate,ticker)
            self.dfanb=self.dfanb.join(anb,how='outer')
        dfIndex=list()
        for dix in range(commons.date_index_internal[commons.min_date],commons.date_index_internal[commons.max_date['WIKI_SP500']]):
           dfIndex.append(commons.date_index_external[dix]) 
        df1=pd.DataFrame(index=dfIndex)
        self.dfanb=self.dfanb.join(df1,how='outer')  
        self.dfanb=self.dfanb.sort_index()
        self.dfanb=self.dfanb.fillna(method='ffill')
        self.dfanb=self.dfanb.fillna(method='backfill')
        self.dfanb.to_hdf(commons.data_path+'anb.h5','table',mode='w')
        print 'Alpha and Beta have been calculated and stored locally'

        
#needed to know from when onwards stats can be calculated and forecasts can be made. 1st date with known prices
    def calcSp5001stDate(self):
        for ticker,dates in commons.sp500CompDates.items():
            min_date=commons.date_index_external[1]
            for date in dates:
                if min_date<date[0]:
                    min_date=date[0]
            self.df1st_date[ticker]=min_date

        for k,v in commons.sp500_index.items():
            self.df1st_date[v[-8:]]=commons.min_date
            
        with open(commons.data_path+'sp500_1st_date.csv','w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            for k,v in self.df1st_date.items():
                csvwriter.writerow([k,v])
            csvfile.close()
        print '1st dates recorded'

#minmax scaler
    def minmaxscale(self,df):
        print 'Calculating MinMax for ',df
        min_max_scaler = MinMaxScaler()
        if len(getattr(self,df).index)>0:
            return pd.DataFrame(data=min_max_scaler.fit_transform(getattr(self,df)).round(2),\
                                                       index=getattr(self,df).index,\
                                                       columns=getattr(self,df).columns)            
        else:
            print 'No data for '+str(df)

#small stuff for dymamics            
    def p_direction(self,df):
        df[df>0]=1
        return df
        
    def n_direction(self,df):
        df[df<0]=-1
        return df


#for the assembly of Xy
    def calcFundamentals(self,ticker,indicator):
        comp=commons.getSp500CompositionAll()        
        for k,v in commons.getIndexCodes().items():
            columns=list([])
            for t in comp[k]:
                columns.append(str(t)+indicator)
        if self.df1st_date[ticker]<min(self.dfFundamentals.index):
            min_date=min(self.dfFundamentals.index)
        else:
            min_date=self.df1st_date[ticker]   
        ret=self.dfFundamentals.ix[min_date:,columns].mean(axis=1).to_frame()
        ret.columns=list([str(indicator).strip('_ARQ')])
        return ret
        
                
#the actual assembly            
    def assemble_xy(self, ticker):
        df=pd.DataFrame()
        min_max_scaler = MinMaxScaler()        
        
        #sentiment            
        select_columns=list(['Bull-Bear Spread'])
        target_columns=list(['Bull_Bear_Spread'])
        np1=self.dfSentiment.ix[self.df1st_date[ticker]:,select_columns].values
        id1=self.dfSentiment.ix[self.df1st_date[ticker]:,select_columns].index
        df1=pd.DataFrame(data=np1,index=id1,columns=target_columns)
        df1=pd.DataFrame(data=min_max_scaler.fit_transform(df1).round(2),index=df1.index,columns=df1.columns)
        df=df.join(df1,how='outer')   
     
        #fundamentals
        a=list(['_PB_ARQ','_EPSDIL_ARQ'])
        for x in a:
            select_columns=list([str(ticker)+str(x)])
            target_columns=list([str(x).strip('_ARQ')])
            if self.df1st_date[ticker]<min(self.dfFundamentals.index):
                min_date=min(self.dfFundamentals.index)
            else:
                min_date=self.df1st_date[ticker]
            np1=self.dfFundamentals.ix[min_date:,select_columns].values
            id1=self.dfFundamentals.ix[min_date:,select_columns].index
            df1=pd.DataFrame(data=np1,index=id1,columns=target_columns)
            df1=df1/self.calcFundamentals(ticker,x)-1
            np1=np.nan_to_num(df1.values)
            df1=pd.DataFrame(data=np1,index=id1,columns=target_columns)            
            df=df.fillna(method='ffill')
            df=df.fillna(method='backfill')
            df=df.fillna(value=0)  
            df1=pd.DataFrame(data=min_max_scaler.fit_transform(df1).round(2),index=df1.index,columns=df1.columns)
            df=df.join(df1,how='outer')

        #rest, incl labels
        select_columns=list([str(ticker)+'_Open',str(ticker)+'_Low',str(ticker)+'_High',str(ticker)+'_Close'])
        a=list(['1dm','2dm','5dm','30dsma','30dmx','30dmn','5dv','bbands','1er','2er','5er'])
        for x in a:
            target_columns=list([str(x)+'_Open',str(x)+'_Low',str(x)+'_High',str(x)+'_Close'])
            np1=getattr(self,'df'+str(x)).ix[self.df1st_date[ticker]:,select_columns].values
            id1=getattr(self,'df'+str(x)).ix[self.df1st_date[ticker]:,select_columns].index
            df1=pd.DataFrame(data=min_max_scaler.fit_transform(np.nan_to_num(np1)),index=id1,columns=target_columns)
            df=df.join(df1,how='outer')
            
        a=list(['ns_ss','ny_ss'])
        for x in a:
            select_columns=list([str(ticker)+'_'+str(x)])
            target_columns=list([str(x)])
            np1=getattr(self,'df'+str(x)).ix[self.df1st_date[ticker]:,select_columns].values
            np1=np.nan_to_num(np1)
            np1=np1.astype(int)   
            id1=getattr(self,'df'+str(x)).ix[self.df1st_date[ticker]:,select_columns].index
            df1=pd.DataFrame(data=np1,index=id1,columns=target_columns)
            df=df.join(df1,how='outer')                 

        select_columns=list([str(ticker)+'_Open',str(ticker)+'_Low',str(ticker)+'_High',str(ticker)+'_Close'])
        a=list(['1dr','5dr','20dr','1dd','5dd','20dd','12dd'])
        for x in a:
            target_columns=list([str(x)+'_Open',str(x)+'_Low',str(x)+'_High',str(x)+'_Close'])
            np1=getattr(self,'df'+str(x)).ix[self.df1st_date[ticker]:,select_columns].values
            np1=np.nan_to_num(np1)
            np1=np1.astype(int)            
            id1=getattr(self,'df'+str(x)).ix[self.df1st_date[ticker]:,select_columns].index
            df1=pd.DataFrame(data=np1,index=id1,columns=target_columns)
            df=df.join(df1,how='outer')
            
        a=list(['clr','chr'])
        for x in a:
            select_columns=list([str(ticker)+'_'+str(x)])
            target_columns=list([str(x)])
            np1=getattr(self,'df'+str(x)).ix[self.df1st_date[ticker]:,select_columns].values
            np1=np1.astype(int)
            id1=getattr(self,'df'+str(x)).ix[self.df1st_date[ticker]:,select_columns].index
            df1=pd.DataFrame(data=np1,index=id1,columns=target_columns)
            df=df.join(df1,how='outer')            
        
        df=self.fillUpIndex(df)
        df=df.fillna(value=0)        

        return df


#get market cap for index composition        
    def getMarketcap(self):
        if self.refresh_data==True:
            #get all tickers that need to be retireved
            tickers=list()
            for dix in range(commons.date_index_internal[commons.max_date['MARKETCAP']],commons.date_index_internal[commons.max_date['WIKI_SP500']]):
                for ticker,sector in commons.getHistSp500Ticker(commons.date_index_external[dix]).items():
                    if ticker not in tickers:
                        tickers.append(commons.getSP1Ticker(ticker)) 
                        
            items=dict()
            for t in tickers:
                dates=dict()
                if t not in self.dfMarketcap.columns: #collect whole history for new tickers
                    dates['startdate']=commons.min_date
                else:
                    dates['startdate']=commons.max_date['MARKETCAP']
                dates['enddate']=commons.max_date['WIKI_SP500']
                items['SF1/'+t+'_MARKETCAP']=dates
        
            consItems=dict() #cosolidate items to have less calls
            for t,d in items.items():
                key=(d['startdate'],d['enddate'])
                try:
                    consItems[key].append(t)
                except KeyError:
                    consItems[key]=list()
                    consItems[key].append(t)

            df1=pd.DataFrame()                    
            for dates,tickerlist in consItems.items():
                tList=list() #max call 50 entries at the time
                for i in range (0,len(tickerlist)):
                    tList.append(tickerlist[i])
#                    print i,' of ',len(tickerlist)
                    if i%100==5 or i==(len(tickerlist)-1):
                        print i
                        df1=pd.DataFrame()
                        df1=Quandl.get(tList,start_date=dates[0],end_date=dates[1])
                        tList=list()
                        
                        columns=list([])
                        for x in df1.columns:
                            if 'Not Found' in x:
                                print 'MarketCap to:',x,'not found.'
                            x=str(x).replace(' - ', '_')
                            x=x.replace('SF1/','')
                            x=x.replace('_Value','')
                            x=x.replace('_MARKETCAP','')
                            columns.append(x)
                        df1.columns=columns
                        
                        self.dfMarketcap=self.processResults(df1,self.dfMarketcap,'MARKETCAP',first='ffill')                   
            print 'Marketcap data refreshed'      
      


#calculate index composition and store        
    def getIndexComposition(self):
        for k,v in commons.sp500_index.items():
            index_t=v[-8:]
            indexComp=commons.read_dataframe(commons.data_path+'HIST_'+index_t+'.h5')
            relMarketCap=self.dfMarketcap.ix[indexComp.index,indexComp.columns]
            dailySum=(indexComp*relMarketCap).sum(axis=1)
            marketCap=indexComp*relMarketCap
            for c in indexComp.columns:
                marketCap.ix[:,c]=marketCap.ix[:,c]/dailySum
            marketCap.to_hdf(commons.data_path+'PCT_'+index_t+'.h5','table',mode='w')
        
        print 'Index composition stored. Update finished.'            
        
    #drop the top/last 1% as outliers
    def drop_outliers(self,df):
        for c in df.columns:
            stock=str(c).replace('_Open','')
            stock=stock.replace('_Close','')
            stock=stock.replace('_High','')
            stock=stock.replace('_Low','')
            stock=stock.replace('_clr','')
            stock=stock.replace('_chr','')
            stock=stock.replace('_ny_ss','')
            stock=stock.replace('_ns_ss','')
            startdate=commons.data_sp500_1st_date[stock]
            series=df.ix[startdate:,c]            
            thold=int(.05*series.value_counts().count())
            if thold>0:
                freq=series.value_counts().sort_index(ascending=True).iloc[:thold].sort_index(ascending=False).iloc[:1].index.values[0]
                series[series<freq]=freq
                freq=series.value_counts().sort_index(ascending=False).iloc[:thold].sort_index(ascending=True).iloc[:1].index.values[0]
                series[series>freq]=freq
                df.ix[startdate:,c]=series
        return df


    def processResults(self,df,target,filename,first='ffill'):
        for c in df.columns:
            if 'Not Found' in c:
                print 'Fundamentals to:',c
            else:
                if c in target.columns:
                    for i in df.index:
                        target.ix[i,c]=df.ix[i,c]        
                else:
                    target=target.join(getattr(df,c),how='outer')
 
        target=self.fillUpIndex(target,first)
        target.to_hdf(commons.data_path+filename+'.h5','table',mode='w')
        return target        

    def fillUpIndex(self,target,first='ffill'):
        dfIndex=list()
        for dix in range(commons.date_index_internal[commons.min_date],commons.date_index_internal[commons.max_date['WIKI_SP500']]+1):
           dfIndex.append(commons.date_index_external[dix]) 
        df1=pd.DataFrame(index=dfIndex)
        target=target.join(df1,how='outer')
        target=target.sort_index()
        if first=='ffill':
            target=target.fillna(method='ffill')
            target=target.fillna(method='backfill')    
        else:
            target=target.fillna(method='backfill')
            target=target.fillna(method='ffill')    
        return target
        
        
    def checkDemoData(self,frame):
        columns=list()

        if not commons.demo_scenario:
            return getattr(self,frame)
        elif frame=='dfSector':
            columns.append('ARCA_VOX'+'_Low')
            columns.append('ARCA_VOX'+'_High')
            columns.append('ARCA_VOX'+'_Open')
            columns.append('ARCA_VOX'+'_Close')
            return getattr(self,frame).ix[:,columns]
        elif frame=='dfFundamentals':
            for ticker in commons.getHistSp500TickerList(1,1,False):
                columns.append(ticker+'_EPSDIL_ARQ')
                columns.append(ticker+'_PB_ARQ')
            return getattr(self,frame).ix[:,columns]
        elif frame=='dfShortSell':
            for ticker in commons.getHistSp500TickerList(1,1,False):
                columns.append('FNSQ_'+ticker+'_ShortVolume')
                columns.append('FNSQ_'+ticker+'_ShortExemptVolume')
                columns.append('FNSQ_'+ticker+'_TotalVolume')
                columns.append('FNYX_'+ticker+'_ShortVolume')
                columns.append('FNYX_'+ticker+'_ShortExemptVolume')
                columns.append('FNYX_'+ticker+'_TotalVolume')
            return getattr(self,frame).ix[:,columns]                
        elif frame=='dfMarketcap':
            for ticker in commons.getHistSp500TickerList(1,1,False):
                columns.append(ticker)
            return getattr(self,frame).ix[:,columns]                
        elif frame=='dfanb':
            for ticker in commons.getHistSp500TickerList(1,1,False):
                columns.append('B_'+ticker+'_Open')
                columns.append('B_'+ticker+'_Close')
                columns.append('B_'+ticker+'_Low')
                columns.append('B_'+ticker+'_High')
                columns.append('A_'+ticker+'_Open')
                columns.append('A_'+ticker+'_Close')
                columns.append('A_'+ticker+'_Low')
                columns.append('A_'+ticker+'_High')
            return getattr(self,frame).ix[:,columns]        
        elif frame=='dfSentiment':
            return getattr(self,frame)