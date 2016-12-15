import commons
import pandas as pd
import numpy as np
import quandl as Quandl
import csv
import datetime as dt
from sklearn.preprocessing import MinMaxScaler

class mydata(object):   
    Quandl.ApiConfig.api_key=''      
    data_sp500_1st_date={}
    last_date=commons.max_date['WIKI_SP500']
    reload_baseline=False
    refresh_data=True
    demo_scenario=True
    sp500_change=dict()
    data_sp500_1dr=pd.DataFrame()
    data_sp500_5dr=pd.DataFrame()
    data_sp500_20dr=pd.DataFrame()
    data_sp500_5dm=pd.DataFrame()
    data_sp500_sector_a_and_b=pd.DataFrame()
    data_sp500_30dsma=pd.DataFrame()
    data_sp500_30dmx=pd.DataFrame()
    data_sp500_30dmn=pd.DataFrame()    
    data_sp500_anb=pd.DataFrame()
    data_sp500_bbands=pd.DataFrame()
    data_sp500_clr=pd.DataFrame()
    data_sp500_chr=pd.DataFrame()
    data_sp500_ny_ss=pd.DataFrame()
    data_sp500_ns_ss=pd.DataFrame()
    data_sp500_1er=pd.DataFrame()
    data_sp500_2er=pd.DataFrame()
    data_sp500_5er=pd.DataFrame()    

        
        
    def __init__(self,refresh_data=False,reload_baseline=False,demo_scenario=True,quandlkey=''):    
        Quandl.ApiConfig.api_key=quandlkey
        self.demo_scenario=demo_scenario
        self.refresh_data=refresh_data
        self.data_sp500_1st_date=commons.data_sp500_1st_date    
        self.data_sp500=commons.read_dataframe(commons.data_path+'WIKI_SP500.h5')
        self.data_sp500_sector=commons.read_dataframe(commons.data_path+'SECTOR_SP500.h5')
        self.data_sp500_sentiment=commons.read_dataframe(commons.data_path+'SENT_SP500.h5')
        self.data_sp500_fundamentals=commons.read_dataframe(commons.data_path+'FUND_SP500.h5')    
        self.data_sp500_short_sell=commons.read_dataframe(commons.data_path+'SHORT_SP500.h5')
        self.data_sp500_marketcap=commons.read_dataframe(commons.data_path+'MARKETCAP.h5')  
        self.data_sp500_anb=commons.read_dataframe(commons.data_path+'anb.h5')  
        self.data_last_calloff=commons.read_dataframe(commons.data_path+'last_calloff.h5')
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
            for c in self.data_sp500.columns:
                if c[-4:] in ['Open', 'lose', 'High', '_Low']:
                    column_selection.append(c)
        self.data_sp500_prices=self.data_sp500.ix[:,column_selection]
         
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
            self.data_sp500_1st_date[k]=min(wiki.ix[k].index)
        wiki_sp500.dropna(axis=0,how='all',inplace=True)
        wiki_sp500.to_hdf(commons.data_path+'WIKI_SP500.h5','table',mode='w')                    
        print 'WIKI based SP500 ticker data stored'



#load index prices
    def getIndexData(self):
        self.data_sp500_sector=commons.read_dataframe(commons.data_path+'SECTOR_SP500.h5')   

        if self.refresh_data==True:
            enddate=commons.idx_today
            startdate=commons.max_date['SECTOR_SP500']

            for k,v in commons.sp500_index.items():
                df=pd.DataFrame()
                df=Quandl.get(v,start_date=startdate,end_date=enddate)
                df.columns=[str(v[-8:])+'_Open',str(v[-8:])+'_High',str(v[-8:])+'_Low',str(v[-8:])+'_Close',str(v[-8:])+'_Volume']
                for c in df.columns:
                    if 'Volume' not in c:
                        if c not in self.data_sp500_sector.columns:
                            self.data_sp500_sector=self.data_sp500_sector.join(getattr(df,c),how='outer')
                        else:
                            for i in df.index:
                                self.data_sp500_sector.ix[i,c]=df.ix[i,c]
            
            self.data_sp500_sector=self.processResults(df,self.data_sp500_sector,'SECTOR_SP500')
            print 'Index prices retrieved and stored'
        else:
            print 'Local index data loaded.'



#load fundamentals            
    def get_fundamentals(self):
#commons.getSP1Ticker(ticker)!!        
        self.data_last_calloff=commons.read_dataframe(commons.data_path+'last_calloff.h5')        
        
        if self.refresh_data==True:
            #get fundamentals
            if self.data_sp500_fundamentals.empty==True:
                max_date=commons.max_date['FUND_SP500']
            else:
                max_date=max(list(self.data_last_calloff.query('fundamentals > 0').index))
                
                #collect unknown days and update data_SP500_fundamentals
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
                    df=Quandl.get(itemList,start_date=max_date-dt.timedelta(days=1),end_date=commons.idx_today)
                
                    columns=list([])
                    for x in df.columns:
                        x=str(x).replace(' - ', '_')
                        x=x.replace('SF1/','')
                        x=x.replace('_Value','')
                        columns.append(x)
                    df.columns=columns
                
                    self.data_sp500_fundamentals=self.processResults(df,self.data_sp500_fundamentals,'FUND_SP500')                

            self.data_last_calloff.ix[dt.datetime.today(),'fundamentals']=1
            self.data_last_calloff.to_hdf(commons.data_path+'last_calloff.h5','table',mode='w')             
            print 'Fundamentals data refreshed'
                
        else:                
            print 'Fundamentals loaded'


#load short selling    
    def get_short_sell(self):
        if self.refresh_data==True:        
            #get short sell
            max_date=commons.max_date['SHORT_SP500']

            short_sell=list(['FINRA/FNSQ_','FINRA/FNYX_'])
            for s in short_sell:
                items=list([])
                for ticker in commons.getHistSp500TickerList(commons.date_index_internal[commons.max_date['SHORT_SP500']],commons.date_index_internal[commons.max_date['WIKI_SP500']]):
                    items=[s+ticker]
                    print items
            
                    df=pd.DataFrame([])
                    df=Quandl.get(items,start_date=max_date+dt.timedelta(days=1),end_date=dt.date.today())
                    columns=list([])
                    for x in df.columns:
                        x=str(x).replace(' - ', '_')
                        x=x.replace('FINRA/','')
                        columns.append(x)
                    df.columns=columns

                    self.data_sp500_short_sell=self.processResults(df,self.data_sp500_short_sell,'SHORT_SP500')

            print 'Short sell data refreshed'
        else:
            print 'No new short sell data to collect'   

#load sentiment                
    def get_sentiment(self):
        if self.refresh_data==True:                  
            #get sentiment
            max_date=commons.max_date['SENT_SP500']
                
            df=pd.DataFrame([])
            df=Quandl.get('AAII/AAII_SENTIMENT',start_date=max_date+dt.timedelta(days=1),end_date=commons.idx_today)
            self.data_sp500_sentiment=self.processResults(df,self.data_sp500_sentiment,'SENT_SP500')
            print 'Sentiment data refreshed'
        else:
            print 'Sentiment data loaded.'                   


#this is not really needed as we have no historical records to index composition. If we had them,
#this method would deliver the deltas            


    def logSp500Changes(self):
        maxDate=max(self.data_sp500.index)
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
    def sp500_fillna(self):
        self.data_sp500=self.data_sp500.sort_index()
        self.data_sp500_fundamentals=self.data_sp500_fundamentals.sort_index()
        self.data_sp500_short_sell=self.data_sp500_short_sell.sort_index()
        self.data_sp500_sentiment=self.data_sp500_sentiment.sort_index()
        self.data_sp500_sector=self.data_sp500_sector.sort_index()
        self.data_last_calloff=self.data_last_calloff.sort_index()
        self.data_sp500_anb=self.data_sp500_anb.sort_index()

        
        self.data_sp500=self.data_sp500.fillna(method='backfill')
        self.data_sp500_fundamentals=self.data_sp500_fundamentals.fillna(method='backfill')
        self.data_sp500_short_sell=self.data_sp500_short_sell.fillna(method='backfill')
        self.data_sp500_sentiment=self.data_sp500_sentiment.fillna(method='backfill')
        self.data_sp500_sector=self.data_sp500_sector.fillna(method='backfill')
        self.data_last_calloff=self.data_last_calloff.fillna(method='backfill')   
        self.data_sp500_anb=self.data_sp500_anb.fillna(method='backfill')
        
        self.data_sp500=self.data_sp500.fillna(method='ffill')
        self.data_sp500_fundamentals=self.data_sp500_fundamentals.fillna(method='ffill')
        self.data_sp500_short_sell=self.data_sp500_short_sell.fillna(method='ffill')
        self.data_sp500_sentiment=self.data_sp500_sentiment.fillna(method='ffill')
        self.data_sp500_sector=self.data_sp500_sector.fillna(method='ffill')
        self.data_last_calloff=self.data_last_calloff.fillna(method='ffill')   
        self.data_sp500_anb=self.data_sp500_anb.fillna(method='ffill')
        print 'FillNa on source data performed.'
        

#calculate features and labels
    def calc_indicators(self):     
        self.data_sp500_fundamentals=self.checkDemoData('data_sp500_fundamentals')
        self.data_sp500_short_sell=self.checkDemoData('data_sp500_short_sell')
        self.data_sp500_sentiment=self.checkDemoData('data_sp500_sentiment')
        self.data_sp500_sector=self.checkDemoData('data_sp500_sector')
        self.data_sp500_anb=self.checkDemoData('data_sp500_anb')
        
        #momentum 5 days
        self.data_sp500_5dm=(self.data_sp500_prices-self.data_sp500_prices.shift(5))/self.data_sp500_prices.shift(5)
        #momentum 2 days
        self.data_sp500_2dm=(self.data_sp500_prices-self.data_sp500_prices.shift(2))/self.data_sp500_prices.shift(2)
        #momentum 1 days
        self.data_sp500_1dm=(self.data_sp500_prices-self.data_sp500_prices.shift(1))/self.data_sp500_prices.shift(1)
        print 'Momentum calculated.'
        
        #delta to expected return
        for k,sector in commons.sp500SectorAssignmentsTicker.items():        
            select_columns=list([str(k)+'_Open',str(k)+'_Low',str(k)+'_High',str(k)+'_Close'])
            df1=self.data_sp500_1dm.ix[:,select_columns]
            df1.columns=list(['Open','Low','High','Close'])
            
            sector_column=list([str(sector[-8:])+'_Open',str(sector[-8:])+'_High',str(sector[-8:])+'_Low',str(sector[-8:])+'_Close'])
            df2=self.data_sp500_sector.ix[:,sector_column]        
            df2.columns=list(['Open','Low','High','Close'])
            
            select_columns=list(['B_'+str(k)+'_Open','B_'+str(k)+'_Low','B_'+str(k)+'_High','B_'+str(k)+'_Close'])
            df3=self.data_sp500_anb.ix[:,select_columns]
            df3.columns=list(['Open','Low','High','Close'])
            
            select_columns=list(['A_'+str(k)+'_Open','A_'+str(k)+'_Low','A_'+str(k)+'_High','A_'+str(k)+'_Close'])
            df4=self.data_sp500_anb.ix[:,select_columns]
            df4.columns=list(['Open','Low','High','Close'])
            
            df1=df1-((df2-df2.shift(1))/df2.shift(1)*df3+df4)
            df1.columns=list([str(k)+'_Open',str(k)+'_Low',str(k)+'_High',str(k)+'_Close'])            
            df1=df1
            self.data_sp500_1er=self.data_sp500_1er.join(df1,how='outer')
            
            select_columns=list([str(k)+'_Open',str(k)+'_Low',str(k)+'_High',str(k)+'_Close'])
            df1=self.data_sp500_2dm.ix[:,select_columns]
            df1.columns=list(['Open','Low','High','Close'])

            df1=df1-df1-((df2-df2.shift(2))/df2.shift(2)*df3+df4)
            df1.columns=list([str(k)+'_Open',str(k)+'_Low',str(k)+'_High',str(k)+'_Close'])
            df1=df1
            self.data_sp500_2er=self.data_sp500_2er.join(df1,how='outer')
            
            
            select_columns=list([str(k)+'_Open',str(k)+'_Low',str(k)+'_High',str(k)+'_Close'])
            df1=self.data_sp500_5dm.ix[:,select_columns]
            df1.columns=list(['Open','Low','High','Close'])

            df1=df1-df1-((df2-df2.shift(5))/df2.shift(5)*df3+df4)
            df1.columns=list([str(k)+'_Open',str(k)+'_Low',str(k)+'_High',str(k)+'_Close'])            
            df1=df1
            self.data_sp500_5er=self.data_sp500_5er.join(df1,how='outer')
        print 'Delta to expected return.'
        
        #sma 30 days
        self.data_sp500_30dsma=self.data_sp500_prices/pd.DataFrame.rolling(self.data_sp500_prices,30).mean()
        #comp to max and min 30 days
        self.data_sp500_30dmx=self.data_sp500_prices/pd.DataFrame.rolling(self.data_sp500_prices,30).max()
        self.data_sp500_30dmn=self.data_sp500_prices/pd.DataFrame.rolling(self.data_sp500_prices,30).min()
        #vola week
        self.data_sp500_5dv=1-pd.DataFrame.rolling(self.data_sp500_prices,30).min()/pd.DataFrame.rolling(self.data_sp500_prices,30).max()
        #bollinger bands
        stock_rm_df=pd.DataFrame.rolling(self.data_sp500_prices,200).mean()
        self.data_sp500_bbands=(self.data_sp500_prices-stock_rm_df)/(2*self.data_sp500_prices.std(axis=0))     
        print 'min, max, sma, vola and bbbands calculated.'
        #returns for labels        
        self.data_sp500_1dr=(self.data_sp500_prices.shift(-1)/self.data_sp500_prices-1).round(2)*100
        self.data_sp500_5dr=(self.data_sp500_prices.shift(-5)/self.data_sp500_prices-1).round(2)*100
        self.data_sp500_20dr=(self.data_sp500_prices.shift(-20)/self.data_sp500_prices-1).round(2)*100
        #directional labels
        self.data_sp500_1dd=(self.data_sp500_prices.shift(-1)/self.data_sp500_prices-1)*100
        self.data_sp500_5dd=(self.data_sp500_prices.shift(-5)/self.data_sp500_prices-1)*100
        self.data_sp500_20dd=(self.data_sp500_prices.shift(-20)/self.data_sp500_prices-1)*100
        #close to low and close to high
        for k,v in commons.sp500SectorAssignmentsTicker.items():        
            #close to low and close to high
            df1=pd.DataFrame(self.data_sp500_prices.ix[:,str(k)+'_Low'].shift(-1)/self.data_sp500_prices.ix[:,str(k)+'_Close']-1).round(2)*100
            df1.columns=list([str(k)+'_clr'])
            self.data_sp500_clr=self.data_sp500_clr.join(df1,how='outer')
            df1=pd.DataFrame(self.data_sp500_prices.ix[:,str(k)+'_High'].shift(-1)/self.data_sp500_prices.ix[:,str(k)+'_Close']-1).round(2)*100
            df1.columns=list([str(k)+'_chr'])
            self.data_sp500_chr=self.data_sp500_chr.join(df1,how='outer')
            
            #short %
            try:
                df1=pd.DataFrame(self.data_sp500_short_sell.ix[:,'FNSQ_'+str(k)+'_ShortVolume']/self.data_sp500_short_sell.ix[:,'FNSQ_'+str(k)+'_TotalVolume'])*10
                df1.columns=list([str(k)+'_ns_ss'])
                self.data_sp500_ns_ss=self.data_sp500_ns_ss.join(df1,how='outer')
                df1=pd.DataFrame(self.data_sp500_short_sell.ix[:,'FNYX_'+str(k)+'_ShortVolume']/self.data_sp500_short_sell.ix[:,'FNYX_'+str(k)+'_TotalVolume'])*10
                df1.columns=list([str(k)+'_ny_ss'])
                self.data_sp500_ny_ss=self.data_sp500_ny_ss.join(df1,how='outer')
            except KeyError:
                print 'Short Sell data for: ',k,' missing.'
                df1=pd.DataFrame(columns=[str(k)+'_ns_ss'])
                self.data_sp500_ns_ss=self.data_sp500_ns_ss.join(df1,how='outer')
                df1=pd.DataFrame(columns=[str(k)+'_ny_ss'])
                self.data_sp500_ny_ss=self.data_sp500_ns_ss.join(df1,how='outer')
                
            

        print 'Labels calculated.'
        #alpha & beta

        #drop outliers; top and bottom 1%    
        a=list(['1dm','2dm','5dm','30dsma','30dmx','30dmn','5dv','bbands','clr','chr','ny_ss','ns_ss','1er','2er','5er'])
        for x in a:
            setattr(self,'data_sp500_'+str(x),self.drop_outliers(getattr(self,'data_sp500_'+str(x))))
                
        
        #fill, minmax and direction
        a=list(['1dm','2dm','5dm','30dsma','30dmx','30dmn','5dv','bbands','1dd','5dd','20dd','clr','chr','ny_ss','ns_ss','1er','2er','5er'])
        for x in a:
            setattr(self,'data_sp500_'+str(x),getattr(self,'data_sp500_'+str(x)).fillna(method='backfill'))
            setattr(self,'data_sp500_'+str(x),getattr(self,'data_sp500_'+str(x)).fillna(method='ffill'))
            
        a=list(['1dm','2dm','5dm','30dsma','30dmx','30dmn','5dv','bbands'])
        for x in a:
            try:
                setattr(self,'data_sp500_'+str(x),self.minmaxscale('data_sp500_'+str(x)))
            except ValueError:
                print 'MinMax value error in data_sp500_'+str(x)
                getattr(self,'data_sp500_'+str(x)).to_hdf(commons.data_path+'data_sp500_'+str(x)+'.h5','table',mode='w')
                

        a=list(['1dd','5dd','20dd'])
        for x in a: 
            setattr(self,'data_sp500_'+str(x),self.p_direction(getattr(self,'data_sp500_'+str(x))))
            setattr(self,'data_sp500_'+str(x),self.n_direction(getattr(self,'data_sp500_'+str(x))))
            
        l_i=505
        for k,v in commons.sp500SectorAssignmentsTicker.items():     
            Xy_all=self.assemble_xy(k)
            Xy_all=Xy_all.fillna(method='backfill')
            Xy_all=Xy_all.fillna(method='ffill')            
            Xy_all.to_hdf(commons.data_path+'Xy_all_'+str(k),'table',mode='w')
            l_i-=1
            print 'Xy_all to '+str(k)+' assembled. '+str(l_i)+' to go.'
            
        #self.drop_obsolete_anb()#only needed 1x due to obsolete index.

#individual alpha and beta, compared to the industry
    def calcRollingSectorBeta(self,startdate,ticker):
        stock_column=list([ticker+'_Open',ticker+'_High',ticker+'_Low',ticker+'_Close'])
        stock_df=self.data_sp500_prices.ix[commons.anb_min_date:,stock_column]
        stock_df.columns=list(['Open','High','Low','Close'])
        sector_column=list([commons.sp500SectorAssignmentsTicker[ticker]+'_Open',commons.sp500SectorAssignmentsTicker[ticker]+'_High',commons.sp500SectorAssignmentsTicker[ticker]+'_Low',commons.sp500SectorAssignmentsTicker[ticker]+'_Close'])
        sector_df=self.data_sp500_sector.ix[commons.anb_min_date:,sector_column]
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

#alphas and betas, calls self.calc_sector_beta()
    def calcSectorBetas(self):
        self.data_sp500_anb=pd.DataFrame() #reset anb
        startdate=commons.getClosestDate(commons.min_date+dt.timedelta(days=-365))
        #calc by ticker
        for ticker,sector in commons.sp500SectorAssignmentsTicker.items():
            print 'Calculating Alpha and Beta for: ',ticker
            anb=self.calcRollingSectorBeta(startdate,ticker)
            self.data_sp500_anb=self.data_sp500_anb.join(anb,how='outer')
        dfIndex=list()
        for dix in range(commons.date_index_internal[commons.min_date],commons.date_index_internal[max(self.data_sp500.index)]):
           dfIndex.append(commons.date_index_external[dix]) 
        df1=pd.DataFrame(index=dfIndex)
        self.data_sp500_anb=self.data_sp500_anb.join(df1,how='outer')            
        self.data_sp500_anb.to_hdf(commons.data_path+'anb.h5','table',mode='w')
        print 'Alpha and Beta have been calculated and stored locally'

        
#needed to know from when onwards stats can be calculated and forecasts can be made. 1st date with known prices
    def calc_sp500_1st_date(self):
        for ticker,dates in commons.sp500CompDates.items():
            min_date=commons.date_index_external[1]
            for date in dates:
                if min_date<date[0]:
                    min_date=date[0]
            self.data_sp500_1st_date[ticker]=min_date

        for k,v in commons.sp500_index.items():
            self.data_sp500_1st_date[v[-8:]]=commons.min_date
            
        with open(commons.data_path+'sp500_1st_date.csv','w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',')
            for k,v in self.data_sp500_1st_date.items():
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
    def calc_fundamentals(self,ticker,indicator):
        comp=commons.getSp500CompositionAll()        
        for k,v in commons.getIndexCodes().items():
            columns=list([])
            for t in comp[k]:
                columns.append(str(t)+indicator)
        if self.data_sp500_1st_date[ticker]<min(self.data_sp500_fundamentals.index):
            min_date=min(self.data_sp500_fundamentals.index)
        else:
            min_date=self.data_sp500_1st_date[ticker]   
        ret=self.data_sp500_fundamentals.ix[min_date:,columns].mean(axis=1).to_frame()
        ret.columns=list([str(indicator).strip('_ARQ')])
        return ret
        
                
#the actual assembly            
    def assemble_xy(self, ticker):
        df=pd.DataFrame()
        min_max_scaler = MinMaxScaler()        
        
        #sentiment            
        select_columns=list(['Bull-Bear Spread'])
        target_columns=list(['Bull_Bear_Spread'])
        np1=self.data_sp500_sentiment.ix[self.data_sp500_1st_date[ticker]:,select_columns].values
        id1=self.data_sp500_sentiment.ix[self.data_sp500_1st_date[ticker]:,select_columns].index
        df1=pd.DataFrame(data=np1,index=id1,columns=target_columns)
        df1=pd.DataFrame(data=min_max_scaler.fit_transform(df1).round(2),index=df1.index,columns=df1.columns)
        df=df.join(df1,how='outer')   
     
        #fundamentals
        a=list(['_PB_ARQ','_EPSDIL_ARQ'])
        for x in a:
            select_columns=list([str(ticker)+str(x)])
            target_columns=list([str(x).strip('_ARQ')])
            if self.data_sp500_1st_date[ticker]<min(self.data_sp500_fundamentals.index):
                min_date=min(self.data_sp500_fundamentals.index)
            else:
                min_date=self.data_sp500_1st_date[ticker]
            np1=self.data_sp500_fundamentals.ix[min_date:,select_columns].values
            id1=self.data_sp500_fundamentals.ix[min_date:,select_columns].index
            df1=pd.DataFrame(data=np1,index=id1,columns=target_columns)
            df1=df1/self.calc_fundamentals(ticker,x)-1
            np1=np.nan_to_num(df1.values)
            df1=pd.DataFrame(data=np1,index=id1,columns=target_columns)            
            df=df.fillna(method='backfill')
            df=df.fillna(method='ffill')
            df=df.fillna(value=0)  
            df1=pd.DataFrame(data=min_max_scaler.fit_transform(df1).round(2),index=df1.index,columns=df1.columns)
            df=df.join(df1,how='outer')

        #rest, incl labels
        select_columns=list([str(ticker)+'_Open',str(ticker)+'_Low',str(ticker)+'_High',str(ticker)+'_Close'])
        a=list(['1dm','2dm','5dm','30dsma','30dmx','30dmn','5dv','bbands','1er','2er','5er'])
        for x in a:
            target_columns=list([str(x)+'_Open',str(x)+'_Low',str(x)+'_High',str(x)+'_Close'])
            np1=getattr(self,'data_sp500_'+str(x)).ix[self.data_sp500_1st_date[ticker]:,select_columns].values
            id1=getattr(self,'data_sp500_'+str(x)).ix[self.data_sp500_1st_date[ticker]:,select_columns].index
            df1=pd.DataFrame(data=min_max_scaler.fit_transform(np.nan_to_num(np1)),index=id1,columns=target_columns)
            df=df.join(df1,how='outer')
            
        a=list(['_ns_ss','_ny_ss'])
        for x in a:
            select_columns=list([str(ticker)+str(x)])
            target_columns=list([str(x)])
            np1=getattr(self,'data_sp500'+str(x)).ix[self.data_sp500_1st_date[ticker]:,select_columns].values
            np1=np.nan_to_num(np1)
            np1=np1.astype(int)   
            id1=getattr(self,'data_sp500'+str(x)).ix[self.data_sp500_1st_date[ticker]:,select_columns].index
            df1=pd.DataFrame(data=np1,index=id1,columns=target_columns)
            df=df.join(df1,how='outer')                 

        select_columns=list([str(ticker)+'_Open',str(ticker)+'_Low',str(ticker)+'_High',str(ticker)+'_Close'])
        a=list(['1dr','5dr','20dr','1dd','5dd','20dd'])
        for x in a:
            target_columns=list([str(x)+'_Open',str(x)+'_Low',str(x)+'_High',str(x)+'_Close'])
            np1=getattr(self,'data_sp500_'+str(x)).ix[self.data_sp500_1st_date[ticker]:,select_columns].values
            np1=np.nan_to_num(np1)
            np1=np1.astype(int)            
            id1=getattr(self,'data_sp500_'+str(x)).ix[self.data_sp500_1st_date[ticker]:,select_columns].index
            df1=pd.DataFrame(data=np1,index=id1,columns=target_columns)
            df=df.join(df1,how='outer')
            
        a=list(['_clr','_chr'])
        for x in a:
            select_columns=list([str(ticker)+str(x)])
            target_columns=list([str(x)])
            np1=getattr(self,'data_sp500'+str(x)).ix[self.data_sp500_1st_date[ticker]:,select_columns].values
            np1=np1.astype(int)
            id1=getattr(self,'data_sp500'+str(x)).ix[self.data_sp500_1st_date[ticker]:,select_columns].index
            df1=pd.DataFrame(data=np1,index=id1,columns=target_columns)
            df=df.join(df1,how='outer')            
        
        df=df.fillna(method='backfill')
        df=df.fillna(method='ffill')
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
                if t not in self.data_sp500_marketcap.columns: #collect whole history for new tickers
                    dates['startdate']=commons.min_date
                else:
                    dates['startdate']=commons.max_date['MARKETCAP']
                dates['enddate']=commons.idx_today
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
                                print x
                            x=str(x).replace(' - ', '_')
                            x=x.replace('SF1/','')
                            x=x.replace('_Value','')
                            x=x.replace('_MARKETCAP','')
                            columns.append(x)
                        df1.columns=columns
                        
                        self.data_sp500_marketcap=self.processResults(df1,self.data_sp500_marketcap,'MARKETCAP',first='ffill')                   
            print 'Marketcap data refreshed'      
      


#calculate index composition and store        
    def get_index_composition(self):
        for k,v in commons.sp500_index.items():
            index_t=v[-8:]
            indexComp=commons.read_dataframe(commons.data_path+'HIST_'+index_t+'.h5')
            relMarketCap=self.data_sp500_marketcap.ix[indexComp.index,indexComp.columns]
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


    def processResults(self,df,target,filename,first='backfill'):
        for c in df.columns:
            if 'Not Found' in c:
                a=1
            else:
                if c in target.columns:
                    for i in df.index:
                        target.ix[i,c]=df.ix[i,c]        
                else:
                    target=target.join(getattr(df,c),how='outer')
 
        dfIndex=list()
        for dix in range(commons.date_index_internal[commons.min_date],commons.date_index_internal[commons.max_date['WIKI_SP500']]):
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
        target.to_hdf(commons.data_path+filename+'.h5','table',mode='w')
        return target        

        
    def checkDemoData(self,frame):
        columns=list()

        if not commons.demo_scenario:
            return getattr(self,frame)
        elif frame=='data_sp500_sector':
            columns.append('ARCA_VOX'+'_Low')
            columns.append('ARCA_VOX'+'_High')
            columns.append('ARCA_VOX'+'_Open')
            columns.append('ARCA_VOX'+'_Close')
            return getattr(self,frame).ix[:,columns]
        elif frame=='data_sp500_fundamentals':
            for ticker in commons.getHistSp500TickerList(1,1,False):
                columns.append(ticker+'_EPSDIL_ARQ')
                columns.append(ticker+'_PB_ARQ')
            return getattr(self,frame).ix[:,columns]
        elif frame=='data_sp500_short_sell':
            for ticker in commons.getHistSp500TickerList(1,1,False):
                columns.append('FNSQ_'+ticker+'_ShortVolume')
                columns.append('FNSQ_'+ticker+'_ShortExemptVolume')
                columns.append('FNSQ_'+ticker+'_TotalVolume')
                columns.append('FNYX_'+ticker+'_ShortVolume')
                columns.append('FNYX_'+ticker+'_ShortExemptVolume')
                columns.append('FNYX_'+ticker+'_TotalVolume')
            return getattr(self,frame).ix[:,columns]                
        elif frame=='data_sp500_marketcap':
            for ticker in commons.getHistSp500TickerList(1,1,False):
                columns.append(ticker)
            return getattr(self,frame).ix[:,columns]                
        elif frame=='data_sp500_anb':
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
        elif frame=='data_sp500_sentiment':
            return getattr(self,frame)