import commons
import pandas as pd
import datetime as dt
import numpy as np

class cl_aNb(object):
    def __init__(self):
        self.data_sp500_anb=commons.read_dataframe(commons.data_path+'anb.h5')
        self.data_sp500_sector=commons.read_dataframe(commons.data_path+'SECTOR_SP500.h5')
        self.data_sp500=commons.read_dataframe(commons.data_path+'WIKI_SP500.h5')
        self.data_sp500_prices=pd.DataFrame()
        column_selection=list([])
        for c in self.data_sp500.columns:
            if c[-4:] in ['Open', 'lose', 'High', '_Low']:
                column_selection.append(c)
        self.data_sp500_prices=self.data_sp500.ix[:,column_selection]   
        self.calcSectorBetas()
        
        
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
        startdate=commons.getClosestDate(commons.max_date['ANB']+dt.timedelta(days=-365))
        
        #calc by ticker
        for ticker,sector in commons.sp500SectorAssignmentsTicker.items():
            print ticker
            anb=self.calcRollingSectorBeta(startdate,ticker)
            self.data_sp500_anb=self.data_sp500_anb.join(anb,how='outer')
        self.data_sp500_anb.to_hdf(commons.data_path+'anb.h5','table',mode='w')
        print 'Alpha and Beta have been calculated and stored locally' 
        
x=cl_aNb()