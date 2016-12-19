import commons
import pandas as pd
from sklearn.externals import joblib
import tables

class forecast(object):
    trading_strategy='aggressiv'
    m=None
    train_uuid=None
    
    def __init__(self,market,train_uuid):  
        self.m=market
        self.train_uuid=train_uuid
        
    def trained(self,train_uuid,ticker):
        dba=tables.open_file(commons.stats_path+'simulation.h5', 'r+')
        t_stats=dba.get_node('/','stats')
        found=False
        stats=t_stats.read_where('(ticker=='+"'"+str(ticker)+"'"+")"+' & (kpi=='+"'"+'1dd_Close'+"')")
        for row in stats:
            if str(self.train_uuid)==row['train_uuid']:
                found=True
                break
        dba.close()
        return found
        
    def getActionUntrained(self,p,sector,ticker,dix):
        pct=commons.read_dataframe(commons.data_path+'PCT_'+sector+'h5')
        try:
            actualVol=p.portfolio[ticker]
        except KeyError:
            actualVol=0
        targetVol=int((p.get_portfolio_value(sector,dix)+p.cash[sector])*pct.ix[commons.date_index_external[dix],'ticker']/self.m.get_closing_price(ticker,dix))
        if actualVol>=targetVol:
            return commons.action_code['sell'],(actualVol-targetVol)
        else:
            return commons.action_code['buy'],(targetVol-actualVol)

    def get_order_price(self,dba,ticker,state,dix,action,closing_price):
        clusters=dba.t_clusters.read_where('(ticker=='+"'"+str(ticker)+"'"+")"+' & (kpi=='+"'"+'_clr'+"')")
        pct_change=0
        if self.trading_strategy=='aggressiv':
            if action==commons.action_code['buy']:
                for row in clusters:
                    for i in range(4,-1,-1):
                        x=state['clr_cluster_'+str(i)]+2
                        if x==3:
                            if i!=0:
                                pct_change=(row['c'+str(i)] + row['c'+str(i-1)])/2.
                            else:
                                pct_change=row['c'+str(i)]
            if action==commons.action_code['sell']:                                           
                clusters=dba.t_clusters.read_where('(train_uuid=='+"'"+self.train_uuid+"')"+' & (ticker=='+"'"+str(ticker)+"'"+")"+' & (kpi=='+"'"+'_chr'+"')")
                for row in clusters:
                    for i in range(0,5):
                        x=state['clr_cluster_'+str(i)]+2
                        if x==3:
                            if i!=4:
                                pct_change=(row['c'+str(i)] + row['c'+str(i+1)])/2.
                            else:
                                pct_change=row['c'+str(i)]                                           
        if closing_price==0:
            print 'Ticker: ',ticker,' Date:',dix,commons.date_index_external[dix],' price==zero!'                                      
        expected_price=closing_price+closing_price*pct_change/100.
        
#if not executable then price should be the closing price
        next_dix=dix+1
        closing_next_day=self.m.get_closing_price(ticker,next_dix)
        low_next_day=self.m.get_low_price(ticker,next_dix)
        high_next_day=self.m.get_high_price(ticker,next_dix)        
        if action==commons.action_code['buy'] and expected_price<low_next_day:
            expected_price=closing_next_day*1.001
        if action==commons.action_code['sell'] and expected_price>high_next_day:
            expected_price=closing_next_day*.999
            
        return expected_price
        
#based on date's feature values predicts the value of label y        
    def get_forecast_state(self,t_stats,ticker,dix):
#        print 'Forecasting for: ',ticker,commons.date_index_external[dix]
        Xy_all=pd.read_hdf(commons.data_path+'Xy_all_'+str(ticker),'table')
        select_columns=commons.Xy_columns(Xy_all,'Close')
        Xy=Xy_all.ix[commons.date_index_external[dix],select_columns]
        X=Xy[:-8]
        X_t=X
        del Xy_all
        state=dict()
        for y in commons.y_labels:
            model=self.get_best_model(t_stats,ticker,y,dix)
#            print model[3]
            if len(model[0])!=0 or len(model[1])!=0 or len(model[2])!=0:
                model_key=model[0]
                X_t=self.transform_X(ticker,X,model[3])
                try:
                    clf=joblib.load(commons.model_path+model_key+'.pkl')
                except IOError:
#                    print 'using model w/o pca no'
                    model_key=model[1]
                    try:
                        clf=joblib.load(commons.model_path+model_key+'.pkl')
                    except IOError:
                        model_key=model[2]
                        X_t=self.transform_X(ticker,X,model[4])
                        clf=joblib.load(commons.model_path+model_key+'.pkl')
                    
                state[y]=clf.predict(X_t.reshape(1,-1))
                state[y]=float(state[y])
            else:
                state[y]=0
            if int(state[y])==state[y]:
                state[y]=int(state[y])
        return state      
        
       
    def get_best_model(self,t_stats,ticker,label,dix):
        sp500_ticker=commons.getHistSp500Ticker(commons.date_index_external[dix])
        model1,model2,generic_model='','',''
        pca,generic_pca=0,0
        models=t_stats.read_where('(ticker=='+"'"+str(ticker)+"')"+' & (kpi=='+"'"+str(label)+"')")
        ticker_accuracy=0
        for row in models:
            if str(self.train_uuid)==row['train_uuid']:
                if row['accuracy']>ticker_accuracy:
                    ticker_accuracy=row['accuracy']
                    model1=str(row['pca'])+'_'+row['model']+'_'+ticker+'_'+label
                    model2=row['model']+'_'+ticker+'_'+label
                    pca=row['pca']


        models=t_stats.read_where('(ticker=='+"'"+sp500_ticker[ticker]+"')"+' & (kpi=='+"'"+str(label)+"')")
        general_accuracy=0
        for row in models:
            if row['accuracy']>general_accuracy and str(self.train_uuid)==row['train_uuid']:
                general_accuracy=row['accuracy']
                generic_model=str(row['pca'])+'generic_'+row['model']+'_'+sp500_ticker[ticker]+label
                generic_pca=row['pca']

        return model1, model2, generic_model,pca,generic_pca,round(ticker_accuracy,2),round(general_accuracy,2)
                
    def transform_X(self,ticker,X,pca):
        if pca!=0:
            pca=joblib.load(commons.model_path+str(pca)+'_PCA_'+ticker+'.pkl')
            X=pca.transform(X.reshape(1,-1))
        return X
        
    def recommendation_followed(self,state,proposed_action,reward):
        if state['1dd_Close']==1 and proposed_action==commons.action_code['sell']:
            reward-=100
        if state['1dd_Close']==-1 and proposed_action==commons.action_code['buy']:
            reward-=100        
        return reward