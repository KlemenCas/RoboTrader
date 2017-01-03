import commons
from sklearn.externals import joblib
import numpy as np


class cl_runQ(object):
    indexComposition=dict()
    train_uuid=None
    dba=None
    
    def __init__(self,dba):
        self.dba=dba
        self.setLastTrainUuid()
        self.initializeIndex()
        self.data_sp500=commons.read_dataframe(commons.data_path+'WIKI_SP500.h5')
    
    def setLastTrainUuid(self):
        dix=0
        for row in self.dba.t_train_h:
            if row['dateint']>=dix:
                self.train_uuid=row['train_uuid']

    #based on date's feature values predicts the value of label y        
    def get_forecast_state(self,ticker,dix):
#        print 'Forecasting for: ',ticker,commons.date_index_external[dix]
        Xy_all=commons.read_dataframe(commons.data_path+'Xy_all_'+str(ticker))
        select_columns=commons.Xy_columns(Xy_all,'Close')
        Xy=Xy_all.ix[commons.date_index_external[dix],select_columns]
        X=Xy[:-9]
        X_t=X
        del Xy_all
        state=dict()
        for y in commons.y_labels:
            model=self.get_best_model(ticker,y,dix)
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
        
    #best model based on the stats from training
    def get_best_model(self,ticker,label,dix):
        sp500_ticker=commons.getHistSp500Ticker(commons.date_index_external[dix])
        model1,model2,generic_model='','',''
        pca,generic_pca=0,0
        models=self.dba.t_stats.read_where('(ticker=='+"'"+str(ticker)+"')"+' & (kpi=='+"'"+str(label)+"')")
        ticker_accuracy=0
        for row in models:
            if str(self.train_uuid)==row['train_uuid']:
                if row['accuracy']>ticker_accuracy:
                    ticker_accuracy=row['accuracy']
                    model1=str(row['pca'])+'_'+row['model']+'_'+ticker+'_'+label
                    model2=row['model']+'_'+ticker+'_'+label
                    pca=row['pca']


        models=self.dba.t_stats.read_where('(ticker=='+"'"+sp500_ticker[ticker]+"')"+' & (kpi=='+"'"+str(label)+"')")
        general_accuracy=0
        for row in models:
            if row['accuracy']>general_accuracy and str(self.train_uuid)==row['train_uuid']:
                general_accuracy=row['accuracy']
                generic_model=str(row['pca'])+'generic_'+row['model']+'_'+sp500_ticker[ticker]+label
                generic_pca=row['pca']

        return model1, model2, generic_model,pca,generic_pca,round(ticker_accuracy,2),round(general_accuracy,2)
        
    #if there was PCA, transform
    def transform_X(self,ticker,X,pca):
        if pca!=0:
            pca=joblib.load(commons.model_path+str(pca)+'_PCA_'+ticker+'.pkl')
            X=pca.transform(X.reshape(1,-1))
        return X        
        
    #best action
    def get_best_action(self,state):
        if state['1dd_Close']==1:
            return commons.action_code['buy']
        elif state['1dd_Close']==0:
            return commons.action_code['hold']
        elif state['1dd_Close']==-1:
            return commons.action_code['sell']        
            
    #order price, based on action and clusters
    def get_order_price(self,ticker,state,dix,action):
        closing_price=self.get_closing_price(ticker,dix)
        clusters=self.dba.t_clusters.read_where('(ticker=='+"'"+str(ticker)+"'"+")"+' & (kpi=='+"'"+'clr'+"')")
        pct_change=0
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
            clusters=self.dba.t_clusters.read_where('(ticker=='+"'"+str(ticker)+"'"+")"+' & (kpi=='+"'"+'chr'+"')")
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
        
        return expected_price            
        
    #closing price        
    def get_closing_price(self,ticker,dix):
        price = self.data_sp500.ix[commons.date_index_external[dix],[ticker+'_Close']]
        if np.isnan(price.values):
            try:
                price = self.data_sp500.ix[commons.date_index_external[dix],[commons.alternative_symbol[ticker]+'_Close']]
                price=float(price.values)
            except KeyError:
                price=0
                print 'Ticker: ',ticker, 'Date: ', commons.date_index_external[dix],' set to zero.'
        else:
            price=float(price.values)
        return price        
    
    #initialize index portfolio        
    def initializeIndex(self):
        for k,v in commons.getIndexCodes().items():
            index_t=v[-8:]    
            self.indexComposition[index_t]=commons.read_dataframe(commons.data_path+'PCT_'+index_t+'.h5')
    
    #index portfolio for target %
    def get_index_portfolio(self,index_t,dix):
        portfolio=dict()
        for t in commons.getHistSp500Composition(commons.date_index_external[dix])[index_t]:
            if dix>=commons.date_index_internal[commons.data_sp500_1st_date[t]]: 
                portfolio[t]=self.indexComposition[index_t].ix[commons.date_index_external[dix],t]
        return portfolio            
        
    def trained(self,train_uuid,ticker):
        found=False
        stats=self.dba.t_stats.read_where('(ticker=='+"'"+str(ticker)+"'"+")"+' & (kpi=='+"'"+'1dd_Close'+"')")
        for row in stats:
            if str(self.train_uuid)==row['train_uuid']:
                found=True
                break
        return found        