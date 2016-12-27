import commons
import numpy as np
import pandas as pd
from database import db
from sklearn.cluster import KMeans
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import mean_squared_error
from sklearn import tree
from sklearn.decomposition import PCA
import uuid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib


class cl_trainSection(object):
    stats_accuracy=dict()
    stats_model=dict()
    stats_kf=dict()    
    scenario=None
    
    def __init__(self,cutoffdix,train_uuid,scenario,trainAll=True):
        self.cutoffdix=cutoffdix
        self.train_uuid=train_uuid
        self.dba=db(self.train_uuid,'r+')
        self.scenario=scenario
        self.trainAll=trainAll

    def predict_labels(self,clf,X,y):
        y_pred = clf.predict(X)
        return accuracy_score(y, y_pred)   

    #log is needed to be able to select the best model during forecasting
    def write_log(self,pca,ticker,model,y,kf):
        self.dba.t_stats.row['train_uuid']=self.train_uuid
        self.dba.t_stats.row['pca']=int(pca)
        self.dba.t_stats.row['ticker']=ticker
        self.dba.t_stats.row['model']=model
        self.dba.t_stats.row['kpi']=y
        self.dba.t_stats.row['accuracy']=kf
        self.dba.t_stats.row.append()
        self.dba.t_stats.flush()

#this logs the best parameters. No really needed, as the best performing configuration is
#being stored as pickle object
    def write_parameters(self,pca,ticker,model,kpi,parameters):
        self.dba.t_parameter.row['train_uuid']=self.train_uuid
        self.dba.t_parameter.row['pca']=int(pca)
        self.dba.t_parameter.row['ticker']=ticker
        self.dba.t_parameter.row['model']=model
        self.dba.t_parameter.row['kpi']=kpi
        
        if model=='SVC':
            self.dba.t_parameter.row['kernel']=parameters['kernel']
            self.dba.t_parameter.row['C']=parameters['C']
        if model=='RF' or model=='DT':
            self.dba.t_parameter.row['max_depth']=parameters['max_depth']
        if model=='kN':
            self.dba.t_parameter.row['n_neighbors']=parameters['n_neighbors']
            self.dba.t_parameter.row['weights']=parameters['weights']
            self.dba.t_parameter.row['algorithm']=parameters['algorithm']
        self.dba.t_parameter.row.append()
        self.dba.t_parameter.flush()
        self.dba.db_main.flush()
    
#the values to the label price change close-to-low and close-to-high are being clustered and the
#label is afterwards whether the expected prince change will be in a certain cluster
    def write_cluster(self,pca,ticker,kf,cluster):
        self.dba.t_clusters.row['train_uuid']=self.train_uuid
        self.dba.t_clusters.row['ticker']=ticker
        self.dba.t_clusters.row['kpi']=kf
        a=cluster[0]
        a.sort()
        self.dba.t_clusters.row['c0']=a[0]
        self.dba.t_clusters.row['c1']=a[1]
        self.dba.t_clusters.row['c2']=a[2]
        self.dba.t_clusters.row['c3']=a[3]
        self.dba.t_clusters.row['c4']=a[4]
        self.dba.t_clusters.row.append()
        self.dba.t_clusters.flush()
    
#this is only being used to check whether we already know the model. If so then 
#the training is being skipped
    def get_parameters(self,pca,ticker,model,kpi):
        q_records=self.dba.t_parameter.where('(train_uuid=='+"'"+self.train_uuid+"'"+') & (pca=='+str(pca)+') & (ticker=='+"'"+ticker+"'"+') & (model=='+"'"+model+"'"+') & (kpi=='+"'"+kpi+"'"+')')
        return any(q_records)

    def getXy(self,mode,modes,ticker,dates,lPca):
        #select relevant columns from Xy_all
#        startdix=self.cutoffdix-500
#        Xy_all=commons.read_dataframe(commons.data_path+'Xy_all_'+ticker).ix[commons.date_index_external[startdix]:commons.date_index_external[self.cutoffdix+1],:]
        Xy_all=commons.read_dataframe(commons.data_path+'Xy_all_'+ticker).ix[:commons.date_index_external[self.cutoffdix+1],:]
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
        Xy_all1=pd.DataFrame()
        for date in dates:
            Xy_all2=pd.DataFrame()
            Xy_all2=Xy_all.ix[date[0]:date[1],select_columns]
            Xy_all1=pd.concat([Xy_all1,Xy_all2])
        Xy_all=Xy_all1
        X_all=Xy_all.ix[:,:-8]

        #reduce dimension space?
        if len(X_all.index)>250:
            if lPca!=0:
                pca=PCA(n_components=lPca)
                pca=pca.fit(X_all)
                X_all=pd.DataFrame(data=pca.transform(X_all),index=X_all.index)
                joblib.dump(pca,commons.model_path+str(lPca)+'_PCA_'+ticker+'.pkl',compress=3)
                del pca

        #get labels and drop the % forecast, as not relevant for the investment decision
        y_all=Xy_all.ix[:,-8:]
        y_all=y_all.drop(['1dr_Close', '5dr_Close', '20dr_Close'],1)
        
        return X_all,y_all
        
#prepare y
    def prepY(self,y_all,lPca,ticker):
        #prep y; tranform the %expectation from the Xy_all to cluster assignment
        for y in y_all.columns:
            if y=='clr' or y=='chr':
                np_overall=dict()
                np_overall[0]=list()    
                np_overall[1]=list()    
                np_overall[2]=list()    
                np_overall[3]=list()    
                np_overall[4]=list()                  
                kmeans=KMeans(n_clusters=5, random_state=0).fit(y_all.ix[:,[y]])
                self.write_cluster(lPca,ticker,y,kmeans.cluster_centers_.reshape(1,-1))                
                for x in y_all[y].values:
                    l_i1=0
                    distance=100.
                    kmeans.cluster_centers_.reshape(1,-1)[0].sort()
                    for c in kmeans.cluster_centers_.reshape(1,-1)[0]:
                        if distance>abs((x-c)):
                            distance=x-c
                            cluster=l_i1
                        l_i1+=1
                    for i in range(0,5):
                        if i==cluster:
                            np_overall[i].append(1)
                        else:
                            np_overall[i].append(0)
                            
                for i in range(0,5):
                    if np_overall[i].count(1)>=3 and np_overall[i].count(0)>=3:
                        np1=np.array(np_overall[i])
                        df1=pd.DataFrame(data=np1.reshape(-1,1),index=y_all.index,columns=[y+'_cluster_'+str(i)])
                        y_all=y_all.join(df1,how='outer')
        
        y_all=y_all.drop(['chr','clr'],1)     
        return y_all
        
#the actual training
    def train(self,mode='Close'):   
        #remove all prices that are not relevant    
        modes=list(['Open','Close','Low','High'])
        modes.remove(mode) 
        
        l_i=505     
        for x_pca in range(8,9):
            if x_pca==8:
                l_pca=0
            else:
                l_pca=x_pca
            for k,dates in commons.sp500CompDates.items():
                for date in dates:
                    if date[0]<=commons.date_index_external[self.cutoffdix] and\
                        date[1]>=commons.date_index_external[self.cutoffdix]:
                        print 'Ticker:',str(k),', l_i=',str(l_i)
                        l_i-=1
            
                        Xy_all=self.getXy(mode,modes,k,dates,l_pca)
                        if len(Xy_all[0].index)>250:
                            X_all=Xy_all[0]
                            y_all=Xy_all[1]
                            y_all=self.prepY(y_all,l_pca,k)
                    
                            #train        
                            scorer = make_scorer(mean_squared_error)            
                    
                            for y in y_all.columns:
        #                        print '-'+str(y)
                                #get training data
                                X_train, X_test, y_train, y_test = train_test_split(X_all, y_all[y], train_size = .7, random_state = 1) 
                                #SVC                            
                                parameters=dict()
                                if (self.get_parameters(l_pca,k,'SVC',y)):
                                   #print 'Trained already.'
                                   a=1
                                else:
                                    clf = SVC()                                                                                
                                    parameters = {'kernel':['linear', 'poly', 'rbf', 'sigmoid'], 'C':[1,10,100]}
                                    grid_obj = GridSearchCV(clf, parameters, scoring = scorer, n_jobs=1)
                                    grid_obj = grid_obj.fit(X_all, y_all[y])
                                    clf = grid_obj.best_estimator_
                                    self.stats_kf[y]=self.predict_labels(clf, X_test, y_test)
                                    self.write_parameters(l_pca,k,'SVC',y,grid_obj.best_params_)
                                    self.write_log(l_pca,k,'SVC',y,self.stats_kf[y])
                                    self.stats_model['SVC']=self.stats_kf
                                    joblib.dump(clf,commons.model_path+str(l_pca)+'_SVC_'+str(k)+'_'+str(y)+'.pkl',compress=3)
                                    del grid_obj,clf
                                    self.dba.t_parameter.flush()
                                if self.trainAll:
                        #        #Random Forest
                                    parameters=dict()
                                    if (self.get_parameters(l_pca,k,'RF',y)):
                                       #print 'Trained already.'
                                       a=1
                                    else:                
                                        parameters = {'max_depth':(1,2,3,4,5,6,7,8,9)}
                                        clf=RandomForestClassifier()
                                        grid_obj = GridSearchCV(clf,parameters,scoring = scorer)
                                        grid_obj = grid_obj.fit(X_train, y_train)
                                        clf=grid_obj.best_estimator_
                                        self.stats_kf[y]=self.predict_labels(clf, X_test, y_test)
                                        self.write_parameters(l_pca,k,'RF',y,grid_obj.best_params_)                
                                        self.write_log(l_pca,k,'RF',y,self.stats_kf[y])                
                                        joblib.dump(clf,commons.model_path+str(l_pca)+'_RF_'+str(k)+'_'+str(y)+'.pkl',compress=3)
                                        del clf
                                        self.stats_model['RF']=self.stats_kf
                                        self.dba.t_parameter.flush()
                        
                        #        #Decision Tree                
                                    parameters=dict()
                                    if (self.get_parameters(l_pca,k,'DT',y)):
                                       #print 'Trained already.'
                                       a=1
                                    else:
                                        parameters = {'max_depth':(1,2,3,4,5,6,7,8,9)}
                                        clf=tree.DecisionTreeClassifier()
                                        grid_obj = GridSearchCV(clf,parameters,scoring = scorer)
                                        grid_obj = grid_obj.fit(X_train, y_train)
                                        clf=grid_obj.best_estimator_
                                        self.stats_kf[y]=self.predict_labels(clf, X_test, y_test)
                                        self.write_parameters(l_pca,k,'DT',y,grid_obj.best_params_)
                                        self.write_log(l_pca,k,'DT',y,self.stats_kf[y])                
                                        joblib.dump(clf,commons.model_path+str(l_pca)+'_DT_'+str(k)+'_'+str(y)+'.pkl',compress=3)
                                        del clf
                                        self.stats_model['DT']=self.stats_kf
                                        self.dba.t_parameter.flush()
                        
                        #        #AdaBoost
                                    parameters=dict()
                                    if (self.get_parameters(l_pca,k,'AB',y)):
                                       #print 'Trained already.'
                                       a=1
                                    else:
                                        clf=AdaBoostClassifier(n_estimators=100)
                                        clf=clf.fit(X_train, y_train)
                                        self.stats_kf[y]=self.predict_labels(clf, X_test, y_test)
                                        self.write_log(l_pca,k,'AB',y,self.stats_kf[y])          
                                        self.write_parameters(l_pca,k,'AB',y,dict())                    
                                        joblib.dump(clf,commons.model_path+str(l_pca)+'_AB_'+str(k)+'_'+str(y)+'.pkl',compress=3)
                                        del clf        
                                        self.stats_model['AB']=self.stats_kf      
                                        self.dba.t_parameter.flush()
                        
                        #        #kNeighbors
                                    parameters=dict()
                                    if (self.get_parameters(l_pca,k,'kN',y)):
                                       #print 'Trained already.'
                                       a=1
                                    else:
                                        parameters = {'n_neighbors':(3,4,5,6,7,8),'weights':('uniform','distance'),'algorithm':('auto', 'ball_tree', 'kd_tree', 'brute')}
                                        clf=KNeighborsClassifier()
                                        grid_obj = GridSearchCV(clf,parameters,scoring = scorer)
                                        grid_obj = grid_obj.fit(X_train, y_train)
                                        clf=grid_obj.best_estimator_
                                        self.stats_kf[y]=self.predict_labels(clf, X_test, y_test)
                                        self.write_parameters(l_pca,k,'kN',y,grid_obj.best_params_)
                                        self.write_log(l_pca,k,'kN',y,self.stats_kf[y])                
                                        joblib.dump(clf,commons.model_path+str(l_pca)+'_kN_'+str(k)+'_'+str(y)+'.pkl',compress=3)
                                        del clf 
                                        self.stats_model['kN']=self.stats_kf
                                        self.dba.t_parameter.flush()
                        
                        #        #GaussianNB
                                    parameters=dict()
                                    if (self.get_parameters(l_pca,k,'GNB',y)):
                                       #print 'Trained already.'
                                       a=1
                                    else:
                                        clf=GaussianNB()
                                        clf=clf.fit(X_train, y_train)
                                        self.stats_kf[y]=self.predict_labels(clf, X_test, y_test)
                                        self.write_log(l_pca,k,'GNB',y,self.stats_kf[y])                
                                        self.write_parameters(l_pca,k,'GNB',y,dict(),)                    
                                        joblib.dump(clf,commons.model_path+str(l_pca)+'_GNB_'+str(k)+'_'+str(y)+'.pkl',compress=3)
                                        self.stats_model['GNB']=self.stats_kf
                                        self.dba.t_parameter.flush()
                            self.stats_accuracy[k]=self.stats_model
                        else:
                            print 'Insufficient data for',k,'. Trade will align with the index.'
                            self.dba.noTrade.row['ticker']=k
                            self.dba.noTrade.row['dix']=self.cutoffdix
                            self.dba.noTrade.row.append()
                            self.dba.noTrade.flush()
                    else:
                        print 'No trade for',k
