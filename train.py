import commons
import numpy as np
import pandas as pd
import datetime as dt
from database import db
from forecasts import forecast
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

#globals locally
sp500_index=commons.sp500_index

#persistence
from sklearn.externals import joblib

accuracy_results=pd.DataFrame()
data_sp500=pd.read_hdf(commons.data_path+'WIKI_SP500.h5','table')        

def predict_labels(clf,X,y):
    y_pred = clf.predict(X)
    return accuracy_score(y, y_pred)   

#log is needed to be able to select the best model during forecasting
def write_log(pca,ticker,model,y,kf):
    dba.t_stats.row['train_uuid']=train_uuid
    dba.t_stats.row['pca']=int(pca)
    dba.t_stats.row['ticker']=ticker
    dba.t_stats.row['model']=model
    dba.t_stats.row['kpi']=y
    dba.t_stats.row['accuracy']=kf
    dba.t_stats.row.append()
    dba.t_stats.flush()

#this logs the best parameters. No really needed, as the best performing configuration is
#being stored as pickle object
def write_parameters(pca,ticker,model,kpi,parameters):
    dba.t_parameter.row['train_uuid']=train_uuid
    dba.t_parameter.row['pca']=int(pca)
    dba.t_parameter.row['ticker']=ticker
    dba.t_parameter.row['model']=model
    dba.t_parameter.row['kpi']=kpi
    
    if model=='SVC':
        dba.t_parameter.row['kernel']=parameters['kernel']
        dba.t_parameter.row['C']=parameters['C']
    if model=='RF' or model=='DT':
        dba.t_parameter.row['max_depth']=parameters['max_depth']
    if model=='kN':
        dba.t_parameter.row['n_neighbors']=parameters['n_neighbors']
        dba.t_parameter.row['weights']=parameters['weights']
        dba.t_parameter.row['algorithm']=parameters['algorithm']
    dba.t_parameter.row.append()
    dba.t_parameter.flush()
    
#the values to the label price change close-to-low and close-to-high are being clustered and the
#label is afterwards whether the expected prince change will be in a certain cluster
def write_cluster(pca,ticker,kf,cluster):
    dba.t_clusters.row['train_uuid']=train_uuid
    dba.t_clusters.row['ticker']=ticker
    dba.t_clusters.row['kpi']=kf
    a=cluster[0]
    a.sort()
    dba.t_clusters.row['c0']=a[0]
    dba.t_clusters.row['c1']=a[1]
    dba.t_clusters.row['c2']=a[2]
    dba.t_clusters.row['c3']=a[3]
    dba.t_clusters.row['c4']=a[4]
    dba.t_clusters.row.append()
    dba.t_clusters.flush()
    
#this is only being used to check whether we already know the model. If so then 
#the training is being skipped
def get_parameters(pca,ticker,model,kpi):
    q_records=dba.t_parameter.where('(train_uuid=='+"'"+train_uuid+"'"+') & (pca=='+str(pca)+') & (ticker=='+"'"+ticker+"'"+') & (model=='+"'"+model+"'"+') & (kpi=='+"'"+kpi+"'"+')')
    return any(q_records)

#the actual training
def train(mode):
    stats_accuracy=dict()
    stats_model=dict()
    stats_kf=dict()

    #remove all prices that are not relevant    
    modes=list(['Open','Close','Low','High'])
    modes.remove(mode) 
    
    l_i=505     
    for x_pca in range(8,11):
        if x_pca==8:
            l_pca=0
        else:
            l_pca=x_pca
        for k,dates in commons.sp500CompDates.items():
            print 'Ticker:',str(k),', l_i=',str(l_i)
            l_i-=1

            #select relevant columns from Xy_all
            Xy_all=pd.read_hdf(commons.data_path+'Xy_all_'+str(k),'table')
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
            if l_pca!=0:
                pca=PCA(n_components=l_pca)
                pca=pca.fit(X_all)
                X_all=pca.transform(X_all)
                del pca
    
            #get labels and drop the % forecast, as not relevant for the investment decision
            y_all=Xy_all.ix[:,-8:]
            y_all=y_all.drop(['1dr_Close', '5dr_Close', '20dr_Close'],1)
            del Xy_all
            
            #prep y; tranform the %expectation from the Xy_all to cluster assignment
            for y in y_all.columns:
                if y=='_clr' or y=='_chr':
                    np_overall=dict()
                    np_overall[0]=list()    
                    np_overall[1]=list()    
                    np_overall[2]=list()    
                    np_overall[3]=list()    
                    np_overall[4]=list()                  
                    kmeans=KMeans(n_clusters=5, random_state=0).fit(y_all.ix[:,[y]])
                    write_cluster(l_pca,k,y,kmeans.cluster_centers_.reshape(1,-1))                
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
                            df1=pd.DataFrame(data=np1.reshape(-1,1),index=y_all.index,columns=[y[1:]+'_cluster_'+str(i)])
                            y_all=y_all.join(df1,how='outer')
            
            y_all=y_all.drop(['_chr','_clr'],1)
    
            #train        
            scorer = make_scorer(mean_squared_error)            
    
            for y in y_all.columns:
                print '-'+str(y)
                #get training data
                X_train, X_test, y_train, y_test = train_test_split(X_all, y_all[y], train_size = .7, random_state = 1) 
                #SVC                            
                parameters=dict()
                if (get_parameters(l_pca,k,'SVC',y)):
                   print 'Trained already.'                        
                else:
                    clf = SVC()                                                                                
                    parameters = {'kernel':['linear', 'poly', 'rbf', 'sigmoid'], 'C':[1,10,100]}
                    grid_obj = GridSearchCV(clf, parameters, scoring = scorer, n_jobs=1)
                    grid_obj = grid_obj.fit(X_all, y_all[y])
                    clf = grid_obj.best_estimator_
                    stats_kf[y]=predict_labels(clf, X_test, y_test)
                    write_parameters(l_pca,k,'SVC',y,grid_obj.best_params_)
                    write_log(l_pca,k,'SVC',y,stats_kf[y])
                    stats_model['SVC']=stats_kf
                    joblib.dump(clf,commons.model_path+str(l_pca)+'_SVC_'+str(k)+'_'+str(y)+'.pkl',compress=3)
                    del grid_obj,clf
                    dba.t_parameter.flush()
    
    #        #Random Forest
                parameters=dict()
                if (get_parameters(l_pca,k,'RF',y)):
                   print 'Trained already.'                        
                else:                
                    parameters = {'max_depth':(1,2,3,4,5,6,7,8,9)}
                    clf=RandomForestClassifier()
                    grid_obj = GridSearchCV(clf,parameters,scoring = scorer)
                    grid_obj = grid_obj.fit(X_train, y_train)
                    clf=grid_obj.best_estimator_
                    stats_kf[y]=predict_labels(clf, X_test, y_test)
                    write_parameters(l_pca,k,'RF',y,grid_obj.best_params_)                
                    write_log(l_pca,k,'RF',y,stats_kf[y])                
                    joblib.dump(clf,commons.model_path+str(l_pca)+'_RF_'+str(k)+'_'+str(y)+'.pkl',compress=3)
                    del clf
                    stats_model['RF']=stats_kf
                    dba.t_parameter.flush()
    
    #        #Decision Tree                
                parameters=dict()
                if (get_parameters(l_pca,k,'DT',y)):
                   print 'Trained already.'                        
                else:
                    parameters = {'max_depth':(1,2,3,4,5,6,7,8,9)}
                    clf=tree.DecisionTreeClassifier()
                    grid_obj = GridSearchCV(clf,parameters,scoring = scorer)
                    grid_obj = grid_obj.fit(X_train, y_train)
                    clf=grid_obj.best_estimator_
                    stats_kf[y]=predict_labels(clf, X_test, y_test)
                    write_parameters(l_pca,k,'DT',y,grid_obj.best_params_)
                    write_log(l_pca,k,'DT',y,stats_kf[y])                
                    joblib.dump(clf,commons.model_path+str(l_pca)+'_DT_'+str(k)+'_'+str(y)+'.pkl',compress=3)
                    del clf
                    stats_model['DT']=stats_kf
                    dba.t_parameter.flush()
    
    #        #AdaBoost
                parameters=dict()
                if (get_parameters(l_pca,k,'AB',y)):
                   print 'Trained already.'                        
                else:
                    clf=AdaBoostClassifier(n_estimators=100)
                    clf=clf.fit(X_train, y_train)
                    stats_kf[y]=predict_labels(clf, X_test, y_test)
                    write_log(l_pca,k,'AB',y,stats_kf[y])          
                    write_parameters(l_pca,k,'AB',y,dict())                    
                    joblib.dump(clf,commons.model_path+str(l_pca)+'_AB_'+str(k)+'_'+str(y)+'.pkl',compress=3)
                    del clf        
                    stats_model['AB']=stats_kf      
                    dba.t_parameter.flush()
    
    #        #kNeighbors
                parameters=dict()
                if (get_parameters(l_pca,k,'kN',y)):
                   print 'Trained already.'                        
                else:
                    parameters = {'n_neighbors':(3,4,5,6,7,8),'weights':('uniform','distance'),'algorithm':('auto', 'ball_tree', 'kd_tree', 'brute')}
                    clf=KNeighborsClassifier()
                    grid_obj = GridSearchCV(clf,parameters,scoring = scorer)
                    grid_obj = grid_obj.fit(X_train, y_train)
                    clf=grid_obj.best_estimator_
                    stats_kf[y]=predict_labels(clf, X_test, y_test)
                    write_parameters(l_pca,k,'kN',y,grid_obj.best_params_)
                    write_log(l_pca,k,'kN',y,stats_kf[y])                
                    joblib.dump(clf,commons.model_path+str(l_pca)+'_kN_'+str(k)+'_'+str(y)+'.pkl',compress=3)
                    del clf 
                    stats_model['kN']=stats_kf
                    dba.t_parameter.flush()
    
    #        #GaussianNB
                parameters=dict()
                if (get_parameters(l_pca,k,'GNB',y)):
                   print 'Trained already.'                        
                else:
                    clf=GaussianNB()
                    clf=clf.fit(X_train, y_train)
                    stats_kf[y]=predict_labels(clf, X_test, y_test)
                    write_log(l_pca,k,'GNB',y,stats_kf[y])                
                    write_parameters(l_pca,k,'GNB',y,dict(),)                    
                    joblib.dump(clf,commons.model_path+str(l_pca)+'_GNB_'+str(k)+'_'+str(y)+'.pkl',compress=3)
                    stats_model['GNB']=stats_kf
                    dba.t_parameter.flush()
            stats_accuracy[k]=stats_model
            del X_all, y_all, X_train, y_train, X_test, y_test
            

#commons.get_sp500_index()
#train_uuid=uuid.uuid1().hex
train_uuid='057fa921c16c11e6a6f5985fd3e5919f'
dba=db(train_uuid,'r+')
#dba.new_training(train_uuid,commons.date_index_internal[commons.max_date['WIKI_SP500']])
train('Close')       
