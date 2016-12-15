import commons
import numpy as np
import pandas as pd
import datetime as dt
import csv
from sklearn.cluster import KMeans
import time

from scipy.stats import kendalltau
import seaborn as sns

#model
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import mean_squared_error
from sklearn import tree
from sklearn.decomposition import PCA

from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#globals locally
local_path=commons.local_path
sp500_ticker=commons.sp500_ticker
sp500_index=commons.sp500_index
sp500_composition=commons.sp500_composition

#persistence
from sklearn.externals import joblib

accuracy_results=pd.DataFrame()
data_sp500=pd.read_hdf(commons.local_path+'data/WIKI_SP500.h5','table')        

#training & testing data
SP500_25=np.array([])
SP500_30=np.array([])
SP500_10=np.array([])
SP500_40=np.array([])
SP500_35=np.array([])
SP500_20=np.array([])
SP500_45=np.array([])
SP500_15=np.array([])
SP500_50=np.array([])
SP500_55=np.array([])

def predict_labels(clf,X,y):
    y_pred = clf.predict(X)
    return accuracy_score(y, y_pred)   

def write_log(pca,index,model,y,kf):
    row=list()
    row.append(index)
    row.append(pca)
    row.append(model)
    row.append(y)
    row.append(kf)
    f=open('C:/Users/kncas/Documents/Python Scripts/stats/stats_generic.csv', 'a')
    writer = csv.writer(f, delimiter=',')
    writer.writerow(row)
    f.close
    
def write_parameters(pca,index,model,kpi,parameters):
    log=dict()
    log['pca']=pca
    log['index']=index
    log['model']=model
    log['kpi']=kpi
    log['kernel']=''
    log['C']=0
    log['max_depth']=0
    log['n_neighbors']=0
    log['weights']=''
    log['algorithm']=''
    
    if model=='SVC':
        log['kernel']=parameters['kernel']
        log['C']=parameters['C']
    if model=='RF' or model=='DT':
        log['max_depth']=parameters['max_depth']
    if model=='kN':
        log['n_neighbors']=parameters['n_neighbors']
        log['weights']=parameters['weights']
        log['algorithm']=parameters['algorithm']

    fieldnames=['pca','index','model','kpi','kernel','C','max_depth','n_neighbors','weights','algorithm']

    f=open('C:/Users/kncas/Documents/Python Scripts/stats/parameters_generic.csv', 'a')
    writer = csv.DictWriter(f,fieldnames=fieldnames)
    writer.writerow(log)
    f.close    

def write_cluster(pca,index,kf,cluster):
    row=list()
    row.append(index)
    row.append(kf)
    row.append(cluster)
    f=open('C:/Users/kncas/Documents/Python Scripts/stats/cluster_generic.csv', 'a')
    writer = csv.writer(f, delimiter=',')
    writer.writerow(row)
    f.close

def get_parameters(pca,index,model,kpi):
    fieldnames=['pca','index','model','kpi','kernel','C']
    f=open('C:/Users/kncas/Documents/Python Scripts/stats/parameters_generic.csv', 'r')
    reader = csv.DictReader(f,fieldnames=fieldnames)
    parameters=dict()
    for row in reader:
        if row['pca']==str(pca) and row['index']==index and row['model']==model and row['kpi']==kpi:
            parameters=row
    f.close        
    return parameters
    
def train(mode):
    modes=list(['Open','Close','Low','High'])
    modes.remove(mode)
    
    stats_accuracy=dict()
    stats_model=dict()
    stats_kf=dict()
    select_columns=list()
    
    for l_pca in range(7,11):             
        for k,v in sp500_composition.items():
            Xy_all=np.array([])
            for t in v:
                df1=pd.DataFrame()
                df1=pd.read_hdf(local_path+'data/Xy_all_'+str(t),'table')
                if not any(select_columns):
                    for c in df1.columns:
                        if mode in str(c):
                            select_columns.append(c)
                        else:
                            m_found=False
                            for m in modes:
                                if m in str(c):
                                    m_found=True
                            if m_found==False:
                                select_columns.append(c)
                startdate=min(df1.index)

                if startdate<dt.datetime.today()-dt.timedelta(days=1825):
                    startdate=dt.datetime.today()-dt.timedelta(days=1825)
                else:
                    startdate=min(df1.index)
                    
                Xy_all=np.append(Xy_all,df1.ix[startdate:,select_columns].values)

            index=commons.sp500_index[k][-10:-2]

#prepare X and y
            Xy_all=Xy_all.reshape(-1,24)
            X_all=Xy_all[:,:-8]
            #reduce space
            if l_pca>7:
                pca=PCA(n_components=l_pca)
                pca=pca.fit(X_all)
                X_all=pca.transform(X_all)
            #get labels and drop the % forecast, as not relevant for the investment decision
            y_all=Xy_all[:,-8:]
            y_columns=select_columns[-8:]
            l=list(['1dr_Close', '5dr_Close', '20dr_Close'])
            for x in l:
                idx=y_columns.index(x)
                y_all=np.delete(y_all,idx,1)
                y_columns.remove(x)
                
            del Xy_all        

            #prep y
            for y in y_columns:
                if y=='_clr' or y=='_chr':
                    np_overall=dict()
                    np_overall[0]=list()    
                    np_overall[1]=list()    
                    np_overall[2]=list()    
                    np_overall[3]=list()    
                    np_overall[4]=list()                  
                    kmeans=KMeans(n_clusters=5, random_state=0).fit(y_all[:,y_columns.index(y)].reshape(-1,1))
                    write_cluster(l_pca,index,y,kmeans.cluster_centers_.reshape(1,-1))                
                    for x in y_all[:,y_columns.index(y)]:
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
                            no_columns=y_all.shape[1]+1
                            y_all=np.insert(y_all,np1,1)
                            y_all=y_all.reshape(-1,no_columns)
                            y_columns.append(y[1:]+'_cluster_'+str(i))
            
            l=list(['_clr', '_chr'])
            for x in l:
                idx=y_columns.index(x)
                y_all=np.delete(y_all,idx,1)
                y_columns.remove(x)
            
            scorer = make_scorer(mean_squared_error)            
    #        #SVC            
            for y in y_columns:
                #get training data
                X_train, X_test, y_train, y_test = train_test_split(X_all, y_all[:,y_columns.index(y)], train_size = .7, random_state = 1) 
    
                parameters=dict()
                parameters=get_parameters(l_pca,index,'SVC',y)
                if any(parameters):
                    print 'Trained already.'                        
                else:
                    clf = SVC()                                                                                
                    parameters = {'kernel':('linear', 'poly', 'rbf', 'sigmoid'), 'C':[1,10,100]}
                    grid_obj = GridSearchCV(clf, parameters, scoring = scorer)
                    grid_obj = grid_obj.fit(X_all, y_all[:,y_columns.index(y)])
                    clf = grid_obj.best_estimator_
                    stats_kf[y]=predict_labels(clf, X_test, y_test)
                    write_parameters(l_pca,index,'SVC',y,grid_obj.best_params_)
                    write_log(l_pca,index,'SVC',y,stats_kf[y])
                    stats_model['SVC']=stats_kf
                    joblib.dump(clf,local_path+'models/'+str(l_pca)+'generic_SVC_'+str(index)+str(y)+'.pkl',compress=3)
                    del grid_obj,clf
                    
    #Random Forest
                parameters=get_parameters(l_pca,index,'RF',y)
                if any(parameters):
                    print 'Trained already.'                        
                else:    
                    parameters = {'max_depth':(1,2,3,4,5,6,7,8,9)}
                    clf=RandomForestClassifier()
                    grid_obj = GridSearchCV(clf,parameters,scoring = scorer)
                    grid_obj = grid_obj.fit(X_train, y_train)
                    clf=grid_obj.best_estimator_
                    stats_kf[y]=predict_labels(clf, X_test, y_test)
                    write_parameters(l_pca,index,'RF',y,grid_obj.best_params_)
                    write_log(l_pca,index,'RF',y,stats_kf[y])
                    joblib.dump(clf,local_path+'models/'+str(l_pca)+'generic_RF_'+str(index)+str(y)+'.pkl',compress=3)
                    stats_model['RF']=stats_kf
                    del clf
    
    #Decision Tree                
                parameters=get_parameters(l_pca,index,'DT',y)
                if any(parameters):
                    print 'Trained already.'                        
                else:    
                    parameters = {'max_depth':(1,2,3,4,5,6,7,8,9)}
                    clf=tree.DecisionTreeClassifier()
                    grid_obj = GridSearchCV(clf,parameters,scoring = scorer)
                    grid_obj = grid_obj.fit(X_train, y_train)
                    clf=grid_obj.best_estimator_
                    stats_kf[y]=predict_labels(clf, X_test, y_test)
                    write_parameters(l_pca,index,'DT',y,grid_obj.best_params_)
                    write_log(l_pca,index,'DT',y,stats_kf[y])
                    joblib.dump(clf,local_path+'models/'+str(l_pca)+'generic_DT_'+str(index)+str(y)+'.pkl',compress=3)
                    stats_model['RF']=stats_kf
                    del clf
    
    #AdaBoost
                parameters=get_parameters(l_pca,index,'AB',y)
                if any(parameters):
                    print 'Trained already.'                        
                else:    
                    clf=AdaBoostClassifier(n_estimators=100)
                    clf=clf.fit(X_train, y_train)
                    stats_kf[y]=predict_labels(clf, X_test, y_test)
                    write_parameters(l_pca,index,'AB',y,grid_obj.best_params_)
                    write_log(l_pca,index,'AB',y,stats_kf[y])
                    joblib.dump(clf,local_path+'models/'+str(l_pca)+'generic_AB_'+str(index)+str(y)+'.pkl',compress=3)
                    stats_model['AB']=stats_kf
                    del clf
    
    #kNeighbors
                parameters=get_parameters(l_pca,index,'kN',y)
                if any(parameters):
                    print 'Trained already.'                        
                else:    
                    parameters = {'n_neighbors':(3,4,5,6,7,8),'weights':('uniform','distance'),'algorithm':('auto', 'ball_tree', 'kd_tree', 'brute')}
                    clf=KNeighborsClassifier()
                    grid_obj = GridSearchCV(clf,parameters,scoring = scorer)
                    grid_obj = grid_obj.fit(X_train, y_train)
                    clf=grid_obj.best_estimator_
                    stats_kf[y]=predict_labels(clf, X_test, y_test)
                    write_parameters(l_pca,index,'kN',y,grid_obj.best_params_)
                    write_log(l_pca,index,'kN',y,stats_kf[y])
                    joblib.dump(clf,local_path+'models/'+str(l_pca)+'generic_kN_'+str(index)+str(y)+'.pkl',compress=3)
                    stats_model['kN']=stats_kf
                    del clf
    
    #GaussianNB
                parameters=get_parameters(l_pca,index,'GNB',y)
                if any(parameters):
                    print 'Trained already.'                        
                else:    
                    clf=GaussianNB()
                    clf=clf.fit(X_train, y_train)
                    stats_kf[y]=predict_labels(clf, X_test, y_test)
                    write_parameters(l_pca,index,'GNB',y,grid_obj.best_params_)                    
                    write_log(l_pca,index,'GNB',y,stats_kf[y])
                    joblib.dump(clf,local_path+'models/'+str(l_pca)+'generic_GNB_'+str(index)+str(y)+'.pkl',compress=3)
                    stats_model['GNB']=stats_kf
                    del clf
                    
                stats_accuracy[k]=stats_model

train('Close')