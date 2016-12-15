import commons
import pandas as pd
import datetime as dt

#model
from sklearn.decomposition import PCA

#globals locally
local_path=commons.local_path

#persistence
from sklearn.externals import joblib

data_sp500=pd.read_hdf(commons.data_path+'WIKI_SP500.h5','table')        

def train(mode):
    modes=list(['Open','Close','Low','High'])
    modes.remove(mode) 
    
    l_i=505     
    for l_pca in range(6,12):
        print 'pca: ', l_pca
        for k,dates in commons.sp500CompDates.items():
            l_i-=1
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
            #reduce space?
            if l_pca!=0:
                pca=PCA(n_components=l_pca)
                pca=pca.fit(X_all)
                X_all=pca.transform(X_all)
                joblib.dump(pca,local_path+'models/'+str(l_pca)+'_PCA_'+str(k)+'.pkl',compress=3)
                del pca

train('Close')