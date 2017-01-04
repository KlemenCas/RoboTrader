import tables
import commons
import csv
import pandas as pd

dbases=dict()
dbases['simulation'] = tables.open_file(commons.stats_path+'simulation.h5', 'r+')
#dbases['t_log']=tables.open_file(commons.stats_path+'t_log.h5', 'r')
#dbases['performance_log']=tables.open_file(commons.stats_path+'performance_log.h5', 'r')

for k, db in dbases.items():
    for x in db.walk_nodes():
        if x.__class__==tables.table.Table:
            tbl=dict()
            tbl[x.name]=db.get_node('/',x.name)
            with open(commons.analytics_path+x.name+'.csv','wb') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',')
                csvwriter.writerow(tbl[x.name].colnames)
                for row in tbl[x.name]:
                    try:
                        xrow=list()         
                        for c in range(0,len(tbl[x.name].colnames)):
                            xrow.append(row[c])
                        csvwriter.writerow(xrow)
                    except KeyError:
                        a=0
                csvfile.close()
    
    db.close()

with open(commons.analytics_path+'dix.csv','w') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',')
    for k,d in commons.date_index_external.items():
        xrow=list()         
        xrow.append(k)
        xrow.append(d)
        csvwriter.writerow(xrow)
csvfile.close()

with open(commons.analytics_path+'action.csv','w') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',')
    for k,d in commons.action_code.items():
        xrow=list()         
        xrow.append(k)
        xrow.append(d)
        csvwriter.writerow(xrow)
csvfile.close()


#export index & portfolio composition
dbases['simulation'] = tables.open_file(commons.stats_path+'simulation.h5', 'r+')
t_sim_h=dbases['simulation'].get_node('/','sim_h')
    
#for index,t in commons.sp500_composition.items():
#    index_t=commons.sp500_index[index][-10:-2]
#    df1=pd.DataFrame()
#    df2=pd.DataFrame()
#    dfa=pd.DataFrame()
#    dfb=pd.DataFrame()
#    
#
#    for row in t_sim_h:
#        try:
#            if df1.empty:
#                df1=pd.read_hdf(commons.local_path+'data/Index_Portfolio_'+index_t+'_'+row['sim_uuid']+'.h5','table')
#                df1.index = pd.MultiIndex.from_tuples([tuple((row['sim_uuid'], i)) for i in df1.index])
#                dfa=pd.read_hdf(commons.local_path+'data/Portfolio_'+index_t+'_'+row['sim_uuid']+'.h5','table')
#                dfa.index = pd.MultiIndex.from_tuples([tuple((row['sim_uuid'], i)) for i in dfa.index])                
##                df1=df1.drop_duplicates()
##                dfa=dfa.drop_duplicates()                
#
#            else:
#                df2=pd.read_hdf(commons.local_path+'data/Index_Portfolio_'+index_t+'_'+row['sim_uuid']+'.h5','table')
#                df2.index = pd.MultiIndex.from_tuples([tuple((row['sim_uuid'], i)) for i in df2.index])
#                dfb=pd.read_hdf(commons.local_path+'data/Portfolio_'+index_t+'_'+row['sim_uuid']+'.h5','table')
#                dfb.index = pd.MultiIndex.from_tuples([tuple((row['sim_uuid'], i)) for i in dfb.index])
#                df1=pd.concat([df1, df2])
#                dfa=pd.concat([dfa, dfb])
##                df1=df1.drop_duplicates()
##                dfa=dfa.drop_duplicates()                
#        except IOError:
#            a=1
#            
#
#    
#    with open(commons.analytics_path+'index.csv','w') as csvfile:
#        csvwriter = csv.writer(csvfile, delimiter=',')
#        xrow=list(['SimUUID','Date','Symbol','Volume'])
#        csvwriter.writerow(xrow)
#        for c in df1.columns:
#            for s,d in df1.index.drop_duplicates().get_values():
#                xrow=list()
#                xrow.append(s)
#                xrow.append(d)
#                xrow.append(c)
#                if df1.ix[(s,d),c].size>1:
#                    xrow.append(float(df1.ix[(s,d),c][0]))
#                else:
#                    xrow.append(float(df1.ix[(s,d),c]))
#                csvwriter.writerow(xrow)
#    csvfile.close()    
#
#    with open(commons.analytics_path+'portfolio.csv','w') as csvfile:
#        csvwriter = csv.writer(csvfile, delimiter=',')
#        xrow=list(['SimUUID','Date','Symbol','Volume'])
#        csvwriter.writerow(xrow)
#        for c in dfa.columns:
#            for s,d in dfa.index.drop_duplicates().get_values():
#                xrow=list()
#                xrow.append(s)
#                xrow.append(d)
#                xrow.append(c)
#                if dfa.ix[(s,d),c].size>1:
#                    xrow.append(float(dfa.ix[(s,d),c][0]))
#                else:
#                    xrow.append(float(dfa.ix[(s,d),c]))
#                csvwriter.writerow(xrow)
#    csvfile.close()  
    
#extract prices
#df=pd.read_hdf(commons.data_path+'WIKI_SP500.h5','table')
#with open(commons.analytics_path+'prices.csv','w') as csvfile:
#    csvwriter = csv.writer(csvfile, delimiter=',')
#    xrow=list(['Symbol','Date','Type','Price'])
#    csvwriter.writerow(xrow)
#    types=list(['Open','Close','Low','High'])    
#    for ticker in commons.getHistSp500TickerList(1,1,False):
#        for t in types:
#            c=str(ticker)+'_'+str(t)
#            print c
#            if c in df.columns:
#                for d in df.index:
#                    xrow=list()
#                    xrow.append(ticker)
#                    xrow.append(d)
#                    xrow.append(t)
#                    xrow.append(float(df.ix[d,c]))
#                    csvwriter.writerow(xrow)
#csvfile.close()