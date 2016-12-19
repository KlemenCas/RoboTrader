import commons
import pandas as pd
import csv
from database import db
from forecasts import forecast
from market import stock_market
import tables

class db(object):
    sim_uuid=None
    t_stats=None
    t_train_h=None
    t_sim_h=None
    t_ticker_symbols=None
    t_q=None
    t_recommend=None
    t_clusters=None
    t_parameter=None
    t_ticker_ids=None
    ti_ticker_ids=dict()
    ti_ticker=dict()
    cl_forecasts=None
    read_mode='w'
    alpha=0
    gamma=0
    
    def __init__(self):
        self.init_db()
        self.init_train_h_table()
        self.init_sim_h_table()
        self.init_stats_table()
        self.init_ticker_ids_table()
        self.init_q_table()
        self.init_q_log()
        self.init_cluster_table()
        self.init_parameter_table()
        self.init_recommendation_table()
        self.load_ticker_ids()
        self.init_transaction_log()
        self.init_portfolio_log()
        self.init_index_log()
        self.init_simulation_log()
        self.init_sp500Changes()
        self.init_noTrade()
        self.db_main.flush()

    def init_db(self):
        self.db_main = tables.open_file(commons.stats_path+'simulation.h5', self.read_mode)

#dropped changes
    def init_sp500Changes(self):
        try:
            desc={'ticker':tables.StringCol(10),
                  'dix':tables.IntCol(),
                  'sector':tables.StringCol(8),
                  'change':tables.StringCol(10)}
            self.sp500Changes=self.db_main.create_table('/','sp500Changes',desc)
            self.sp500Changes.cols.ticker.create_index()
            self.sp500Changes.cols.dix.create_index(kind='full')            
            print 'sp500Changes table created.'
        except tables.exceptions.NodeError:
            sp500Changes=self.db_main.get_node('/','sp500Changes')
            print 'sp500Changes opened.'   

#no trade
    def init_noTrade(self):
        try:
            desc={'ticker':tables.StringCol(10),
                  'dix':tables.IntCol()}
            self.noTrade=self.db_main.create_table('/','noTrade',desc)
            self.noTrade.cols.ticker.create_index()
            self.noTrade.cols.dix.create_index(kind='full')            
            print 'noTrade table created.'
        except tables.exceptions.NodeError:
            noTrade=self.db_main.get_node('/','noTrade')
            print 'noTrade opened.'   
            
#simulation log
    def init_simulation_log(self):
        try:
            perf_desc={'sim_uuid':tables.StringCol(32),
                       'gamma':tables.FloatCol(),
                       'alpha':tables.FloatCol(),
                       'simrun':tables.IntCol(),
                       'dix':tables.IntCol(),
                       'index':tables.StringCol(10),
                        'p_value':tables.FloatCol(),
                        'cash':tables.FloatCol(),
                        'i_value':tables.IntCol()}
            self.s_log=self.db_main.create_table('/','s_log',perf_desc)
            self.s_log.cols.dix.create_index(kind='full')
            print 'simulation log table created.'
        except tables.exceptions.NodeError:
            s_log=self.db_main.get_node('/','s_log')
            print 'simulation log opened.'   
    
#index log
    def init_index_log(self):
        try:
            i_log_desc={'sim_uuid':tables.StringCol(32),
                        'index_t':tables.StringCol(10),
                        'dix':tables.IntCol(),
                        'ticker':tables.StringCol(10),
                        'volume':tables.IntCol()}
            self.i_log=self.db_main.create_table('/','i_log',i_log_desc)
            self.i_log.cols.sim_uuid.create_index()
            self.i_log.cols.ticker.create_index()
            print 'index log table created.'
        except tables.exceptions.NodeError:
            self.i_log=self.db_main.get_node('/','i_log')
            print 'index log opened.'  
        
#portfolio log
    def init_portfolio_log(self):
        try:
            p_log_desc={'sim_uuid':tables.StringCol(32),
                        'index_t':tables.StringCol(10),
                        'dix':tables.IntCol(),
                        'ticker':tables.StringCol(10),
                        'volume':tables.IntCol()}
            self.p_log=self.db_main.create_table('/','p_log',p_log_desc)
            self.p_log.cols.sim_uuid.create_index()
            self.p_log.cols.ticker.create_index()
            print 'portfolio log table created.'
        except tables.exceptions.NodeError:
            self.p_log=self.db_main.get_node('/','p_log')
            print 'portfolio log opened.'  
        

#transaction log
    def init_transaction_log(self):
        try:
            t_log_desc={'sim_uuid':tables.StringCol(32),
                        'dix':tables.IntCol(),
                        'ticker':tables.StringCol(10),
                        'tx':tables.StringCol(10),
                        'price':tables.FloatCol(),
                        'volume':tables.IntCol(),
                        'close':tables.FloatCol(),
                        'cash_before':tables.FloatCol(),
                        'cash_after':tables.FloatCol()}
            self.t_log=self.db_main.create_table('/','t_log',t_log_desc)
            self.t_log.cols.sim_uuid.create_index()
            self.t_log.cols.ticker.create_index()
            print 'transaction log table created.'
        except tables.exceptions.NodeError:
            self.t_log=self.db_main.get_node('/','t_log')
            print 'transaction log opened.'  
            
#parameter header
    def init_parameter_table(self):
        if self.read_mode=='w':        
            try:
                self.db_main.remove_node('/', 'parameter')
                print 'paramter table dropped.'
            except tables.exceptions.NoSuchNodeError:
                print 'parameter table not exist yet. nothing to initialize.'         
            try:
                parameter_desc={'train_uuid':tables.StringCol(32),
                               'pca':tables.IntCol(),
                               'ticker':tables.StringCol(10),
                               'model':tables.StringCol(3),
                               'kpi':tables.StringCol(15),
                               'kernel':tables.StringCol(10),
                               'C':tables.IntCol(),
                               'max_depth':tables.IntCol(),
                               'n_neighbors':tables.IntCol(),
                               'weights':tables.StringCol(10),
                               'algorithm':tables.StringCol(10)}

                self.t_parameter=self.db_main.create_table('/','parameter',parameter_desc)
                self.t_parameter.cols.train_uuid.create_index()
                self.t_parameter.cols.pca.create_index()
                self.t_parameter.cols.ticker.create_index()
                self.t_parameter.cols.model.create_index()
                self.t_parameter.cols.kpi.create_index()
                print 'parameter table created.'
            except tables.exceptions.NodeError:
                self.t_parameter=self.db_main.get_node('/','parameter')
                print 'parameter table opened.'
        else:
            self.t_parameter=self.db_main.get_node('/','parameter')
            print 'parameter table opened.'
            
#training header
    def init_train_h_table(self):
        if self.read_mode=='w':        
            try:
                self.db_main.remove_node('/', 'train_h')
                print 'train header table dropped.'
            except tables.exceptions.NoSuchNodeError:
                print 'train header table not exist yet. nothing to initialize.'         
            try:
                train_h_desc={'train_uuid':tables.StringCol(32),
                              'enddix':tables.IntCol()}
                self.t_train_h=self.db_main.create_table('/','train_h',train_h_desc)
                self.t_train_h.cols.train_uuid.create_index()
                print 'train header table created.'
            except tables.exceptions.NodeError:
                self.t_train_h=self.db_main.get_node('/','train_h')
                print 'train header table opened.'
        else:
            self.t_train_h=self.db_main.get_node('/','train_h')
            print 'train header table opened.'

#simulation header
    def init_sim_h_table(self):
        if self.read_mode=='w':        
            try:
                self.db_main.remove_node('/', 'sim_h')
                print 'simulation header table dropped.'
            except tables.exceptions.NoSuchNodeError:
                print 'simulation header table not exist yet. nothing to initialize.'         
            try:
                sim_h_desc={'sim_uuid':tables.StringCol(32),
                              'dix':tables.IntCol(),
                              'startdix':tables.IntCol(),
                              'enddix':tables.IntCol()}
                self.t_sim_h=self.db_main.create_table('/','sim_h',sim_h_desc)
                self.t_sim_h.cols.sim_uuid.create_index()
                self.t_sim_h.cols.dix.create_index()
                print 'simulation header table created.'
            except tables.exceptions.NodeError:
                self.t_sim_h=self.db_main.get_node('/','sim_h')
                print 'simulation header table opened.'
        else:
            self.t_sim_h=self.db_main.get_node('/','sim_h')
            print 'simulation header table opened.'

#recommendations
    def init_recommendation_table(self):
        if self.read_mode=='w':        
            try:
                self.db_main.remove_node('/', 'recommend')
                print 'recommendation table dropped.'
            except tables.exceptions.NoSuchNodeError:
                print 'recommendation table does not exist yet. nothing to initialize.'         
            try:
                recommend_desc={'sim_uuid':tables.StringCol(32),
                                'dix':tables.IntCol(),
                                'ticker':tables.StringCol(10),
                                'action':tables.StringCol(10),
                                'volume':tables.IntCol()}
                self.t_recommend=self.db_main.create_table('/','recommend',recommend_desc)
                self.t_recommend.cols.sim_uuid.create_index()
                self.t_recommend.cols.dix.create_index()
                print 'recommendation table created.'
            except tables.exceptions.NodeError:
                self.t_recommend=self.db_main.get_node('/','recommend')
                print 'recommendation table opened.'
        else:
            try:
                self.t_recommend=self.db_main.get_node('/','recommend')
                print 'recommendation table opened.'
            except tables.exceptions.NoSuchNodeError:
                recommend_desc={'sim_uuid':tables.StringCol(32),
                                'dix':tables.IntCol(),
                                'ticker':tables.StringCol(10),
                                'action':tables.StringCol(10),
                                'volume':tables.IntCol()}
                self.t_recommend=self.db_main.create_table('/','recommend',recommend_desc)
                self.t_recommend.cols.sim_uuid.create_index()
                self.t_recommend.cols.dix.create_index()
            
#accuracy stats            
    def init_stats_table(self):
        if self.read_mode=='w':        
            try:
                self.db_main.remove_node('/', 'stats')
                print 'stats table dropped.'
            except tables.exceptions.NoSuchNodeError:
                print 'stats table not exist yet. nothing to initialize.'         
            try:
                stats_desc={'train_uuid':tables.StringCol(32),
                            'pca':tables.IntCol(),
                            'ticker':tables.StringCol(10),
                            'model': tables.StringCol(3),
                            'kpi':   tables.StringCol(15),
                            'accuracy':tables.FloatCol()}
                self.t_stats=self.db_main.create_table('/','stats',stats_desc)
                self.t_stats.cols.ticker.create_index()
                self.t_stats.cols.kpi.create_index()
                print 'statistics table created.'
            except tables.exceptions.NodeError:
                self.t_stats=self.db_main.get_node('/','stats')
                print 'stats table opened.'
        else:
            self.t_stats=self.db_main.get_node('/','stats')
            print 'stats table opened.'
            

    def init_ticker_ids_table(self):
        if self.read_mode=='w':    
            try:
                ticker_desc={'ticker':tables.StringCol(10),
                             'id':tables.IntCol()}
                self.t_ticker_ids=self.db_main.create_table('/','ticker_symbols',ticker_desc)
                self.t_ticker_ids.cols.ticker.create_index()
                print 'ticker ids table created.'
            except tables.exceptions.NodeError:
                self.t_ticker_ids=self.db_main.get_node('/','ticker_symbols')            
                print 'ticker symbols table opened.'
        else:
            self.t_ticker_ids=self.db_main.get_node('/','ticker_symbols')            
            print 'ticker symbols table opened.'
            

    def init_q_table(self):
        if self.read_mode=='w': 
            try:
                self.db_main.remove_node('/', 'q_table')
                print 'q_table table dropped.'            
            except tables.exceptions.NoSuchNodeError:
                print 'no q table to drop.'
            try:
                q_table_desc={'ticker':tables.IntCol(),
                              'state':tables.IntCol(),
                              'action':tables.IntCol(),
                              'reward':tables.FloatCol()}
                self.t_q=self.db_main.create_table('/','q_table',q_table_desc)
                self.t_q.cols.ticker.create_index()
                self.t_q.cols.state.create_index()
                self.t_q.cols.action.create_index()
                print 'q table created.'
            except tables.exceptions.NodeError:
                self.t_q=self.db_main.get_node('/','q_table')
                print 'q table opened.'
        else:
            self.t_q=self.db_main.get_node('/','q_table')
            print 'q table opened.'

                
    def init_q_log(self):
        if self.read_mode=='w':   
            try:
                self.db_main.remove_node('/', 'q_log')
                print 'stats table dropped.'            
            except tables.exceptions.NoSuchNodeError:
                print 'no q_log to drop.'
            try:
                q_log_desc={'sim_uuid':tables.StringCol(32),
                            'ticker':tables.StringCol(10),
                            'dix':tables.IntCol(),
                            'state':tables.Int64Col(),
                            'action':tables.IntCol(),
                            'reward':tables.FloatCol()}
                self.q_log=self.db_main.create_table('/','q_log',q_log_desc)
                self.q_log.cols.ticker.create_index()
                self.q_log.cols.state.create_index()
                self.q_log.cols.action.create_index()
                print 'q log table created.'
            except tables.exceptions.NodeError:
                self.q_log=self.db_main.get_node('/','q_log')
                print 'q_log table opened.'
        else:
            self.q_log=self.db_main.get_node('/','q_log')
            print 'q_log table opened.'
            
            
    def init_cluster_table(self):
        if self.read_mode=='w':   
            try:
                self.db_main.remove_node('/', 'cluster_table')
                print 'Cluster table dropped.'
            except tables.exceptions.NoSuchNodeError:
                print 'cluster table does not exist yet. nothing to drop.'        
            try:
                cluster_desc={'train_uuid':tables.StringCol(32),
                              'ticker':tables.StringCol(10),
                              'kpi':tables.StringCol(15),
                              'c0':tables.FloatCol(),
                              'c1':tables.FloatCol(),
                              'c2':tables.FloatCol(),
                              'c3':tables.FloatCol(),
                              'c4':tables.FloatCol()}
                self.t_clusters=self.db_main.create_table('/','cluster_table',cluster_desc)
                self.t_clusters.cols.ticker.create_index()
                self.t_clusters.cols.kpi.create_index()
                print 'cluster table created.'
            except tables.exceptions.NodeError:
                self.t_clusters=self.db_main.get_node('/','cluster_table')
                print 'cluster table exists already.'
        else:
            self.t_clusters=self.db_main.get_node('/','cluster_table')
            print 'cluster table exists already.'
            
    def load_ticker_ids(self):   
        all_rows=self.t_ticker_ids.read()
        for ticker in commons.getHistSp500TickerList(commons.min_date,commons.min_date,False):
            ticker_id=self.t_ticker_ids.read_where('ticker=='+"'"+ticker+"'")
            if not any(ticker_id):
                if any(all_rows):
                    new_id=self.t_ticker_ids.nrows+1
                else:
                    new_id=1
                self.t_ticker_ids.row['ticker']=ticker
                self.t_ticker_ids.row['id']=new_id
                self.t_ticker_ids.row.append()
                self.t_ticker_ids.flush()
                all_rows=self.t_ticker_ids.read()
        print 'Ticker IDs table initialized.'    
    
        
dba=db()       
#with open(commons.local_path+'backup/stats.csv','r') as csvfile:
#    csvreader = csv.reader(csvfile, delimiter=',')
#    l_i=0
#    for row in csvreader:
#        if l_i>0:
#            dba.t_stats.row['train_uuid']=row[5]
#            dba.t_stats.row['pca']=int(row[3])
#            dba.t_stats.row['ticker']=row[4]
#            dba.t_stats.row['model']=row[2]
#            dba.t_stats.row['kpi']=row[1]
#            dba.t_stats.row['accuracy']=float(row[0])
#            dba.t_stats.row.append()
#            dba.t_stats.flush()
#        l_i+=1
#csvfile.close()    
#print 'stats loaded'
#    
#with open(commons.local_path+'backup/parameter.csv','r') as csvfile:
#    csvreader = csv.reader(csvfile, delimiter=',')
#    l_i=0
#    for row in csvreader:
#        if l_i>0:
#            dba.t_parameter.row['C']=int(row[0])
#            dba.t_parameter.row['algorithm']=row[1]
#            dba.t_parameter.row['kernel']=row[2]
#            dba.t_parameter.row['kpi']=row[3]
#            dba.t_parameter.row['max_depth']=int(row[4])
#            dba.t_parameter.row['model']=row[5]
#            dba.t_parameter.row['n_neighbors']=int(row[6])
#            dba.t_parameter.row['pca']=int(row[7])
#            dba.t_parameter.row['ticker']=row[8]
#            dba.t_parameter.row['train_uuid']=row[9]
#            dba.t_parameter.row['weights']=row[10]        
#            dba.t_parameter.row.append()
#            dba.t_parameter.flush()
#        l_i+=1
#csvfile.close()    
#print 'parameter loaded'
#    
#
#    
#with open(commons.local_path+'backup/cluster_table.csv','r') as csvfile:
#    csvreader = csv.reader(csvfile, delimiter=',')
#    l_i=0
#    for row in csvreader:
#        if l_i>0:        
#            dba.t_clusters.row['c0']=float(row[0])
#            dba.t_clusters.row['c1']=float(row[1])
#            dba.t_clusters.row['c2']=float(row[2])
#            dba.t_clusters.row['c3']=float(row[3])
#            dba.t_clusters.row['c4']=float(row[4])
#            dba.t_clusters.row['kpi']=row[5]
#            dba.t_clusters.row['ticker']=row[6]
#            dba.t_clusters.row['train_uuid']=row[7]
#            dba.t_clusters.row.append()
#            dba.t_clusters.flush()
#        l_i+=1
#csvfile.close()    
#print 'cluster loaded'  
#
#
#with open(commons.local_path+'backup/sp500_changes.csv','r') as csvfile:
#    csvreader = csv.reader(csvfile, delimiter=',')
#    for row in csvreader:
#        dba.sp500Changes.row['ticker']=row[0]
#        dba.sp500Changes.row['dix']=commons.date_index_external[dt.datetime.strptime(row[2], '%m/%d/%Y')]
#        dba.sp500Changes.row['sector']=commons.sp500_index[row[1]][-10:-2]
#        dba.sp500Changes.row['change']=row[3]
#        dba.sp500Changes.row.append()
#        dba.sp500Changes.flush()
#
#csvfile.close()    
#print 'sp500Changes loaded'  