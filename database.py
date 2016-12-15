import commons
import tables
import csv
import numpy as np
import random
from math import exp


class db(object):
    sim_uuid=None
    s_log=None
    t_log=None
    p_log=None
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
    read_mode='r+'
    alpha=0
    gamma=0
    
    def __init__(self,sim_uuid,read_mode='r+'):
        self.sim_uuid=sim_uuid
        self.read_mode=read_mode
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
        self.init_transaction_log()
        self.init_portfolio_log()
        self.init_index_log()
        self.init_simulation_log()
        if self.read_mode=='w':
            self.load_ticker_ids()
        self.load_internal_ticker_ids()
        

    def init_db(self):
        self.db_main = tables.open_file(commons.stats_path+'simulation.h5', self.read_mode)

#transaction log
    def init_transaction_log(self):
        self.t_log=self.db_main.get_node('/','t_log')
        print 'transaction log opened.'  

#simulation log
    def init_simulation_log(self):
        self.s_log=self.db_main.get_node('/','s_log')
        print 'simulation log opened.'  

        #index log
    def init_index_log(self):
        self.i_log=self.db_main.get_node('/','i_log')
        print 'index log opened.'   
        
#portfolio log
    def init_portfolio_log(self):
        self.p_log=self.db_main.get_node('/','p_log')
        print 'portfolio log opened.'         
#parameter header
    def init_parameter_table(self):
        self.t_parameter=self.db_main.get_node('/','parameter')
        print 'parameter table opened.'
            
#training header
    def init_train_h_table(self):
        self.t_train_h=self.db_main.get_node('/','train_h')
        print 'train header table opened.'

#simulation header
    def init_sim_h_table(self):
        self.t_sim_h=self.db_main.get_node('/','sim_h')
        print 'simulation header table opened.'

#recommendations
    def init_recommendation_table(self):
        self.t_recommend=self.db_main.get_node('/','recommend')
        print 'recommendation table opened.'
            
#accuracy stats            
    def init_stats_table(self):
        self.t_stats=self.db_main.get_node('/','stats')
        print 'stats table opened.'
            

    def init_ticker_ids_table(self):
        self.t_ticker_ids=self.db_main.get_node('/','ticker_symbols')            
        print 'ticker symbols table opened.'
            

    def init_q_table(self):
        self.t_q=self.db_main.get_node('/','q_table')
        print 'q table opened.'

                
    def init_q_log(self):
        self.q_log=self.db_main.get_node('/','q_log')
        print 'q_log table opened.'
            
            
    def init_cluster_table(self):
        self.t_clusters=self.db_main.get_node('/','cluster_table')
        print 'cluster table exists already.'
                
    def load_internal_ticker_ids(self):
        for ticker in commons.getHistSp500TickerList(commons.min_date,commons.min_date,False):
            for row in self.t_ticker_ids.read_where('ticker=='+"'"+ticker+"'"):
                self.ti_ticker_ids[row[1]]=row[0]
                self.ti_ticker[row[0]]=row[1]
        
    
    def update_q_table(self,ticker,state,action,reward,dix):
        q_records=self.t_q.where('(ticker=='+str(self.ti_ticker_ids[ticker])+') & (state=='+str(self.get_q_key(state))+') & (action=='+str(action)+')')
        if any(q_records):
            for row in self.t_q.where('(ticker=='+str(self.ti_ticker_ids[ticker])+') & (state=='+str(self.get_q_key(state))+') & (action=='+str(action)+')'):
                row['reward']=reward
                row.update()
        else:
            self.t_q.row['ticker']=self.ti_ticker_ids[ticker]
            self.t_q.row['state']=self.get_q_key(state)
            self.t_q.row['action']=action
            self.t_q.row['reward']=reward
            self.t_q.row.append()
            self.t_q.flush()
        self.q_log.row['dix']=dix
        self.q_log.row['sim_uuid']=self.sim_uuid
        self.q_log.row['ticker']=ticker       
        self.q_log.row['state']=self.get_q_key(state)
        self.q_log.row['action']=action
        self.q_log.row['reward']=reward
        self.q_log.row.append()
        self.q_log.flush()
    
            
    def get_max_q(self,ticker,state):
        max_action=''
        max_q=-1000
        x_random=False
        q_records=self.t_q.read_where('(ticker=='+str(self.ti_ticker_ids[ticker])+') & (state=='+str(self.get_q_key(state))+')')
        if any(q_records):
            q_records_n=np.array(q_records,dtype=[('ticker',int),('state',int),('action',int),('reward',float)])
            q_records_s=np.sort(q_records_n,order='reward')
            max_action=q_records_s[-1]['action']
            max_q=q_records_s[-1]['reward']
        if max_q==-1000:
            max_q=100
            max_action=random.choice([commons.action_code['sell'],commons.action_code['hold'],commons.action_code['buy']])
            x_random=True
        return max_q, max_action,x_random
        
    def get_reward(self,ticker,state,action_code):
        q_records=self.t_q.read_where('(ticker=='+str(self.ti_ticker_ids[ticker])+') & (state=='+str(self.get_q_key(state))+') & (action=='+str(action_code)+')')
        if any(q_records):
            for row in q_records:
                return float(row['reward'])
        else:
            return 50.
            
    def get_softmax_action(self,ticker,state,t):
        actions=[commons.action_code['sell'],commons.action_code['buy']]
#commons.action_code['hold'],
        distr=np.array([])
        e_q_sum=0        
        for b in actions:
            e_q_sum+=exp(self.get_reward(ticker,state,b)/t)
        for a in actions:
            e_q=0.
            e_q=exp(self.get_reward(ticker,state,a)/t)
            distr_a=np.array([])
            for i in range(int(e_q/e_q_sum*100)):
                distr_a=np.append(distr_a,a)
            distr=np.append(distr,distr_a)
        return distr[int(distr[int(random.random()*len(distr))])]

    def get_q_key(self,state):
        q_key=1
        for y in commons.y_dd_labels:            
            q_key=q_key+pow(10,commons.qkc[y])*(state[y]+2)
        return int(q_key)
                     
    def string_to_list(self,a):
        x=list()
        f_n=0.
        n=''
        for c in a:
            if c in ['1','2','3','4','5','6','7','8','9','0','-','.','E','e']:
                n=n+c
            else:
                if len(n)>0 and c in [' ',']']:
                    f_n=float(n)
                    x.append(f_n)
                    n=''
        
        return x
        
    def new_training(self,uuid,enddix):
        self.t_train_h.row['train_uuid']=uuid
        self.t_train_h.row['enddix']=enddix
        self.t_train_h.row.append()
        self.t_train_h.flush()
        
    def new_simulation(self,uuid,dix,startdix,enddix):
        self.t_sim_h.row['sim_uuid']=uuid
        self.t_sim_h.row['dix']=dix
        self.t_sim_h.row['startdix']=startdix
        self.t_sim_h.row['enddix']=enddix
        self.t_sim_h.row.append()
        self.t_sim_h.flush()
        
    
    def log_recommendation(self,sim_uuid,dix,ticker,action,volume=0):
        self.t_recommend.row['sim_uuid']=sim_uuid
        self.t_recommend.row['dix']=dix
        self.t_recommend.row['ticker']=ticker
        self.t_recommend.row['action']=action
        self.t_recommend.row['volume']=volume
        self.t_recommend.row.append()
        self.t_recommend.flush()