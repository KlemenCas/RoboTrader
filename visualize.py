import commons
from scipy.stats import kendalltau
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import tables
import numpy as np
#from prettytable import PrettyTable
import csv

  
#
dba = tables.open_file(commons.stats_path+'simulation.h5', 'r+')
p_log=dba.get_node('/','s_log')
#p_log.cols.dix.remove_index()
#p_log.cols.dix.create_index(kind='full')

maxsim=0
for row in p_log:
    if maxsim<row['simrun']:
        maxsim=row['simrun']

for simrun in range(1,maxsim+1):
    df=pd.DataFrame(columns=['Index','P_Total'])
    first='X'

    for row in p_log.itersorted('dix'):
        if row['simrun']==simrun:
            if first=='X':
                first=''
                p_base=row['p_value']+row['cash']
                i_base=row['i_value']*1.
            df.ix[commons.date_index_external[row['dix']],['Index']]=(row['i_value'])/i_base
            df.ix[commons.date_index_external[row['dix']],['P_Total']]=(row['p_value']+row['cash'])/p_base

    print 'simulation:',simrun
    df.plot(kind='line',fontsize=20)
    plt.show()