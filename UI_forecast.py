import commons
import datetime as dt
from market import stock_market
from database import db
from forecasts import forecast
import Tkinter as tk


class user_forecasts(object):    
    def __init__(self, top):
        if commons.demo_scenario:
            self.demo_date=dt.datetime(2016,10,31)    
        else:
            self.demo_date=dt.date.today()
        self.dix=commons.date_index_internal[self.demo_date]
        self.m=stock_market()
        self.f=forecast()
        self.dba=db(self.f,'r')
        self.f.set_dba=self.dba        

#        self.mLabel=tk.Tk()
        self.mlx = tk.Listbox(top)
        for k,v in commons.sp500_ticker.items():
            self.mlx.insert('end', k)   
        self.mlx.bind('<<ListboxSelect>>',self.user_forecast)
        self.mlx.pack()
        self.temperature=.5
   
    def user_forecast(self,evt):
        w = evt.widget
        index = int(w.curselection()[0])
        ticker = w.get(index)
        master.destroy()
        
        self.mLabel=tk.Tk()
        self.mLabel.geometry("800x100+300+350")
        mtext=str(ticker)+' selected. Preparing forecast.'
        mlb=tk.Message(self.mLabel,text=mtext,anchor='ne',width=600)  
        mlb.pack()
        mlb.update()
        
        state=self.f.get_forecast_state(self.dba.t_stats,ticker,self.dix)
        proposed_action=self.dba.get_softmax_action(ticker,state,self.temperature)
        best1dd=self.f.get_best_model(self.dba.t_stats,ticker,'1dd_Close')
        best5dd=self.f.get_best_model(self.dba.t_stats,ticker,'5dd_Close')
        best20dd=self.f.get_best_model(self.dba.t_stats,ticker,'20dd_Close')
        
        self.mLabel.destroy()
        
        self.mLabel=tk.Tk()
        self.mLabel.geometry("500x200+300+200")  
        mlb=tk.Message(self.mLabel,text=self.forecast_text(state,ticker,proposed_action,best1dd,best5dd,best20dd),anchor='ne',width=600)    
        mlb.pack()
        mlb.update()
        
    def forecast_text(self,state,ticker,proposed_action,best1dd,best5dd,best20dd):
        forecast_text='Forecast for '+ticker+' performed. The model delivered the following results: \n'
        if state['1dd_Close']==-1:
            forecast_text+='- Stock price will fall tomorrow. The confidence in this prediction is: '+str(max([best1dd[5],best1dd[6]]))+'.\n'
        else:
            forecast_text+='- Stock price will raise tomorrow. The confidence in this prediction is: '+str(max([best1dd[5],best1dd[6]]))+'.\n'
        if state['5dd_Close']==-1:
            forecast_text+='- Stock price will fall over the next 5 days. The confidence in this prediction is: '+str(max([best5dd[5],best5dd[6]]))+'.\n'
        else:
            forecast_text+='- Stock price will raise over the next 5 days. The confidence in this prediction is: '+str(max([best5dd[5],best5dd[6]]))+'.\n'
        if state['20dd_Close']==-1:
            forecast_text+='- Stock price will fall over the next 20 days. The confidence in this prediction is: '+str(max([best20dd[5],best20dd[6]]))+'.\n'
        else:
            forecast_text+='- Stock price will raise over the next 20 days. The confidence in this prediction is: '+str(max([best20dd[5],best20dd[6]]))+'.\n'
        price_recommendation=self.f.get_order_price(self.dba,self.dba.ti_ticker_ids[ticker],state,\
                                               self.dix,proposed_action,self.m.get_closing_price(ticker,self.dix))
        forecast_text+='In case of a transaction the recommended order price is: '+"{0:.2f}".format(round(price_recommendation,2))+'. '
        return forecast_text
        
master = tk.Tk()
master.title('Please pick a stock')
master.geometry("300x200+300+200")
FC = user_forecasts(master)
master.mainloop()