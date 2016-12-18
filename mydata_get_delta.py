#the code calls the methods in mydata to refresh everything to until today
from mydata import mydata
import ttk as tt
import Tkinter as tk
import time

class cl_get_delta(object):

    def __init__(self):    
        self.requery_data=True
        self.QuandlKey='QWQMKc-NYsoYgtfadPZs'
        self.get_delta(self.QuandlKey)

    
    def set_key(self):
        self.QuandlKey=self.mk.get()
        self.mKey.destroy()
        self.get_delta(self.QuandlKey)

    def get_delta(self,quandlkey):
        if self.requery_data==True:
            self.mGui = tk.Tk()
            self.mLabel=tk.Tk()

            self.mGui.geometry("800x100+300+500")
            self.mLabel.geometry("800x100+300+350")
        
            self.mLabel.title('Updates')
            mlb=tk.Message(self.mLabel,text='The database needs to be refreshed. This will take a while.',anchor='ne',width=600)
            mlb.pack()
            mlb.update()
            self.mGui.title('Progress')
            mpb = tt.Progressbar(self.mGui,orient ="horizontal",length = 700, mode ="determinate",variable='i')
            mpb.pack()
            mpb["maximum"] = 12
            mpb.update()
            time.sleep(5)
            
            start=time.time()
            mlb["text"]='Reading Marketcap (all S&P500 tickers).'
            mlb.update()
            mpb["value"]=1
            mpb.update()
            dba=mydata(True,False,True,self.QuandlKey)
            dba.getMarketcap()
            end=time.time()
            print 'Runtime:',end-start
            
            start=time.time()
            mlb["text"]='Refreshing index composition.'    
            mlb.update()
            mpb["value"]=2
            mpb.update()        
            dba.getIndexComposition()
            end=time.time()
            print 'Runtime:',end-start
            
            start=time.time()
            mlb["text"]='Reading SP500 index prices.'    
            mlb.update()
            mpb["value"]=3
            mpb.update()
            dba.getIndexData()
            end=time.time()
            print 'Runtime:',end-start
            
            start=time.time()
            mlb["text"]='Reading fundamentals.'    
            mlb.update()
            mpb["value"]=4
            mpb.update()    
            dba.getFundamentals()
            end=time.time()
            print 'Runtime:',end-start
            
            start=time.time()
            mlb["text"]='Reading short sell data.'   
            mlb.update()
            mpb["value"]=5
            mpb.update()    
            dba.getShortSell()
            end=time.time()
            print 'Runtime:',end-start
            
            start=time.time()
            mlb["text"]='Reading sentiment.'    
            mlb.update()
            mpb["value"]=6
            mpb.update()    
            dba.getSentiment()
            end=time.time()
            print 'Runtime:',end-start
            
            start=time.time()
            mlb["text"]='Calculating dates.'    
            mlb.update()
            mpb["value"]=7
            mpb.update()    
            dba.calcSp5001stDate()
            end=time.time()
            print 'Runtime:',end-start
            
            start=time.time()
            mlb["text"]='Updating Alpha and Beta.'    
            mlb.update()
            mpb["value"]=8
            mpb.update()    
            dba.calcSectorBetas()
            end=time.time()
            print 'Runtime:',end-start
            
            start=time.time()
            mlb["text"]='Slicing Alpha and Beta.'    
            mlb.update()
            mpb["value"]=9
            mpb.update()    
            dba.sliceSectorBetas()
            end=time.time()
            print 'Runtime:',end-start
            
            start=time.time()
            mlb["text"]='Treatment of NaN values.'    
            mlb.update()
            mpb["value"]=10
            mpb.update()    
            dba.sp500fillna()
            end=time.time()
            print 'Runtime:',end-start
            
            start=time.time()
            mlb["text"]='Slicing Index.'    
            mlb.update()
            mpb["value"]=11
            mpb.update()    
            dba.sliceIndex()
            end=time.time()
            print 'Runtime:',end-start
            
            start=time.time()
            mlb["text"]='Calculating indicators.'    
            mlb.update()
            mpb["value"]=12
            mpb.update()        
            dba.calcIndicators()
            end=time.time()
            print 'Runtime:',end-start

            mlb["text"]='Done. Closing in 3 sec.'    
            mlb.update()
            time.sleep(1)
            mlb["text"]='Done. Closing in 2 sec.'    
            mlb.update()
            time.sleep(1)
            mlb["text"]='Done. Closing in 1 sec.'    
            mlb.update()
            time.sleep(1)
            
            self.mGui.destroy()
            self.mLabel.destroy()            
                    
x=cl_get_delta()