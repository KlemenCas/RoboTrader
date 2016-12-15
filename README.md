**WORKFLOW**

1.Check https://en.wikipedia.org/wiki/List_of_S%26P_500_companies for any SP500 changes. If there are, maintain in C:\Users\kleme\OneDrive\HF_Trading\RoboTrader\backup\sp500_changes.xls



**DEMO:**
In demo mode only data to the SP500 sector 'Telecommunications Services' will be loaded and processed. Demo flag is being set in commons.py - see variable _demo_scenario_. Otherwise all stock symbols will be processed (500 stocks!), and it will take a long time...

**Data Retrieval:**

1. For data retrieval please run the python module _mydata\_get\_delta.py_. 
2. Quandl Key needs to be provided.
3. Subscription to 2 databases is required: SP0 (needs subscription, but is free) and SF1 (not free!) 


**Training:**
For training please run _train.py_. Note!: the l_pca in the method _train()_ needs to be set to the dimension reduction that you want to train. Currently it's set to 0 (= no reduction). Training is currently only being called with mode=Close, which means that the forecast is on the Close prices. The Xy_all contains all values (Open, Low, High, Close). For labels where the system already knows the best parameters and has the model saved, there will be _Trained already._ message. 


**Simulation**
For simulation please run _simulation.py_. It will simulate price development from -5 years to today. At -5 years two portfolios will be initialized; one will track closely the index, like an ETF, adjusting the composition daily to the market cap based composition. The other portfolio will make adjustments based on the ML recommendations. The results (= the peformance) will be saved in _performance_log.h5_.


**Individual Stock Recommendation**
For the individual stock recommendation please run _UI_forecast.py_. It will open a listbox, the selected stock will be forecasted. Note; for demo the forecast date has been set to September 30th 2016. The date can be changed in the code, note though that quandl only provides the stock prices towards the end of the day. Do not remove the demo_scenario flag in commons, as only the pickle object for the demo have been uploaded to Github. 


**Short Description to all Modules** 

1. _commons.py_: common definitions, constants and methods that are are being reused in various modules
2. _database.py_: handling of the database, including Q-Learning related methods (softmax, max_q, q_update
3. _forecasts.py_: forecast related methods (next state, forecasted order price, best model
4. _market.py_: index portfolio; composition and value, methods delivering market prices on a certain day
5. _mydata.py_:data retrieval methods and local storage
6. _mydata_get_delta.py_: data retrieval process
7. _portfolio.py_: user portfolio, buying list alignment, order execution
8. _simulate.py_: simulation run
9. _train.py_: training of scikit models by stock symbol and local storage
10. _train_generic.py_: training of scikit models by index and local storage
11. _train_pca.py_: calculation and local storage of PCA objects, for later reuse
12. _UI_forecast.py_: forecast to one stock symbol
13. _visualize.py_: helper to visualize and export data.
