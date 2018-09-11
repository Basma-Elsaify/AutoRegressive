
# coding: utf-8

# In[2]:


from pandas import Series
from matplotlib import pyplot
import numpy as np
from pandas.tools.plotting import lag_plot
from pandas import DataFrame
from pandas import concat
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf
# pandas.plotting.autocorrelation_plot

series = Series.from_csv('file.csv', index_col=0, header = 0 )


# df = pd.read_csv('data/file.csv', delimiter = ';')    
# df = df.replace({'t': {',': '.'}}, regex=True)
# df['t'] = df['t'].astype(np.float64)
# df = df.replace({'deflexion11': {',': '.'}}, regex=True)
# df['deflexion11'] = df['deflexion11'].astype(np.float64)
# df.to_csv('file_export.csv')

series = series.astype(np.float64)
print(series)


# In[2]:


series.plot()
pyplot.show()


# In[31]:


lag_plot(series)
pyplot.show()


# In[5]:


values = DataFrame(series.values)
dataframe = concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
result = dataframe.corr()
print(result)


# In[6]:


autocorrelation_plot(series)
pyplot.show()


# In[9]:


plot_acf(series, lags=31)
pyplot.show()


# In[5]:


from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error

X = series.values
train, test = X[1:len(X)-50000], X[len(X)-50000:]

model = AR(train)
model_fit = model.fit()
wd = model_fit.k_ar
coef = model_fit.params 
h = train[len(train)-wd:]
h = [h[i] for i in range(len(h))]
predictions = list()
opf = open('AR_result.csv', 'a')
opf.write('predicted, expected')
for t in range(len(test)):
    length = len(h)
    lag = [h[i] for i in range(length-wd,length)]
    y = coef[0]
    for d in range(wd):
        y += coef[d+1] * lag[wd-d-1]
    obs = test[t]
    predictions.append(y)
    h.append(obs)
    opf.write('\n %f, %f' % (y, obs))
    print('predicted=%f, expected=%f' % (y, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
# pyplot.savefig("ar_graph.png")
pyplot.show()
pyplot.plot(test)
# pyplot.savefig("ar_test.png")
pyplot.show()


# In[ ]:


import warnings
from pandas import Series
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

def evaluate_arima_model(X, arima_order):    
    train_size = int(len(X) * 0.75)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
        
    error = mean_squared_error(test, predictions)
    return error

def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.3f' % (order,mse))
                except:
                    continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))

p_values = [1, 4, 8]
d_values = range(0, 2)
q_values = range(0, 2)
warnings.filterwarnings("ignore")
evaluate_models(series.values, p_values, d_values, q_values)

