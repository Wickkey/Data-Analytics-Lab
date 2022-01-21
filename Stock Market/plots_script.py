#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf #matplotlibfinance


# In[2]:


# plt.style.use(['science'])


# In[3]:


df_cognizant = pd.read_csv('Datasets/Cognizant share prices 2019_2021.csv',parse_dates = ['Date'],infer_datetime_format=True,index_col='Date')
df_hcl = pd.read_csv('Datasets/HCL Technologies share prices 2019_2021.csv',parse_dates = ['Date'],infer_datetime_format=True,index_col='Date')
df_hdfc = pd.read_csv('Datasets/HDFC Bank Share Prices 2019_2021.csv',parse_dates = ['Date'],infer_datetime_format=True,index_col='Date')
df_icici = pd.read_csv('Datasets/ICICI Bank Share Prices 2019_2021.csv',parse_dates = ['Date'],infer_datetime_format=True,index_col='Date')
df_infosys = pd.read_csv('Datasets/Infosys Share Prices 2019_2021.csv',parse_dates = ['Date'],infer_datetime_format=True,index_col='Date')
df_sbi = pd.read_csv('Datasets/SBI Share Prices 2019_2021.csv',parse_dates = ['Date'],infer_datetime_format=True,index_col='Date')
df_inr = pd.read_csv('Datasets/USD-INR Exchange rate 2019_2021.csv',parse_dates = ['Date'],infer_datetime_format=True,index_col='Date')


# ## Remove null rows

# In[4]:


df_cognizant.dropna(inplace = True)
df_hcl.dropna(inplace=True)
df_hdfc.dropna(inplace =True)
df_icici.dropna(inplace=True)
df_infosys.dropna(inplace=True)
df_sbi.dropna(inplace=True)
df_inr.dropna(inplace=True)


# ## Check for null values

# In[5]:


print(df_cognizant.isnull().sum())
print(df_cognizant.dtypes)


# In[6]:


print(df_hcl.isnull().sum())
print(df_hcl.dtypes)


# In[7]:


print(df_hdfc.isnull().sum())
print(df_hdfc.dtypes)


# In[8]:


print(df_icici.isnull().sum())
print(df_icici.dtypes)


# In[9]:


print(df_infosys.isnull().sum())
print(df_infosys.dtypes)


# In[10]:


print(df_sbi.isnull().sum())
print(df_sbi.dtypes)


# In[11]:


print(df_inr.isnull().sum())
print(df_inr.dtypes)


# ## plot

# In[69]:


fig,ax = mpf.plot(df_cognizant ,volume=True,type='line',mav=50,tight_layout=True,style='classic',returnfig=True,title='Cognizant')
ax[0].legend(['Stock Price','SMA(30)'])
ax[0].set_ylabel('Cognizant Stock Price')
fig.savefig('Cognizant_linechart.png')


# In[31]:


fig,ax =mpf.plot(df_hcl ,volume=True,type='line',mav=50,tight_layout=True,style='classic',title='HCL',returnfig=True)
ax[0].legend(['Stock Price','SMA(30)'])
ax[0].set_ylabel('HCL Stock Price')


# In[32]:


fig,ax =mpf.plot(df_hdfc ,volume=True,type='line',mav=50,tight_layout=True,style='classic',title='HDFC',returnfig=True)
ax[0].legend(['Stock Price','SMA(30)'])
ax[0].set_ylabel('HDFC Stock Price')


# In[33]:


fig,ax =mpf.plot(df_infosys ,volume=True,type='line',mav=50,tight_layout=True,style='classic',title='Infosys',returnfig=True)
ax[0].legend(['Stock Price','SMA(30)'])
ax[0].set_ylabel('Infosys Stock Price')


# In[34]:


fig,ax =mpf.plot(df_sbi ,volume=True,type='line',mav=50,tight_layout=True,style='classic',title='SBI',returnfig=True)
ax[0].legend(['Stock Price','SMA(30)'])
ax[0].set_ylabel('SBI Stock Price')


# In[35]:


fig,ax =mpf.plot(df_icici ,volume=True,type='line',mav=50,tight_layout=True,style='classic',title='ICICI',returnfig=True)
ax[0].legend(['Stock Price','SMA(30)'])
ax[0].set_ylabel('Icici Stock Price')


# In[ ]:

fig,ax =mpf.plot(df_inr ,type='line',mav=50,tight_layout=True,style='classic',title='USD-INR',returnfig=True)
ax[0].legend(['Stock Price','SMA(30)'])
ax[0].set_ylabel('INR USD Price')


