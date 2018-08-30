
# coding: utf-8

# In[1]:


#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


#Load the Data and take a quick look
df = pd.read_csv('events.csv', low_memory=False)
df.tail()


# In[9]:


# Information about the dataset
df.info()


# In[11]:


# Some stats about the numeric columns in our dataset
df.describe(include="all")


# In[12]:


# Name of columns
df.columns


# In[3]:


events = df['event']
print(events.size == events.count())


# In[4]:


events_vc = df['event'].value_counts()
events_vc


# In[5]:


g = sns.barplot(x=events_vc.values, y=events_vc.index)
g.set_title("Eventos generados por usuarios.", fontsize=18)
g.set_xlabel("Frecuencia", fontsize=18)
g.set_ylabel("Evento", fontsize=18)

