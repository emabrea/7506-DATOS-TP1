

# coding: utf-8

# # Analisis Exploratorio del set de datos

# In[1]:


#Import libraries
import pandas as pd
import numpy as np
#Plots
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Load the Data and take a quick look
trocafone = pd.read_csv('events.csv', low_memory=False)
trocafone.tail()


# In[3]:


# Information about the dataset
trocafone.info()


# In[4]:


# Some stats about the numeric columns in our dataset
trocafone.describe()


# In[5]:


# Name of columns
trocafone.columns


# In[6]:


trocafone.dtypes


# In[7]:


events = trocafone['event']
print(events.size == events.count())


# In[8]:


events_vc=trocafone['event'].value_counts()
events_vc


# ## Eventos generados por usuarios

# In[9]:


fig, ax = plt.subplots(figsize=(18, 14))
ax = sns.barplot(x=events_vc.values, y=events_vc.index)
ax.set_title("Eventos generados por usuarios", fontsize=22)
ax.set_xlabel("Frecuencia", fontsize=20)
ax.set_ylabel("Evento", fontsize=20)
ax.tick_params(labelsize=16)
fig.tight_layout()


rects = ax.patches

# For each bar: Place a label
for rect in rects:
    # Get X and Y placement of label from rect.
    x_value = rect.get_width()
    y_value = rect.get_y() + rect.get_height() / 2

    # Number of points between bar and label. Change to your liking.
    space = 5
    # Vertical alignment for positive values
    ha = 'left'

    # Use X value as label and format number with one decimal place
    label = x_value.astype(int)

    # Create annotation
    plt.annotate(
        label,                      # Use `label` as label
        (x_value, y_value),         # Place label at end of the bar
        xytext=(space, 0),          # Horizontally shift label by `space`
        textcoords="offset points", # Interpret `xytext` as offset in points
        va='center',                # Vertically center label
        ha=ha)         


# # Análisis temporal

# ## Distribucion de eventos por hora
# A continuación se analiza la cantidad de eventos ocurridos en cada hora del día.

# In[10]:


# Convert Date
import calendar
trocafone['timestamp'] = pd.to_datetime(trocafone['timestamp'])
trocafone['Year']=trocafone['timestamp'].map(lambda x:x.year)
trocafone['Month'] = trocafone['timestamp'].dt.month_name()
trocafone['day_of_week'] = trocafone['timestamp'].dt.weekday_name
trocafone['day'] = trocafone['timestamp'].map(lambda x: x.day)
trocafone['Hour'] = pd.to_datetime(trocafone['timestamp'], format='%H:%M',errors='coerce').dt.hour


# In[11]:


trocafone['Hour'].value_counts().sort_index().plot(kind='bar',figsize=(24,10))

plt.title("Distribución de eventos por hora",fontsize='25')
plt.xlabel('Hora',fontsize='22')
plt.ylabel('Eventos',fontsize='22')
plt.xticks(range(24),size = 15)
plt.yticks(size = 15)


# Como se observa, a partir de las 01:00 hay una reducción importante en la cantidad de eventos, los cuales entre las 14:00 y las 00:00 se mantienen casi constantes en valores altos. La hora con más eventos es entre las 22:00 y las 23:00.

# ### Tipos de eventos por hora

# In[12]:


trocafone.groupby(['Hour','event']).size().unstack().plot(kind='bar',stacked=True,figsize=(24,10),rot=0).legend(bbox_to_anchor=(0.4, 1.00),prop={'size': 15})

plt.title("Tipos de eventos por hora",fontsize='22')
plt.xlabel('Hora',fontsize='22')
plt.ylabel('Eventos',fontsize='22')
plt.xticks(size = 15)
plt.yticks(size = 15)


# ### Distribucion de eventos por dia de la semana
# A continuación se analizaran qué días de la semana tienen más eventos.

# In[13]:


#Create a column with days of the week
trocafone['day_of_week'] = pd.Categorical(trocafone['day_of_week'], categories=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday', 'Sunday'], ordered=True)
events_by_day = trocafone['day_of_week'].value_counts().sort_index().plot(kind='bar',figsize=(24,10),rot=0)

plt.title("Cantidad de eventos por día de la semana",fontsize='22')
plt.xlabel('Día de la semana',fontsize=20)
plt.ylabel('Eventos',fontsize=20)
plt.xticks(size = 15)
plt.yticks(size = 15)


# In[14]:


df_week=trocafone['day_of_week'].value_counts().to_frame()
promedio_semana=df_week.iloc[0:5].mean()
promedio_fin_semana=df_week.iloc[6:8].mean()
diferencia=promedio_semana-promedio_fin_semana
[promedio_semana,promedio_fin_semana, diferencia]


# Se observa que el día martes es el más ocupado, y a partir de ahí se reduce gradualmente la cantidad de eventos, para luego aumentar nuevamente el lunes. Es de destacar que los fines de semana hay muy poca actividad en relación a los demás días (56% menos).

# ### Distribución de compras por _hora del dia_ y _dia del mes_.
# 
# Continuando con el analisis temporal, se realizó un Heatmap para  evidenciar la distribución de eventos durante el mes.

# In[15]:


trocafone['nonNaN'] = trocafone['event'].map(lambda x : 1) #Defino una columna para poder sumar

# llevamos a una representacion de ese tipo usando una tabla pivot.
for_heatmap = trocafone.pivot_table(index='Hour', columns='day' , values = 'nonNaN' , aggfunc = 'sum')


# In[16]:


dims = (12, 10)
fig, ax = plt.subplots(figsize=dims)
g = sns.heatmap(for_heatmap.T , cmap="YlGnBu")
g.set_title("Cantidad de eventos (Primer semestre 2018)", fontsize=20)
g.set_xlabel("Hora del día",fontsize=13)
g.set_ylabel("Día del mes", fontsize=13)


# Se puede observar un descenso en la cantidad de eventos en las horas de la mañana y un incremento particularmente interesante
# en los dias 14 y 15 del mes, lo cual podría ser de utilidad a la hora de elegir la fecha para las ofertas o promociones mensuales.

# ### Distribucion de compras por dia de la semana
# Continuando con el análisis de la sección anterior, se filtran los eventos que resultaron en compras (conversion).

# In[17]:


trocafone.loc[trocafone['event']=='conversion']['day_of_week'].value_counts().sort_index().plot(kind='bar',figsize=(24,10),rot=0)

plt.title("Cantidad de compras por día de la semana",fontsize='22')
plt.xlabel('Día de la semana',fontsize=20)
plt.ylabel('Compras',fontsize=20)
plt.xticks(size = 15)
plt.yticks(size = 15) 


# Se observa que la cantidad de compras mantiene la relación con la cantidad de eventos. Es por esto que los martes son los días donde hubo más compras, mientras que en los fines de semana se registraron pocas compras.

# ### Evolución de la cantidad de eventos a lo largo del año
# Se procede a realizar un histograma mostrando la evolución en la cantidad de eventos a lo largo del año.

# In[18]:


trocafone.sort_values(by='timestamp')['timestamp'].hist(figsize=(24,10),bins=166)

plt.title("Cantidad de eventos a lo largo del año",fontsize='22')
plt.xlabel('Fecha',fontsize=20)
plt.ylabel('Eventos',fontsize=20)
plt.xticks(size = 15)
plt.yticks(size = 15) 


# A partir del gráfico se deduce que, o bien los primeros 4 meses no hubo una cantidad significativa de eventos, o estos no fueron registrados. Igualmente, En los meses de Mayo y Junio se observa un gran incremento en la cantidad de eventos, con un gran salto a mitad del mes de Mayo. Se necesitarán más datos para predecir el progreso en los futuros meses.

# ### Evolución de la cantidad de compras a lo largo del año

# In[19]:


trocafone.loc[trocafone["event"] == "conversion"].sort_values(by='timestamp')['timestamp'].hist(figsize=(24,10),bins=166,color='green')

plt.title("Cantidad de compras a lo largo del año",fontsize='22')
plt.xlabel('Fecha',fontsize=20)
plt.ylabel('Compras',fontsize=20)
plt.xticks(size = 15)
plt.yticks(size = 15) 


# In[20]:


trocafone['Month'] = pd.Categorical(trocafone['Month'], categories=['January','February','March','April','May','June'], ordered=True)
df=trocafone.groupby('Month').size().to_frame()
df['compras']=trocafone.loc[trocafone['event']=='conversion'].groupby('Month').size()
df.columns = ['total', 'compras']
df['relacion']=df['compras']/df['total']

df['relacion'].plot(kind='bar',figsize=(24,10),rot=0)
plt.title("Relacion entre eventos y compras",fontsize='22')
plt.xlabel('Mes',fontsize=20)
plt.ylabel('Compras/Eventos',fontsize=20)
plt.xticks(size = 15)
plt.yticks(size = 15) 


# ### Evolución de la cantidad de eventos a lo largo del mes

# In[21]:


trocafone["day"].value_counts().sort_index().plot(kind='bar',figsize=(24,10),rot=0)

plt.title("Cantidad de eventos a lo largo del mes",fontsize='22')
plt.xlabel('Día del mes',fontsize=20)
plt.ylabel('Eventos',fontsize=20)
plt.xticks(size = 15)
plt.yticks(size = 15) 


# In[22]:


trocafone.loc[(trocafone['event']=='conversion') | (trocafone['event']=='lead')].groupby(['Month','event']).size().unstack().plot(kind='bar',figsize=(24,10),rot=0).legend(prop={'size': 22})

plt.title("Relacion compras/subscripciones por mes",fontsize='22')
plt.xlabel('Mes',fontsize='22')
plt.ylabel('Compras/Subscripciones',fontsize='22')
plt.xticks(size = 20)
plt.yticks(size = 20)
plt.grid(axis='y')


# ## Modelos mas vendidos

# In[23]:


#Top Models 
grouped = trocafone.groupby('event')
compras = grouped.get_group('conversion')
top_compras = compras['model'].value_counts().head(20)
top_compras


# In[24]:


fig, ax = plt.subplots(figsize=(18, 14))
ax = sns.barplot(x = top_compras.values , y = top_compras.index)
ax.set_title("Compras por Modelo", fontsize=25)
ax.set_xlabel("Cantidad de Compras", fontsize=22)
ax.set_ylabel("Modelo", fontsize=22)
fig.tight_layout()
ax.tick_params(labelsize=14)
rects = ax.patches

# For each bar: Place a label
for rect in rects:
    # Get X and Y placement of label from rect.
    x_value = rect.get_width()
    y_value = rect.get_y() + rect.get_height() / 2

    # Number of points between bar and label. Change to your liking.
    space = 5
    # Vertical alignment for positive values
    ha = 'left'    

    # Use X value as label and format number with one decimal place
    label = x_value.astype(int)

    # Create annotation
    plt.annotate(
        label,                      # Use `label` as label
        (x_value, y_value),         # Place label at end of the bar
        xytext=(space, 0),          # Horizontally shift label by `space`
        textcoords="offset points", # Interpret `xytext` as offset in points
        va='center',                # Vertically center label
        ha=ha)  


# In[25]:



celulares=trocafone.loc[trocafone['event']=='conversion']
celulares=celulares['model'].dropna().str.split().str.get(0)
celulares=celulares.value_counts()

fig, ax = plt.subplots(figsize=(24, 10))
ax = sns.barplot(x = celulares.values , y = celulares.index)
ax.set_title("Compras por Marca", fontsize=25)
ax.set_xlabel("Cantidad de compras", fontsize=22)
ax.set_ylabel("Modelo", fontsize=22)
fig.tight_layout()
ax.tick_params(labelsize=14)

rects = ax.patches

# For each bar: Place a label
for rect in rects:
    # Get X and Y placement of label from rect.
    x_value = rect.get_width()
    y_value = rect.get_y() + rect.get_height() / 2

    # Number of points between bar and label. Change to your liking.
    space = 5
    # Vertical alignment for positive values
    ha = 'left'   

    # Use X value as label and format number with one decimal place
    label = x_value.astype(int)

    # Create annotation
    plt.annotate(
        label,                      # Use `label` as label
        (x_value, y_value),         # Place label at end of the bar
        xytext=(space, 0),          # Horizontally shift label by `space`
        textcoords="offset points", # Interpret `xytext` as offset in points
        va='center',                # Vertically center label
        ha=ha)


# In[26]:


celulares=trocafone['model'].dropna().str.split().str.get(0)
celulares=celulares.value_counts()

fig, ax = plt.subplots(figsize=(24, 10))
ax = sns.barplot(x = celulares.values , y = celulares.index)
ax.set_title("Visitas por Modelo", fontsize=25)
ax.set_xlabel("Cantidad de visitas", fontsize=22)
ax.set_ylabel("Modelo", fontsize=22)
fig.tight_layout()
ax.tick_params(labelsize=14)

rects = ax.patches

# For each bar: Place a label
for rect in rects:
    # Get X and Y placement of label from rect.
    x_value = rect.get_width()
    y_value = rect.get_y() + rect.get_height() / 2

    # Number of points between bar and label. Change to your liking.
    space = 5
    # Vertical alignment for positive values
    ha = 'left'

    # If value of bar is negative: Place label left of bar
    if x_value < 0:
        # Invert space to place label to the left
        space *= -1
        # Horizontally align label at right
        ha = 'right'

    # Use X value as label and format number with one decimal place
    label = x_value.astype(int)

    # Create annotation
    plt.annotate(
        label,                      # Use `label` as label
        (x_value, y_value),         # Place label at end of the bar
        xytext=(space, 0),          # Horizontally shift label by `space`
        textcoords="offset points", # Interpret `xytext` as offset in points
        va='center',                # Vertically center label
        ha=ha)


# In[27]:


trocafone['marcaCel'] =trocafone['model'].dropna().str.split().str.get(0)

df=trocafone.loc[(trocafone['event']=='conversion')].groupby(['Month','marcaCel']).size().unstack()
df.drop([col for col, val in df.sum().iteritems() if val < 50], axis=1, inplace=True)
df.plot(kind='bar',figsize=(24,10),rot=0).legend(prop={'size': 22})

plt.title("Marcas más compradas por mes",fontsize='22')
plt.xlabel('Mes',fontsize='22')
plt.ylabel('Compras',fontsize='22')
plt.xticks(size = 20)
plt.yticks(size = 20)
plt.grid(axis='y')


# In[28]:


capacidad= trocafone.loc[trocafone['storage'].notnull()]
capacidad['storage'].value_counts().plot(kind="bar",rot=0, figsize=(24,10))

plt.xlabel('Capacidad',fontsize='22')
plt.ylabel('Visitas',fontsize='22')
plt.title("Celulares vistos por capacidad",fontsize='22')
plt.xticks(size = 20)
plt.yticks(size = 20)


# In[29]:


capacidad= trocafone.loc[(trocafone['storage'].notnull()) & (trocafone['event']=='conversion')]
capacidad['storage'].value_counts().plot(kind="bar", rot=0,figsize=(24,10))

plt.xlabel('Capacidad',fontsize='22')
plt.ylabel('Compras',fontsize='22')
plt.title("Celulares comprados por capacidad",fontsize='22')
plt.xticks(size = 20)
plt.yticks(size = 20)


# In[30]:


campaña=trocafone.groupby(['Month','person','event'])
campaña=campaña.size().unstack()
campaña=campaña.loc[(campaña['ad campaign hit'].notnull()) & (campaña['conversion'].notnull())]
campaña.groupby('Month').size().plot(kind='bar',rot=0,figsize=(24,8)) 

plt.xlabel('Mes',fontsize='22')
plt.ylabel('Usuarios',fontsize='22')
plt.title("Usuarios que compraron a partir de una campaña",fontsize='22')
plt.xticks(size = 20)
plt.yticks(size = 20)


# ### Cantidad de eventos por tipo de dispositivo

# In[31]:


device_types=trocafone[(trocafone['device_type']!='Unknown') & (trocafone['device_type'].notnull())]
device_types['device_type'].value_counts().plot(kind='bar',rot=0,figsize=(24,8)) 

plt.xlabel('Dispositivo',fontsize='22')
plt.ylabel('Eventos',fontsize='22')
plt.title("Tipos de dispositivos utilizados",fontsize='22')
plt.xticks(size = 20)
plt.yticks(size = 20)


# In[32]:


device_types['event'].nunique()
#No hay datos sobre tipos de dispositivos con otros eventos


# # Analisis Geografico

# ### Paises con mas usuarios

# In[33]:


countries = trocafone.loc[trocafone["country"] != "Unknown"]
countries = countries.drop_duplicates(subset=['person', 'country'])
countries = countries["country"].value_counts().to_frame("cantidad")
countries.index.title = "pais"
countries = countries.assign(porcentaje=countries/countries.sum()*100)
countries.head(10)


# ### Ciudades con mas usuarios

# In[34]:


cities = trocafone.loc[trocafone["city"] != "Unknown"]
cities = cities.drop_duplicates(subset=['person', 'city'])
cities = cities["city"].value_counts().to_frame("cantidad")
cities.index.title = "ciudad"
cities = cities.assign(porcentaje=cities/cities.sum()*100)
cities.head(10)


# In[35]:


fig, ax = plt.subplots(figsize=(20, 12))
ax = sns.barplot(x=cities.head(10).index, y=cities.head(10)["cantidad"])
ax.set_title("Ciudades con más usuarios", fontsize=25)
ax.set_xlabel("Ciudad", fontsize=22)
ax.set_ylabel("Usuarios", fontsize=22)
ax.tick_params(labelsize=14)


# ### Ciudades con mas eventos

# In[36]:


cities = trocafone.loc[trocafone["city"] != "Unknown"]
cities = cities["city"].value_counts().to_frame("frecuencia")
cities.index.title = "ciudad"
fig, ax = plt.subplots(figsize=(20, 12))
ax = sns.barplot(x=cities.head(10).index, y=cities.head(10)["frecuencia"])
ax.set_title("Ciudades con más eventos", fontsize=25)
ax.set_xlabel("Ciudad", fontsize=22)
ax.set_ylabel("Eventos", fontsize=22)
ax.tick_params(labelsize=14)


# ### Paises con mas eventos

# In[37]:


countries = trocafone.loc[trocafone["country"] != "Unknown"]
countries = countries["country"].value_counts().to_frame("cantidad")
countries.index.title = "pais"
countries = countries.assign(porcentaje=countries/countries.sum()*100)
countries.head(10)


# In[38]:


cities_loc = pd.read_csv("coordinates.csv")
cities_loc = cities_loc.dropna()


# In[39]:


import folium

cm = plt.get_cmap("winter")

folium_map = folium.Map(tiles="Mapbox Bright", location=(0,0), zoom_start=2.47)

for city in cities_loc.values:
    marker = folium.CircleMarker(location=[city[2], city[3]], radius=1, color='red', opacity=0.5)
    marker.add_to(folium_map)
folium_map.zoom_control = False
folium_map


# # Analisis del usuario

# ### Cantidad de eventos por usuario

# In[40]:


eventos_por_persona = trocafone.groupby('person')['event'].count()
max = eventos_por_persona.max()
min = eventos_por_persona.min()
promedio = eventos_por_persona.mean()
media = eventos_por_persona.median()
std = eventos_por_persona.std()
d = {"promedio":promedio, "media":media, "std":std, "max":max, "min":min}
pd.DataFrame(data=d, index=["eventos por persona"])


# ## Navegadores preferidos

# In[41]:


browser=trocafone['browser_version'].dropna().str.replace('\d+', '').str.replace('.', '')
browser_df=browser.value_counts().to_frame()
browser_df['percentages']=browser_df['browser_version']/browser_df['browser_version'].sum()*100
browser_df


# In[42]:


browser_df['percentages'].head(5).plot(kind='bar',rot=0,figsize=(24,8)) 

plt.title("Tipos de navegadores utilizados",fontsize='22')
plt.xlabel('Navegador',fontsize='22')
plt.ylabel('Eventos(%)',fontsize='22')
plt.xticks(size = 20)
plt.yticks(size = 20)


# ## Analisis de usuarios que compran

# ### Cantidad de personas que compraron

# In[43]:


trocafone.loc[trocafone['event']=='conversion']['person'].nunique()


# ### Promedio de visitas de personas que compran

# In[44]:


compras=trocafone.loc[trocafone['event']=='conversion'].groupby('person')
compras.size().mean()


# ### Máximo de compras de una persona

# In[45]:


compras.size().max()


# In[46]:


compras.size().hist(figsize=(24,15),bins=716,width=1)

plt.title("Compras por usuario",fontsize='22')
plt.xlabel('Compras',fontsize=20)
plt.ylabel('Usuarios',fontsize=20)
plt.xticks(size = 15)
plt.yticks(size = 15) 


# ### Porcentaje de usuarios que realizaron solo una compra

# In[47]:


(compras.size()==1).sum()/len(compras)*100


# ### Cantidad de personas que realizaron ckeckouts

# In[48]:


checkouts=trocafone.loc[trocafone['event']=='checkout'].groupby('person')
len(checkouts)


# ### Promedio de checkouts por persona

# In[49]:


checkouts.size().mean()


# ### Relación entre personas que hicieron checkouts y personas que compraron (%)

# In[50]:


len(compras)/len(checkouts)*100


# ### Relación entre total de checkouts y total de compras (%)

# In[51]:


total_checkouts=trocafone.loc[trocafone['event']=='checkout']
len(total_checkouts)


# In[52]:


total_compras=trocafone.loc[trocafone['event']=='conversion']
len(total_compras)


# In[53]:


len(total_compras)/len(total_checkouts)*100


# ### Cantidad de productos vistos por persona

# In[54]:


vistos_por_persona = trocafone.loc[trocafone['event'] == 'viewed product'].groupby('person')['sku'].nunique()
max = vistos_por_persona.max()
min = vistos_por_persona.min()
promedio = vistos_por_persona.mean()
media = vistos_por_persona.median()
std = vistos_por_persona.std()
d = {"promedio":promedio, "media":media, "std":std, "max":max, "min":min}
pd.DataFrame(data=d, index=["productos vistos por persona"])


# ### Usuarios nuevos

# In[55]:


primeros_ingresos = trocafone.groupby(['person'])['timestamp'].min().to_frame()
primeros_ingresos['timestamp'] = primeros_ingresos['timestamp'].map(lambda x: x.date())
primeros_ingresos_por_dia = primeros_ingresos.groupby('timestamp').agg('size')
primeros_ingresos_por_dia.plot(figsize=(24,10))
plt.title("Ingreso de usuarios nuevos",fontsize='22')
plt.xlabel('Tiempo',fontsize=20)
plt.ylabel('Cantidad de usuarios',fontsize=20)
plt.xticks(size = 15)
plt.yticks(size = 15) 
plt.grid()


# ### Proporcion de usuarios que ingresan por un buscador (%)

# In[56]:


buscadores = trocafone.loc[trocafone['event'] == 'search engine hit']
buscadores.index.size / trocafone.index.size * 100


# #### Buscadores mas utilizados

# In[57]:


frec_buscadores = buscadores['search_engine'].value_counts().to_frame('frecuencia')
frec_buscadores.index.title = 'Buscador'
frec_buscadores['porcentaje'] = buscadores['search_engine'].value_counts(normalize=True)*100
frec_buscadores


# ### Usuarios que ingresan por una campaña publicitaria (%)

# In[58]:


publicidad = trocafone.loc[trocafone['event'] == 'ad campaign hit']
publicidad.index.size / trocafone.index.size * 100


# In[59]:


frec_publicidad = publicidad['campaign_source'].value_counts(normalize=True).to_frame('porcentaje')
frec_publicidad.index.title = 'Buscador'
frec_publicidad['porcentaje'] = frec_publicidad['porcentaje']*100
frec_publicidad.head(10).plot(kind='bar', figsize=(20, 15),rot=0)
plt.title("Ingreso de usuarios por publicidad",fontsize='22')
plt.xlabel('Campaña',fontsize=20)
plt.ylabel('Porcentaje',fontsize=20)
plt.xticks(size = 15)
plt.yticks(size = 15) 
plt.grid(axis='y')

