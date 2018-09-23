
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


events_vc=trocafone['event'].value_counts()
events_vc


# # Análisis temporal

# ## Distribucion de eventos por hora
# A continuación se analiza la cantidad de eventos ocurridos en cada hora del día.

# In[6]:


# Convert Date
import calendar
trocafone['timestamp'] = pd.to_datetime(trocafone['timestamp'])
trocafone['Year']=trocafone['timestamp'].map(lambda x:x.year)
trocafone['Month'] = trocafone['timestamp'].dt.month_name()
trocafone['day_of_week'] = trocafone['timestamp'].dt.weekday_name
trocafone['day'] = trocafone['timestamp'].map(lambda x: x.day)
trocafone['Hour'] = pd.to_datetime(trocafone['timestamp'], format='%H:%M',errors='coerce').dt.hour


# In[7]:


trocafone['Hour'].value_counts().sort_index().plot(kind='bar',rot=0,figsize=(24,10))

plt.title("Distribución de eventos por hora",fontsize='25')
plt.xlabel('Hora',fontsize='22')
plt.ylabel('Eventos',fontsize='22')
plt.xticks(range(24),size = 15)
plt.yticks(size = 15)


# Como se observa, a partir de las 01:00 hay una reducción importante en la cantidad de eventos, los cuales entre las 14:00 y las 00:00 se mantienen casi constantes en valores altos. La hora con más eventos es entre las 22:00 y las 23:00.

# ### Tipos de eventos por hora

# In[8]:


eventos_por_hora = trocafone.groupby(['Hour','event']).size().unstack()

eventos_por_hora.plot(kind='bar',stacked=True,figsize=(24,10),rot=0).legend(bbox_to_anchor=(0.4, 1.00),prop={'size': 15})

plt.title("Tipos de eventos por hora",fontsize='22')
plt.xlabel('Hora',fontsize='22')
plt.ylabel('Eventos',fontsize='22')
plt.xticks(size = 15)
plt.yticks(size = 15)


# En este gráfico de barras apiladas se evidencia el predominio del evento "viewed product" frente al resto, siendo de un orden de magnitud mayor al evento que lo sigue en cantidad, podríamos filtrar a dicho evento para poder apreciar mejor la relación de los demás:

# In[9]:


eventos_por_hora


# In[10]:


filter_viewed_product = trocafone.loc[trocafone['event'] != 'viewed product']                                .groupby(['Hour','event']).size().unstack()

#Defino las filas cuyo evento es viewed product 


# In[11]:


fig, ax = plt.subplots(figsize=(18,6))
labels = filter_viewed_product.columns
ax.stackplot(filter_viewed_product.index , [filter_viewed_product[column] for column in filter_viewed_product.columns], labels=labels)
plt.legend(loc= (0.25,0.3) , prop={'size': 12})
ax.xaxis.set_ticks(filter_viewed_product.index)
plt.xlabel('Hora del día',fontsize=20)
plt.ylabel('Eventos',fontsize=20)
plt.xticks(size = 12)
plt.yticks(size = 12)

plt.show()


# En este gráfico se pueden apreciar mejor las similitudes entre algunos eventos , mostramos un gráfico ilustrando algunos de los eventos de mayor ocurrencia durante el dia (excluyendo a "viewed product")

# In[12]:


fig, ax = plt.subplots(figsize=(8,6))
plt.plot(eventos_por_hora['visited site'] , 'lightblue' , linestyle = 'dashed' , linewidth = 3)
plt.plot(eventos_por_hora['brand listing'] , 'orange' ,linestyle = '-.' , linewidth = 3)
plt.plot(eventos_por_hora['searched products'] , 'go')
plt.plot(eventos_por_hora['generic listing'] , 'darkviolet',linestyle='-.',linewidth = 3)
plt.legend(prop={'size': 12})
ax.xaxis.set_ticks(eventos_por_hora.index)
plt.xlabel('Hora del día',fontsize=20)
plt.ylabel('Eventos',fontsize=20)
plt.xticks(size = 12)
plt.yticks(size = 12)
plt.grid()


# En el gráfico se aprecian los diferentes eventos de busqueda, pudiendo ver una similitud en su tendencia temporal.

# ### Evolución de la cantidad de eventos a lo largo del mes

# In[13]:


trocafone["day"].value_counts().sort_index().plot(kind='bar',figsize=(24,10),rot=0 )

plt.title("Cantidad de eventos a lo largo del mes",fontsize='22')
plt.xlabel('Día del mes',fontsize=20)
plt.ylabel('Eventos',fontsize=20)
plt.xticks(size = 15)
plt.yticks(size = 15) 


# ### Distribución de compras por _hora del dia_ y _dia del mes_.
# 
# Continuando con el analisis temporal, se realizó un Heatmap para  evidenciar la distribución de eventos durante el mes.

# In[14]:


trocafone['nonNaN'] = trocafone['event'].map(lambda x : 1) #Defino una columna para poder sumar

# llevamos a una representacion de ese tipo usando una tabla pivot.
for_heatmap = trocafone.pivot_table(index='Hour', columns='day' , values = 'nonNaN' , aggfunc = 'sum')


# In[15]:


dims = (11, 9)
fig, ax = plt.subplots(figsize=dims)
g = sns.heatmap(for_heatmap.T , cmap="YlGnBu")
g.set_title("Cantidad de eventos (Primer semestre 2018)", fontsize=20)
g.set_xlabel("Hora del día",fontsize=13)
g.set_ylabel("Día del mes", fontsize=13)


# Se puede observar un descenso en la cantidad de eventos en las horas de la mañana y un incremento particularmente interesante
# en los dias 14 y 15 del mes, lo cual podría ser de utilidad a la hora de elegir la fecha para las ofertas o promociones mensuales.

# ### Distribucion de eventos por dia de la semana
# A continuación se analizaran qué días de la semana tienen más eventos.

# In[16]:


#Create a column with days of the week
trocafone['day_of_week'] = pd.Categorical(trocafone['day_of_week'], categories=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday', 'Sunday'], ordered=True)
events_by_day = trocafone['day_of_week'].value_counts().sort_index().plot(kind='bar',figsize=(24,10),rot=0)

plt.title("Cantidad de eventos por día de la semana",fontsize='22')
plt.xlabel('Día de la semana',fontsize=20)
plt.ylabel('Eventos',fontsize=20)
plt.xticks(size = 15)
plt.yticks(size = 15)


# In[17]:


df_week=trocafone['day_of_week'].value_counts().to_frame()
promedio_semana=df_week.iloc[0:5].mean()
promedio_fin_semana=df_week.iloc[6:8].mean()
diferencia=promedio_semana-promedio_fin_semana
print("El promedio de visitas de lunes a viernes es",promedio_semana.values[0],", mientras que los fines de semana es",promedio_fin_semana.values[0])


# Se observa que el día martes es el más ocupado, y a partir de ahí se reduce gradualmente la cantidad de eventos, para luego aumentar nuevamente el lunes. Es de destacar que los fines de semana hay muy poca actividad en relación a los demás días.

# ### Distribucion de compras por dia de la semana
# Continuando con el análisis de la sección anterior, se filtran los eventos que resultaron en compras (conversion).

# In[18]:


trocafone.loc[trocafone['event']=='conversion']['day_of_week'].value_counts().sort_index().plot(kind='bar',figsize=(24,10),rot=0)

plt.title("Cantidad de compras por día de la semana",fontsize='22')
plt.xlabel('Día de la semana',fontsize=20)
plt.ylabel('Compras',fontsize=20)
plt.xticks(size = 15)
plt.yticks(size = 15) 


# Se observa que la cantidad de compras mantiene la relación con la cantidad de eventos. Es por esto que los martes son los días donde hubo más compras, mientras que en los fines de semana se registraron pocas compras.

# ### Evolución de la cantidad de eventos a lo largo del año
# Se procede a realizar un histograma mostrando la evolución en la cantidad de eventos a lo largo del año.

# In[19]:


trocafone.sort_values(by='timestamp')['timestamp'].hist(figsize=(24,10),bins=166)

plt.title("Cantidad de eventos a lo largo del año",fontsize='22')
plt.xlabel('Fecha',fontsize=20)
plt.ylabel('Eventos',fontsize=20)
plt.xticks(size = 15)
plt.yticks(size = 15) 


# A partir del gráfico se deduce que, o bien los primeros 4 meses no hubo una cantidad significativa de eventos, o estos no fueron registrados. Igualmente, En los meses de Mayo y Junio se observa un gran incremento en la cantidad de eventos, con un gran salto a mitad del mes de Mayo. Se necesitarán más datos para predecir el progreso en los futuros meses.

# ### Evolución de la cantidad de compras a lo largo del año

# In[20]:


trocafone.loc[trocafone["event"] == "conversion"].sort_values(by='timestamp')['timestamp'].hist(figsize=(24,10),bins=166,color='green')

plt.title("Cantidad de compras a lo largo del año",fontsize='22')
plt.xlabel('Fecha',fontsize=20)
plt.ylabel('Compras',fontsize=20)
plt.xticks(size = 15)
plt.yticks(size = 15) 


# Si bien los ultimos meses aumentaron visitas, no lo hicieron las compras diarias, que mantienen un umbral de 20 compras por dia. Dicho umbral se rompio solo en 4 ocasiones muy cercanas, tal vez debido a alguna promocion.

# In[21]:


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


# Aca se obsera lo destacado anteriormente, la relacion entre visitas y compras fue disminuyendo con los meses.

# In[22]:


trocafone.loc[(trocafone['event']=='conversion') | (trocafone['event']=='lead')]        .groupby(['Month','event']).size().unstack()        .plot(kind='bar',figsize=(24,10),rot=0).legend(prop={'size': 22})

plt.title("Relacion compras/subscripciones por mes",fontsize='22')
plt.xlabel('Mes',fontsize='22')
plt.ylabel('Compras/Subscripciones',fontsize='22')
plt.xticks(size = 20)
plt.yticks(size = 20)
plt.grid(axis='y')


# Se observa que las compras tienen una tendencia alcista, pero no es asi con las subscripciones, que van variando mucho en cada mes. Es de destacar que el ultimo mes, casi todas las compras resultaron en subscripciones, lo cual indica algun cambio beneficioso.

# ### Usuarios nuevos

# In[23]:


primeros_ingresos = trocafone.groupby(['person'])['timestamp'].min().to_frame()
primeros_ingresos['timestamp'] = primeros_ingresos['timestamp'].map(lambda x: x.date())
primeros_ingresos_por_dia = primeros_ingresos.groupby('timestamp').agg('size')
primeros_ingresos_por_dia.plot(figsize=(24,10) , linewidth = 4)
plt.title("Ingreso de usuarios nuevos",fontsize='22')
plt.xlabel('Tiempo',fontsize=20)
plt.ylabel('Cantidad de usuarios',fontsize=20)
plt.xticks(size = 15)
plt.yticks(size = 15) 
plt.grid()


# La cantidad de usuarios nuevos que visitaron la pagina se mantuvo casi constante hasta mediados de mayo, donde, tal vez debido a una campaña publicitaria, hubo un gran salto de usuarios nuevos. Como en los ultimos dias hay maximos cada vez mas altos, la tendencia es alcista.

# # Análisis de eventos

# ### Cantidad de eventos por usuario

# In[24]:


eventos_por_persona = trocafone.groupby('person')['event'].count()
max = eventos_por_persona.max()
min = eventos_por_persona.min()
promedio = eventos_por_persona.mean()
media = eventos_por_persona.median()
std = eventos_por_persona.std()
d = {"promedio":promedio, "media":media, "std":std, "max":max, "min":min}
pd.DataFrame(data=d, index=["eventos por persona"])


# Veamos que pese a que la media es de 14, nuestro valor máximo es de 2771, lo que puede predecir que vamos a tener outliers, si graficamos la cantidad de veces que cada valor ocurre, obtenemos:

# - Previamente defino una funcion f que nos filtra los valores mayores a n:

# In[25]:


f = (lambda n : eventos_por_persona[eventos_por_persona < n]) 


# In[26]:


f(200).value_counts().sort_index()        .plot(color='r', figsize=(20,8) ,linewidth = 4,  label ='Usuarios por cantidad de eventos')

plt.title("Ingreso de usuarios nuevos",fontsize='22')
plt.ylabel('Cantidad de usuarios',fontsize=18)
plt.xlabel('Cantidad de eventos realizados',fontsize=18)
plt.legend(prop={'size': 15})
plt.xticks(size = 15)
plt.yticks(size = 15) 
plt.grid()


# Se ve una clara disminución en el numero de usuarios con mas de 75 eventos, tendiendo a 0 cuando se supera este valor.
# 
# - Esto hace que dichos valores sean desestimables frente a los mas de cien usuarios que presenta cada valor menor a 25
# 
# Por lo tanto podríamos filtrar dichos valores, para poder apreciar mejor en un violin plot, su distribución
# 
# (Adicionalmente realizo un violin plot filtrando los mayores a 30)

# In[27]:


f = (lambda n : eventos_por_persona[eventos_por_persona < n])
s1 , s2 = f(75) , f(30)


# In[28]:


fig, axs = plt.subplots(nrows=2 , figsize=(6,6))
plt.tight_layout(h_pad=3.0)
g1 =sns.violinplot(s1.values , orient = 'v' , ax=axs[0] , palette = 'Greens')
g2 =sns.violinplot(s2.values , orient = 'v' , ax=axs[1] , palette = 'Greens')

g1.set_title("Personas con menos de 75 eventos ", fontsize=15)
g1.set_ylabel("Eventos",fontsize=13)
g1.set_xlabel("Cantidad de personas", fontsize=13)

g2.set_title("Personas con menos de 30 eventos ", fontsize=15)
g2.set_ylabel("Eventos",fontsize=13)
g2.set_xlabel("Cantidad de personas", fontsize=13)


# Esta tabla puede ser acompañada por un boxplot, que nos ayuda a visualizar dichas entradas, junto con los outliers, de forma elegante y directa.

# ## Eventos generados por usuarios

# In[29]:


fig, ax = plt.subplots(figsize=(18, 14))
ax = sns.barplot(x=events_vc.values, y=events_vc.index , palette = 'rocket')
ax.set_title("Eventos generados por usuarios", fontsize=22)
ax.set_xlabel("Frecuencia absoluta", fontsize=20)
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


# ## Modelos mas vendidos

# In[30]:


#Top Models 
grouped = trocafone.groupby('event')
compras = grouped.get_group('conversion')
top_compras = compras['model'].value_counts().head(20)
top_compras


# In[31]:


fig, ax = plt.subplots(figsize=(18, 14))
ax = sns.barplot(x = top_compras.values , y = top_compras.index )
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


# En este grafico se ve que hay una gran variedad de modelos, de los cuales la mayoria se encuentra entre 17 y 30 modelos vendidos. Pero lo destacable es que solo aparecen 3 marcas, iPhone, Samsung y Motorola. Esta última solo con 2 modelos muy vendidos (Moto G3  y Moto G4 plus). Por otro lado, iPhone ocupa 4 de las primeras 6 posiciones.

# In[32]:



celulares=trocafone.loc[trocafone['event']=='conversion']
celulares=celulares['model'].dropna().str.split().str.get(0)
celulares=celulares.value_counts()

fig, ax = plt.subplots(figsize=(24, 10))
ax = sns.barplot(x = celulares.values , y = celulares.index , palette = "RdBu")
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


# Vemos que Samsung domina el escenario de modelos vendidos, duplicando los vendidos por iPhone. Seguramente por el precio y por la mayor variedad de modelos.

# In[33]:


celulares=trocafone['model'].dropna().str.split().str.get(0)
celulares=celulares.value_counts()

fig, ax = plt.subplots(figsize=(24, 10))
ax = sns.barplot(x = celulares.values , y = celulares.index , palette = 'cubehelix')
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


# Es interesante que aunque iPhone no son los más comprados, sí son los más vistos. Esto puede deberse a su diseño y fama de ser celulares de alta gama.

# In[34]:


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


# Las ventas de Samsung por mes van en aumento razonable, pero las de iPhone crecen muy lentamente, aproximadamente 10 ventas más que el mes pasado. Por otro lado, los Motorola son más dificil de predecir, aunque no suelen pasar las 50 ventas mensuales.

# In[35]:


capacidad= trocafone.loc[trocafone['storage'].notnull()]
capacidad['storage'].value_counts().plot(kind="bar", rot=0, figsize=(24,10) )

plt.xlabel('Capacidad',fontsize='22')
plt.ylabel('Visitas',fontsize='22')
plt.title("Celulares vistos por capacidad",fontsize='22')
plt.xticks(size = 20)
plt.yticks(size = 20)


# In[36]:


capacidad= trocafone.loc[(trocafone['storage'].notnull()) & (trocafone['event']=='conversion')]
capacidad['storage'].value_counts().plot(kind="bar", rot=0,figsize=(24,10))

plt.xlabel('Capacidad',fontsize='22')
plt.ylabel('Compras',fontsize='22')
plt.title("Celulares comprados por capacidad",fontsize='22')
plt.xticks(size = 20)
plt.yticks(size = 20)


# Es razonable que 16gb sea la capacidad más comprada, pero lo interesante es que ya hay muchas compras de celulares de 32gb. Esto se debe a que ultimamente los 8gb no alcanzan para almacenar la cantidad de informacion que se maneja.

# In[37]:


estado=trocafone.groupby('condition')
estado=estado.size().sort_values(ascending=False)
estado.rename(index={"Bom": 'Bueno',"Muito Bom":"Muy bueno","Bom - Sem Touch ID":"Bueno-Sin touch ID","Novo":"Nuevo"}).plot(kind='bar',rot=0,figsize=(24,8) ) 

plt.xlabel('Condición',fontsize='22')
plt.ylabel('Visitas',fontsize='22')
plt.title("Celulares vistos por condicion",fontsize='22')
plt.xticks(size = 20)
plt.yticks(size = 20)


# In[38]:


estado=trocafone.loc[trocafone['event']=='conversion']
estado=estado.groupby('condition')
estado=estado.size().sort_values(ascending=False)
estado.rename(index={"Bom": 'Bueno',"Muito Bom":"Muy bueno","Bom - Sem Touch ID":"Bueno-Sin touch ID","Novo":"Nuevo"}).plot(kind='bar',rot=0,figsize=(24,8)) 

plt.xlabel('Condición',fontsize='22')
plt.ylabel('Compras',fontsize='22')
plt.title("Celulares comprados por condición",fontsize='22')
plt.xticks(size = 20)
plt.yticks(size = 20)


# Vemos que a la gente no le molesta compras celulares en estado Bueno, seguramente prefiriendo el precio antes que el estado.

# In[39]:


campaña=trocafone.groupby(['Month','person','event'])
campaña=campaña.size().unstack()
campaña=campaña.loc[(campaña['ad campaign hit'].notnull()) & (campaña['conversion'].notnull())]
campaña.groupby('Month').size().plot(kind='bar',rot=0,figsize=(24,8)) 

plt.xlabel('Mes',fontsize='22')
plt.ylabel('Usuarios',fontsize='22')
plt.title("Usuarios que compraron a partir de una campaña",fontsize='22')
plt.xticks(size = 20)
plt.yticks(size = 20)


# Se observa una tendencia alcista en la cantidad de compras debido a campañas, por lo que deberia ser un area en la que seguir invirtiendo.

# ### Cantidad de eventos por tipo de dispositivo

# In[40]:


device_types=trocafone[(trocafone['device_type']!='Unknown') & (trocafone['device_type'].notnull())]
device_types['device_type'].value_counts().plot(kind='bar',rot=0,figsize=(24,8)) 

plt.xlabel('Dispositivo',fontsize='22')
plt.ylabel('Eventos',fontsize='22')
plt.title("Tipos de dispositivos utilizados",fontsize='22')
plt.xticks(size = 20)
plt.yticks(size = 20)


# Como hoy en día lo más utilizado son los celulares, resulta llamativo que no tenga una gran diferencia con respecto a las computadoras. Una posible respuesta es que la gente usa las computadoras del trabajo para visitar la página web. Por otro lado, las tablet ya estan casi en desuso, pues fueron reemplazadas por poderosos smartphones.

# In[41]:


device_types['event'].nunique()
#No hay datos sobre tipos de dispositivos con otros eventos


# ### Navegadores preferidos

# In[42]:


browser=trocafone['browser_version'].dropna().str.replace('\d+', '').str.replace('.', '')
browser_df=browser.value_counts().to_frame()
browser_df['percentages']=browser_df['browser_version']/browser_df['browser_version'].sum()*100
browser_df.head(10)


# In[43]:


browser_df['percentages'].head(5).plot(kind='bar',rot=0,figsize=(24,8)) 

plt.title("Tipos de navegadores utilizados",fontsize='22')
plt.xlabel('Navegador',fontsize='22')
plt.ylabel('Eventos(%)',fontsize='22')
plt.xticks(size = 20)
plt.yticks(size = 20)


# Google Chrome es el navegador más usado hoy en día, y este grafico lo (de)muestra claramente, pues comprende casi un 90% del total de los navegadores utilizados para acceder a la página web. Por esto se debería optimizar y dar soporte para que funcione mejor en Chrome que en otros navegadores.

# ### Proporcion de usuarios que ingresan por un buscador (%)

# In[44]:


buscadores = trocafone.loc[trocafone['event'] == 'search engine hit']
buscadores.index.size / trocafone.index.size * 100


# #### Buscadores mas utilizados

# In[45]:


frec_buscadores = buscadores['search_engine'].value_counts().to_frame('frecuencia')
frec_buscadores.index.title = 'Buscador'
frec_buscadores['porcentaje'] = buscadores['search_engine'].value_counts(normalize=True)*100
frec_buscadores


# Claramente Google es el buscador más usado hoy en día, y por eso las campañas publicitarias deberían venir a partir de ahí.

# # Analisis Geografico

# ### Paises con mas usuarios

# In[46]:


countries = trocafone.loc[trocafone["country"] != "Unknown"]
countries = countries.drop_duplicates(subset=['person', 'country'])
countries = countries["country"].value_counts().to_frame("cantidad")
countries.index.title = "pais"
countries = countries.assign(porcentaje=countries/countries.sum()*100)
countries.head(10)


# ### Ciudades con mas usuarios

# In[47]:


cities = trocafone.loc[trocafone["city"] != "Unknown"]
cities = cities.drop_duplicates(subset=['person', 'city'])
cities = cities["city"].value_counts().to_frame("cantidad")
cities.index.title = "ciudad"
cities = cities.assign(porcentaje=cities/cities.sum()*100)
cities.head(10)


# In[48]:


fig, ax = plt.subplots(figsize=(20, 12))
ax = sns.barplot(x=cities.head(10).index, y=cities.head(10)["cantidad"] , palette = "Reds_d")
ax.set_title("Ciudades con más usuarios", fontsize=25)
ax.set_xlabel("Ciudad", fontsize=22)
ax.set_ylabel("Usuarios", fontsize=22)
ax.tick_params(labelsize=14)


# ### Ciudades con mas eventos

# In[49]:


cities = trocafone.loc[trocafone["city"] != "Unknown"]
cities = cities["city"].value_counts().to_frame("frecuencia")
cities.index.title = "ciudad"
fig, ax = plt.subplots(figsize=(20, 12))
ax = sns.barplot(x=cities.head(10).index, y=cities.head(10)["frecuencia"] , palette = 'Reds_d')
ax.set_title("Ciudades con más eventos", fontsize=25)
ax.set_xlabel("Ciudad", fontsize=22)
ax.set_ylabel("Eventos", fontsize=22)
ax.tick_params(labelsize=14)


# ### Paises con mas eventos

# In[50]:


countries = trocafone.loc[trocafone["country"] != "Unknown"]
countries = countries["country"].value_counts().to_frame("cantidad")
countries.index.title = "pais"
countries = countries.assign(porcentaje=countries/countries.sum()*100)
countries.head(10)


# In[51]:


cities_loc = pd.read_csv("coordinates.csv")
cities_loc = cities_loc.dropna()


# In[52]:


import folium

cm = plt.get_cmap("winter")

folium_map = folium.Map(tiles="Mapbox Bright", location=(0,0), zoom_start=2.47)

for city in cities_loc.values:
    marker = folium.CircleMarker(location=[city[2], city[3]], radius=1, color='red', opacity=0.5)
    marker.add_to(folium_map)
folium_map.zoom_control = False
folium_map


# # Analisis del usuario

# ## Analisis de usuarios que compran

# ### Cantidad de personas que compraron

# In[53]:


trocafone.loc[trocafone['event']=='conversion']['person'].nunique()


# ### Promedio de visitas de personas que compran

# In[54]:


compras=trocafone.loc[trocafone['event']=='conversion'].groupby('person')
compras.size().mean()


# ### Máximo de compras de una persona

# In[55]:


compras.size().max()


# In[56]:


compras.size().hist(figsize=(24,15),bins=716,width=1)

plt.title("Compras por usuario",fontsize='22')
plt.xlabel('Compras',fontsize=20)
plt.ylabel('Usuarios',fontsize=20)
plt.xticks(size = 15)
plt.yticks(size = 15) 


# ### Porcentaje de usuarios que realizaron solo una compra

# In[57]:


(compras.size()==1).sum()/len(compras)*100


# ### Cantidad de personas que realizaron ckeckouts

# In[58]:


checkouts=trocafone.loc[trocafone['event']=='checkout'].groupby('person')
len(checkouts)


# ### Promedio de checkouts por persona

# In[59]:


checkouts.size().mean()


# ### Relación entre personas que hicieron checkouts y personas que compraron (%)

# In[60]:


len(compras)/len(checkouts)*100


# ### Relación entre total de checkouts y total de compras (%)

# In[61]:


total_checkouts=trocafone.loc[trocafone['event']=='checkout']
len(total_checkouts)


# In[62]:


total_compras=trocafone.loc[trocafone['event']=='conversion']
len(total_compras)


# In[63]:


len(total_compras)/len(total_checkouts)*100


# ### Cantidad de productos vistos por persona

# In[64]:


vistos_por_persona = trocafone.loc[trocafone['event'] == 'viewed product'].groupby('person')['sku'].nunique()
max = vistos_por_persona.max()
min = vistos_por_persona.min()
promedio = vistos_por_persona.mean()
media = vistos_por_persona.median()
std = vistos_por_persona.std()
d = {"promedio":promedio, "media":media, "std":std, "max":max, "min":min}
pd.DataFrame(data=d, index=["productos vistos por persona"])


# ### Usuarios que ingresan por una campaña publicitaria (%)

# In[65]:


publicidad = trocafone.loc[trocafone['event'] == 'ad campaign hit']
publicidad.index.size / trocafone.index.size * 100


# In[66]:


frec_publicidad = publicidad['campaign_source'].value_counts(normalize=True).to_frame('porcentaje')
frec_publicidad.index.title = 'Buscador'
frec_publicidad['porcentaje'] = frec_publicidad['porcentaje']*100
a = frec_publicidad.head(10).plot(kind='bar', figsize=(20, 15),rot=0 )
plt.title("Ingreso de usuarios por publicidad",fontsize='22')
plt.xlabel('Campaña',fontsize=20)
plt.ylabel('Porcentaje',fontsize=20)
plt.xticks(size = 15)
plt.yticks(size = 15) 
plt.grid(axis='y')

