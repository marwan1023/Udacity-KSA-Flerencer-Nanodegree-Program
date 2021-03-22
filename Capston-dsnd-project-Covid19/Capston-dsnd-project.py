#!/usr/bin/env python
# coding: utf-8

# ## Data Scientist Capstone
# 
# ### By Marwan Saeed Alsharabbi
# 
# 

# # COVID-19 Analysis ,visualization & Prediction

# <h1 style='background:#27A2AB; border:0; color:black'><center>Introduction</center></h1> 
# 
# Coronavirus is a family of viruses that are named after their spiky crown. The novel coronavirus, also known as SARS-CoV-2, is a contagious respiratory virus that first reported in Wuhan, China. On 2/11/2020, the World Health Organization designated the name COVID-19 for the disease caused by the novel coronavirus. This notebook aims at exploring COVID-19 through data analysis and projections.
# The world is going through a difficult time and fighting with a deadly virus called COVID-19. Coronavirus disease 2019 (COVID-19) is an infectious disease caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2). It was first identified in December 2019 in Wuhan, China, and has resulted in an ongoing pandemic. The first case may be traced back to 17 November 2019.As of 8 June 2020, more than 7.06 million cases have been reported across 188 countries and territories, resulting in more than 403,000 deaths. More than 3.16 million people have recovered.
# 
# ## Step 1: Select a real-world dataset 
# 
# I chose the Covid 19 data set from the following site(https://ourworldindata.org/coronavirus), and I will analyze the data, clean and perform some interesting processes and conclusions. I will strengthen the analysis and cleaning of global data. 
# The data was downloaded from https://covid.ourworldindata.org/data/owid-covid-data.csv.
# 
# 
# ## Data Sources:
# 
# Confirmed cases and deaths: Data comes from the European Centre for Disease Prevention and Control (ECDC)
# Testing for COVID-19: Data is collected by the Our World in Data team from official reports; you can find the source information for every country and further details in the post on COVID-19 testing. The testing dataset is updated around twice a week.
# Confirmed cases and deaths: Data is collected from a variety of sources (United Nations, World Bank, Global Burden of Disease, etc.)
# 
# ## License:
# The information on this page is summarized from OWID's COVID-19 github page. All of Our World in Data is completely open access and all work is licensed under the Creative Commons BY license. More information about the usage of content can be found OWID github page.https://github.com/owid/covid-19-data/tree/master/public/data
# 
# ## Authors:
# OWID's COVID19 github page the data has been collected, aggregated, and documented by Diana Beltekian, Daniel Gavrilov, Joe Hasell, Bobbie Macdonald, Edouard Mathieu, Esteban Ortiz-Ospina, Hannah Ritchie, Max Roser.

# ## Step 2: Business Questions

# 
# - How many total population in each location by continents from our dataset
# - The 10 top population total in each location by continents from our dataset
# - Show countries in Asia, Europe and North America the total_cases and total_deaths,new_cases,total_tests, total_vaccinations by mean, and max
# - Let's see the speed of transmission of the Corona virus between countries on the map . 
# - Let's see number of total_cases,total_deaths,total_deaths_per_million,test per confirmed(%) on map .
# - Top 15 countries for the total_cases,total_deaths,total_deaths_per_million,total_tests ,people_fully_vaccinated and total_vaccinations on plot_hbar and Visulizing Treemaps
# - How many the New Deaths Smoothed day by day in continents 
# - How many the New Tests Smoothed day by day in continents
# - find some gdp_per_capita and new_cases clusters over countries
# - find some new_deaths_smoothed_per_million, life_expectancy and hospital_beds_per_thousand clusters over countries
# - find some new_deaths_smoothed_per_million, handwashing_facilities and extreme_poverty clusters over countries
# - find some new_deaths_smoothed_per_million, life_expectancy and hospital_beds_per_thousand clusters over countries
# - Which vaccination scheme is used most?
# - Vaccinations (Total vs. Daily) grouped per country and vaccines
# - How the vaccination progressed

# ## Step 3: Problems and Modeling

# ### 1- Problem Question : 
# 
# Created a Linear regression model and fit the model with owid COVID19 data, predicted the world death projection for the next 30 days. In this project I have used sklearn for creating Linear Regression model and created training split with 80 to 20%. The trained the model and predicted the death for next 30 days. Also created model using XGBoost for improving the linear regression model and fit the model with owid COVID19 data, predicted the world death projection for the next 30 days.
# 
# ### 2- Problem Question:
# I will create a model that can predict the risk for the Case Mortality Ratio of a Country utilizing its Life Expectancy, Percentage of Population over 65, and Percentage of diabetes_prevalence and cardiovasc_death_rate ?
# 
# It decided on using Population Over Age 65 and Obesity because in the world, over 80% of the deaths were in the population 65 and over, and the CDC has stated that 94% of deaths had some underlying health condition. We also used Life Expectency per country to account for possible deficiencies in the health care system. John Hopkins University has listed several diseases such as heart disease and Diabetes which are known to be exacerbated by Obesity. Our idea is that we can more accurately predict the Mortality Ratio of COVID-19 by using both population 65 and over and Obesity rather than just population 65 and over. This may show that creating a healthier population is the best way to prevent the devastation in future pandemics that the world is currently facing
# 

# ### Importing Libraries

# In[1]:


# os to manipulate files
import os
# Importing pandas to work with DataFrames.
import pandas as pd
# Importing numpy to general methods.
import numpy as np
import time
import datetime
from datetime import datetime, date,timedelta
# Importing the matplotlib to create graphics
import matplotlib 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.style.use('ggplot')

# Import seaborn to better the visualization
import seaborn as sns
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.express as px


# Scipy for statistics
from scipy import stats
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy import integrate, optimize

# ML libraries
import lightgbm as lgb
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import preprocessing, svm
from sklearn import linear_model
from sklearn.metrics import mean_squared_error,explained_variance_score
import sklearn
import matplotlib.dates as dates
import seaborn as seabornInstance 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from matplotlib import rcParams
#sns.set()
#sns.set_context('talk')
import warnings
warnings.filterwarnings('ignore')


# ### In this link, every execution process pulls the new and updated data from the link

# In[2]:


# We'll download this file using the urlretrieve function from the urllib.request module.
from urllib.request import urlretrieve

urlretrieve('https://covid.ourworldindata.org/data/owid-covid-data.csv','owid-covid-data.csv')


# In[3]:


#Read data from a CSV file into a Pandas DataFrame object
world_covid19_df = pd.read_csv('owid-covid-data.csv')


# In[4]:


owidcovidcodebook=pd.read_csv('https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-codebook.csv',index_col=0)


# In[5]:


owidcovidcodebook


# In[6]:


world_covid19_df


# Data from the file is read and stored in a DataFrame object - one of the core data structures in Pandas for storing and working with tabular data. We typically use the _df suffix in the variable names for dataframes.

# In[7]:


type(world_covid19_df)


# In[8]:


#Get the number of rows & columns as a tuple
world_covid19_df.shape


# In[9]:


#View basic infomation about rows, columns & data types
world_covid19_df.info()


# ## Step 2: Perform data preparation & Cleaning

# For now, let's assume this was indeed a data entry error. We can use one of the following approaches for dealing with the missing or faulty value:
# 
# - Replace it with 0.
# - Replace it with the average of the entire column
# - Replace it with the average of the values on the previous & next date
# - Discard the row entirely
# Which approach you pick requires some context about the data and the problem. In this case, since we are dealing with data ordered by date, we can go ahead with the one approach
# 
# It is not really logical to delete Nan values but replace with 0, because that would confirm that the result was static because the data is historical and adopts high time series, we cannot replace or delete even the most data in the rows because it is data historical 

# ### Numerical Features
# I'd rather copy from the list than from Pandas Profiling

# In[10]:


# ets first handle numerical features with nan value
Numerical_feat = [feature for feature in world_covid19_df.columns if world_covid19_df[feature].dtypes != 'O']
print('Total numerical features: ', len(Numerical_feat))
print('\nNumerical Features: ', Numerical_feat)


# ### Categorical Features

# In[11]:


# categorical features
categorical_feat = [feature for feature in world_covid19_df.columns if world_covid19_df[feature].dtypes=='O']
print('Total categorical features: ', len(categorical_feat))
print('\n',categorical_feat)


# In[12]:


## Replacing the numerical Missing Values
for feature in Numerical_feat:
    ## We will replace by using median since there are outliers
        world_covid19_df[feature].fillna(0,inplace=True)
    
world_covid19_df[Numerical_feat].isnull().sum()


# In[13]:


world_covid19_df


# In[14]:


#Store the clean DataFrame in a CSV file
world_covid19_df.to_csv('covid19_df_master.csv',index=False)


# <a id="1"></a><h1 style='background:#26A2AB; border:0; color:black'><center>Analysis preparation</center></h1>

# ## Step 3: Perform exploratory Analysis & Visualization
# 

# ### Loading the cleaned Data and Exploring the histogram of the data looks like

# In[15]:


covid_df=pd.read_csv('covid19_df_master.csv')
#covid_df.hist(figsize=(15,15));


# It appears that each column contains values of a specific data type. For the numeric columns, you can view the some statistical information like mean, standard deviation, minimum/maximum values and number of non-empty values using the .describe method

# #### It appears that each column contains values of a specific data type. You can view statistical information for numerical columns (mean, standard deviation, minimum/maximum values, and the number of non-empty values) using the .describe method.

# In[16]:


covid_df.describe().style.background_gradient(cmap="CMRmap_r")


# ## Business Questions

# 
# - How many total population in each location by continents from our dataset
# - The 10 top population total in each location by continents from our dataset
# - Show countries in Asia, Europe and North America the total_cases and total_deaths,new_cases,total_tests, total_vaccinations by mean, and max
# - Let's see the speed of transmission of the Corona virus between countries on the map . 
# - Let's see number of total_cases,total_deaths,total_deaths_per_million,test per confirmed(%) on map .
# - Top 15 countries for the total_cases,total_deaths,total_deaths_per_million,total_tests ,people_fully_vaccinated and total_vaccinations on plot_hbar and Visulizing Treemaps
# - How many the New Deaths Smoothed day by day in continents 
# - How many the New Tests Smoothed day by day in continents
# - find some gdp_per_capita and new_cases clusters over countries
# - find some new_deaths_smoothed_per_million, life_expectancy and hospital_beds_per_thousand clusters over countries
# - find some new_deaths_smoothed_per_million, handwashing_facilities and extreme_poverty clusters over countries
# - find some new_deaths_smoothed_per_million, life_expectancy and hospital_beds_per_thousand clusters over countries
# - Which vaccination scheme is used most?
# - Vaccinations (Total vs. Daily) grouped per country and vaccines
# - How the vaccination progressed

# # Data Understanding

# While we ahve looked at overall numbers for the cases, tests, positive rate etc., it would be also be useful to study these numbers on a month-by-month basis. The date column might come in handy here, as Pandas provides many utilities for working with dates.

# In[17]:


#covid_df['date'] = pd.to_datetime(covid_df.date)


# You can see that it now has the datatype datetime64. We can now extract different parts of the data into separate columns, using the DatetimeIndex class 

# In[18]:


#covid_df['year'] = pd.DatetimeIndex(covid_df.date).year
#covid_df['month'] = pd.DatetimeIndex(covid_df.date).month
#covid_df['day'] = pd.DatetimeIndex(covid_df.date).day
#covid_df['weekday'] = pd.DatetimeIndex(covid_df.date).weekday


# In[19]:


covid_df.head(10)


# In[20]:


sum(covid_df.duplicated())


# In[21]:


covid_df.isnull().sum()


# In[22]:


covid_df.info()


# In[23]:


covid_df.describe().T.style.background_gradient(cmap="CMRmap_r")


# ### Question#1 
# 
# ### How the many  total population in each location  by continents  from our datase

# ### Question#1 
# 
# ### How the many  total population in each location  by continents  from our datase

# In[24]:


data_popu=covid_df.groupby('continent').sum()


# In[25]:


plt.figure(figsize = (20,18)) 
sns.set_style('ticks')
#sum countries population in Asia
plt.subplot(221)
sns.barplot(y='location', x='population', data=covid_df[covid_df['continent'] == 'Asia']).set_title('sum countries population in Asia')

#sum countries population in North America
plt.subplot(222)
sns.barplot(y='location', x='population', data=covid_df[covid_df['continent'] == 'North America']).set_title('sum countries population in North America')

#sum countries population in South America
plt.subplot(223)
sns.barplot(y='location', x='population',data=covid_df[covid_df['continent'] == 'South America']).set_title('sum countries population in South America')

#sum countries population in Europe
plt.subplot(224)
sns.set_style('ticks')
sns.barplot(y='location', x='population', data=covid_df[covid_df['continent'] == 'Europe']).set_title('sum countries population in Europe')
plt.subplots_adjust(left=0.1, 
                    bottom=0.1,  
                    right=0.9,  
                    top=0.9,  
                    wspace=0.4,  
                    hspace=0.2) 
plt.show();


# In[26]:


#sum countries population in Africa
plt.figure(figsize = (20,18))
plt.subplot(221)
sns.set_style('ticks')
#sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.barplot(y='location', x='population', data=covid_df[covid_df['continent'] == 'Africa']).set_title('sum countries population in Africa')
#sum countries population in Oceania
plt.subplot(222)
sns.barplot(y='location', x='population', data=covid_df[covid_df['continent'] == 'Oceania']).set_title('sum countries population in Oceania')
plt.subplots_adjust(left=0.1, 
                    bottom=0.1,  
                    right=0.9,  
                    top=0.9,  
                    wspace=0.4,  
                    hspace=0.2) 
plt.show();


# ### Question#2
# 
# ### The 10 top  population total in each location by continents  from  our dataset 

# In[27]:


#top countries total population in asia
top_popu_asia=covid_df[covid_df['continent'] == 'Asia']
print("The top 10 countries total population in the continent of Asia is :\n",top_popu_asia.groupby(['continent','location'])['population'].max().nlargest(10))


# In[28]:


#top countries total population in North America
top_popu_north_america=covid_df[covid_df['continent'] == 'North America']
print("The top 10 countries total population in the continent of North America is :\n",top_popu_north_america.groupby(['continent','location'])['population'].max().nlargest(10))


# In[29]:


#top countries total population in South America
top_popu_south_america=covid_df[covid_df['continent'] == 'South America']
print("The top 10 countries total population in the continent of South America is :\n",top_popu_south_america.groupby(['continent','location'])['population'].max().nlargest(10))


# In[30]:


#top countries total population in Europe
top_popu_europe=covid_df[covid_df['continent'] == 'Europe']
print("The top 10 countries total population in the continent of Europe is :\n",top_popu_europe.groupby(['continent','location'])['population'].max().nlargest(10))


# In[31]:


##Show two or more countries total population numbers the min
data=covid_df[covid_df['continent'] == 'Europe']
data.groupby(['continent','location'])['population'].min().nsmallest(10)


# #### The 10 top population total in each location by continent 'Africa from our dataset

# In[32]:


#top countries total population in Europe
top_popu_africa=covid_df[covid_df['continent'] == 'Africa']
print(" The top 10 countries total population in the continent of Africa is :\n",top_popu_africa.groupby(['continent','location'])['population'].max().nlargest(10))


# In[33]:


#top 10 countries total population in Oceania
top_popu_oceania=covid_df[covid_df['continent'] == 'Oceania']
print(" The top 10 countries total population in the continent of Oceania is :\n",top_popu_oceania.groupby(['continent','location'])['population'].max().nlargest(10))


# #### Create a data frame showing the total population of each continent

# In[34]:


continent_populations_df = covid_df.groupby(['continent'])['population'].sum()
continent_populations_df


# ### Question#3
# ### Show countries in  Asia,Europe and North America the  total_cases and total_deaths,new_cases,total_tests, total_vaccinations by mean, and max

# In[35]:


#Show countries in  asia the  total_cases and total_deaths,new_cases,total_tests, numbers mean, and max
data_total=covid_df[covid_df['continent'] == 'Asia']
data_total.groupby(['continent','location']).agg({'total_cases': ['mean','max'],'total_deaths':['mean','max'],'total_tests':['mean','max'],'total_vaccinations':['mean','max']}).style.background_gradient(cmap="CMRmap_r")


# In[36]:


#Show countries in  Europe the  total_cases and total_deaths,new_cases,total_tests, numbers mean, and max
data_total=covid_df[covid_df['continent'] == 'Europe']
data_total.groupby(['continent','location']).agg({'total_cases': ['mean','max'],'total_deaths':['mean','max'],'total_tests':['mean','max'],'total_vaccinations':['mean','max']}).style.background_gradient(cmap="CMRmap_r")


# In[37]:


#Show countries in  Europe the  total_cases and total_deaths,new_cases,total_tests, numbers mean, and max
data_total=covid_df[covid_df['continent'] == 'North America']
data_total.groupby(['continent','location']).agg({'total_cases': ['mean','max'],'total_deaths':['mean','max'],'total_tests':['mean','max'],'total_vaccinations':['mean','max']}).style.background_gradient(cmap="CMRmap_r")


# ### Question#4
# ### Let's see the speed of transmission of the Corona virus between countries on the map . 

# ## Worldwide spread

# Coronavirus is continuing its spread across the world with almost 100 million confirmed cases in 191 countries and more than two million deaths. and the virus has been detected in nearly every country, as these maps show.
# 
# #### We can see trend covid-19 moving to China -> Europe -> US

# In[38]:


worldwide_spread=covid_df[["continent","location","total_cases","total_tests","date","total_deaths","positive_rate","total_vaccinations","people_fully_vaccinated"]]

df=worldwide_spread.dropna(axis=0)
df.sort_values("total_tests",ascending=False)
df_loc=df.groupby(['location']).max()
df_loc.drop(["date"],axis=1,inplace=True)
df_loc
for i,r in df_loc.iterrows():
    if r["total_tests"]>0:
        df_loc.loc[i,"test per confirmed(%)"]=(r["total_cases"]/r["total_tests"])*100
df_covid=df_loc.reset_index()  
df_covid.style.background_gradient(cmap="CMRmap_r")


# You can click each country and see the number representing the spread of the virus.
# 
# 

# In[39]:


fig = px.choropleth(covid_df, locations="location", 
                    color=np.log(covid_df["total_cases"]),
                    locationmode="country names", hover_name="location", 
                    animation_frame=covid_df["date"],
                    title='Cases over time', color_continuous_scale=px.colors.sequential.matter)
#fig.update(layout_coloraxis_showscale=False)
fig.show()


# #### We can see trend covid-19 moving to China -> Europe -> US on map

# ### Question#5
# ### Let's see number of total_cases,total_deaths,total_deaths_per_million,test per confirmed(%) on map.

# ### COVID-19 maps

# In[40]:


def plot_map(df, col, pal):
      
    fig = px.choropleth(df, locations="location", locationmode='country names', 
                  color=col, hover_name="location", 
                  title=col, hover_data=[col], color_continuous_scale=pal)
#    fig.update_layout(coloraxis_showscale=False)
    fig.show()


# In[41]:


covid_deaths=covid_df[["continent","location","total_cases","date","total_deaths","total_deaths_per_million","total_cases_per_million","total_vaccinations"]]
df=covid_deaths.dropna(axis=0)
df_data=df.groupby(['location']).max()
df_data.drop(["date"],axis=1,inplace=True)
df_data.reset_index(inplace=True)

#df_data.drop(index=171,inplace=True)
df_data
df_data[df_data["continent"]=="Africa"].sum()


# ### Let's see number of confirmed cases on map.
# 
# For africa regions, the confirmed cases is lower than other continents, I guess this is due to the fact that number of tests is quite low.

# You can click each country and see the number of the total confirmed cases.
# 
# 

# In[42]:


plot_map(df_data,'total_cases', 'matter')


#  We can see US,Brazil and India are distinctive

# ### Let's see number of deaths on map.
# 
#  You can click each country and see the number of the total deaths.

# In[43]:


plot_map(df_data,'total_deaths', 'matter')


# We can see US,Brazil,Mexico and India are distinctive

# ### Let's see number of total deaths per million on map. 
# 
# You can click each country and see the number of the total deaths per million

# In[44]:


plot_map(df_data,'total_deaths_per_million', 'matter')


# #### We can see that south,north America and europe has the most number of total deaths per million

# ### Question#5
# ### Top 15 countries for the total_cases,total_deaths,total_deaths_per_million,total_tests,people_fully_vaccinated and total_vaccinations on plot_hbar and Visulizing Treemaps

# ### Top 15 countries

# In[45]:


def plot_hbar(df, col, n, hover_data=[]):
    fig = px.bar(df.sort_values(col).tail(n), 
                 x=col, y="location", color='continent',  
                 text=col, orientation='h', width=700, hover_data=hover_data,
                 color_discrete_sequence = px.colors.qualitative.Dark2)
    fig.update_layout(title=col, xaxis_title="", yaxis_title="", 
                      yaxis_categoryorder = 'total ascending',
                      uniformtext_minsize=8, uniformtext_mode='hide')
    fig.show()


# In[46]:


plot_hbar(df_data, 'total_cases', 15)


# In[47]:


plot_hbar(df_data, 'total_deaths', 15)


# In[48]:


plot_hbar(df_data, 'total_deaths_per_million', 15)


# In[49]:


plot_hbar(df_covid, "total_tests", 15)


# In[50]:


plot_hbar(df_covid,"total_vaccinations", 15)


# In[51]:


plot_hbar(df_covid,"people_fully_vaccinated", 15)


# ### Visulizing Treemaps
# We used this technique of data visulizing to display hierarchical data using nested rectangles,And accurately display multiple elements together

# In[52]:


def plot_treemap(col):
    fig = px.treemap(df_data, path=["location"], values=col, height=700,
                 title=col, color_discrete_sequence = px.colors.qualitative.Dark2)
    fig.data[0].textinfo = 'label+text+value'
    fig.show()
def plot_treemap_(col):
    fig = px.treemap(df_covid, path=["location"], values=col, height=700,
                 title=col, color_discrete_sequence = px.colors.qualitative.Dark2)
    fig.data[0].textinfo = 'label+text+value'
    fig.show()    


# In[53]:


plot_treemap('total_cases')


# In[54]:


plot_treemap('total_deaths')


# In[55]:


plot_treemap_('total_tests')


# In[56]:


plot_treemap_('test per confirmed(%)')


# In[57]:


plot_treemap_('total_vaccinations')


# In[58]:


plot_treemap_('people_fully_vaccinated')


# In[59]:


covid_df['death_rate'] = (covid_df['new_deaths_smoothed_per_million'] / covid_df['new_cases_smoothed_per_million']).replace(np.inf,np.nan)
covid_df['population_coverage'] = covid_df['total_tests'] / covid_df['population']


# In[60]:


trace1 = go.Scatter(
    x=covid_df.groupby(['date'])['date'].apply(lambda x: np.unique(x)[0]),
    y=covid_df.groupby(['date'])['new_deaths_smoothed_per_million'].mean(),
        xaxis='x2',
    yaxis='y2',
    name = "mean new deaths smoothed per million"
)
trace2 = go.Scatter(
    x=covid_df.groupby(['date'])['date'].apply(lambda x: np.unique(x)[0]),
    y=covid_df.groupby(['date'])['new_tests_smoothed_per_thousand'].mean(),
    name = "mean new tests smoothed per thousand"
)
trace3 = go.Scatter(
    x=covid_df.groupby(['date'])['date'].apply(lambda x: np.unique(x)[0]),
    y=(covid_df.groupby(['date'])['death_rate'].mean().replace([np.inf],np.nan).interpolate(method='linear', limit_direction='forward', axis=0) * 100).round(3),
    xaxis='x3',
    yaxis='y3',
    name = "interpolated death rate %"
)
trace4 = go.Scatter(
    x=covid_df.groupby(['date'])['date'].apply(lambda x: np.unique(x)[0]),
    y=((covid_df.groupby(['date'])['new_cases_per_million'].apply(lambda x: np.mean(x/1e+6))) * 100).round(6),
    xaxis='x4',
    yaxis='y4',
    name = "mean covid population d2d coverage %"
)

data = [trace1, trace2, trace3, trace4]
layout = go.Layout(
    xaxis=dict(
        domain=[0, 0.45]
    ),
    yaxis=dict(
        domain=[0, 0.45]
    ),
    xaxis2=dict(
        domain=[0.55, 1]
    ),
    xaxis3=dict(
        domain=[0, 0.45],
        anchor='y3'
    ),
    xaxis4=dict(
        domain=[0.55, 1],
        anchor='y4'
    ),
    yaxis2=dict(
        domain=[0, 0.45],
        anchor='x2'
    ),
    yaxis3=dict(
        domain=[0.55, 1]
    ),
    yaxis4=dict(
        domain=[0.55, 1],
        anchor='x4'
    ),
    title = 'Mean new deaths per 1M, new tests per 1K, death rate and covid mean coverage'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# ### Line Plot function
# We used this technique of data visualization to plot line display day by day trend ,And accurately display multiple elements together

# In[61]:


def plot_line(col,title):
    trace1 = go.Scatter(
                    x = covid_df[(covid_df['continent']=='Asia')].groupby(['date','continent'])['date'].apply(lambda x: np.unique(x)[0]),
                    y = covid_df[(covid_df['continent']=='Asia')].groupby(['date','continent'])[col].sum(),
                    mode = "lines",
                    name = "Asia",
                    marker = dict(color = 'green'),
    )

    trace2 = go.Scatter(
                        x = covid_df[(covid_df['continent']=='Europe')].groupby(['date','continent'])['date'].apply(lambda x: np.unique(x)[0]),
                        y = covid_df[(covid_df['continent']=='Europe')].groupby(['date','continent'])[col].sum(),
                        mode = "lines",
                        name = "Europe",
                        marker = dict(color = 'red'),
    )

    trace3 = go.Scatter(
                        x = covid_df[(covid_df['continent']=='Africa')].groupby(['date','continent'])['date'].apply(lambda x: np.unique(x)[0]),
                        y = covid_df[(covid_df['continent']=='Africa')].groupby(['date','continent'])[col].sum(),
                        mode = "lines",
                        name = "Africa",
                        marker = dict(color = 'blue'),
                        #text= covid_df.university_name
    )

    trace4 = go.Scatter(
                        x = covid_df[(covid_df['continent']=='North America')].groupby(['date','continent'])['date'].apply(lambda x: np.unique(x)[0]),
                        y = covid_df[(covid_df['continent']=='North America')].groupby(['date','continent'])[col].sum(),
                        mode = "lines",
                        name = "North America",
                        marker = dict(color = 'black'),
    )

    trace5 = go.Scatter(
                        x = covid_df[(covid_df['continent']=='South America')].groupby(['date','continent'])['date'].apply(lambda x: np.unique(x)[0]),
                        y = covid_df[(covid_df['continent']=='South America')].groupby(['date','continent'])[col].sum(),
                        mode = "lines",
                        name = "South America",
                        marker = dict(color = 'brown'),
    )

    data = [trace1,trace2,trace3,trace4,trace5]
    layout = dict(title = title,
                  xaxis= dict(title= "#{} day by day".format(title),ticklen= 5,zeroline= False)
                 )
    fig = dict(data = data, layout = layout)
    iplot(fig)


# ### Question#6
# ### How many the New Deaths Smoothed day by day in continents

# In[62]:


plot_line('new_deaths_smoothed','New Deaths Smoothed')


# ### Question#7
# ### How many the new vaccinations smoothed day by day in continents

# In[63]:


plot_line('new_vaccinations_smoothed','new vaccinations smoothed')


# In[64]:


plot_line('total_vaccinations','total_vaccinations')


# ### Question#8
# ### How many the New Tests Smoothed day by day in continents

# In[65]:


plot_line('new_tests_smoothed','New tests smoothed')


# ### Question#8
# ### How many the positive_rate day by day in continents

# In[66]:


#covid_df['date'] = pd.to_datetime(covid_df.date)
plot_line('positive_rate','positive_rate')


# In[67]:


def plot_line_mean(col,title):
    
    
    trace1 = go.Scatter(
                        x = covid_df[(covid_df['continent']=='Asia')&(covid_df['date']>='2020-03-01')].groupby(['date','continent'])['date'].apply(lambda x: np.unique(x)[0]),
                        y = covid_df[(covid_df['continent']=='Asia')&(covid_df['date']>='2020-03-01')].groupby(['date','continent'])[col].mean()*100,
                        mode = "lines",
                        name = "Asia",
                        marker = dict(color = 'green'),
    )

    trace2 = go.Scatter(
                        x = covid_df[(covid_df['continent']=='Europe')&(covid_df['date']>='2020-03-01')].groupby(['date','continent'])['date'].apply(lambda x: np.unique(x)[0]),
                        y = covid_df[(covid_df['continent']=='Europe')&(covid_df['date']>='2020-03-01')].groupby(['date','continent'])[col].mean()*100,
                        mode = "lines",
                        name = "Europe",
                        marker = dict(color = 'red'),
    )

    trace3 = go.Scatter(
                        x = covid_df[(covid_df['continent']=='Africa')&(covid_df['date']>='2020-03-01')].groupby(['date','continent'])['date'].apply(lambda x: np.unique(x)[0]),
                        y = covid_df[(covid_df['continent']=='Africa')&(covid_df['date']>='2020-03-01')].groupby(['date','continent'])[col].mean()*100,
                        mode = "lines",
                        name = "Africa",
                        marker = dict(color = 'blue'),
    )

    trace4 = go.Scatter(
                        x = covid_df[(covid_df['continent']=='North America')&(covid_df['date']>='2020-03-01')].groupby(['date','continent'])['date'].apply(lambda x: np.unique(x)[0]),
                        y = covid_df[(covid_df['continent']=='North America')&(covid_df['date']>='2020-03-01')].groupby(['date','continent'])[col].mean()*100,
                        mode = "lines",
                        name = "North America",
                        marker = dict(color = 'black'),
    )

    trace5 = go.Scatter(
                        x = covid_df[(covid_df['continent']=='South America')&(covid_df['date']>='2020-03-01')].groupby(['date','continent'])['date'].apply(lambda x: np.unique(x)[0]),
                        y = covid_df[(covid_df['continent']=='South America')&(covid_df['date']>='2020-03-01')].groupby(['date','continent'])[col].mean(),
                        mode = "lines",
                        name = "South America",
                        marker = dict(color = 'brown'),
    )

    data = [trace1,trace2,trace3,trace4,trace5]
    layout = dict(title = title,
                  xaxis= dict(title= 'mean deaths/cases %',ticklen= 5,zeroline= False)
                 )
    fig = dict(data = data, layout = layout)
    iplot(fig)


# In[68]:


plot_line_mean('death_rate','Mean death rate over continents')


# ### Test population coverage

# In[69]:


plot_line_mean('population_coverage','Mean population test coverage over continents')


# In[70]:


trace1 = go.Scatter(
                    x = covid_df[(covid_df['continent']=='Asia')&(covid_df['date']>='2020-03-01')].groupby(['date','continent'])['date'].apply(lambda x: np.unique(x)[0]),
                    y = covid_df[(covid_df['continent']=='Asia')&(covid_df['date']>='2020-03-01')].groupby(['date','continent'])['death_rate'].mean()*100,
                    mode = "lines",
                    name = "Asia",
                    marker = dict(color = 'green'),
)

trace2 = go.Scatter(
                    x = covid_df[(covid_df['continent']=='Europe')&(covid_df['date']>='2020-03-01')].groupby(['date','continent'])['date'].apply(lambda x: np.unique(x)[0]),
                    y = covid_df[(covid_df['continent']=='Europe')&(covid_df['date']>='2020-03-01')].groupby(['date','continent'])['population_coverage'].mean()*100,
                    mode = "lines",
                    name = "Europe",
                    marker = dict(color = 'red'),
)

trace3 = go.Scatter(
                    x = covid_df[(covid_df['continent']=='Africa')&(covid_df['date']>='2020-03-01')].groupby(['date','continent'])['date'].apply(lambda x: np.unique(x)[0]),
                    y = covid_df[(covid_df['continent']=='Africa')&(covid_df['date']>='2020-03-01')].groupby(['date','continent'])['population_coverage'].mean()*100,
                    mode = "lines",
                    name = "Africa",
                    marker = dict(color = 'blue'),
)

trace4 = go.Scatter(
                    x = covid_df[(covid_df['continent']=='North America')&(covid_df['date']>='2020-03-01')].groupby(['date','continent'])['date'].apply(lambda x: np.unique(x)[0]),
                    y = covid_df[(covid_df['continent']=='North America')&(covid_df['date']>='2020-03-01')].groupby(['date','continent'])['population_coverage'].mean()*100,
                    mode = "lines",
                    name = "North America",
                    marker = dict(color = 'black'),
)

trace5 = go.Scatter(
                    x = covid_df[(covid_df['continent']=='South America')&(covid_df['date']>='2020-03-01')].groupby(['date','continent'])['date'].apply(lambda x: np.unique(x)[0]),
                    y = covid_df[(covid_df['continent']=='South America')&(covid_df['date']>='2020-03-01')].groupby(['date','continent'])['population_coverage'].mean(),
                    mode = "lines",
                    name = "South America",
                    marker = dict(color = 'brown'),
)

data = [trace1,trace2,trace3,trace4,trace5]
layout = dict(title = 'Mean population test coverage over continents',
              xaxis= dict(title= 'mean tests/population %',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)


# ### Question#9
# ### find some gdp_per_capita and new_cases clusters over countries

# In[71]:


covid_df_grouped = covid_df.groupby(['location','continent']).agg({'new_deaths': np.sum, 'gdp_per_capita': np.mean, 'new_cases':np.sum}).reset_index()
covid_df_grouped = covid_df_grouped[(~covid_df_grouped['new_deaths'].isnull())&(~covid_df_grouped['new_cases'].isnull())&(~covid_df_grouped['gdp_per_capita'].isnull())&(~covid_df_grouped['continent'].isnull())]


# In[72]:


fig = px.scatter(covid_df_grouped, 
                 x="new_deaths", y="gdp_per_capita", size="new_cases", color="continent",
                 hover_name="location", log_x=True, size_max=60)
fig.show()


# ### Question#10
# ### find some new_deaths_smoothed_per_million, handwashing_facilities and extreme_poverty clusters over countries

# In[73]:


covid_df_grouped = covid_df.groupby(['location','continent']).agg({'handwashing_facilities': np.mean, 'new_deaths_smoothed_per_million': np.sum, 'extreme_poverty':np.mean}).reset_index()
covid_df_grouped = covid_df_grouped[(~covid_df_grouped['handwashing_facilities'].isnull())&(~covid_df_grouped['new_deaths_smoothed_per_million'].isnull())&(~covid_df_grouped['extreme_poverty'].isnull())&(~covid_df_grouped['continent'].isnull())]


# In[74]:


fig = px.scatter(covid_df_grouped, 
                 x="new_deaths_smoothed_per_million", y="handwashing_facilities", size="extreme_poverty", color="continent",
                 hover_name="location", log_x=True, size_max=60)
fig.show()


# ### Question#11
# ### find some new_deaths_smoothed_per_million, aged_70_older and population_density clusters over countries

# In[75]:


covid_df_grouped = covid_df.groupby(['location','continent']).agg({'population_density': np.mean, 'new_deaths_smoothed_per_million': np.sum, 'aged_70_older':np.mean}).reset_index()
covid_df_grouped = covid_df_grouped[(~covid_df_grouped['population_density'].isnull())&(~covid_df_grouped['new_deaths_smoothed_per_million'].isnull())&(~covid_df_grouped['aged_70_older'].isnull())&(~covid_df_grouped['continent'].isnull())]


# In[76]:


fig = px.scatter(covid_df_grouped, 
                 x="new_deaths_smoothed_per_million", y="aged_70_older", size="population_density", color="continent",
                 hover_name="location", log_x=True, size_max=60)
fig.show()


# ### Question#12
# ### find some new_deaths_smoothed_per_million, life_expectancy and hospital_beds_per_thousand clusters over countries

# In[77]:


covid_df_grouped = covid_df.groupby(['location','continent']).agg({'life_expectancy': np.mean, 'new_deaths_smoothed_per_million': np.sum, 'hospital_beds_per_thousand':np.mean}).reset_index()
covid_df_grouped = covid_df_grouped[(~covid_df_grouped['life_expectancy'].isnull())&(~covid_df_grouped['new_deaths_smoothed_per_million'].isnull())&(~covid_df_grouped['hospital_beds_per_thousand'].isnull())&(~covid_df_grouped['continent'].isnull())]


# In[78]:


fig = px.scatter(covid_df_grouped, 
                 x="new_deaths_smoothed_per_million", y="life_expectancy", size="hospital_beds_per_thousand", color="continent",
                 hover_name="location", log_x=True, size_max=60)
fig.show()


# ### Stringency Index and death rate correlation

# In[79]:


covid_df_grouped = covid_df.groupby(['location','continent']).agg({'death_rate': np.mean, 'stringency_index': np.mean, 'new_cases':np.sum}).reset_index()
covid_df_grouped = covid_df_grouped[(~covid_df_grouped['death_rate'].isnull())&(~covid_df_grouped['stringency_index'].isnull())&(~covid_df_grouped['new_cases'].isnull())&(~covid_df_grouped['continent'].isnull())]


# In[80]:


fig = px.scatter(covid_df_grouped, 
                 x="death_rate", y="stringency_index", size="new_cases", color="continent",
                 hover_name="location", log_x=True, size_max=60)
fig.show()


# <h1 style='background:#27A2AB; border:0; color:black'><center>Modeling</center></h1> 

# In[81]:


covid_df_copy = world_covid19_df.copy()


# In[82]:


covid_df_copy = covid_df.copy()


# In[83]:


covid_df_copy.head(10)


# ### Correlation Analysis

# In[84]:


correlations = covid_df_copy.corr()['total_cases'].abs().sort_values(ascending=False).drop('total_cases',axis=0).to_frame()
correlations.plot(kind='bar',figsize=(12,10));


# In[85]:


# Function to see the correlation of each features

def corr(df):
    "argument df tp get the correlation for"
    return df.corr()


# In[86]:


corr(covid_df_copy).style.background_gradient(cmap="CMRmap_r")


# <h1 style='background:#27A2AB; border:0; color:black'><center>Linear Regression-Forecast</center></h1> 

# #### Created a Linear regression model and fit the model with owid COVID19 data, predicted the world death projection for the next 30 days. In this project I have used sklearn for creating Linear Regression model and created training split with 80 to 20%. The trained the model and predicted the death for next 30 days. Also created model using XGBoost for improving the linear regression model and fit the model with owid COVID19 data, predicted the world death projection for the next 30 days.

# In[87]:


#owid_covid_data = pd.read_csv('https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv')
#owid_covid_data.head()


# In[88]:


#Select Coloumn to clean
ColumnToClean = ['total_cases', 'new_cases', 'total_deaths', 'new_deaths','aged_65_older','aged_70_older','gdp_per_capita','diabetes_prevalence','female_smokers','male_smokers','hospital_beds_per_thousand']
#Replace the nan with emty string
covid_df_copy[['location']] = covid_df_copy[['location']].fillna('')
#Replace the Nan with 0
covid_df_copy[ColumnToClean] = covid_df_copy[ColumnToClean].fillna(0)
#Filter the data so we will get only overall world data
covid_df_copy = covid_df_copy.query('location=="World"' )
Data_For_Regression = pd.DataFrame(columns=['date','total_cases', 'new_cases', 'total_deaths', 'new_deaths','aged_65_older','aged_70_older','gdp_per_capita','diabetes_prevalence','female_smokers','male_smokers','hospital_beds_per_thousand'], data=covid_df_copy[['date','total_cases', 'new_cases', 'total_deaths', 'new_deaths','aged_65_older','aged_70_older','gdp_per_capita','diabetes_prevalence','female_smokers','male_smokers','hospital_beds_per_thousand']].values)
Data_For_Regression.head()


# In[89]:


#set the index as date
Data_For_Regression['date'] = pd.to_datetime(Data_For_Regression['date'])
Data_For_Regression = Data_For_Regression.set_index('date')
Data_For_Regression.head()


# In[90]:


#Plot the graph
Data_For_Regression['total_cases'].plot(figsize=(15,6), color="green")
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Cases')
plt.show()

Data_For_Regression['total_deaths'].plot(figsize=(15,6), color="red")
plt.xlabel('Date')
plt.ylabel('Death')
plt.show()

Data_For_Regression['new_cases'].plot(figsize=(15,6), color="blue")
plt.xlabel('Date')
plt.ylabel('New Cases')
plt.show()


# In[91]:


# pick total death as forecast column
forecast_col = 'total_deaths'

# Chosing 30 days as number of forecast days
forecast_out = int(30)
print('length =',len(Data_For_Regression), "and forecast_out =", forecast_out)


# In[92]:


# Creating label by shifting 'total_deaths' according to 'forecast_out'
Data_For_Regression['temp'] = Data_For_Regression[forecast_col].shift(-forecast_out)
print(Data_For_Regression.head(2))
print('\n')
# verify rows with NAN in Label column 
print(Data_For_Regression.tail(2))


# In[93]:


# Define features Matrix X by excluding the label column which we just created 
X = np.array(Data_For_Regression.drop(['temp'], 1))

# Using a feature in sklearn, preposessing to scale features
X = preprocessing.scale(X)
print(X[1,:])


# In[94]:


# X contains last 'n= forecast_out' rows for which we don't have label data
# Put those rows in different Matrix X_forecast_out by X_forecast_out = X[end-forecast_out:end]

X_forecast_out = X[-forecast_out:]
X = X[:-forecast_out]
print ("Length of X_forecast_out:", len(X_forecast_out), "& Length of X :", len(X))


# In[95]:


# Define vector y for the data we have prediction for
# make sure length of X and y are identical
y = np.array(Data_For_Regression['temp'])
y = y[:-forecast_out]
print('Length of y: ',len(y))


# In[96]:


# (split into test and train data)
# test_size = 0.2 ==> 20% data is test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print('length of X_train and x_test: ', len(X_train), len(X_test))


# In[97]:


# Create linear regression object
lr = LinearRegression()

# Train the model using the training sets
lr.fit(X_train, y_train)
# Test
accuracy = lr.score(X_test, y_test)
print("Accuracy of Linear Regression: ", accuracy)


# In[98]:


# Predict using our Model
forecast_prediction = lr.predict(X_forecast_out)
print(forecast_prediction)


# In[99]:


Data_For_Regression.tail()


# In[100]:


last_date = Data_For_Regression.iloc[-1].name 
last_date


# In[101]:


todays_date = datetime.strptime(last_date.strftime("%Y-%m-%d"), "%Y-%m-%d")
todays_date = todays_date + timedelta(days=1)
todays_date = datetime.strptime(todays_date.strftime("%Y-%m-%d"), "%Y-%m-%d")
index = pd.date_range(todays_date, periods=30, freq='D')
columns = ['total_cases', 'new_cases', 'total_deaths', 'new_deaths','aged_65_older','aged_70_older','gdp_per_capita','diabetes_prevalence','female_smokers','male_smokers','hospital_beds_per_thousand','temp','forecast']
temp_df = pd.DataFrame(index=index, columns=columns)
temp_df


# In[102]:


j=0
for i in forecast_prediction:
    temp_df.iat[j,12] = i
    j= j+1

temp_df


# In[103]:


#Append the forcasted - Initially did it for easness but kater decided to use xgboost also
Data_For_Regression['total_deaths'].plot(figsize=(15,6), color="red")
temp_df['forecast'].plot(figsize=(15,6), color="orange")
plt.xlabel('Date')
plt.ylabel('Death')
plt.show()


# In[104]:


#  XGboost algorithm to see if we can get better results
xgb_model = xgb.XGBRegressor(objective ='reg:squarederror',colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.07,
                 max_depth=3,
                 min_child_weight=1.5,
                 n_estimators=10000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6) 

traindf, testdf = train_test_split(X_train, test_size = 0.2)

xgb_model.fit(X_train,y_train)

xgforecast_prediction = xgb_model.predict(X_forecast_out)

xgforecast_prediction


# In[105]:


#Setting the temperory df with XGboost forecasted data
j=0
for i in xgforecast_prediction:
    temp_df.iat[j,12] = i
    j= j+1


# In[106]:


Data_For_Regression['total_deaths'].plot(figsize=(15,6), color="red")
temp_df['forecast'].plot(figsize=(15,6), color="orange")
plt.xlabel('Date')
plt.ylabel('Death')
plt.show()


# <h1 style='background:#27A2AB; border:0; color:black'><center>k-nearest neighbors(KNN) algorithm </center></h1> 

# ### I will create a model that can predict the risk for the Case Mortality Ratio of a Country utilizing its Life Expectancy, Percentage of Population over 65, and Percentage of diabetes_prevalence and cardiovasc_death_rate ?
# It decided on using Population Over Age 65 and Obesity because in the world, over 80% of the deaths were in the population 65 and over, and the CDC has stated that 94% of deaths had some underlying health condition. We also used Life Expectency per country to account for possible deficiencies in the health care system. John Hopkins University has listed several diseases such as heart disease and Diabetes which are known to be exacerbated by Obesity. Our idea is that we can more accurately predict the Mortality Ratio of COVID-19 by using both population 65 and over and Obesity rather than just population 65 and over. This may show that creating a healthier population is the best way to prevent the devastation in future pandemics that the world is currently facing

# In[ ]:





# After viewing the graphs in Linear Regression-Forecast we are concerned about the accuracy that XGboost  algorithms can achieve with this data. The data may improve as more accurate case data is produced from Antibody testing. We will continue and see if our ML Algorithm can do better than we are expecting. We have initially chosen to use categorization with the HighRisk category as that may be more accurate than regression.

# In[107]:


covid_df_copy = covid_df.copy()


# In[108]:


covid_df_copy.info()


# In[109]:


#it decided to update the High Risk and base it off of total_deaths_per_million which is the Total deaths attributed to COVID-19 per 1,000,000 people
covid_df_copy['HighRisk'] = zscore(covid_df_copy['total_deaths_per_million']) > 0.65


# In[110]:


covid_df_copy.head(20)


# ### Correlation Analysis

# #### We will be using the diabetes, cardiovascular health, percent of poplation above 70 and any other data we find to be the most useful to see if we can get better results with these  features.

# In[111]:


corr = covid_df_copy[['death_rate', 'total_deaths_per_million', 'aged_65_older','extreme_poverty', 'icu_patients', 'life_expectancy', 'cardiovasc_death_rate', 'diabetes_prevalence', 'human_development_index', 'population_density', 'aged_70_older', 'population_coverage']].corr()
corr.style.background_gradient(cmap='coolwarm')
#Total deaths attributed to COVID-19 per 1,000,000 people
#With the new data the correlations seem stronger on aged_70_older, 'cardiovasc_death_rate', 'diabetes_prevalence', 'human_development_index', 'population_density', 'aged_70_older',and  'population_coverage' than before.


# 
# ### Now we will split our data for the Machine Learning Algorithm using the High Risk Category as our target and Life_Expectancy,icu_patients ,diabetes_prevalence, and aged_65_older as features

# In[112]:



predictors = ['diabetes_prevalence','icu_patients','life_expectancy', 'cardiovasc_death_rate', 'human_development_index', 'aged_70_older', 'population_density', 'female_smokers', 'male_smokers', 'extreme_poverty']
target = 'HighRisk'
X = covid_df_copy[predictors].values
y = covid_df_copy[target].values

# Split the data into training and test sets, and scale
scaler = StandardScaler()

# unscaled version (note that scaling is only used on predictor variables)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# scaled version
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)
print('First 10 Rows of Scaled Data: \n\n', X_train[0:10:,], '\n')

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
accuracy = (predictions == y_test).mean()
print('Accuracy:', round(accuracy * 100, 2), '%')


# ### It looks like despite our initial reservations that KNN was able to get a decent accuracy of 90.23 %
# #### Let's test which k value gets us our best accuracy.

# In[113]:


n = 8
accuracies = []
ks = np.arange(1, n+1, 2)
for k in ks:
    print(k, ' ', end='')
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    acc = (predictions == y_test).mean()
    accuracies.append(acc)
print('done')

def get_best(ks, accuracies):
    maximum = np.array(accuracies).max()
    indexMax = np.where(accuracies == maximum)
    return ks[indexMax], maximum

best_k, best_acc = get_best(ks, accuracies)
print('best k = {}, best accuracy: {:0.3f}%'.format(best_k, best_acc * 100))


# #### Interestingly a slightly better classification of  90.226% with k = 5.

# In[114]:


print('Comparison of predictions to y_test values: \n\n', predictions == y_test)
print('\nPredictions:\n\n', predictions)
print('\nY_test values:\n\n', y_test)


# A further look at our predictions and Y_test values show that we get 90.226% simply by predicting almost everything as False so this model's features and data should be improved

# ##### Now we will try to test all the features we currently have and select through a greedy algorithm the best features utilizing KNN and all K-Ranges between 1 and 7

# In[115]:


predictors = ['diabetes_prevalence','icu_patients','life_expectancy', 'cardiovasc_death_rate','human_development_index', 'aged_70_older', 'population_density', 'female_smokers', 'male_smokers', 'extreme_poverty']
target = 'HighRisk'
X = covid_df_copy[predictors].values
y = covid_df_copy[target].values
# Split the data into training and test sets, and scale
scaler = StandardScaler()

# unscaled version (note that scaling is only used on predictor variables)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# scaled version
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

ks = np.arange(1, 8, 2)
for k in ks:
    best_acc = 0
    selected = []
    remaining = list(range(X_train.shape[1]))
    n = 11
    better = True
    while len(selected) < n and better == True:
        # find the single features that works best in conjunction
        # with the already selected features
        acc_max = 0
        for i in remaining:
            # make a version of the training data with just selected, feature i
            selectedFi = selected.copy()
            selectedFi.append(i)
            X_si = X_train[:,selectedFi]
            y_siTrain = y_train[~np.isnan(X_si).any(axis=1)]
            X_si=X_si[~np.isnan(X_si).any(axis=1)]
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_si, y_siTrain)
            X_testSi = X_test[:,selectedFi]
            y_siTest = y_test[~np.isnan(X_testSi).any(axis=1)]
            X_testSi = X_testSi[~np.isnan(X_testSi).any(axis=1)]
            predictions = knn.predict(X_testSi)

            acc = (predictions == y_siTest).mean()
            if (acc > acc_max):
                acc_max = acc
                i_min = i
                if (best_acc < acc):
                    best_acc = acc
                    better = True
                else:
                    better = False


        if (better == True):        
            remaining.remove(i_min)
            selected.append(i_min)
            print('k: {}; num features: {}; features: {}; bestAcc: {:.2f}%'.format(k, len(selected), [predictors[x] for x in selected], best_acc*100))
    


# ### Looking at the data above we have gotten a bit better accuracy using k=7 with the following 4 features:
# aged_70_older', 'icu_patients', 'life_expectancy', 'cardiovasc_death_rate'

# In[116]:


# change default plot size
rcParams['figure.figsize'] = 10,8
sns.scatterplot(data=covid_df_copy, x='male_smokers', y='cardiovasc_death_rate', hue='HighRisk', style='HighRisk');
plt.xlabel("Cardiovascular Death Rate")
plt.ylabel("Male Smokers");


# In[117]:


sns.scatterplot(data=covid_df_copy, x='life_expectancy', y='diabetes_prevalence', hue='HighRisk', style='HighRisk')
plt.xlabel("Life Expectancy")
plt.ylabel("% of Obesity");


# In[118]:


sns.scatterplot(data=covid_df_copy, x='human_development_index', y='positive_rate', hue='HighRisk', style='HighRisk');
plt.xlabel("Human Development Index")
plt.ylabel("positive_rate");


# In[119]:


sns.scatterplot(data=covid_df_copy, x='aged_65_older', y='diabetes_prevalence', hue='HighRisk', style='HighRisk');
plt.xlabel("% Age 65 and older")
plt.ylabel("% of diabetes_prevalence");


# In[120]:


k = 7
predictors = ['diabetes_prevalence','icu_patients','female_smokers','male_smokers', 'human_development_index', 'life_expectancy','cardiovasc_death_rate','positive_rate']
target = 'HighRisk'

# unscaled version (note that scaling is only used on predictor variables)
X = covid_df_copy[predictors].values
y = covid_df_copy[target].values

# Split the data into training and test sets, and scale
scaler = StandardScaler()

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
# scaled version
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

#Remove data rows with nan
y_train = y_train[~np.isnan(X_train).any(axis=1)]
X_train = X_train[~np.isnan(X_train).any(axis=1)]

y_test = y_test[~np.isnan(X_test).any(axis=1)]
X_test = X_test[~np.isnan(X_test).any(axis=1)]

knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

acc = (predictions == y_test).mean()

print('Comparison of predictions to y_test values: \n\n', predictions == y_test)
print('\nPredictions:\n\n', predictions)
print('\nY_test values:\n\n', y_test)

print('\nAccurancy: ', acc)


# #### The model using the extra features especially the human_development_index, smoker data and more recent target data has gotten better at predicting a countries rate of mortality vs population going from 90% to 93% accuracy depending on the randomization with a mean of 89% using slightly diff features
# 
# ##### This may be due to different reporting systems for what is and is not a covid death and overall accuracy of the inputs.
# 
# #### Our original Hypotheses that Age and Obesity would be factors seem to have been proven true through the data, one might even be able to try regression on the normalized mortality / population and if we had the One World Data originally we may have even gone further and tried that as the correlation seems to be stronger

# ## Inferences and Conclusion

# Two questions guide this daily updated publication on the global COVID-19 pandemic:
# 
# How can we make progress against the pandemic?
# And, are we making progress?
# To answer these questions we need data. But data is not enough. This is especially true in this pandemic because even the best available data is far from perfect. Much of our work therefore focuses on explaining what the data can – and can not – tell us about the pandemic. 
# 
# Our goal is two-fold:
# 
# To provide reliable, global and open data and research on how the COVID-19 pandemic is spreading, what impact the pandemic has, how we can make progress against the pandemic, and whether the measures countries are taking are successful or not;
# And to build an infrastructure that allows research colleagues – and everyone who is interested – to navigate and understand this data and research.
# Before we study how to make progress we should consider the more basic question: is it possible to do so?
# 
# The answer is very clear: While some countries have failed in their response to the pandemic, others met the challenge much more successfully. Perhaps the most important thing to know about the pandemic is that it is possible to fight the pandemic.
# 
# Responding successfully means two things: limiting the direct and the indirect impact of the pandemic. Countries that have responded most successfully were able to avoid choosing between the two: they avoided the trade-off between a high mortality and a high socio-economic impact of the pandemic. New Zealand has been able to bring infections down and open up their country internally. Other island nations were also able to almost entirely prevent an outbreak (like Taiwan, Australia, and Iceland). But not only islands were able to bend the curve of infections and prevent large outbreaks – Norway, Uruguay, Switzerland, South Korea, and Germany are examples. These countries suffered a smaller direct impact, but they also limited the indirect impacts because they were able to release lockdown measures earlier.
# 
# Together with colleagues at the Robert Koch Institute, the Chan School of Public Health, the UK Public Health Rapid Support Team, the London School of Hygiene and Tropical Medicine and other institutions we study countries that responded most successfully in detail.

# Among the countries with the highest death toll are some of the most populous countries in the world such as the US, Brazil, and Mexico. If you prefer to adjust for the differences in population size you can switch to per capita statistics by clicking the ‘per million people’ tickbox.
# 
# We can see three different ways in which the pandemic has affected countries:
# 
# - Some countries have not been able to contain the pandemic. The death toll there continues to rise quickly week after week.
# - Some countries saw large outbreaks, but then ‘bent the curve’ and brought the number of deaths down again. Italy, Germany, and many European countries followed this trajectory.
# - Some were able to prevent a large outbreak altogether. Shown in the chart are South Korea and Norway. These countries had rapid outbreaks, but were then able to reduce the number of deaths very quickly to low numbers.
# 
# While some commentaries on the pandemic have the premise that all countries failed to respond well to the pandemic the exact opposite stands out to us: Even at this early stage of the pandemic we see very large differences between countries – as the chart shows. While some suffer terrible outbreaks others have managed to contain rapid outbreaks or even prevented bad outbreaks entirely. It is possible to respond successfully to the pandemic.

# - Some countries, like Australia, South Korea and Slovenia do hundreds, or even thousands of tests for each case they find – the positive rate of these countries is therefore below 1% or even 0.1%.
# - Others, such as Mexico, Nigeria, and Bangladesh do very few tests – five or fewer – for every confirmed case. Their positive rate is very high.

# - The data for Slovakia, Thailand, New Zealand, South Korea, and Germany shows that these countries monitored the outbreak well from the start or caught up rapidly after an initial outbreak. Eventually they were able to bend the curve and bring down the number of confirmed cases, while increasing the ratio of tests to confirmed cases. These are not the only countries that achieved this; you can add for example Austria, Iceland, Slovenia, Tunisia, or Latvia to the chart and you will find similar trajectories.
# - The data for Brazil, Mexico, the United States, Panama, India, Pakistan, South Africa, and Nigeria shows that these countries test little relative to the size of the outbreak. Additionally these countries report unfortunately still very high daily case counts – their lines are red and far from zero.

# Fighting the pandemic: What can everyone of us do to flatten the curve?
# Some measures against the pandemic are beyond what any individual can do. The development of a vaccine, R&D in pharmaceutical research, building the infrastructure to allow large-scale testing, and coordinated policy responses require large-scale collaboration and are society-wide efforts. We will explore these later.
# 
# But, as with all big problems, there are many ways to make progress and some of the most important measures are up to all of us.
# 
# In the fight against the pandemic we are in the fortunate situation that what is good for ourselves is also good for everyone else. By protecting yourself you are slowing the spread of the pandemic.
# 
# You and everyone else have the same two clear personal goals during this pandemic: Don’t get infected and don’t infect others.
# 
# To not get infected you have to do what you can to prevent the virus from entering your body through your mouth, nose, or eyes. To not infect others your goal is to prevent the virus from traveling from your body to the mouth, nose or eyes of somebody else.
# 
# What can you do? How can all of us – you and me – do our part to flatten the curve? The three main measures are called the three Ws: Wash your hands, wear a mask, watch your distance.

# ## References and Future Work
#     
# 1- https://www.geeksforgeeks.org/python-programming-language/?ref=leftbar
# 
# 2- https://www.python-course.eu/python3_class_and_instance_attributes.php
# 
# 3- https://thispointer.com/data-analysis-in-python-using-pandas/
# 
# 4- https://jovian.ml/learn/data-analysis-with-python-zero-to-pandas
# 
# 5- https://ourworldindata.org/coronavirus
# 
# 6- https://covid19.moh.gov.sa/
# 
# 7-https://github.com/
# 
# 8-https://www.kaggle.com/

# In[ ]:




