#!/usr/bin/env python
# coding: utf-8

# # Graduation project
# ## The aim is to anlayse the dataset of an unknown company, provide insight in ways to improve and give recommendations from the analytical persective
# ### 1.  I will look into the period when the Orders came to determine the best time for resupply 
# ### 2.  I will classify the customers to determine the most beneficial customers
# ### 3. I will check which products are the most beneficial and which ones to get rid of
# 

# In[146]:


import pandas as pd 
from pandas import ExcelWriter,ExcelFile
import numpy as np
from datetime import datetime, date, time
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pmdarima as pm


# In[147]:


df = pd.read_excel('practice.xlsx')
df.head()


# In[82]:


df.shape


# In[148]:


# Convert period from object to daytime for the possibility of future analysis
df['Period']=pd.to_datetime(df['Period'])


# In[149]:


# Drop orders with 0 quantity ordered as it is a bug
print(df[df['Qty']==0].count()[0])
df.drop(df[df['Qty']==0].index, inplace=True, axis=0)


# In[85]:


# General infromation about all the data in the dataframe. 
# Qty tells us that on average clients buy 1-2 products of the same type
# The average client tops about 18-19 units of sale, however there is one client who reached 925.
# There are 355 unique Products out of which Product 25 is the most popular as it was ordered 3265 times
# There are 889 unique Clients with client 9 being the one who ordered the most items  
# At last the comany operates from January 2018 up to December 2021 possibly future values are pre-orders
print(df.describe())
print(df.describe(include=['object']))
df.describe(include=['datetime64'])


# In[320]:


#The most popular product is Product 25, The client with the biggest number of items ordered is Client 9
pr=df['Product'].value_counts()
cl=df['Client'].value_counts()
print(pr,'\n', cl) 


# In[92]:


# Here we can see the top 5 most valuable customers who order the most
# and top 5 customers who ordered the least in terms of unit of sales
df1=df.groupby(['Client'])['Sales'].describe().reset_index().sort_values('mean',ascending=False)
df1


# In[107]:


df.groupby(['Order'])['Sales'].mean().sort_values(ascending=False)


# In[305]:


# Divide period into years. 
df2018=df[df['Period'] < '2019-01-01']
df2019=df[(df['Period'] < '2020-01-01') & (df['Period']>='2019-01-01')]
df2020=df[(df['Period'] < '2021-01-01') & (df['Period']>='2020-01-01')]
df2021=df[(df['Period']>='2021-01-01')]


# In[206]:


plt.figure(figsize=(12,8))

plt.subplot(221)
plt.title('2018',loc='left')
df2018.groupby(df["Period"].dt.month)["Sales"].sum().plot.bar(color='green')
# axes[0,0].title('2018 Data',size=20)
plt.ylabel("Sales")

plt.subplot(222)
plt.title('2019',loc='left')
df2019.groupby(df["Period"].dt.month)["Sales"].sum().plot.bar(color='blue')
plt.ylabel("Sales")

plt.subplot(223)
plt.title('2020',loc='left')
df2020.groupby(df["Period"].dt.month)["Sales"].sum().plot.bar(color='purple')
plt.ylabel("Sales")

plt.subplot(224)
plt.title('2021',loc='left')
df2021.groupby(df["Period"].dt.month)["Sales"].sum().plot.bar(color='tomato')
plt.ylabel("Sales")

plt.show()


# From the sales data over 4 years period we could see that the company had a slow start in 2018, managing to reach about 6800 sales in December 2018. This data does not represent the full picture, as that is a typical selling behaviour in the first year of many companies. However, in 2019 the company started opearting well from the start as they had a minimum of about 13500 sales in December, which was the worst month, where September proved to be the best selling month. In 2020 the sales finally took a seasonal perspective, with peaks and falls. Here we could see that March, April , September and October are on the top, where January, June and December are at the bottom of sales per month. At last, 2021 data is given in terms of pre-orders which the company recieved. Therefore, it is not elegible for analysis. 
# 
# ### Overall, we could see the seasonal behaviour (during spring and autumn sales increase) in two fully operational years

# In[289]:


# As sales data showed, it is crucial to use only 2 years in between, which indicate full operational years
df2019_20=df[(df['Period'] < '2021-01-01') & (df['Period']>='2019-01-01')]


# In[200]:


df2019_20.groupby(df['Period'].dt.weekday)['Sales'].sum().plot(kind='bar',color='orange')
plt.xlabel("Days of the week")
plt.ylabel("Sales")
plt.title("Sales per week day")


# It appears that the majority of the sales happen in the beginning of the week, and drop significantly on the weekend.
# 
# ### The company should make sure to resupply before the beginning of the week, as otherwise they would lack products to proceed with sales.

# In[204]:


fig,axes = plt.subplots(figsize=(14,8))
df2019_20.groupby(df['Period'].dt.hour)['Sales'].sum().plot(kind='bar',color='orange')


# ### The majority of sales happen during the normal working days for European companies (from 9 to 17) where there is a drop during 13:00 which is the lunch break in most companies. 

# In[9]:


#preparing data for ABC analysis
df1=df[['Client','Sales']]
df1=df1.groupby('Client')['Sales'].sum().reset_index()
df1.sort_values('Sales',inplace=True,ascending = False)
df1.head()


# In[10]:


# Add accumulated column to count the percentage
df1['Accumulated'] = df1['Sales'].cumsum()
df1['Percent'] = df1['Accumulated']/df1['Accumulated'].max()*100
df1


# In[11]:


# classify clients by their percentige of overall sales
def ABC(x):
    if x <= 20:
        return 'A'
    elif x > 20 and x <= 90:
        return 'B'
    else:
        return 'C'
df1['ABC'] = df1['Percent'].map(ABC)
df1.head(20)


# In[12]:


# Number of Customers in each category
df1['ABC'].value_counts()


# ### There are only 7 clients of type A, while the majority of the clients are of type C

# In[13]:


# Plotting the number of Customers of each group 
df1plot = df1.groupby('ABC')['Client'].count().reset_index()
plt.figure(figsize = (11,7))

barlist=plt.bar('ABC','Client',data=df1plot)
barlist[0].set_color('r')

barlist[1].set_color('orange')

barlist[2].set_color('b')

plt.xlabel('Client class')
plt.ylabel('Number of clients')
plt.title('Number of clients per category')

plt.show()


# In[73]:


df2plot = df1.groupby('ABC')['Sales'].sum().reset_index()
plt.figure(figsize = (11,7))
# df['a'].hist()
barlist=plt.bar('ABC','Sales',data=df2plot)
barlist[0].set_color('r')
barlist[1].set_color('orange')
barlist[2].set_color('b')
plt.xlabel('Client class')
plt.ylabel('Sales')
plt.title('Client sales per group')

plt.show()


# It appears that even so there are only 7 people in group A, they raised about 19.4% of overall sales. Group B has an unnatural behaviour, as it was expected to be the second best group in terms of sales, but it provides the vast majority of sales with 70.5% of all the sales. The biggest group of clients which is group 3 reached only 10% of sales. It is also visible by the graph that group A has on average higher sales than of any other group.

# In[20]:


# Percentage of sales per group
df2=df1.groupby(df1['ABC'])['Sales'].sum()
A=df2[0]/df1['Accumulated'].max() * 100
B=df2[1]/df1['Accumulated'].max() * 100 
C=df2[2]/df1['Accumulated'].max() * 100
print(A,"A\n", B,"B\n",C,"C\n")


# In[35]:


# Preparation for vizualisation 
A = df1[df1['ABC']=='A']
B = df1[df1['ABC']=='B']
C = df1[df1['ABC']=='C']


# In[71]:


# Whisker plot diagram to see the average sum of sales for each group.
plt.figure(figsize=(12,8))


plt.subplot(221)

plt.boxplot(A['Sales'],meanline=True, showmeans=True, showcaps=True,showbox=True, showfliers=True)
plt.ylabel("Sales")
plt.title('A')

plt.subplot(222)
plt.boxplot(B['Sales'],meanline=True, showmeans=True, showcaps=True,showbox=True, showfliers=True)
plt.ylabel("Sales")
plt.title('B')

plt.subplot(223)
plt.boxplot(C['Sales'],meanline=True, showmeans=True, showcaps=True,showbox=True, showfliers=True)
plt.ylabel("Sales")
plt.title('C')


# Whisker plot diagrams are all plotted in different scales, so it is important to pay attention to the y-axis. From it we could see that group A has the biggest sales in the group with mean at around 14500. On the other hand, the biggest group with the biggest percentage contribution to sales has mean at around 1200. At last, Group C with the most customers in a group has a mean of 90. 
# 
# 

# In[80]:


print(A.describe(), "A")
print(B.describe(), "B")
print(C.describe(), "C")


# ### Overall, the company should think of loyalty programs to try and attract customers to increase their class. The best thibg to concentrate on in the beginning would probably be to make as much people from Group C to convert to B as here lies the majority of sales. I would recommend offering discounts and personal assistance as a starting point.

# # XYZ analysis for products

# In[319]:


# Use 2020 data as it is a stable year for ABC analysis, divide it to 12 month
df2020=df2020.assign(month = pd.to_datetime(df2020['Period']).dt.month)
df2020.head()


# In[320]:


# sum quantity for each product and month
df_units=df2020.groupby(['Product','month'])['Qty'].sum().to_frame().reset_index()
df_units.head()


# In[321]:


#reshape data into wide format
df_units = df_units.pivot(index='Product',columns='month',values='Qty').add_prefix('m').reset_index().fillna(0)


# In[322]:


#calculate standard deviation for each product and month in terms of demand
df_units['std_demand'] = df_units[['m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12']].std(axis=1)


# In[323]:


#Calculate total demand for each month
df_units = df_units.assign(total_demand = df_units['m1'] + df_units['m2'] + df_units['m3'] 
                    + df_units['m4'] + df_units['m5'] + df_units['m6'] 
                    + df_units['m7'] + df_units['m8'] + df_units['m9'] 
                    + df_units['m10'] + df_units['m11'] + df_units['m12'] )


# In[324]:


#Average monthly demand
df_units=df_units.assign(avg_demand = df_units['total_demand'] / 12)
df_units


# In[326]:


# Coefficient of variation to determine forecastability
df_units['cov_demand'] = df_units['std_demand'] / df_units['avg_demand']
df_units.sort_values(by='cov_demand', ascending=True).head()


# In[328]:


print(df_units['cov_demand'].min(),"Min\n", df_units['cov_demand'].max(),"Max\n",df_units['cov_demand'].mean(),"Mean")


# In[330]:


f, ax = plt.subplots(figsize=(15, 6))
ax = sns.distplot(df_units['cov_demand']).set_title("Coefficient of Variation",fontsize=15)


# It appears that some products with coefficient of variation bigger than 1 exist, it is quite hard to predict how these products would sale due to fluctuation in demand.

# In[331]:


def xyz_classify_product(cov):
    
    if cov <= 0.5:
        return 'X'
    elif cov > 0.5 and cov <= 1.0:
        return 'Y'
    else:
        return 'Z'


# In[332]:


# Assign XYZ classes
df_units['xyz_class'] = df_units['cov_demand'].apply(xyz_classify_product)


# In[333]:


df_units.xyz_class.value_counts()


# ### We could see that the majority of the products are hard to forecast as these have a very high coefficient of variation. However, products X and Y are much easier to forecast and predict, thus, solving the supply issue won't be a problem for these groups.

# In[334]:


df_units.head()


# In[337]:


# Overview of X,Y,Z where we cans see that 27 group X products have
# higher total demand than group Y and Z combined.
df_units.groupby('xyz_class').agg(
    total_Products=('Product', 'nunique'),
    total_demand=('total_demand', 'sum'),    
    std_demand=('std_demand', 'mean'),      
    avg_demand=('avg_demand', 'mean'),
    avg_cov_demand=('cov_demand', 'mean'),
)


# In[339]:


# new dataframe for another reshape to plot
df_monthly = df_units.groupby('xyz_class').agg(
    m1=('m1', 'sum'),
    m2=('m2', 'sum'),
    m3=('m3', 'sum'),
    m4=('m4', 'sum'),
    m5=('m5', 'sum'),
    m6=('m6', 'sum'),
    m7=('m7', 'sum'),
    m8=('m8', 'sum'),
    m9=('m9', 'sum'),
    m10=('m10', 'sum'),
    m11=('m11', 'sum'),
    m12=('m12', 'sum'),
)
df_monthly.head()


# In[340]:


df_monthly_unstacked = df_monthly.unstack('xyz_class').to_frame()
df_monthly_unstacked = df_monthly_unstacked.reset_index().rename(columns={'level_0': 'month', 0: 'demand'})
df_monthly_unstacked.head()


# In[342]:


f, ax = plt.subplots(figsize=(15, 6))
ax = sns.barplot(x="month", 
                 y="demand", 
                 hue="xyz_class", 
                 data=df_monthly_unstacked,
                 palette="Blues_d")\
                .set_title("XYZ demand by month",fontsize=15)


# We could see that products of category X have the constantly higher demand than any other group. We could see that there is a drop in winter and summer times for all the groups.Therefore, the company needs to constantly keep the stock of these products up and running. Also, the demand of group Z is low overall, which means that for Customers this product is used quite rarely, so it would be a good idea not to pile this product in stock and get supplies only when it runs out. 

# # ABC inventory Analysis

# In[345]:


# use previous data, add Sales
df_product = df2020.groupby('Product').agg(unique_purchases=('Product', 'nunique'),
    total_units=('Qty', 'sum'),
    total_revenue=('Sales', 'sum'),
).sort_values(by='total_revenue', ascending=False).reset_index()
df_product


# In[349]:


df_product['revenue_cumsum'] = df_product['total_revenue'].cumsum()
df_product['revenue_total'] = df_product['total_revenue'].sum()
df_product['revenue_running_percentage'] = (df_product['revenue_cumsum'] / df_product['revenue_total'])
df_product


# In[367]:


def abc_classify_product(percentage):
    
    if percentage > 0 and percentage <= 0.5:
        return 'A'
    elif percentage > 0.5 and percentage <= 0.9:
        return 'B'
    else:
        return 'C'


# In[368]:


# Apply class and rank 
df_product['abc_class'] = df_product['revenue_running_percentage'].apply(abc_classify_product)
df_product['abc_rank'] = df_product['revenue_running_percentage'].rank().astype(int)
df_product.head()


# In[369]:


# Overall information about ABC categories 
df_abc = df_product.groupby('abc_class').agg(
    total_product=('Product', 'nunique'),
    total_units=('total_units', sum),
    total_revenue=('total_revenue', sum),
).reset_index()

df_abc.head()


# In[374]:


f, ax = plt.subplots(figsize=(15, 6))
ax = sns.barplot(x="abc_class", 
                 y="total_revenue", 
                 data=df_abc,
                 palette="Blues_d")\
                .set_title("Revenue by ABC class",fontsize=15)


# Group A has the least members of the group-14, but generates the biggest sale benefit for the company due to having the biggest number of items in stock. It is also interesting to note that group B has unnaturally high revenue. Group C has the least products in stock, but yet the highest diversity of the products presented. 

# In[375]:


f, ax = plt.subplots(figsize=(15, 6))
ax = sns.barplot(x="abc_class", 
                 y="total_units", 
                 data=df_abc,
                 palette="Blues_d")\
                .set_title("Quantity by ABC class",fontsize=15)


# # ABCXYZ Analysis

# In[376]:


# Make one df out of abc and xyz dfs
df_abc = df_product[['Product','abc_class','abc_rank','total_revenue']]
df_xyz = df_units.copy()
df_abc_xyz = df_abc.merge(df_xyz, on='Product', how='left')
df_abc_xyz.head()


# In[377]:


df_abc_xyz['abc_xyz_class'] = df_abc_xyz['abc_class'].astype(str) + df_abc_xyz['xyz_class'].astype(str)


# In[380]:


# summarize statistics from both dfs
df_abc_xyz_summary = df_abc_xyz.groupby('abc_xyz_class').agg(
    total_products=('Product', 'nunique'),
    total_demand=('total_demand', sum),
    avg_demand=('avg_demand', 'mean'),    
    total_revenue=('total_revenue', sum),    
).reset_index()

df_abc_xyz_summary.sort_values(by='total_revenue', ascending=False)


# In[384]:


f, ax = plt.subplots(figsize=(15, 6))
ax = sns.barplot(x="abc_xyz_class", 
                 y="total_products", 
                 data=df_abc_xyz_summary,
                 palette="Blues_d")\
                .set_title("Products by ABC-XYZ class",fontsize=15)


# ## The majority of the productws in stock are of type CZ which is hard to manage and predict, while having a low product value. The company should consieder finding substitution for these products, by analysing the demand of their customers or rival companies.

# In[383]:


f, ax = plt.subplots(figsize=(15, 6))
ax = sns.barplot(x="abc_xyz_class", 
                 y="total_revenue", 
                 data=df_abc_xyz_summary, 
                 palette="Blues_d")\
                .set_title("Revenue by ABC-XYZ class",fontsize=15)


# In[385]:


f, ax = plt.subplots(figsize=(15, 6))
ax = sns.barplot(x="abc_xyz_class", 
                 y="total_demand", 
                 data=df_abc_xyz_summary, 
                 palette="Blues_d")\
                .set_title("Demand by ABC-XYZ class",fontsize=15)


# ### There is a high demand for AX products, which also implies higher sales from this group as well. It is interesting that the biggest group in terms of quantity is group CZ, but it has one of the lowest demands in line. Overall, I would still advice to look into the products that are of the type CZ and trying to find other products thtat would be more interesting for the customer to purchase.

# ## AX
# - High value
# - Steady demand
# - Easy to forecast
# - Easy to manage
# 
# ## BX
# - Medium value
# - Steady demand
# - Easy to forecast
# - Easy to manage
# 
# ## CX
# - Low value
# - Steady demand
# - Easy to forecast
# - Easy to manage
# 
# ## AY
# - High value
# - Variable demand
# - Harder to forecast
# - Harder to manage
# 
# ## BY
# - Medium value
# - Variable demand
# - Harder to forecast
# - Harder to manage
# 
# ## CY
# - Low value
# - Variable demand
# - Harder to forecast
# - Harder to manage
# 
# ## AZ
# - High value
# - Sporadic demand
# - Difficult to forecast
# - Difficult to manage
# 
# ## BZ
# - Medium value
# - Sporadic demand
# - Difficult to forecast
# - Difficult to manage
# 
# ## CZ
# - Low value
# - Sporadic demand
# - Difficult to forecast
# - Difficult to manage

# # In conclusion
# 
# - There is an effect of seasons on sales. Sales drop in summer and winter and raise in autumn and spring.
# - During the weekends there is a drop in sales.
# - Sales happen mostly during normal working hours 9-17.
# 
# ### Solution: Resupply products on weekend, especially by the end of summer/winter
# 
# - there are a lot of products that have low demand and high stock
# 
# ### Solution: try to find products interesting for the customers in winter and summer time instead of resupplying the CZ products.
# 
# - There are 3 groups of Customers with certain buying patterns. The majority lies in group C.
# 
# ### Solution: Implement loyalty card scheme to motivate customers with discounts and special events.
