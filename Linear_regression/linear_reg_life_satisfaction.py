# -*- coding: utf-8 -*-
"""
Created on Tue May 14 03:56:46 2019

@author: SonyTF
"""


# For example, suppose you want to know if money makes people happy, 
# so you download the Better Life Index

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Better Life Index
bli=pd.read_csv("BLI2015.csv",thousands=',')
bli_tot_tmp=bli[bli["INEQUALITY"]=='TOT']
bli_tot=bli_tot_tmp.pivot_table(index='Country',columns='Indicator',values='Value')
    
bli_tot.info()


# GDP per capita
gdp_per_capita =pd.read_csv("WEO_Data.xls",thousands=',',delimiter='\t',encoding='latin1',na_values="n/a",index_col='Country')
gdp_per_capita.rename(columns={"2015":"GDPperCapita"}, inplace =True)

# Join the tables and sort by GDP per capita
full_country_stats=pd.merge(left=bli_tot,right=gdp_per_capita,left_index=True,right_index=True)
#full_country_stats.sort_values(by="GDP per capita", inplace=True)

full_country_stats.sort_values(by='GDPperCapita',inplace=True)

# =============================================================================
# remove_indices =[0,1,6,8,33,34,35]
# remove_indeices = ['Brazil','Mexico','Chile','Czech Republic','Norway','Switzerland','Luxembourg']
# =============================================================================

# removing the outliers 
# visualizing the scatter plot we get to see some extreme values which can be treated accordingly

plt.scatter(full_country_stats.index,full_country_stats.GDPperCapita)
plt.xticks(ticks=full_country_stats.index,rotation='vertical')
plt.grid()
plt.show()

# =============================================================================
# remove_indices1 =[0,1,6,8,33,34,35]
# remove_indices2 =[0,1,2,3,33,34,35]
# new_stats=full_country_stats.drop(full_country_stats.index[remove_indices1],axis=0)
# 
# =============================================================================

remove_indices = [0, 1, 6, 8, 33, 34, 35]
keep_indices = list(set(range(36)) - set(remove_indices))
new_stats=full_country_stats[["GDPperCapita", 'Life satisfaction']].iloc[keep_indices]


#full_country_stats[["GDPperCapita", 'Life satisfaction']].head()
plt.figure()
plt.scatter(new_stats.index,new_stats.GDPperCapita)
plt.xticks(ticks=new_stats.index,rotation='vertical')
plt.grid()
plt.show()

X=np.c_[new_stats['GDPperCapita']]
y=np.c_[new_stats['Life satisfaction']]


plt.figure()

plt.scatter(X,y)
plt.show()

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

regressor.fit(X,y)

X_test=[[25864]]
print(regressor.predict(X_test))

from sklearn.neighbors import KNeighborsRegressor
clf = KNeighborsRegressor(n_neighbors=3)
clf.fit(X,y)

print(clf.predict(X_test))
