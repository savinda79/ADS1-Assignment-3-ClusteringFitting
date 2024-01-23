# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 18:55:44 2024

@author: Savinda Rangika De Abrew
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib 
import sklearn.preprocessing as pp
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import scipy.optimize as opt
import importlib as imlib

# Import the custize libarry and Loading it
import errors as err
imlib.reload(err)


# setup a scaler object
scaler = pp.RobustScaler()
# and no to the scaling

# cluster by cluster
#plt.figure(figsize=(10.0, 10.0))

# Select a colour map. For distinuishing groups qualitative colour maps are best suited.
# One wants contrast.
cm = matplotlib.colormaps["Paired"]

ncluster = 3


#df_climatechange = pd.read_csv("wb_climatechange_data.csv",skiprows=4)
#print(df_climatechange.info())


def read_file(filename):
    #Read and return the data frame
    climate_df = pd.read_csv( filename, skiprows=4)
    
    return climate_df


#def filerwe_data()


df_climatechange = read_file("wb_climatechange_data.csv")
#print(df_climatechange.info())

#Filter data from df_climatechange belons to Oil consumption and CO2

code=('EN.ATM.CO2E.EG.ZS','EG.USE.PCAP.KG.OE')
filtered_data = df_climatechange[df_climatechange['Indicator Code'].isin(code)]
df_Energy = filtered_data[['Country Name','Indicator Code', '2010']].copy()
#df_Energy.set_index('Country Name',inplace=True)
df_Energy.set_index('Country Name',inplace=True)


df_Energy['2010'] = df_Energy['2010'].replace(np.nan, 0)
print(df_Energy.dtypes)

#Trnaspose data from Pivot Method

pivoted_df = df_Energy.pivot( columns='Indicator Code', values=['2010'])

print(pivoted_df)


plt.figure(figsize=(8, 8))
plt.scatter(pivoted_df[pivoted_df.columns[0]], pivoted_df[pivoted_df.columns[1]])

plt.xlabel("Energy use (kg of oil equivalent per capita)")
plt.ylabel("CO2 intensity (kg per kg of oil equivalent energy use)")

plt.show()

# and set up the scaler
# extract the columns for clustering
df_ex = pivoted_df[[pivoted_df.columns[0], pivoted_df.columns[1]]]
scaler.fit(df_ex)

norm = scaler.transform(df_ex)

plt.figure(figsize=(8, 8))
plt.scatter(norm[:, 0], norm[:, 1])

plt.xlabel("Energy use (kg of oil equivalent per capita)")
plt.ylabel("CO2 intensity (kg per kg of oil equivalent energy use)")

plt.show()



def one_silhoutte(xy, n):
    """ Calculates silhoutte score for n clusters """

    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=n, n_init=20)

    # Fit the data, results are stored in the kmeans object
    kmeans.fit(xy)     # fit done on x,y pairs

    labels = kmeans.labels_
    
    # calculate the silhoutte score
    score = (skmet.silhouette_score(xy, labels))

    return score


# calculate silhouette score for 2 to 10 clusters
for ic in range(2, 11):
    score = one_silhoutte(norm, ic)
    print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")   # allow for minus signs


# set up the clusterer with the number of expected clusters
kmeans = cluster.KMeans(n_clusters=3, n_init=20)

# Fit the data, results are stored in the kmeans object
kmeans.fit(norm)     # fit done on x,y pairs

# extract cluster labels
labels = kmeans.labels_

# extract the estimated cluster centres and convert to original scales
cen = kmeans.cluster_centers_
cen = scaler.inverse_transform(cen)
xkmeans = cen[:, 0]
ykmeans = cen[:, 1]    

# print(df_norm)
#print(labels)
#print(norm)

plt.figure(figsize=(8.0, 8.0))

# plot data with kmeans cluster number
plt.scatter(pivoted_df[pivoted_df.columns[0]], pivoted_df[pivoted_df.columns[1]], 10, labels, marker="o", cmap=cm)
    
# show cluster centres
plt.scatter(xkmeans, ykmeans, 45, "k", marker="d")
    
#plt.xlabel("GDP per head 1980")
#plt.ylabel("GDP growth/year [%]")

plt.xlabel("Energy use (kg of oil equivalent per capita)")
plt.ylabel("CO2 intensity (kg per kg of oil equivalent energy use)")
plt.title("Energy useag vs CO2 intensity year 2010")
#plt.savefig('Cluster.png')
plt.show()


""" 

Start module of Data Fitting and Prediction.

"""


#df_climatefit = pd.read_csv("wb_climatechange_data.csv",skiprows=4)

#Call the same function again and asiing it to df_climatefit
df_climatefit = read_file("wb_climatechange_data.csv")


df_climatefit = df_climatefit[df_climatefit['Indicator Code']=='EG.USE.PCAP.KG.OE']


#Filter the Canda data

df_climatefit_Canada = df_climatefit[df_climatefit['Country Name']=='Canada']




"""
Funtion to select the Columns to filler out to 
create the df_canada data frame

"""
def filter_trnaspose(df_climatefit_Canada):


    df_canada = df_climatefit_Canada[['Country Name',
                                  '1960', '1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968',
                                  '1969', '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977',
                                  '1978', '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986',
                                  '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995',
                                  '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004',
                                  '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
                                  '2014', '2015']]
    df_canada.set_index('Country Name', inplace=True)


    # Using Transpos mehtod trans pos the df_canada data frame
    df_canada_Tp = df_canada.T

    return df_canada_Tp


#Call the filter_trnaspose function 

df_canada_Tp = filter_trnaspose(df_climatefit_Canada)


df_canada_Tp.plot()
plt.show()

df_canada_Tp['Year'] = df_canada_Tp.index
df_canada_Tp['Year'] = df_canada_Tp.index

df_canada_Tp=df_canada_Tp.reset_index(drop=True)

df_canada_Tp.plot("Year", "Canada")
plt.show()



def exponential(t, n0, g):
    """Calculates exponential function with scale factor n0 and growth rate g."""
    
    # makes it easier to get a guess for initial parameters
    t = t - 1990
    
    f = n0 * np.exp(g*t)
    
    return f




df_canada_Tp['Year'] = pd.to_numeric(df_canada_Tp['Year'])

param, covar = opt.curve_fit(exponential, df_canada_Tp["Year"], df_canada_Tp["Canada"])

print("Canada oil 1980", param[0]/1e9)
print("growth rate", param[1])

plt.figure()

plt.plot(df_canada_Tp["Year"], exponential(df_canada_Tp["Year"], 1.2e12, 0.03), label = "trial fit")
plt.plot(df_canada_Tp["Year"], df_canada_Tp["Canada"])

plt.xlabel("Year")
#plt.legend()

plt.show()
param, covar = opt.curve_fit(exponential, df_canada_Tp["Year"], df_canada_Tp["Canada"], p0=(1.2e12, 0.03))

print(f"GDP 1960: {param[0]/1e9:6.1f} billion $")
print(f"growth rate: {param[1]*100:4.2f}%")

param, covar = opt.curve_fit(exponential, df_canada_Tp["Year"], df_canada_Tp["Canada"], p0=(1.2e12, 0.03))

print(f"GDP 1960: {param[0]/1e9:6.1f} billion $")
print(f"growth rate: {param[1]*100:4.2f}%")


df_canada_Tp["fit"] = exponential(df_canada_Tp["Year"], *param)
df_canada_Tp.plot("Year", ["Canada", "fit"])
plt.show()



def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0 and growth rate g"""
    
    f = n0 / (1 + np.exp(-g*(t - t0)))
    
    return f




param, covar = opt.curve_fit(logistic, df_canada_Tp["Year"], df_canada_Tp["Canada"])
df_canada_Tp["trial"] = logistic(df_canada_Tp["Year"], 3e12, 0.10, 1960)

df_canada_Tp.plot("Year", ["Canada", "trial"])
plt.show()

param, covar = opt.curve_fit(logistic, df_canada_Tp["Year"], df_canada_Tp["Canada"], 
                            p0=(3e12, 0.1, 1960))
df_canada_Tp["fit"] = logistic(df_canada_Tp["Year"], *param)

df_canada_Tp.plot("Year", ["Canada", "fit"])
plt.title("Logistic Curve fit of Energy Usage in Canada 1960-2015,")

plt.show()

#plt.savefig('Curvefit2_Canada.png')



# extract variances and take square root to get sigmas
var = np.diag(covar)
sigma = np.sqrt(var)

print(f"turning point {param[2]: 6.1f} +/- {sigma[2]: 4.1f}")
print(f"GDP at turning point {param[0]: 7.3e} +/- {sigma[0]: 7.3e}")
print(f"growth rate {param[1]: 6.4f} +/- {sigma[1]: 6.4f}")


# create array for forecasting
year = np.linspace(1960, 2030, 100)
forecast = logistic(year, *param)

plt.figure()
plt.plot(df_canada_Tp["Year"], df_canada_Tp["Canada"], label="Oil Kg")
plt.plot(year, forecast, label="forecast")

plt.xlabel("year")
plt.ylabel("Energy use (kg of oil equivalent per capita)")
plt.title("Forecast Energy Usage in Canada in Kg of Oil, 1960-2030")

plt.legend()
plt.show()



"""
create array for forecasting

"""


year = np.linspace(1960, 2030, 100)
forecast = logistic(year, *param)
sigma = err.error_prop(year, logistic, param, covar)
up = forecast + sigma
low = forecast - sigma

plt.figure()
plt.plot(df_canada_Tp["Year"], df_canada_Tp["Canada"], label="Oil Kg")
plt.plot(year, forecast, label="forecast")

plt.fill_between(year, low, up, color="yellow", alpha=0.7)
plt.xlabel("Year")
plt.ylabel("Energy use (kg of oil equivalent per capita)")
plt.title("Forecast Energy Usage in Canada in Kg of Oil, 1960-2030")
plt.legend()
#plt.savefig('Forcast_Canada.png')
plt.show()


