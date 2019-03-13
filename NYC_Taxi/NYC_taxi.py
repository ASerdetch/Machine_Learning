"""
NEW YORK TRIP DURATION DATASET FROM KAGGLE 

---
TO DO LIST: 
    1. set stricter limit on feature importance 
    2. do further tests with xgboost
    3. create 2 seperate datasets for in and outside the city and train them seperately 
    4. do pca with steps to reduce dimensionality
"""
#%% IMPORT KEY DATASETS 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from geopy.distance import geodesic
import datetime
import re

#%% PRELIMINARY EDA-TRAIN DATA

#load data 
df=pd.read_csv('train.csv')

#determine number of unique & nans values in each column 
for x in df:
    print(x, df[x].nunique())
    print(x, df[x].isnull().sum())
# no nan values
# vendor_id and store_and_fwd_flag only have 2 values and can be label encoded
# id is unique per ride 
    
#get description and summary stats on all columns 
df_head=df.head()
df_desc=df.describe()
df.dtypes
# dropoff_datetime column will need to be removed to avoid data leakage 
# trip_duration has some outliers, based on stats, will need to be looked at
# lat/lomg also seem to have outliers, will see if they are linked with trip duration outliers  
#no unexpected or problematic data types; all objects will be processed or removed

#drop dropff_datetime to avoid data leakage
df=df.drop(['dropoff_datetime'], axis=1)

#=======INITIAL DATA VISUALIZATION=============

#pass count, vendor id & trip duration

#passenger_count vs vendor_id 
sns.pairplot(df, hue='vendor_id',vars=['passenger_count'], size=5)
# vendor 2 seems to get more rides with multiple passengers 
# majority are one person trips 

#trip duration vs vendor id
sns.pairplot(df[df['trip_duration'] < 3000], hue='vendor_id',vars=['trip_duration'], size=5)
# trip duration limited to avoid outlier skewing visualization 
# majority of trips are <2000s 
# relatively even distribution between vendor_ids 

#passenger count vs. trip duration
sns.pairplot(df[df['trip_duration'] < 3000], hue='passenger_count',vars=['trip_duration'], size=5)
# no real difference; seems to be pretty uniform acorss  

#trip_duration
plt.scatter(np.arange(len(df)), df['trip_duration'].sort_values(ascending=True))
# very few, very extreme outliers

#trip_duration, w/o outliers 
plt.scatter(np.arange(len(df)), df['trip_duration'].sort_values(ascending=True), s=2)
plt.ylim([0,10000]) 
# without outliers, a nice polynomial shape 

#SPATIAL DISTRIBUTION EXPLORATION 

plt.scatter(df['pickup_longitude'], df['pickup_latitude'], s=0.1)
plt.scatter(df['dropoff_longitude'],df['dropoff_latitude'], s=0.1)
# there are some extreme outliers in lat/long, as discovered before 
# are outliers in trip duraiton related to outliers in time? 

#will make a seperate df based on the trip duration conditions; gives opportunity to play around 
df_regdur=df[df['trip_duration'] < 10000]
plt.scatter(df_regdur['pickup_longitude'], df_regdur['pickup_latitude'], s=0.1)
plt.scatter(df_regdur['dropoff_longitude'],df_regdur['dropoff_latitude'], s=0.1)
# this is not only a trip duration issue; will first finish spatial data exploration, then look at outliers 

#look at distribution of non-outlier data 
plt.hist(df_regdur['trip_duration'], bins=100)
# majority of trips are <2K seconds 


#city limits: #ny city limits: 41-40.5/ -74.2, -73.7
#will set up shortcuts for limits
city_lat=[40.5,41]
city_long=[-74.2,-73.7]
trip_lim=df['trip_duration']<10000
df_city=(((df['pickup_latitude']<41)&(df['pickup_latitude'] >40.5))&
        ((df['dropoff_latitude'] <41)&(df['dropoff_latitude'] >40.5))&
        ((df['pickup_longitude'] > -74.2)&(df['pickup_longitude'] < -73.7))&
        ((df['dropoff_longitude'] > -74.2)&(df['dropoff_longitude'] < -73.7)))

#will do analysis in the city
#will not put distribution maps into for loop, since they are already computationally expensive 

#trip duration spatial distribution, limit trip duraiton for better visualization
plt.figure(figsize=(20,20))
plt.scatter(df['pickup_longitude'], df['pickup_latitude'], s=0.1, c=df['trip_duration'], vmax=2000)
plt.scatter(df['dropoff_longitude'],df['dropoff_latitude'], s=0.1, c=df['trip_duration'], vmax=2000)
plt.xlim(city_long)
plt.ylim(city_lat)
plt.colorbar()
#futher outside city center=longer trips, but the 'most touristy areas' (central park, etc) also see long trip times

#spatial distribution of vendors
plt.figure(figsize=(20,20))
plt.scatter(df['pickup_longitude'], df['pickup_latitude'], s=0.1, c=df['vendor_id'])
plt.scatter(df['dropoff_longitude'],df['dropoff_latitude'], s=0.1, c=df['vendor_id'])
plt.xlim(city_long)
plt.ylim(city_lat)
plt.colorbar()
# even distribution 

#spatial distribution of passenger count: 
plt.figure(figsize=(20,20))
plt.scatter(df['pickup_longitude'], df['pickup_latitude'], s=0.1, c=df['passenger_count'])
plt.scatter(df['dropoff_longitude'],df['dropoff_latitude'], s=0.1, c=df['passenger_count'])
plt.xlim(city_long)
plt.ylim(city_lat)
plt.colorbar()
# nothing particularly suprirsing/interesting 

#lat/long vs tripcount and trip duration and removing outliers 
fig, [ax1, ax2]=plt.subplots(2,1)
col='blue'
for x in ['pickup_longitude','dropoff_longitude']:
    ax1.hist(df[x][df_city], bins=100, alpha=0.5, color=col)
    ax2.scatter(df[x][trip_lim], df['trip_duration'][trip_lim], s=0.1, alpha=0.2, color=col)
    ax2.set_xlim(city_long)
    col='red'
plt.show()
    
fig, [ax1, ax2]=plt.subplots(2,1)
col='blue'
for x in ['pickup_latitude','dropoff_latitude']:
    ax1.hist(df[x][df_city], bins=1000, alpha=0.5, color=col)
    ax2.scatter(df[x][trip_lim], df['trip_duration'][trip_lim], s=0.1, alpha=0.2, color=col)
    ax2.set_xlim(city_lat)
    col='red'
plt.show()
# longitude primarily in manhattan, no general difference between the two
# latitudeprimarily around central park/times square area (and then soho, etc. on south end)
# significant increases in trip duration at ~40.645 and 40.745
# significant increases in trip duration at ~-74.00/73.88/73.78
# will want to find better descriptors for these once we can confirm this is not an artifact of travel distance

#%%ANALYSIS OF OUTLIERS 
#determine if there are common themes among them 

#===========SPATIAL OUTLIERS==============

#data analysis outside city limits 
df_far= df[(df['pickup_latitude'] >41)|(df['pickup_latitude'] <40.5)|
        (df['dropoff_latitude'] >41)|(df['dropoff_latitude'] <40.5)|
        (df['pickup_longitude'] > -73.7)|(df['pickup_longitude'] < -74.2)|
        (df['dropoff_longitude'] > -73.7)|(df['dropoff_longitude'] < -74.2)]

df_dist_out=df[df['id'].isin(df['id'][(abs(df['pickup_latitude'] - np.mean(df['pickup_latitude']))
                                 < 2 * np.std(df['pickup_latitude'])) & 
                                 (abs(df['pickup_longitude'] - np.mean(df['pickup_longitude']))
                                 < 2 * np.std(df['pickup_longitude'])) &
                                              (abs(df['dropoff_latitude'] - np.mean(df['dropoff_latitude']))
                                 < 2 * np.std(df['dropoff_latitude'])) &
                                 (abs(df['dropoff_longitude'] - np.mean(df['dropoff_longitude']))
                                 < 2 * np.std(df['dropoff_longitude']))])==False]
                                              
plt.figure(figsize=(20,20))                                              
plt.scatter(df_far['pickup_longitude'],df_far['pickup_latitude'],s=0.8,c='red')
plt.scatter(df_far['dropoff_longitude'], df_far['dropoff_latitude'], s=0.8, c='red')
plt.scatter(df_dist_out['pickup_longitude'],df_dist_out['pickup_latitude'],s=0.1, c='blue', alpha=0.5)
plt.scatter(df_dist_out['dropoff_longitude'], df_dist_out['dropoff_latitude'], s=0.1, c='blue', alpha=0.5)
#plt.ylim([40.4,41.2])
#plt.xlim([-74.5,-73.5])
plt.xlim(city_long)
plt.ylim(city_lat)
# both the city limit and stat outliers have all the super extreme outliers out there 
# however, outliers do not limit city; if need to limit or define, will use city limits, if anything

plt.scatter(np.arange(len(df_far)),df_far['trip_duration'].sort_values(ascending=True), s=0.5)
plt.ylim([0,10000])
# more or less the same structure as within the city

# trip duration vs vendor id/passanger count for outside city
sns.pairplot(df_far[df_far['trip_duration'] < 3000], hue='vendor_id',vars=['trip_duration'], size=5)
sns.pairplot(df_far[df_far['trip_duration'] < 3000], hue='passenger_count',vars=['trip_duration'], size=5)
#nothing significant 

#vendor id & passenger count
sns.pairplot(df[df_far], hue='passenger_count',vars=['vendor_id'], size=5)
# relatively even; though vendor 2 the only one that carries 6-7 passengers in these locations 

plt.hist(df['trip_duration'][df_far], bins=100)
# trip duration more or less matches the rest 
# nothing out of the ordnary for location other than locaiton itself 

#===================TEMPORAL OUTLIERS================

#isolate trip duration outliers, see if they can be categorized/identified 
df_long=df[df['id'].isin(df['id'][abs(df['trip_duration'] - np.mean(df['trip_duration']))
                                 < 2 * np.std(df['trip_duration'])])==False]
    
plt.figure(figsize=(20,20))                                              
plt.scatter(df_long['pickup_longitude'],df_long['pickup_latitude'], c='red', alpha=0.5)
plt.scatter(df_long['dropoff_longitude'], df_long['dropoff_latitude'], c='blue', alpha=0.2)
#plt.ylim([40.4,41.2])
#plt.xlim([-74.5,-73.5])
#plt.xlim(city_long)
#plt.ylim(city_lat)
# still exists within city and outside of it 

#look at lat/long components individually 
plt.scatter(df_long['pickup_longitude'], df_long['trip_duration'], c='red')
plt.scatter(df_long['dropoff_longitude'], df_long['trip_duration'],c='blue', alpha=0.3)

plt.scatter(df_long['pickup_latitude'], df_long['trip_duration'], c='red')
plt.scatter(df_long['dropoff_latitude'], df_long['trip_duration'],c='blue', alpha=0.3)

#there are 4 trip durations that are far higher than others; will see if the 4 of them have anything in common 
df_long=df_long.sort_values(by='trip_duration', ascending=False)[:4]

#lets do the above visualizations again

#major outliers in context of rest of points 
plt.figure(figsize=(20,20))   
plt.scatter(df['pickup_longitude'], df['pickup_latitude'], s=0.1, c=[0.5,0.5,0.5])
plt.scatter(df['dropoff_longitude'],df['dropoff_latitude'], s=0.1, c=[0.5,0.5,0.5])                                           
plt.scatter(df_long['pickup_longitude'],df_long['pickup_latitude'], c='red', alpha=0.5)
plt.scatter(df_long['dropoff_longitude'], df_long['dropoff_latitude'], c='blue', alpha=0.2)
plt.ylim([40,41.2])
plt.xlim([-74.5,-73.5])

# these trips are all within the city; maybe the drivers were asked to wait? 
# these trips occured on Jan 5 and Feb 15, early in the morning/late at night. 
# these may be subject to removal; will see how model does with and without them 

#%%FEATURE ENGINEERING ROUND 1 (& TO MAKE REST OF EDA EASIER)

#convert date string into datetime and seperate
df['pickup_datetime']=df['pickup_datetime'].apply(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))

df['year']=df['pickup_datetime'].apply(lambda x: x.year)
df['month']=df['pickup_datetime'].apply(lambda x: x.month)
df['day']=df['pickup_datetime'].apply(lambda x: x.day)
df['hour']=df['pickup_datetime'].apply(lambda x: x.hour)
df['minute']=df['pickup_datetime'].apply(lambda x: x.minute)
df['second']=df['pickup_datetime'].apply(lambda x: x.second)
df['day_of_week']=df['pickup_datetime'].apply(lambda x: x.isoweekday())
df['week_of_year']=df['pickup_datetime'].apply(lambda x: x.strftime('%W'))

#will keep datetime for further conversion, and will create new version for label encoding 
df['pickup_datetime_labelenc']=df['pickup_datetime']

#label encoding so we can analyze store_and_fwd_flag and label encode pickup datetime 
for_labels=[
        'vendor_id',
        'pickup_datetime_labelenc',
        'store_and_fwd_flag']

for x in for_labels:
    labelenc=LabelEncoder()
    df[x]=labelenc.fit_transform(df[x])


#%%EDA ROUND 2
    
#new stats and head 
df_head=df.head()
df_desc=df.describe()

#see yearrange
df['year'].value_counts()
# can delete this column 

#explore store_and_fwd_flag

plt.hist(df['store_and_fwd_flag'])
df['store_and_fwd_flag'].value_counts()
#only 8000 of these; meaning that system does not fail often 

sns.pairplot(df[df['store_and_fwd_flag'] ==1], hue='vendor_id',vars=['trip_duration'], size=5)
# only with 1 vendor, trip duration has more or less same distribution 

sns.pairplot(df[df['store_and_fwd_flag'] ==1], hue='passenger_count',vars=['trip_duration'], size=5)
# no discrimination on passenger count 

#when do these failures occur? 
plt.hist(df['pickup_datetime_labelenc'][df['store_and_fwd_flag']==1], bins=100)
# no real trend or one big event, just seems to happen, though more on a few days than others 

#are these events allocated to specific part of city? 
plt.figure(figsize=(20,20))
plt.scatter(df['pickup_longitude'], df['pickup_latitude'], s=0.1, c=df['store_and_fwd_flag'])
plt.scatter(df['dropoff_longitude'],df['dropoff_latitude'], s=0.1, c=df['store_and_fwd_flag'])
plt.xlim([-74.2,-73.7])
plt.ylim([40.5,41])
plt.colorbar()
# no real pattern 

#explore basic temporal variablity
for x in df.iloc[:,-8:-1].columns:
    plt.plot(df.groupby(x).mean()['trip_duration'])
    plt.title('mean trip duration in '+ x)
    plt.show()
    plt.plot(df.groupby(x).count()['trip_duration'])
    plt.title('trip number in ' + x)
    plt.show()
# trip duration increses with increasing months
# #of trips increases until march and then starts decreasing 
# trip duration highest during middle of day/end of day rush hour (4-6pm)
# number of trips increases until 10pm or so 
# trip duration high during weekdays, decreases during weekends 
# number of trips highest during weekend
# 2 spikes early in the year, but otherwise trip duration rises over weeks 
    
#day and day of week temporal analysis 
for x in range(1,7):
    plt.plot(df[df['month'] ==x].groupby('day')
                .mean()['trip_duration'])
    plt.title('DAY Mean, Month ' + str(x))
    plt.show()
    plt.plot(df[df['month'] ==x].groupby('day')
                .count()['trip_duration'])
    plt.title('DAY Count, Month ' + str(x))
    plt.show()
    plt.plot(df[df['month'] ==x].groupby('day_of_week')
                .mean()['trip_duration'])
    plt.title('Day of Week Mean, Month ' + str(x))
    plt.show()
    plt.plot(df[df['month'] ==x].groupby('day_of_week')
                .count()['trip_duration'])
    plt.title('Day of Week Count, Month ' + str(x))
    plt.show()
#Jan: trip duration very high on 5th
    # no. of trips very low on 23rd 
#Feb: trip duration very high on 13th
    # monday/saturday seem like days with most rides 
#March, April, June: nothing too significant 
#May: 30th low ride count 
    
    
""" 
KEY TAKEAWAYS FROM EDA ROUNDS 1 & 2
1. A few trip_duration outliers (>100000s) need to be removed 
2. ~1000 lat/long outliers (away from city limits) need to be removed (outliers not linked to trip_duration outliers)
3. Significant increases in trip duration at ~ lat: 40.645/40.745 and long: -74.00/73.88/73.78
        each point should be encoded as distance from these locations after further investigation
4. Only vendor 1 had issues with recording at proper interval; beyond that no pattern of date or location, etc. 
5. Some irregularities in temporal trip duration/trip count data, should see if connection can be found 

"""
#%% TRAIN VS TEST DATA: EDA ROUND 3

#compare train vs test data 

#load data 
df_test=pd.read_csv('test.csv')
df_test_head=df_test.head()
df_test_stats=df_test.describe()

#check spatial consistency 
plt.figure(figsize=(20,20))
plt.scatter(df['pickup_longitude'], df['pickup_latitude'], c='blue', alpha=0.8)
plt.scatter(df['dropoff_longitude'],df['dropoff_latitude'], c='blue', alpha=0.8)
plt.scatter(df_test['pickup_longitude'], df_test['pickup_latitude'], c='red', alpha=0.5)
plt.scatter(df_test['dropoff_longitude'],df_test['dropoff_latitude'], c='red', alpha=0.5)
plt.title('Test vs Train Data')
# test data has same lat/long outliers, meaning that we will not remove lat/long outliers in any way

#initial feature engineering on test data 
df_test['pickup_datetime']=df_test['pickup_datetime'].apply(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))

df_test['year']=df_test['pickup_datetime'].apply(lambda x: x.year)
df_test['month']=df_test['pickup_datetime'].apply(lambda x: x.month)
df_test['day']=df_test['pickup_datetime'].apply(lambda x: x.day)
df_test['hour']=df_test['pickup_datetime'].apply(lambda x: x.hour)
df_test['minute']=df_test['pickup_datetime'].apply(lambda x: x.minute)
df_test['second']=df_test['pickup_datetime'].apply(lambda x: x.second)
df_test['day_of_week']=df_test['pickup_datetime'].apply(lambda x: x.isoweekday())
df_test['week_of_year']=df_test['pickup_datetime'].apply(lambda x: x.strftime('%W'))

#will keep datetime for further conversion, and will create new version for label encoding 
df_test['pickup_datetime_labelenc']=df_test['pickup_datetime']

#label encoding for test data
for_labels=[
        'vendor_id',
        'pickup_datetime_labelenc',
        'store_and_fwd_flag']

for x in for_labels:
    labelenc=LabelEncoder()
    df_test[x]=labelenc.fit_transform(df_test[x])
    

#comparison of meter failures 
plt.hist(df_test['pickup_datetime_labelenc'][df_test['store_and_fwd_flag']==1], bins=1000, color='blue', alpha=0.5)

plt.hist(df['pickup_datetime_labelenc'][df['store_and_fwd_flag']==1], bins=1000, color='red', alpha=0.6)
# there is a few on test dataset in which the number of failures goes up in a way not found in training set 
# however, based on prior investigation, there seems to be no impact trip_duration; will keep in mind but not act on it 
    
#are meter failers in specific part of city?  
plt.figure(figsize=(20,20))
plt.scatter(df_test['pickup_longitude'], df_test['pickup_latitude'], s=0.1, c=df_test['store_and_fwd_flag'])
plt.scatter(df_test['dropoff_longitude'],df_test['dropoff_latitude'], s=0.1, c=df_test['store_and_fwd_flag'])
plt.xlim([-74.2,-73.7])
plt.ylim([40.5,41])
plt.colorbar()
# still nope 

#temporal distribution analysis
plt.hist(df_test['pickup_datetime_labelenc'], bins=1000)
plt.hist(df['pickup_datetime_labelenc'], bins=1000)
# nothing outstanding 

"""
KEY TAKEAWAYS: 
    -test data has similar distribution as train dataset
    - FIGURE OUT HOW TO ENCODE TRIP DURATION LAT/LONG INCREASES??
    - will not remove lat/long outliers, as they exist in the test data as well 
    - may need to remove the 4 large trip duration outliers; will see how model does
    - may need to investgate further into the high number of meter failures in train dataset 
        (if model performs poorly and the cause is unknown; will need to be a consideration)
"""
#%% DATA MGMT

#create backups
df_back=df
df_test_back=df_test
#%% ADDITION OF SUPPLEMENTARY DATA AND EDA ROUND 4

#============EVENTS============ 
#public holidays and parades that may divert traffic 

#add data
df_events=pd.read_csv('NYC_events.csv')

#convert datetime in original dataset so it can be compared 
df['date']=df['pickup_datetime'].apply(lambda x: x.strftime('%Y-%m-%d'))
df_test['date']=df_test['pickup_datetime'].apply(lambda x: x.strftime('%Y-%m-%d'))

df=df.merge(df_events, on='date', how='left').fillna(0)
df_test=df_test.merge(df_events, on='date', how='left').fillna(0)

#explore if events have impact on trip duration/count
for x in range(1,7):
    plt.plot(df[df['month'] ==x].groupby('day').mean()['trip_duration'])
    plt.title('DAY Mean, Month ' + str(x))
    month=df[df['month'] ==x].groupby('day').mean()
    for z in range(1,(len(month)+1)):
        if df[(df['month'] ==x)&(df['day']==z)]['event'].iloc[0]!=0:
            plt.axvline(x=z, color='red')
    plt.show()

for x in range(1,7):
    plt.plot(df[df['month'] ==x].groupby('day').count()['trip_duration'])
    plt.title('DAY Count, Month ' + str(x))
    month=df[df['month'] ==x].groupby('day').mean()
    for z in range(1,(len(month)+1)):
        if df[(df['month'] ==x)&(df['day']==z)]['event'].iloc[0]!=0:
            plt.axvline(x=z, color='red')
    plt.show()
# these events fall on some of the local minimums for some months and are therefore worth keeping
# will one hot encode these  

#====================WEATHER DATA=======================

#load weather data
df_clim=pd.read_csv('weather_data_nyc_centralpark_2016.csv')

#format date column so we can merge 
df_clim['date']=df_clim['date'].apply(lambda x: (datetime.datetime.strptime(x, '%d-%m-%Y'))
                .strftime('%Y-%m-%d'))
    
#change 'T'(trace) accum into 0
df_clim=df_clim.replace({'T':0}) 
   
#change all numbers to float 
df_clim.iloc[:,1:]=df_clim.iloc[:,1:].astype(float)

#merge dates on isolated column created and date column 
df=df.merge(df_clim, on='date', how='left').fillna(0)
df_test=df_test.merge(df_clim, on='date', how='left').fillna(0)

#analyze weather in comparison to trip count/duration 
of_interest=['average temperature','precipitation','snow fall','snow depth']

for x in of_interest: 
    for y in range(1,7):
        fig, [ax1, ax3]=plt.subplots(2,1, sharex=True)
        fig.set_figheight(8)
        ax1.plot(df[df['month'] ==y].groupby('day')
                .count()['trip_duration'], c='red')
        ax1.set_ylabel('ride count', color='red')
        ax2 = ax1.twinx()
        ax2.plot(df[df['month'] ==y].groupby('day').mean()[x],c='blue')
        ax2.set_ylabel(x, color='blue')
        ax1.set_title(y)
        ax3.plot(df[df['month'] ==y].groupby('day')
                .mean()['trip_duration'], c='red')
        ax3.set_ylabel('ride duration', color='red')
        ax3.set_xlabel('day of month')
        ax4=ax3.twinx()
        ax4.plot(df[df['month'] ==y].groupby('day').mean()[x],c='blue')
        ax4.set_ylabel(x, color='blue')
        fig.subplots_adjust(wspace=0, hspace=0)
        fig.show()
#temperature shows no relationship for ride count/duration in january 
        # +ve correlation in feb-april
        #-'ve correlation in may/june

#precipitation; extreme event on 23rd of january accounts for dip in no of rides 
#and ride duration therafter for a few days 
    # ditto in feb 
    # more or less ditto in march 
    # starts to become less related in april 
    # may: extreme precip event on may 30th -'ve correlated to ride count and duration
    # correlation isn't as strong in june 
    #snowfall causes correlation in earlier months 
    #snow depth: -'ve correlation to ride count, +'ve correlation to ride duration 
        #but only above a certain depth(evident in one big snowfall event at end of jan)


#update head and stats
df_head=df.head()
df_desc=df.describe()
df_test_head=df_test.head()
df_test_desc=df_test.describe()


#===============FASTEST ROUTE DATA===============
#load data
df_route1=pd.read_csv('fastest_routes_train_part_1.csv')
df_route2=pd.read_csv('fastest_routes_train_part_2.csv')
df_route=pd.concat([df_route1, df_route2], axis=0,).reset_index().iloc[:,1:]
df_route_test=pd.read_csv('fastest_routes_test.csv')

df_route_head=df_route.head()
df_route_test_head=df_route_test.head()

#do eda where possible beofre feature engineering 
df_route['starting_street'].nunique()
df_route['end_street'].nunique()
# very high number of variables here 

valc_start=df_route['starting_street'].value_counts()
valc_end=df_route['end_street'].value_counts()
# a lot of the 'touristy' streets are frequent start/end streets, will freq encode 

#to do any further analysis, should prepare rest of features so they can be analyzed 

"""
KEY TAKEAWAYS FROM SUPPLEMENTARY DATA EDA:
1. events seem to correlate at least somewhat, need to one hot encode them 
2. engineer rest of info in fastest_route for analysis 
3. frequency encode street names (will do after rest of feature engineering to see if it can be done together)

"""

#%%FEATURE ENGINEERING ROUND 3

#feature engineering supplementary data 

#=========EVENTS AND ADDITIONAL DISTANCE CALCULATIONS==========
#one hot encode city events 
city_events=pd.get_dummies(df['event'].astype(str), drop_first=True)
city_events_test=pd.get_dummies(df_test['event'].astype(str), drop_first=True)

#find the geometric distances of travel
eclu=[]
vert=[]
horiz=[]
for x in range(len(df)):
    print(x)
    eclu.append(geodesic([df['pickup_latitude'][x], df['pickup_longitude'][x]], 
               [df['dropoff_latitude'][x], df['dropoff_longitude'][x]]).meters)
    vert.append(geodesic([df['pickup_latitude'][x], 0], 
               [df['dropoff_latitude'][x], 0]).meters)
    horiz.append(geodesic([0, df['pickup_longitude'][x]], 
               [0, df['dropoff_longitude'][x]]).meters)
#test data
eclu_test=[]
vert_test=[]
horiz_test=[]
for x in range(len(df_test)):
    print(x)
    eclu_test.append(geodesic([df_test['pickup_latitude'][x], df_test['pickup_longitude'][x]], 
               [df_test['dropoff_latitude'][x], df_test['dropoff_longitude'][x]]).meters)
    vert_test.append(geodesic([df_test['pickup_latitude'][x], 0], 
               [df_test['dropoff_latitude'][x], 0]).meters)
    horiz_test.append(geodesic([0, df_test['pickup_longitude'][x]], 
               [0, df_test['dropoff_longitude'][x]]).meters)

#%%
#========FASTEST ROUTE DATA============

#1. SPLIT STRINGS INTO DATASETS OF STEPS
#split all strings that need splitting in fastest route data 
#ensure all of the columns that are split are string format
df_route.iloc[:,-6:]=df_route.iloc[:,-6:].astype(str)
df_route_test.iloc[:,-6:]=df_route_test.iloc[:,-6:].astype(str)

#train data
step_street=[]
step_dist=[]
step_time=[]
step_man=[]
step_direct=[]
step_loc=[]

for x in range(len(df_route)):
    print(x)
    step_street.append(df_route['street_for_each_step'][x].split('|'))
    step_dist.append(df_route['distance_per_step'][x].split('|'))
    step_time.append(df_route['travel_time_per_step'][x].split('|'))
    step_man.append(df_route['step_maneuvers'][x].split('|'))
    step_direct.append(df_route['step_direction'][x].split('|'))
    step_loc.append(re.split(',|\|',df_route['step_location_list'][x]))

step_street_train=pd.DataFrame(step_street)
step_dist_train=pd.DataFrame(step_dist)
step_time_train=pd.DataFrame(step_time) 
step_man_train=pd.DataFrame(step_man)
step_direct_train=pd.DataFrame(step_direct)
step_loc_train=pd.DataFrame(step_loc)

#test data
step_street=[]
step_dist=[]
step_time=[]
step_man=[]
step_direct=[]
step_loc=[]

for x in range(len(df_route_test)):
    print(x)
    step_street.append(df_route_test['street_for_each_step'][x].split('|'))
    step_dist.append(df_route_test['distance_per_step'][x].split('|'))
    step_time.append(df_route_test['travel_time_per_step'][x].split('|'))
    step_man.append(df_route_test['step_maneuvers'][x].split('|'))
    step_direct.append(df_route_test['step_direction'][x].split('|'))
    step_loc.append(re.split(',|\|',df_route_test['step_location_list'][x]))

#will need to reduce number of columns to 47 to match train data, if individual columns get used
step_street_test=pd.DataFrame(step_street)
step_dist_test=pd.DataFrame(step_dist)
step_time_test=pd.DataFrame(step_time) 
step_man_test=pd.DataFrame(step_man)
step_direct_test=pd.DataFrame(step_direct)
step_loc_test=pd.DataFrame(step_loc)
# step time, step location, step distance are numerical and do not need further analysis
# step street will be frequency encoded
# step maneuver and step direciton need further analysis

#1 b) IDENTIFY TURNS/BUSY STREETS FOR FURTHER ANALYSIS
#turns will likely take more time;
#distinguish turns from other maneuvers, and their directions 
turns=[]
for x in range(len(df_route)):
    print(x)
    turn_n=[]
    for y in range(len(step_direct_train.iloc[x,:])):
        if step_man_train.iloc[x,y] == 'turn': 
            turn_n.append(step_direct_train.iloc[x,y])
    turns.append(turn_n)
turns_train=pd.DataFrame(turns)

turns=[]
for x in range(len(df_route_test)):
    print(x)
    turn_n=[]
    for y in range(len(step_direct_test.iloc[x,:])):
        if step_man_test.iloc[x,y] == 'turn': 
            turn_n.append(step_direct_test.iloc[x,y])
    turns.append(turn_n)
turns_test=pd.DataFrame(turns)

#street name; will find the most stopped at places (will likely have most traffic) and see how often 
#they come up in travel, as these streets will likely be busy and increas trip duration
#this should not change with dataset; instead will use the larger dataset to guage which streets are busiest
start_street=pd.Series(df_route['starting_street'].value_counts()[df_route['starting_street']
            .value_counts() > 5000].index)
end_street=pd.Series(df_route['end_street'].value_counts()[df_route['end_street']
            .value_counts() > 5000].index)
busy_streets=pd.concat([start_street,end_street]).unique()


#%%
#2. DETERMINE UNIQUE VALUES IN DATAFRAMES THAT NEED FURTHER ANALYSIS
#determine the unique values in step_man, turn, and step_direct to determine how to id them 
uni_direct=pd.Series()
uni_man=pd.Series() 
uni_turn=pd.Series()
for x in range(step_direct.shape[1]):
    print(x)
    uni_direct=uni_direct.append(pd.Series(step_direct.iloc[:,x].unique()), ignore_index=True)
    uni_man=uni_man.append(pd.Series(step_man.iloc[:,x].unique()), ignore_index=True)
for x in range(turns.shape[1]):
    uni_turn=uni_turn.append(pd.Series(turns.iloc[:,x].unique()), ignore_index=True)
print(uni_direct.unique())
print(uni_man.unique())
print(uni_turn.unique())

"""
unique_direct=['depart' 'rotary' 'turn' 'fork' 'arrive' 'new name' 'end of road'
 'continue' 'merge' 'roundabout turn' 'on ramp' 'off ramp' 'roundabout',nan]
unique_maneuv=['left' 'none' 'right' 'straight' 'slight right' 'arrive' 'slight left'
 'uturn' 'sharp left' 'sharp right', nan]
unique_turn=['right' 'left' None 'straight' 'slight right' 'slight left' 'sharp left'
 'sharp right' 'uturn']
"""
#3. DETERMINE FREQUENCY OF IMPORTANT STEPS

busy_street_freq=[]
turn_right=[]
turn_left=[]
turn_uturn=[]
direct_right=[]
direct_left=[]
straight=[]
direct_uturn=[]
end_road=[]
cont=[]
ramp=[]
fork=[]
turn=[]
merge=[]
round_abt=[]

for x in range(len(df_route)):
    print(x)
    busy_street_freq.append(sum(step_street_train.iloc[x].isin(busy_streets)))
    turn_right.append(sum((turns_train.iloc[x] == 'right') | (turns_train.iloc[x] =='slight right')
                            |(turns_train.iloc[x] =='sharp right')))
    turn_left.append(sum((turns_train.iloc[x] == 'left') | (turns_train.iloc[x] =='slight left')
                            |(turns_train.iloc[x] =='sharp left')))
    turn_uturn.append(sum(turns_train.iloc[x] =='uturn'))
    direct_right.append(sum((step_direct_train.iloc[x] == 'right') | (step_direct_train.iloc[x] =='slight right')
                            |(step_direct_train.iloc[x] =='sharp right')))
    direct_left.append(sum((step_direct_train.iloc[x] == 'left') | (step_man_train.iloc[x] =='slight left')
                            |(step_direct_train.iloc[x] =='sharp left')))
    direct_uturn.append(sum(step_direct_train.iloc[x] =='uturn'))
    end_road.append(sum(step_man_train.iloc[x] == 'end of road'))
    cont.append(sum(step_man_train.iloc[x] == 'continue'))
    ramp.append(sum((step_man_train.iloc[x] == 'on ramp') |(step_man_train.iloc[x] == 'off ramp')))
    fork.append(sum(step_man_train.iloc[x] == 'fork'))
    turn.append(sum(step_man_train.iloc[x] == 'turn'))
    merge.append(sum(step_man_train.iloc[x] == 'merge'))
    round_abt.append(sum((step_man_train.iloc[x] == 'roundabout') | (step_man_train.iloc[x] == 'roundabout turn')|
                        (step_man_train.iloc[x] == 'rotary')))

busy_street_freq_test=[]
turn_right_test=[]
turn_left_test=[]
turn_uturn_test=[]
direct_right_test=[]
direct_left_test=[]
straight_test=[]
direct_uturn_test=[]
end_road_test=[]
cont_test=[]
ramp_test=[]
fork_test=[]
turn_test=[]
merge_test=[]
round_abt_test=[]

for x in range(len(df_route_test)):
    print(x)
    busy_street_freq_test.append(sum(step_street_test.iloc[x].isin(busy_streets)))
    turn_right_test.append(sum((turns_test.iloc[x] == 'right') | (turns_test.iloc[x] =='slight right')
                            |(turns_test.iloc[x] =='sharp right')))
    turn_left_test.append(sum((turns_test.iloc[x] == 'left') | (turns_test.iloc[x] =='slight left')
                            |(turns_test.iloc[x] =='sharp left')))
    turn_uturn_test.append(sum(turns_test.iloc[x] =='uturn'))
    direct_right_test.append(sum((step_direct_test.iloc[x] == 'right') | (step_direct_test.iloc[x] =='slight right')
                            |(step_direct_test.iloc[x] =='sharp right')))
    direct_left_test.append(sum((step_direct_test.iloc[x] == 'left') | (step_man_test.iloc[x] =='slight left')
                            |(step_direct_test.iloc[x] =='sharp left')))
    direct_uturn_test.append(sum(step_direct_test.iloc[x] =='uturn'))
    end_road_test.append(sum(step_man_test.iloc[x] == 'end of road'))
    cont_test.append(sum(step_man_test.iloc[x] == 'continue'))
    ramp_test.append(sum((step_man_test.iloc[x] == 'on ramp') |(step_man_test.iloc[x] == 'off ramp')))
    fork_test.append(sum(step_man_test.iloc[x] == 'fork'))
    turn_test.append(sum(step_man_test.iloc[x] == 'turn'))
    merge_test.append(sum(step_man_test.iloc[x] == 'merge'))
    round_abt_test.append(sum((step_man_test.iloc[x] == 'roundabout') | (step_man_test.iloc[x] == 'roundabout turn')|
                        (step_man_test.iloc[x] == 'rotary')))


#frequency encode start and end streets 
    
# no point in frequency encoding because will not be consistant between datasets
# will only frequency encode by combining datasets and splitting in two, depending on
# how well model does 
for x in df_route[['starting_street','end_street']]: 
    for y in df_route[x].unique():
        print(x)
        df_route[x][df_route[x]== y]=df_route[x].value_counts()[y]
        
for x in df_route_test[['starting_street','end_street']]: 
    for y in df_route_test[x].unique():
        print(x)
        df_route_test[x][df_route_test[x]== y]=df_route_test[x].value_counts()[y]

#%%
#4. FINAL MODIFICATION AND DATASET ORGINALIZATION

#give column names to numerical datasets that are not undergoing further modification 
#& ones that will be used 
 
step_dist_test=step_dist_test.iloc[:,:46]
step_time_test=step_time_test.iloc[:,:46]
step_loc_test=step_loc_test.iloc[:,:92]
step_street_test=step_street_test.iloc[:,:46]
    
#naming of columns for datasets before merging 
street_dist=[]
street_time=[]
street_loc=[]
for x in range(int(step_dist_train.shape[1])):
    z=x+1
    street_dist.append('STEP_dist_'+ str(z))
    street_time.append('STEP_time_'+ str(z))
    street_loc.append('STEP_lat_'+ str(z))
    street_loc.append('STEP_long_'+str(z))
    
step_dist_train.columns=street_dist
step_time_train.columns=street_time
step_loc_train.columns=street_loc

step_dist_test.columns=street_dist
step_time_test.columns=street_time
step_loc_test.columns=street_loc

#combine route data with newly created columns before merging 
df_route=pd.concat([df_route, step_dist_train, step_time_train, step_loc_train, 
                          pd.Series(busy_street_freq).rename('busy_street_freq'),
                          pd.Series(turn_right).rename('turn_right'),
                          pd.Series(turn_left).rename('turn_left'),
                          pd.Series(turn_uturn).rename('turn_uturn'),
                          pd.Series(direct_right).rename('direct_right'),
                          pd.Series(direct_left).rename('direct_left'),
                          pd.Series(straight).rename('direct_straight'),
                          pd.Series(direct_uturn).rename('direct_uturn'),
                          pd.Series(end_road).rename('maneuv_end_road'),
                          pd.Series(cont).rename('maneuv_cont'),
                          pd.Series(ramp).rename('maneuv_ramp'),
                          pd.Series(fork).rename('maneuv_fork'),
                          pd.Series(turn).rename('maneuv_turn'),
                          pd.Series(merge).rename('maneuv_merge'),
                          pd.Series(round_abt).rename('maneuv_round_abt')], axis=1)
    
df_route_test=pd.concat([df_route_test, step_dist_test, step_time_test, step_loc_test, 
                          pd.Series(busy_street_freq_test).rename('busy_street_freq'),
                          pd.Series(turn_right_test).rename('turn_right'),
                          pd.Series(turn_left_test).rename('turn_left'),
                          pd.Series(turn_uturn_test).rename('turn_uturn'),
                          pd.Series(direct_right_test).rename('direct_right'),
                          pd.Series(direct_left_test).rename('direct_left'),
                          pd.Series(straight_test).rename('direct_straight'),
                          pd.Series(direct_uturn_test).rename('direct_uturn'),
                          pd.Series(end_road_test).rename('maneuv_end_road'),
                          pd.Series(cont_test).rename('maneuv_cont'),
                          pd.Series(ramp_test).rename('maneuv_ramp'),
                          pd.Series(fork_test).rename('maneuv_fork'),
                          pd.Series(turn_test).rename('maneuv_turn'),
                          pd.Series(merge_test).rename('maneuv_merge'),
                          pd.Series(round_abt_test).rename('maneuv_round_abt')], axis=1)

    
#merge semi-final dataset   
df=df.merge(df_route, on='id', how='left').fillna(0)    
df_test=df_test.merge(df_route_test, on='id', how='left').fillna(0)   
 
#remove data not found in route data 
missingno=df[df['id'].isin(df_route['id'])==False]
df_final=df_final[df_final['id'] != 'id3008062']

df_head=df.head()
df_test_head=df_test.head()
#%% get rid of trip duration outliers 
df_final=df_final[abs(df_final['trip_duration'] - np.mean(df_final['trip_duration']))
                                 < 2 * np.std(df_final['trip_duration'])]
#%% TESTING THE LOCAL TRIP_DURATION MAXIMUMS 

#test how both speed and expected vs actual trip duration relates to lat/long
#will remove the 4 largest outliers to do so and see if it makes a difference 
df_no_out=df_final

ave_speed=df_no_out['total_distance']/df_no_out['trip_duration']
trip_diff=df_no_out['trip_duration']-df_no_out['total_travel_time']
trip_dur_test=pd.DataFrame([round(df_no_out['pickup_longitude'], 3),round(df_no_out['pickup_latitude'],3), 
                            df_no_out['trip_duration'], ave_speed.rename('ave_speed'), trip_diff.rename('trip_diff')]).transpose()

#by longitude
trip_dur_mean=trip_dur_test.groupby('pickup_longitude').mean()
trip_dur_mean_lat=trip_dur_test.groupby('pickup_latitude').mean()
trip_dur_count=trip_dur_test.groupby('pickup_longitude').count()
trip_dur_count_lat=trip_dur_test.groupby('pickup_latitude').count()


#speed vs trip duration
fig, ax1 = plt.subplots()
ax1.plot(trip_dur_mean['ave_speed'], c='red')
ax1.set_ylabel('ave speed')
ax2 = ax1.twinx()
ax2.plot(trip_dur_mean['trip_duration'], c='blue')
ax2.set_ylabel('trip_duration')
ax1.set_xlim([-80,-70])
# no real correlation? a lot of variation around -74 long, but almost opposite; difficult to tell without further analysis


#trip duration vs trip duration difference
fig, ax1 = plt.subplots()
ax1.plot(trip_dur_mean['trip_diff'], c='red')
ax1.set_ylabel('trip_diff')
ax2 = ax1.twinx()
ax2.plot(trip_dur_mean['trip_duration'], c='blue')
ax2.set_ylabel('trip_duration')
ax1.set_xlim([-80,-70])
# these are more or less identical, meaning that there is a correlation between location and trips taking longer 
# will do peak analysis 


fig, ax1 = plt.subplots()
ax1.plot(trip_dur_mean_lat['trip_diff'], c='red')
ax1.set_ylabel('trip_diff')
ax2 = ax1.twinx()
ax2.plot(trip_dur_mean_lat['trip_duration'], c='blue', alpha=0.3)
ax2.set_ylabel('trip_duration')
# yup 

plt.plot(trip_dur_mean_lat['trip_duration'])
plt.plot(trip_dur_count_lat['trip_duration'])
plt.xlim([38,42])

plt.plot(trip_dur_mean['trip_duration'])
plt.plot(trip_dur_count['trip_duration'])
plt.xlim([-80,-70])

#count very low in some places where trip duration is high, meaning it relies on one or two trips
#will only count where trip count > 15

trip_long=trip_dur_mean['trip_duration'][trip_dur_count['trip_duration'] > 15]
trip_lat=trip_dur_mean_lat['trip_duration'][trip_dur_count_lat['trip_duration'] > 15]
#check shape 
plt.plot(trip_long)
plt.plot(trip_lat)
# still good 

#peak analysis 
from scipy.signal import find_peaks

peaks=find_peaks(trip_lat, height=1120, distance=10)[0]
lat_peaks=trip_lat.iloc[list(peaks)]

peaks=find_peaks(trip_long, height=1500, distance=20)[0]
long_peaks=trip_long.iloc[list(peaks)]

#4 locations in lat/long that have significant increases, will code these into train and test

#put these into datasets 
for x in lat_peaks.index:
    df[str(x)+'_pickup_lat']=abs(df['pickup_latitude']-x)
    df[str(x)+'_dropoff_lat']=abs(df['dropoff_latitude']-x)
    df_test[str(x)+'_pickup_lat']=abs(df['pickup_latitude']-x)
    df_test[str(x)+'_dropoff_lat']=abs(df['dropoff_latitude']-x)

for x in long_peaks.index:
    df[str(x)+'_pickup_long']=abs(df['pickup_longitude']-x)
    df[str(x)+'_dropoff_long']=abs(df['dropoff_longitude']-x)
    df_test[str(x)+'_pickup_long']=abs(df['pickup_longitude']-x)
    df_test[str(x)+'_dropoff_long']=abs(df['dropoff_longitude']-x)
#%%FINAL DATA COMPILATION

to_drop=['pickup_datetime','year','date','event','street_for_each_step','distance_per_step',
         'travel_time_per_step','step_maneuvers','step_direction','step_location_list', 'starting_street','end_street']
    

#combine all data 
df_final=pd.concat([df.drop(to_drop, axis=1), city_events,
              pd.Series(eclu).rename('euclidean_dist'),
              pd.Series(vert).rename('vertical_dist'),
              pd.Series(horiz).rename('horizontal_dist')], axis=1)

df_test_final=pd.concat([df_test.drop(to_drop, axis=1), city_events_test,
              pd.Series(eclu_test).rename('euclidean_dist'),
              pd.Series(vert_test).rename('vertical_dist'),
              pd.Series(horiz_test).rename('horizontal_dist')], axis=1)


df_final_head=df_final.head()
df_test_final_head=df_test_final.head()
df_final.to_csv('df_train_final.csv')
df_test_final.to_csv('df_test_final.csv')

#%% EDA 4 (Final)

# see if any correlation between the frequency categories engineered in the last section
of_interest=['number_of_steps', 'busy_street_freq','turn_right','turn_left','turn_uturn','direct_right',
             'direct_left','direct_straight','direct_uturn','maneuv_end_road','maneuv_cont','maneuv_ramp',
             'maneuv_fork','maneuv_turn','maneuv_merge','maneuv_round_abt','euclidean_dist', 'vertical_dist',
             'horizontal_dist']

for x in of_interest:
    plt.scatter(df_final[x], df_final['trip_duration'], s=0.1)
    plt.title(x)
    plt.show()
    
#while nothing easily quantifiable, will keep (until 2nd round of lbgm) as there are some interesting things going on 
#%% REMOVING EXTREME TRIP DURATION OUTLIERS
df_final=df_final.sort_values('trip_duration',ascending=False).iloc[4:,:]
#%% FORMATTING DATA FOR MODEL
    
#assign x and y data
x=df_final.drop(['id','trip_duration'],axis=1)
y=df_final['trip_duration']
x_for_test=df_test_final.drop(['id'], axis=1).astype(float)

#scale data (not necessary for lgbm, but I find it works better anyway)
scale=StandardScaler()
#exclude binary from scaling 
for_scaling=[]
for z in x:
    if (max(x[z]) !=1):
        for_scaling.append(z)
x[for_scaling]=scale.fit_transform(x[for_scaling])
x_for_test[for_scaling]=scale.fit_transform(x_for_test[for_scaling])

# train test split
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.2, random_state=0)

#format for model input
x_train=x_train.values.astype(float)
x_valid=x_valid.values.astype(float)
y_train=y_train.values
y_valid=y_valid.values
x_for_test=x_for_test.values.astype(float)

#%%

import lightgbm as lgb

param = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2_root', 'l1'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

lgb_train = lgb.Dataset(x_train, label=y_train)
lgb_valid = lgb.Dataset(x_valid, label=y_valid)

model = lgb.train(param, lgb_train, 50000, valid_sets=[lgb_train, lgb_valid], 
                  early_stopping_rounds=1000, verbose_eval=100)

#train_predictions_raw = model.predict(x_train, num_iteration=model.best_iteration)
validation_predictions=model.predict(x_valid, num_iteration=model.best_iteration)
test_predictions= model.predict(x_for_test, num_iteration=model.best_iteration)

#remove negative values
validation_predictions[validation_predictions < 0]= np.mean(validation_predictions[validation_predictions < 0])
test_predictions[test_predictions < 0]= np.mean(test_predictions[test_predictions < 0])

#check feature importance
feat_imp=pd.DataFrame(sorted(zip(model.feature_importance(), x.columns)), columns=['Value','Feature'])

plt.figure(figsize=(20, 40))
sns.barplot(x="Value", y="Feature", data=feat_imp.sort_values(by="Value", ascending=False))

plt.scatter(y_valid ,validation_predictions, s=0.1)
plt.xlabel('y_test')
plt.ylabel('y_pred')

#check shape of results
plt.hist(validation_predictions,bins=100)

plt.hist(test_predictions, bins=100)

from sklearn.metrics import mean_absolute_error
mae=mean_absolute_error(y_valid, validation_predictions)

submission=pd.concat([df_test_final['id'], pd.Series(test_predictions)],axis=1)
submission.columns=['id','trip_duration']

submission.to_csv('submission_all_feat.csv', index=False)

#%% REMOVE IRRELEVANT FEATURES AND REDO MODEL
feat_list=(feat_imp[feat_imp['Value'] < mae]['Feature'])

df_final_mod=df_final.drop(feat_list, axis=1)
df_test_final_mod=df_test_final.drop(feat_list, axis=1)

#assign x and y data
x=df_final_mod.drop(['id','trip_duration'],axis=1)
y=df_final_mod['trip_duration']
x_for_test=df_test_final_mod.drop(['id'], axis=1).astype(float)

#scale data (not necessary for lgbm, but I find it works better anyway)
scale=StandardScaler()
#exclude binary from scaling 
for_scaling=[]
for z in x:
    if (max(x[z]) !=1):
        for_scaling.append(z)
x[for_scaling]=scale.fit_transform(x[for_scaling])
x_for_test[for_scaling]=scale.fit_transform(x_for_test[for_scaling])

#format for model input
x=x.values.astype(float)
y=y.values
x_for_test=x_for_test.values.astype(float)
#%%MODEL ROUND 2
#LGBM WITH 5 KFOLD CV


param = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2_root', 'l1'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

from sklearn.model_selection import KFold

K = 5
kf = KFold(n_splits=K)
kf.get_n_splits(x)
 
lgb_pred=[]
cv_best_iter=[]
cv_best_score=[]
for train_index, test_index in kf.split(x):
    print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_valid = x[train_index], x[test_index]
    y_train, y_valid = y[train_index], y[test_index]
    

    lgb_train = lgb.Dataset(x_train, label=y_train)
    lgb_valid = lgb.Dataset(x_valid, label=y_valid)

    model = lgb.train(param, lgb_train, 50000, valid_sets=[lgb_train, lgb_valid], 
                  early_stopping_rounds=500, verbose_eval=100)

    test_predictions= model.predict(x_for_test, num_iteration=model.best_iteration)
    test_predictions[test_predictions < 0]= np.mean(test_predictions[test_predictions < 0])
    lgb_pred.append(list(test_predictions))
    cv_best_iter.append(model.best_iteration)
    cv_best_score.append(model.best_score['valid_1']['l1'])

preds=[]
for i in range(len(lgb_pred[0])):
    sum=0
    for j in range(K):
        sum+=lgb_pred[j][i]
    preds.append(sum / K)

std=[]
np_lgb_pred=np.array(lgb_pred)
for i in range(len(lgb_pred[0])):
    std.append(np.std(np_lgb_pred[:,i]))

#check feature importance
feat_imp=pd.DataFrame(sorted(zip(model.feature_importance(), x.columns)), columns=['Value','Feature'])

plt.figure(figsize=(20, 40))
sns.barplot(x="Value", y="Feature", data=feat_imp.sort_values(by="Value", ascending=False))

plt.scatter(y_valid ,validation_predictions, s=0.1)
plt.xlabel('y_test')
plt.ylabel('y_pred')

from sklearn.metrics import mean_absolute_error
mae=mean_absolute_error(y_valid, validation_predictions)

submission_mod=pd.concat([df_test_final['id'], pd.Series(preds)],axis=1)
submission_mod.columns=['id','trip_duration']

submission_mod.to_csv('submission_lgbm_kfold.csv', index=False)



#%% CHECK SCORE WITH XGBOOST
x_train=x_train.values.astype(float)
x_valid=x_valid.values.astype(float)
y_train=y_train.values
y_valid=y_valid.values
x_for_test=x_for_test.values.astype(float)


from xgboost import XGBRegressor

xgb = XGBRegressor(n_estimators=10000, max_depth=12, min_child_weight=150, 
                   subsample=0.7, colsample_bytree=0.3, silent=False)

xgb.fit(x_train, y_train, eval_set=[(x_valid, y_valid)],
            early_stopping_rounds=500, verbose=25)

test_predictions= xgb.predict(x_for_test, ntree_limit=xgb.best_ntree_limit)
validation_predictions= xgb.predict(x_valid, ntree_limit=xgb.best_ntree_limit)

plt.scatter(y_valid ,validation_predictions, s=0.1)
plt.xlabel('y_test')
plt.ylabel('y_pred')

#check shape of results
plt.hist(validation_predictions,bins=100)
plt.hist(test_predictions, bins=100)

feat_imp=pd.DataFrame(sorted(zip(xgb.feature_importances_, df_final.drop('trip_duration', axis=1).columns)), columns=['Value','Feature'])

submission=pd.concat([df_test_final['id'], pd.Series(test_predictions)],axis=1)
submission.columns=['id','trip_duration']

submission.to_csv('submission_xgboost.csv', index=False)
#%%DATA BACKUP

#data backup    
df_route.to_csv('df_route_train.csv')
df_route_test.to_csv('df_route_test.csv')

df.to_csv('df_train_intermed.csv')
df=pd.read_csv('df_train_intermed.csv').iloc[:,1:]

df_test.to_csv('df_test_intermed.csv')
df_test=pd.read_csv('df_test_intermed.csv').iloc[:,1:]


df_final=pd.read_csv('df_train_final.csv').iloc[:,1:]
df_test_final=pd.read_csv('df_test_final.csv').iloc[:,1:]
#do initial feature engineering; put everything into proper formats 
# do train vs test comparison (spatial and temporal)
# remove outliers if appropriate 
#add additional datasets to both train and test (in a more fast running format; list?? )
    
# for duration maximums; round lat/long to nearest 3rd decimal? and group by/mean with std
    #then do peak analysis
# do final eda and see if anything needs to be removed or changed? 
# do initial ML 
#%%% SCRATCH


