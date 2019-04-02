
"""
ANIMAL SHELTER OUTCOMES 
evaluation metric: logloss
"""
import numpy as np
import pandas as pd 
import datetime
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import re
#%% IMPORT DATA /FEATURE ENGINEERING (CANNOT REALLY VISUALIZE DATA AT THIS POINT)
df=pd.read_csv('train.csv')

#NAME
#change names to binary: if there is name=0, no name=1 
df['Name'][df['Name'].isnull()==False]=0
df['Name'][df['Name'].isnull()]=1
#change all data to intiger 
df['Name']=df['Name'].astype(int)

#DATETIME
#convert date string into datetime and seperate
df['DateTime']=df['DateTime'].apply(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))
#expand out date until hour 
df['year']=df['DateTime'].apply(lambda x: x.year)
df['month']=df['DateTime'].apply(lambda x: x.month)
df['day']=df['DateTime'].apply(lambda x: x.day)
df['hour']=df['DateTime'].apply(lambda x: x.hour)
df['day_of_week']=df['DateTime'].apply(lambda x: x.isoweekday())
df['week_of_year']=df['DateTime'].apply(lambda x: x.strftime('%W')).astype(int)

#ANIMAL TYPE
#determine number of unique animals in this dataset
df['AnimalType'].nunique()
# 2; cats and dogs 

#label encode 
labelenc=LabelEncoder()
df['AnimalType']=labelenc.fit_transform(df['AnimalType'])
# cat=0, dog=1

#SEX UPON OUTCOME CATEGORY
#determine what the unique categories are 
df['SexuponOutcome'].value_counts()
# will not remove unknowns, because there is over 1K, and dataset isn't that large

#will convert to string and split 
#for convenience sake, will double unknown, so won't have to modify later
df['SexuponOutcome']=df['SexuponOutcome'].astype(str).replace({'Unknown': 'Unknown Unknown','nan':'Unknown Unknown'})
gender=df['SexuponOutcome'].apply(lambda x: pd.Series(x.split(' ')))
gender.columns=['Status','Sex']
#replace spayed and neutered into 'fixed', since male and female are stated 
#check unique values first, just in case
gender['Status'].value_counts()
# good to go
gender['Status']=gender['Status'].replace({'Spayed':'Fixed','Neutered':'Fixed'})

#one hot encode these values 
gender_dum=pd.get_dummies(gender,drop_first=True)


#AGE UPON OUTCOME
#split and ensure that everything is strings 
ages=df['AgeuponOutcome'].astype(str).apply(lambda x: pd.Series(x.split(' ')))
ages.columns=['Num','Int']

#make sure that no missing data in split 
ages['Num'].value_counts()
ages['Int'].value_counts()
#18 nan values; will encode as -1 so it cannot be mistaken 
#there are plurals, etc. 
#first make sure it is proparly formatted: 
ages['Num']=ages['Num'].astype(float)
ages['Int']=ages['Int'].astype(str)
age_nums=[]
for x in range(len(ages)):
    if ages['Int'][x]=='year' or ages['Int'][x]=='years':
        age_nums.append(ages['Num'][x])
    elif ages['Int'][x]=='months' or ages['Int'][x]=='month':
        age_nums.append(ages['Num'][x]/12)
    elif ages['Int'][x]=='weeks' or ages['Int'][x]=='week':
        age_nums.append(ages['Num'][x]/52)
    elif ages['Int'][x]=='days' or ages['Int'][x]=='day':
        age_nums.append(ages['Num'][x]/365)
    elif np.isnan(ages['Num'][x]) == True:
        age_nums.append(-1)

age_nums=pd.Series(age_nums).rename('age_in_yrs')

#BREED

#will see the number of unique types
breeds=df['Breed'].value_counts()
df['Breed'].nunique()
# quite a lot, can't just one hot encode them 

#confirm everything is string and lower case everything for consistancy

df['Breed']=df['Breed'].astype(str).apply(lambda x: x.lower())
#exception to split; this is a color rather than description
df['Breed']=df['Breed'].apply(lambda x: x.replace('black/tan','black-tan'))

#check if 'mix' or '/' is their breed, will also apply shorthair medium hair and longhair; apply to boht cats and dogs 
breed_mix=df['Breed'].apply(lambda x: 'mix' in x)
breed_short=df['Breed'].apply(lambda x: 'shorthair' in x)
breed_med=df['Breed'].apply(lambda x: 'medium hair' in x)
breed_long=df['Breed'].apply(lambda x: 'longhair' in x)

#now will remove 'mix'/hair length
breed=df['Breed'].apply(lambda x: x.replace('mix',''))
breed=breed.apply(lambda x: x.replace('shorthair', ''))
breed=breed.apply(lambda x: x.replace('medium hair', ''))
breed=breed.apply(lambda x: x.replace('longhair', ''))
breed=breed.apply(lambda x: x.replace('  ', ' '))
breed=breed.apply(lambda x: x.strip())
breed_split=breed.apply(lambda x: pd.Series(x.split('/')))

#will recound number of unique breeds
breeds=breed_split[0].value_counts()
breeds_2=breed_split[1].value_counts()
# remove anything that has less than 10 values; may have to play around with these
breeds=breeds[breeds >= 10]
breeds_2=breeds_2[breeds_2 >= 10]

breed3=pd.Series(pd.concat([pd.Series(breeds.index),pd.Series(breeds_2.index)], axis=0).unique())
breed3=pd.Series(breed3.apply(lambda x: x.strip()).unique())
breed3=breed3.apply(lambda x: x.replace('  ', ' '))

breed_legend=breed3

#convert each column's breed into these categories, and if not there, 'other' then one hot encode them 
breed_final=pd.DataFrame(np.empty((len(df),len(breed3))),columns=breed3)


for x in breed_legend:
    print(x)
    for y in range(len(df)):
        if all(z in df['Breed'][y] for z in x): 
            breed_final[x][y]=1
        else:
            breed_final[x][y]=0


##too computationally expensive
#for x in breed3:
#    print(x)
#    x=list(x.split())
#    for y in range(len(df)):
#        print(y)
#        if all(z in df['Breed'][y] for z in x): 
#            breed_final[x][y]=1
#        else:
#            breed_final[x][y]=0

#COLOR
#see number of unique colors 
colors=df['Color'].value_counts()
# won't split up as is, instead will do similar process as for breeds
#format properly
df['Color']=df['Color'].astype(str).apply(lambda x: x.lower())

#split up by / and by spaces 
colors=df['Color'].apply(lambda x: pd.Series(re.split('/|\s',x)))

colors=pd.concat([colors[0], colors[1], colors[2], colors[3]], axis=0).reset_index(drop=True)
#get unique values
color_test=colors.value_counts()

#will elimainate rare colors that do not have more than 10 values 
color_test=color_test[color_test >=10]
#get index values for sorting 

color_idx=pd.Series(pd.Series(color_test.index).apply(lambda x: x.strip()).unique())

color_legend=color_idx

color_final=pd.DataFrame(np.empty((len(df), len(color_idx))), columns=color_idx)

for x in color_legend:
    print(x)
    for y in range(len(df)):
        if x in df['Color'][y]:
            color_final[x][y]=1
        else:
            color_final[x][y]=0
            
#label encode booleans and results          
labelenc=LabelEncoder()
breed_mix=pd.Series(labelenc.fit_transform(breed_mix))
breed_short=pd.Series(labelenc.fit_transform(breed_short))
breed_med=pd.Series(labelenc.fit_transform(breed_med))
breed_long=pd.Series(labelenc.fit_transform(breed_long))
df['OutcomeType']=pd.Series(labelenc.fit_transform(df['OutcomeType']))

#0=Adoption, 1=Died, 2=Euthanaisa 3=Return to owner, 4=Transfer
#determine which columns to drop: 
to_drop=['DateTime','OutcomeSubtype','SexuponOutcome','AgeuponOutcome','Breed','Color']


df_final=pd.concat((df.drop(to_drop,axis=1),breed_final,breed_mix.rename('mix'),breed_short.rename('shorthair'),
                   breed_med.rename('medium_hair'),breed_long.rename('longhair'),color_final,
                   gender_dum,age_nums),axis=1)


#%%TEST DATA FEATURE ENGINEERING 
df_test=pd.read_csv('test.csv')

#change names to binary: if there is name=0, no name=1 
df_test['Name'][df_test['Name'].isnull()==False]=0
df_test['Name'][df_test['Name'].isnull()]=1
#change all data to intiger 
df_test['Name']=df_test['Name'].astype(int)

#DATETIME
#convert date string into datetime and seperate
df_test['DateTime']=df_test['DateTime'].apply(lambda x: datetime.datetime.strptime(x,"%Y-%m-%d %H:%M:%S"))
#expand out date until hour 
df_test['year']=df_test['DateTime'].apply(lambda x: x.year)
df_test['month']=df_test['DateTime'].apply(lambda x: x.month)
df_test['day']=df_test['DateTime'].apply(lambda x: x.day)
df_test['hour']=df_test['DateTime'].apply(lambda x: x.hour)
df_test['day_of_week']=df_test['DateTime'].apply(lambda x: x.isoweekday())
df_test['week_of_year']=df_test['DateTime'].apply(lambda x: x.strftime('%W')).astype(int)

#ANIMAL TYPE
#determine number of unique animals in this dataset


#label encode 
labelenc=LabelEncoder()
df_test['AnimalType']=labelenc.fit_transform(df_test['AnimalType'])
# cat=0, dog=1

#SEX UPON OUTCOME CATEGORY

#will convert to string and split 
#for convenience sake, will double unknown, so won't have to modify later
df_test['SexuponOutcome']=df_test['SexuponOutcome'].astype(str).replace({'Unknown': 'Unknown Unknown','nan':'Unknown Unknown'})
gender=df_test['SexuponOutcome'].apply(lambda x: pd.Series(x.split(' ')))
gender.columns=['Status','Sex']
#replace spayed and neutered into 'fixed', since male and female are stated 
gender['Status']=gender['Status'].replace({'Spayed':'Fixed','Neutered':'Fixed'})

#one hot encode these values 
gender_dum=pd.get_dummies(gender,drop_first=True)


#AGE UPON OUTCOME
#split and ensure that everything is strings 
ages=df_test['AgeuponOutcome'].astype(str).apply(lambda x: pd.Series(x.split(' ')))
ages.columns=['Num','Int']

#first make sure it is proparly formatted: 
ages['Num']=ages['Num'].astype(float)
ages['Int']=ages['Int'].astype(str)
age_nums=[]
for x in range(len(ages)):
    if ages['Int'][x]=='year' or ages['Int'][x]=='years':
        age_nums.append(ages['Num'][x])
    elif ages['Int'][x]=='months' or ages['Int'][x]=='month':
        age_nums.append(ages['Num'][x]/12)
    elif ages['Int'][x]=='weeks' or ages['Int'][x]=='week':
        age_nums.append(ages['Num'][x]/52)
    elif ages['Int'][x]=='days' or ages['Int'][x]=='day':
        age_nums.append(ages['Num'][x]/365)
    elif np.isnan(ages['Num'][x]) == True:
        age_nums.append(-1)

age_nums=pd.Series(age_nums).rename('age_in_yrs')

#BREED

#will see the number of unique types
breeds=df_test['Breed'].value_counts()

#confirm everything is string and lower case everything for consistancy

df_test['Breed']=df_test['Breed'].astype(str).apply(lambda x: x.lower())
#exception to split; this is a color rather than description
df_test['Breed']=df_test['Breed'].apply(lambda x: x.replace('black/tan','black-tan'))

#check if 'mix' or '/' is their breed, will also apply shorthair medium hair and longhair; apply to boht cats and dogs 
breed_mix=df_test['Breed'].apply(lambda x: 'mix' in x)
breed_short=df_test['Breed'].apply(lambda x: 'shorthair' in x)
breed_med=df_test['Breed'].apply(lambda x: 'medium hair' in x)
breed_long=df_test['Breed'].apply(lambda x: 'longhair' in x)

#now will remove 'mix'/hair length
breed=df_test['Breed'].apply(lambda x: x.replace('mix',''))
breed=breed.apply(lambda x: x.replace('shorthair', ''))
breed=breed.apply(lambda x: x.replace('medium hair', ''))
breed=breed.apply(lambda x: x.replace('longhair', ''))
breed=breed.apply(lambda x: x.replace('  ', ' '))
breed=breed.apply(lambda x: x.strip())
breed_split=breed.apply(lambda x: pd.Series(x.split('/')))

#will recound number of unique breeds
breeds=breed_split[0].value_counts()
breeds_2=breed_split[1].value_counts()
# remove anything that has less than 10 values; may have to play around with these
breeds=breeds[breeds >= 10]
breeds_2=breeds_2[breeds_2 >= 10]

breed3=pd.Series(pd.concat([pd.Series(breeds.index),pd.Series(breeds_2.index)], axis=0).unique())
breed3=pd.Series(breed3.apply(lambda x: x.strip()).unique())
breed3=breed3.apply(lambda x: x.replace('  ', ' '))

#convert each column's breed into these categories, and if not there, 'other' then one hot encode them 
breed_final=pd.DataFrame(np.empty((len(df_test),len(breed_legend))),columns=breed_legend)


for x in breed_legend:
    print(x)
    for y in range(len(df_test)):
        if all(z in df_test['Breed'][y] for z in x): 
            breed_final[x][y]=1
        else:
            breed_final[x][y]=0


#COLOR
#see number of unique colors 

#format properly
df_test['Color']=df_test['Color'].astype(str).apply(lambda x: x.lower())

#split up by / and by spaces 
colors=df_test['Color'].apply(lambda x: pd.Series(re.split('/|\s',x)))

colors=pd.concat([colors[0], colors[1], colors[2], colors[3]], axis=0).reset_index(drop=True)
#get unique values
color_test=colors.value_counts()

#will elimainate rare colors that do not have more than 10 values 
color_test=color_test[color_test >=10]
#get index values for sorting 

color_idx=pd.Series(pd.Series(color_test.index).apply(lambda x: x.strip()).unique())

color_final=pd.DataFrame(np.empty((len(df_test), len(color_legend))), columns=color_legend)

for x in color_legend:
    print(x)
    for y in range(len(df_test)):
        if x in df_test['Color'][y]:
            color_final[x][y]=1
        else:
            color_final[x][y]=0
            
#label encode booleans and results          
labelenc=LabelEncoder()
breed_mix=pd.Series(labelenc.fit_transform(breed_mix))
breed_short=pd.Series(labelenc.fit_transform(breed_short))
breed_med=pd.Series(labelenc.fit_transform(breed_med))
breed_long=pd.Series(labelenc.fit_transform(breed_long))
 
to_drop=['DateTime','SexuponOutcome','AgeuponOutcome','Breed','Color']


df_test_final=pd.concat((df_test.drop(to_drop,axis=1),breed_final,breed_mix.rename('mix'),breed_short.rename('shorthair'),
                   breed_med.rename('medium_hair'),breed_long.rename('longhair'),color_final,
                   gender_dum,age_nums),axis=1)

#%% SPLIT DATA FOR TEST/TRAIN

x=df_final.drop(['AnimalID','OutcomeType'],axis=1).astype(float)
y=df_final['OutcomeType'].astype(float)
x_test=df_test_final.drop(['ID'], axis=1).astype(float)

#scale x data that isn't binary 
for_scaling=[]
scale=StandardScaler()
for z in x:
    if x[z].nunique() !=2:
        print(z)
        for_scaling.append(z)
x[for_scaling]=scale.fit_transform(x[for_scaling])
x_test[for_scaling]=scale.fit_transform(x_test[for_scaling])

x=x.values
y=y.values
x_test=x_test.values
#%%FIRST XGBOOST

from xgboost import XGBClassifier

xgb=XGBClassifier(
                              n_estimators=1000,
                              objective='multi:softmax',
                              eval_metric= 'logloss',
                              num_class=5,
                              verbosity=-1)

from sklearn.model_selection import StratifiedKFold
K = 2
kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=0)
kf.get_n_splits(x)

xgb_valid=[]
xgb_pred=[]
for train_index, test_index in kf.split(x, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_valid = x[train_index], x[test_index]
    y_train, y_valid = y[train_index], y[test_index]

    xgb.fit(x_train, y_train,verbose=True)

    validation_predictions= pd.DataFrame(xgb.predict_proba(x_valid))
    test_predictions=pd.DataFrame(xgb.predict_proba(x_test))
    xgb_valid.append(validation_predictions)
    xgb_pred.append(test_predictions)

pred_final=[]
for z in range(len(test_predictions)):
     pred_final.append(np.mean([xgb_pred[0].iloc[z], xgb_pred[1].iloc[z]],axis=0))


valid_final=[]
for z in range(len(validation_predictions)):
     valid_final.append(np.mean([xgb_valid[0].iloc[z], xgb_valid[1].iloc[z]],axis=0))
     
from sklearn.metrics import log_loss

log_loss(y_valid, valid_final)

#%% ANN 
import keras
from keras.constraints import maxnorm
from keras.models import Sequential
from keras.layers import Dense

x=df_final.drop(['AnimalID','OutcomeType'],axis=1).astype(float)
y=df_final['OutcomeType'].astype(float)
x_test=df_test_final.drop(['ID'], axis=1).astype(float)

#scale x data that isn't binary 
for_scaling=[]
scale=StandardScaler()
for z in x:
    if x[z].nunique() !=2:
        print(z)
        for_scaling.append(z)
x[for_scaling]=scale.fit_transform(x[for_scaling])
x_test[for_scaling]=scale.fit_transform(x_test[for_scaling])

y=pd.get_dummies(y)

x=x.values
y=y.values
x_test=x_test.values

x_train,x_valid,y_train,y_valid=train_test_split(x, y, test_size = 0.2, random_state=0)


classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 100, activation = 'relu', input_dim =165))

classifier.add(Dense(5, activation="sigmoid"))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 100, activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 5, activation = 'softmax'))

# Compiling the ANN
classifier.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy'])


# Fitting the ANN to the Training set
classifier.fit(x_train,y_train, batch_size = 2000, verbose=1,nb_epoch = 1000, validation_split=0.2)

validation_predictions= pd.DataFrame(classifier.predict(x_valid))
test_predictions=pd.DataFrame(classifier.predict(x_test))

#valid_final=[]
#for z in range(len(validation_predictions)):
#    valid_final.append(validation_predictions.iloc[z].values.argmax())
#    
#valid_final=pd.get_dummies(valid_final)

log_loss(y_valid, validation_predictions)
#%%RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, 
                                    criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)


y_val = classifier.predict_proba(x_valid)

log_loss(y_valid, y_val)
#%%KNN

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(x_train, y_train)
y_val = classifier.predict_proba(x_valid)

log_loss(y_valid, y_val)
#%%backup
df2=df

df=df2
#%%

"""
Next Steps: 
    1. Expand on feat eng of names
    2. frequency encode breeds 
    3. conduct PCA on breeds 

    
"""