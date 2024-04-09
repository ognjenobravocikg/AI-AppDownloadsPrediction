#!/usr/local/bin/python3.12

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.externals import joblib


def convert_to_mb(size):
    size_str = str(size)

    if (size_str[-1] == 'k'):
        return float(size_str[:-1].replace(',', '.')) / 1024

    if (size_str[-1] == 'M'):
        return float(size_str[:-1].replace(',', '.'))

    if (size_str[-1] == 'G'):
        return float(size_str[:-1]) * 1024.0

    else:
        return None
        

def iseci(x):
    return str(x)[:3]


def days_since_epoch(date_string):
    date_object = datetime.strptime(date_string, "%d.%m.%Y")
    epoch = datetime(1970, 1, 1)
    delta = date_object - epoch
    return delta.days


def nadji_verziju(x):
    searched = days_since_epoch(x)
    for index in range(android_versions_release_date['since_111970'].count()):
        if searched <= android_versions_release_date['since_111970'][index]:
            if index == 0:
                return 'to_be_droped'
            return android_versions_release_date['Release date'][index-1]
        

data = pd.read_csv('./Google-Playstore.csv')

data.drop('App Id', axis=1, inplace=True)
data.drop('Scraped Time', axis=1, inplace=True)
data.drop('Developer Website', axis=1, inplace=True)
data.drop('Developer Email', axis=1, inplace=True)
data.drop('Privacy Policy', axis=1, inplace=True)
data.drop('Currency', axis=1, inplace=True)
data.drop('Installs', axis=1, inplace=True)
data.drop('Minimum Installs', axis=1, inplace=True)

data['Free'].replace({True: 1, False: 0}, inplace=True)
data['Ad Supported'].replace({True: 1, False: 0}, inplace=True)
data['In App Purchases'].replace({True: 1, False: 0}, inplace=True)
data['Editors Choice'].replace({True: 1, False: 0}, inplace=True)

list_of_categories = data['Category'].unique()
categories = pd.get_dummies(data['Category'])
categories.replace({True: 1, False: 0}, inplace=True)
data = pd.concat([data, categories], axis=1)
data.drop('Category', axis=1, inplace=True)

_ = pd.get_dummies(data['Content Rating'])
_.replace({True: 1, False: 0}, inplace=True)
data = pd.concat([data, _], axis=1)
data.drop('Content Rating', axis=1, inplace=True)

list_of_developers = pd.DataFrame()
list_of_developers['Developer'] = data['Developer Id'].unique()
list_of_developers['Id'] = range(1, len(list_of_developers) + 1)

developers_dict = dict(zip(list_of_developers['Developer'], list_of_developers['Id']))

data['Developer Id'] = data['Developer Id'].map(developers_dict)

data['Size'] = data['Size'].apply(lambda x: convert_to_mb(x))

data['Released'] = pd.to_datetime(data['Released'])
data['Released'] = data['Released'].dt.strftime('%d.%m.%Y') 
data['Last Updated'] = pd.to_datetime(data['Last Updated'])
data['Last Updated'] = data['Last Updated'].dt.strftime('%d.%m.%Y') 

android_versions_dict = pd.read_csv('./Android-Versions.csv')
android_versions_dict['Android version'] = android_versions_dict['Android version'].apply(lambda x: str(x))

android_versions_release_date = pd.DataFrame(android_versions_dict['Release date'].copy(deep=True))

android_versions_dict = dict(zip(android_versions_dict['Android version'], android_versions_dict['Release date']))

data['Minimum Android'] = data['Minimum Android'].apply(lambda x: iseci(x)) 

android_versions_dict.update({'nan': 0})
android_versions_dict.update({'Var': 'Varies with device'})

data['Minimum Android'] = data['Minimum Android'].map(android_versions_dict)

data['Rating'].fillna(0.0, inplace=True)
data['Rating Count'].fillna(0.0, inplace=True)

for category in list_of_categories:

    average_size = data.loc[data[category] == 1]['Size'].mean()

    _ = pd.DataFrame(data.loc[data['Size'].isna()][category])
    indexes = _.index[_[category] == 1].to_list()
    data.loc[indexes, 'Size'] = average_size

to_be_removed = data.loc[data['Minimum Android'] == 'Varies with device'].index.to_list()
data.drop(to_be_removed, inplace=True)

to_be_fixed = data.loc[data['Released'].isna()].index.to_list()
data.loc[to_be_fixed, 'Released'] = data.loc[to_be_fixed]['Minimum Android']

to_be_removed = data.loc[data['Released'].isna()].index.to_list()
data.drop(to_be_removed, inplace=True)
to_be_removed = data.loc[data['Released'] == 0].index.to_list()
data.drop(to_be_removed, inplace=True)

android_versions_release_date['since_111970'] = android_versions_release_date['Release date'].apply(lambda x: days_since_epoch(x))
        
to_be_fixed = data.loc[data['Minimum Android'].isna()].index.to_list()
data.loc[to_be_fixed, 'Minimum Android'] = data.loc[to_be_fixed, 'Released'].apply(lambda x: nadji_verziju(x))

to_be_removed = data.loc[data['Minimum Android'] == 0].index.to_list()
data.drop(to_be_removed, inplace=True)

data['Minimum Android'] = data['Minimum Android'].apply(lambda x: days_since_epoch(x))
data['Released'] = data['Released'].apply(lambda x: days_since_epoch(x))
data['Last Updated'] = data['Last Updated'].apply(lambda x: days_since_epoch(x))

to_be_removed = data.loc[data['App Name'].isna()].index.to_list()
data.drop(to_be_removed, inplace=True)

vectorizer = TfidfVectorizer(ngram_range=(1,4))
X = vectorizer.fit_transform(data['App Name'])

kmeans = KMeans(n_clusters=30)
kmeans.fit(X)

app_name_categories = kmeans.predict(X) 
data['App Name'] = app_name_categories 

joblib.dump(kmeans, 'kategorizacija_imena.joblib')

app_names = pd.get_dummies(data['App Name']) 
app_names.replace({True: 1, False: 0}, inplace=True) 
data = pd.concat([data, app_names], axis=1)
data.drop('App Name', axis=1, inplace=True)

data.to_csv('Finalni_Podaci.csv')

y = data['Maximum Installs'].copy(deep=True)
X = data.copy(deep=True)
X.drop('Maximum Installs', axis=1)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, 'main_model.joblib')

