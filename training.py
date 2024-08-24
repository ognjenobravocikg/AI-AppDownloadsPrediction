import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import joblib
import warnings

# Some annoying legacy functions warrnings pop up
warnings.filterwarnings('ignore')

# Convert file size to MB
def convert_to_mb(size):
    size_str = str(size)
    if size_str[-1] == 'k':
        return float(size_str[:-1].replace(',', '.')) / 1024
    if size_str[-1] == 'M':
        return float(size_str[:-1].replace(',', '.'))
    if size_str[-1] == 'G':
        return float(size_str[:-1]) * 1024.0
    else:
        return None

# Extract first 3 characters
def iseci(x):
    return str(x)[:3]

# Convert date to days since Unix epoch
def days_since_epoch(date_string):
    if isinstance(date_string, str):
        date_object = datetime.strptime(date_string, "%d.%m.%Y")
        epoch = datetime(1970, 1, 1)
        delta = date_object - epoch
        return delta.days
    return 0  # Default to 0 or some sentinel value if the conversion fails

# Find version by date
def nadji_verziju(x):
    searched = days_since_epoch(x)
    for index in range(android_versions_release_date['since_111970'].count()):
        if searched <= android_versions_release_date['since_111970'][index]:
            if index == 0:
                return 'to_be_dropped'
            return android_versions_release_date['Release date'][index-1]

data = pd.read_csv('datasets/Google-Playstore.csv')
data = data.sample(n=200000, random_state=42)
data.drop(['App Id', 'Scraped Time', 'Developer Website', 'Developer Email',
           'Privacy Policy', 'Currency', 'Installs', 'Minimum Installs'], axis=1, inplace=True)

# Convert boolean columns to integers
data['Free'] = data['Free'].replace({True: 1, False: 0})
data['Ad Supported'] = data['Ad Supported'].replace({True: 1, False: 0})
data['In App Purchases'] = data['In App Purchases'].replace({True: 1, False: 0})
data['Editors Choice'] = data['Editors Choice'].replace({True: 1, False: 0})

# One Hot Encoding for the Category, Content Rating, Developer Id
categories = pd.get_dummies(data['Category'])
data = pd.concat([data, categories], axis=1)
data.drop('Category', axis=1, inplace=True)

content_ratings = pd.get_dummies(data['Content Rating'])
data = pd.concat([data, content_ratings], axis=1)
data.drop('Content Rating', axis=1, inplace=True)

list_of_developers = pd.DataFrame()
list_of_developers['Developer'] = data['Developer Id'].unique()
list_of_developers['Id'] = range(1, len(list_of_developers) + 1)

developers_dict = dict(zip(list_of_developers['Developer'], list_of_developers['Id']))
data['Developer Id'] = data['Developer Id'].map(developers_dict)

# Convert Size to MB
data['Size'] = data['Size'].apply(lambda x: convert_to_mb(x))

# Format dates
data['Released'] = pd.to_datetime(data['Released']).dt.strftime('%d.%m.%Y')
data['Last Updated'] = pd.to_datetime(data['Last Updated']).dt.strftime('%d.%m.%Y')

# Load Android Versions data
android_versions_dict = pd.read_csv('datasets/Android-Versions.csv')
android_versions_dict['Android version'] = android_versions_dict['Android version'].apply(lambda x: str(x))

android_versions_release_date = pd.DataFrame(android_versions_dict['Release date'].copy(deep=True))
android_versions_dict = dict(zip(android_versions_dict['Android version'], android_versions_dict['Release date']))

# Process Minimum Android version column
data['Minimum Android'] = data['Minimum Android'].apply(lambda x: iseci(x)) 
android_versions_dict.update({'nan': 0})
android_versions_dict.update({'Var': 'Varies with device'})
data['Minimum Android'] = data['Minimum Android'].map(android_versions_dict)

# Fill missing values
data['Rating'] = data['Rating'].fillna(0.0)
data['Rating Count'] = data['Rating Count'].fillna(0.0)

# Fill missing size values with the average size for each category
for category in categories.columns:
    average_size = data.loc[data[category] == 1, 'Size'].mean()
    data.loc[data['Size'].isna() & (data[category] == 1), 'Size'] = average_size

# Drop rows where Minimum Android is 'Varies with device'
data = data[data['Minimum Android'] != 'Varies with device']

# Handle missing and incorrect 'Released' values
data['Released'].fillna(data['Minimum Android'], inplace=True)
data = data[data['Released'].notna() & (data['Released'] != 0)]

# Convert dates to days since Unix epoch
android_versions_release_date['since_111970'] = android_versions_release_date['Release date'].apply(lambda x: days_since_epoch(x))
data['Minimum Android'] = data['Minimum Android'].apply(lambda x: days_since_epoch(x))
data['Released'] = data['Released'].apply(lambda x: days_since_epoch(x))
data['Last Updated'] = data['Last Updated'].apply(lambda x: days_since_epoch(x))

# Handle missing 'Minimum Android' values, drop rows where 'app name' is missing
to_be_fixed = data[data['Minimum Android'].isna()].index
data.loc[to_be_fixed, 'Minimum Android'] = data.loc[to_be_fixed, 'Released'].apply(lambda x: nadji_verziju(x))
data = data[data['Minimum Android'] != 0]

data = data[data['App Name'].notna()]

# Vectorize 'App Name' and apply KMeans clustering
vectorizer = TfidfVectorizer(ngram_range=(1, 4))
X = vectorizer.fit_transform(data['App Name'])

kmeans = KMeans(n_clusters=30, random_state=42)
data['App Name'] = kmeans.fit_predict(X)

joblib.dump(kmeans, 'final-results/kategorizacija_imena.joblib')

# One Hot Encoding for 'App Name' categories
app_names = pd.get_dummies(data['App Name'])
data = pd.concat([data, app_names], axis=1)
data.drop('App Name', axis=1, inplace=True)

# Save the final data
data.to_csv('final-results/Finalni_Podaci.csv', index=False)

y = data['Maximum Installs']
X = data.drop('Maximum Installs', axis=1)

X.columns = X.columns.astype(str)

# Training and saving
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, 'final-results/main_model.joblib')
