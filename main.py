#!/usr/local/bin/python3.12

import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

app = pd.read_csv('./App_Template.csv')

app_categories = [  'Action','Adventure', 'Arcade', 'Art & Design', 'Auto & Vehicles', 'Beauty',
                    'Board', 'Books & Reference', 'Business', 'Card', 'Casino', 'Casual',
                    'Comics', 'Communication', 'Dating', 'Education', 'Educational',
                    'Entertainment', 'Events', 'Finance', 'Food & Drink',
                    'Health & Fitness', 'House & Home', 'Libraries & Demo', 'Lifestyle',
                    'Maps & Navigation', 'Medical', 'Music', 'Music & Audio',
                    'News & Magazines', 'Parenting', 'Personalization', 'Photography',
                    'Productivity', 'Puzzle', 'Racing', 'Role Playing', 'Shopping',
                    'Simulation', 'Social', 'Sports', 'Strategy', 'Tools', 'Travel & Local',
                    'Trivia', 'Video Players & Editors', 'Weather', 'Word']

app_content_rating = ['Adults only 18+', 'Everyone', 'Everyone 10+', 'Mature 17+', 'Teen', 'Unrated']
                      
app_unsupervised_learning_categories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11',
                                        '12', '13', '14', '15', '16', '17', '18', '19', '20', '21',
                                        '22', '23','24', '25', '26', '27', '28', '29']

app_developers = pd.read_csv('./Developers.csv')
app_developers_list = app_developers['Developer'].apply(lambda x: str(x))
app_developers_list = app_developers['Developer'].to_list()

global filtered
filtered = app_developers_list.copy()

app_android_versions_dict = pd.read_csv('./Android-Versions.csv')
app_android_versions_list = app_android_versions_dict['Android version'].to_list()
app_android_versions_dict['Android version'] = app_android_versions_dict['Android version'].apply(lambda x: str(x))

android_versions_release_date = pd.DataFrame(app_android_versions_dict['Release date'].copy(deep=True))
app_android_versions_dict = dict(zip(app_android_versions_dict['Android version'], app_android_versions_dict['Release date']))

def update_results(event):

    query = developer_entry.get()
    if query != '':
       print('FILTRIRAM')
       filtered = [str(dev) for dev in app_developers_list if query in str(dev)]
    if query == '':
       filtered = app_developers_list.copy()
    developer_entry['values'] = filtered


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

def predict():
    
    app_name = app_name_entry.get()
    free_price = free_price_entry.get()
    size = size_entry.get()
    minimum_android = minimum_android_combobox.get()
    developer = developer_entry.get()
    ad_supported = ad_supported_var.get()
    in_app_purchases = in_app_purchases_var.get()
    category = category_combobox.get()
    content_rating = content_rating_combobox.get()

    if app_name == '' or free_price == '' or size == '' or minimum_android == '' or developer == '' or category == '' or content_rating == '':
        output_text.delete(1.0, tk.END)
        output_text.insert(tk.END, f"Sva polja moraju biti popunjena!\n\n")
        output_text.insert(tk.END, f"Ako je aplikacija besplatna ostaviti 0.0 vrednost.\n")
        output_text.insert(tk.END, f"Za veličinu aplikacje koristiti oznake:\n")
        output_text.insert(tk.END, f"'k', 'M', 'G'\n")
        output_text.insert(tk.END, f"Primer: 3.14M\n")
        return -1

    kmeans = joblib.load('./kategorizacija_imena.joblib')
    app_name_categories = kmeans.predict(X=app_name)
    app = pd.concat([app, app_name_categories], axis=1)
    del kmeans

    free_price = float(free_price)
    if free_price != 0.0:
        app['Price'] == float(free_price)

    app['Size'] = convert_to_mb(size)

    app['Minimum Android'] = minimum_android
    app['Minimum Android'].map(app_android_versions_dict)

    if developer not in app_developers_list:
        new_developer = {'Developer': developer, 'Id': (max(app_developers['Id'])+1)}
        app_developers.append(new_developer, ignore_index=True)
        app_developers.to_csv('Developers.csv', index=False)
    app['Developer Id'] = app_developers.loc[app_developers['Developer'] == developer]['Id']

    app['Ad Supported'] = ad_supported
    app['Ad Supported'].replace({True: 1, False: 0}, inplace=True)

    app['In App Purchases'] = in_app_purchases
    app['In App Purchases'].replace({True: 1, False: 0}, inplace=True)

    app[category] = 1

    app[content_rating] = 1
    app.to_csv('to_predict.csv')

    model = joblib.load('./main_model.joblib')
    prediction = model.predict(X=app)

    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, f"Shodno zadatim parametrima predvidjanje modela je:\n"
                               f"{prediction} preuzimanja.\n\n"
                               f"Predvidjanje je informativnog karaktera!")
    

root = tk.Tk()
root.title("App Prediction")

app_name_label = ttk.Label(root, text="App Name:")
app_name_label.grid(row=0, column=0, padx=5, pady=10)
app_name_entry = ttk.Entry(root)
app_name_entry.grid(row=0, column=1, padx=5, pady=10)

free_price_label = ttk.Label(root, text="Price:")
free_price_label.grid(row=1, column=0, padx=5, pady=10)
free_price_entry = ttk.Entry(root)
free_price_entry.insert(0, "0.0") 
free_price_entry.grid(row=1, column=1, padx=5, pady=10)

size_label = ttk.Label(root, text="Size:")
size_label.grid(row=2, column=0, padx=5, pady=10)
size_entry = ttk.Entry(root)
size_entry.grid(row=2, column=1, padx=5, pady=10)

minimum_android_label = ttk.Label(root, text="Minimum Android:")
minimum_android_label.grid(row=3, column=0, padx=5, pady=10)
minimum_android_combobox = ttk.Combobox(root, values=app_android_versions_list)
minimum_android_combobox.grid(row=3, column=1, padx=5, pady=10)

developer_label = ttk.Label(root, text="Developer:")
developer_label.grid(row=4, column=0, padx=5, pady=10)
developer_entry = ttk.Combobox(root, values=filtered)
developer_entry.grid(row=4, column=1, padx=5, pady=10)
developer_entry.bind('<KeyRelease>', update_results)

ad_supported_var = tk.BooleanVar()
ad_supported_check = ttk.Checkbutton(root, text="Ad Supported", variable=ad_supported_var)
ad_supported_check.grid(row=5, column=0, padx=5, pady=10)

in_app_purchases_var = tk.BooleanVar()
in_app_purchases_check = ttk.Checkbutton(root, text="In-app Purchases", variable=in_app_purchases_var)
in_app_purchases_check.grid(row=5, column=1, padx=5, pady=10)

category_label = ttk.Label(root, text="Category:")
category_label.grid(row=6, column=0, padx=5, pady=10)
category_combobox = ttk.Combobox(root, values=app_categories)
category_combobox.grid(row=6, column=1, padx=5, pady=10)

content_rating_label = ttk.Label(root, text="Content Rating:")
content_rating_label.grid(row=7, column=0, padx=5, pady=10)
content_rating_combobox = ttk.Combobox(root, values=app_content_rating)
content_rating_combobox.grid(row=7, column=1, padx=5, pady=10)

predict_button = ttk.Button(root, text="Predict", command=predict)
predict_button.grid(row=8, column=0, columnspan=2, padx=5, pady=10)

output_text = tk.Text(root, font=14, height=10, width=40)
output_text.grid(row=9, column=0, columnspan=2, padx=5, pady=10)

output_text.delete(1.0, tk.END)
output_text.insert(tk.END, f"Ako je aplikacija besplatna ostaviti 0.0 vrednost.\n")
output_text.insert(tk.END, f"Za veličinu aplikacje koristiti oznake:\n")
output_text.insert(tk.END, f"'k', 'M', 'G'\n")
output_text.insert(tk.END, f"Primer: 3.14M\n")

root.mainloop()
