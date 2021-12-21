import pandas as pd
import kaggle
import zipfile
from linear_regression import *

kaggle.api.authenticate()
kaggle.api.dataset_download_file('budincsevity/szeged-weather', file_name='weatherHistory.csv',  path='data/')

file_name = r"C:\Users\user\Documents\Python Environment\davidhw2\data\weatherHistory.csv.zip"
with zipfile.ZipFile(file_name, 'r') as zip_ref:
    zip_ref.extractall()


df = pd.read_csv(r'C:\Users\user\Documents\Python Environment\davidhw2\weatherHistory.csv', usecols=['Temperature (C)', 'Humidity', 'Wind Speed (km/h)'])
df.head()

print(df.head())

df.to_csv('weatherHistory2.csv', index=False)
df = pd.read_csv('weatherHistory2.csv')
df.head()

regression_estimates, standard_errors, credible_intervals, X, Y = linear_regression(data_set='weatherHistory2.csv')
