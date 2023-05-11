import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pytz
import json
import seaborn as sns
import sweetviz as sv
from scipy import stats

## Data Cleaning & Feature Engineer

# Removing Features: 'ph_country_code', 'fb_gender', 'fb_dob', 'ph_other_device_info'
df = df.drop('ph_country_code', axis=1)
df = df.drop('fb_dob', axis=1)
df = df.drop('fb_gender', axis=1)
df = df.drop('ph_other_device_info', axis=1)

# Standardizing data types: 'de_date_joined', 'fb_last_update_date'
df['de_date_joined'] = pd.to_datetime(df['de_date_joined'], format='%d/%m/%Y %H:%M')
df['days_since_joined'] = (datetime.now() - df['de_date_joined']).dt.days

utc = pytz.utc
df['fb_last_updated_date'] = pd.to_datetime(df['fb_last_updated_date'])
df['last_updated_utc'] = df['fb_last_updated_date'].dt.tz_convert(utc)
df['days_since_fb_update'] = (datetime.now(utc) - df['last_updated_utc']).dt.days
df = df.drop('fb_last_updated_date', axis=1)
df = df.drop('last_updated_utc', axis=1)
df = df.drop('days_since_fb_update', axis=1)

# 'ph_app_list': create a new column for the result of the length of the app list and drop 'ph_app_list'
def calculate_length(x):
    return len(eval(x))
df['app_list_length'] = df['ph_app_list'].apply(calculate_length)
df = df.drop('ph_app_list', axis=1)
print(df['app_list_length'])

# 'ph_call_log_stats': create new columns for each key-value pair in the dictionary to one-hot encoding
def extract_json_values(row, key):
    try:
        values = json.loads(row['ph_call_log_stats'])
        return values[key]
    except:
        return 0
keys = ['percent_incoming_nighttime', 'percent_outgoing_daytime', 'duration_incoming_daytime',
        'duration_outgoing_daytime', 'percent_incoming_daytime', 'percent_other_calls',
        'duration_outgoing_nighttime', 'percent_outgoing_nighttime', 'total_calls', 'duration_incoming_nighttime']
for key in keys:
    df[key] = df.apply(lambda row: extract_json_values(row, key), axis=1)

# Handle Missing Value: 'fb_relation', 'ph_call_log_stats', 'days_since_fb_update', 'last_updated_utc'
df['fb_relation']=df['fb_relation'].fillna("No_Relation")
df['ph_call_log_stats']=df['ph_call_log_stats'].fillna("No_Stats")

# 'fb_relation': recotegorise the variable types, create new columns for each key-value pair in the dictionary to one-hot encoding
df['fb_relation'] = df['fb_relation'].replace([
       "It's complicated", 'Widowed', 'Divorced',
       'In a civil union', "It's complicated (Pending)"], 'others')

df['fb_relation'] = df['fb_relation'].replace([
       'In an open relationship', 'Engaged',
       'In a domestic partnership', 'Separated',
       'Engaged (Pending)', 'Married (Pending)', 
       'In an open relationship (Pending)', 'In a relationship (Pending)',
       'In a domestic partnership (Pending)'], 'In a relationship')
counts = df["fb_relation"].value_counts()
plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%")
plt.title("fb_relation")
plt.show()

df['fb_relation'].unique()
def extract_json_values(row, key):
    try:
        values = json.loads(row['fb_relation'])
        return values[key]
    except:
        return 0

keys = ['Married', 'No_Relation', 'In a relationship', 'Single', 'others']
for key in keys:
    df[key] = df.apply(lambda row: extract_json_values(row, key), axis=1)

# Remove the outliers:
def remove_outliers(df, col_name, threshold):
    z_scores = stats.zscore(df[col_name])
    abs_z_scores = np.abs(z_scores)
    outliers = abs_z_scores > threshold
    df[col_name][outliers] = np.nan
    df.drop(df.index[outliers], inplace=True)
cols_to_process = ['de_monthly_salary', 'de_employment_duration', 'ph_total_contacts', 'days_since_joined', 'app_list_length', 'total_calls',
                   'duration_incoming_daytime', 'duration_incoming_nighttime', 'duration_outgoing_daytime', 'duration_outgoing_nighttime']
threshold = 3
for col in cols_to_process:
    remove_outliers(df, col, threshold)
