#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
import random
from faker import Faker
import datetime
import matplotlib.pyplot as plt
fake = Faker()

# Создать собственный DataFrame с придуманными данными (не менее 50 записей, минимум 4–5 колонок с разными типами данных).
num_rows = 75

data = {
    'ID': [],
    'Name': [],
    'Date': [],  # Day of activity
    'Start_Time': [],  # Start time of the activity
    'Duration_Minutes': [],  # Duration in minutes
    'Value': [],   # number of events
    'Active': []  # Whether the activity is currently running
}

user_ids = list(range(1000, 1075))
user_names = [fake.name() for _ in range(len(user_ids))]
for _ in range(num_rows):
    user_index = random.randint(0, len(user_ids) - 1)
    user_id = user_ids[user_index]
    user_name = user_names[user_index]
    data['ID'].append(user_id)
    data['Name'].append(user_name)

    date_start = fake.date_between(start_date='-14d', end_date='today')
    data['Date'].append(date_start)

    start_hour = random.randint(6, 22)
    start_minute = random.randint(0, 59)
    start_time = datetime.time(start_hour, start_minute)
    data['Start_Time'].append(start_time)

    duration_minutes = random.randint(5, 120)
    data['Duration_Minutes'].append(duration_minutes)

    value = random.randint(0, 50)
    data['Value'].append(value)

    data['Active'].append(random.choice([True, False]))

df = pd.DataFrame(data)

df['Date_Time'] = df.apply(lambda row: datetime.datetime.combine(row['Date'], row['Start_Time']), axis=1)
df['Duration'] = df['Duration_Minutes'].apply(lambda x: datetime.timedelta(minutes=x))
df['Duration_Formatted'] = df['Duration'].dt.components.minutes

print(df[['ID', 'Name', 'Date', 'Start_Time', 'Duration_Minutes', 'Value', 'Active', 'Date_Time', 'Duration_Formatted']].head().to_string())
print("\nData Types:")
print(df.dtypes)
print("\nDataFrame Info:")
df.info()
print("\nDataFrame Head:")
print(df.head().to_string())
print("\nDataFrame Description:")
print(df.describe().to_string())
print("\nMissing Values:")
print(df.isnull().sum())


def events_per_duration(df):
    active_df = df[df['Active'] == True]  # Only active Users
    # Count unique active users here
    unique_active_users = active_df['Name'].nunique()
    print(f"\nNumber of unique active users: {unique_active_users}")

    grouped = active_df.groupby('Name').agg(
        total_events=('Value', 'sum'),
        total_duration=('Duration_Minutes', 'sum')
    )
    grouped['events_per_minute'] = grouped['total_events'] / grouped['total_duration']
    plt.figure(figsize=(10, 6))  # Adjust figure size as needed
    grouped['events_per_minute'].hist(bins=15)  # You can adjust the number of bins
    plt.title('Distribution of Events per Minute for Active Users')
    plt.xlabel('Events per Minute')
    plt.ylabel('Frequency')
    plt.grid(False)  #remove the grid
    plt.show()
    return grouped['events_per_minute']


events_per_name = events_per_duration(df)
print("\nEvents per Minute for Each Active Name:")
print(events_per_name.to_string())

# Count active and inactive users
active_counts = df['Active'].value_counts()
print("\nActive and Inactive User Counts:")
print(active_counts)
def calculate_stats(events_per_minute):
    """Calculates mean, variance, and coefficient of variation for the 'events_per_minute' data."""
    mean = events_per_minute.mean()
    variance = events_per_minute.var()
    std_dev = events_per_minute.std()
    cv = std_dev / mean if mean != 0 else 0  # Coefficient of Variation

    print("\nStatistics for Events per Minute:")
    print(f"Mean: {mean:.4f}")
    print(f"Variance: {variance:.4f}")
    print(f"Coefficient of Variation: {cv:.4f}")

    # Create the boxplot
    plt.figure(figsize=(8, 6))  # Adjust figure size as needed
    plt.boxplot(events_per_minute, vert=False, patch_artist=True)  # Create horizontal boxplot
    plt.title('Boxplot of Events per Minute for Active Users')
    plt.xlabel('Events per Minute')
    plt.yticks([])  # Remove y-axis ticks as they are not needed
    plt.grid(False)
    plt.show()
    return mean, cv, variance  # Return mean, CV, and variance
calculate_stats(events_per_name)
def calculate_kendall_correlation(df, column1, column2):
    try:
        correlation = df[column1].corr(df[column2], method='kendall')
        print(f"\nKendall's Tau correlation between '{column1}' and '{column2}': {correlation:.4f}")
    except KeyError:
        print(f"Error: One or both columns ('{column1}', '{column2}') not found in the DataFrame.")
    except Exception as e:
        print(f"An error occurred during correlation calculation: {e}")


# Example usage: Calculate Kendall's Tau correlation between 'Value' and 'Duration_Minutes'
calculate_kendall_correlation(df, 'Value', 'Duration_Minutes')
def interpret_results(events_per_minute):
    """Interprets events per minute data using IQR and identifies high and low performers."""

    Q1 = events_per_minute.quantile(0.25)
    Q3 = events_per_minute.quantile(0.75)
    IQR = Q3 - Q1

    upper_bound = Q3 + 1.5 * IQR
    lower_bound = Q1 - 1.5 * IQR

    # Identify high performers (bonus candidates)
    high_performers = events_per_minute[events_per_minute >= upper_bound].index.tolist()

    # Identify low performers (private conversation needed)
    low_performers = events_per_minute[events_per_minute <= lower_bound].index.tolist()

    print("\nPerformance Interpretation:")
    if high_performers:
        print("\nBonus Eligible:")
        for name in high_performers:
            print(f"- {name}")
    else:
        print("\nNo candidates for bonus.")

    if low_performers:
        print("\nPrivate Conversation Needed:")
        for name in low_performers:
            print(f"- {name}")
    else:
        print("\nNo users for private conversation.")

interpret_results(events_per_name)


# In[ ]:




