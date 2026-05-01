##############################################
# Programmer: Noah Barron
# Class: CPSC 222-02, Spring 2026
# Data Assignment #8 / Project S26
# May 1, 2026
# Description: This utils.py file holds the utility functions for the final project.
#              It does ...
##############################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def clean_data(df):
    '''
    Cleans the typing data by reversing, dropping columns, and re-doing the ids
    Parameter df: the dataframe that needs cleaning
    Returns: the cleaned dataframe
    '''
    # reversing the order of the dataframe
    # Source: “Reverse the Rows of a Pandas Data Frame?” Tutorialspoint.com, 2026,
    #         www.tutorialspoint.com/article/reverse-the-rows-of-a-pandas-data-frame.
    df = df[::-1]

    remove_cols = ["charStats", "mode", "mode2", "quoteLength", "restartCount", "afkDuration",
               "incompleteTestSeconds", "punctuation", "numbers", "language", "funbox",
               "difficulty", "lazyMode", "blindMode", "bailedOut", "tags", "testDuration",
               "timestamp"]
    df = df.drop(columns=remove_cols)

    # replacing the ids with numbers 0 to n:
    size = df["_id"].size
    for i in range(size):
        df.loc[size - i - 1, "_id"] = i

    return df

def add_columns(df):
    '''
    Adding "date" and "night" columns to the typing table (I know this
    is very inelegant, but I don't know if I'll have time to fix it.)
    Parameter df: the dataframe that we're adding columns to
    Returns: the dataframe with the columns added
    '''
    night_list = ["true", "false", "true", "false", "true", "false", "true", "false", "true", "false", "true", "false", "true",
                  "true", "false", "true", "false", "true", "false", "false", "true", "false", "true", "false", "true", "false",
                  "true", "false", "true", "false", "true", "false", "true", "false", "true", "true", "false", "false", "true",
                  "false", "true", "false", "true", "false", "true", "false", "true", "false", "true", "true", "false", "true",
                  "false", "true", "false", "true", "false", "true", "false", "true", "false", "true", "false", "true", "false",
                  "false", "true", "false", "true", "false", "true", "false", "true", "false", "true", "true", "true", "true",
                  "true", "false", "true", "false", "true", "false", "true","false", "true", "false", "true", "false", "false",
                  "true", "true", "true", "false", "true", "false", "true", "false", "true", "false", "true", "false", "true",
                  "true", "false", "true", "false", "true", "false", "false", "true", "false", "true", "false", "false", "false",
                  "true", "false", "true", "false", "true", "false", "true", "false", "true"]
    df.insert(1, "night", night_list)

    date_list = ["2-9-26", "2-10-26", "2-10-26", "2-11-26", "2-11-26", "2-12-26", "2-12-26", "2-13-26", "2-13-26", "2-14-26",
                 "2-14-26", "2-15-26", "2-15-26", "2-16-26", "2-17-26", "2-17-26", "2-18-26", "2-18-26", "2-19-26", "2-20-26",
                 "2-20-26", "2-21-26", "2-21-26", "2-22-26", "2-22-26", "2-23-26", "2-23-26", "2-24-26", "2-24-26", "2-25-26",
                 "2-25-26", "2-26-26", "2-26-26", "2-27-26", "2-27-26", "2-28-26", "3-1-26", "3-2-26", "3-2-26", "3-3-26",
                 "3-3-26", "3-4-26", "3-4-26", "3-5-26", "3-5-26", "3-6-26", "3-6-26", "3-7-26", "3-7-26", "3-8-26", "3-9-26",
                 "3-9-26", "3-10-26", "3-10-26", "3-11-26", "3-11-26", "3-12-26", "3-12-26", "3-13-26", "3-13-26", "3-14-26",
                 "3-14-26", "3-15-26", "3-15-26", "3-16-26", "3-17-26", "3-17-26", "3-18-26", "3-18-26", "3-19-26", "3-19-26",
                 "3-21-26", "3-21-26", "3-22-26", "3-22-26", "3-23-26", "3-24-26", "3-25-26", "3-26-26", "3-27-26", "3-27-26",
                 "3-28-26", "3-28-26", "3-29-26", "3-29-26", "3-30-26", "3-30-26", "3-31-26", "4-1-26", "4-2-26", "4-7-26",
                 "4-7-26", "4-8-26", "4-9-26", "4-10-26", "4-12-26", "4-13-26", "4-13-26", "4-14-26", "4-14-26", "4-15-26",
                 "4-15-26", "4-16-26", "4-16-26", "4-17-26", "4-19-26", "4-19-26", "4-20-26", "4-20-26", "4-21-26", "4-22-26",
                 "4-22-26", "4-23-26", "4-23-26", "4-24-26", "4-25-26", "4-26-26", "4-26-26", "4-27-26", "4-27-26", "4-28-26",
                 "4-28-26", "4-29-26", "4-29-26", "4-30-26", "4-30-26"]
    df.insert(1, "date", date_list)

    return df

def split_months(df) :
    '''
    Splitting the typing data into it's months
    Parameter df: the dataframe that will be split
    Returns: a list of 3 dataframes, each holding one month's worth of data
    '''
    # hard-coding the numbers in, unless I have time to go back and make it more dynamic
    feb_df = df.iloc[0:36] # the numbers are the indexes of where the months start and stop
    mar_df = df.iloc[36:88]
    apr_df = df.iloc[88:126]

    return [feb_df, mar_df, apr_df]

def wpm_avg_over_time(months_split):
    '''
    ...
    '''
    morning_wpm_avg = []
    night_wpm_avg = []

    num_months = 3
    for i in range(num_months):
        # split into morning/day
        grouped_by_time = months_split[i].groupby("night")
        morning_df = grouped_by_time.get_group("false")
        night_df = grouped_by_time.get_group("true")

        morning_wpm_avg.append(morning_df["wpm"].mean())
        night_wpm_avg.append(night_df["wpm"].mean())
    
    return morning_wpm_avg, night_wpm_avg

def acc_avg_over_time(months_split):
    '''
    ...
    '''
    morning_acc_avg = []
    night_acc_avg = []

    num_months = 3
    for i in range(num_months):
        # split into morning/day
        grouped_by_time = months_split[i].groupby("night")
        morning_df = grouped_by_time.get_group("false")
        night_df = grouped_by_time.get_group("true")

        morning_acc_avg.append(morning_df["acc"].mean())
        night_acc_avg.append(night_df["acc"].mean())
    
    return morning_acc_avg, night_acc_avg

def days_raw_wpm(df, days):
    '''
    ...
    '''
    days_raw_wpm = []
    for day in days:
        day_raw_wpm = df.groupby("dayOfTheWeek").get_group(day)["rawWpm"].mean()
        days_raw_wpm.append(day_raw_wpm)
    return days_raw_wpm

def plot_grouped_bar(month_names, heights1, heights2, ylim, ylabel, title):
    '''
    ...
    '''
    plt.figure()
    plt.bar(month_names, height=heights1, width=-0.2, align="edge", label="morning")
    plt.bar(month_names, height=heights2, width=0.2, align="edge", label="night")
    plt.ylim(ylim)
    plt.legend(loc="upper right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def plot_scatter(x, y, xlabel, ylabel, title):
    '''
    ...
    '''
    plt.figure()
    plt.scatter(x, y)
    plt.xticks(rotation=45)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def plot_bar(days, heights, ylim, ylabel, title):
    '''
    ...
    '''
    plt.figure()
    plt.bar(days, height=heights)
    plt.ylim(ylim)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def hypothesis_test(x, y, alpha, two_tailed, dependent):
    '''
    ...
    '''
    t_computed = 0
    p_val = 0
    if dependent:
        t_computed, p_val = stats.ttest_rel(x, y)
    else:
        t_computed, p_val = stats.ttest_ind(x, y)
    if two_tailed:
        p_val /= 2 # because it's one-tailed

    print("Step 3 (Cont.):")
    print("t computed:", t_computed, "p value:", p_val)

    print()
    print("Step 4:\nIf our p-value is less than our alpha, we reject the null hypothesis.")
    print()
    print("Step 5:")
    if p_val < alpha:
        print("We reject the null hypothesis.")
    else:
        print("We do NOT reject the null hypothesis.")

def encode_data(df):
    '''
    ...
    '''
    pb_le = LabelEncoder()
    night_le = LabelEncoder()
    days_le = LabelEncoder()
    pb_le = pb_le.fit([np.nan, True])
    night_le = night_le.fit(["false", "true"])
    days_le = days_le.fit(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    df["isPb"] = pb_le.transform(df["isPb"])
    df["night"] = night_le.transform(df["night"])
    df["dayOfTheWeek"] = days_le.transform(df["dayOfTheWeek"])

    return df

def knn_clf(X, y):
    '''
    ...
    ...
    ...
    Returns: the accuracy score of the created kNN classifer
    '''
    scaler = MinMaxScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=0)

    knn_clf = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
    knn_clf.fit(X_train, y_train)
    y_predicted = knn_clf.predict(X_test)
    return accuracy_score(y_test, y_predicted)

def tree_clf(X, y):
    '''
    ...
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    tree_clf = DecisionTreeClassifier()
    tree_clf.fit(X_train, y_train)
    y_predicted = tree_clf.predict(X_test)
    return accuracy_score(y_test, y_predicted)
