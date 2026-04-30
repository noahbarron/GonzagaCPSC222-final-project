##############################################
# Programmer: Noah Barron
# Class: CPSC 222-02, Spring 2026
# Data Assignment #8 / Project S26
# May 1, 2026
# Description: This utils.py file holds the utility functions for the final project.
#              It does ...
##############################################

import pandas as pd

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
                  "true", "false", "true", "false", "true", "false", "true"]
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
                 "4-28-26", "4-29-26", "4-29-26"]
    df.insert(1, "date", date_list)

    return df