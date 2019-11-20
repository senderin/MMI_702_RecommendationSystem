# standard library imports
from ast import literal_eval
import itertools
import time
import re

# third-party imports
import numpy as np
import pandas as pd
from IPython.display import display

#detect the columns with more than 50% missing values to remove
def detectColumnsWithMissingValue():
    threshold = raw_steam_data.shape[0] // 2

    print('Drop columns with more than {} missing rows'.format(threshold))

    # see how many missing values we have in each column
    null_counts = raw_steam_data.isnull().sum()
    drop_rows = raw_steam_data.columns[null_counts > threshold]

    print('Columns to drop: {}'.format(list(drop_rows)))

def detectRowsWithNoInformation():
    print('Rows to remove:', raw_steam_data[raw_steam_data['type'].isnull()].shape[0])
    # preview rows with missing type data
    display(raw_steam_data[raw_steam_data['type'].isnull()].head(3))


def drop_null_cols(df, thresh=0.5):
    """Drop columns with more than a certain proportion of missing values (Default 50%)."""
    cutoff_count = len(df) * thresh

    return df.dropna(thresh=cutoff_count, axis=1)


def process_name_type(df):
    """Remove null values in name and type columns, and remove type column."""
    df = df[df['type'].notnull()]

    df = df[df['name'].notnull()]
    df = df[df['name'] != 'none']

    df = df.drop('type', axis=1)

    return df


def process(df):
    """Process data set. Will eventually contain calls to all functions we write."""

    # Copy the input dataframe to avoid accidentally modifying original data
    df = df.copy()

    # Remove duplicate rows - all appids should be unique
    df = df.drop_duplicates()

    # Remove collumns with more than 50% null values
    df = drop_null_cols(df)

    # Process rest of columns
    df = process_name_type(df)

    return df

def analyze_dataset():
    # print out number of rows and columns
    print('Rows:', raw_steam_data.shape[0])
    print('Columns:', raw_steam_data.shape[1])

    # view first five rows
    # display(raw_steam_data.head())

    # see how many missing values we have in each column
    null_counts = raw_steam_data.isnull().sum()
    # display(null_counts)

    # detect the columns with more than 50% missing values to remove
    detectColumnsWithMissingValue()

    # detect the rows if no information was returned from the request -only the name and appid was stored
    detectRowsWithNoInformation()

    # remove 'type' column
    # the counts of unique values in a column by using the pandas Series.value_counts method
    display(raw_steam_data['type'].value_counts(dropna=False))

    # check 'name' column for rows which either have a null value or a string containing 'none'
    raw_steam_data[(raw_steam_data['name'].isnull()) | (raw_steam_data['name'] == 'none')]

    # remove the extra rows
    # view duplicated rows using the DataFrame.duplicated() method of pandas
    # keep=False to view all duplicated rows, (keep='first') to skip over the first row
    # a column label into subset if we want to filter by a single column
    duplicate_rows = raw_steam_data[raw_steam_data.duplicated()]
    print('Duplicate rows to remove:', duplicate_rows.shape[0])
    display(duplicate_rows.head(3))


def process_age(df):
    """Format ratings in age column to be in line with the PEGI Age Ratings system."""
    # PEGI Age ratings: 3, 7, 12, 16, 18
    cut_points = [-1, 0, 3, 7, 12, 16, 2000]
    label_values = [0, 3, 7, 12, 16, 18]

    df['required_age'] = pd.cut(df['required_age'], bins=cut_points, labels=label_values)

    return df

def process_platforms(df):
    """Split platforms column into separate boolean columns for each platform."""
    # evaluate values in platforms column, so can index into dictionaries
    df = df.copy()

    def parse_platforms(x):
        d = literal_eval(x)

        return ';'.join(platform for platform in d.keys() if d[platform])

    #process the rows
    df['platforms'] = df['platforms'].apply(parse_platforms)

    return df

def print_steam_links(df):
    """Print links to store page for apps in a dataframe."""
    url_base = "https://store.steampowered.com/app/"

    for i, row in df.iterrows():
        appid = row['steam_appid']
        name = row['name']

        print(name + ':', url_base + str(appid))


def process_price(df):
    """Process price_overview column into formatted price column."""
    df = df.copy()

    def parse_price(x):
        if x is not np.nan:
            return literal_eval(x)
        else:
            return {'currency': 'TRY', 'initial': -1}

    # evaluate as dictionary and set to -1 if missing
    df['price_overview'] = df['price_overview'].apply(parse_price)

    # create columns from currency and initial values
    df['currency'] = df['price_overview'].apply(lambda x: x['currency'])
    df['price'] = df['price_overview'].apply(lambda x: x['initial'])

    # set price of free games to 0
    df.loc[df['is_free'], 'price'] = 0

    # remove non-TRY rows
    df = df[df['currency'] == 'TRY']

    # remove rows where price is -1
    df = df[df['price'] != -1]

    # change price to display in pounds (only applying to rows with a value greater than 0)
    df.loc[df['price'] > 0, 'price'] /= 100

    # remove columns no longer needed
    df = df.drop(['is_free', 'currency', 'price_overview'], axis=1)

    return df


def process_language(df):
    """Process supported_languages column into a boolean 'is english' column."""
    df = df.copy()

    # drop rows with missing language data
    df = df.dropna(subset=['supported_languages'])

    df['english'] = df['supported_languages'].apply(lambda x: 1 if 'english' in x.lower() else 0)
    df = df.drop('supported_languages', axis=1)

    return df


def process_developers_and_publishers(df):
    # remove rows with missing data
    df = df[(df['developers'].notnull()) & (df['publishers'] != "['']")].copy()

    for col in ['developers', 'publishers']:
        df[col] = df[col].apply(lambda x: literal_eval(x))

        # filter dataframe to rows with lists longer than 1, and store the number of rows
        num_rows = df[df[col].str.len() > 1].shape[0]

        print('Rows in {} column with multiple values:'.format(col), num_rows)


# customisations
pd.set_option("max_columns", None)
pd.set_option("max_rows", None)

# read in downloaded data
raw_steam_data = pd.read_csv('steam_app_data.csv')

#analyze the dataset
#analyze_dataset()

#start to cleaning
print ("Starting to cleaning...")
#print(raw_steam_data.shape)
initial_processing = process(raw_steam_data)
#print(initial_processing.shape)
#display(initial_processing.head())

#processing age
display(initial_processing['required_age'].value_counts(dropna=False).sort_index())
#reducing the number of categories that ages fall into
age_df = process_age(initial_processing)
display(age_df['required_age'].value_counts().sort_index())

#processing the 'platforms' column
display(age_df['platforms'].head())
#even though the data looks like a dictionary it is in fact stored as a string
platforms_first_row = age_df['platforms'].iloc[0]
print(type(platforms_first_row))
display(platforms_first_row)
#to recognise the data in the columns as dictionaries rather than just strings
#literal_eval evaluates the string, and then index into it as a dictionary
eval_first_row = literal_eval(platforms_first_row)
print(type(eval_first_row))
print(eval_first_row)
display(eval_first_row['windows'])
#check for missing values
print (age_df['platforms'].isnull().sum())
# create string of keys, joined on a semi-colon
#create the desired list by calling the str.join() method on a string, and passing an iterable into the function
display(';'.join(eval_first_row.keys()))
#return a list of supported platforms
#example of how to do it
platforms = {'windows': True, 'mac': True, 'linux': False}
# list comprehension
print([x for x in platforms.keys() if platforms[x]])
# using list comprehension in join
';'.join(x for x in platforms.keys() if platforms[x])
#putting this all together
platforms_df = process_platforms(age_df)
print (platforms_df['platforms'].value_counts())

#processing price
#check how many null values there are in 'price_overview'
print (platforms_df['price_overview'].isnull().sum())
#check with 'is_free' column
free_and_null_price = platforms_df[(platforms_df['is_free']) & (platforms_df['price_overview'].isnull())]
print (free_and_null_price.shape[0])
#Difference between them means that there are almost 850 rows which aren't free
#but have null values in the price_overview column
not_free_and_null_price = platforms_df[(platforms_df['is_free']==False) & (platforms_df['price_overview'].isnull())]
print (not_free_and_null_price.shape[0])
#display(not_free_and_null_price)
#It looks like we can rule out data errors,
#so let's dig a little deeper and see
# if we can find out what is going on
print_steam_links(not_free_and_null_price)
#...
#check the value under the currency key
print (platforms_df['price_overview'][37])
print (platforms_df['price_overview'][0])
#process on dataset
price_df = process_price(platforms_df)
display(price_df[['name', 'price']].head())

#processing packages
# temporarily set a pandas option using with and option_context
with pd.option_context("display.max_colwidth", 100):
    display(price_df[['steam_appid', 'packages', 'package_groups', 'price']].head(3))
#missing price data, which we previously set to -1
print(price_df[price_df['price'] == -1].shape[0])

#split these rows into two categories:
#those with package_groups data and those without
print('Null counts:', price_df['package_groups'].isnull().sum())
print('Empty list counts:', price_df[price_df['package_groups'] == "[]"].shape[0])
#find out how many rows have both missing price and package_group data
#these will be removed
missing_price_and_package = price_df[(price_df['price'] == -1) & (price_df['package_groups'] == "[]")]
print('Number of rows:', missing_price_and_package.shape[0])
print('First few rows:\n')
print_steam_links(missing_price_and_package[:5])
print('\nLast few rows:\n')
print_steam_links(missing_price_and_package[-10:-5])
#the rows that have missing price data but do have package_groups data
#these will be kept
missing_price_have_package = price_df.loc[(price_df['price'] == -1) & (price_df['package_groups'] != "[]"), ['name', 'steam_appid', 'package_groups', 'price']]
print('Number of rows:', missing_price_have_package.shape[0], '\n')
print('First few rows:\n')
print_steam_links(missing_price_have_package[:5])
print('\nLast few rows:\n')
print_steam_links(missing_price_have_package[-10:-5])

#processing languages
#create a column marking english games with a boolean value - True or False
#looking for rows with null values
print (price_df['supported_languages'].isnull().sum())
display(price_df[price_df['supported_languages'].isnull()])
#take a look at the structure of the column
print(price_df['supported_languages'][0])
display(price_df['supported_languages'].value_counts().head(10))
#do not apply this process
language_df = price_df
#language_df = process_language(price_df)
#language_df[['name', 'english']].head()
#language_df['english'].value_counts()

#detect rows missing either developer or publisher information
no_dev = language_df[language_df['developers'].isnull()]
print('Total games missing developer:', no_dev.shape[0])
print_steam_links(no_dev[:5])
no_pub = language_df[language_df['publishers'] == "['']"]
print('Total games missing publisher:', no_pub.shape[0])
print_steam_links(no_pub[:5])
no_dev_or_pub = language_df[(language_df['developers'].isnull()) & (language_df['publishers'] == "['']")]
print('Total games missing developer and publisher:', no_dev_or_pub.shape[0], '\n')
print_steam_links(no_dev_or_pub[:5])
display(language_df[['developers', 'publishers']].iloc[24:28])
#find the number of rows with more than one value in each column
process_developers_and_publishers(language_df)

display(language_df.loc[language_df['developers'].str.contains(",", na=False), ['steam_appid', 'developers', 'publishers']].head(4))
display(language_df.loc[language_df['developers'].str.contains(";", na=False), ['steam_appid', 'developers', 'publishers']])