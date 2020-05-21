# ETL Pipeline
# This python utility processes data file and loads them in a sqlite database

# Import python libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(message_file, category_file):
    """
    Load data from files

    Input:
    :param message_file: file containing message data to be loaded in the database
    :param category_file: file containing category data to be loaded in the database
    :return: pandas dataframe containing messages and categories
    """
    # Load message data set
    messages = pd.read_csv(message_file)
    if messages.shape[0] > 0:
        print('messages loaded successfully with {} rows and {} columns' .format(messages.shape[0], messages.shape[1]))

    # load categories data set
    categories = pd.read_csv(category_file)
    if categories.shape[0] > 0:
        print('messages loaded successfully with {} rows and {} columns' .format(categories.shape[0], categories.shape[1]))

    # merge data sets
    df = messages.merge(categories, how='inner', on='id')

    return df

def clean_data(df):
    """
    clean data for analysis
    :param df: dataframe
    :return: cleaned dataframe
    """

    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0, ]

    # use lambda functions to get a list of column names for categories
    category_colnames = list(row.apply(lambda x: x[0:len(x)-2]))

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

    # convert column from string to numeric
    categories[column] = categories[column].astype('int')

    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    # drop rows where related=2. We are only working with binary response for a category
    df = df[df.related != 2]

    # drop child_alone column as no message is tied to this category
    df = df.drop(['child_alone'], axis=1)

    return df


def save_data(df, database_file):
    """
    Save data in sqlite database
    :param database_file: path to database
    :param df: dataframe to be saved as table in the database
    :return: None
    """
    engine = create_engine('sqlite:///' + database_file)
    df.to_sql('InsertTableName', engine, if_exists='replace', index=False)
    print('data successfully loaded in the database')
    return None


def main():
    if len(sys.argv) == 4:
        message_file, category_file, database_file = sys.argv[1:4]
        df = load_data(message_file, category_file)

        df = clean_data(df)

        save_data(df, database_file)
    else:
        print('\n Please provide message, category and database file path in that order.'\
              '\n Execute: python process_data.py messages.csv categories.csv InsertDatabaseName.db')


main()
