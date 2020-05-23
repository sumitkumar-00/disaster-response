# ETL Pipeline
# This python utility processes data file and loads them in a sqlite database

# Import python libraries
import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load data from files

    Input:
    :param messages_filepath: file containing message data to be loaded in the database
    :param categories_filepath: file containing category data to be loaded in the database
    :return: pandas dataframe containing messages and categories
    """
    # Load message data set
    messages = pd.read_csv(messages_filepath)
    if messages.shape[0] > 0:
        print('messages loaded successfully with {} rows and {} columns' .format(messages.shape[0], messages.shape[1]))

    # load categories data set
    categories = pd.read_csv(categories_filepath)
    if categories.shape[0] > 0:
        print('categories loaded successfully with {} rows and {} columns' .format(categories.shape[0], categories.shape[1]))

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
    df.to_sql('disaster_response', engine, if_exists='replace', index=False)
    print('data successfully saved in the database')
    return None


def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
