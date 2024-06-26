import sys
from sqlalchemy import create_engine
import pandas as pd


def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories data from CSV files and merge them into a single DataFrame.

    Parameters:
    messages_filepath (str): Filepath of the messages CSV file.
    categories_filepath (str): Filepath of the categories CSV file.

    Returns:
    df (DataFrame): Merged DataFrame containing messages and corresponding categories.
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge dataframes
    df = messages.merge(categories, how='outer', on='id')

    return df

def clean_data(df):
    """
    Clean and preprocess the merged DataFrame.

    Parameters:
    df (DataFrame): Merged DataFrame containing messages and categories.

    Returns:
    df (DataFrame): Cleaned and preprocessed DataFrame.
    """

    # split cate columns
    categories = df['categories'].str.split(";", expand=True)

    # create columns for splited cate
    row = categories.iloc[0, :]
    category_colnames = row.str.rstrip("-01")

    # rename the columns of `categories`
    categories.columns = category_colnames

    # only takes last values
    categories = categories.transform(lambda x: x.str.get(-1).astype(int))

    # drop the original categories column from `df`
    df = df.drop(columns=["categories"])
    df = pd.concat([df, categories], axis=1)

    # drop values that is not zero and one
    df = df.drop(df[df["related"] == 2].index)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df

def save_data(df, database_filename):
    """
    Save the cleaned DataFrame to an SQLite database.

    Parameters:
    df (DataFrame): Cleaned and preprocessed DataFrame.
    database_filename (str): Filepath of the SQLite database to save the data.
    """
    
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('clean_disaster_data', engine, index=False, if_exists='replace')


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
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()