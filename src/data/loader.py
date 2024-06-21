import pandas as pd
from typing import Callable
from functools import partial, reduce
# import babel.dates
import datetime as dt
# import i18n
# Defining the type of function that Preprocessor is, it takes a pd.df as input
# and returns a pd.df as output.
Preprocessor = Callable[[pd.DataFrame], pd.DataFrame]

class DataSchema:
    AMOUNT = 'amount'
    CATEGORY = 'category'
    DATE = 'date'
    MONTH = 'month'
    YEAR = 'year'


def create_year_column(df: pd.DataFrame) -> pd.DataFrame:
    df[DataSchema.YEAR] = df[DataSchema.DATE].dt.year.astype(str)
    return df

def create_month_column(df: pd.DataFrame) -> pd.DataFrame:
    df[DataSchema.MONTH] = df[DataSchema.DATE].dt.month.astype(str)
    return df
# returns a "lambda x: g(f(x))" so this returns a lambda function that is a
# composition of the input functions. If you give it more than two functions
# as input it will compose the functions from left to right due to the usage of
# reduce.
# Exm:
# compose(f,g,h) -> lambda x: h(g(f(x))
# it first takes f,g and creates "lambda x: g(f(x))", then in the second round
# it takes g,h and creates "lambda x: h(g(f(x)))", etc...
# This is usefull if you want to create columns on a df.
def compose(*functions: Preprocessor) -> Preprocessor:
    return reduce(lambda f, g: lambda x: g(f(x)), functions)


def load_transaction_data(path: str) -> pd.DataFrame:
    # load the data from csv file
    data = pd.read_csv(
        path,
        dtype={
            DataSchema.AMOUNT: float,
            DataSchema.CATEGORY: str,
            # You can't pre-load DATE as str in python 3.12 since it seems to persist as str
            # and doesn't care that you later parse it as a Date type column.
            # DataSchema.DATE: str,
        },
        parse_dates=[DataSchema.DATE]
    )
    # NOTE: preprocessor = lambda x: create_month_column(create_year_column(x))
    # 'x' is the expected df that will be modified.
    preprocessor = compose(
        create_year_column,
        create_month_column,
        # because all the funcs are of type 'Preproccesor' which expects only a df as input, we need to use partial
        # since translate_date has two inputs, df and locale, so we partially call it with the locale kwarg defined
        # then it will only need a df kwarg.
        # partial(translate_date, locale=locale),
    )
    return preprocessor(data)