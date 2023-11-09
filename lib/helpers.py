#!usr/bin/env python

# Internal libraries

# External libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ABOUT: common setup & library to de-cluter notebook
pd.set_option('display.max_columns', None)  # don't collapse
pd.set_option('display.max_rows', 500)      # show more rows
pd.set_option('display.max_colwidth', 400)


# Shows raw stats per column. Codee modified from:
# https://www.kaggle.com/code/zelalemgetahun/eda-of-7-million-company-dataset/notebook
def df_overview(df: pd.DataFrame) -> pd.DataFrame:
    def percentage(lst):
        return [f'{str(round(value / df.shape[0] * 100, 2))}%' for value in lst]

    data_columns = list(df.columns)
    columns = [
        'column_name',
        'counts',
        '%_missing',
        '%_unique',
        'dtype'
    ]
    return pd.DataFrame(
        data=zip(
            data_columns,
            df.count().values,
            percentage(list(df.isna().sum())),
            percentage([df[column].value_counts().shape[0] for column in df]),
            df.dtypes
        ),
        columns=columns
    ).set_index('column_name')
