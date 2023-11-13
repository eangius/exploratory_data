#!usr/bin/env python

# ABOUT: specific functionality pertaining to the dataset & type of analysis desired.
from address_parser import Parser
from datetime import datetime
import matplotlib.pyplot as plt
import calendar
import pandas as pd
import seaborn as sns
sns.set_theme(style="darkgrid")

parser = Parser()


# Attempt to parse address & extract its name.
def parse_addr_name(addr: str) -> str:
    try:
        return parser.parse(addr).road.name
    except Exception:
        return 'unknown'


# See if road type keywords exist near the end of the address.
def parse_addr_type(addr: str) -> str:
    try:
        for token in set(addr.upper().split()[-2:]):
            if token in {
                'RD', 'AVE', 'ST', 'WAY',
                'BLVD', 'DR', 'ROAD', 'CRT',
                'CRES', 'LANE', 'TRL',
            }:
                return token
        raise
    except Exception:
        return 'unknown'


# Converts a numerical 24hr clock value formatted as "234.0" to denote "02:34"
# Handles time cycles.
def parse_hour_time(t: float) -> datetime.time:
    t = f'{round(t):04}'         # ignore secs & zero pad
    h = f'{int(t[:2]) % 24:02}'  # loop & accumulate hrs
    m = f'{int(t[2:]) % 60:02}'  # loop & accumulate min
    return datetime.strptime(f"{h}{m}", "%H%M").time()


# Time series of weekly volume of infractions.
def plot_timeline_infractions(df: pd.DataFrame):
    ax = sns.lineplot(
        data=df.groupby([pd.Grouper(key='date_of_infraction', freq='W')]).size().to_frame('volume'),
    )
    ax.set_title('Weekly Infractions')
    ax.set_xlabel('date')
    ax.set_ylabel('volume')
    ax.tick_params(axis='x', rotation=90)
    ax.get_legend().set_visible(False)
    return ax


# Compare infraction cost by its frequency of occurrence
def plot_scatter_cost_by_freq(df: pd.DataFrame):
    df_tmp = df.groupby(['set_fine_amount', 'infraction_description']).size().to_frame('counts')
    ax = sns.scatterplot(
        data=df_tmp,
        x='set_fine_amount',
        y='counts',
        legend='full'
    )
    plt.title('Infraction Profitability')
    plt.xlabel('Cost $')
    plt.ylabel('Frequency')
    # for row in df_tmp.iterrows():
    #     idx, cost, infraction, size = row
    #     ax.text(cost + .02, infraction, str(size))

    return ax


# Tally specific infraction types, typically the top most ones & group them by (hour, day, month) period
def plot_histogram_infractions(df: pd.DataFrame, period: str, infractions: set):
    lbls = \
        list(range(24)) if period == 'hour' else \
        list(calendar.day_abbr) if period == 'day' else \
        list(calendar.month_abbr) if period == 'month' else \
        []

    df_tmp = (
        df['time_of_infraction'].apply(lambda t: t.hour) if period == 'hour' else
        df['date_of_infraction'].dt.dayofweek if period == 'day' else
        df['date_of_infraction'].dt.month if period == 'month' else
        df['date_of_infraction']
    ).to_frame(period)

    df_tmp = df_tmp.join(df['infraction_description'].apply(
        lambda infraction: infraction if infraction in infractions else "OTHER"
    ).to_frame('infraction_description'))
    df_tmp = df_tmp.groupby([period, 'infraction_description']).size().to_frame('counts')
    df_tmp = df_tmp.unstack(level=-1, fill_value=0)
    df_tmp = 100 * df_tmp / df_tmp.sum(axis='columns').sum()  # relative
    df_tmp.rename(
        index=dict(enumerate(lbls)),
        inplace=True,
        errors='ignore',
    )
    df_tmp.columns = df_tmp.columns.droplevel()
    return df_tmp


# Track relative volume of infraction by license plate of offender
def plot_prejudice_by_type(df: pd.DataFrame, infractions: set):
    df_tmp = df['province'].apply(
        lambda x: 'LOCAL' if x == 'ON' else 'FOREIGN'
    ).to_frame('offender')

    df_tmp = df_tmp.join(
        df['infraction_description'].apply(
            lambda x: x if x in infractions else 'OTHER'
        ).to_frame('infraction_description')
    )
    df_tmp = df_tmp.groupby(['infraction_description', 'offender']) \
        .size().to_frame('counts').unstack(level=-1, fill_value=0)
    df_tmp.columns = df_tmp.columns.droplevel()

    df_tmp = 100 * df_tmp / df_tmp.sum(axis='rows')
    return df_tmp


