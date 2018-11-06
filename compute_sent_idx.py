import pandas as pd
import numpy as np


df = pd.read_csv('stock_comments_analyzed.csv', parse_dates=['created_time'])
grouped = df['polarity'].groupby(df.created_time.dt.date)


def BI_Simple_func(row):
    pos = row[row == 1].count()
    neg = row[row == 0].count()

    return (pos-neg)/(pos+neg)

BI_Simple_index = grouped.apply(BI_Simple_func)


def BI_func(row):
    pos = row[row == 1].count()
    neg = row[row == 0].count()

    bi = np.log(1.0 * (1+pos) / (1+neg))

    return bi


BI_index = grouped.apply(BI_func)

sentiment_idx = pd.concat([BI_index.rename('BI'), BI_Simple_index.rename('BI_Simple')], axis=1)

quotes = pd.read_csv('./data/sh000001.csv', parse_dates=['date'])
quotes.set_index('date', inplace=True)

sentiment_idx.index = pd.to_datetime(sentiment_idx.index)
merged = pd.merge(sentiment_idx, quotes, how='left', left_index=True, right_index=True)

merged.fillna(method='ffill', inplace=True)
merged['BI_MA'] = merged['BI'].rolling(window=10, center=False).mean()
merged['BI_Simple_MA'] = merged['BI_Simple'].rolling(window=10, center=False).mean()


merged.to_csv('merged_sentiment_idx.csv')
