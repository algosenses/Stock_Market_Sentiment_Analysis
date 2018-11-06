import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('merged_sentiment_idx.csv', parse_dates=['created_time'])
df.set_index(df.created_time, inplace=True)
df = df.loc['2017-4-15':'2018-4-15']

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(df.index, df['BI_MA'], color='#1F77B4', linestyle=':')
ax2.plot(df.index, df['close'], color='#4B73B1')
ax1.set_xlabel('日期')
ax1.set_ylabel('BI指标')
ax2.set_ylabel('上证指数')

plt.show()
