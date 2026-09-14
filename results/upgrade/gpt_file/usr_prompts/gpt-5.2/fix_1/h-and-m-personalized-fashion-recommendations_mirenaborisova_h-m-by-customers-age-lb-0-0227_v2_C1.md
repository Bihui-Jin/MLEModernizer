# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
The training data is the purchase history of customers across time. The task is to predict what articles each customer will purchase in the 7-day period immediately after the training data ends.

## Metric
Mean Average Precision @ 12 (MAP@12):

$$
\text{MAP@12}=\frac{1}{U} \sum_{u=1}^U \frac{1}{\min (m, 12)} \sum_{k=1}^{\min (n, 12)} P(k) \times \text{rel}(k)
$$

where $U$ is the number of customers, $P(k)$ is the precision at cutoff $k, n$ is the number predictions per customer, $m$ is the number of ground truth values per customer, and $\text{rel}(k)$ is an indicator function equaling 1 if the item at rank $k$ is a relevant (correct) label, zero otherwise.

You must make predictions for all `customer_id` values found in the sample submission. All customers who made purchases during the test period are scored, regardless of whether they had purchase history in the training data.

## Submission Format
For each `customer_id` observed in the training data, you may predict up to 12 labels for the `article_id`, which is the predicted items a customer will buy in the next 7-day period after the training time period. The file should contain a header and have the following format:

```
customer_id,prediction
00000dba,0706016001 0706016002 0372860001 ...
0000423b,0706016001 0706016002 0372860001 ...
...
```

## Dataset
- **images/** - a folder of images corresponding to each `article_id`; images are placed in subfolders starting with the first three digits of the `article_id`; note, not all `article_id` values have a corresponding image.
- **articles.csv** - detailed metadata for each `article_id` available for purchase
- **customers.csv** - metadata for each `customer_id` in dataset
- **sample_submission.csv** - a sample submission file in the correct format
- **transactions_train.csv** - the training data, consisting of the purchases each customer for each date, as well as additional information. Duplicate rows correspond to multiple purchases of the same item. Your task is to predict the `article_id`s each customer will purchase during the 7-day period immediately after the training data period.

# 2. Python version

3.10

# 3. Installed packages

cudf-cu12==25.2.2
cudf-polars-cu12==25.6.0
dask-cudf-cu12==25.2.2
geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
libcudf-cu12==25.2.2
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pylibcudf-cu12==25.2.2
seaborn==0.12.2
sklearn-pandas==2.2.0
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            articles.csv (105543 lines)
            articles.csv.zip (4.4 MB)
            customers.csv (1371981 lines)
            customers.csv.zip (102.4 MB)
            description.md (74 lines)
            images.zip (30.0 GB)
            sample_submission.csv (1371981 lines)
            sample_submission.csv.zip (53.3 MB)
            transactions_train.csv (31521961 lines)
            transactions_train.csv.zip (604.1 MB)
            h-and-m-personalized-fashion-recommendations/
                articles.csv (105543 lines)
                articles.csv.zip (4.4 MB)
                ... and 8 other files
                h-and-m-personalized-fashion-recommendations/
                images/
                    010/
                        0108775015.jpg (154.6 kB)
                        0108775044.jpg (106.7 kB)
                        ... and 1 other files
                    011/
                        0110065001.jpg (148.6 kB)
                        0110065002.jpg (85.7 kB)
                        ... and 18 other files
                    ... and 84 other folders
            images/
                010/
                    0108775015.jpg (154.6 kB)
                    0108775044.jpg (106.7 kB)
                    ... and 1 other files
                011/
                    0110065001.jpg (148.6 kB)
                    0110065002.jpg (85.7 kB)
                    ... and 18 other files
                ... and 84 other folders
        input/
            articles.csv (105543 lines)
            articles.csv.zip (4.4 MB)
            customers.csv (1371981 lines)
            customers.csv.zip (102.4 MB)
            description.md (74 lines)
            images.zip (30.0 GB)
            sample_submission.csv (1371981 lines)
            sample_submission.csv.zip (53.3 MB)
            transactions_train.csv (31521961 lines)
            transactions_train.csv.zip (604.1 MB)
            h-and-m-personalized-fashion-recommendations/
                articles.csv (105543 lines)
                articles.csv.zip (4.4 MB)
                ... and 8 other files
                h-and-m-personalized-fashion-recommendations/
                images/
                    010/
                        0108775015.jpg (154.6 kB)
                        0108775044.jpg (106.7 kB)
                        ... and 1 other files
                    011/
                        0110065001.jpg (148.6 kB)
                        0110065002.jpg (85.7 kB)
                        ... and 18 other files
                    ... and 84 other folders
            images/
                010/
                    0108775015.jpg (154.6 kB)
                    0108775044.jpg (106.7 kB)
                    ... and 1 other files
                011/
                    0110065001.jpg (148.6 kB)
                    0110065002.jpg (85.7 kB)
                    ... and 18 other files
                ... and 84 other folders
        working/
            h-and-m-personalized-fashion-recommendations/
                articles.csv (105543 lines)
                articles.csv.zip (4.4 MB)
                ... and 8 other files
                h-and-m-personalized-fashion-recommendations/
                images/
                    010/
                        0108775015.jpg (154.6 kB)
                        0108775044.jpg (106.7 kB)
                        ... and 1 other files
                    011/
                        0110065001.jpg (148.6 kB)
                        0110065002.jpg (85.7 kB)
                        ... and 18 other files
                    ... and 84 other folders
```

-> data/articles.csv has 105542 rows and 25 columns.
The columns are: article_id, product_code, prod_name, product_type_no, product_type_name, product_group_name, graphical_appearance_no, graphical_appearance_name, colour_group_code, colour_group_name, perceived_colour_value_id, perceived_colour_value_name, perceived_colour_master_id, perceived_colour_master_name, department_no... and 10 more columns

-> data/customers.csv has 1371980 rows and 7 columns.
The columns are: customer_id, FN, Active, club_member_status, fashion_news_frequency, age, postal_code

-> data/h-and-m-personalized-fashion-recommendations/articles.csv has 105542 rows and 25 columns.
The columns are: article_id, product_code, prod_name, product_type_no, product_type_name, product_group_name, graphical_appearance_no, graphical_appearance_name, colour_group_code, colour_group_name, perceived_colour_value_id, perceived_colour_value_name, perceived_colour_master_id, perceived_colour_master_name, department_no... and 10 more columns

-> data/h-and-m-personalized-fashion-recommendations/customers.csv has 1371980 rows and 7 columns.
The columns are: customer_id, FN, Active, club_member_status, fashion_news_frequency, age, postal_code

-> data/h-and-m-personalized-fashion-recommendations/sample_submission.csv has 1371980 rows and 2 columns.
The columns are: customer_id, prediction

-> data/h-and-m-personalized-fashion-recommendations/transactions_train.csv has 31521960 rows and 5 columns.
The columns are: t_dat, customer_id, article_id, price, sales_channel_id

-> data/sample_submission.csv has 1371980 rows and 2 columns.
The columns are: customer_id, prediction

-> data/transactions_train.csv has 31521960 rows and 5 columns.
The columns are: t_dat, customer_id, article_id, price, sales_channel_id

-> (stopped after 10 files for performance)

# 5. Target score

0.02244

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import sys
import warnings
warnings.filterwarnings('ignore')
import time
import os
import copy
import gc
import re
import random
import pickle
import cudf

from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 10000)

import seaborn as sns
sns.set()

from pandas.io.json import json_normalize
from pprint import pprint
from pathlib import Path
from tqdm import tqdm
tqdm.pandas()
from collections import Counter
from datetime import datetime, timedelta


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_10/3776904508.py in <cell line: 0>()
     22 sns.set()
     23 
---> 24 from pandas.io.json import json_normalize
     25 from pprint import pprint
     26 from pathlib import Path

ImportError: cannot import name 'json_normalize' from 'pandas.io.json' (/usr/local/lib/python3.11/dist-packages/pandas/io/json/__init__.py)

## === cell 1
DEBUG = False
PATH_INPUT = r'../input/h-and-m-personalized-fashion-recommendations/'


## === cell 2
COUNT = 1

ORDER = 0
N = 12


## === cell 3
def display_df(df, head=3):
    print(f'SHAPE: {df.shape}\n')
    display(df.head(head))


## === cell 4
def pre_increment(name, local={}):
    if name in local:
        local[name] += 1
        
        return local[name]
    
    globals()[name] += 1
    
    return globals()[name]


## === cell 5
def info_df(df, count, order):
    if count:
        try:
            name = [x for x in globals() if globals()[x] is df][0]
        except IndexError:
            name = ''
        
        order = pre_increment('order')   
        print('=' * 30)
        print(f'{order} INFO_DF {name}:\n')
        display_df(df)


## === cell 6
articles_df = cudf.read_csv(PATH_INPUT + 'articles.csv', 
                            usecols=['article_id', 
                                     'product_group_name', 
                                     'perceived_colour_master_name'])
display_df(articles_df)


## === cell 7
customers_df = cudf.read_csv(PATH_INPUT + 'customers.csv',
                             usecols=['customer_id', 'age'])
display_df(customers_df)


## === cell 8
customers_df = customers_df.to_pandas()
bin_list = [-1, 19, 29, 39, 49, 59, 69, 119]
customers_df['age_bins'] = pd.cut(customers_df['age'], bin_list)

display_df(customers_df)


## === cell 9
age_missing = customers_df[customers_df['age_bins'].isnull()].shape[0]
age_missing


## === cell 10
transactions_df = cudf.read_csv(PATH_INPUT + 'transactions_train.csv',
                                usecols=['t_dat', 'customer_id', 'article_id'],
                                dtype={'t_dat': 'string',
                                       'customer_id': 'string',
                                       'article_id': 'int32'})
transactions_df['t_dat'] = cudf.to_datetime(transactions_df['t_dat'])
transactions_df.set_index('t_dat', inplace=True)

display_df(transactions_df)


## === cell 11
recent_df = transactions_df.loc['2020-09-01':'2020-09-21']

display_df(recent_df)


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_10/3785786911.py in <cell line: 0>()
----> 1 recent_df = transactions_df.loc['2020-09-01':'2020-09-21']
      2 
      3 display_df(recent_df)

/usr/local/lib/python3.11/dist-packages/cudf/core/dataframe.py in __getitem__(self, arg)
    138             if not isinstance(arg, tuple):
    139                 arg = (arg, slice(None))
--> 140             return self._getitem_tuple_arg(arg)
    141 
    142     def __setitem__(self, key, value):

/usr/local/lib/python3.11/dist-packages/cudf/utils/performance_tracking.py in wrapper(*args, **kwargs)
     49                     )
     50                 )
---> 51             return func(*args, **kwargs)
     52 
     53     return wrapper

/usr/local/lib/python3.11/dist-packages/cudf/core/dataframe.py in _getitem_tuple_arg(self, arg)
    275         else:
    276             if isinstance(arg[0], slice):
--> 277                 out = _get_label_range_or_mask(
    278                     columns_df.index, arg[0].start, arg[0].stop, arg[0].step
    279                 )

/usr/local/lib/python3.11/dist-packages/cudf/core/indexed_frame.py in _get_label_range_or_mask(index, start, stop, step)
    201                 return slice(start_loc, stop_loc)
    202             else:
--> 203                 raise KeyError(
    204                     "Value based partial slicing on non-monotonic "
    205                     "DatetimeIndexes with non-existing keys is not allowed.",

KeyError: 'Value based partial slicing on non-monotonic DatetimeIndexes with non-existing keys is not allowed.'

## === cell 12
recent_df = recent_df.to_pandas()
recent_df = recent_df.merge(customers_df[['customer_id', 'age_bins']], on='customer_id', how='inner')

display_df(recent_df)


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/681558899.py in <cell line: 0>()
----> 1 recent_df = recent_df.to_pandas()
      2 recent_df = recent_df.merge(customers_df[['customer_id', 'age_bins']], on='customer_id', how='inner')
      3 
      4 display_df(recent_df)

NameError: name 'recent_df' is not defined

## === cell 13
recent_df = recent_df.groupby(['age_bins', 'article_id']).count().reset_index()\
    .rename(columns={'customer_id': 'counts'})

display_df(recent_df)


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/2676331017.py in <cell line: 0>()
----> 1 recent_df = recent_df.groupby(['age_bins', 'article_id']).count().reset_index()\
      2     .rename(columns={'customer_id': 'counts'})
      3 
      4 display_df(recent_df)

NameError: name 'recent_df' is not defined

## === cell 14
bins_unique_list = recent_df['age_bins'].unique().tolist()
bins_unique_list


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/3724590386.py in <cell line: 0>()
----> 1 bins_unique_list = recent_df['age_bins'].unique().tolist()
      2 bins_unique_list

NameError: name 'recent_df' is not defined

## === cell 15
bins_unique_list[0]


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/2311883249.py in <cell line: 0>()
----> 1 bins_unique_list[0]

NameError: name 'bins_unique_list' is not defined

## === cell 16
ages_article_id = {}
count = COUNT
order = ORDER
for bins_unique in bins_unique_list:
    temp_df = recent_df[recent_df['age_bins'] == bins_unique]
    info_df(temp_df, count,order)
    
    temp_df = temp_df.sort_values(by='counts', ascending=False)
    info_df(temp_df, count, order)
    
    ages_article_id[bins_unique] = temp_df.head(100)['article_id'].values.tolist()
    
    count = 0


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/210998673.py in <cell line: 0>()
      2 count = COUNT
      3 order = ORDER
----> 4 for bins_unique in bins_unique_list:
      5     temp_df = recent_df[recent_df['age_bins'] == bins_unique]
      6     info_df(temp_df, count,order)

NameError: name 'bins_unique_list' is not defined

## === cell 17
from itertools import islice

for key, value in islice(ages_article_id.items(), 3):
    print(f'{key} {len(value)}')


## === cell 18
topcustcnt_byage_df = pd.DataFrame([ages_article_id])

display_df(topcustcnt_byage_df)


## === cell 19
topcustcnt_byage_df = pd.DataFrame([ages_article_id]).T.rename(columns={0: 'top_100'})

display_df(topcustcnt_byage_df)


## === cell 20
for i in topcustcnt_byage_df.index:
    topcustcnt_byage_df[i] = [len(set(topcustcnt_byage_df.at[i, 'top_100']) & \
                                  set(topcustcnt_byage_df.at[j, 'top_100'])) / 100 for j in topcustcnt_byage_df.index]
    
display_df(topcustcnt_byage_df)


## === cell 21
topcustcnt_byage_df = topcustcnt_byage_df.drop(columns='top_100')

display_df(topcustcnt_byage_df, head=10)


## === cell 22
plt.figure(figsize=(10, 6))
sns.heatmap(topcustcnt_byage_df, cmap='winter', annot=True, cbar=False)


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_10/1520051716.py in <cell line: 0>()
      1 plt.figure(figsize=(10, 6))
----> 2 sns.heatmap(topcustcnt_byage_df, cmap='winter', annot=True, cbar=False)

/usr/local/lib/python3.11/dist-packages/seaborn/matrix.py in heatmap(data, vmin, vmax, cmap, center, robust, annot, fmt, annot_kws, linewidths, linecolor, cbar, cbar_kws, cbar_ax, square, xticklabels, yticklabels, mask, ax, **kwargs)
    444     """
    445     # Initialize the plotter object
--> 446     plotter = _HeatMapper(data, vmin, vmax, cmap, center, robust, annot, fmt,
    447                           annot_kws, cbar, cbar_kws, xticklabels,
    448                           yticklabels, mask)

/usr/local/lib/python3.11/dist-packages/seaborn/matrix.py in __init__(self, data, vmin, vmax, cmap, center, robust, annot, fmt, annot_kws, cbar, cbar_kws, xticklabels, yticklabels, mask)
    161 
    162         # Determine good default values for the colormapping
--> 163         self._determine_cmap_params(plot_data, vmin, vmax,
    164                                     cmap, center, robust)
    165 

/usr/local/lib/python3.11/dist-packages/seaborn/matrix.py in _determine_cmap_params(self, plot_data, vmin, vmax, cmap, center, robust)
    200                 vmin = np.nanpercentile(calc_data, 2)
    201             else:
--> 202                 vmin = np.nanmin(calc_data)
    203         if vmax is None:
    204             if robust:

/usr/local/lib/python3.11/dist-packages/numpy/lib/nanfunctions.py in nanmin(a, axis, out, keepdims, initial, where)
    341         # Fast, but not safe for subclasses of ndarray, or object arrays,
    342         # which do not implement isnan (gh-9009), or fmin correctly (gh-8975)
--> 343         res = np.fmin.reduce(a, axis=axis, out=out, **kwargs)
    344         if np.isnan(res).any():
    345             warnings.warn("All-NaN slice encountered", RuntimeWarning,

ValueError: zero-size array to reduction operation fmin which has no identity

## === cell 23
bins_unique_list


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/4218793739.py in <cell line: 0>()
----> 1 bins_unique_list

NameError: name 'bins_unique_list' is not defined

## === cell 24
bins_unique_list = customers_df['age_bins'].unique().tolist()
bins_unique_list


## === cell 25
count = COUNT
order = ORDER
for bin_unique in bins_unique_list:
    
    df = cudf.read_csv('../input/h-and-m-personalized-fashion-recommendations/transactions_train.csv',
                       usecols=['t_dat', 'customer_id', 'article_id'],
                       dtype={'t_dat': 'string',
                              'customer_id': 'string',
                              'article_id': 'int32'})
    info_df(df, count, order)
    
    if str(bin_unique) == 'nan':
        temp_customers_df = customers_df[customers_df['age_bins'].isnull()]
    else:
        temp_customers_df = customers_df[customers_df['age_bins'] == bin_unique]
        
    temp_customers_df = temp_customers_df.drop(['age_bins'], axis=1)
    temp_customers_df = cudf.from_pandas(temp_customers_df)
    info_df(temp_customers_df, count, order)
    
    df = df.merge(temp_customers_df[['customer_id', 'age']], on='customer_id', how='inner')
    info_df(df, count, order)
    
    print(f'TRANSACTION SHAPE FOR SCOPE {bin_unique}: {df.shape}\n')
    
    df['customer_id'] = df['customer_id'].str[-16:].str.hex_to_int().astype('int64')
    df['t_dat'] = cudf.to_datetime(df['t_dat'])
    
    last_date_train = df['t_dat'].max()
    
    if count:
        print('=' * 30 + f'\n4 LAST_DATE_TRAIN:\n{last_date_train}')
    
    tmp = df[['t_dat']].copy().to_pandas()
    tmp['weekday'] = tmp['t_dat'].dt.dayofweek
    info_df(tmp, count, order)
    
    tmp['t_dat_shiftday'] = tmp['t_dat'] - pd.TimedeltaIndex(tmp['weekday'] - 1, unit='D')
    info_df(tmp, count, order)
    
    tmp.loc[tmp['weekday'] >= 2, 't_dat_shiftday'] = tmp.loc[tmp['weekday'] >= 2, 't_dat_shiftday'] + \
        pd.TimedeltaIndex(np.ones(len(tmp.loc[tmp['weekday'] >= 2])) * 7, unit='D')
    info_df(tmp, count, order)
    
    df['t_dat_shiftday'] = tmp['t_dat_shiftday'].values
    info_df(df, count, order)
    
    weekly_sales = df.drop('customer_id', axis=1).groupby(['t_dat_shiftday', 'article_id']).count().reset_index()
    info_df(weekly_sales, count, order)
    
    weekly_sales = weekly_sales.rename(columns={'t_dat': 'count'})
    info_df(weekly_sales, count, order)
    
    df = df.merge(weekly_sales, on=['t_dat_shiftday', 'article_id'], how='left')
    info_df(df, count, order)
    
    weekly_sales = weekly_sales.reset_index().set_index('article_id')
    info_df(weekly_sales, count, order)
    
    df = df.merge(weekly_sales.loc[weekly_sales['t_dat_shiftday'] == last_date_train, ['count']], 
                 on='article_id',
                 suffixes=('', '_targ'))
    info_df(df, count, order)
    
    df['count_targ'].fillna(0, inplace=True)
    del weekly_sales
    
    df['quotient'] = df['count_targ'] / df['count']
    info_df(df, count, order)
    
    target_sales = df.drop('customer_id', axis=1).groupby('article_id')['quotient'].sum()
    info_df(target_sales, count, order)
    
    general_pred = target_sales.nlargest(N).index.to_pandas().tolist()
    general_pred = ['0' + str(article_id) for article_id in general_pred]
    general_pred_str = ' '.join(general_pred)
    
    del target_sales
    
    purchase_dict = {}
    tmp = df.copy().to_pandas()
    info_df(tmp, count, order)
    
    tmp['x'] = ((last_date_train - tmp['t_dat']) / np.timedelta64(1, 'D')).astype(int)
    info_df(tmp, count, order)
    
    tmp['dummy_1'] = 1
    tmp['x'] = tmp[['x', 'dummy_1']].max(axis=1)
    info_df(tmp, count, order)
    
    a, b, c, d = 2.5e4, 1.5e5, 2e-1, 1e3
    tmp['y'] = a / np.sqrt(tmp['x']) + b * np.exp(-c * tmp['x']) - d
    info_df(tmp, count, order)
    
    tmp['dummy_0'] = 0
    tmp['y'] = tmp[['y', 'dummy_0']].max(axis=1)
    tmp['value'] = tmp['quotient'] * tmp['y']
    info_df(tmp, count, order)
    
    tmp = tmp.groupby(['customer_id', 'article_id']).agg({'value': 'sum'})
    info_df(tmp, count, order)
    
    tmp = tmp.reset_index()
    
    tmp = tmp.loc[tmp['value'] > 0]
    info_df(tmp, count, order)
    
    tmp['rank'] = tmp.groupby('customer_id')['value'].rank('dense', ascending=False)
    info_df(tmp, count, order)
    
    tmp = tmp.loc[tmp['rank'] <= 12]
    info_df(tmp, count, order)
    
    purchase_df = tmp.sort_values(['customer_id', 'value'], ascending=False).reset_index(drop=True)
    info_df(purchase_df, count, order)
    
    purchase_df['prediction'] = '0' + purchase_df['article_id'].astype('str') + ' '
    info_df(purchase_df, count, order)
    
    purchase_df = purchase_df.groupby('customer_id').agg({'prediction': sum}).reset_index()
    info_df(purchase_df, count, order)
    
    purchase_df = cudf.DataFrame(purchase_df)
    
    sub = cudf.read_csv('../input/h-and-m-personalized-fashion-recommendations/sample_submission.csv',
                        usecols=['customer_id'],
                        dtype={'customer_id': 'string'})
    info_df(sub, count, order)
    
    num_customers = sub.shape[0]
    sub = sub.merge(temp_customers_df[['customer_id', 'age']], on='customer_id', how='inner')
    info_df(sub, count, order)
    
    sub['customer_id_2'] = sub['customer_id'].str[-16:].str.hex_to_int().astype('int64')
    info_df(sub, count, order)
    
    sub = sub.merge(purchase_df,
                    left_on='customer_id_2',
                    right_on='customer_id',
                    how='left',
                    suffixes={'', '_ignored'})
    info_df(sub, count, order)
    
    sub = sub.to_pandas()
    sub['prediction'] = sub['prediction'].fillna(general_pred_str)
    sub['prediction'] = sub['prediction'] + ' ' + general_pred_str
    sub['prediction'] = sub['prediction'].str.strip()
    sub['prediction'] = sub['prediction'].str[:131]
    
    sub = sub[['customer_id', 'prediction']]
    sub.to_csv(f'submission_{str(bin_unique)}.csv', index=False)
    info_df(sub, count, order)
    
    print(f'SAVED PREDICTION FOR {bin_unique}, SHAPE: {sub.shape}\n')
    print('-' * 50)
    
    count = 0
    
print('FINISHED')
print('=' * 50)


## === cell 26
for i, bin_unique in enumerate(bins_unique_list):
    temp_df = cudf.read_csv(f'submission_{str(bin_unique)}.csv')
    if i == 0:
        sub_df = temp_df
    else:
        sub_df = cudf.concat([sub_df, temp_df], axis=0)
        
assert sub_df.shape[0] == num_customers,\
    f'SUB_DF ROWS NUMBER IS NOT CORRECT {sub_df.shape[0]} VS {num_customers}'
    
sub_df.to_csv('by_cust_age__.csv', index=False)

display_df(sub_df)


## === cell 27
check_df = cudf.read_csv('./by_cust_age__.csv')

display_df(check_df)


## --- ERROR in outputing the csv:
Invalid submission: Submission customer_id must be a superset of answers customer_id
