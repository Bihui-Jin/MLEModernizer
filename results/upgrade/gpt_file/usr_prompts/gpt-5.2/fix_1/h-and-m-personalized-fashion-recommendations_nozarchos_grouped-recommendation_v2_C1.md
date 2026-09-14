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

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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

0.00544

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


## === cell 1
general_path = '../input/h-and-m-personalized-fashion-recommendations/'


## === cell 2
articles = pd.read_csv(general_path + 'articles.csv')
customers = pd.read_csv(general_path + 'customers.csv')
sample_submission = pd.read_csv(general_path + 'sample_submission.csv')
transactions_train = pd.read_csv(general_path + 'transactions_train.csv')


## === cell 3
transactions_train.info()


## === cell 4
transactions_train.sample(5)


## === cell 5
transactions_train['article_id'] = '0' + transactions_train['article_id'].astype(str)


## === cell 6
articles['article_id'] = '0' + articles['article_id'].astype(str)


## === cell 7
transactions_train['t_dat'] = pd.to_datetime(transactions_train['t_dat'], format='%Y-%m-%d')


## === cell 8
maximum_history_date = transactions_train['t_dat'].max()


## === cell 9
print(f'We have {len(customers)} unique customers')
print(f'And {len(customers["postal_code"].unique())} unique locations')


## === cell 10
customer_per_location = customers.pivot_table(index='postal_code', aggfunc={'customer_id': 'count'})


## === cell 11
customer_per_location.sort_values(by=['customer_id'])


## === cell 12
customers['location'] = 'other'
customers.loc[customers['postal_code'] == '2c29ae653a9282cce4151bd87643c907644e09541abc28ae87dea0d1f6603b1c',
          'location'] = 'big_city'


## === cell 13
transactions_train['price'].hist(bins=30, figsize=(16, 5))
plt.title('Prices')
plt.show()


## === cell 14
articles.nunique()


## === cell 15
articles.sample(1)


## === cell 16
articles['index_name'].unique()


## === cell 17
customers['age_group'] = '<20'
customers.loc[customers['age'] > 20, 'age_group'] = '20-45'
customers.loc[customers['age'] > 45, 'age_group'] = '>45'


## === cell 18
def aggregate_to_list(row):
    return [i for i in row]


## === cell 19
transactions_train = pd.merge(transactions_train,
                              articles[['article_id', 'index_name']],
                              on='article_id',
                              how='left')


## === cell 20
transactions_train = pd.merge(transactions_train,
                              customers[['customer_id', 'age_group']],
                              on='customer_id',
                              how='left')


## === cell 21
sample_submission = customers[['customer_id', 'age_group']]
sample_submission = pd.merge(sample_submission,
                             transactions_train.pivot_table(index=['customer_id', 'index_name'],
                                                            aggfunc={'article_id': ['count', 
                                                                                    aggregate_to_list]}
                                                           ).reset_index(),
                             on='customer_id',
                             how='left')

sample_submission.columns = ['customer_id', 'age_group', 'index_name', 'article_purchased', 
                             'article_id_count']


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
MergeError                                Traceback (most recent call last)
/tmp/ipykernel_11/2294336074.py in <cell line: 0>()
      1 sample_submission = customers[['customer_id', 'age_group']]
----> 2 sample_submission = pd.merge(sample_submission,
      3                              transactions_train.pivot_table(index=['customer_id', 'index_name'],
      4                                                             aggfunc={'article_id': ['count', 
      5                                                                                     aggregate_to_list]}

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in merge(left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
    168         )
    169     else:
--> 170         op = _MergeOperation(
    171             left_df,
    172             right_df,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in __init__(self, left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, indicator, validate)
    782                 f"{_right.columns.nlevels} on the right)"
    783             )
--> 784             raise MergeError(msg)
    785 
    786         self.left_on, self.right_on = self._validate_left_right_on(left_on, right_on)

MergeError: Not allowed to merge between different levels. (1 levels on the left, 2 on the right)

## === cell 22
articles_sale_interval = transactions_train.pivot_table(index='article_id', aggfunc={'t_dat': ['min', 'max']}).reset_index()
articles_sale_interval.columns = [i[0] if (pd.isna(i[1]) or i[1] == '') else i[0] + '_' + i[1] for i in articles_sale_interval.columns]
long_time_have_not_sold = articles_sale_interval[articles_sale_interval['t_dat_max'] < maximum_history_date - pd.Timedelta('30 days')]


## === cell 23
len_before = len(transactions_train)
transactions_train = transactions_train[~transactions_train['article_id'].isin(long_time_have_not_sold['article_id'])]
print('Removed {:.2%} of articles'.format(1 - (len(transactions_train) / len_before)))


## === cell 24
transactions_train = transactions_train[transactions_train['t_dat'] > 
                                        transactions_train['t_dat'].max() - pd.Timedelta('21 days')]


## === cell 25
group_recomendation = transactions_train.pivot_table(index=['age_group', 'index_name', 'article_id'],
                                                     aggfunc={'customer_id': 'count'}).reset_index()


## === cell 26
group_recomendation.sort_values(by=['age_group', 'index_name', 'customer_id'], ascending=False, 
                                inplace=True)
group_recomendation.query('customer_id > 2', inplace=True)
group_recomendation.rename({'customer_id': 'article_raiting'}, inplace=True)


## === cell 27
sample_submission = pd.merge(sample_submission,
                             group_recomendation.pivot_table(index=['age_group', 'index_name'],
                                                             aggfunc={'article_id': aggregate_to_list}
                                                            ).reset_index(),
                             on=['age_group', 'index_name'],
                             how='left')
sample_submission.rename(columns={'article_id' : 'top_article_id'}, inplace=True)


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1467961919.py in <cell line: 0>()
----> 1 sample_submission = pd.merge(sample_submission,
      2                              group_recomendation.pivot_table(index=['age_group', 'index_name'],
      3                                                              aggfunc={'article_id': aggregate_to_list}
      4                                                             ).reset_index(),
      5                              on=['age_group', 'index_name'],

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in merge(left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
    168         )
    169     else:
--> 170         op = _MergeOperation(
    171             left_df,
    172             right_df,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in __init__(self, left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, indicator, validate)
    792             left_drop,
    793             right_drop,
--> 794         ) = self._get_merge_keys()
    795 
    796         if left_drop:

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in _get_merge_keys(self)
   1308                         #  the latter of which will raise
   1309                         lk = cast(Hashable, lk)
-> 1310                         left_keys.append(left._get_label_or_level_values(lk))
   1311                         join_names.append(lk)
   1312                     else:

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _get_label_or_level_values(self, key, axis)
   1909             values = self.axes[axis].get_level_values(key)._values
   1910         else:
-> 1911             raise KeyError(key)
   1912 
   1913         # Check for duplicates

KeyError: 'index_name'

## === cell 28
sample_submission['group_possibility'] = sample_submission.groupby('customer_id'
                                                                  )['article_id_count'].transform('sum')
sample_submission.query('group_possibility != 0', inplace=True)
sample_submission['article_id_count'] /= sample_submission['group_possibility']
sample_submission['article_id_count'] *= 12
sample_submission['article_id_count'] = sample_submission['article_id_count'].astype(float).round().astype('Int64')
sample_submission.rename(columns={'article_id_count': 'qty_to_recomend'}, inplace=True)


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/758037902.py in <cell line: 0>()
----> 1 sample_submission['group_possibility'] = sample_submission.groupby('customer_id'
      2                                                                   )['article_id_count'].transform('sum')
      3 sample_submission.query('group_possibility != 0', inplace=True)
      4 sample_submission['article_id_count'] /= sample_submission['group_possibility']
      5 sample_submission['article_id_count'] *= 12

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/generic.py in __getitem__(self, key)
   1949                 "Use a list instead."
   1950             )
-> 1951         return super().__getitem__(key)
   1952 
   1953     def _gotitem(self, key, ndim: int, subset=None):

/usr/local/lib/python3.11/dist-packages/pandas/core/base.py in __getitem__(self, key)
    242         else:
    243             if key not in self.obj:
--> 244                 raise KeyError(f"Column not found: {key}")
    245             ndim = self.obj[key].ndim
    246             return self._gotitem(key, ndim=ndim)

KeyError: 'Column not found: article_id_count'

## === cell 29
sample_submission.info()


## === cell 30
sample_submission = sample_submission[sample_submission['qty_to_recomend'] != 0]


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'qty_to_recomend'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/730055301.py in <cell line: 0>()
----> 1 sample_submission = sample_submission[sample_submission['qty_to_recomend'] != 0]

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'qty_to_recomend'

## === cell 31
def articles_remove(row):
    bought = row['article_purchased']
    recomendation = []
    if type(bought) != list:
        recomendation.append(row['top_article_id'])
    i = 0
    while len(recomendation) < row['qty_to_recomend'] and i < len(row['top_article_id']):
        current_acticle = row['top_article_id'][i]
        i += 1
        if current_acticle in bought:
            continue
            
        if i == len(row['top_article_id']):
            break
        recomendation.append(current_acticle)
    return recomendation


## === cell 32
sample_submission[sample_submission['customer_id'] == '0118ed570ff6ff085cde55a6e801c6861a4e9ff9a8d9e82ee36b33c8b3af8f59']


## === cell 33
sample_submission['recomendation'] = sample_submission.apply(articles_remove, axis=1)


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'article_purchased'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/4031271387.py in <cell line: 0>()
----> 1 sample_submission['recomendation'] = sample_submission.apply(articles_remove, axis=1)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in apply(self, func, axis, raw, result_type, args, by_row, engine, engine_kwargs, **kwargs)
  10372             kwargs=kwargs,
  10373         )
> 10374         return op.apply().__finalize__(self, method="apply")
  10375 
  10376     def map(

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply(self)
    914             return self.apply_raw(engine=self.engine, engine_kwargs=self.engine_kwargs)
    915 
--> 916         return self.apply_standard()
    917 
    918     def agg(self):

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_standard(self)
   1061     def apply_standard(self):
   1062         if self.engine == "python":
-> 1063             results, res_index = self.apply_series_generator()
   1064         else:
   1065             results, res_index = self.apply_series_numba()

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_series_generator(self)
   1079             for i, v in enumerate(series_gen):
   1080                 # ignore SettingWithCopy here in case the user mutates
-> 1081                 results[i] = self.func(v, *self.args, **self.kwargs)
   1082                 if isinstance(results[i], ABCSeries):
   1083                     # If we have a view on v, we need to make a copy because

/tmp/ipykernel_11/4292082698.py in articles_remove(row)
      1 def articles_remove(row):
----> 2     bought = row['article_purchased']
      3     recomendation = []
      4     if type(bought) != list:
      5         recomendation.append(row['top_article_id'])

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in __getitem__(self, key)
   1119 
   1120         elif key_is_scalar:
-> 1121             return self._get_value(key)
   1122 
   1123         # Convert generator to list before going through hashable part

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in _get_value(self, label, takeable)
   1235 
   1236         # Similar to Index.get_value, but we do not fall back to positional
-> 1237         loc = self.index.get_loc(label)
   1238 
   1239         if is_integer(loc):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'article_purchased'

## === cell 34
def lists_aggregate_to_list(row):
    output = []
    for i in row:

        output += i
    return output


## === cell 35
sample_submission = sample_submission.pivot_table(index='customer_id',
                                                  aggfunc={'recomendation': lists_aggregate_to_list}
                                                 ).reset_index()


## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/4015777016.py in <cell line: 0>()
----> 1 sample_submission = sample_submission.pivot_table(index='customer_id',
      2                                                   aggfunc={'recomendation': lists_aggregate_to_list}
      3                                                  ).reset_index()

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in pivot_table(self, values, index, columns, aggfunc, fill_value, margins, dropna, margins_name, observed, sort)
   9507         from pandas.core.reshape.pivot import pivot_table
   9508 
-> 9509         return pivot_table(
   9510             self,
   9511             values=values,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/pivot.py in pivot_table(data, values, index, columns, aggfunc, fill_value, margins, dropna, margins_name, observed, sort)
    100         return table.__finalize__(data, method="pivot_table")
    101 
--> 102     table = __internal_pivot_table(
    103         data,
    104         values,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/pivot.py in __internal_pivot_table(data, values, index, columns, aggfunc, fill_value, margins, dropna, margins_name, observed, sort)
    181             stacklevel=find_stack_level(),
    182         )
--> 183     agged = grouped.agg(aggfunc)
    184 
    185     if dropna and isinstance(agged, ABCDataFrame) and len(agged.columns):

/usr/local/lib/python3.11/dist-packages/pandas/core/groupby/generic.py in aggregate(self, func, engine, engine_kwargs, *args, **kwargs)
   1430 
   1431         op = GroupByApply(self, func, args=args, kwargs=kwargs)
-> 1432         result = op.agg()
   1433         if not is_dict_like(func) and result is not None:
   1434             # GH #52849

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in agg(self)
    188 
    189         if is_dict_like(func):
--> 190             return self.agg_dict_like()
    191         elif is_list_like(func):
    192             # we require a list, but not a 'str'

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in agg_dict_like(self)
    421         Result of aggregation.
    422         """
--> 423         return self.agg_or_apply_dict_like(op_name="agg")
    424 
    425     def compute_dict_like(

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in agg_or_apply_dict_like(self, op_name)
   1606             obj, "as_index", True, condition=hasattr(obj, "as_index")
   1607         ):
-> 1608             result_index, result_data = self.compute_dict_like(
   1609                 op_name, selected_obj, selection, kwargs
   1610             )

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in compute_dict_like(self, op_name, selected_obj, selection, kwargs)
    460         is_groupby = isinstance(obj, (DataFrameGroupBy, SeriesGroupBy))
    461         func = cast(AggFuncTypeDict, self.func)
--> 462         func = self.normalize_dictlike_arg(op_name, selected_obj, func)
    463 
    464         is_non_unique_col = (

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in normalize_dictlike_arg(self, how, obj, func)
    661             cols = Index(list(func.keys())).difference(obj.columns, sort=True)
    662             if len(cols) > 0:
--> 663                 raise KeyError(f"Column(s) {list(cols)} do not exist")
    664 
    665         aggregator_types = (list, tuple, dict)

KeyError: "Column(s) ['recomendation'] do not exist"

## === cell 36
unknown_customer_id = customers[~customers['customer_id'].isin(sample_submission['customer_id'])]


## === cell 37
unknown_customer_id = unknown_customer_id[['customer_id']]


## === cell 38
unknown_customer_id['prediction'] = ' '.join((transactions_train.pivot_table(index='article_id', 
                                                                             aggfunc={'customer_id': 
                                                                                      'count'})
                                                                .reset_index()
                                                                .sort_values(by='customer_id',
                                                                             ascending=False)
                                                )['article_id'].values[:12])


## === cell 39
sample_submission['prediction'] = sample_submission['recomendation'].str.join(' ')


## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'recomendation'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2717044569.py in <cell line: 0>()
----> 1 sample_submission['prediction'] = sample_submission['recomendation'].str.join(' ')

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'recomendation'

## === cell 40
sample_submission = pd.concat([sample_submission, unknown_customer_id])


## === cell 41
sample_submission.drop('recomendation', axis=1, inplace=True)


## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/762648384.py in <cell line: 0>()
----> 1 sample_submission.drop('recomendation', axis=1, inplace=True)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in drop(self, labels, axis, index, columns, level, inplace, errors)
   5579                 weight  1.0     0.8
   5580         """
-> 5581         return super().drop(
   5582             labels=labels,
   5583             axis=axis,

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in drop(self, labels, axis, index, columns, level, inplace, errors)
   4786         for axis, labels in axes.items():
   4787             if labels is not None:
-> 4788                 obj = obj._drop_axis(labels, axis, level=level, errors=errors)
   4789 
   4790         if inplace:

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _drop_axis(self, labels, axis, level, errors, only_slice)
   4828                 new_axis = axis.drop(labels, level=level, errors=errors)
   4829             else:
-> 4830                 new_axis = axis.drop(labels, errors=errors)
   4831             indexer = axis.get_indexer(new_axis)
   4832 

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in drop(self, labels, errors)
   7068         if mask.any():
   7069             if errors != "ignore":
-> 7070                 raise KeyError(f"{labels[mask].tolist()} not found in axis")
   7071             indexer = indexer[~mask]
   7072         return self.delete(indexer)

KeyError: "['recomendation'] not found in axis"

## === cell 42
sample_submission.to_csv('predict.csv', index=False)


## --- ERROR in outputing the csv:
Invalid submission: Submission DataFrame must have 'customer_id' and 'prediction' columns.
