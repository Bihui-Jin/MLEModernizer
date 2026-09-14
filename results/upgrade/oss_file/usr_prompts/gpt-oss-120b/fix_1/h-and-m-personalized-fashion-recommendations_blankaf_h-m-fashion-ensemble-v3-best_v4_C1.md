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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

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

0.0237733507340115

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import gc

## === cell 1
sub0 = pd.read_csv('../input/fashions/submission231.csv').sort_values('customer_id').reset_index(drop=True)
sub1 = pd.read_csv('../input/fashions/sub-age.csv').sort_values('customer_id').reset_index(drop=True) #227
sub2 = pd.read_csv('../input/fashions/sub-weekly.csv').sort_values('customer_id').reset_index(drop=True) #225
sub3 = pd.read_csv('../input/fashions/submission225.csv').sort_values('customer_id').reset_index(drop=True)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3112241280.py in <cell line: 0>()
----> 1 sub0 = pd.read_csv('../input/fashions/submission231.csv').sort_values('customer_id').reset_index(drop=True)
      2 sub1 = pd.read_csv('../input/fashions/sub-age.csv').sort_values('customer_id').reset_index(drop=True) #227
      3 sub2 = pd.read_csv('../input/fashions/sub-weekly.csv').sort_values('customer_id').reset_index(drop=True) #225
      4 sub3 = pd.read_csv('../input/fashions/submission225.csv').sort_values('customer_id').reset_index(drop=True)
      5 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: '../input/fashions/submission231.csv'

## === cell 2
sub0.columns = ['customer_id', 'prediction0']
sub0['prediction1'] = sub1['prediction']
sub0['prediction2'] = sub2['prediction']
sub0['prediction3'] = sub3['prediction'].astype(str)

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/529648108.py in <cell line: 0>()
----> 1 sub0.columns = ['customer_id', 'prediction0']
      2 sub0['prediction1'] = sub1['prediction']
      3 sub0['prediction2'] = sub2['prediction']
      4 sub0['prediction3'] = sub3['prediction'].astype(str)

NameError: name 'sub0' is not defined

## === cell 3
del sub1, sub2, sub3, 
gc.collect()
sub0.head()

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1400773299.py in <cell line: 0>()
----> 1 del sub1, sub2, sub3,
      2 gc.collect()
      3 sub0.head()

NameError: name 'sub1' is not defined

## === cell 4
def cust_blend(dt, W = [1,1,1,1]):
    REC = []
    REC.append(dt['prediction0'].split())
    REC.append(dt['prediction1'].split())
    REC.append(dt['prediction2'].split())
    REC.append(dt['prediction3'].split())
    res = {}
    for M in range(len(REC)):
        for n, v in enumerate(REC[M]):
            if v in res:
                res[v] += (W[M]/(n+1))
            else:
                res[v] = (W[M]/(n+1))
    res = list(dict(sorted(res.items(), key=lambda item: -item[1])).keys())
    return ' '.join(res[:12])

sub0['prediction'] = sub0.apply(cust_blend, W = [0.25, 0.25, 0.25, 0.25], axis=1)
sub0.head()

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3318222656.py in <cell line: 0>()
     15     return ' '.join(res[:12])
     16 
---> 17 sub0['prediction'] = sub0.apply(cust_blend, W = [0.25, 0.25, 0.25, 0.25], axis=1)
     18 sub0.head()

NameError: name 'sub0' is not defined

## === cell 5
del sub0['prediction0']
del sub0['prediction1']
del sub0['prediction2']
del sub0['prediction3']
gc.collect()

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3817733294.py in <cell line: 0>()
----> 1 del sub0['prediction0']
      2 del sub0['prediction1']
      3 del sub0['prediction2']
      4 del sub0['prediction3']
      5 gc.collect()

NameError: name 'sub0' is not defined

## === cell 6
sub00 = pd.read_csv('../input/fashions/submission224.csv').sort_values('customer_id').reset_index(drop=True)
sub1 = pd.read_csv('../input/fashions/sub-lstm.csv').sort_values('customer_id').reset_index(drop=True) #221
sub2 = pd.read_csv('../input/fashions/sub-time.csv').sort_values('customer_id').reset_index(drop=True) #220
sub3 = pd.read_csv('../input/fashions/sub-decay.csv').sort_values('customer_id').reset_index(drop=True) #217



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2577888004.py in <cell line: 0>()
----> 1 sub00 = pd.read_csv('../input/fashions/submission224.csv').sort_values('customer_id').reset_index(drop=True)
      2 sub1 = pd.read_csv('../input/fashions/sub-lstm.csv').sort_values('customer_id').reset_index(drop=True) #221
      3 sub2 = pd.read_csv('../input/fashions/sub-time.csv').sort_values('customer_id').reset_index(drop=True) #220
      4 sub3 = pd.read_csv('../input/fashions/sub-decay.csv').sort_values('customer_id').reset_index(drop=True) #217
      5 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: '../input/fashions/submission224.csv'

## === cell 7
sub00.columns = ['customer_id', 'prediction0']
sub00['prediction1'] = sub1['prediction']
sub00['prediction2'] = sub2['prediction']
sub00['prediction3'] = sub3['prediction'].astype(str)

del sub1, sub2, sub3, 
gc.collect()
sub00.head()

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2336707452.py in <cell line: 0>()
----> 1 sub00.columns = ['customer_id', 'prediction0']
      2 sub00['prediction1'] = sub1['prediction']
      3 sub00['prediction2'] = sub2['prediction']
      4 sub00['prediction3'] = sub3['prediction'].astype(str)
      5 

NameError: name 'sub00' is not defined

## === cell 8
sub00['prediction'] = sub00.apply(cust_blend, W = [0.25, 0.25, 0.25, 0.25], axis=1)
sub00.head()

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4238712120.py in <cell line: 0>()
----> 1 sub00['prediction'] = sub00.apply(cust_blend, W = [0.25, 0.25, 0.25, 0.25], axis=1)
      2 sub00.head()

NameError: name 'sub00' is not defined

## === cell 9
del sub00['prediction0']
del sub00['prediction1']
del sub00['prediction2']
del sub00['prediction3']
gc.collect()

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3946625123.py in <cell line: 0>()
----> 1 del sub00['prediction0']
      2 del sub00['prediction1']
      3 del sub00['prediction2']
      4 del sub00['prediction3']
      5 gc.collect()

NameError: name 'sub00' is not defined

## === cell 10
sub1 = pd.read_csv('../input/fashions/sub-parq.csv').sort_values('customer_id').reset_index(drop=True)
sub1['prediction'] = sub1['prediction'].astype(str)
sub0['prediction'] = sub0['prediction'].astype(str)
sub00['prediction'] = sub00['prediction'].astype(str)
sub1.columns = ['customer_id', 'prediction0']
sub1['prediction1'] = sub0['prediction']
sub1['prediction2'] = sub00['prediction']
del sub0, sub00
gc.collect()

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2480844963.py in <cell line: 0>()
----> 1 sub1 = pd.read_csv('../input/fashions/sub-parq.csv').sort_values('customer_id').reset_index(drop=True)
      2 sub1['prediction'] = sub1['prediction'].astype(str)
      3 sub0['prediction'] = sub0['prediction'].astype(str)
      4 sub00['prediction'] = sub00['prediction'].astype(str)
      5 sub1.columns = ['customer_id', 'prediction0']

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: '../input/fashions/sub-parq.csv'

## === cell 11
def cust_blend(dt, W = [1,1,1]):
    REC = []
    REC.append(dt['prediction0'].split())
    REC.append(dt['prediction1'].split())
    REC.append(dt['prediction2'].split())
    res = {}
    for M in range(len(REC)):
        for n, v in enumerate(REC[M]):
            if v in res:
                res[v] += (W[M]/(n+1))
            else:
                res[v] = (W[M]/(n+1))
    res = list(dict(sorted(res.items(), key=lambda item: -item[1])).keys())
    return ' '.join(res[:12])

sub1['prediction'] = sub1.apply(cust_blend, W = [0.415, 0.565, 0.22], axis=1)

del sub1['prediction0']
del sub1['prediction1']
del sub1['prediction2']

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1390811398.py in <cell line: 0>()
     14     return ' '.join(res[:12])
     15 
---> 16 sub1['prediction'] = sub1.apply(cust_blend, W = [0.415, 0.565, 0.22], axis=1)
     17 
     18 del sub1['prediction0']

NameError: name 'sub1' is not defined

## === cell 12
sub1.to_csv('submission.csv', index=False)

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1338966873.py in <cell line: 0>()
----> 1 sub1.to_csv('submission.csv', index=False)

NameError: name 'sub1' is not defined
