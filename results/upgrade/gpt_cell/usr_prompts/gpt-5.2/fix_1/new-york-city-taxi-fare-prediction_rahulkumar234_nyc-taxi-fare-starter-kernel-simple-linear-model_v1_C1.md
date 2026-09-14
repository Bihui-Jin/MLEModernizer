# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.9

# 2. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0

# 3. Data file paths

```
/
    kaggle/
        data/
            GCP-Coupons-Instructions.rtf (486 Bytes)
            description.md (100 lines)
            labels.csv (55413943 lines)
            labels.csv.zip (1.6 GB)
            sample_submission.csv (9915 lines)
            sample_submission.csv.zip (76.2 kB)
            test.csv (9915 lines)
            test.csv.zip (273.0 kB)
            train.csv (55423857 lines)
            train.csv.zip (1.6 GB)
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 8 other files
                new-york-city-taxi-fare-prediction/
        input/
            GCP-Coupons-Instructions.rtf (486 Bytes)
            description.md (100 lines)
            labels.csv (55413943 lines)
            labels.csv.zip (1.6 GB)
            sample_submission.csv (9915 lines)
            sample_submission.csv.zip (76.2 kB)
            test.csv (9915 lines)
            test.csv.zip (273.0 kB)
            train.csv (55423857 lines)
            train.csv.zip (1.6 GB)
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 8 other files
                new-york-city-taxi-fare-prediction/
        working/
            new-york-city-taxi-fare-prediction/
                GCP-Coupons-Instructions.rtf (486 Bytes)
                description.md (100 lines)
                ... and 8 other files
                new-york-city-taxi-fare-prediction/
```

-> data/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/labels.csv has 55413942 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/sample_submission.csv has 9914 rows and 2 columns.
The columns are: key, fare_amount

-> data/new-york-city-taxi-fare-prediction/test.csv has 9914 rows and 7 columns.
The columns are: key, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/new-york-city-taxi-fare-prediction/train.csv has 55423856 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/sample_submission.csv has 9914 rows and 2 columns.
The columns are: key, fare_amount

-> data/test.csv has 9914 rows and 7 columns.
The columns are: key, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> data/train.csv has 55423856 rows and 8 columns.
The columns are: key, fare_amount, pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude, passenger_count

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
import numpy as np # linear algebra
import pandas as pd # CSV file I/O (e.g. pd.read_csv)
import os # reading the input files we have access to

print(os.listdir('../input'))


## === cell 1
train_df =  pd.read_csv('../input/train.csv', nrows = 10_000_000)
test_df=pd.read_csv('../input/test.csv')
train_df.dtypes


## === cell 2
test_df=pd.read_csv('../input/test.csv')
test_df.head()


## === cell 3
def add_travel_vector_features(df):
    df['Difference_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['Difference_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()

add_travel_vector_features(train_df)
add_travel_vector_features(test_df)


## === cell 4
print(train_df.isnull().sum())


## === cell 5
print('Old size: %d' % len(train_df))
train_df = train_df.dropna(how = 'any', axis = 'rows')
print('New size: %d' % len(train_df))


## === cell 6
plot = train_df.iloc[:2000].plot.scatter('abs_diff_longitude', 'abs_diff_latitude')


## --- ERROR in cell 6, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mKeyError[0m                                  Traceback (most recent call last)
[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py[0m in [0;36mget_loc[0;34m(self, key)[0m
[1;32m   3804[0m         [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 3805[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_engine[0m[0;34m.[0m[0mget_loc[0m[0;34m([0m[0mcasted_key[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   3806[0m         [0;32mexcept[0m [0mKeyError[0m [0;32mas[0m [0merr[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32mindex.pyx[0m in [0;36mpandas._libs.index.IndexEngine.get_loc[0;34m()[0m

[0;32mindex.pyx[0m in [0;36mpandas._libs.index.IndexEngine.get_loc[0;34m()[0m

[0;32mpandas/_libs/hashtable_class_helper.pxi[0m in [0;36mpandas._libs.hashtable.PyObjectHashTable.get_item[0;34m()[0m

[0;32mpandas/_libs/hashtable_class_helper.pxi[0m in [0;36mpandas._libs.hashtable.PyObjectHashTable.get_item[0;34m()[0m

[0;31mKeyError[0m: 'abs_diff_longitude'

The above exception was the direct cause of the following exception:

[0;31mKeyError[0m                                  Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/285943716.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0mplot[0m [0;34m=[0m [0mtrain_df[0m[0;34m.[0m[0miloc[0m[0;34m[[0m[0;34m:[0m[0;36m2000[0m[0;34m][0m[0;34m.[0m[0mplot[0m[0;34m.[0m[0mscatter[0m[0;34m([0m[0;34m'abs_diff_longitude'[0m[0;34m,[0m [0;34m'abs_diff_latitude'[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/usr/local/lib/python3.11/dist-packages/pandas/plotting/_core.py[0m in [0;36mscatter[0;34m(self, x, y, s, c, **kwargs)[0m
[1;32m   1746[0m             ...                       colormap='viridis')
[1;32m   1747[0m         """
[0;32m-> 1748[0;31m         [0;32mreturn[0m [0mself[0m[0;34m([0m[0mkind[0m[0;34m=[0m[0;34m"scatter"[0m[0;34m,[0m [0mx[0m[0;34m=[0m[0mx[0m[0;34m,[0m [0my[0m[0;34m=[0m[0my[0m[0;34m,[0m [0ms[0m[0;34m=[0m[0ms[0m[0;34m,[0m [0mc[0m[0;34m=[0m[0mc[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1749[0m [0;34m[0m[0m
[1;32m   1750[0m     def hexbin(

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/plotting/_core.py[0m in [0;36m__call__[0;34m(self, *args, **kwargs)[0m
[1;32m    973[0m         [0;32mif[0m [0mkind[0m [0;32min[0m [0mself[0m[0;34m.[0m[0m_dataframe_kinds[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    974[0m             [0;32mif[0m [0misinstance[0m[0;34m([0m[0mdata[0m[0;34m,[0m [0mABCDataFrame[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 975[0;31m                 [0;32mreturn[0m [0mplot_backend[0m[0;34m.[0m[0mplot[0m[0;34m([0m[0mdata[0m[0;34m,[0m [0mx[0m[0;34m=[0m[0mx[0m[0;34m,[0m [0my[0m[0;34m=[0m[0my[0m[0;34m,[0m [0mkind[0m[0;34m=[0m[0mkind[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    976[0m             [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    977[0m                 [0;32mraise[0m [0mValueError[0m[0;34m([0m[0;34mf"plot kind {kind} can only be used for data frames"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/plotting/_matplotlib/__init__.py[0m in [0;36mplot[0;34m(data, kind, **kwargs)[0m
[1;32m     69[0m             [0mkwargs[0m[0;34m[[0m[0;34m"ax"[0m[0;34m][0m [0;34m=[0m [0mgetattr[0m[0;34m([0m[0max[0m[0;34m,[0m [0;34m"left_ax"[0m[0;34m,[0m [0max[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     70[0m     [0mplot_obj[0m [0;34m=[0m [0mPLOT_CLASSES[0m[0;34m[[0m[0mkind[0m[0;34m][0m[0;34m([0m[0mdata[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 71[0;31m     [0mplot_obj[0m[0;34m.[0m[0mgenerate[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     72[0m     [0mplot_obj[0m[0;34m.[0m[0mdraw[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     73[0m     [0;32mreturn[0m [0mplot_obj[0m[0;34m.[0m[0mresult[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/plotting/_matplotlib/core.py[0m in [0;36mgenerate[0;34m(self)[0m
[1;32m    499[0m         [0mself[0m[0;34m.[0m[0m_compute_plot_data[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    500[0m         [0mfig[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mfig[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 501[0;31m         [0mself[0m[0;34m.[0m[0m_make_plot[0m[0;34m([0m[0mfig[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    502[0m         [0mself[0m[0;34m.[0m[0m_add_table[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    503[0m         [0mself[0m[0;34m.[0m[0m_make_legend[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/plotting/_matplotlib/core.py[0m in [0;36m_make_plot[0;34m(self, fig)[0m
[1;32m   1350[0m             [0mlabel[0m [0;34m=[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1351[0m         scatter = ax.scatter(
[0;32m-> 1352[0;31m             [0mdata[0m[0;34m[[0m[0mx[0m[0;34m][0m[0;34m.[0m[0mvalues[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1353[0m             [0mdata[0m[0;34m[[0m[0my[0m[0;34m][0m[0;34m.[0m[0mvalues[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1354[0m             [0mc[0m[0;34m=[0m[0mc_values[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py[0m in [0;36m__getitem__[0;34m(self, key)[0m
[1;32m   4100[0m             [0;32mif[0m [0mself[0m[0;34m.[0m[0mcolumns[0m[0;34m.[0m[0mnlevels[0m [0;34m>[0m [0;36m1[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   4101[0m                 [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_getitem_multilevel[0m[0;34m([0m[0mkey[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 4102[0;31m             [0mindexer[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mcolumns[0m[0;34m.[0m[0mget_loc[0m[0;34m([0m[0mkey[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   4103[0m             [0;32mif[0m [0mis_integer[0m[0;34m([0m[0mindexer[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   4104[0m                 [0mindexer[0m [0;34m=[0m [0;34m[[0m[0mindexer[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py[0m in [0;36mget_loc[0;34m(self, key)[0m
[1;32m   3810[0m             ):
[1;32m   3811[0m                 [0;32mraise[0m [0mInvalidIndexError[0m[0;34m([0m[0mkey[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 3812[0;31m             [0;32mraise[0m [0mKeyError[0m[0;34m([0m[0mkey[0m[0;34m)[0m [0;32mfrom[0m [0merr[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   3813[0m         [0;32mexcept[0m [0mTypeError[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   3814[0m             [0;31m# If we have a listlike key, _check_indexing_error will raise[0m[0;34m[0m[0;34m[0m[0m

[0;31mKeyError[0m: 'abs_diff_longitude'

## === cell 7
print('Old size: %d' % len(train_df))
train_df = train_df[(train_df.Difference_longitude < 5.0) & (train_df.Difference_latitude < 5.0)]
print('New size: %d' % len(train_df))
