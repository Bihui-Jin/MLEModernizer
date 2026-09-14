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
Predict a neutrino particle's direction. 

## Metric
Mean angular error between the predicted and true event origins.

## Submission Format
For each `event_id` in the test set, you must predict the `azimuth` and `zenith`. The file should contain a header and have the following format:

```
event_id,azimuth,zenith
730,1,1
769,1,1
774,1,1
etc.
```

## Dataset 
[train/test]_meta.parquet

-   `batch_id` (`int`): the ID of the batch the event was placed into.
-   `event_id` (`int`): the event ID.
-   `[first/last]_pulse_index` (`int`): index of the first/last row in the features dataframe belonging to this event.
-   `[azimuth/zenith]` (`float32`): the [azimuth/zenith] angle in radians of the neutrino. A value between 0 and 2*pi for the azimuth and 0 and pi for zenith. The target columns. Not provided for the test set. The direction vector represented by zenith and azimuth points to where the neutrino came from.
-   NB: Other quantities regarding the event, such as the interaction point in `x, y, z` (vertex position), the neutrino energy, or the interaction type and kinematics are not included in the dataset.

[train/test]/batch_[n].parquet Each batch contains tens of thousands of events. Each event may contain thousands of pulses, each of which is the digitized output from a photomultiplier tube and occupies one row.

-   `event_id` (`int`): the event ID. Saved as the index column in parquet.
-   `time` (`int`): the time of the pulse in nanoseconds in the current event time window. The absolute time of a pulse has no relevance, and only the relative time with respect to other pulses within an event is of relevance.
-   `sensor_id` (`int`): the ID of which of the 5160 IceCube photomultiplier sensors recorded this pulse.
-   `charge` (`float32`): An estimate of the amount of light in the pulse, in units of photoelectrons (p.e.). A physical photon does not exactly result in a measurement of 1 p.e. but rather can take values spread around 1 p.e. As an example, a pulse with charge 2.7 p.e. could quite likely be the result of two or three photons hitting the photomultiplier tube around the same time. This data has `float16` precision but is stored as `float32` due to limitations of the version of pyarrow the data was prepared with.
-   `auxiliary` (`bool`): If `True`, the pulse was not fully digitized, is of lower quality, and was more likely to originate from noise. If `False`, then this pulse was contributed to the trigger decision and the pulse was fully digitized.

sample_submission.parquet An example submission with the correct columns and properly ordered event IDs. The sample submission is provided in the parquet format so it can be read quickly but *your final submission must be a csv*.

`sensor_geometry.csv` The `x`, `y`, and `z` positions for each of the 5160 IceCube sensors. The row index corresponds to the `sensor_idx` feature of pulses. The `x`, `y`, and `z` coordinates are in units of meters, with the origin at the center of the IceCube detector. The coordinate system is right-handed, and the z-axis points upwards when standing at the South Pole. You can convert from these coordinates to `azimuth` and `zenith` with the following formulas (here the vector (x,y,z) is normalized):

```
x = cos(azimuth) * sin(zenith)
y = sin(azimuth) * sin(zenith)
z = cos(zenith)

```

# 2. Python version

3.11

# 3. Installed packages

geopandas==0.14.4
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
pyarrow==19.0.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (231 lines)
            sample_submission.csv (13200001 lines)
            sample_submission.csv.zip (35.3 MB)
            sensor_geometry.csv (5161 lines)
            sensor_geometry.csv.zip (36.0 kB)
            test.zip (9.3 GB)
            test_meta.parquet (172.5 MB)
            train.zip (83.9 GB)
            train_meta.parquet (3.5 GB)
            icecube-neutrinos-in-deep-ice/
                description.md (231 lines)
                sample_submission.csv (13200001 lines)
                ... and 7 other files
                icecube-neutrinos-in-deep-ice/
                test/
                    batch_104.parquet (172.9 MB)
                    batch_128.parquet (172.7 MB)
                    ... and 64 other files
                    test/
                train/
                    batch_1.parquet (172.1 MB)
                    batch_10.parquet (173.4 MB)
                    ... and 592 other files
                    train/
            test/
                batch_104.parquet (172.9 MB)
                batch_128.parquet (172.7 MB)
                ... and 64 other files
                test/
            train/
                batch_1.parquet (172.1 MB)
                batch_10.parquet (173.4 MB)
                ... and 592 other files
                train/
        input/
            description.md (231 lines)
            sample_submission.csv (13200001 lines)
            sample_submission.csv.zip (35.3 MB)
            sensor_geometry.csv (5161 lines)
            sensor_geometry.csv.zip (36.0 kB)
            test.zip (9.3 GB)
            test_meta.parquet (172.5 MB)
            train.zip (83.9 GB)
            train_meta.parquet (3.5 GB)
            icecube-neutrinos-in-deep-ice/
                description.md (231 lines)
                sample_submission.csv (13200001 lines)
                ... and 7 other files
                icecube-neutrinos-in-deep-ice/
                test/
                    batch_104.parquet (172.9 MB)
                    batch_128.parquet (172.7 MB)
                    ... and 64 other files
                    test/
                train/
                    batch_1.parquet (172.1 MB)
                    batch_10.parquet (173.4 MB)
                    ... and 592 other files
                    train/
            test/
                batch_104.parquet (172.9 MB)
                batch_128.parquet (172.7 MB)
                ... and 64 other files
                test/
                    batch_104.parquet (172.9 MB)
                    batch_128.parquet (172.7 MB)
                    ... and 64 other files
                    test/
            train/
                batch_1.parquet (172.1 MB)
                batch_10.parquet (173.4 MB)
                ... and 592 other files
                train/
                    batch_1.parquet (172.1 MB)
                    batch_10.parquet (173.4 MB)
                    ... and 592 other files
                    train/
        working/
            icecube-neutrinos-in-deep-ice/
                description.md (231 lines)
                sample_submission.csv (13200001 lines)
                ... and 7 other files
                icecube-neutrinos-in-deep-ice/
                test/
                    batch_104.parquet (172.9 MB)
                    batch_128.parquet (172.7 MB)
                    ... and 64 other files
                    test/
                train/
                    batch_1.parquet (172.1 MB)
                    batch_10.parquet (173.4 MB)
                    ... and 592 other files
                    train/
```

-> data/icecube-neutrinos-in-deep-ice/sample_submission.csv has 13200000 rows and 3 columns.
The columns are: event_id, azimuth, zenith

-> data/icecube-neutrinos-in-deep-ice/sensor_geometry.csv has 5160 rows and 4 columns.
The columns are: sensor_id, x, y, z

-> data/sample_submission.csv has 13200000 rows and 3 columns.
The columns are: event_id, azimuth, zenith

-> data/sensor_geometry.csv has 5160 rows and 4 columns.
The columns are: sensor_id, x, y, z

-> input/icecube-neutrinos-in-deep-ice/sample_submission.csv has 13200000 rows and 3 columns.
The columns are: event_id, azimuth, zenith

-> input/icecube-neutrinos-in-deep-ice/sensor_geometry.csv has 5160 rows and 4 columns.
The columns are: sensor_id, x, y, z

-> (stopped after 10 files for performance)

# 5. Target score

1.567501

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyarrow.parquet as pq


## === cell 1
train_meta = pq.read_pandas('/kaggle/input/icecube-neutrinos-in-deep-ice/train_meta.parquet').to_pandas()
data = '/kaggle/input/icecube-neutrinos-in-deep-ice/train/'
batch_files = [data+'batch_1.parquet', data+'batch_2.parquet',data+'batch_10.parquet']
batch_data = pd.concat([pq.read_pandas(file, columns=['event_id', 'time', 'sensor_id', 'charge', 'auxiliary']).to_pandas() for file in batch_files])


## === cell 2
sensor_geometry = pd.read_csv('/kaggle/input/icecube-neutrinos-in-deep-ice/sensor_geometry.csv')
sub = pq.read_pandas('/kaggle/input/icecube-neutrinos-in-deep-ice/sample_submission.parquet').to_pandas()
test = pq.read_pandas('/kaggle/input/icecube-neutrinos-in-deep-ice/test/batch_661.parquet').to_pandas()
test_meta = pq.read_pandas('/kaggle/input/icecube-neutrinos-in-deep-ice/test_meta.parquet').to_pandas()
test_data = pd.read_parquet('/kaggle/input/icecube-neutrinos-in-deep-ice/test_meta.parquet')


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2298447522.py in <cell line: 0>()
      1 # read sensor_geometry.csv using pandas
      2 sensor_geometry = pd.read_csv('/kaggle/input/icecube-neutrinos-in-deep-ice/sensor_geometry.csv')
----> 3 sub = pq.read_pandas('/kaggle/input/icecube-neutrinos-in-deep-ice/sample_submission.parquet').to_pandas()
      4 test = pq.read_pandas('/kaggle/input/icecube-neutrinos-in-deep-ice/test/batch_661.parquet').to_pandas()
      5 test_meta = pq.read_pandas('/kaggle/input/icecube-neutrinos-in-deep-ice/test_meta.parquet').to_pandas()

/usr/local/lib/python3.11/dist-packages/pyarrow/parquet/core.py in read_pandas(source, columns, **kwargs)
   1856 
   1857 def read_pandas(source, columns=None, **kwargs):
-> 1858     return read_table(
   1859         source, columns=columns, use_pandas_metadata=True, **kwargs
   1860     )

/usr/local/lib/python3.11/dist-packages/pyarrow/parquet/core.py in read_table(source, columns, use_threads, schema, use_pandas_metadata, read_dictionary, memory_map, buffer_size, partitioning, filesystem, filters, use_legacy_dataset, ignore_prefixes, pre_buffer, coerce_int96_timestamp_unit, decryption_properties, thrift_string_size_limit, thrift_container_size_limit, page_checksum_verification)
   1791 
   1792     try:
-> 1793         dataset = ParquetDataset(
   1794             source,
   1795             schema=schema,

/usr/local/lib/python3.11/dist-packages/pyarrow/parquet/core.py in __init__(self, path_or_paths, filesystem, schema, filters, read_dictionary, memory_map, buffer_size, partitioning, ignore_prefixes, pre_buffer, coerce_int96_timestamp_unit, decryption_properties, thrift_string_size_limit, thrift_container_size_limit, page_checksum_verification, use_legacy_dataset)
   1369                 infer_dictionary=True)
   1370 
-> 1371         self._dataset = ds.dataset(path_or_paths, filesystem=filesystem,
   1372                                    schema=schema, format=parquet_format,
   1373                                    partitioning=partitioning,

/usr/local/lib/python3.11/dist-packages/pyarrow/dataset.py in dataset(source, schema, format, filesystem, partitioning, partition_base_dir, exclude_invalid_files, ignore_prefixes)
    792 
    793     if _is_path_like(source):
--> 794         return _filesystem_dataset(source, **kwargs)
    795     elif isinstance(source, (tuple, list)):
    796         if all(_is_path_like(elem) or isinstance(elem, FileInfo) for elem in source):

/usr/local/lib/python3.11/dist-packages/pyarrow/dataset.py in _filesystem_dataset(source, schema, filesystem, partitioning, format, partition_base_dir, exclude_invalid_files, selector_ignore_prefixes)
    474             fs, paths_or_selector = _ensure_multiple_sources(source, filesystem)
    475     else:
--> 476         fs, paths_or_selector = _ensure_single_source(source, filesystem)
    477 
    478     options = FileSystemFactoryOptions(

/usr/local/lib/python3.11/dist-packages/pyarrow/dataset.py in _ensure_single_source(path, filesystem)
    439         paths_or_selector = [path]
    440     else:
--> 441         raise FileNotFoundError(path)
    442 
    443     return filesystem, paths_or_selector

FileNotFoundError: /kaggle/input/icecube-neutrinos-in-deep-ice/sample_submission.parquet

## === cell 3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


## === cell 4
subset = train_meta.sample(frac=0.02, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(subset.drop(['azimuth', 'zenith'], axis=1), subset[['azimuth', 'zenith']], test_size=0.2, random_state=42)


## === cell 5
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


## === cell 6
test_predictions = model.predict(test_data)


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/410537113.py in <cell line: 0>()
      1 # make predictions on the test data
----> 2 test_predictions = model.predict(test_data)

NameError: name 'test_data' is not defined

## === cell 7
test_predictions


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2630918204.py in <cell line: 0>()
----> 1 test_predictions

NameError: name 'test_predictions' is not defined

## === cell 8
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.fit_transform(X_test)

model = LinearRegression()

model.fit(X_train_poly, y_train)

y_pred = model.predict(X_test_poly)


## === cell 9
X_test_poly = poly.transform(test_data)

y_pred = model.predict(X_test_poly)


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1958809293.py in <cell line: 0>()
----> 1 X_test_poly = poly.transform(test_data)
      2 
      3 # Make predictions on the test data
      4 y_pred = model.predict(X_test_poly)

NameError: name 'test_data' is not defined

## === cell 10
submission = pd.DataFrame({'event_id': test_data.index, 'azimuth': y_pred[:,0], 'zenith': y_pred[:,1]})


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3241677793.py in <cell line: 0>()
      1 # create the submission dataframe
----> 2 submission = pd.DataFrame({'event_id': test_data.index, 'azimuth': y_pred[:,0], 'zenith': y_pred[:,1]})

NameError: name 'test_data' is not defined

## === cell 11
sub_df = pd.read_parquet('/kaggle/input/icecube-neutrinos-in-deep-ice/sample_submission.parquet')


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/228252576.py in <cell line: 0>()
----> 1 sub_df = pd.read_parquet('/kaggle/input/icecube-neutrinos-in-deep-ice/sample_submission.parquet')

/usr/local/lib/python3.11/dist-packages/pandas/io/parquet.py in read_parquet(path, engine, columns, storage_options, use_nullable_dtypes, dtype_backend, filesystem, filters, **kwargs)
    665     check_dtype_backend(dtype_backend)
    666 
--> 667     return impl.read(
    668         path,
    669         columns=columns,

/usr/local/lib/python3.11/dist-packages/pandas/io/parquet.py in read(self, path, columns, filters, use_nullable_dtypes, dtype_backend, storage_options, filesystem, **kwargs)
    265             to_pandas_kwargs["split_blocks"] = True  # type: ignore[assignment]
    266 
--> 267         path_or_handle, handles, filesystem = _get_path_or_handle(
    268             path,
    269             filesystem,

/usr/local/lib/python3.11/dist-packages/pandas/io/parquet.py in _get_path_or_handle(path, fs, storage_options, mode, is_dir)
    138         # fsspec resources can also point to directories
    139         # this branch is used for example when reading from non-fsspec URLs
--> 140         handles = get_handle(
    141             path_or_handle, mode, is_text=False, storage_options=storage_options
    142         )

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    880         else:
    881             # Binary mode
--> 882             handle = open(handle, ioargs.mode)
    883         handles.append(handle)
    884 

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/icecube-neutrinos-in-deep-ice/sample_submission.parquet'

## === cell 12
sub_df['azimuth']=submission['azimuth']
sub_df['zenith']=submission['zenith']


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2741858588.py in <cell line: 0>()
----> 1 sub_df['azimuth']=submission['azimuth']
      2 sub_df['zenith']=submission['zenith']

NameError: name 'submission' is not defined

## === cell 13
sub_df.to_csv('submission.csv', index=False)


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3763899528.py in <cell line: 0>()
      1 # write the submission dataframe to a CSV file
----> 2 sub_df.to_csv('submission.csv', index=False)

NameError: name 'sub_df' is not defined

## === cell 14
sub_df


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2580825199.py in <cell line: 0>()
----> 1 sub_df

NameError: name 'sub_df' is not defined
