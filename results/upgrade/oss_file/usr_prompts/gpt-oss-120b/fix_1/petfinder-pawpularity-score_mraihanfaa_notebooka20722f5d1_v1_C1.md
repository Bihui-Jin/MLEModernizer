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
Predict engagement with a pet's profile based on the photograph for that profile.

## Metric
Root mean squared error.

## Submission Format
For each `Id` in the test set, you must predict a probability for the target variable, `Pawpularity`. The file should contain a header and have the following format:

```
Id, Pawpularity
0008dbfb52aa1dc6ee51ee02adf13537, 99.24
0014a7b528f1682f0cf3b73a991c17a0, 61.71
0019c1388dfcd30ac8b112fb4250c251, 6.23
00307b779c82716b240a24f028b0031b, 9.43
00320c6dd5b4223c62a9670110d47911, 70.89
etc.
```

## Dataset
- **train/** - Folder containing training set photos of the form **{id}.jpg**, where **{id}** is a unique Pet Profile ID.
- **train.csv** - Metadata (described below) for each photo in the training set as well as the target, the photo's Pawpularity score. The Id column gives the photo's unique Pet Profile ID corresponding the photo's file name.

The train.csv and test.csv files contain metadata for photos in the training set and test set, respectively. Each pet photo is labeled with the value of 1 (Yes) or 0 (No) for each of the following features:

- **Focus** - Pet stands out against uncluttered background, not too close / far.
- **Eyes** - Both eyes are facing front or near-front, with at least 1 eye / pupil decently clear.
- **Face** - Decently clear face, facing front or near-front.
- **Near** - Single pet taking up significant portion of photo (roughly over 50% of photo width or height).
- **Action** - Pet in the middle of an action (e.g., jumping).
- **Accessory** - Accompanying physical or digital accessory / prop (i.e. toy, digital sticker), excluding collar and leash.
- **Group** - More than 1 pet in the photo.
- **Collage** - Digitally-retouched photo (i.e. with digital photo frame, combination of multiple photos).
- **Human** - Human in the photo.
- **Occlusion** - Specific undesirable objects blocking part of the pet (i.e. human, cage or fence). Note that not all blocking objects are considered occlusion.
- **Info** - Custom-added text or labels (i.e. pet name, description).
- **Blur** - Noticeably out of focus or noisy, especially for the pet's eyes and face. For Blur entries, "Eyes" column is always set to 0.

# 2. Python version

3.10

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
missingno==0.5.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
        input/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
        working/
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
```

-> data/petfinder-pawpularity-score/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/petfinder-pawpularity-score/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/petfinder-pawpularity-score/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> data/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> (stopped after 10 files for performance)

# 5. Target score

20.510964682588504

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
df_train_metadata = pd.read_csv('/kaggle/input/petfinder-pawpularity-score/train.csv')
df_test_metadata = pd.read_csv('/kaggle/input/petfinder-pawpularity-score/test.csv')
df_train_metadata.head

## === cell 2
import tensorflow as tf
import cv2
import missingno as msno
import matplotlib.pyplot as plt

train_image = df_train_metadata.copy()
test_image = df_test_metadata.copy()

train_image["file_path"] = df_train_metadata["Id"].apply(lambda x: "../input/petfinder-pawpularity-score/train/" + x + ".jpg")
test_image["file_path"] = df_test_metadata["Id"].apply(lambda x: "../input/petfinder-pawpularity-score/test/" + x + ".jpg")



plt.figure(figsize=(20, 20))
row, col = 5, 4
for i in range(row * col):
    plt.subplot(row, col, i+1)
    image = cv2.imread(train_image.loc[i, 'file_path'])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    target = train_image.loc[i, 'Pawpularity']
    plt.imshow(image)
    plt.title(f"No: {i}" f"   Pawpularity: {target}")
plt.show()

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
image_features_train = pd.read_csv('../input/image-features/image_features_train.csv')
image_features_test = pd.read_csv('../input/image-features/image_features_test.csv')


df_train = pd.concat([df_train_metadata['Pawpularity'], df_train_metadata.loc[:,'Subject Focus':'Blur'],image_features_train.loc[:,'centroid_pr_x':'kurtosis_pr_y']], axis=1)
df_test = pd.concat([df_test_metadata.loc[:,'Subject Focus':'Blur'],image_features_test.loc[:,'centroid_pr_x':'kurtosis_pr_y']], axis=1)

df_test.head(3)

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3432602787.py in <cell line: 0>()
      1 #load data google drive
----> 2 image_features_train = pd.read_csv('../input/image-features/image_features_train.csv')
      3 image_features_test = pd.read_csv('../input/image-features/image_features_test.csv')
      4 
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

FileNotFoundError: [Errno 2] No such file or directory: '../input/image-features/image_features_train.csv'

## === cell 4
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

label_diskrit = np.asarray(df_train['Pawpularity'])

label_kontinu = np.asarray(df_train['Pawpularity'])

features = np.asarray(df_train_metadata.loc[:,'Subject Focus':'Blur'])


scaler = preprocessing.MinMaxScaler(feature_range=(0, 10)).fit(features)
scaled_feature = scaler.transform(features)



ranked_index = [ 2, 9, 1, 10, 4, 5, 8, 0, 3, 11, 6, 7 ]
print("\nMIM")
print(ranked_index)


result = scaled_feature[:, ranked_index[:]]
print(result)

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/382085897.py in <cell line: 0>()
      3 
      4 #ambil kolom untuk label diskrit (class)
----> 5 label_diskrit = np.asarray(df_train['Pawpularity'])
      6 #print(label_diskrit)
      7 

NameError: name 'df_train' is not defined

## === cell 5
def rmse(predict, actual):
    predict = np.array(predict)
    actual = np.array(actual)

    distance = predict - actual

    square_distance = distance ** 2

    mean_square_distance = square_distance.mean()

    score = np.sqrt(mean_square_distance)

    return score

## === cell 6
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split

jumlah_fitur = range(1,scaled_feature.shape[1]+1)
scores = []
score = 1000
best_score = 1000
best_feature_number = 0

for jumlah_fitur_terbaik in jumlah_fitur:
    selected_features = result[:,0:jumlah_fitur_terbaik]
    X_train, X_test, y_train, y_test = train_test_split(selected_features, label_kontinu, test_size=0.3, random_state=0)

    regressor = DecisionTreeRegressor()
    
    param_grid = [
    {'splitter':['best','random'],'max_depth':[5, 10, 15], 'min_samples_split':[0.1, 1.0, 10], 'min_samples_leaf':[0.1, 0.5, 5]}
    ]
    
    metric = make_scorer(mean_squared_error, greater_is_better=False)

    model = GridSearchCV(regressor, param_grid, scoring=metric, cv=5, refit = True, verbose = 3) 
    model.fit(X_train, y_train)

    print(model.best_params_) 
      
    print(model.best_estimator_)


    score = abs(model.score(X_test, y_test))
    scores.append(score)

    if(best_score > score):
      best_score = score
      best_model = model
      best_feature_number = jumlah_fitur_terbaik
      best_parameter = model.best_params_
      
      best_X_test = X_test


plt.figure()
plt.xlabel('jumlah_fitur_terbaik')
plt.ylabel('score')
plt.scatter(jumlah_fitur, scores)
plt.grid()

print(scores);
print('Jumlah fitur terbaik adalah: ',best_feature_number)
print('Score terbaik adalah: ',best_score)
print('Parameter terbaik adalah: ',best_parameter)

final_predictions = best_model.predict(best_X_test) 
  



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3911488511.py in <cell line: 0>()
     11 from sklearn.model_selection import train_test_split
     12 
---> 13 jumlah_fitur = range(1,scaled_feature.shape[1]+1)
     14 scores = []
     15 score = 1000

NameError: name 'scaled_feature' is not defined

## === cell 7
from sklearn.metrics import r2_score

fig, ax = plt.subplots()
ax.text(1, 9.5,'$R^2=$'+str(round(r2_score(y_test, final_predictions),4)), fontsize=12, verticalalignment='top', multialignment='center')
ax.text(1, 9,'$MSE=$'+str(round(rmse(y_test, final_predictions),4)), fontsize=12, verticalalignment='top', multialignment='center')

ax.set_xlim(xmin=1)
ax.set_ylim(ymin=1)
ax.set_xlim(xmax=100)
ax.set_ylim(ymax=100)

ax.set_xlabel('Actual Value', fontsize=14)
ax.set_ylabel('Predicted Value', fontsize=14)
ax.scatter(y_test, final_predictions, s=100, c=y_test, cmap='viridis')

lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]

ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.grid(True, which='both')

xvalue = np.linspace(1,10,10)
lsigma = ax.fill_between(xvalue, xvalue+1, xvalue-1, color='blue', alpha=0.3)

plt.show()

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3718180729.py in <cell line: 0>()
      2 
      3 fig, ax = plt.subplots()
----> 4 ax.text(1, 9.5,'$R^2=$'+str(round(r2_score(y_test, final_predictions),4)), fontsize=12, verticalalignment='top', multialignment='center')
      5 ax.text(1, 9,'$MSE=$'+str(round(rmse(y_test, final_predictions),4)), fontsize=12, verticalalignment='top', multialignment='center')
      6 

NameError: name 'y_test' is not defined

## === cell 8
features_test = np.asarray(df_test.loc[:, "Subject Focus":"kurtosis_pr_y"])
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(features_test)
features_test = scaler.transform(features_test)

result_test = features_test[:, ranked_index[:]]
selected_features_kaggle_test = result_test[:, 0:best_feature_number]

final_predictions_kaggle = best_model.predict(selected_features_kaggle_test) 
data_test = np.array(df_test_metadata.Id)
features_test = np.asarray(df_test.loc[:, "Subject Focus":"kurtosis_pr_y"])
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(features_test)
features_test = scaler.transform(features_test)

result_test = features_test[:, ranked_index[:]]
selected_features_kaggle_test = result_test[:, 0:best_feature_number]

final_predictions_kaggle = best_model.predict(selected_features_kaggle_test) 
data_test = np.array(df_test_metadata.Id)


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/86848047.py in <cell line: 0>()
----> 1 features_test = np.asarray(df_test.loc[:, "Subject Focus":"kurtosis_pr_y"])
      2 scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(features_test)
      3 features_test = scaler.transform(features_test)
      4 
      5 result_test = features_test[:, ranked_index[:]]

NameError: name 'df_test' is not defined

## === cell 9
selected_features_kaggle_test = result_test[:, 0:best_feature_number]
final_predictions_kaggle = best_model.predict(selected_features_kaggle_test) 
final_predictions_kaggle

df_hasil = pd.DataFrame({"Id":df_test_metadata.Id,"Pawpularity":final_predictions_kaggle})

df_hasil.to_csv('submission.csv', index=False)
df_hasil.head()

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2871590614.py in <cell line: 0>()
      1 #generate file prediksi
----> 2 selected_features_kaggle_test = result_test[:, 0:best_feature_number]
      3 final_predictions_kaggle = best_model.predict(selected_features_kaggle_test)
      4 final_predictions_kaggle
      5 

NameError: name 'result_test' is not defined
