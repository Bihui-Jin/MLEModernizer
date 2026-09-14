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
pillow==11.3.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tqdm==4.67.1

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os



## === cell 1
df_train = pd.read_csv(r'/kaggle/input/cassava-leaf-disease-classification/train.csv')
df_train.head()


## === cell 2
import json 

with open(r'/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json') as json_file: 
    label_map = json.load(json_file) 

label_map = {int(k):v for k,v in label_map.items()}
label_map


## === cell 3
df_train['disease'] = df_train['label'].map(label_map)
df_train


## === cell 4
import glob
train_path = glob.glob(r'/kaggle/input/cassava-leaf-disease-classification/train_images/*.jpg')
train_path.sort()
print(len(train_path))


## === cell 5
df_train['path'] = train_path
df_train


## === cell 6
df_train.groupby(['disease']).size().plot(kind='bar')


## === cell 7
import matplotlib.pyplot as plt
from PIL import Image


## === cell 8
img = Image.open(df_train.path[0])
img


## === cell 9
img.size


## === cell 10
from tqdm.notebook import tqdm #to monitor progress
np.random.seed(42) #to get reproducible results


## === cell 11
df_samp = pd.DataFrame()

df_samp = pd.concat([df_samp, df_train.sample(1000)], ignore_index=True)

df_samp.groupby(by="disease").count()


## === cell 12
from sklearn.utils import shuffle

df_samp = shuffle(df_samp).reset_index(drop=True) #shuffling the dataframe


## === cell 13
from sklearn.model_selection import train_test_split

X = df_samp.drop(columns=['label'])
y = df_samp.label

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, stratify=y)

print(X_train.shape)
print(len(y_train))
print(X_valid.shape)
print(len(y_valid))


## === cell 14
compressed_size = (200,150)


## === cell 15
resample_filter = (
    Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
)

train_array = np.array(
    [
        np.asarray(Image.open(path).resize(compressed_size, resample_filter))
        for path in tqdm(X_train.path)
    ]
)
valid_array = np.array(
    [
        np.asarray(Image.open(path).resize(compressed_size, resample_filter))
        for path in tqdm(X_valid.path)
    ]
)

print(train_array.shape)
print(valid_array.shape)


## === cell 16
plt.figure(figsize=(20,12))

for i, img in tqdm(enumerate(train_array[:5])):
    plt.subplot(1, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img)
    plt.title(X_train.disease.iloc[i])
    plt.xlabel(X_train.image_id.iloc[i])

plt.show()


## === cell 17
plt.figure(figsize=(20,12))

for i, img in tqdm(enumerate(valid_array[:5])):
    plt.subplot(1, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img)
    plt.title(X_valid.disease.iloc[i])
    plt.xlabel(X_valid.image_id.iloc[i])

plt.show()


## === cell 18
print(f'Length of the training array is {len(train_array)}')
print(f'Shape of the training array is {train_array.shape}')
print(f'Shape of each training image array is {train_array[0].shape}')


## === cell 19
train_array.resize(len(train_array), train_array.shape[1]*train_array.shape[2]*train_array.shape[3])

print(f'New length of the training array is {len(train_array)}')
print(f'New shape of the training array is {train_array.shape}')
print(f'New shape of each training image array is {train_array[0].shape}')


## === cell 20
valid_array.resize(len(valid_array), valid_array.shape[1]*valid_array.shape[2]*valid_array.shape[3])

print(f'New length of the validation array is {len(valid_array)}')
print(f'New shape of the validation array is {valid_array.shape}')
print(f'New shape of each validation image array is {valid_array[0].shape}')


## === cell 21
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(class_weight='balanced', verbose=5, n_jobs=-1)


## === cell 22
lr.fit(train_array, y_train)


## === cell 23
preds = lr.predict(valid_array)
preds


## === cell 24
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns

label = sorted(y_valid.unique())
sns.heatmap(confusion_matrix(y_valid, preds), annot=True, square=True, fmt='g', 
            xticklabels=label, yticklabels=label, cbar=False)

plt.title('Confusion matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print(classification_report(y_valid, preds))


## === cell 25
test_path = glob.glob(r'/kaggle/input/cassava-leaf-disease-classification/test_images/*.jpg')
test_path.sort()
print(len(test_path))
test_path


## === cell 26
Image.open(test_path[0])


## === cell 27
test_array = np.array([np.asarray(Image.open(path).resize(compressed_size, Image.ANTIALIAS)) for path in tqdm(test_path)])
test_array.resize(len(test_array), test_array.shape[1]*test_array.shape[2]*test_array.shape[3])


## --- ERROR in cell 27, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2132551979.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0mtest_array[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0marray[0m[0;34m([0m[0;34m[[0m[0mnp[0m[0;34m.[0m[0masarray[0m[0;34m([0m[0mImage[0m[0;34m.[0m[0mopen[0m[0;34m([0m[0mpath[0m[0;34m)[0m[0;34m.[0m[0mresize[0m[0;34m([0m[0mcompressed_size[0m[0;34m,[0m [0mImage[0m[0;34m.[0m[0mANTIALIAS[0m[0;34m)[0m[0;34m)[0m [0;32mfor[0m [0mpath[0m [0;32min[0m [0mtqdm[0m[0;34m([0m[0mtest_path[0m[0;34m)[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0mtest_array[0m[0;34m.[0m[0mresize[0m[0;34m([0m[0mlen[0m[0;34m([0m[0mtest_array[0m[0;34m)[0m[0;34m,[0m [0mtest_array[0m[0;34m.[0m[0mshape[0m[0;34m[[0m[0;36m1[0m[0;34m][0m[0;34m*[0m[0mtest_array[0m[0;34m.[0m[0mshape[0m[0;34m[[0m[0;36m2[0m[0;34m][0m[0;34m*[0m[0mtest_array[0m[0;34m.[0m[0mshape[0m[0;34m[[0m[0;36m3[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/2132551979.py[0m in [0;36m<listcomp>[0;34m(.0)[0m
[0;32m----> 1[0;31m [0mtest_array[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0marray[0m[0;34m([0m[0;34m[[0m[0mnp[0m[0;34m.[0m[0masarray[0m[0;34m([0m[0mImage[0m[0;34m.[0m[0mopen[0m[0;34m([0m[0mpath[0m[0;34m)[0m[0;34m.[0m[0mresize[0m[0;34m([0m[0mcompressed_size[0m[0;34m,[0m [0mImage[0m[0;34m.[0m[0mANTIALIAS[0m[0;34m)[0m[0;34m)[0m [0;32mfor[0m [0mpath[0m [0;32min[0m [0mtqdm[0m[0;34m([0m[0mtest_path[0m[0;34m)[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0mtest_array[0m[0;34m.[0m[0mresize[0m[0;34m([0m[0mlen[0m[0;34m([0m[0mtest_array[0m[0;34m)[0m[0;34m,[0m [0mtest_array[0m[0;34m.[0m[0mshape[0m[0;34m[[0m[0;36m1[0m[0;34m][0m[0;34m*[0m[0mtest_array[0m[0;34m.[0m[0mshape[0m[0;34m[[0m[0;36m2[0m[0;34m][0m[0;34m*[0m[0mtest_array[0m[0;34m.[0m[0mshape[0m[0;34m[[0m[0;36m3[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mAttributeError[0m: module 'PIL.Image' has no attribute 'ANTIALIAS'

## === cell 28
submission = lr.predict(test_array)
submission
