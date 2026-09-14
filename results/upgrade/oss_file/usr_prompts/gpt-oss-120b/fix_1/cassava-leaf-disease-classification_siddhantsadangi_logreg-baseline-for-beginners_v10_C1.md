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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

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
pillow==11.3.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tqdm==4.67.1

# 4. Data file paths

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

# 5. Target score

0.5157

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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

df_samp = df_samp.append(df_train.sample(2000), ignore_index=True)

df_samp.groupby(by='disease').count()


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3980020134.py in <cell line: 0>()
      1 df_samp = pd.DataFrame()
      2 
----> 3 df_samp = df_samp.append(df_train.sample(2000), ignore_index=True)
      4 
      5 df_samp.groupby(by='disease').count()

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'append'

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


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/73640213.py in <cell line: 0>()
      1 from sklearn.model_selection import train_test_split
      2 
----> 3 X = df_samp.drop(columns=['label'])
      4 y = df_samp.label
      5 

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

KeyError: "['label'] not found in axis"

## === cell 14
compressed_size = (200,150)


## === cell 15
train_array = np.array([np.asarray(Image.open(path).resize(compressed_size, Image.ANTIALIAS)) for path in tqdm(X_train.path)])
valid_array = np.array([np.asarray(Image.open(path).resize(compressed_size, Image.ANTIALIAS)) for path in tqdm(X_valid.path)])

print(train_array.shape)
print(valid_array.shape)


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3723642037.py in <cell line: 0>()
----> 1 train_array = np.array([np.asarray(Image.open(path).resize(compressed_size, Image.ANTIALIAS)) for path in tqdm(X_train.path)])
      2 valid_array = np.array([np.asarray(Image.open(path).resize(compressed_size, Image.ANTIALIAS)) for path in tqdm(X_valid.path)])
      3 
      4 print(train_array.shape)
      5 print(valid_array.shape)

NameError: name 'X_train' is not defined

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


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3291671438.py in <cell line: 0>()
      1 plt.figure(figsize=(20,12))
      2 
----> 3 for i, img in tqdm(enumerate(train_array[:5])):
      4     plt.subplot(1, 5, i+1)
      5     plt.xticks([])

NameError: name 'train_array' is not defined

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


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3789382677.py in <cell line: 0>()
      1 plt.figure(figsize=(20,12))
      2 
----> 3 for i, img in tqdm(enumerate(valid_array[:5])):
      4     plt.subplot(1, 5, i+1)
      5     plt.xticks([])

NameError: name 'valid_array' is not defined

## === cell 18
print(f'Length of the training array is {len(train_array)}')
print(f'Shape of the training array is {train_array.shape}')
print(f'Shape of each training image array is {train_array[0].shape}')


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/243702640.py in <cell line: 0>()
----> 1 print(f'Length of the training array is {len(train_array)}')
      2 print(f'Shape of the training array is {train_array.shape}')
      3 print(f'Shape of each training image array is {train_array[0].shape}')

NameError: name 'train_array' is not defined

## === cell 19
train_array.resize(len(train_array), train_array.shape[1]*train_array.shape[2]*train_array.shape[3])

print(f'New length of the training array is {len(train_array)}')
print(f'New shape of the training array is {train_array.shape}')
print(f'New shape of each training image array is {train_array[0].shape}')


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2142493536.py in <cell line: 0>()
----> 1 train_array.resize(len(train_array), train_array.shape[1]*train_array.shape[2]*train_array.shape[3])
      2 
      3 print(f'New length of the training array is {len(train_array)}')
      4 print(f'New shape of the training array is {train_array.shape}')
      5 print(f'New shape of each training image array is {train_array[0].shape}')

NameError: name 'train_array' is not defined

## === cell 20
valid_array.resize(len(valid_array), valid_array.shape[1]*valid_array.shape[2]*valid_array.shape[3])

print(f'New length of the validation array is {len(valid_array)}')
print(f'New shape of the validation array is {valid_array.shape}')
print(f'New shape of each validation image array is {valid_array[0].shape}')


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2198808829.py in <cell line: 0>()
----> 1 valid_array.resize(len(valid_array), valid_array.shape[1]*valid_array.shape[2]*valid_array.shape[3])
      2 
      3 print(f'New length of the validation array is {len(valid_array)}')
      4 print(f'New shape of the validation array is {valid_array.shape}')
      5 print(f'New shape of each validation image array is {valid_array[0].shape}')

NameError: name 'valid_array' is not defined

## === cell 21
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(class_weight='balanced', verbose=5, n_jobs=-1)


## === cell 22
lr.fit(train_array, y_train)


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/931164985.py in <cell line: 0>()
----> 1 lr.fit(train_array, y_train)

NameError: name 'train_array' is not defined

## === cell 23
preds = lr.predict(valid_array)
preds


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1633854246.py in <cell line: 0>()
----> 1 preds = lr.predict(valid_array)
      2 preds

NameError: name 'valid_array' is not defined

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


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2954814632.py in <cell line: 0>()
      2 import seaborn as sns
      3 
----> 4 label = sorted(y_valid.unique())
      5 sns.heatmap(confusion_matrix(y_valid, preds), annot=True, square=True, fmt='g', 
      6             xticklabels=label, yticklabels=label, cbar=False)

NameError: name 'y_valid' is not defined

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
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2132551979.py in <cell line: 0>()
----> 1 test_array = np.array([np.asarray(Image.open(path).resize(compressed_size, Image.ANTIALIAS)) for path in tqdm(test_path)])
      2 test_array.resize(len(test_array), test_array.shape[1]*test_array.shape[2]*test_array.shape[3])

/tmp/ipykernel_11/2132551979.py in <listcomp>(.0)
----> 1 test_array = np.array([np.asarray(Image.open(path).resize(compressed_size, Image.ANTIALIAS)) for path in tqdm(test_path)])
      2 test_array.resize(len(test_array), test_array.shape[1]*test_array.shape[2]*test_array.shape[3])

AttributeError: module 'PIL.Image' has no attribute 'ANTIALIAS'

## === cell 28
submission = lr.predict(test_array)
submission


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/798566040.py in <cell line: 0>()
----> 1 submission = lr.predict(test_array)
      2 submission

NameError: name 'test_array' is not defined

## === cell 29
submission_df = pd.DataFrame({'image_id':[path.split('/')[-1] for path in test_path], 
                              'label':submission})
submission_df


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1757876400.py in <cell line: 0>()
      1 submission_df = pd.DataFrame({'image_id':[path.split('/')[-1] for path in test_path], 
----> 2                               'label':submission})
      3 submission_df

NameError: name 'submission' is not defined

## === cell 30
submission_df.to_csv('/kaggle/working/submission.csv', index=False)


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/241923186.py in <cell line: 0>()
----> 1 submission_df.to_csv('/kaggle/working/submission.csv', index=False)

NameError: name 'submission_df' is not defined
