# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Predict the individual whale species in images.

## Metric
Mean Average Precision @ 5 (MAP@5).

## Submission Format
For each `Image` in the test set, you may predict up to 5 labels for the whale `Id`. Whales that are not predicted to be one of the labels in the training data should be labeled as `new_whale`. The file should contain a header and have the following format:

```
Image,Id
00029b3a.jpg,new_whale w_1287fbc w_98baff9 w_7554f44 w_1eafe46
0003c693.jpg,new_whale w_1287fbc w_98baff9 w_7554f44 w_1eafe46
...
```

## Dataset
This training data contains thousands of images of humpback whale flukes. Individual whales have been identified by researchers and given an `Id`. The challenge is to predict the whale `Id` of images in the test set. What makes this such a challenge is that there are only a few examples for each of 3,000+ whale Ids.

- **train.zip** - a folder containing the training images
- **train.csv** - maps the training `Image` to the appropriate whale `Id`. Whales that are not predicted to have a label identified in the training data should be labeled as `new_whale`.
- **test.zip** - a folder containing the test images to predict the whale `Id`
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (51 lines)
            sample_submission.csv (2611 lines)
            sample_submission.csv.zip (15.5 kB)
            test.zip (72.1 MB)
            train.csv (7241 lines)
            train.csv.zip (68.3 kB)
            train.zip (200.4 MB)
            test/
                11482c0d.jpg (27.8 kB)
                68edd063.jpg (11.5 kB)
                ... and 2608 other files
                test/
            train/
                6fc790ac.jpg (17.1 kB)
                f4bb7bb0.jpg (11.6 kB)
                ... and 7238 other files
                train/
            whale-categorization-playground/
                description.md (51 lines)
                sample_submission.csv (2611 lines)
                ... and 5 other files
                test/
                    11482c0d.jpg (27.8 kB)
                    68edd063.jpg (11.5 kB)
                    ... and 2608 other files
                    test/
                train/
                    6fc790ac.jpg (17.1 kB)
                    f4bb7bb0.jpg (11.6 kB)
                    ... and 7238 other files
                    train/
                whale-categorization-playground/
        input/
            description.md (51 lines)
            sample_submission.csv (2611 lines)
            sample_submission.csv.zip (15.5 kB)
            test.zip (72.1 MB)
            train.csv (7241 lines)
            train.csv.zip (68.3 kB)
            train.zip (200.4 MB)
            test/
                11482c0d.jpg (27.8 kB)
                68edd063.jpg (11.5 kB)
                ... and 2608 other files
                test/
                    11482c0d.jpg (27.8 kB)
                    68edd063.jpg (11.5 kB)
                    ... and 2608 other files
                    test/
            train/
                6fc790ac.jpg (17.1 kB)
                f4bb7bb0.jpg (11.6 kB)
                ... and 7238 other files
                train/
                    6fc790ac.jpg (17.1 kB)
                    f4bb7bb0.jpg (11.6 kB)
                    ... and 7238 other files
                    train/
            whale-categorization-playground/
                description.md (51 lines)
                sample_submission.csv (2611 lines)
                ... and 5 other files
                test/
                    11482c0d.jpg (27.8 kB)
                    68edd063.jpg (11.5 kB)
                    ... and 2608 other files
                    test/
                train/
                    6fc790ac.jpg (17.1 kB)
                    f4bb7bb0.jpg (11.6 kB)
                    ... and 7238 other files
                    train/
                whale-categorization-playground/
        working/
            whale-categorization-playground/
                description.md (51 lines)
                sample_submission.csv (2611 lines)
                ... and 5 other files
                test/
                    11482c0d.jpg (27.8 kB)
                    68edd063.jpg (11.5 kB)
                    ... and 2608 other files
                    test/
                train/
                    6fc790ac.jpg (17.1 kB)
                    f4bb7bb0.jpg (11.6 kB)
                    ... and 7238 other files
                    train/
                whale-categorization-playground/
```

-> data/sample_submission.csv has 2610 rows and 2 columns.
The columns are: Image, Id

-> data/train.csv has 7240 rows and 2 columns.
The columns are: Image, Id

-> data/whale-categorization-playground/sample_submission.csv has 2610 rows and 2 columns.
The columns are: Image, Id

-> data/whale-categorization-playground/train.csv has 7240 rows and 2 columns.
The columns are: Image, Id

-> input/sample_submission.csv has 2610 rows and 2 columns.
The columns are: Image, Id

-> input/train.csv has 7240 rows and 2 columns.
The columns are: Image, Id

-> (stopped after 10 files for performance)

# 5. Target score

0.00093

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import time
import math

import numpy as np
import pandas as pd

import matplotlib.image as mpimg

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


## === cell 1
def rgb_to_gray(img):
    if len(img.shape) == 2: #already gray
        return(img)
    grayImage = np.zeros(img.shape)
    R = img[:,:,0]*0.299
    G = img[:,:,1]*0.587
    B = img[:,:,2]*0.114
    
    grayImage = R + G + B
    
    return(grayImage)


## === cell 2
    
def img_compress(img, x_bins=100,y_bins=100):
    x_splits = np.linspace(0,img.shape[1]-1,x_bins+1, dtype = int)
    y_splits=np.linspace(0,img.shape[0]-1,y_bins+1, dtype = int)
    
    compressed = np.zeros((y_bins,x_bins))
    
    for i in range(y_bins):
        for j in range(x_bins):
            temp = np.mean(img[y_splits[i]:y_splits[i+1],
                                      x_splits[j]:x_splits[j+1]])
            if math.isnan(temp):
                if y_splits[i]==y_splits[i+1]:
                    compressed[i,j]=compressed[i-1,j]
                else:
                    compressed[i,j] = compressed[i,j-1]
            else:
                compressed[i,j] = int(temp)
    return(compressed)


## === cell 3
train_dir = '../input/train/train'
test_dir = '../input/test/test'
min_cols = 138
min_rows = 54


## === cell 4
%%time

train_file_names = os.listdir(train_dir)[::25]
imgs_train = [rgb_to_gray(mpimg.imread(train_dir + '/' + file, format = 'JPG')) 
              for file in train_file_names]
print(len(imgs_train), 'images loaded')


## === cell 5
good_pics = [i for i in range(len(imgs_train)) if (imgs_train[i].shape[0]>=min_rows)and(imgs_train[i].shape[1]>=min_cols)]
imgs_train = [imgs_train[i] for i in good_pics]


## === cell 6


compressed_train_imgs = [img_compress(img, min_cols, min_rows) for img in imgs_train]
del imgs_train


## === cell 7
filenames = [file for file in train_file_names]
filenames = [filenames[i] for i in good_pics]
df1 = pd.DataFrame(data = [img.ravel() for img in compressed_train_imgs])
df2 = pd.DataFrame(data = filenames, columns = ['FileName'])
data1 = pd.concat([df2,df1],axis = 1)
data2 = pd.read_csv('../input/train.csv',names = ['FileName','WhaleID'])
data = data2.merge(data1, on = 'FileName')
data = data.drop('FileName',axis = 1)
del compressed_train_imgs, df1, df2, data1, data2


## === cell 8
data.sample(5)


## === cell 9
X_train = data.iloc[:,1:]
y_train = data['WhaleID']


## === cell 10
%%time
pca = PCA(random_state = 42, n_components = 100, whiten = True)
pca.fit(X_train)


## === cell 11
logreg = LogisticRegression(C = 1e-2)

clf = Pipeline([('pca',pca),
                ('logreg',logreg),
                ])

clf.fit(X_train,y_train)


## === cell 12
print("Score on training set:", clf.score(X_train,y_train))
del data, X_train, y_train


## === cell 13
all_test_filenames = [file for file in os.listdir(test_dir)]
filenames_test = all_test_filenames[:100]
imgs_test = [rgb_to_gray(mpimg.imread(test_dir + '/' + file)) for file in filenames_test]
compressed_test_imgs = [img_compress(img, min_cols, min_rows) for img in imgs_test]
del imgs_test
X_test = pd.DataFrame([img.ravel() for img in compressed_test_imgs])
del compressed_test_imgs


## === cell 14
%%time
y_preds = clf.predict_proba(X_test)


## === cell 15
results = pd.DataFrame(data = [clf.classes_[np.argsort(y_preds[i,:])[-5:]] for i in range(y_preds.shape[0])],
                       index = filenames_test)
def list_to_str(L):
    string = ""
    for word in L:
        string = string + word + " "
    return(string)
        
results2 = pd.DataFrame(data = [list_to_str(results.iloc[i].values) for i in range(results.shape[0])],
                        index = filenames_test,columns = ['Id'])


## === cell 16
most_common = results2['Id'].value_counts().index[0]
missing_files = [x for x in all_test_filenames if x not in filenames_test]
full_results_df = pd.concat([
    results2,
    pd.DataFrame([most_common]*len(missing_files), 
                 index=missing_files,
                columns = ['Id'])
])


## === cell 17
full_results_df.sample(3)


## === cell 18
full_results_df.to_csv('submission.csv', 
                sep = ',', 
                index_label = 'Image', 
                header = True)
