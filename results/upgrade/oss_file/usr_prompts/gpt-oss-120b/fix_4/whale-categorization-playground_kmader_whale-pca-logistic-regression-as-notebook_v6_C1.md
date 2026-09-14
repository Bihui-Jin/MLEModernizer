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

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import math

import numpy as np
import pandas as pd

import matplotlib.image as mpimg

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline




## === cell 1
def rgb_to_gray(img):
    if len(img.shape) == 2:  # already gray
        return img
    R = img[:, :, 0] * 0.299
    G = img[:, :, 1] * 0.587
    B = img[:, :, 2] * 0.114
    grayImage = R + G + B
    return grayImage




## === cell 2
def img_compress(img, x_bins=100, y_bins=100):
    x_splits = np.linspace(0, img.shape[1] - 1, x_bins + 1, dtype=int)
    y_splits = np.linspace(0, img.shape[0] - 1, y_bins + 1, dtype=int)

    compressed = np.zeros((y_bins, x_bins))

    for i in range(y_bins):
        for j in range(x_bins):
            temp = np.mean(
                img[y_splits[i] : y_splits[i + 1], x_splits[j] : x_splits[j + 1]]
            )
            if math.isnan(temp):
                if y_splits[i] == y_splits[i + 1]:
                    compressed[i, j] = compressed[i - 1, j]
                else:
                    compressed[i, j] = compressed[i, j - 1]
            else:
                compressed[i, j] = int(temp)
    return compressed




## === cell 3
base_path = "/kaggle/input"
train_dir = os.path.join(base_path, "train", "train")
test_dir = os.path.join(base_path, "test", "test")
min_cols = 138
min_rows = 54

train_file_names = os.listdir(train_dir)
imgs_train = [
    rgb_to_gray(mpimg.imread(os.path.join(train_dir, file)))
    for file in train_file_names
]
print(len(imgs_train), "images loaded")




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
IsADirectoryError                         Traceback (most recent call last)
/tmp/ipykernel_11/1999312495.py in <cell line: 0>()
      7 # Load **all** training images instead of a sparse subset
      8 train_file_names = os.listdir(train_dir)
----> 9 imgs_train = [
     10     rgb_to_gray(mpimg.imread(os.path.join(train_dir, file)))
     11     for file in train_file_names

/tmp/ipykernel_11/1999312495.py in <listcomp>(.0)
      8 train_file_names = os.listdir(train_dir)
      9 imgs_train = [
---> 10     rgb_to_gray(mpimg.imread(os.path.join(train_dir, file)))
     11     for file in train_file_names
     12 ]

/usr/local/lib/python3.11/dist-packages/matplotlib/image.py in imread(fname, format)
   1561             "``np.array(PIL.Image.open(urllib.request.urlopen(url)))``."
   1562             )
-> 1563     with img_open(fname) as image:
   1564         return (_pil_png_to_float_array(image)
   1565                 if isinstance(image, PIL.PngImagePlugin.PngImageFile) else

/usr/local/lib/python3.11/dist-packages/PIL/Image.py in open(fp, mode, formats)
   3511     if is_path(fp):
   3512         filename = os.fspath(fp)
-> 3513         fp = builtins.open(filename, "rb")
   3514         exclusive_fp = True
   3515     else:

IsADirectoryError: [Errno 21] Is a directory: '/kaggle/input/train/train/train'

## === cell 4
good_pics = [
    i
    for i in range(len(imgs_train))
    if (imgs_train[i].shape[0] >= min_rows) and (imgs_train[i].shape[1] >= min_cols)
]
imgs_train = [imgs_train[i] for i in good_pics]




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2856390536.py in <cell line: 0>()
      1 good_pics = [
      2     i
----> 3     for i in range(len(imgs_train))
      4     if (imgs_train[i].shape[0] >= min_rows) and (imgs_train[i].shape[1] >= min_cols)
      5 ]

NameError: name 'imgs_train' is not defined

## === cell 5
compressed_train_imgs = [img_compress(img, min_cols, min_rows) for img in imgs_train]
del imgs_train




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3032120709.py in <cell line: 0>()
----> 1 compressed_train_imgs = [img_compress(img, min_cols, min_rows) for img in imgs_train]
      2 del imgs_train
      3 
      4 

NameError: name 'imgs_train' is not defined

## === cell 6
filenames = [file for file in train_file_names]
filenames = [filenames[i] for i in good_pics]

df1 = pd.DataFrame(data=[img.ravel() for img in compressed_train_imgs])
df2 = pd.DataFrame(data=filenames, columns=["FileName"])
data1 = pd.concat([df2, df1], axis=1)

data2 = pd.read_csv(os.path.join(base_path, "train.csv"))
data2 = data2.rename(columns={"Image": "FileName", "Id": "WhaleID"})
data = data2.merge(data1, on="FileName")
data = data.drop("FileName", axis=1)
del compressed_train_imgs, df1, df2, data1, data2




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1807566353.py in <cell line: 0>()
      1 filenames = [file for file in train_file_names]
----> 2 filenames = [filenames[i] for i in good_pics]
      3 
      4 df1 = pd.DataFrame(data=[img.ravel() for img in compressed_train_imgs])
      5 df2 = pd.DataFrame(data=filenames, columns=["FileName"])

NameError: name 'good_pics' is not defined

## === cell 7
print(data.sample(5))




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/765211772.py in <cell line: 0>()
----> 1 print(data.sample(5))
      2 
      3 

NameError: name 'data' is not defined

## === cell 8
X_train = data.iloc[:, 1:]
y_train = data["WhaleID"]




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4066224436.py in <cell line: 0>()
----> 1 X_train = data.iloc[:, 1:]
      2 y_train = data["WhaleID"]
      3 
      4 

NameError: name 'data' is not defined

## === cell 9
pca = PCA(random_state=42, n_components=100, whiten=True)
pca.fit(X_train)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2920791293.py in <cell line: 0>()
      1 pca = PCA(random_state=42, n_components=100, whiten=True)
----> 2 pca.fit(X_train)
      3 
      4 

NameError: name 'X_train' is not defined

## === cell 10
logreg = LogisticRegression(C=1e-2, max_iter=1000)
clf = Pipeline([("pca", pca), ("logreg", logreg)])
clf.fit(X_train, y_train)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1957359660.py in <cell line: 0>()
      1 logreg = LogisticRegression(C=1e-2, max_iter=1000)
      2 clf = Pipeline([("pca", pca), ("logreg", logreg)])
----> 3 clf.fit(X_train, y_train)
      4 
      5 

NameError: name 'X_train' is not defined

## === cell 11
print("Score on training set:", clf.score(X_train, y_train))
del data, X_train, y_train




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2787492356.py in <cell line: 0>()
----> 1 print("Score on training set:", clf.score(X_train, y_train))
      2 del data, X_train, y_train
      3 
      4 

NameError: name 'X_train' is not defined

## === cell 12
all_test_filenames = os.listdir(test_dir)
filenames_test = all_test_filenames
imgs_test = [
    rgb_to_gray(mpimg.imread(os.path.join(test_dir, file))) for file in filenames_test
]
compressed_test_imgs = [img_compress(img, min_cols, min_rows) for img in imgs_test]
del imgs_test
X_test = pd.DataFrame([img.ravel() for img in compressed_test_imgs])
del compressed_test_imgs




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
IsADirectoryError                         Traceback (most recent call last)
/tmp/ipykernel_11/3564055344.py in <cell line: 0>()
      2 # Use **all** test images for a complete submission
      3 filenames_test = all_test_filenames
----> 4 imgs_test = [
      5     rgb_to_gray(mpimg.imread(os.path.join(test_dir, file))) for file in filenames_test
      6 ]

/tmp/ipykernel_11/3564055344.py in <listcomp>(.0)
      3 filenames_test = all_test_filenames
      4 imgs_test = [
----> 5     rgb_to_gray(mpimg.imread(os.path.join(test_dir, file))) for file in filenames_test
      6 ]
      7 compressed_test_imgs = [img_compress(img, min_cols, min_rows) for img in imgs_test]

/usr/local/lib/python3.11/dist-packages/matplotlib/image.py in imread(fname, format)
   1561             "``np.array(PIL.Image.open(urllib.request.urlopen(url)))``."
   1562             )
-> 1563     with img_open(fname) as image:
   1564         return (_pil_png_to_float_array(image)
   1565                 if isinstance(image, PIL.PngImagePlugin.PngImageFile) else

/usr/local/lib/python3.11/dist-packages/PIL/Image.py in open(fp, mode, formats)
   3511     if is_path(fp):
   3512         filename = os.fspath(fp)
-> 3513         fp = builtins.open(filename, "rb")
   3514         exclusive_fp = True
   3515     else:

IsADirectoryError: [Errno 21] Is a directory: '/kaggle/input/test/test/test'

## === cell 13
y_preds = clf.predict_proba(X_test)




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2435435360.py in <cell line: 0>()
----> 1 y_preds = clf.predict_proba(X_test)
      2 
      3 

NameError: name 'X_test' is not defined

## === cell 14
results = pd.DataFrame(
    data=[
        clf.classes_[np.argsort(y_preds[i, :])[-5:]] for i in range(y_preds.shape[0])
    ],
    index=filenames_test,
)


def list_to_str(L):
    return " ".join(L)


results2 = pd.DataFrame(
    data=[list_to_str(results.iloc[i].values) for i in range(results.shape[0])],
    index=filenames_test,
    columns=["Id"],
)




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3324947110.py in <cell line: 0>()
      1 results = pd.DataFrame(
      2     data=[
----> 3         clf.classes_[np.argsort(y_preds[i, :])[-5:]] for i in range(y_preds.shape[0])
      4     ],
      5     index=filenames_test,

NameError: name 'y_preds' is not defined

## === cell 15
most_common = results2["Id"].value_counts().index[0]
missing_files = [x for x in all_test_filenames if x not in filenames_test]
full_results_df = pd.concat(
    [
        results2,
        pd.DataFrame(
            [most_common] * len(missing_files), index=missing_files, columns=["Id"]
        ),
    ]
)




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3818296752.py in <cell line: 0>()
----> 1 most_common = results2["Id"].value_counts().index[0]
      2 missing_files = [x for x in all_test_filenames if x not in filenames_test]
      3 full_results_df = pd.concat(
      4     [
      5         results2,

NameError: name 'results2' is not defined

## === cell 16
print(full_results_df.sample(3))




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3670609642.py in <cell line: 0>()
----> 1 print(full_results_df.sample(3))
      2 
      3 

NameError: name 'full_results_df' is not defined

## === cell 17
full_results_df.to_csv("submission.csv", sep=",", index_label="Image", header=True)

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2512376298.py in <cell line: 0>()
----> 1 full_results_df.to_csv("submission.csv", sep=",", index_label="Image", header=True)

NameError: name 'full_results_df' is not defined
