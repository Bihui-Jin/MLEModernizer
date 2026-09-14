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

- What this solution (achieved 0.11697) has done: 'I fix the crash by filtering out non-image entries inside the test folder (your `../input/test` contains a nested `test/` directory, which caused `IsADirectoryError`). I also make the train/test directory resolution robust to both `../input/...` and `/kaggle/input/...` layouts so it runs reliably in this environment without changing the modeling approach. Finally, I ensure the submission is aligned to `sample_submission.csv` ordering and always writes a valid `submission.csv` with `Image,Id` columns so Kaggle accepts it.'

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

    compressed = np.zeros((y_bins, x_bins), dtype=np.float32)

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
def _first_existing_dir(candidates):
    for p in candidates:
        if os.path.isdir(p):
            return p
    return None


def _first_existing_file(candidates):
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


train_dir = _first_existing_dir(
    [
        "../input/train",
        "/kaggle/input/train",
        "/kaggle/input/whale-categorization-playground/train",
    ]
)
test_dir = _first_existing_dir(
    [
        "../input/test",
        "/kaggle/input/test",
        "/kaggle/input/whale-categorization-playground/test",
    ]
)
train_csv = _first_existing_file(
    [
        "../input/train.csv",
        "/kaggle/input/train.csv",
        "/kaggle/input/whale-categorization-playground/train.csv",
    ]
)
sample_sub_csv = _first_existing_file(
    [
        "../input/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
        "/kaggle/input/whale-categorization-playground/sample_submission.csv",
    ]
)

min_cols = 138
min_rows = 54

assert train_dir is not None, "Could not find train directory"
assert test_dir is not None, "Could not find test directory"
assert train_csv is not None, "Could not find train.csv"
assert sample_sub_csv is not None, "Could not find sample_submission.csv"

print("Using paths:")
print(" train_dir:", train_dir)
print(" test_dir :", test_dir)
print(" train_csv:", train_csv)
print(" sample_sub_csv:", sample_sub_csv)



## === cell 4
t0 = time.time()

TRAIN_STRIDE = 250  # was 25

train_file_names = os.listdir(train_dir)[::TRAIN_STRIDE]
imgs_train = [
    rgb_to_gray(mpimg.imread(os.path.join(train_dir, file), format="JPG"))
    for file in train_file_names
    if os.path.isfile(os.path.join(train_dir, file))
]
print(len(imgs_train), "images loaded")
print("Load time (s):", round(time.time() - t0, 2))



## === cell 5
good_pics = [
    i
    for i in range(len(imgs_train))
    if (imgs_train[i].shape[0] >= min_rows) and (imgs_train[i].shape[1] >= min_cols)
]
imgs_train = [imgs_train[i] for i in good_pics]



## === cell 6
compressed_train_imgs = [img_compress(img, min_cols, min_rows) for img in imgs_train]
del imgs_train



## === cell 7
filenames = [file for file in train_file_names]
filenames = [filenames[i] for i in good_pics]

df1 = pd.DataFrame(data=[img.ravel() for img in compressed_train_imgs])
df2 = pd.DataFrame(data=filenames, columns=["Image"])
data1 = pd.concat([df2, df1], axis=1)

data2 = pd.read_csv(train_csv)  # columns: Image, Id
data2 = data2.rename(columns={"Id": "WhaleID"})

data = data2.merge(data1, on="Image", how="inner")
data = data.drop("Image", axis=1)

del compressed_train_imgs, df1, df2, data1, data2



## === cell 8
_ = data.sample(5, random_state=42)



## === cell 9
X_train = data.iloc[:, 1:]
y_train = data["WhaleID"]



## === cell 10
t0 = time.time()
pca = PCA(random_state=42, n_components=100, whiten=True)
pca.fit(X_train)
print("PCA fit time (s):", round(time.time() - t0, 2))



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2620268916.py in <cell line: 0>()
      1 t0 = time.time()
      2 pca = PCA(random_state=42, n_components=100, whiten=True)
----> 3 pca.fit(X_train)
      4 print("PCA fit time (s):", round(time.time() - t0, 2))
      5 

/usr/local/lib/python3.11/dist-packages/sklearn/decomposition/_pca.py in fit(self, X, y)
    433         self._validate_params()
    434 
--> 435         self._fit(X)
    436         return self
    437 

/usr/local/lib/python3.11/dist-packages/sklearn/decomposition/_pca.py in _fit(self, X)
    510         # Call different fits for either full or truncated SVD
    511         if self._fit_svd_solver == "full":
--> 512             return self._fit_full(X, n_components)
    513         elif self._fit_svd_solver in ["arpack", "randomized"]:
    514             return self._fit_truncated(X, n_components, self._fit_svd_solver)

/usr/local/lib/python3.11/dist-packages/sklearn/decomposition/_pca.py in _fit_full(self, X, n_components)
    524                 )
    525         elif not 0 <= n_components <= min(n_samples, n_features):
--> 526             raise ValueError(
    527                 "n_components=%r must be between 0 and "
    528                 "min(n_samples, n_features)=%r with "

ValueError: n_components=100 must be between 0 and min(n_samples, n_features)=29 with svd_solver='full'

## === cell 11
logreg = LogisticRegression(C=1e-6, max_iter=1000)

clf = Pipeline(
    [
        ("pca", pca),
        ("logreg", logreg),
    ]
)

clf.fit(X_train, y_train)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4107522877.py in <cell line: 0>()
     10 )
     11 
---> 12 clf.fit(X_train, y_train)
     13 

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in fit(self, X, y, **fit_params)
    399         """
    400         fit_params_steps = self._check_fit_params(**fit_params)
--> 401         Xt = self._fit(X, y, **fit_params_steps)
    402         with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
    403             if self._final_estimator != "passthrough":

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in _fit(self, X, y, **fit_params_steps)
    357                 cloned_transformer = clone(transformer)
    358             # Fit or load from cache the current transformer
--> 359             X, fitted_transformer = fit_transform_one_cached(
    360                 cloned_transformer,
    361                 X,

/usr/local/lib/python3.11/dist-packages/joblib/memory.py in __call__(self, *args, **kwargs)
    324 
    325     def __call__(self, *args, **kwargs):
--> 326         return self.func(*args, **kwargs)
    327 
    328     def call_and_shelve(self, *args, **kwargs):

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in _fit_transform_one(transformer, X, y, weight, message_clsname, message, **fit_params)
    891     with _print_elapsed_time(message_clsname, message):
    892         if hasattr(transformer, "fit_transform"):
--> 893             res = transformer.fit_transform(X, y, **fit_params)
    894         else:
    895             res = transformer.fit(X, y, **fit_params).transform(X)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/decomposition/_pca.py in fit_transform(self, X, y)
    460         self._validate_params()
    461 
--> 462         U, S, Vt = self._fit(X)
    463         U = U[:, : self.n_components_]
    464 

/usr/local/lib/python3.11/dist-packages/sklearn/decomposition/_pca.py in _fit(self, X)
    510         # Call different fits for either full or truncated SVD
    511         if self._fit_svd_solver == "full":
--> 512             return self._fit_full(X, n_components)
    513         elif self._fit_svd_solver in ["arpack", "randomized"]:
    514             return self._fit_truncated(X, n_components, self._fit_svd_solver)

/usr/local/lib/python3.11/dist-packages/sklearn/decomposition/_pca.py in _fit_full(self, X, n_components)
    524                 )
    525         elif not 0 <= n_components <= min(n_samples, n_features):
--> 526             raise ValueError(
    527                 "n_components=%r must be between 0 and "
    528                 "min(n_samples, n_features)=%r with "

ValueError: n_components=100 must be between 0 and min(n_samples, n_features)=29 with svd_solver='full'

## === cell 12
print("Score on training set:", clf.score(X_train, y_train))
del data, X_train, y_train




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2787492356.py in <cell line: 0>()
----> 1 print("Score on training set:", clf.score(X_train, y_train))
      2 del data, X_train, y_train
      3 
      4 

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in score(self, X, y, sample_weight)
    716         Xt = X
    717         for _, name, transform in self._iter(with_final=False):
--> 718             Xt = transform.transform(Xt)
    719         score_params = {}
    720         if sample_weight is not None:

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/decomposition/_base.py in transform(self, X)
    119 
    120         X = self._validate_data(X, dtype=[np.float64, np.float32], reset=False)
--> 121         if self.mean_ is not None:
    122             X = X - self.mean_
    123         X_transformed = np.dot(X, self.components_.T)

AttributeError: 'PCA' object has no attribute 'mean_'

## === cell 13
def _is_image_file(fn):
    ext = os.path.splitext(fn)[1].lower()
    return ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


all_test_entries = sorted(os.listdir(test_dir))
filenames_test = [
    fn
    for fn in all_test_entries
    if os.path.isfile(os.path.join(test_dir, fn)) and _is_image_file(fn)
]

assert len(filenames_test) > 0, f"No image files found in test_dir={test_dir}"
print("Found test images:", len(filenames_test))

t0 = time.time()
imgs_test = [
    rgb_to_gray(mpimg.imread(os.path.join(test_dir, file))) for file in filenames_test
]
compressed_test_imgs = [img_compress(img, min_cols, min_rows) for img in imgs_test]
del imgs_test
X_test = pd.DataFrame([img.ravel() for img in compressed_test_imgs])
del compressed_test_imgs
print("Test feature build time (s):", round(time.time() - t0, 2))



## === cell 14
t0 = time.time()
y_preds = clf.predict_proba(X_test)
print("Predict_proba time (s):", round(time.time() - t0, 2))



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1079944138.py in <cell line: 0>()
      1 t0 = time.time()
----> 2 y_preds = clf.predict_proba(X_test)
      3 print("Predict_proba time (s):", round(time.time() - t0, 2))
      4 

/usr/local/lib/python3.11/dist-packages/sklearn/pipeline.py in predict_proba(self, X, **predict_proba_params)
    544         Xt = X
    545         for _, name, transform in self._iter(with_final=False):
--> 546             Xt = transform.transform(Xt)
    547         return self.steps[-1][1].predict_proba(Xt, **predict_proba_params)
    548 

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/decomposition/_base.py in transform(self, X)
    119 
    120         X = self._validate_data(X, dtype=[np.float64, np.float32], reset=False)
--> 121         if self.mean_ is not None:
    122             X = X - self.mean_
    123         X_transformed = np.dot(X, self.components_.T)

AttributeError: 'PCA' object has no attribute 'mean_'

## === cell 15
top5 = np.argsort(y_preds, axis=1)[:, -5:][:, ::-1]
top5_labels = clf.classes_[top5]


def list_to_str(L):
    return " ".join([str(x) for x in L])


results2 = pd.DataFrame(
    data=[list_to_str(top5_labels[i]) for i in range(top5_labels.shape[0])],
    index=filenames_test,
    columns=["Id"],
)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4001126752.py in <cell line: 0>()
----> 1 top5 = np.argsort(y_preds, axis=1)[:, -5:][:, ::-1]
      2 top5_labels = clf.classes_[top5]
      3 
      4 
      5 def list_to_str(L):

NameError: name 'y_preds' is not defined

## === cell 16
sample_sub = pd.read_csv(sample_sub_csv)
sample_sub = sample_sub[["Image"]].copy()

full_results_df = sample_sub.merge(
    results2.reset_index().rename(columns={"index": "Image"}),
    on="Image",
    how="left",
)

full_results_df["Id"] = full_results_df["Id"].fillna(
    "new_whale new_whale new_whale new_whale new_whale"
)

full_results_df.to_csv(
    "submission.csv",
    sep=",",
    index=False,
    header=True,
)
print("Wrote submission.csv with shape:", full_results_df.shape)
print(full_results_df.head())

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3093454908.py in <cell line: 0>()
      3 
      4 full_results_df = sample_sub.merge(
----> 5     results2.reset_index().rename(columns={"index": "Image"}),
      6     on="Image",
      7     how="left",

NameError: name 'results2' is not defined
