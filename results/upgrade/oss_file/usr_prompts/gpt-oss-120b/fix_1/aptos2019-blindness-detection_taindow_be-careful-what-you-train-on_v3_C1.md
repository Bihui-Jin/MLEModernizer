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
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
shap==0.44.1
shapely==2.1.2
sklearn-pandas==2.2.0
tqdm==4.67.1
xgboost==2.0.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.463379

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import cv2
import shap
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, confusion_matrix

sns.set(rc={'figure.figsize':(11.7,8.27)})


## === cell 1

train = pd.read_csv('../input/train.csv')
train_results = []

for index, row in tqdm(train.iterrows(), total=train.shape[0]):
    img = cv2.imread('../input//train_images/{}.png'.format(row['id_code']))

    height, width, channels = img.shape
    ratio = width/height
    pixel_count = width*height
    gray_scaled = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    black_cnt = pixel_count-cv2.countNonZero(gray_scaled)
    black_pct = black_cnt/pixel_count
    mean0, mean1, mean2, _ = cv2.mean(img)

    observation = np.array(
        (row['diagnosis'], height, width, ratio, pixel_count, black_cnt, black_pct, mean0, mean1, mean2))

    train_results.append(observation)

train_results_df = pd.DataFrame(train_results)
train_results_df.columns = [
    'diagnosis','height','width','ratio','pixel_count','black_cnt','black_pct','mean_c0','mean_c1','mean_c2']


## === cell 2

test = pd.read_csv('../input/test.csv')
test_results = []

for index, row in tqdm(test.iterrows(), total=test.shape[0]):
    img = cv2.imread('../input//test_images/{}.png'.format(row['id_code']))

    height, width, channels = img.shape
    ratio = width/height
    pixel_count = width*height
    gray_scaled = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    black_cnt = pixel_count-cv2.countNonZero(gray_scaled)
    black_pct = black_cnt/pixel_count
    mean0, mean1, mean2, _ = cv2.mean(img)

    observation = np.array(
        (np.nan, height, width, ratio, pixel_count, black_cnt, black_pct, mean0, mean1, mean2))

    test_results.append(observation)

test_results_df = pd.DataFrame(test_results)
test_results_df.columns = [
    'diagnosis','height','width','ratio','pixel_count','black_cnt','black_pct','mean_c0','mean_c1','mean_c2']


## === cell 3
params = {
    'booster': 'gbtree',
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'eta': 0.005,
    'max_depth': 10,
    'subsample': 1.0,
    'colsample_bytree': 1.0,
    'tree_method': 'gpu_hist',
    'num_class': 5}


## === cell 4
X = train_results_df.drop(columns=['diagnosis'])
y = train_results_df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1337)

dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_test, label=y_test)

watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

bst = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=100000,
    evals=watchlist,
    early_stopping_rounds=250,
    verbose_eval=100
)


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
XGBoostError                              Traceback (most recent call last)
/tmp/ipykernel_11/3241303046.py in <cell line: 0>()
      9 watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
     10 
---> 11 bst = xgb.train(
     12     params=params,
     13     dtrain=dtrain,

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in inner_f(*args, **kwargs)
    728             for k, arg in zip(sig.parameters, args):
    729                 kwargs[k] = arg
--> 730             return func(**kwargs)
    731 
    732         return inner_f

/usr/local/lib/python3.11/dist-packages/xgboost/training.py in train(params, dtrain, num_boost_round, evals, obj, feval, maximize, early_stopping_rounds, evals_result, verbose_eval, xgb_model, callbacks, custom_metric)
    179         if cb_container.before_iteration(bst, i, dtrain, evals):
    180             break
--> 181         bst.update(dtrain, i, obj)
    182         if cb_container.after_iteration(bst, i, dtrain, evals):
    183             break

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in update(self, dtrain, iteration, fobj)
   2048 
   2049         if fobj is None:
-> 2050             _check_call(
   2051                 _LIB.XGBoosterUpdateOneIter(
   2052                     self.handle, ctypes.c_int(iteration), dtrain.handle

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in _check_call(ret)
    280     """
    281     if ret != 0:
--> 282         raise XGBoostError(py_str(_LIB.XGBGetLastError()))
    283 
    284 

XGBoostError: [03:06:59] /workspace/src/tree/updater_gpu_hist.cu:781: Exception in gpu_hist: [03:06:59] /workspace/src/tree/updater_gpu_hist.cu:787: Check failed: ctx_->gpu_id >= 0 (-1 vs. 0) : Must have at least one device
Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb27f2a) [0x7ffeb125bf2a]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb3e95a) [0x7ffeb127295a]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb483cd) [0x7ffeb127c3cd]
  [bt] (3) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x460c79) [0x7ffeb0b94c79]
  [bt] (4) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x461d09) [0x7ffeb0b95d09]
  [bt] (5) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x4c54f7) [0x7ffeb0bf94f7]
  [bt] (6) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGBoosterUpdateOneIter+0x70) [0x7ffeb0895ef0]
  [bt] (7) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7ffff63ace2e]
  [bt] (8) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7ffff63a9493]



Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb27f2a) [0x7ffeb125bf2a]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0xb485c9) [0x7ffeb127c5c9]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x460c79) [0x7ffeb0b94c79]
  [bt] (3) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x461d09) [0x7ffeb0b95d09]
  [bt] (4) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x4c54f7) [0x7ffeb0bf94f7]
  [bt] (5) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGBoosterUpdateOneIter+0x70) [0x7ffeb0895ef0]
  [bt] (6) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7ffff63ace2e]
  [bt] (7) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7ffff63a9493]
  [bt] (8) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0xa4d8) [0x7ffff63bc4d8]



## === cell 5
pred = pd.DataFrame(np.argmax(bst.predict(dvalid), axis=1))
results = pd.concat([y_test.reset_index(drop=True), pred], axis=1)
score = cohen_kappa_score(results.iloc[:,0], results.iloc[:,1], weights="quadratic")

print('Validation Kappa:', score)


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/471122854.py in <cell line: 0>()
----> 1 pred = pd.DataFrame(np.argmax(bst.predict(dvalid), axis=1))
      2 results = pd.concat([y_test.reset_index(drop=True), pred], axis=1)
      3 score = cohen_kappa_score(results.iloc[:,0], results.iloc[:,1], weights="quadratic")
      4 
      5 print('Validation Kappa:', score)

NameError: name 'bst' is not defined

## === cell 6
cm = confusion_matrix(y_true=results.iloc[:,0], y_pred=results.iloc[:,1])
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True)
ax.set(ylabel='True label', xlabel='Predicted label')


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1272455945.py in <cell line: 0>()
----> 1 cm = confusion_matrix(y_true=results.iloc[:,0], y_pred=results.iloc[:,1])
      2 cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      3 
      4 fig, ax = plt.subplots()
      5 sns.heatmap(cm, annot=True)

NameError: name 'results' is not defined

## === cell 7
train['diagnosis'].value_counts()/train.shape[0]


## === cell 8
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(bst, importance_type='gain', height=0.8, ax=ax)


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2523996396.py in <cell line: 0>()
      1 fig, ax = plt.subplots(figsize=(12,18))
----> 2 xgb.plot_importance(bst, importance_type='gain', height=0.8, ax=ax)

NameError: name 'bst' is not defined

## === cell 9
train_results_df.groupby(['ratio', 'diagnosis'])['pixel_count'].count()


## === cell 10
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(20,10))

img0a = cv2.imread(
    '../input/train_images/{}.png'.format(train.iloc[train_results_df[np.round(train_results_df['ratio'], 6)==1.000000].index[0], 0]))

ax[0,0].imshow(cv2.cvtColor(img0a, cv2.COLOR_BGR2RGB))
ax[0,0].axis('off')

img0b = cv2.imread(
    '../input/train_images/{}.png'.format(train.iloc[train_results_df[np.round(train_results_df['ratio'], 6)==1.000000].index[1], 0]))

ax[1,0].imshow(cv2.cvtColor(img0b, cv2.COLOR_BGR2RGB))
ax[1,0].axis('off')

img0c = cv2.imread(
    '../input/train_images/{}.png'.format(train.iloc[train_results_df[np.round(train_results_df['ratio'], 6)==1.000000].index[2], 0]))

ax[2,0].imshow(cv2.cvtColor(img0c, cv2.COLOR_BGR2RGB))
ax[2,0].axis('off')

img1a = cv2.imread(
    '../input/train_images/{}.png'.format(train.iloc[train_results_df[np.round(train_results_df['ratio'], 6)==1.333333].index[0], 0]))

ax[0,1].imshow(cv2.cvtColor(img1a, cv2.COLOR_BGR2RGB))
ax[0,1].axis('off')

img1b = cv2.imread(
    '../input/train_images/{}.png'.format(train.iloc[train_results_df[np.round(train_results_df['ratio'], 6)==1.333333].index[1], 0]))

ax[1,1].imshow(cv2.cvtColor(img1b, cv2.COLOR_BGR2RGB))
ax[1,1].axis('off')

img1c = cv2.imread(
    '../input/train_images/{}.png'.format(train.iloc[train_results_df[np.round(train_results_df['ratio'], 6)==1.333333].index[2], 0]))

ax[2,1].imshow(cv2.cvtColor(img1c, cv2.COLOR_BGR2RGB))
ax[2,1].axis('off')

img2a = cv2.imread(
    '../input/train_images/{}.png'.format(train.iloc[train_results_df[np.round(train_results_df['ratio'], 6)==1.505618].index[0], 0]))

ax[0,2].imshow(cv2.cvtColor(img2a, cv2.COLOR_BGR2RGB))
ax[0,2].axis('off')

img2b = cv2.imread(
    '../input/train_images/{}.png'.format(train.iloc[train_results_df[np.round(train_results_df['ratio'], 6)==1.505618].index[1], 0]))

ax[1,2].imshow(cv2.cvtColor(img2b, cv2.COLOR_BGR2RGB))
ax[1,2].axis('off')

img2c = cv2.imread(
    '../input/train_images/{}.png'.format(train.iloc[train_results_df[np.round(train_results_df['ratio'], 6)==1.505618].index[2], 0]))

ax[2,2].imshow(cv2.cvtColor(img2c, cv2.COLOR_BGR2RGB))
ax[2,2].axis('off')


## === cell 11
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(20,10))

img0a = cv2.imread(
    '../input/train_images/{}.png'.format(train.iloc[train_results_df[np.round(train_results_df['ratio'], 6)==1.000000].index[0], 0]))
img0a = cv2.resize(img0a, (224,224))
ax[0,0].imshow(cv2.cvtColor(img0a, cv2.COLOR_BGR2RGB))
ax[0,0].axis('off')

img0b = cv2.imread(
    '../input/train_images/{}.png'.format(train.iloc[train_results_df[np.round(train_results_df['ratio'], 6)==1.000000].index[1], 0]))
img0b = cv2.resize(img0b, (224,224))
ax[1,0].imshow(cv2.cvtColor(img0b, cv2.COLOR_BGR2RGB))
ax[1,0].axis('off')

img0c = cv2.imread(
    '../input/train_images/{}.png'.format(train.iloc[train_results_df[np.round(train_results_df['ratio'], 6)==1.000000].index[2], 0]))
img0c = cv2.resize(img0c, (224,224))
ax[2,0].imshow(cv2.cvtColor(img0c, cv2.COLOR_BGR2RGB))
ax[2,0].axis('off')

img1a = cv2.imread(
    '../input/train_images/{}.png'.format(train.iloc[train_results_df[np.round(train_results_df['ratio'], 6)==1.333333].index[0], 0]))
img1a = cv2.resize(img1a, (224,224))
ax[0,1].imshow(cv2.cvtColor(img1a, cv2.COLOR_BGR2RGB))
ax[0,1].axis('off')

img1b = cv2.imread(
    '../input/train_images/{}.png'.format(train.iloc[train_results_df[np.round(train_results_df['ratio'], 6)==1.333333].index[1], 0]))
img1b = cv2.resize(img1b, (224,224))
ax[1,1].imshow(cv2.cvtColor(img1b, cv2.COLOR_BGR2RGB))
ax[1,1].axis('off')

img1c = cv2.imread(
    '../input/train_images/{}.png'.format(train.iloc[train_results_df[np.round(train_results_df['ratio'], 6)==1.333333].index[2], 0]))
img1c = cv2.resize(img1c, (224,224))
ax[2,1].imshow(cv2.cvtColor(img1c, cv2.COLOR_BGR2RGB))
ax[2,1].axis('off')

img2a = cv2.imread(
    '../input/train_images/{}.png'.format(train.iloc[train_results_df[np.round(train_results_df['ratio'], 6)==1.505618].index[0], 0]))
img2a = cv2.resize(img2a, (224,224))
ax[0,2].imshow(cv2.cvtColor(img2a, cv2.COLOR_BGR2RGB))
ax[0,2].axis('off')

img2b = cv2.imread(
    '../input/train_images/{}.png'.format(train.iloc[train_results_df[np.round(train_results_df['ratio'], 6)==1.505618].index[1], 0]))
img2b = cv2.resize(img2b, (224,224))
ax[1,2].imshow(cv2.cvtColor(img2b, cv2.COLOR_BGR2RGB))
ax[1,2].axis('off')

img2c = cv2.imread(
    '../input/train_images/{}.png'.format(train.iloc[train_results_df[np.round(train_results_df['ratio'], 6)==1.505618].index[2], 0]))
img2c = cv2.resize(img2c, (224,224))
ax[2,2].imshow(cv2.cvtColor(img2c, cv2.COLOR_BGR2RGB))
ax[2,2].axis('off')


## === cell 12
dtest = xgb.DMatrix(test_results_df.drop(columns=['diagnosis']))
test_pred = pd.DataFrame(np.argmax(bst.predict(dtest), axis=1))
submission = pd.concat([test['id_code'], test_pred], axis=1)
submission.columns = ['id_code','diagnosis']
submission.to_csv('submission.csv', index=False)
submission.head()


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/301761620.py in <cell line: 0>()
      1 dtest = xgb.DMatrix(test_results_df.drop(columns=['diagnosis']))
----> 2 test_pred = pd.DataFrame(np.argmax(bst.predict(dtest), axis=1))
      3 submission = pd.concat([test['id_code'], test_pred], axis=1)
      4 submission.columns = ['id_code','diagnosis']
      5 submission.to_csv('submission.csv', index=False)

NameError: name 'bst' is not defined
