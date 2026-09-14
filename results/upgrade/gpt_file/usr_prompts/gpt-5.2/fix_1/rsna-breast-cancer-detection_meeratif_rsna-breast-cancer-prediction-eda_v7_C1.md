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
Detect breast cancer in mammograms.

## Metric
[Probabilistic F1 score](https://aclanthology.org/2020.eval4nlp-1.9.pdf) (pF1). This extension of the traditional F score accepts probabilities instead of binary classifications. 

With pX as the probabilistic version of X:

$$
pF_1 = 2 \frac{pPrecision \cdot pRecall}{pPrecision + pRecall}
$$

where:

$$
pPrecision = \frac{pTP}{pTP + pFP}
$$

$$
pRecall = \frac{pTP}{TP + FN}
$$

## Submission Format
For each `prediction_id`, you should predict the likelihood of cancer in the corresponding `cancer` column. The submission file should have the following format:

```
prediction_id,cancer
0-L,0
0-R,0.5
0-R,0.5
1-L,1
...
# Dataset

**[train/test]_images/[patient_id]/[image_id].dcm** The mammograms, in dicom format. You can expect roughly 8,000 patients in the hidden test set. There are usually but not always 4 images per patient. Note that many of the images use the jpeg 2000 format which may you may need special libraries to load.

**sample_submission.csv** A valid sample submission.

**[train/test].csv** Metadata for each patient and image. Only the first few rows of the test set are available for download.

- `site_id` - ID code for the source hospital.
- `patient_id` - ID code for the patient.
- `image_id` - ID code for the image.
- `laterality` - Whether the image is of the left or right breast.
- `view` - The orientation of the image. The default for a screening exam is to capture two views per breast.
- `age` - The patient's age in years.
- `implant` - Whether or not the patient had breast implants. Site 1 only provides breast implant information at the patient level, not at the breast level.
- `density` - A rating for how dense the breast tissue is, with A being the least dense and D being the most dense. Extremely dense tissue can make diagnosis more difficult. Only provided for train.
- `machine_id` - An ID code for the imaging device.
- `cancer` - Whether or not the breast was positive for malignant cancer. The target value. Only provided for train.
- `biopsy` - Whether or not a follow-up biopsy was performed on the breast. Only provided for train.
- `invasive` - If the breast is positive for cancer, whether or not the cancer proved to be invasive. Only provided for train.
- `BIRADS` - 0 if the breast required follow-up, 1 if the breast was rated as negative for cancer, and 2 if the breast was rated as normal. Only provided for train.
- `prediction_id` - The ID for the matching submission row. Multiple images will share the same prediction ID. Test only.
- `difficult_negative_case` - True if the case was unusually difficult. Only provided for train.

# 2. Python version

3.11

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pydicom==3.0.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tf_keras==2.18.0
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (190 lines)
            sample_submission.csv (2385 lines)
            sample_submission.csv.zip (6.5 kB)
            test.csv (5475 lines)
            test.csv.zip (60.6 kB)
            test.zip (160 Bytes)
            test_images.zip (29.1 GB)
            train.csv (49233 lines)
            train.csv.zip (513.7 kB)
            train.zip (162 Bytes)
            train_images.zip (260.7 GB)
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
            test_images/
                10116/
                    1470873094.dcm (8.5 MB)
                    472095321.dcm (5.6 MB)
                    ... and 2 other files
                10130/
                    1013166704.dcm (9.7 MB)
                    1165309236.dcm (8.7 MB)
                    ... and 5 other files
                ... and 1191 other folders
            train_images/
                10006/
                    1459541791.dcm (4.4 MB)
                    1864590858.dcm (4.0 MB)
                    ... and 2 other files
                10011/
                    1031443799.dcm (2.1 MB)
                    220375232.dcm (1.7 MB)
                    ... and 2 other files
                ... and 10720 other folders
        input/
            description.md (190 lines)
            sample_submission.csv (2385 lines)
            sample_submission.csv.zip (6.5 kB)
            test.csv (5475 lines)
            test.csv.zip (60.6 kB)
            test.zip (160 Bytes)
            test_images.zip (29.1 GB)
            train.csv (49233 lines)
            train.csv.zip (513.7 kB)
            train.zip (162 Bytes)
            train_images.zip (260.7 GB)
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
            test_images/
                10116/
                    1470873094.dcm (8.5 MB)
                    472095321.dcm (5.6 MB)
                    ... and 2 other files
                10130/
                    1013166704.dcm (9.7 MB)
                    1165309236.dcm (8.7 MB)
                    ... and 5 other files
                ... and 1191 other folders
            train_images/
                10006/
                    1459541791.dcm (4.4 MB)
                    1864590858.dcm (4.0 MB)
                    ... and 2 other files
                10011/
                    1031443799.dcm (2.1 MB)
                    220375232.dcm (1.7 MB)
                    ... and 2 other files
                ... and 10720 other folders
        working/
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
```

-> data/rsna-breast-cancer-detection/sample_submission.csv has 2384 rows and 2 columns.
The columns are: prediction_id, cancer

-> data/rsna-breast-cancer-detection/test.csv has 5474 rows and 9 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, implant, machine_id, prediction_id

-> data/rsna-breast-cancer-detection/train.csv has 49232 rows and 14 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, cancer, biopsy, invasive, BIRADS, implant, density, machine_id, difficult_negative_case

-> data/sample_submission.csv has 2384 rows and 2 columns.
The columns are: prediction_id, cancer

-> data/test.csv has 5474 rows and 9 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, implant, machine_id, prediction_id

-> data/train.csv has 49232 rows and 14 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, cancer, biopsy, invasive, BIRADS, implant, density, machine_id, difficult_negative_case

-> (stopped after 10 files for performance)

# 5. Target score

0.02

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
import pydicom
import random
import matplotlib.pyplot as plt
import seaborn as sns
import PIL
import tqdm
import cv2


## === cell 2
from keras.preprocessing.image import ImageDataGenerator
train = pd.read_csv('/kaggle/input/rsna-breast-cancer-detection/train.csv')
test = pd.read_csv('/kaggle/input/rsna-breast-cancer-detection/test.csv')
test1 = pd.read_csv('/kaggle/input/rsna-breast-cancer-detection/test.csv')
img_data  = "/kaggle/input/rsna-breast-cancer-detection"


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
test


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1939250918.py in <cell line: 0>()
----> 1 test

NameError: name 'test' is not defined

## === cell 4
train


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3436609520.py in <cell line: 0>()
----> 1 train

NameError: name 'train' is not defined

## === cell 5
import glob
train_images  = glob.glob(img_data+"/train_images/*/*")
(train_images)


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3117305640.py in <cell line: 0>()
      1 import glob
----> 2 train_images  = glob.glob(img_data+"/train_images/*/*")
      3 (train_images)

NameError: name 'img_data' is not defined

## === cell 6
train.info()


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3994350071.py in <cell line: 0>()
----> 1 train.info()

NameError: name 'train' is not defined

## === cell 7
train.isnull().sum()


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/102480753.py in <cell line: 0>()
----> 1 train.isnull().sum()

NameError: name 'train' is not defined

## === cell 9
train = train.fillna(train.mean())
test = test.fillna(test.mean())


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1353366467.py in <cell line: 0>()
----> 1 train = train.fillna(train.mean())
      2 test = test.fillna(test.mean())

NameError: name 'train' is not defined

## === cell 10
bl_col = train.select_dtypes(include=('boolean'))
int_col = train.select_dtypes(include=('int'))
str_col = train.select_dtypes(include=('object'))
flt_col = train.select_dtypes(include=('float'))


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4029465344.py in <cell line: 0>()
----> 1 bl_col = train.select_dtypes(include=('boolean'))
      2 int_col = train.select_dtypes(include=('int'))
      3 str_col = train.select_dtypes(include=('object'))
      4 flt_col = train.select_dtypes(include=('float'))

NameError: name 'train' is not defined

## === cell 11
for i, col in enumerate(str_col):
    plt.figure(i)
    sns.countplot(x=col, data=str_col)


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2867705500.py in <cell line: 0>()
----> 1 for i, col in enumerate(str_col):
      2     plt.figure(i)
      3     sns.countplot(x=col, data=str_col)

NameError: name 'str_col' is not defined

## === cell 12
for i, col in enumerate(bl_col):
    plt.figure(i)
    sns.countplot(x=col, data=bl_col)


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3438875635.py in <cell line: 0>()
----> 1 for i, col in enumerate(bl_col):
      2     plt.figure(i)
      3     sns.countplot(x=col, data=bl_col)

NameError: name 'bl_col' is not defined

## === cell 13
print(train.patient_id.nunique())
print(train.site_id.nunique())
print(train.image_id.nunique())


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4059999216.py in <cell line: 0>()
----> 1 print(train.patient_id.nunique())
      2 print(train.site_id.nunique())
      3 print(train.image_id.nunique())

NameError: name 'train' is not defined

## === cell 14
sns.jointplot(data=train, x='biopsy', y='age')
sns.jointplot(data=train, x='cancer', y='age')
sns.jointplot(data=train, x='invasive', y='age')
sns.jointplot(data=train, x='implant', y='age')


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1720888967.py in <cell line: 0>()
----> 1 sns.jointplot(data=train, x='biopsy', y='age')
      2 sns.jointplot(data=train, x='cancer', y='age')
      3 sns.jointplot(data=train, x='invasive', y='age')
      4 sns.jointplot(data=train, x='implant', y='age')

NameError: name 'train' is not defined

## === cell 15
train.describe().T


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4026630191.py in <cell line: 0>()
----> 1 train.describe().T

NameError: name 'train' is not defined

## === cell 17
test.info()


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3722404436.py in <cell line: 0>()
----> 1 test.info()

NameError: name 'test' is not defined

## === cell 18
train = pd.get_dummies(train, columns=['laterality', 'view', 'implant'])
test = pd.get_dummies(test, columns=['laterality', 'view', 'implant'])


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/779743588.py in <cell line: 0>()
      5 # for col in str_col:
      6 #     test[col] = label_encoding.fit_transform(test[col].astype('str'))
----> 7 train = pd.get_dummies(train, columns=['laterality', 'view', 'implant'])
      8 test = pd.get_dummies(test, columns=['laterality', 'view', 'implant'])

NameError: name 'train' is not defined

## === cell 19
def fun_process(path:str):
    read_dcm = pydicom.dcmread(path)
    print(read_dcm)
    print("\n")
    img = read_dcm.pixel_array
    print(img)
    print("\n", img.shape, "\n")
    img_show = PIL.Image.fromarray(img)
    plt.figure(figsize=(6,6))
    plt.imshow(img_show)
    img_show.save('show1.png')
fun_process(train_images[1])


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3892217969.py in <cell line: 0>()
     10     plt.imshow(img_show)
     11     img_show.save('show1.png')
---> 12 fun_process(train_images[1])

NameError: name 'train_images' is not defined

## === cell 20
img_show  = PIL.Image.open(r"/kaggle/working/show1.png") 
plt.imshow(img_show)
plt.show()


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3621409299.py in <cell line: 0>()
----> 1 img_show  = PIL.Image.open(r"/kaggle/working/show1.png")
      2 plt.imshow(img_show)
      3 plt.show()

/usr/local/lib/python3.11/dist-packages/PIL/Image.py in open(fp, mode, formats)
   3511     if is_path(fp):
   3512         filename = os.fspath(fp)
-> 3513         fp = builtins.open(filename, "rb")
   3514         exclusive_fp = True
   3515     else:

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/show1.png'

## === cell 21
from multiprocessing import Pool,cpu_count


## === cell 22
for file_name in["train", "test"]:
     os.makedirs(file_name, exist_ok=True)


## === cell 24
!ls


## === cell 27
train


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3436609520.py in <cell line: 0>()
----> 1 train

NameError: name 'train' is not defined

## === cell 28
train.isnull().sum()


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/102480753.py in <cell line: 0>()
----> 1 train.isnull().sum()

NameError: name 'train' is not defined

## === cell 29
df_new_0 = train[train['cancer']==0]
df_new_1 = train[train['cancer']==1]


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1200610376.py in <cell line: 0>()
----> 1 df_new_0 = train[train['cancer']==0]
      2 df_new_1 = train[train['cancer']==1]

NameError: name 'train' is not defined

## === cell 30
from sklearn.model_selection import *
from sklearn.metrics import *
from sklearn.model_selection import * 
from sklearn.utils import *
from sklearn.preprocessing import *
from sklearn.linear_model import *


## === cell 31
df_new_sampled = resample(df_new_1, replace=True, n_samples=len(df_new_1), random_state=20)


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1420847023.py in <cell line: 0>()
----> 1 df_new_sampled = resample(df_new_1, replace=True, n_samples=len(df_new_1), random_state=20)

NameError: name 'df_new_1' is not defined

## === cell 32
data_upsampled = pd.concat([df_new_0, df_new_sampled])


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/548094470.py in <cell line: 0>()
----> 1 data_upsampled = pd.concat([df_new_0, df_new_sampled])

NameError: name 'df_new_0' is not defined

## === cell 33
div_col_scale = ['age', 'machine_id']


## === cell 34
x = data_upsampled.drop('cancer', axis=1)
y = data_upsampled['cancer']


## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4275831871.py in <cell line: 0>()
----> 1 x = data_upsampled.drop('cancer', axis=1)
      2 y = data_upsampled['cancer']

NameError: name 'data_upsampled' is not defined

## === cell 35
y


## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/859018229.py in <cell line: 0>()
----> 1 y

NameError: name 'y' is not defined

## === cell 36
stand_data = StandardScaler()
x[div_col_scale] = stand_data.fit_transform(x[div_col_scale])


## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/881677238.py in <cell line: 0>()
      1 stand_data = StandardScaler()
----> 2 x[div_col_scale] = stand_data.fit_transform(x[div_col_scale])

NameError: name 'x' is not defined

## === cell 37
all_col = set(x.columns) & set(test.columns)
x = x[list(all_col)]


## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1291246258.py in <cell line: 0>()
----> 1 all_col = set(x.columns) & set(test.columns)
      2 x = x[list(all_col)]

NameError: name 'x' is not defined

## === cell 38
test[div_col_scale] = stand_data.transform(test[div_col_scale])
test = test[list(all_col - {'prediction_id'})]


## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1600742238.py in <cell line: 0>()
----> 1 test[div_col_scale] = stand_data.transform(test[div_col_scale])
      2 test = test[list(all_col - {'prediction_id'})]

NameError: name 'test' is not defined

## === cell 39
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.33, random_state=42)


## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/805636407.py in <cell line: 0>()
----> 1 x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.33, random_state=42)

NameError: name 'x' is not defined

## === cell 40
y_train.value_counts()


## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2314264361.py in <cell line: 0>()
----> 1 y_train.value_counts()

NameError: name 'y_train' is not defined

## === cell 41
x_val


## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2729026052.py in <cell line: 0>()
----> 1 x_val

NameError: name 'x_val' is not defined

## === cell 42
l_r = LogisticRegression(random_state=42)
l_r.fit(x_train, y_train)
y_pred = l_r.predict(x_val)


## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3596215469.py in <cell line: 0>()
      1 l_r = LogisticRegression(random_state=42)
----> 2 l_r.fit(x_train, y_train)
      3 y_pred = l_r.predict(x_val)

NameError: name 'x_train' is not defined

## === cell 43
y_pred


## --- ERROR in cell 43, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3830458035.py in <cell line: 0>()
----> 1 y_pred

NameError: name 'y_pred' is not defined

## === cell 44
test[div_col_scale] = stand_data.transform(test[div_col_scale])


## --- ERROR in cell 44, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1220549068.py in <cell line: 0>()
----> 1 test[div_col_scale] = stand_data.transform(test[div_col_scale])

NameError: name 'test' is not defined

## === cell 46
x_val.info()


## --- ERROR in cell 46, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4238347935.py in <cell line: 0>()
----> 1 x_val.info()

NameError: name 'x_val' is not defined

## === cell 47
test.info()


## --- ERROR in cell 47, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3722404436.py in <cell line: 0>()
----> 1 test.info()

NameError: name 'test' is not defined

## === cell 48
test


## --- ERROR in cell 48, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1939250918.py in <cell line: 0>()
----> 1 test

NameError: name 'test' is not defined

## === cell 49
test1


## --- ERROR in cell 49, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2079699178.py in <cell line: 0>()
----> 1 test1

NameError: name 'test1' is not defined

## === cell 50
y_pred1 = l_r.predict_proba(test)[:, 1]
prediction_ids = test1['prediction_id'].copy()


## --- ERROR in cell 50, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1424024218.py in <cell line: 0>()
----> 1 y_pred1 = l_r.predict_proba(test)[:, 1]
      2 prediction_ids = test1['prediction_id'].copy()

NameError: name 'test' is not defined

## === cell 51
submission = pd.DataFrame({'prediction_id': prediction_ids, 'cancer': y_pred1}).groupby('prediction_id').mean().reset_index()
submission.to_csv('submission.csv', index=False)


## --- ERROR in cell 51, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3319983763.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({'prediction_id': prediction_ids, 'cancer': y_pred1}).groupby('prediction_id').mean().reset_index()
      2 submission.to_csv('submission.csv', index=False)

NameError: name 'prediction_ids' is not defined
