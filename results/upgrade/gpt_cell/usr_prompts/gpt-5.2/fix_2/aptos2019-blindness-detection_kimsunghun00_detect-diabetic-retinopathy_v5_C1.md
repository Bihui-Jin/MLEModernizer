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

3.7

# 2. Installed packages

No external packages required in the script and installed.

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm_notebook as tqdm
%matplotlib inline

pd.set_option('display.max_rows', 10)


## === cell 1
base_data_folder = "/kaggle/input"
train_data_folder = os.path.join(base_data_folder, "train_images")

print(os.listdir(base_data_folder))


## === cell 2
train_files_names = os.listdir(train_data_folder)
train_files_names[:5]


## === cell 3
train_images = []
for file in tqdm(glob.glob(train_data_folder + '/*.png')):
    image_bgr = cv2.imread(file, cv2.IMREAD_COLOR)
    image_resized = cv2.resize(image_bgr, dsize=(0,0), fx = 0.12, fy = 0.12)
    train_images.append(image_resized)


## === cell 4
len(train_images)


## === cell 5
train_labels = pd.read_csv(base_data_folder+"/train.csv", index_col = 0)
train_labels


## === cell 6
image_labels = pd.DataFrame(columns = ['id_code'])

for i in train_files_names:
    splited = i.split('.')[0]
    temp = pd.DataFrame({'id_code':[splited]})
    image_labels = pd.concat([image_labels, temp], ignore_index=True)

image_labels


## === cell 7
labels = pd.merge(image_labels, train_labels, on='id_code')
labels


## === cell 8
y_data = labels['diagnosis']
y_data[:5]


## === cell 9
labels['diagnosis'].hist()
labels['diagnosis'].value_counts()


## === cell 10
def crop_image_from_gray(img,tol=7):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray_img > tol
    
    img1=img[:,:,0][np.ix_(mask.any(axis=1),mask.any(axis=0))]
    img2=img[:,:,1][np.ix_(mask.any(axis=1),mask.any(axis=0))]
    img3=img[:,:,2][np.ix_(mask.any(axis=1),mask.any(axis=0))]
    img = np.stack([img1,img2,img3], axis=-1)
    
    return img


## === cell 11
def circle_crop(img):
    img = crop_image_from_gray(img)
    
    height, width, depth = img.shape
    largest_side = np.max((height, width))
    img = cv2.resize(img, dsize=(largest_side, largest_side),interpolation = cv2.INTER_CUBIC)
    
    height, width, depth = img.shape
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))
    
    
    background = np.zeros(shape=(height, width), dtype=np.uint8)
    circle_mask = cv2.circle(background, (x,y), int(r), 1, thickness=-1)
    
    img = cv2.bitwise_and(img, img, mask = background)
    
    return img


## === cell 12
pic_num = 43

img = train_images[pic_num]
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb);


## === cell 13
img1 = crop_image_from_gray(img_rgb)
plt.imshow(img1);


## === cell 14
img2 = circle_crop(img_rgb)
plt.imshow(img2);


## === cell 15
X_data = []
for image in tqdm(train_images):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    circle_img = circle_crop(img_rgb)
    image_resized = cv2.resize(circle_img, dsize=(224, 224), interpolation = cv2.INTER_CUBIC)
    X_data.append(image_resized)


## === cell 16
X_data = np.array(X_data).reshape(-1, 224, 224, 3)
print(X_data.shape)


## === cell 17
def values_in_mask(X):
    background = np.zeros(shape=(224, 224), dtype=np.uint8)
    circle_mask = cv2.circle(background, (112,112), 110, 1, thickness=-1)
    
    dim1 = X[:,:,0]
    dim2 = X[:,:,1]
    dim3 = X[:,:,2]
    
    circle_locations = (circle_mask == 1)
    R = dim1[circle_locations]
    G = dim2[circle_locations]
    B = dim3[circle_locations]
    
    return R, G, B


## === cell 18
def min_max_scaler_rgb(X):
    
    R, G, B = values_in_mask(X)   
    
    dim1 = X[:,:,0].astype('float32')
    dim2 = X[:,:,1].astype('float32')
    dim3 = X[:,:,2].astype('float32')
    
    min_R = np.min(R)
    min_G = np.min(G)
    min_B = np.min(B)
    
    max_R = np.max(R)
    max_G = np.max(G)
    max_B = np.max(B)
    
    img_R = (dim1 - min_R) / (max_R - min_R)
    img_G = (dim2 - min_G) / (max_G - min_G)
    img_B = (dim3 - min_B) / (max_B - min_B)
    
    img_R = np.where(img_R < 0, 0, img_R)
    img_G = np.where(img_G < 0, 0, img_G)
    img_B = np.where(img_B < 0, 0, img_B)
    
    img_R = np.where(img_R > 1, 1, img_R)
    img_G = np.where(img_G > 1, 1, img_G)
    img_B = np.where(img_B > 1, 1, img_B)
    
    img = np.stack([img_R, img_G, img_B], axis=-1)
    return img


## === cell 19
def min_max_scaler_gray(X):
    img = (X-np.min(X)) / (np.max(X)-np.min(X))
    return img


## === cell 20
fig = plt.figure(figsize=(14,8))

for idx, image in enumerate(train_images[:10]):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig.add_subplot(2, 5, idx+1)
    plt.imshow(image_rgb)
    plt.title("Label:{0}".format(labels['diagnosis'][idx]))
    plt.xlabel(labels['id_code'][idx])
    plt.tight_layout()


## === cell 21
fig = plt.figure(figsize=(14,8))

for idx, image in enumerate(X_data[:10]):
    fig.add_subplot(2, 5, idx+1)
    plt.imshow(image)
    plt.title("Label:{0}".format(labels['diagnosis'][idx]))
    plt.xlabel(labels['id_code'][idx])
    plt.tight_layout()


## === cell 22
del train_images


## === cell 23
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_data, y_data, test_size=0.2,
                                                      stratify = y_data, random_state = 123)

print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)


## === cell 24
def to_categorical_np(y, num_classes):
    y = np.asarray(y).astype(np.int64)
    y = y.reshape(-1)
    out = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


y_train_onehot = to_categorical_np(y_train, num_classes=5)
y_valid_onehot = to_categorical_np(y_valid, num_classes=5)

print(y_train_onehot.shape)
print(y_valid_onehot.shape)


## === cell 25
plt.hist(y_train)
plt.hist(y_valid)
plt.title("Train and Validation set Distribution")
plt.legend(['Train', 'Validation'])
plt.show()


## === cell 26
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=123456)
X_resampled, y_resampled = ros.fit_resample(X_train.reshape(-1, 224*224*3), y_train)


## --- ERROR in cell 26, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mModuleNotFoundError[0m                       Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1867660675.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0;32mfrom[0m [0mimblearn[0m[0;34m.[0m[0mover_sampling[0m [0;32mimport[0m [0mRandomOverSampler[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0;34m[0m[0m
[1;32m      3[0m [0mros[0m [0;34m=[0m [0mRandomOverSampler[0m[0;34m([0m[0mrandom_state[0m[0;34m=[0m[0;36m123456[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0mX_resampled[0m[0;34m,[0m [0my_resampled[0m [0;34m=[0m [0mros[0m[0;34m.[0m[0mfit_resample[0m[0;34m([0m[0mX_train[0m[0;34m.[0m[0mreshape[0m[0;34m([0m[0;34m-[0m[0;36m1[0m[0;34m,[0m [0;36m224[0m[0;34m*[0m[0;36m224[0m[0;34m*[0m[0;36m3[0m[0;34m)[0m[0;34m,[0m [0my_train[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/imblearn/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m     50[0m     [0;31m# process, as it may not be compiled yet[0m[0;34m[0m[0;34m[0m[0m
[1;32m     51[0m [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 52[0;31m     from . import (
[0m[1;32m     53[0m         [0mcombine[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     54[0m         [0mensemble[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/imblearn/combine/__init__.py[0m in [0;36m<module>[0;34m[0m
[1;32m      3[0m """
[1;32m      4[0m [0;34m[0m[0m
[0;32m----> 5[0;31m [0;32mfrom[0m [0;34m.[0m[0m_smote_enn[0m [0;32mimport[0m [0mSMOTEENN[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      6[0m [0;32mfrom[0m [0;34m.[0m[0m_smote_tomek[0m [0;32mimport[0m [0mSMOTETomek[0m[0;34m[0m[0;34m[0m[0m
[1;32m      7[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/imblearn/combine/_smote_enn.py[0m in [0;36m<module>[0;34m[0m
[1;32m     10[0m [0;32mfrom[0m [0msklearn[0m[0;34m.[0m[0mutils[0m [0;32mimport[0m [0mcheck_X_y[0m[0;34m[0m[0;34m[0m[0m
[1;32m     11[0m [0;34m[0m[0m
[0;32m---> 12[0;31m [0;32mfrom[0m [0;34m.[0m[0;34m.[0m[0mbase[0m [0;32mimport[0m [0mBaseSampler[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     13[0m [0;32mfrom[0m [0;34m.[0m[0;34m.[0m[0mover_sampling[0m [0;32mimport[0m [0mSMOTE[0m[0;34m[0m[0;34m[0m[0m
[1;32m     14[0m [0;32mfrom[0m [0;34m.[0m[0;34m.[0m[0mover_sampling[0m[0;34m.[0m[0mbase[0m [0;32mimport[0m [0mBaseOverSampler[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/imblearn/base.py[0m in [0;36m<module>[0;34m[0m
[1;32m     10[0m [0;32mfrom[0m [0msklearn[0m[0;34m.[0m[0mbase[0m [0;32mimport[0m [0mBaseEstimator[0m[0;34m,[0m [0mOneToOneFeatureMixin[0m[0;34m[0m[0;34m[0m[0m
[1;32m     11[0m [0;32mfrom[0m [0msklearn[0m[0;34m.[0m[0mpreprocessing[0m [0;32mimport[0m [0mlabel_binarize[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 12[0;31m [0;32mfrom[0m [0msklearn[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0m_metadata_requests[0m [0;32mimport[0m [0mMETHODS[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     13[0m [0;32mfrom[0m [0msklearn[0m[0;34m.[0m[0mutils[0m[0;34m.[0m[0mmulticlass[0m [0;32mimport[0m [0mcheck_classification_targets[0m[0;34m[0m[0;34m[0m[0m
[1;32m     14[0m [0;34m[0m[0m

[0;31mModuleNotFoundError[0m: No module named 'sklearn.utils._metadata_requests'

## === cell 27
X_resampled = X_resampled.reshape(-1, 224, 224, 3)
y_resampled_onehot = to_categorical(y_resampled, num_classes=5)

print(X_resampled.shape, y_resampled_onehot.shape)
