# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given a dataset of images from digital pathology scans, predict if the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. 

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability that center 32x32px region of a patch contains at least one pixel of tumor tissue. The file should contain a header and have the following format:

```
id,label
0b2ea2a822ad23fdb1b5dd26653da899fbd2c0d5,0
95596b92e5066c5c52466c90b69ff089b39f2737,0
248e6738860e2ebcf6258cdc1f32f299e0c76914,0
etc.
```

## Dataset
Files are named with an image `id`. The `train_labels.csv` file provides the ground truth for the images in the `train` folder. You are predicting the labels for the images in the `test` folder.

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-image==0.25.2
sklearn-pandas==2.2.0
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
        input/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
        working/
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
```

-> data/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

import tf_keras as keras
from tf_keras.models import Sequential
from tf_keras.layers import Dense, Dropout, Flatten
from tf_keras.layers import Conv2D, MaxPooling2D

from skimage.io import imread
import gc

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(0)
try:
    keras.utils.set_random_seed(0)
except Exception:
    pass



## === cell 1
train = pd.read_csv("../input/train_labels.csv")



## === cell 2
train.head()



## === cell 3
print("Number of training smaples -->", len(train))




## === cell 4
def train_func_image_file(x):
    folder = "../input/train/"
    path = folder + x + ".tif"
    return path




## === cell 5
train["path"] = train["id"].apply(train_func_image_file)



## === cell 6
print(train["path"][0])




## === cell 7
def _load_center_crop_paths_to_array(paths, out_h=48, out_w=48):
    n = len(paths)
    x = np.empty((n, out_h, out_w, 3), dtype=np.uint8)
    for i, p in enumerate(paths):
        img = imread(p)  # original image is 96x96x3
        x[i] = img[24:72, 24:72]
    return x


train_paths = train["path"].iloc[0:215000].tolist()
x_train = _load_center_crop_paths_to_array(train_paths, 48, 48)



## === cell 8
print("Loaded training crops:", x_train.shape, x_train.dtype)




## === cell 9
def crop(x):
    return x[24:72, 24:72]




## === cell 10
pass



## === cell 11
pass



## === cell 12
print("Dimension of crop image --->", x_train[0].shape)



## === cell 13
print("Dimension of crop image --->", x_train[0].shape)



## === cell 14
train = train.drop(["path"], axis=1)



## === cell 15
pass



## === cell 16
gc.collect()



## === cell 17
pass



## === cell 18
pass



## === cell 19
gc.collect()



## === cell 20
x_train = x_train.astype("float32")



## === cell 21
x_train /= 255.0



## === cell 22
num_classes = 2



## === cell 23
y_train = train["label"][0:215000]



## === cell 24
y_train = keras.utils.to_categorical(y_train, num_classes)



## === cell 25
del train



## === cell 26
gc.collect()



## === cell 27
img_rows, img_cols = 48, 48
input_shape = (img_rows, img_cols, 3)

batch_size = 128
epochs = 3



## === cell 28
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation="softmax"))



## === cell 29
model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adadelta(),
    metrics=["accuracy"],
)



## === cell 30
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)



## === cell 31
del x_train
del y_train



## === cell 32
gc.collect()



## === cell 33
image_file = os.listdir("../input/test/")



## === cell 34
test = pd.DataFrame(image_file, columns=["file"])



## === cell 35
test.head()




## === cell 36
def test_func_image_file(x):
    folder = "../input/test/"
    path = folder + x
    return path




## === cell 37
test["path"] = test["file"].apply(test_func_image_file)



## === cell 38
pass



## === cell 39
pass



## === cell 40
pass




## === cell 41
def _predict_test_in_batches(model, paths, batch_size=512):
    n = len(paths)
    preds = np.empty((n,), dtype=np.int64)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_paths = paths[start:end]
        xb = (
            _load_center_crop_paths_to_array(batch_paths, 48, 48).astype("float32")
            / 255.0
        )
        prob = model.predict(xb, batch_size=128, verbose=0)
        preds[start:end] = np.argmax(prob, axis=1)
    return preds




## === cell 42
pass



## === cell 43
gc.collect()



## === cell 44
pass



## === cell 45
pass



## === cell 46
test["id"] = test["file"].apply(lambda x: os.path.splitext(x)[0])



## === cell 47
predictions = _predict_test_in_batches(model, test["path"].tolist(), batch_size=512)



## === cell 48
test["label"] = pd.Series(predictions)



## === cell 49
print("Cancer Detected - True Positive --> ", len(test["label"][test["label"] == 1]))



## === cell 50
print("NO Cancer Detected - True Negative --> ", len(test["label"][test["label"] == 0]))



## === cell 51
test = test.drop(["file", "path"], axis=1)



## === cell 52
test.head()



## === cell 53
test.to_csv("submission.csv", columns=test.columns, index=False)
