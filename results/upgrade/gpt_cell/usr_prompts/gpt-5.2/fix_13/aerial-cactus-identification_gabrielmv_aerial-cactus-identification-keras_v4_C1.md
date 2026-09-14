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
sklearn-pandas==2.2.0
tf_keras==2.18.0
tqdm==4.67.1

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
import os
import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<4"])

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, BatchNormalization, LeakyReLU, Dropout

from keras.layers import GlobalAveragePooling2D
from keras.optimizers import Adam

from tf_keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.applications import DenseNet169

from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split

from tqdm import tqdm, tqdm_notebook

import numpy as np
import pandas as pd
import cv2 as cv
import seaborn as sns
import matplotlib.pyplot as plt


## === cell 1
os.listdir('../input')


## === cell 2
train_data = pd.read_csv('../input/train.csv')


## === cell 3
len(train_data)


## === cell 4
train_data.head()


## === cell 5
def create_model():
    model = Sequential()
    
    model.add(Conv2D(filters = 16, kernel_size = 3, activation = 'relu', input_shape = (32, 32, 3)))
    model.add(Conv2D(filters = 16, kernel_size = 3, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    model.add(Conv2D(filters = 32, kernel_size = 3, activation = 'relu'))
    model.add(Conv2D(filters = 64, kernel_size = 3, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    model.add(Conv2D(filters = 64, kernel_size = 3, activation = 'relu'))
    model.add(Conv2D(filters = 128, kernel_size = 3, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    model.add(Conv2D(filters = 128, kernel_size = 3, activation = 'relu'))
    model.add(Conv2D(filters = 256, kernel_size = 3, activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    model.add(GlobalAveragePooling2D())

    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(Adam(lr = 0.01), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return model


## === cell 6
def plot_training_curves(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(1, len(acc) + 1)
    
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'g', label='Validation loss')
    plt.title('Losses')
    plt.legend()
    plt.figure()
    
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'g', label='Validation acc')
    plt.title('Accuracies')
    plt.legend()
    plt.figure()
    
    plt.show()
    


## === cell 7
file_path = 'weights-aerial-cactus.h5'

callbacks = [
        ModelCheckpoint(file_path, monitor = 'val_acc', verbose = 1, save_best_only = True, mode = 'max'),
        ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 3, verbose = 1, mode = 'min', min_lr = 0.00001),
        EarlyStopping(monitor = 'val_loss', min_delta = 1e-10, patience = 5, verbose = 1, restore_best_weights = True)
        ]

training_path = '../input/train/train/'
test_path = '../input/test/test/'


## === cell 8
images_train = []
labels_train = []

images = train_data['id'].values

for image_id in tqdm_notebook(images):
    image = np.array(cv.imread(training_path + image_id))
    images_train.append(image)
    labels_train.append(train_data[train_data['id'] == image_id]['has_cactus'].values[0])
    
images_train = np.asarray(images_train)
images_train = images_train.astype('float32')
images_train /= 255.

labels_train = np.asarray(labels_train)


## === cell 9
x_tr = images_train
y_tr = labels_train

images_train = x_tr
labels_train - y_tr


## === cell 10
test_images_names = []

for filename in os.listdir(test_path):
    if filename.lower().endswith(".jpg"):
        test_images_names.append(filename)

test_images_names.sort()

images_test = []

for image_id in tqdm_notebook(test_images_names):
    img = cv.imread(test_path + image_id)
    if img is None:
        continue
    images_test.append(np.array(img))

images_test = np.asarray(images_test)
images_test = images_test.astype("float32")
images_test /= 255


## === cell 11
x_train, x_test, y_train, y_test = train_test_split(images_train, labels_train, test_size = 0.15, stratify = labels_train)


## === cell 12
import keras

_Adam_orig = Adam


def _Adam_compat(*args, **kwargs):
    if "lr" in kwargs and "learning_rate" not in kwargs:
        kwargs["learning_rate"] = kwargs.pop("lr")
    else:
        kwargs.pop("lr", None)
    return _Adam_orig(*args, **kwargs)


Adam = _Adam_compat
try:
    model = create_model()
finally:
    Adam = _Adam_orig


## === cell 13
history = model.fit(x_train, 
            y_train, 
            batch_size = 32, 
            epochs = 50, 
            validation_data = (x_test, y_test),
            verbose = 1,
            callbacks = callbacks)


## === cell 14
import os

if os.path.exists(file_path):
    model.load_weights(file_path)
else:
    print(f"Warning: checkpoint '{file_path}' not found; using current model weights.")


## === cell 15
predictions = model.predict(images_test, verbose = 1)


## === cell 16
predictions


## === cell 17
def plot_training_curves(history):
    hist = history.history

    if "acc" in hist:
        acc_key, val_acc_key = "acc", "val_acc"
    else:
        acc_key, val_acc_key = "accuracy", "val_accuracy"

    acc = hist.get(acc_key, [])
    val_acc = hist.get(val_acc_key, [])
    loss = hist.get("loss", [])
    val_loss = hist.get("val_loss", [])

    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, "r", label="Training loss")
    if len(val_loss) == len(loss) and len(val_loss) > 0:
        plt.plot(epochs, val_loss, "g", label="Validation loss")
    plt.title("Losses")
    plt.legend()
    plt.figure()

    if len(acc) == len(loss) and len(acc) > 0:
        plt.plot(epochs, acc, "r", label="Training acc")
    if len(val_acc) == len(loss) and len(val_acc) > 0:
        plt.plot(epochs, val_acc, "g", label="Validation acc")
    plt.title("Accuracies")
    plt.legend()
    plt.figure()

    plt.show()


## === cell 18
y_pred_probability = model.predict_proba(x_tr)

y_pred = model.predict_classes(x_tr)
conf_matrix = confusion_matrix(y_tr, y_pred)
fig, ax = plt.subplots(figsize = (10, 10))

sns.heatmap(conf_matrix, annot = True, fmt = 'd', xticklabels = ['0', '1'], yticklabels = ['0', '1'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

print(classification_report(y_tr, y_pred, target_names = ['0','1']))
print("\n\n AUC: {:<0.4f}".format(roc_auc_score(y_tr, y_pred_probability)))


## --- ERROR in cell 18, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3094669797.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0my_pred_probability[0m [0;34m=[0m [0mmodel[0m[0;34m.[0m[0mpredict_proba[0m[0;34m([0m[0mx_tr[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0;34m[0m[0m
[1;32m      3[0m [0my_pred[0m [0;34m=[0m [0mmodel[0m[0;34m.[0m[0mpredict_classes[0m[0;34m([0m[0mx_tr[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0mconf_matrix[0m [0;34m=[0m [0mconfusion_matrix[0m[0;34m([0m[0my_tr[0m[0;34m,[0m [0my_pred[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0mfig[0m[0;34m,[0m [0max[0m [0;34m=[0m [0mplt[0m[0;34m.[0m[0msubplots[0m[0;34m([0m[0mfigsize[0m [0;34m=[0m [0;34m([0m[0;36m10[0m[0;34m,[0m [0;36m10[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mAttributeError[0m: 'Sequential' object has no attribute 'predict_proba'

## === cell 19
test_df = pd.read_csv('../input/sample_submission.csv')
X_test = []
images_test = test_df['id'].values

for img_id in tqdm_notebook(images_test):
    X_test.append(cv.imread(test_path + img_id))
    
X_test = np.asarray(X_test)
X_test = X_test.astype('float32')
X_test /= 255

y_test_pred = model.predict_proba(X_test)

test_df['has_cactus'] = y_test_pred
test_df.to_csv('aerial-cactus-submission.csv', index = False)
