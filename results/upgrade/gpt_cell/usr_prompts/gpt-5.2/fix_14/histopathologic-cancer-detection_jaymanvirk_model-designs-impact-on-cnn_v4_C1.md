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

3.12

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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tf_keras==2.18.0

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

try:
    import google.protobuf as _pb
    from packaging.version import Version as _V

    _pb_ver = _V(getattr(_pb, "__version__", "0"))
    if _pb_ver.major >= 5:
        import sys, subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
        )
        import importlib

        importlib.invalidate_caches()
        import google.protobuf  # noqa: F401
except Exception:
    pass

import time
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import tensorflow_io as tfio
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from tensorflow.keras import layers, models
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam

from PIL import Image
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import load_model


## === cell 1
input_dir = '/kaggle/input/histopathologic-cancer-detection'
list_l = [os.path.join(input_dir, x) for x in os.listdir(input_dir)]
list_l


## === cell 2
sample_path = os.path.join(input_dir, "sample_submission.csv")
train_labels_path = os.path.join(input_dir, "train_labels.csv")
train_dir = os.path.join(input_dir, "train") + "/"
test_dir = os.path.join(input_dir, "test") + "/"

sample_data = pd.read_csv(sample_path)
train_data = pd.read_csv(train_labels_path)


## === cell 3
def print_short_summary(name, data):
    """
    Prints data head, shape and info.
    Args:
        name (str): name of dataset
        data (dataframe): dataset in a pd.DataFrame format
    """
    print(name)
    print('\n1. Data head:')
    print(data.head())
    print('\n2. Data shape: {}'.format(data.shape))
    print('\n3. Data info:')
    data.info()
    
def print_number_files(dirpath):
    print('{}: {} files'.format(dirpath, len(os.listdir(dirpath))))


## === cell 4
print_short_summary('Train data', train_data)


## === cell 5
print_short_summary('Sample data', sample_data)


## === cell 6
print_number_files(train_dir)


## === cell 7
print_number_files(test_dir)


## === cell 8
plt.figure(figsize=(16, 9))
tmp = train_data['label'].value_counts()
sns.barplot(y=['No Cancer', 'Cancer'], x=tmp.values, orient='h')
plt.xlabel('Number of records')
plt.ylabel('Label')
plt.title('Number of records per label')
plt.show()


## === cell 9
def get_images_to_plot(file_names):
    """
    Returns list of images
    Args:
        file_names: list of filenames
    Returns:
        list of image objects
    """
    return [Image.open(f) for f in file_names]

def get_image_label(dirname, data, labels, n = 5):
    dict_img = {}
    for l in labels:
        indexes = data['label'] == l
        tmp = data[indexes][:n]
        tmp = dirname + tmp['id'] + '.tif'
        tmp = tmp.values
        tmp = get_images_to_plot(tmp)
        dict_img[l] = tmp
        
    return dict_img


## === cell 10
img_path = train_dir + train_data['id'][0] + '.tif'
img = Image.open(img_path)
print('Original image size: {}'.format(img.size))


## === cell 11
data = get_image_label(train_dir,train_data, [0,1])


## === cell 12
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(16, 9))

labels = ['No Cancer', 'Cancer']
for i in range(10):
    row = i // 5
    col = i % 5
    axes[row, col].imshow(data[row][col])
    axes[row, col].set_title(labels[row])
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()


## === cell 13
SAMPLE_SIZE = 0.2
no_cancer = train_data[train_data['label'] == 0]
cancer = train_data[train_data['label'] == 1]
cancer = cancer[:int(SAMPLE_SIZE*len(cancer))]

no_cancer_downsampled = resample(no_cancer,
                              replace=False, 
                              n_samples=len(cancer),
                              random_state=0)

balanced_train_data = pd.concat([no_cancer_downsampled, cancer])

balanced_train_data = balanced_train_data.sample(frac=1, random_state=0).reset_index(drop=True)


## === cell 14
image_paths = train_dir + balanced_train_data['id'] + '.tif'
image_paths = image_paths.values

labels = balanced_train_data['label'].values

X_train, X_test, y_train, y_test = train_test_split(image_paths
                                                    , labels
                                                    , test_size = 0.25
                                                    , shuffle = True
                                                    , random_state = 0)


## === cell 15
def get_decoded_image(image_path, label=None):
    """
    Load and preprocess images using TensorFlow I/O.
    Decode image with 4 channels RGBA.
    Resize image to 32x32px.
    Scale pixels from 0 to 1.
    Args:
        image_path: path to TIFF image
        label (optional): true label from train data
    Returns:
        (img, label): for train data
        img: for test data
    """
    img = tf.io.read_file(image_path)
    img = tfio.experimental.image.decode_tiff(img)
    img = tf.image.resize(img, [32, 32])
    img = tf.cast(img, tf.float32) / 255.0
    
    return img if label is None else (img, label)

def get_prefetched_data(data, batch_size, buffer_size):
    """
    Create a TensorFlow dataset from image paths and labels.
    Execution in parallel.
    Load, preprocess images, shuffle and batch the data.
    Prefetch batches to improve training performance.
    Args:
        data (tuple): image paths and corresponding labels
        batch_size (int): number of samples per batch
        buffer_size (int): number of elements from the dataset to buffer while shuffling
    Returns:
        tf.data.Dataset: preprocessed and preloaded TensorFlow dataset for keras CNN
    """
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    
    dataset = tf.data.Dataset.from_tensor_slices(data)

    dataset = dataset.map(get_decoded_image, num_parallel_calls=AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    
    return dataset


## === cell 16
BATCH_SIZE = 64
TRAIN_BUFFER_SIZE = X_train.shape[0]
TEST_BUFFER_SIZE = X_test.shape[0]


def get_decoded_image(image_path, label=None):
    img_bytes = tf.io.read_file(image_path)

    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])

    img = tf.image.resize(img, [32, 32])
    img = tf.cast(img, tf.float32) / 255.0

    return img if label is None else (img, label)


train_dataset = get_prefetched_data((X_train, y_train), BATCH_SIZE, TRAIN_BUFFER_SIZE)
test_dataset = get_prefetched_data((X_test, y_test), BATCH_SIZE, TEST_BUFFER_SIZE)


## === cell 17
def roc_auc_score_(y_true, y_pred):
    """
    Calculate ROC AUC score using sklearn built-in function.
    Used in a model.compile as a custom metric.
    Args:
        y_true: true labels
        y_pred: predicted labels
    Returns:
        ROC AUC score (float)
    """
    return tf.py_function(roc_auc_score, (y_true, y_pred), tf.float64)


## === cell 18
model_base = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 4)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])


## === cell 19
model_drop_bn = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 4))
    , layers.BatchNormalization()
    , layers.MaxPooling2D((2, 2))
    , layers.Flatten()
    , layers.Dense(64, activation='relu')
    , layers.Dropout(0.25)
    , layers.Dense(1, activation='sigmoid')
])


## === cell 20
model_tuned = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 4))
    , layers.BatchNormalization()
    , layers.MaxPooling2D((4, 4))
    , layers.Flatten()
    , layers.Dense(64, activation='relu')
    , layers.Dropout(0.35)
    , layers.Dense(1, activation='sigmoid')
])


## === cell 21
def plot_model_scores(scores):
    """
    Plot train and test ROC AUC scores of a model by epoch
    """
    train_scores, test_scores = scores
    epochs = range(1, len(train_scores) + 1)

    plt.figure(figsize=(16, 9))
    plt.plot(epochs, train_scores, 'b', label='Train score')
    plt.plot(epochs, test_scores, 'r', label='Test score')
    plt.title('Train and test ROC AUC scores')
    plt.xlabel('Epoch')
    plt.ylabel('ROC AUC Score')
    plt.legend()
    plt.grid(True)
    plt.show()

    
def get_model_results(model_name, model):
    """
    Return tuple of runtime, train and test scores.
    Compile, fit and save model along the way.
    Args:
        model_name: model name
        model: fitted model
    Returns:
        (runtime, (train_scores, test_scores) )
    """
    st = time.time()
    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=[roc_auc_score_])
    model.fit(train_dataset, epochs=5, validation_data=test_dataset)
    runtime = time.time() - st
    model.save('{}.h5'.format(model_name))
    train_scores = model.history.history['roc_auc_score_']
    test_scores = model.history.history['val_roc_auc_score_']
    del model
    
    return (runtime, (train_scores, test_scores))


## === cell 22
tf.config.optimizer.set_jit(False)
tf.config.run_functions_eagerly(False)


def get_decoded_image_4ch(image_path, label=None):
    def _load_rgba(path):
        path = path.numpy().decode("utf-8")
        with Image.open(path) as im:
            im = im.convert("RGBA")
            arr = np.array(im, dtype=np.uint8)
        return arr

    img = tf.py_function(_load_rgba, [image_path], Tout=tf.uint8)
    img.set_shape([None, None, 4])

    img = tf.image.resize(img, [32, 32])
    img = tf.cast(img, tf.float32) / 255.0

    return img if label is None else (img, label)


def roc_auc_score_(y_true, y_pred):
    out = tf.py_function(roc_auc_score, (y_true, y_pred), tf.float64)
    out.set_shape([])  # scalar
    return out


def get_model_results(model_name, model):
    st = time.time()
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[roc_auc_score_],
        jit_compile=False,
    )
    model.fit(train_dataset, epochs=5, validation_data=test_dataset)
    runtime = time.time() - st
    model.save("{}.h5".format(model_name))
    train_scores = model.history.history["roc_auc_score_"]
    test_scores = model.history.history["val_roc_auc_score_"]
    del model

    return (runtime, (train_scores, test_scores))


AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.map(get_decoded_image_4ch, num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.shuffle(buffer_size=TRAIN_BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.map(get_decoded_image_4ch, num_parallel_calls=AUTOTUNE)
test_dataset = test_dataset.shuffle(buffer_size=TEST_BUFFER_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

model_base.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[roc_auc_score_],
    jit_compile=False,
)

runtime_base, scores_base = get_model_results("base", model_base)


## === cell 23
plot_model_scores(scores_base)


## === cell 24
runtime_drop_bn, scores_drop_bn = get_model_results('drop_bn', model_drop_bn)


## === cell 25
plot_model_scores(scores_drop_bn)


## === cell 26
runtime_tuned, scores_tuned = get_model_results('tuned', model_tuned)


## === cell 27
plot_model_scores(scores_tuned)


## === cell 28
table = [
    {
        'model':'Base'
        , 'sample_size': SAMPLE_SIZE
        , 'runtime': runtime_base
        , 'train_roc_auc_score': scores_base[0][-1]
        , 'test_roc_auc_score': scores_base[1][-1]
    }
    ,{
        'model':'Drop and BN'
        , 'sample_size': SAMPLE_SIZE
        , 'runtime': runtime_drop_bn
        , 'train_roc_auc_score': scores_drop_bn[0][-1]
        , 'test_roc_auc_score': scores_drop_bn[1][-1]
    }
    ,{
        'model':'Tuned'
        , 'sample_size': SAMPLE_SIZE
        , 'runtime': runtime_tuned
        , 'train_roc_auc_score': scores_tuned[0][-1]
        , 'test_roc_auc_score': scores_tuned[1][-1]
    }
]

pd.DataFrame(table).sort_values(by = ['test_roc_auc_score','runtime']
                                , ascending = [False, True])


## === cell 29
model_tuned_20 = load_model('tuned.h5'
                           , custom_objects = {'roc_auc_score_': roc_auc_score_})


## === cell 30
submis_files = os.listdir(test_dir)
submis_data = np.core.defchararray.add(test_dir, np.array(submis_files))

BATCH_SIZE = 64
SUBMIS_BUFFER_SIZE = submis_data.shape[0]

submis_dataset = get_prefetched_data((submis_data)
                                    , BATCH_SIZE
                                    , SUBMIS_BUFFER_SIZE)


## === cell 31
AUTOTUNE = tf.data.experimental.AUTOTUNE

submis_dataset = tf.data.Dataset.from_tensor_slices(submis_data)
submis_dataset = submis_dataset.map(get_decoded_image_4ch, num_parallel_calls=AUTOTUNE)
submis_dataset = submis_dataset.batch(BATCH_SIZE)
submis_dataset = submis_dataset.prefetch(buffer_size=AUTOTUNE)

result_20 = model_tuned_20.predict(submis_dataset)


## --- ERROR in cell 31, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mUnknownError[0m                              Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1509730438.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     11[0m [0msubmis_dataset[0m [0;34m=[0m [0msubmis_dataset[0m[0;34m.[0m[0mprefetch[0m[0;34m([0m[0mbuffer_size[0m[0;34m=[0m[0mAUTOTUNE[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     12[0m [0;34m[0m[0m
[0;32m---> 13[0;31m [0mresult_20[0m [0;34m=[0m [0mmodel_tuned_20[0m[0;34m.[0m[0mpredict[0m[0;34m([0m[0msubmis_dataset[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m
[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py[0m in [0;36merror_handler[0;34m(*args, **kwargs)[0m
[1;32m    120[0m             [0;31m# To get the full stack trace, call:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    121[0m             [0;31m# `keras.config.disable_traceback_filtering()`[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 122[0;31m             [0;32mraise[0m [0me[0m[0;34m.[0m[0mwith_traceback[0m[0;34m([0m[0mfiltered_tb[0m[0;34m)[0m [0;32mfrom[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    123[0m         [0;32mfinally[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    124[0m             [0;32mdel[0m [0mfiltered_tb[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py[0m in [0;36mraise_from_not_ok_status[0;34m(e, name)[0m
[1;32m   6000[0m [0;32mdef[0m [0mraise_from_not_ok_status[0m[0;34m([0m[0me[0m[0;34m,[0m [0mname[0m[0;34m)[0m [0;34m->[0m [0mNoReturn[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   6001[0m   [0me[0m[0;34m.[0m[0mmessage[0m [0;34m+=[0m [0;34m([0m[0;34m" name: "[0m [0;34m+[0m [0mstr[0m[0;34m([0m[0mname[0m [0;32mif[0m [0mname[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m [0;32melse[0m [0;34m""[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 6002[0;31m   [0;32mraise[0m [0mcore[0m[0;34m.[0m[0m_status_to_exception[0m[0;34m([0m[0me[0m[0;34m)[0m [0;32mfrom[0m [0;32mNone[0m  [0;31m# pylint: disable=protected-access[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   6003[0m [0;34m[0m[0m
[1;32m   6004[0m [0;34m[0m[0m

[0;31mUnknownError[0m: {{function_node __wrapped__IteratorGetNext_output_types_1_device_/job:localhost/replica:0/task:0/device:CPU:0}} IsADirectoryError: [Errno 21] Is a directory: '/kaggle/input/histopathologic-cancer-detection/test/test'
Traceback (most recent call last):

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 267, in __call__
    return func(device, token, args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 145, in __call__
    outputs = self._call(device, args)
              ^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 152, in _call
    ret = self._func(*args)
          ^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py", line 643, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/__autograph_generated_filef7utnyke.py", line 17, in _load_rgba
    with ag__.ld(Image).open(ag__.ld(path)) as im:
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/PIL/Image.py", line 3513, in open
    fp = builtins.open(filename, "rb")
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

IsADirectoryError: [Errno 21] Is a directory: '/kaggle/input/histopathologic-cancer-detection/test/test'


	 [[{{node EagerPyFunc}}]] [Op:IteratorGetNext] name: 

## === cell 32
id_ = np.char.replace(submis_files, '.tif', '')
label_ = np.ravel(np.round(result_20))
table = {
         'id': id_
         , 'label': label_
        }
table = pd.DataFrame(table)
