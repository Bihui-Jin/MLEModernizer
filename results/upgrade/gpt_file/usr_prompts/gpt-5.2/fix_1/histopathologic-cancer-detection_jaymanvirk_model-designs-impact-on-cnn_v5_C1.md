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

3.12

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

# 5. Target score

0.5035

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
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


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
input_dir = '/kaggle/input/histopathologic-cancer-detection'
list_l = [os.path.join(input_dir, x) for x in os.listdir(input_dir)]
list_l


## === cell 2
sample_data = pd.read_csv(list_l[0])
train_data = pd.read_csv(list_l[1])
train_dir = list_l[3] + '/'
test_dir = list_l[2] + '/'


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ParserError                               Traceback (most recent call last)
/tmp/ipykernel_11/2514562187.py in <cell line: 0>()
----> 1 sample_data = pd.read_csv(list_l[0])
      2 train_data = pd.read_csv(list_l[1])
      3 train_dir = list_l[3] + '/'
      4 test_dir = list_l[2] + '/'

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    624 
    625     with parser:
--> 626         return parser.read(nrows)
    627 
    628 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read(self, nrows)
   1921                     columns,
   1922                     col_dict,
-> 1923                 ) = self._engine.read(  # type: ignore[attr-defined]
   1924                     nrows
   1925                 )

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/c_parser_wrapper.py in read(self, nrows)
    232         try:
    233             if self.low_memory:
--> 234                 chunks = self._reader.read_low_memory(nrows)
    235                 # destructive to chunks
    236                 data = _concatenate_chunks(chunks)

parsers.pyx in pandas._libs.parsers.TextReader.read_low_memory()

parsers.pyx in pandas._libs.parsers.TextReader._read_rows()

parsers.pyx in pandas._libs.parsers.TextReader._tokenize_rows()

parsers.pyx in pandas._libs.parsers.TextReader._check_tokenize_status()

parsers.pyx in pandas._libs.parsers.raise_parser_error()

ParserError: Error tokenizing data. C error: Expected 1 fields in line 7, saw 4


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


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1144702278.py in <cell line: 0>()
----> 1 print_short_summary('Train data', train_data)

NameError: name 'train_data' is not defined

## === cell 5
print_short_summary('Sample data', sample_data)


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2335617714.py in <cell line: 0>()
----> 1 print_short_summary('Sample data', sample_data)

NameError: name 'sample_data' is not defined

## === cell 6
print_number_files(train_dir)


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/738094132.py in <cell line: 0>()
----> 1 print_number_files(train_dir)

NameError: name 'train_dir' is not defined

## === cell 7
print_number_files(test_dir)


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2424364139.py in <cell line: 0>()
----> 1 print_number_files(test_dir)

NameError: name 'test_dir' is not defined

## === cell 8
plt.figure(figsize=(16, 9))
tmp = train_data['label'].value_counts()
sns.barplot(y=['No Cancer', 'Cancer'], x=tmp.values, orient='h')
plt.xlabel('Number of records')
plt.ylabel('Label')
plt.title('Number of records per label')
plt.show()


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2738037214.py in <cell line: 0>()
      1 # Plot horizontal barplot of number of records per label
      2 plt.figure(figsize=(16, 9))
----> 3 tmp = train_data['label'].value_counts()
      4 sns.barplot(y=['No Cancer', 'Cancer'], x=tmp.values, orient='h')
      5 plt.xlabel('Number of records')

NameError: name 'train_data' is not defined

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


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1148453031.py in <cell line: 0>()
      1 # Print original image size
----> 2 img_path = train_dir + train_data['id'][0] + '.tif'
      3 img = Image.open(img_path)
      4 print('Original image size: {}'.format(img.size))

NameError: name 'train_dir' is not defined

## === cell 11
data = get_image_label(train_dir,train_data, [0,1])


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2271866679.py in <cell line: 0>()
      1 # Get 5 filenames per label
----> 2 data = get_image_label(train_dir,train_data, [0,1])

NameError: name 'train_dir' is not defined

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


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/473166997.py in <cell line: 0>()
      7     row = i // 5
      8     col = i % 5
----> 9     axes[row, col].imshow(data[row][col])
     10     axes[row, col].set_title(labels[row])
     11     axes[row, col].axis('off')

NameError: name 'data' is not defined

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


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1719085415.py in <cell line: 0>()
      2 SAMPLE_SIZE = 0.2
      3 # Majority class
----> 4 no_cancer = train_data[train_data['label'] == 0]
      5 # Minority class
      6 cancer = train_data[train_data['label'] == 1]

NameError: name 'train_data' is not defined

## === cell 14
image_paths = train_dir + balanced_train_data['id'] + '.tif'
image_paths = image_paths.values

labels = balanced_train_data['label'].values

X_train, X_test, y_train, y_test = train_test_split(image_paths
                                                    , labels
                                                    , test_size = 0.25
                                                    , shuffle = True
                                                    , random_state = 0)


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2340008347.py in <cell line: 0>()
      1 # Get full path to image including extension
----> 2 image_paths = train_dir + balanced_train_data['id'] + '.tif'
      3 image_paths = image_paths.values
      4 
      5 labels = balanced_train_data['label'].values

NameError: name 'train_dir' is not defined

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

train_dataset = get_prefetched_data((X_train, y_train)
                                    , BATCH_SIZE
                                    , TRAIN_BUFFER_SIZE)
test_dataset = get_prefetched_data((X_test, y_test)
                                   , BATCH_SIZE
                                   , TEST_BUFFER_SIZE)


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1155466183.py in <cell line: 0>()
      1 # Get train and test datasets for optimal performance
      2 BATCH_SIZE = 64
----> 3 TRAIN_BUFFER_SIZE = X_train.shape[0]
      4 TEST_BUFFER_SIZE = X_test.shape[0]
      5 

NameError: name 'X_train' is not defined

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
runtime_base, scores_base = get_model_results('base', model_base)


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/85090611.py in <cell line: 0>()
      1 # Get train and test scores of every epoch
----> 2 runtime_base, scores_base = get_model_results('base', model_base)

/tmp/ipykernel_11/2842347828.py in get_model_results(model_name, model)
     32               loss='binary_crossentropy',
     33               metrics=[roc_auc_score_])
---> 34     model.fit(train_dataset, epochs=5, validation_data=test_dataset)
     35     runtime = time.time() - st
     36     model.save('{}.h5'.format(model_name))

NameError: name 'train_dataset' is not defined

## === cell 23
plot_model_scores(scores_base)


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2685839975.py in <cell line: 0>()
      1 # Plot scores
----> 2 plot_model_scores(scores_base)

NameError: name 'scores_base' is not defined

## === cell 24
runtime_drop_bn, scores_drop_bn = get_model_results('drop_bn', model_drop_bn)


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2568431152.py in <cell line: 0>()
      1 # Get train and test scores of every epoch
----> 2 runtime_drop_bn, scores_drop_bn = get_model_results('drop_bn', model_drop_bn)

/tmp/ipykernel_11/2842347828.py in get_model_results(model_name, model)
     32               loss='binary_crossentropy',
     33               metrics=[roc_auc_score_])
---> 34     model.fit(train_dataset, epochs=5, validation_data=test_dataset)
     35     runtime = time.time() - st
     36     model.save('{}.h5'.format(model_name))

NameError: name 'train_dataset' is not defined

## === cell 25
plot_model_scores(scores_drop_bn)


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2137608740.py in <cell line: 0>()
      1 # Plot scores
----> 2 plot_model_scores(scores_drop_bn)

NameError: name 'scores_drop_bn' is not defined

## === cell 26
runtime_tuned, scores_tuned = get_model_results('tuned', model_tuned)


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1918389726.py in <cell line: 0>()
      1 # Get train and test scores of every epoch
----> 2 runtime_tuned, scores_tuned = get_model_results('tuned', model_tuned)

/tmp/ipykernel_11/2842347828.py in get_model_results(model_name, model)
     32               loss='binary_crossentropy',
     33               metrics=[roc_auc_score_])
---> 34     model.fit(train_dataset, epochs=5, validation_data=test_dataset)
     35     runtime = time.time() - st
     36     model.save('{}.h5'.format(model_name))

NameError: name 'train_dataset' is not defined

## === cell 27
plot_model_scores(scores_tuned)


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1478226984.py in <cell line: 0>()
      1 # Plot scores
----> 2 plot_model_scores(scores_tuned)

NameError: name 'scores_tuned' is not defined

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


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2131177517.py in <cell line: 0>()
      3         'model':'Base'
      4         , 'sample_size': SAMPLE_SIZE
----> 5         , 'runtime': runtime_base
      6         , 'train_roc_auc_score': scores_base[0][-1]
      7         , 'test_roc_auc_score': scores_base[1][-1]

NameError: name 'runtime_base' is not defined

## === cell 29
model_tuned_20 = load_model('tuned.h5'
                           , custom_objects = {'roc_auc_score_': roc_auc_score_})


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1314845611.py in <cell line: 0>()
      1 # Load save tuned model with custom metric parameter
----> 2 model_tuned_20 = load_model('tuned.h5'
      3                            , custom_objects = {'roc_auc_score_': roc_auc_score_})

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    194         )
    195     if str(filepath).endswith((".h5", ".hdf5")):
--> 196         return legacy_h5_format.load_model_from_hdf5(
    197             filepath, custom_objects=custom_objects, compile=compile
    198         )

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/saving/legacy_h5_format.py in load_model_from_hdf5(filepath, custom_objects, compile)
    114     opened_new_file = not isinstance(filepath, h5py.File)
    115     if opened_new_file:
--> 116         f = h5py.File(filepath, mode="r")
    117     else:
    118         f = filepath

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in __init__(self, name, mode, driver, libver, userblock_size, swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order, fs_strategy, fs_persist, fs_threshold, fs_page_size, page_buf_size, min_meta_keep, min_raw_keep, locking, alignment_threshold, alignment_interval, meta_block_size, **kwds)
    562                                  fs_persist=fs_persist, fs_threshold=fs_threshold,
    563                                  fs_page_size=fs_page_size)
--> 564                 fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)
    565 
    566             if isinstance(libver, tuple):

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in make_fid(name, mode, userblock_size, fapl, fcpl, swmr)
    236         if swmr and swmr_support:
    237             flags |= h5f.ACC_SWMR_READ
--> 238         fid = h5f.open(name, flags, fapl=fapl)
    239     elif mode == 'r+':
    240         fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/h5f.pyx in h5py.h5f.open()

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = 'tuned.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 30
submis_data = test_dir + sample_data['id'] + '.tif'
submis_data = submis_data.values

BATCH_SIZE = 64
SUBMIS_BUFFER_SIZE = submis_data.shape[0]

submis_dataset = get_prefetched_data((submis_data)
                                    , BATCH_SIZE
                                    , SUBMIS_BUFFER_SIZE)


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/22209112.py in <cell line: 0>()
      1 # Create prefethed dataset of images to classify
----> 2 submis_data = test_dir + sample_data['id'] + '.tif'
      3 submis_data = submis_data.values
      4 
      5 BATCH_SIZE = 64

NameError: name 'test_dir' is not defined

## === cell 31
result_20 = model_tuned_20.predict(submis_dataset)


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1556081006.py in <cell line: 0>()
      1 # Set predictions to result_20
----> 2 result_20 = model_tuned_20.predict(submis_dataset)

NameError: name 'model_tuned_20' is not defined

## === cell 32
sample_data['label'] = np.ravel(np.round(result_20))


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3554755914.py in <cell line: 0>()
      1 # Create table of ids and labels like sample_submission
----> 2 sample_data['label'] = np.ravel(np.round(result_20))

NameError: name 'result_20' is not defined

## === cell 33
sample_data


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1812181612.py in <cell line: 0>()
      1 # Print submission table
----> 2 sample_data

NameError: name 'sample_data' is not defined

## === cell 34
sample_data.to_csv('submission_20.csv', index=False)


## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4199607534.py in <cell line: 0>()
      1 # Make submission
----> 2 sample_data.to_csv('submission_20.csv', index=False)

NameError: name 'sample_data' is not defined
