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

0.7367

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
from PIL import Image

import tensorflow as tf
import tensorflow_io as tfio
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from tensorflow.keras import layers, models

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
/tmp/ipykernel_11/2976446992.py in <cell line: 0>()
      1 # Set datasets and directory names
----> 2 sample_data = pd.read_csv(list_l[0])
      3 train_data = pd.read_csv(list_l[1])
      4 train_dir = list_l[3] + '/'
      5 test_dir = list_l[2] + '/'

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
del list_l


## === cell 4
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


## === cell 5
print_short_summary('Train data', train_data)


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1144702278.py in <cell line: 0>()
----> 1 print_short_summary('Train data', train_data)

NameError: name 'train_data' is not defined

## === cell 6
print_short_summary('Sample data', sample_data)


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2335617714.py in <cell line: 0>()
----> 1 print_short_summary('Sample data', sample_data)

NameError: name 'sample_data' is not defined

## === cell 7
print_number_files(train_dir)


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/738094132.py in <cell line: 0>()
----> 1 print_number_files(train_dir)

NameError: name 'train_dir' is not defined

## === cell 8
print_number_files(test_dir)


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2424364139.py in <cell line: 0>()
----> 1 print_number_files(test_dir)

NameError: name 'test_dir' is not defined

## === cell 9
del print_short_summary, print_number_files


## === cell 10
plt.figure(figsize=(16, 9))
tmp = train_data['label'].value_counts()
sns.barplot(y=['No Cancer', 'Cancer'], x=tmp.values, orient='h')
plt.xlabel('Number of records')
plt.ylabel('Label')
plt.title('Number of records per label')
plt.show()


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2738037214.py in <cell line: 0>()
      1 # Plot horizontal barplot of number of records per label
      2 plt.figure(figsize=(16, 9))
----> 3 tmp = train_data['label'].value_counts()
      4 sns.barplot(y=['No Cancer', 'Cancer'], x=tmp.values, orient='h')
      5 plt.xlabel('Number of records')

NameError: name 'train_data' is not defined

## === cell 11
del tmp


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2082842287.py in <cell line: 0>()
      1 # Cleaning
----> 2 del tmp

NameError: name 'tmp' is not defined

## === cell 12
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
    """
    Return dictionary with label-imagepath
    Args:
        dirname: name of the directory
        data: dataset of file names
        labels: list of labels
        n (opt): number of images per label
    Returns:
        dict_img: dictionary with label-imagepath pairs
    """
    dict_img = {}
    for l in labels:
        indexes = data['label'] == l
        tmp = data[indexes][:n]
        tmp = dirname + tmp['id'] + '.tif'
        tmp = tmp.values
        tmp = get_images_to_plot(tmp)
        dict_img[l] = tmp
        
    return dict_img


## === cell 13
img_path = train_dir + train_data['id'][0] + '.tif'
img = Image.open(img_path)
print('Original image size: {}'.format(img.size))


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1148453031.py in <cell line: 0>()
      1 # Print original image size
----> 2 img_path = train_dir + train_data['id'][0] + '.tif'
      3 img = Image.open(img_path)
      4 print('Original image size: {}'.format(img.size))

NameError: name 'train_dir' is not defined

## === cell 14
data = get_image_label(train_dir,train_data, [0,1])


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2271866679.py in <cell line: 0>()
      1 # Get 5 filenames per label
----> 2 data = get_image_label(train_dir,train_data, [0,1])

NameError: name 'train_dir' is not defined

## === cell 15
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


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2544390445.py in <cell line: 0>()
      7     row = i // 5
      8     col = i % 5
----> 9     axes[row, col].imshow(data[row][col])
     10     axes[row, col].set_title(labels[row])
     11     axes[row, col].axis('off')

NameError: name 'data' is not defined

## === cell 16
del get_images_to_plot, get_image_label, img_path, img
del data, fig, axes, labels, row, col


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/317569615.py in <cell line: 0>()
      1 # Cleaning
----> 2 del get_images_to_plot, get_image_label, img_path, img
      3 del data, fig, axes, labels, row, col

NameError: name 'img_path' is not defined

## === cell 17
no_cancer = train_data[train_data['label'] == 0]
cancer = train_data[train_data['label'] == 1]

no_cancer_downsampled = resample(no_cancer,
                              replace=False, 
                              n_samples=len(cancer),
                              random_state=0)

balanced_train_data = pd.concat([no_cancer_downsampled, cancer])

balanced_train_data = balanced_train_data.sample(frac=1, random_state=0).reset_index(drop=True)


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4017910861.py in <cell line: 0>()
      1 # Majority class
----> 2 no_cancer = train_data[train_data['label'] == 0]
      3 # Minority class
      4 cancer = train_data[train_data['label'] == 1]
      5 

NameError: name 'train_data' is not defined

## === cell 18
del no_cancer, cancer, no_cancer_downsampled


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1269975198.py in <cell line: 0>()
      1 # Cleaning
----> 2 del no_cancer, cancer, no_cancer_downsampled

NameError: name 'no_cancer' is not defined

## === cell 19
image_paths = train_dir + balanced_train_data['id'] + '.tif'
image_paths = image_paths.values

labels = balanced_train_data['label'].values

X_train, X_test, y_train, y_test = train_test_split(image_paths
                                                    , labels
                                                    , test_size = 0.25
                                                    , shuffle = True
                                                    , random_state = 0)


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2340008347.py in <cell line: 0>()
      1 # Get full path to image including extension
----> 2 image_paths = train_dir + balanced_train_data['id'] + '.tif'
      3 image_paths = image_paths.values
      4 
      5 labels = balanced_train_data['label'].values

NameError: name 'train_dir' is not defined

## === cell 20
del image_paths, labels


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2603046264.py in <cell line: 0>()
      1 # Cleaning
----> 2 del image_paths, labels

NameError: name 'image_paths' is not defined

## === cell 21
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

def get_prefetched_data(data, batch_size):
    """
    Create a TensorFlow dataset from image paths and labels.
    Execution in parallel.
    Load, preprocess images and batch the data.
    Prefetch batches to improve training performance.
    Args:
        data (tuple): image paths and corresponding labels
        batch_size (int): number of samples per batch
    Returns:
        tf.data.Dataset: preprocessed and preloaded TensorFlow dataset for keras CNN
    """
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    
    dataset = tf.data.Dataset.from_tensor_slices(data)
    
    dataset = dataset.map(get_decoded_image, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    
    return dataset


## === cell 22
BATCH_SIZE = 128

train_dataset = get_prefetched_data((X_train, y_train)
                                    , BATCH_SIZE)
test_dataset = get_prefetched_data((X_test, y_test)
                                   , BATCH_SIZE)


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1722225702.py in <cell line: 0>()
      2 BATCH_SIZE = 128
      3 
----> 4 train_dataset = get_prefetched_data((X_train, y_train)
      5                                     , BATCH_SIZE)
      6 test_dataset = get_prefetched_data((X_test, y_test)

NameError: name 'X_train' is not defined

## === cell 23
del X_train, y_train, X_test, y_test


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3054284884.py in <cell line: 0>()
----> 1 del X_train, y_train, X_test, y_test

NameError: name 'X_train' is not defined

## === cell 24
def get_model_base():
    """
    Return base model architecture
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 4))

        , layers.Flatten()

        , layers.Dense(32, activation='relu')

        , layers.Dense(1, activation='sigmoid')
    ])
    
    return model


## === cell 25
def get_model_base_deep():
    """
    Return deeper model architecture
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 4))
        , layers.Conv2D(32, (3, 3), activation='relu')

        , layers.Flatten()

        , layers.Dense(32, activation='relu')
        , layers.Dense(32, activation='relu')

        , layers.Dense(1, activation='sigmoid')
    ])
    
    return model


## === cell 26
def get_model_base_wide():
    """
    Return wider model architecture
    """
    model_drop_bn = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 4))

        , layers.Flatten()
        
        , layers.Dense(64, activation='relu')
 
        , layers.Dense(1, activation='sigmoid')
    ])
    
    return model_drop_bn


## === cell 27
def get_model_base_maxpool():
    """
    Return maxpool model architecture
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 4))
        , layers.MaxPooling2D((2, 2), strides = (2,2))

        , layers.Flatten()

        , layers.Dense(32, activation='relu')
        
        , layers.Dense(1, activation='sigmoid')
    ])
    
    return model


## === cell 28
def get_model_base_dropout():
    """
    Return dropout model architecture
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 4))

        , layers.Flatten()

        , layers.Dense(32, activation='relu')
        
        , layers.Dropout(0.25)
        
        , layers.Dense(1, activation='sigmoid')
    ])
    
    return model


## === cell 29
def get_compiled_model(func):
    """
    Create model to be trained with a multi-GPU strategy.
    Args:
        func: function to get model architecture
    Returns:
        compiled_model: tensorflow model that performs data parallelism
                            by copying all of the model's variables
                            to each processor
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        strategy = tf.distribute.MirroredStrategy()

        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
        print('No GPU available, falling back to CPU.')

    with strategy.scope():
        compiled_model = func()
        compiled_model.compile(optimizer = tf.keras.optimizers.Adam()
                              , loss = tf.keras.losses.BinaryCrossentropy()
                              , metrics = [tf.keras.metrics.AUC()])

    return compiled_model


## === cell 30
def plot_model_scores(scores, model_name):
    """
    Plot train and test ROC AUC scores of a model by epoch
    """
    train_scores, test_scores = scores
    epochs = range(1, len(train_scores) + 1)

    plt.figure(figsize=(16, 9))
    plt.plot(epochs, train_scores, label='Train score')
    plt.plot(epochs, test_scores, label='Test score')
    plt.title('Train and test ROC AUC scores of the {}'.format(model_name))
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
        model: fitted model
    Returns:
        (runtime, (train_scores, test_scores) )
    """
    model = get_compiled_model(model)
    
    st = time.time()
    model.fit(train_dataset, epochs=5, validation_data=test_dataset)
    runtime = time.time() - st
    
    model.save('{}.h5'.format(model_name))
    
    train_scores = model.history.history['auc']
    test_scores = model.history.history['val_auc']
    
    tf.keras.backend.clear_session()
    
    return (runtime, (train_scores, test_scores))


## === cell 31
runtime_base, scores_base = get_model_results('model_base',get_model_base)


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4110197565.py in <cell line: 0>()
      1 # Get train and test scores of every epoch
----> 2 runtime_base, scores_base = get_model_results('model_base',get_model_base)

/tmp/ipykernel_11/519111943.py in get_model_results(model_name, model)
     30 
     31     st = time.time()
---> 32     model.fit(train_dataset, epochs=5, validation_data=test_dataset)
     33     runtime = time.time() - st
     34 

NameError: name 'train_dataset' is not defined

## === cell 32
plot_model_scores(scores_base, 'base model')


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/573228175.py in <cell line: 0>()
      1 # Plot scores
----> 2 plot_model_scores(scores_base, 'base model')

NameError: name 'scores_base' is not defined

## === cell 33
runtime_base_deep, scores_base_deep = get_model_results('model_base_deep'
                                                          ,get_model_base_deep)


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3798284138.py in <cell line: 0>()
      1 # Get train and test scores of every epoch
----> 2 runtime_base_deep, scores_base_deep = get_model_results('model_base_deep'
      3                                                           ,get_model_base_deep)

/tmp/ipykernel_11/519111943.py in get_model_results(model_name, model)
     30 
     31     st = time.time()
---> 32     model.fit(train_dataset, epochs=5, validation_data=test_dataset)
     33     runtime = time.time() - st
     34 

NameError: name 'train_dataset' is not defined

## === cell 34
plot_model_scores(scores_base_deep, 'base + additional layers model')


## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4161757532.py in <cell line: 0>()
      1 # Plot scores
----> 2 plot_model_scores(scores_base_deep, 'base + additional layers model')

NameError: name 'scores_base_deep' is not defined

## === cell 35
runtime_base_wide, scores_base_wide = get_model_results('model_base_wide'
                                                          ,get_model_base_wide)


## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1867656141.py in <cell line: 0>()
      1 # Get train and test scores of every epoch
----> 2 runtime_base_wide, scores_base_wide = get_model_results('model_base_wide'
      3                                                           ,get_model_base_wide)

/tmp/ipykernel_11/519111943.py in get_model_results(model_name, model)
     30 
     31     st = time.time()
---> 32     model.fit(train_dataset, epochs=5, validation_data=test_dataset)
     33     runtime = time.time() - st
     34 

NameError: name 'train_dataset' is not defined

## === cell 36
plot_model_scores(scores_base_wide, 'base + wider layers model')


## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1933104148.py in <cell line: 0>()
      1 # Plot scores
----> 2 plot_model_scores(scores_base_wide, 'base + wider layers model')

NameError: name 'scores_base_wide' is not defined

## === cell 37
runtime_base_maxpool, scores_base_maxpool = get_model_results('model_base_maxpool'
                                                                ,get_model_base_maxpool)


## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1755838132.py in <cell line: 0>()
      1 # Get train and test scores of every epoch
----> 2 runtime_base_maxpool, scores_base_maxpool = get_model_results('model_base_maxpool'
      3                                                                 ,get_model_base_maxpool)

/tmp/ipykernel_11/519111943.py in get_model_results(model_name, model)
     30 
     31     st = time.time()
---> 32     model.fit(train_dataset, epochs=5, validation_data=test_dataset)
     33     runtime = time.time() - st
     34 

NameError: name 'train_dataset' is not defined

## === cell 38
plot_model_scores(scores_base_maxpool, 'base + max pooling model')


## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/436528656.py in <cell line: 0>()
      1 # Plot scores
----> 2 plot_model_scores(scores_base_maxpool, 'base + max pooling model')

NameError: name 'scores_base_maxpool' is not defined

## === cell 39
runtime_base_dropout, scores_base_dropout = get_model_results('model_base_dropout'
                                                                ,get_model_base_dropout)


## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4099413398.py in <cell line: 0>()
      1 # Get train and test scores of every epoch
----> 2 runtime_base_dropout, scores_base_dropout = get_model_results('model_base_dropout'
      3                                                                 ,get_model_base_dropout)

/tmp/ipykernel_11/519111943.py in get_model_results(model_name, model)
     30 
     31     st = time.time()
---> 32     model.fit(train_dataset, epochs=5, validation_data=test_dataset)
     33     runtime = time.time() - st
     34 

NameError: name 'train_dataset' is not defined

## === cell 40
plot_model_scores(scores_base_dropout, 'base + dropout model')


## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3736353129.py in <cell line: 0>()
      1 # Plot scores
----> 2 plot_model_scores(scores_base_dropout, 'base + dropout model')

NameError: name 'scores_base_dropout' is not defined

## === cell 41
results = [('Base', runtime_base, scores_base)
          ,('Base + Add. layers', runtime_base_deep, scores_base_deep)
          ,('Base + Wider layers', runtime_base_wide, scores_base_wide)
          ,('Base + Max pooling', runtime_base_maxpool, scores_base_maxpool)
          ,('Base + Dropout', runtime_base_dropout, scores_base_dropout)]
table = []
for i in range(len(results)):
    tmp = {
            'model': results[i][0]
            , 'runtime (sec)': results[i][1]
            , 'train_roc_auc_score': results[i][2][0][-1]
            , 'test_roc_auc_score': results[i][2][1][-1]
        }
    table.append(tmp)


pd.DataFrame(table).sort_values(by = ['test_roc_auc_score'
                                      ,'runtime (sec)']
                                , ascending = [False, True])


## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4076084121.py in <cell line: 0>()
      1 # Print table results comparison
----> 2 results = [('Base', runtime_base, scores_base)
      3           ,('Base + Add. layers', runtime_base_deep, scores_base_deep)
      4           ,('Base + Wider layers', runtime_base_wide, scores_base_wide)
      5           ,('Base + Max pooling', runtime_base_maxpool, scores_base_maxpool)

NameError: name 'runtime_base' is not defined

## === cell 42
del results, tmp, table, train_dataset, test_dataset, train_data, train_dir, test_dir
del get_model_results, plot_model_scores, get_compiled_model


## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2386080944.py in <cell line: 0>()
      1 # Cleaning
----> 2 del results, tmp, table, train_dataset, test_dataset, train_data, train_dir, test_dir
      3 del get_model_results, plot_model_scores, get_compiled_model

NameError: name 'results' is not defined

## === cell 43
model = load_model('model_base_deep.h5')


## --- ERROR in cell 43, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/806139185.py in <cell line: 0>()
      1 # Load top performed model
----> 2 model = load_model('model_base_deep.h5')

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

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = 'model_base_deep.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 44
submis_data = test_dir + sample_data['id'] + '.tif'
submis_data = submis_data.values

submis_dataset = get_prefetched_data((submis_data)
                                    , BATCH_SIZE)


## --- ERROR in cell 44, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3810886281.py in <cell line: 0>()
      1 # Create prefethed dataset of images to classify
----> 2 submis_data = test_dir + sample_data['id'] + '.tif'
      3 submis_data = submis_data.values
      4 
      5 submis_dataset = get_prefetched_data((submis_data)

NameError: name 'test_dir' is not defined

## === cell 45
results = model.predict(submis_dataset)


## --- ERROR in cell 45, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4238099676.py in <cell line: 0>()
      1 # Set results
----> 2 results = model.predict(submis_dataset)

NameError: name 'model' is not defined

## === cell 46
sample_data['label'] = np.ravel(np.round(results))


## --- ERROR in cell 46, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4020228035.py in <cell line: 0>()
      1 # Create table of ids and labels like sample_submission
----> 2 sample_data['label'] = np.ravel(np.round(results))

NameError: name 'results' is not defined

## === cell 47
sample_data


## --- ERROR in cell 47, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1812181612.py in <cell line: 0>()
      1 # Print submission table
----> 2 sample_data

NameError: name 'sample_data' is not defined

## === cell 48
sample_data.to_csv('submission.csv', index=False)


## --- ERROR in cell 48, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3271207801.py in <cell line: 0>()
      1 # Make submission
----> 2 sample_data.to_csv('submission.csv', index=False)

NameError: name 'sample_data' is not defined

## === cell 49
del submis_data, submis_dataset, sample_data
del get_decoded_image, get_prefetched_data


## --- ERROR in cell 49, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2126114509.py in <cell line: 0>()
      1 # Cleaning
----> 2 del submis_data, submis_dataset, sample_data
      3 del get_decoded_image, get_prefetched_data

NameError: name 'submis_data' is not defined
