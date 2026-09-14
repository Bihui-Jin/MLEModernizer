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
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

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
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.7155493998153282

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 1
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pandas as pd

import tensorflow as tf

import tensorflow_addons as tfa
print("Using tensorflow ", tf.__version__)

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
class CFG:
    classes = [
        'complex', 
        'frog_eye_leaf_spot', 
        'powdery_mildew', 
        'rust', 
        'scab',
        'healthy']
    batch_size = 16
    test_size = 0.25
    img_size = 512 # image size
    seed = 42 # random seed 
    retrain = False



## === cell 4
!mkdir /kaggle/working/pngs/

## === cell 5
train_dir = '../input/plant-pathology-2021-fgvc8/train_images/'
train_csv = '../input/plant-pathology-2021-fgvc8/train.csv'
test_dir = '../input/plant-pathology-2021-fgvc8/test_images/'
duplicates = '../input/duplicatescsv/duplicates.csv'
png_dir = "/kaggle/working/pngs/" 
model_dir = "../input/models/"

print(os.path.exists(train_dir))
print(os.path.exists(train_csv))
print(os.path.exists(test_dir))
print(os.path.exists(duplicates))
print(os.path.exists(png_dir))

## === cell 7
imgs = os.listdir(train_dir)
df_train = pd.read_csv(train_csv)
for ind in range(df_train.shape[0]):
    if df_train['image'][ind] not in imgs:
        print("{} not in train_images".foramt(df_train['image'][id]))

## === cell 8
df_duplicates = pd.read_csv(duplicates,header=None)
df_duplicates.columns = ['img1','img2']


print("The train dataset is composed of {} labeled images".format(df_train.shape[0]))

print("The test dataset is composed of {} unlabeled images".format(len(os.listdir(test_dir))))
print(df_train.head())

print("\nThere are {} duplicated images.\n".format(df_duplicates.shape[0]))

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2985615576.py in <cell line: 0>()
----> 1 df_duplicates = pd.read_csv(duplicates,header=None)
      2 df_duplicates.columns = ['img1','img2']
      3 
      4 
      5 print("The train dataset is composed of {} labeled images".format(df_train.shape[0]))

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: '../input/duplicatescsv/duplicates.csv'

## === cell 9
true_duplicates = []
false_duplicates = []
for ind in range(df_duplicates.shape[0]):
    if not df_duplicates['img1'][ind] in list(df_train["image"])  : print("{} not in training dataset".format(df_duplicates['img1'][ind]))
    elif not df_duplicates['img2'][ind] in list(df_train["image"])  : print("{} not in training dataset".format(df_duplicates['img2'][ind]))
    else : 
        if np.all(df_train[df_train["image"]==df_duplicates['img2'][ind]].reset_index()["labels"] == df_train[df_train["image"]==df_duplicates['img1'][ind]].reset_index()["labels"]):
            true_duplicates.append(df_duplicates['img1'][ind])
        else :
            false_duplicates.append((df_duplicates['img1'][ind],df_duplicates['img2'][ind]))

print('There are {} true duplicates and {} false ones.'.format(len(true_duplicates),len(false_duplicates)))
print('Lets display the false duplicates')

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/344786139.py in <cell line: 0>()
      1 true_duplicates = []
      2 false_duplicates = []
----> 3 for ind in range(df_duplicates.shape[0]):
      4     # First, check if all images are in the train dataset
      5     if not df_duplicates['img1'][ind] in list(df_train["image"])  : print("{} not in training dataset".format(df_duplicates['img1'][ind]))

NameError: name 'df_duplicates' is not defined

## === cell 10
count = 0
for img1, img2 in false_duplicates[:5]:
    fig, axs = plt.subplots(1,2)
    axs[0].imshow(plt.imread(train_dir+img1))
    axs[0].set_title(df_train[df_train["image"]==img1].reset_index()["labels"][0])
    axs[0].axis('off')
    axs[1].imshow(plt.imread(train_dir+img2))
    axs[1].set_title(df_train[df_train["image"]==img2].reset_index()["labels"][0])
    axs[1].axis('off')
    plt.savefig(png_dir+"compare_false_dup"+str(count)+".png")
    count += 1
    plt.show()

## === cell 12
labels = [x.split(' ') for x in df_train['labels']]
labels = [l for label in labels for l in label ]

uniques = np.unique(labels)
assert len(uniques)==len(CFG.classes), 'ERROR : labels and CFG.classes mismatch'
for unique in uniques : 
    assert unique in CFG.classes , 'ERROR : labels and CFG.classes mismatch'

## === cell 13
df_train['labels'] = [x.split(' ') for x in df_train['labels']]
labels = MultiLabelBinarizer(classes=CFG.classes).fit_transform(df_train['labels'].values)
labels = pd.DataFrame(columns=CFG.classes, data=labels, index=df_train.index)
df_train.drop('labels', axis = 1, inplace = True)
for col in labels.columns:
    df_train[col] = labels[col]
print(df_train.head())

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3799681172.py in <cell line: 0>()
      1 df_train['labels'] = [x.split(' ') for x in df_train['labels']]
----> 2 labels = MultiLabelBinarizer(classes=CFG.classes).fit_transform(df_train['labels'].values)
      3 labels = pd.DataFrame(columns=CFG.classes, data=labels, index=df_train.index)
      4 df_train.drop('labels', axis = 1, inplace = True)
      5 for col in labels.columns:

NameError: name 'MultiLabelBinarizer' is not defined

## === cell 14
init = df_train.shape[0]

for img1, img2 in false_duplicates:
    df_train = df_train[df_train["image"]!=img1]
    df_train = df_train[df_train["image"]!=img2]
for img in true_duplicates:
    df_train = df_train[df_train["image"]!=img]

end = df_train.shape[0]
df_train.reset_index(drop = True, inplace = True)
print("Deleted {} files".format(init-end))

## === cell 16
value_counts = lambda x: pd.Series.value_counts(x, normalize=True)
df_occurence = pd.DataFrame({
    'origin': df_train[CFG.classes].apply(value_counts).loc[1]})

bar = df_occurence.plot.barh(figsize=[15, 5], colormap='plasma')

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2154895891.py in <cell line: 0>()
      1 value_counts = lambda x: pd.Series.value_counts(x, normalize=True)
      2 df_occurence = pd.DataFrame({
----> 3     'origin': df_train[CFG.classes].apply(value_counts).loc[1]})
      4 
      5 bar = df_occurence.plot.barh(figsize=[15, 5], colormap='plasma')

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6247         if nmissing:
   6248             if nmissing == len(indexer):
-> 6249                 raise KeyError(f"None of [{key}] are in the [{axis_name}]")
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())

KeyError: "None of [Index(['complex', 'frog_eye_leaf_spot', 'powdery_mildew', 'rust', 'scab',\n       'healthy'],\n      dtype='object')] are in the [columns]"

## === cell 18
sss = StratifiedShuffleSplit(n_splits=1, test_size= CFG.test_size, random_state=CFG.seed)
X = df_train['image']
y = df_train[CFG.classes]
for train_index, test_index in sss.split(X, y):
    X_train, X_test = df_train.loc[train_index], df_train.loc[test_index]

compare_X_train, compare_X_test = train_test_split( df_train,  test_size= CFG.test_size, random_state= CFG.seed)

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1368720213.py in <cell line: 0>()
----> 1 sss = StratifiedShuffleSplit(n_splits=1, test_size= CFG.test_size, random_state=CFG.seed)
      2 X = df_train['image']
      3 y = df_train[CFG.classes]
      4 for train_index, test_index in sss.split(X, y):
      5     X_train, X_test = df_train.loc[train_index], df_train.loc[test_index]

NameError: name 'StratifiedShuffleSplit' is not defined

## === cell 19
df_occurence = pd.DataFrame({
    'origin': df_train[CFG.classes].apply(value_counts).loc[1],
    'stratified_train': X_train[CFG.classes].apply(value_counts).loc[1],
    'stratified_test': X_test[CFG.classes].apply(value_counts).loc[1],
    'compare_train': compare_X_train[CFG.classes].apply(value_counts).loc[1],
    'compare_test': compare_X_test[CFG.classes].apply(value_counts).loc[1]
})

bar = df_occurence.plot.barh(figsize=[15, 5], colormap='plasma')
plt.savefig(png_dir+'comparison_stratified.png')

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/829283298.py in <cell line: 0>()
      1 df_occurence = pd.DataFrame({
----> 2     'origin': df_train[CFG.classes].apply(value_counts).loc[1],
      3     'stratified_train': X_train[CFG.classes].apply(value_counts).loc[1],
      4     'stratified_test': X_test[CFG.classes].apply(value_counts).loc[1],
      5     'compare_train': compare_X_train[CFG.classes].apply(value_counts).loc[1],

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6247         if nmissing:
   6248             if nmissing == len(indexer):
-> 6249                 raise KeyError(f"None of [{key}] are in the [{axis_name}]")
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())

KeyError: "None of [Index(['complex', 'frog_eye_leaf_spot', 'powdery_mildew', 'rust', 'scab',\n       'healthy'],\n      dtype='object')] are in the [columns]"

## === cell 21
def pred2labels(pred, thresh = 0.5, labels = CFG.classes):

    assert len(pred)==len(labels), 'Predictions must have shape : ({},)'.format(len(labels))
    pred = [labels[i] for i in range(len(labels)) if pred[i]>thresh]
    pred = np.array(pred)
    res = ''
    for p in pred :
        if res == '':
            res += p
        else:
            res += ' '+p
    return res

## === cell 22
AUTOTUNE = tf.data.AUTOTUNE
data_augmentation = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical", seed = CFG.seed),
                                         tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
                                         tf.keras.layers.experimental.preprocessing.RandomContrast([0,0.3], seed= CFG.seed ),
                                         tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=0.2, width_factor=0.2)
                                        ])

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1227881770.py in <cell line: 0>()
      1 AUTOTUNE = tf.data.AUTOTUNE
----> 2 data_augmentation = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical", seed = CFG.seed),
      3                                          tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
      4                                          tf.keras.layers.experimental.preprocessing.RandomContrast([0,0.3], seed= CFG.seed ),
      5                                          tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=0.2, width_factor=0.2)

AttributeError: module 'keras._tf_keras.keras.layers' has no attribute 'experimental'

## === cell 23
def parse_image(file_path):
    img = tf.io.read_file(train_dir + file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [CFG.img_size,CFG.img_size])
    return img

def prepare_dataset(X, augmentation = False):
    dataset = tf.data.Dataset.from_tensor_slices((X['image'].values, X[CFG.classes].values ))
    dataset = dataset.map(lambda x ,y : (parse_image(x),y) )
    dataset = dataset.batch(CFG.batch_size)
    
    if augmentation :
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y), 
                                            num_parallel_calls=AUTOTUNE)
    dataset = dataset.repeat().prefetch(buffer_size=AUTOTUNE)
    
    return dataset

## === cell 24
ds_train = prepare_dataset(X_train, augmentation = True)
ds_test = prepare_dataset(X_test)

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/342475226.py in <cell line: 0>()
----> 1 ds_train = prepare_dataset(X_train, augmentation = True)
      2 ds_test = prepare_dataset(X_test)

NameError: name 'X_train' is not defined

## === cell 25
for inputs, outputs in ds_train.as_numpy_iterator():
    print("Input shape is:", inputs.shape, "output shape is:", outputs.shape)

    plt.imshow(inputs[0])
    plt.show()
    print('label of this input is', outputs[0], 'corresponding to', pred2labels(outputs[0]))

    break

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2000438112.py in <cell line: 0>()
----> 1 for inputs, outputs in ds_train.as_numpy_iterator():
      2     # Verify the shapes are still as we expect
      3     print("Input shape is:", inputs.shape, "output shape is:", outputs.shape)
      4 
      5     # Print the first element and the label

NameError: name 'ds_train' is not defined

## === cell 27
def create_cnn(input_shape, output_length,
               nb_cnn=3, nb_filters = 64, activation_cnn = 'relu', 
               model_transfert = None, fine_tune = False, 
               nb_FC_layer = 3, nb_FC_neurons = 512, reducing = False, activation_FC = 'relu',
               dropout = 0.0,
               activation_output = 'sigmoid',
               name = 'my_cnn_model'
               ):
    '''Create a CNN based model is model_transfert is None. Else, the model_transfert is used for feature extraction. 
    If reducing is not False, nb_FC_neurons must be multiple of 2**nb_FC_layer '''
    
    assert input_shape[-1] == 3, 'For the moment only models with rgb input is dealt'
    if reducing : assert nb_FC_neurons % 2**nb_FC_layer == 0 , 'If reducing, nb_FC_neurons must be multiple of 2**nb_FC_layer '
        
    model = tf.keras.models.Sequential(name=name)
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape, name = 'Input_layer'))
    
    if model_transfert == None: 
        for cnn in range(nb_cnn):
            model.add(tf.keras.layers.Conv2D( filters = nb_filters, kernel_size = (3,3), padding='same', activation = activation_cnn, name ='Conv2D_'+str(cnn+1) ))
            model.add(tf.keras.layers.MaxPooling2D( pool_size=(2, 2), name ='MaxPool_'+str(cnn+1)))
    else : 
        if not fine_tune : model_transfert.trainable = False
            
        model.add(model_transfert)
        model.add(tf.keras.layers.MaxPooling2D( pool_size=(2, 2), name ='MaxPool_transfer'))
        
    model.add(tf.keras.layers.Flatten())
    
    if reducing : 
        for FC in range(nb_FC_layer):
            model.add(tf.keras.layers.Dense(nb_FC_neurons/2**FC, activation= activation_FC, name='FC_layer_'+str(FC+1)))
            
            if dropout != 0.0: 
                model.add(tf.keras.layers.Dropout(dropout, name = 'Dropout_'+str(FC+1)))
    else:
        for FC in range(nb_FC_layer):
            model.add(tf.keras.layers.Dense(nb_FC_neurons, activation= activation_FC, name='FC_layer_'+str(FC+1)))
        if dropout != 0.0:  
            model.add(tf.keras.layers.Dropout(dropout, name = 'Dropout_'+str(FC+1)))

    model.add(tf.keras.layers.Dense(output_length, activation = activation_output ,name='Output_layer'))

    return model


def get_callbacks(monitor='val_loss',save_name=None,patience=8):
    '''Returns the wanted callbacks to save models and avoid overfitting.
    monitor (str, optional): the monitor to check for the early stopping. Default is 'val_loss'
    save_name (str, optional): if not None, uses modelcheckpoint and saves checkpoints at the save_name. Default is None.
    patience (int, optional): number of epoch to wait for improvment of monitor. Default is 8.'''
    if save_name :
        return [tf.keras.callbacks.ModelCheckpoint(filepath=save_name,
                                                   monitor=monitor, 
                                                   save_best_only=True,
                                                   verbose=0),
                tf.keras.callbacks.EarlyStopping(monitor=monitor, 
                                                 patience=patience,
                                                 restore_best_weights=True)
                ]
    else:
        return [tf.keras.callbacks.EarlyStopping(monitor=monitor, 
                                                 patience=patience,
                                                 restore_best_weights=True)
                ]

## === cell 28
try : 
    base = tf.keras.applications.Xception( include_top=False, weights='imagenet', input_shape=(CFG.img_size, CFG.img_size, 3), classes=len(CFG.classes) )

    model = create_cnn(input_shape=(CFG.img_size, CFG.img_size, 3), output_length=len(CFG.classes),
                   model_transfert = base, fine_tune = True, 
                   nb_FC_layer = 2, nb_FC_neurons = 512, reducing = True, activation_FC = 'relu',
                   dropout = 0,
                   activation_output = 'sigmoid',
                   name='my_model'
                   )
    optimizer = tf.keras.optimizers.Adam(learning_rate=3.5e-5)

    model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=[
                        tf.keras.metrics.BinaryAccuracy(name='acc'), 
                        tfa.metrics.F1Score(
                            num_classes=len(CFG.classes), 
                            average='micro')])

    model.summary()
except : 
    print('Internet not available')

## === cell 29
if os.path.exists(model_dir + 'model.h5') and not CFG.retrain : 
    print('Loading model from file')
    model = tf.keras.models.load_model(model_dir+'model.h5')
    history = None
else :
    history = model.fit(ds_train,
                          validation_data=ds_test,
                          steps_per_epoch=(X_train.shape[0]*0.8)//CFG.batch_size, 
                          validation_steps= (X_train.shape[0]*0.2)//CFG.batch_size,
                          callbacks = get_callbacks(monitor = 'val_f1_score', save_name = '/kaggle/working/model.h5', patience = 2),
                        epochs = 6)

## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1327298128.py in <cell line: 0>()
      4     history = None
      5 else :
----> 6     history = model.fit(ds_train,
      7                           validation_data=ds_test,
      8                           steps_per_epoch=(X_train.shape[0]*0.8)//CFG.batch_size,

NameError: name 'model' is not defined

## === cell 31
if history : 
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))

    axes[0].plot(history.history['loss'], label = 'Train loss')
    axes[0].plot(history.history['val_loss'], label = 'Validation loss')
    axes[0].set_title('Loss')

    axes[1].plot(history.history['acc'], label = 'Train accuracy')
    axes[1].plot(history.history['val_acc'], label = 'Validation accuracy')
    axes[1].set_title('Accuracy')

    axes[2].plot(history.history['f1_score'], label = 'Train micro-F1')
    axes[2].plot(history.history['val_f1_score'], label = 'Validation micro-F1')
    axes[2].set_title('micro-F1')

    plt.savefig(png_dir+'history_xception.png')

    plt.show()
else : 
    print('There is no history')

## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2435680254.py in <cell line: 0>()
----> 1 if history :
      2     fig, axes = plt.subplots(1, 3, figsize=(10, 5))
      3 
      4     axes[0].plot(history.history['loss'], label = 'Train loss')
      5     axes[0].plot(history.history['val_loss'], label = 'Validation loss')

NameError: name 'history' is not defined

## === cell 33
def parse_test_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [CFG.img_size,CFG.img_size])
    return img



def predict_new(path, model):
    img = parse_test_image(path)
    img = tf.expand_dims(img,axis = 0)
    pred = model.predict(img)
    return pred2labels(pred[0])

## === cell 34
df_sub = pd.DataFrame(columns=['image','labels'])
for path in os.listdir(test_dir):
    pred = predict_new(test_dir+path, model)
    
    df_sub = df_sub.append( {'image': path, 'labels': pred}, ignore_index = True )
    
print(df_sub.head())
df_sub.to_csv('submission.csv', index=False)
print('Submission completed')

## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3339695308.py in <cell line: 0>()
      1 df_sub = pd.DataFrame(columns=['image','labels'])
      2 for path in os.listdir(test_dir):
----> 3     pred = predict_new(test_dir+path, model)
      4 
      5     df_sub = df_sub.append( {'image': path, 'labels': pred}, ignore_index = True )

NameError: name 'model' is not defined
