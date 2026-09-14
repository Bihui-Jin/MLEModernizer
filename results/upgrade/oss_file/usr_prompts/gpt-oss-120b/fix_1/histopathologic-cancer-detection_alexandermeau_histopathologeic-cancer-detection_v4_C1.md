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

3.14

# 3. Installed packages

No external packages required in the script and installed.

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

0.9374697534977384

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 2

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
tif_files = []

for dirname, _, filenames in os.walk('/kaggle/input'):
    print(f"\nos path: [{dirname}]")
    tif_file_count =0
    for filename in filenames:
        if filename.endswith('.tif'):
            tif_file_count +=1
            if tif_file_count < 3:
                print(os.path.join(dirname, filename))
            elif tif_file_count == 3:
                print("...")
        else:
            print(os.path.join(dirname, filename))  
    if tif_file_count > 0:
        print(f"\nTotal number of .tif files in {dirname}: {tif_file_count}")
        tif_file_count =0

        


## === cell 4
import pandas as pd
import os 

train_labels = pd.read_csv('/kaggle/input/competitions/histopathologic-cancer-detection/train_labels.csv')
print("shape:",train_labels.shape)
print("-- Label head ---")
print("################")
print(train_labels.head())
print("-----------------")
print("Missing values: ")
print("################")
print(train_labels.isnull().sum())
print("-----------------")
print(f"Sum of duplicated labels: {train_labels.duplicated().sum()}.")

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1638646809.py in <cell line: 0>()
      2 import os
      3 
----> 4 train_labels = pd.read_csv('/kaggle/input/competitions/histopathologic-cancer-detection/train_labels.csv')
      5 print("shape:",train_labels.shape)
      6 print("-- Label head ---")

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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/competitions/histopathologic-cancer-detection/train_labels.csv'

## === cell 5
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='label', data=train_labels)
plt.xticks([0, 1], ['Non-Cancer (0)', 'Cancer (1)'])
plt.title("Distribution of Cancer vs Non-Cancer Images")
plt.ylabel("Image Count")
plt.xlabel("Label")
plt.show()


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/593333409.py in <cell line: 0>()
      2 import matplotlib.pyplot as plt
      3 
----> 4 sns.countplot(x='label', data=train_labels)
      5 plt.xticks([0, 1], ['Non-Cancer (0)', 'Cancer (1)'])
      6 plt.title("Distribution of Cancer vs Non-Cancer Images")

NameError: name 'train_labels' is not defined

## === cell 6
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

train_dir = Path("/kaggle/input/competitions/histopathologic-cancer-detection/train")
train_labels = pd.read_csv('/kaggle/input/competitions/histopathologic-cancer-detection/train_labels.csv')
train_labels['train_filepath'] = train_labels['id'].apply(lambda x: str(train_dir / f'{x}.tif'))

def show_samples(label, df=train_labels, num_images=5):
    subset = df[df['label'] == label].sample(num_images, random_state=42)
    plt.figure(figsize=(12, 4))
    for i, row in enumerate(subset.itertuples()):
        img = cv2.imread(row.train_filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        center = (32, 32, 64, 64)  
        cv2.rectangle(img, (center[0], center[1]), (center[2], center[3]), (255, 0, 0), 1)
        plt.subplot(1, num_images, i+1)
        plt.imshow(img)
        plt.title(f'Label: {row.label}')
        plt.axis('off')
    plt.show()


show_samples(0)  # Non-cancerous
show_samples(1)  # Cancerous

img_path = f"/kaggle/input/competitions/histopathologic-cancer-detection/train/{train_labels['id'].iloc[0]}.tif"
img = Image.open(img_path)
print(f"image size: {img.size}") 
print(f"image mode: {img.mode}") 

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2779736353.py in <cell line: 0>()
      7 # add 'train_filepath' to dataframe
      8 train_dir = Path("/kaggle/input/competitions/histopathologic-cancer-detection/train")
----> 9 train_labels = pd.read_csv('/kaggle/input/competitions/histopathologic-cancer-detection/train_labels.csv')
     10 train_labels['train_filepath'] = train_labels['id'].apply(lambda x: str(train_dir / f'{x}.tif'))
     11 # # convert 'label' field to string.

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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/competitions/histopathologic-cancer-detection/train_labels.csv'

## === cell 7
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import tensorflow_io as tfio
print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 9
from sklearn.model_selection import train_test_split
import cv2


train_df, val_df = train_test_split(
    train_labels,
    test_size=0.2,
    stratify=train_labels['label'],
    random_state=42
)

def load_image(path, label):
    def _load_image_py(path_tensor):
        path_str = path_tensor.numpy()
        if isinstance(path_str, bytes):
            path_str = path_str.decode("utf-8")
        img = cv2.imread(path_str)
        if img is None:
            raise ValueError(f"Failed to load image at path: {path_str}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img.astype(np.uint8)

    image = tf.py_function(func=_load_image_py, inp=[path], Tout=tf.uint8)
    image.set_shape([96, 96, 3])
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(label, tf.int32)  
    return image, label

train_dataset = tf.data.Dataset.from_tensor_slices((train_df['train_filepath'], train_df['label']))
train_dataset = train_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(1000).batch(64).prefetch(tf.data.AUTOTUNE)

validation_dataset = tf.data.Dataset.from_tensor_slices((val_df['train_filepath'], val_df['label']))
validation_dataset = validation_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.batch(64).prefetch(tf.data.AUTOTUNE)

print("Data Pre-processing complete..")

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3196099194.py in <cell line: 0>()
      5 # Split the DataFrame into training and validation sets
      6 train_df, val_df = train_test_split(
----> 7     train_labels,
      8     test_size=0.2,
      9     stratify=train_labels['label'],

NameError: name 'train_labels' is not defined

## === cell 11
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.metrics import AUC

cnn_model = Sequential([
    Conv2D(32, kernel_size=5, strides=1, padding='same', input_shape=(96, 96, 3)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=2, strides=2), 
    
    Conv2D(64, kernel_size=3, strides=1, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=2, strides=2), 
    
    Conv2D(128, kernel_size=3, strides=1, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=2, strides=2), 
    
    Flatten(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid') 
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', AUC(name='auroc')])
cnn_model.summary()
three_layer_history = cnn_model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/376880218.py in <cell line: 0>()
     34 cnn_model.summary()
     35 three_layer_history = cnn_model.fit(
---> 36     train_dataset,
     37     validation_data=validation_dataset,
     38     epochs=10

NameError: name 'train_dataset' is not defined

## === cell 12
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

print("\nTraining Metrics:")
print("---------------------------")
for metric in ['loss', 'accuracy', 'auroc']:
    print(f"{metric.capitalize()}: {three_layer_history.history[metric][-1]:.4f}")

print("\nValidation Metrics:")
print("---------------------------")
for metric in ['val_loss', 'val_accuracy', 'val_auroc']:
    print(f"{metric.replace('val_', 'Validation ').capitalize()}: {three_layer_history.history[metric][-1]:.4f}")



y_true = []
y_pred = []
for images, labels in validation_dataset:
    preds = cnn_model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend((preds > 0.5).astype(int).flatten())
y_true = np.array(y_true)
y_pred = np.array(y_pred)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'])

plt.title('Confusion Matrix (Validation DataSet)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print("\nConfusion Matrix Metrics:")
print(f"True Negatives (TN): {cm[0,0]}")
print(f"False Positives (FP): {cm[0,1]}")
print(f"False Negatives (FN): {cm[1,0]}")
print(f"True Positives (TP): {cm[1,1]}")

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2764352205.py in <cell line: 0>()
      7 print("---------------------------")
      8 for metric in ['loss', 'accuracy', 'auroc']:
----> 9     print(f"{metric.capitalize()}: {three_layer_history.history[metric][-1]:.4f}")
     10 
     11 print("\nValidation Metrics:")

NameError: name 'three_layer_history' is not defined

## === cell 13
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(three_layer_history.history['loss'], label='Training Loss')
plt.plot(three_layer_history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(three_layer_history.history['accuracy'], label='Training Accuracy')
plt.plot(three_layer_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(three_layer_history.history['auroc'], label='Training AUC')
plt.plot(three_layer_history.history['val_auroc'], label='Validation AUC')
plt.title('Training vs Validation AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1934018870.py in <cell line: 0>()
      8 plt.figure(figsize=(15, 5))
      9 plt.subplot(1, 3, 1)
---> 10 plt.plot(three_layer_history.history['loss'], label='Training Loss')
     11 plt.plot(three_layer_history.history['val_loss'], label='Validation Loss')
     12 plt.title('Training vs Validation Loss')

NameError: name 'three_layer_history' is not defined

## === cell 16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.metrics import AUC

two_layer_cnn_model = Sequential([
    Conv2D(32, kernel_size=5, strides=1, padding='same', input_shape=(96, 96, 3)),
    BatchNormalization(), Activation('relu'), MaxPooling2D(pool_size=2, strides=2),

    Conv2D(64, kernel_size=3, strides=1, padding='same'),
    BatchNormalization(), Activation('relu'), MaxPooling2D(pool_size=2, strides=2),

    Flatten(),Dropout(0.5),
    Dense(256, activation='relu'),Dropout(0.3), Dense(1, activation='sigmoid') 
])

two_layer_cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', AUC(name='auroc')])
two_layer_history = two_layer_cnn_model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)

print("\nTraining Metrics:")
print("---------------------------")
for metric in ['loss', 'accuracy', 'auroc']:
    print(f"{metric.capitalize()}: {two_layer_history.history[metric][-1]:.4f}")

print("\nValidation Metrics:")
print("---------------------------")
for metric in ['val_loss', 'val_accuracy', 'val_auroc']:
    print(f"{metric.replace('val_', 'Validation ').capitalize()}: {two_layer_history.history[metric][-1]:.4f}")


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1827737584.py in <cell line: 0>()
     18 two_layer_cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', AUC(name='auroc')])
     19 two_layer_history = two_layer_cnn_model.fit(
---> 20     train_dataset,
     21     validation_data=validation_dataset,
     22     epochs=10

NameError: name 'train_dataset' is not defined

## === cell 17
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.metrics import AUC

five_layer_cnn_model = Sequential([
    Conv2D(32, kernel_size=5, strides=1, padding='same', input_shape=(96, 96, 3)),
    BatchNormalization(), Activation('relu'), MaxPooling2D(pool_size=2, strides=2),

    Conv2D(64, kernel_size=3, strides=1, padding='same'),
    BatchNormalization(), Activation('relu'), MaxPooling2D(pool_size=2, strides=2),

    Conv2D(128, kernel_size=3, strides=1, padding='same'),
    BatchNormalization(), Activation('relu'), MaxPooling2D(pool_size=2, strides=2),

    Conv2D(256, kernel_size=3, strides=1, padding='same'),
    BatchNormalization(), Activation('relu'), MaxPooling2D(pool_size=2, strides=2),

    Conv2D(512, kernel_size=3, strides=1, padding='same'),
    BatchNormalization(), Activation('relu'), MaxPooling2D(pool_size=2, strides=2),
        
    Flatten(),Dropout(0.5),
    Dense(256, activation='relu'),Dropout(0.3), Dense(1, activation='sigmoid') 
])

five_layer_cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', AUC(name='auroc')])
five_layer_history = five_layer_cnn_model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)

print("\nTraining Metrics:")
print("---------------------------")
for metric in ['loss', 'accuracy', 'auroc']:
    print(f"{metric.capitalize()}: {five_layer_history.history[metric][-1]:.4f}")

print("\nValidation Metrics:")
print("---------------------------")
for metric in ['val_loss', 'val_accuracy', 'val_auroc']:
    print(f"{metric.replace('val_', 'Validation ').capitalize()}: {five_layer_history.history[metric][-1]:.4f}")


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2638186377.py in <cell line: 0>()
     29 five_layer_cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', AUC(name='auroc')])
     30 five_layer_history = five_layer_cnn_model.fit(
---> 31     train_dataset,
     32     validation_data=validation_dataset,
     33     epochs=10

NameError: name 'train_dataset' is not defined

## === cell 18
import matplotlib.pyplot as plt

histories = {
    "2-layer CNN": two_layer_history,
    "3-layer CNN": three_layer_history,
    "5-layer CNN": five_layer_history
}

metrics = ['accuracy', 'val_accuracy']
titles = {
    'accuracy': 'Training Accuracy',
    'val_accuracy': 'Validation Accuracy'
}

for metric in ['accuracy', 'val_accuracy']:    
    plt.figure(figsize=(6, 4))
    for label, history in histories.items():
        plt.plot(history.history[metric], label=label)
    plt.title(f"{titles[metric]} Comparison")
    plt.xlabel("Epoch")
    plt.ylabel(metric.split('_')[-1].capitalize())
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

records = []
for name, hist in histories.items():
    max_train_accuracy = max(hist.history['accuracy'])
    max_val_accuracy = max(hist.history['val_accuracy'])
    last_train_accuracy = hist.history['accuracy'][-1]
    last_val_accuracy = hist.history['val_accuracy'][-1]
    train_loss = hist.history['loss'][-1]
    val_loss = hist.history['val_loss'][-1]
    max_auroc = max(hist.history['val_auroc'])
    last_auroc = hist.history['val_auroc'][-1]    
    records.append({
        "Model": name,
        "Max Training Accuracy": round(max_train_accuracy, 4),
        "Max Validation Accuracy": round(max_val_accuracy, 4),
        "Last Train Acc": round(last_train_accuracy, 4),
        "Last Val Acc": round(last_val_accuracy, 4),
        "Train Loss": round(train_loss, 4),
        "Val Loss": round(val_loss, 4),
        "Max Val AUC": round(max_auroc, 4),
        "Last Val AUC": round(last_auroc, 4)         
    })
metrics_df = pd.DataFrame(records)
metrics_df

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3625225363.py in <cell line: 0>()
      2 
      3 histories = {
----> 4     "2-layer CNN": two_layer_history,
      5     "3-layer CNN": three_layer_history,
      6     "5-layer CNN": five_layer_history

NameError: name 'two_layer_history' is not defined

## === cell 20
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(five_layer_history.history['loss'], label='Training Loss')
plt.plot(five_layer_history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss \n(Best Model Architecture)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(five_layer_history.history['accuracy'], label='Training Accuracy')
plt.plot(five_layer_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy \n(Best Model Architecture)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(five_layer_history.history['auroc'], label='Training AUC')
plt.plot(five_layer_history.history['val_auroc'], label='Validation AUC')
plt.title('Training vs Validation AUC \n(Best Model Architecture)')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3438908836.py in <cell line: 0>()
      6 plt.figure(figsize=(15, 5))
      7 plt.subplot(1, 3, 1)
----> 8 plt.plot(five_layer_history.history['loss'], label='Training Loss')
      9 plt.plot(five_layer_history.history['val_loss'], label='Validation Loss')
     10 plt.title('Training vs Validation Loss \n(Best Model Architecture)')

NameError: name 'five_layer_history' is not defined

## === cell 22
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.metrics import AUC


filter_config = [(16, 32, 64, 128, 256), (24, 48, 96, 192, 384)]
history_dict = {} 
history_dict[(32, 64, 128, 256, 512)] = five_layer_history
best_validation_accuracy = five_layer_history.history['val_accuracy'][-1]
best_filter_config = (32, 64, 128, 256, 512)
best_filter_history = five_layer_history

for idx, (f1, f2, f3, f4, f5) in enumerate(filter_config):
    filter_label = (f1, f2, f3, f4, f5)
    print(f"\nTraining Model with Filters: {filter_label}...")
    model_filter_tuning = Sequential([
        Conv2D(f1, kernel_size=5, strides=1, padding='same', input_shape=(96, 96, 3)),
        BatchNormalization(), Activation('relu'), MaxPooling2D(pool_size=2, strides=2),
    
        Conv2D(f2, kernel_size=3, strides=1, padding='same'),
        BatchNormalization(), Activation('relu'), MaxPooling2D(pool_size=2, strides=2),
    
        Conv2D(f3, kernel_size=3, strides=1, padding='same'),
        BatchNormalization(), Activation('relu'), MaxPooling2D(pool_size=2, strides=2),
    
        Conv2D(f4, kernel_size=3, strides=1, padding='same'),
        BatchNormalization(), Activation('relu'), MaxPooling2D(pool_size=2, strides=2),
    
        Conv2D(f5, kernel_size=3, strides=1, padding='same'),
        BatchNormalization(), Activation('relu'), MaxPooling2D(pool_size=2, strides=2),
            
        Flatten(),Dropout(0.5),
        Dense(256, activation='relu'),Dropout(0.3), Dense(1, activation='sigmoid') 
    ])

        
    model_filter_tuning.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', AUC(name='auroc')])
    history_filter_tuning = model_filter_tuning.fit(train_dataset, validation_data=validation_dataset, epochs=10, verbose=1)
    history_dict[filter_label] = history_filter_tuning
    final_validation_accuracy =  history_filter_tuning.history['val_accuracy'][-1]
    print(f"Final Validation Accuracy for {filter_label}: {final_validation_accuracy:.4f}")

    if final_validation_accuracy > best_validation_accuracy:
        best_validation_accuracy = final_validation_accuracy
        best_filter_config = filter_label
        best_filter_history = history_filter_tuning

print("Best Filter Configuration:")
print(f"Filters: {best_filter_config}")
print(f"Validation Accuracy: {best_validation_accuracy:.4f}")

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1298897845.py in <cell line: 0>()
      7 # add previous model result
      8 history_dict = {}
----> 9 history_dict[(32, 64, 128, 256, 512)] = five_layer_history
     10 best_validation_accuracy = five_layer_history.history['val_accuracy'][-1]
     11 best_filter_config = (32, 64, 128, 256, 512)

NameError: name 'five_layer_history' is not defined

## === cell 23
records_filter = []
for name, hist in history_dict.items():
    max_train_accuracy = max(hist.history['accuracy'])
    max_val_accuracy = max(hist.history['val_accuracy'])
    last_train_accuracy = hist.history['accuracy'][-1]
    last_val_accuracy = hist.history['val_accuracy'][-1]
    train_loss = hist.history['loss'][-1]
    val_loss = hist.history['val_loss'][-1]
    max_auroc = max(hist.history['val_auroc'])
    last_auroc = hist.history['val_auroc'][-1]
    
    records_filter.append({
        "name": name,
        "Max Train Accuracy": round(max_train_accuracy, 4),
        "Max Validation Accuracy": round(max_val_accuracy, 4),
        "Last Train Acc": round(last_train_accuracy, 4),
        "Last Validation Acc": round(last_val_accuracy, 4),
        "Train Loss": round(train_loss, 4),
        "Validation Loss": round(val_loss, 4),
        "Max Train AUC": round(max_auroc, 4),
        "Last Validation AUC": round(last_auroc, 4)        
    })
metrics_df_filter = pd.DataFrame(records_filter)
metrics_df_filter

## === cell 24
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(best_filter_history.history['loss'], label='Training Loss')
plt.plot(best_filter_history.history['val_loss'], label='Validation Loss')
plt.title(f"Training vs Validation Loss  \n(Best Num of Filters {best_filter_config}")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(best_filter_history.history['accuracy'], label='Training Accuracy')
plt.plot(best_filter_history.history['val_accuracy'], label='Validation Accuracy')
plt.title(f"Training vs Validation Accuracy \n(Best Num of Filters {best_filter_config}")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(best_filter_history.history['auroc'], label='Training AUC')
plt.plot(best_filter_history.history['val_auroc'], label='Validation AUC')
plt.title(f"Training vs Validation AUC \n(Best Num of Filters {best_filter_config}")
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3736170152.py in <cell line: 0>()
      8 plt.figure(figsize=(15, 5))
      9 plt.subplot(1, 3, 1)
---> 10 plt.plot(best_filter_history.history['loss'], label='Training Loss')
     11 plt.plot(best_filter_history.history['val_loss'], label='Validation Loss')
     12 plt.title(f"Training vs Validation Loss  \n(Best Num of Filters {best_filter_config}")

NameError: name 'best_filter_history' is not defined

## === cell 27
import tensorflow as tf

optimizer = tf.keras.optimizers.Adam()
default_lr = optimizer.learning_rate.numpy()
print(f"Default learning rate: {default_lr}")

## === cell 28
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

learning_rates_to_epoch = {
    5e-4: 20,   # (lower learning rate: 0.0005)
    5e-3: 10    # (higher learning rate=0.005)
}

history_dict_lr = {} 
history_dict_lr[default_lr] = five_layer_history
best_lr_validation_accuracy = five_layer_history.history['val_accuracy'][-1]
best_lr = default_lr
best_lr_history = five_layer_history

for lr, num_epochs in learning_rates_to_epoch.items():
    print(f"\nTraining Model with Learning Rate: {lr}...")
    
    model_lr_tuning = Sequential([
        Conv2D(f1, kernel_size=5, strides=1, padding='same', input_shape=(96, 96, 3)),
        BatchNormalization(), Activation('relu'), MaxPooling2D(pool_size=2, strides=2),
    
        Conv2D(f2, kernel_size=3, strides=1, padding='same'),
        BatchNormalization(), Activation('relu'), MaxPooling2D(pool_size=2, strides=2),
    
        Conv2D(f3, kernel_size=3, strides=1, padding='same'),
        BatchNormalization(), Activation('relu'), MaxPooling2D(pool_size=2, strides=2),
    
        Conv2D(f4, kernel_size=3, strides=1, padding='same'),
        BatchNormalization(), Activation('relu'), MaxPooling2D(pool_size=2, strides=2),
    
        Conv2D(f5, kernel_size=3, strides=1, padding='same'),
        BatchNormalization(), Activation('relu'), MaxPooling2D(pool_size=2, strides=2),
            
        Flatten(),Dropout(0.5),
        Dense(256, activation='relu'),Dropout(0.3), Dense(1, activation='sigmoid') 
    ])


    model_lr_tuning.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy', AUC(name='auroc')])

    early_stopping = EarlyStopping(monitor='val_auroc', patience=3, mode='max', restore_best_weights=True)
    history_lr_turning = model_lr_tuning.fit(train_dataset, validation_data=validation_dataset,epochs=num_epochs, callbacks=[early_stopping])
    history_dict_lr[lr] = history_lr_turning
    final_lr_validation_accuracy =  history_lr_turning.history['val_accuracy'][-1]
    print(f"Final Validation Accuracy for {lr}: {final_lr_validation_accuracy:.4f}")
    
    if final_lr_validation_accuracy > best_lr_validation_accuracy:
        best_lr_validation_accuracy = final_lr_validation_accuracy
        best_lr = lr
        best_lr_history = history_lr_turning

print("Best Learing Rate Configuration:")
print(f"Learning Rate: {best_lr}")
print(f"Validation Accuracy: {best_lr_validation_accuracy:.4f}")

## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2177349579.py in <cell line: 0>()
     13 
     14 history_dict_lr = {}
---> 15 history_dict_lr[default_lr] = five_layer_history
     16 best_lr_validation_accuracy = five_layer_history.history['val_accuracy'][-1]
     17 best_lr = default_lr

NameError: name 'five_layer_history' is not defined

## === cell 29
records_lr = []
for name, hist in history_dict_lr.items():
    max_train_accuracy = max(hist.history['accuracy'])
    max_val_accuracy = max(hist.history['val_accuracy'])
    last_train_accuracy = hist.history['accuracy'][-1]
    last_val_accuracy = hist.history['val_accuracy'][-1]
    train_loss = hist.history['loss'][-1]
    val_loss = hist.history['val_loss'][-1]
    max_auroc = max(hist.history['val_auroc'])
    last_auroc = hist.history['val_auroc'][-1]    
    records_lr.append({
        "Learning Rate": name,
        "Max Training Accuracy": round(max_train_accuracy, 4),
        "Max Validation Accuracy": round(max_val_accuracy, 4),
        "Last Train Acc": round(last_train_accuracy, 4),
        "Last Val Acc": round(last_val_accuracy, 4),
        "Train Loss": round(train_loss, 4),
        "Val Loss": round(val_loss, 4),
        "Max Val AUC": round(max_auroc, 4),
        "Last Val AUC": round(last_auroc, 4)                
    })
metrics_df_lr = pd.DataFrame(records_lr)
metrics_df_lr

## === cell 32
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


model_best = Sequential([
    Conv2D(32, kernel_size=5, strides=1, padding='same', input_shape=(96, 96, 3)),
    BatchNormalization(), Activation('relu'), MaxPooling2D(pool_size=2, strides=2),

    Conv2D(64, kernel_size=3, strides=1, padding='same'),
    BatchNormalization(), Activation('relu'), MaxPooling2D(pool_size=2, strides=2),

    Conv2D(128, kernel_size=3, strides=1, padding='same'),
    BatchNormalization(), Activation('relu'), MaxPooling2D(pool_size=2, strides=2),

    Conv2D(256, kernel_size=3, strides=1, padding='same'),
    BatchNormalization(), Activation('relu'), MaxPooling2D(pool_size=2, strides=2),

    Conv2D(512, kernel_size=3, strides=1, padding='same'),
    BatchNormalization(), Activation('relu'), MaxPooling2D(pool_size=2, strides=2),
        
    Flatten(),Dropout(0.5),
    Dense(256, activation='relu'),Dropout(0.3), Dense(1, activation='sigmoid') 
])

model_best.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy', AUC(name='auroc')])
checkpoint = ModelCheckpoint('model_best.h5', monitor='val_auroc', save_best_only=True, mode='max')

early_stopping = EarlyStopping(monitor='val_auroc', patience=3, mode='max', restore_best_weights=True)
history_best_model = model_best.fit(train_dataset, validation_data=validation_dataset,epochs=num_epochs, callbacks=[early_stopping, checkpoint])


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3388504025.py in <cell line: 0>()
     36 
     37 early_stopping = EarlyStopping(monitor='val_auroc', patience=3, mode='max', restore_best_weights=True)
---> 38 history_best_model = model_best.fit(train_dataset, validation_data=validation_dataset,epochs=num_epochs, callbacks=[early_stopping, checkpoint])

NameError: name 'train_dataset' is not defined

## === cell 33
from tensorflow.keras.models import load_model
import pandas as pd
import os


test_dir = Path("/kaggle/input/competitions/histopathologic-cancer-detection/test")
test_df = pd.read_csv('/kaggle/input/competitions/histopathologic-cancer-detection/sample_submission.csv')  
test_df['test_filepath'] = test_df['id'].apply(lambda x: str(test_dir / f'{x}.tif'))

test_dataset = tf.data.Dataset.from_tensor_slices((test_df['test_filepath'], test_df['label']))
test_dataset = test_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(64).prefetch(tf.data.AUTOTUNE)


predictions = model_best.predict(test_dataset, verbose=0)
test_df['label'] = predictions.flatten()

submission_path = 'submission.csv'
test_df[['id', 'label']].to_csv(submission_path, index=False)
print(f"Submission file saved to {submission_path}")


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3871544921.py in <cell line: 0>()
      7 
      8 test_dir = Path("/kaggle/input/competitions/histopathologic-cancer-detection/test")
----> 9 test_df = pd.read_csv('/kaggle/input/competitions/histopathologic-cancer-detection/sample_submission.csv')
     10 test_df['test_filepath'] = test_df['id'].apply(lambda x: str(test_dir / f'{x}.tif'))
     11 

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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/competitions/histopathologic-cancer-detection/sample_submission.csv'

## === cell 34
submission_path = '/kaggle/working/submission.csv'
test_df[['id', 'label']].to_csv(submission_path, index=False)
print(f"Submission file saved to {submission_path}")

## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1474738847.py in <cell line: 0>()
      1 submission_path = '/kaggle/working/submission.csv'
----> 2 test_df[['id', 'label']].to_csv(submission_path, index=False)
      3 print(f"Submission file saved to {submission_path}")

NameError: name 'test_df' is not defined

## === cell 38
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix

y_true_best_model = []
for _, labels in validation_dataset:
    y_true_best_model.extend(labels.numpy())
y_true_best_model = np.array(y_true_best_model)
y_pred_prob_best_model = model_best.predict(validation_dataset).flatten()
y_pred_best_model = (y_pred_prob_best_model >= 0.5).astype(int)

cm = confusion_matrix(y_true_best_model, y_pred_best_model)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'])

plt.title('Confusion Matrix (Validation DataSet)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print("\nConfusion Matrix Metrics:")
print(f"True Negatives (TN): {cm[0,0]}")
print(f"False Positives (FP): {cm[0,1]}")
print(f"False Negatives (FN): {cm[1,0]}")
print(f"True Positives (TP): {cm[1,1]}")

## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3957785298.py in <cell line: 0>()
      6 
      7 y_true_best_model = []
----> 8 for _, labels in validation_dataset:
      9     y_true_best_model.extend(labels.numpy())
     10 y_true_best_model = np.array(y_true_best_model)

NameError: name 'validation_dataset' is not defined

## === cell 39
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix


plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(history_best_model.history['loss'], label='Training Loss')
plt.plot(history_best_model.history['val_loss'], label='Validation Loss')
plt.title(f"Training vs Validation Loss (Best Model)")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(history_best_model.history['accuracy'], label='Training Accuracy')
plt.plot(history_best_model.history['val_accuracy'], label='Validation Accuracy')
plt.title(f"Training vs Validation Accuracy (Best Model)")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(history_best_model.history['auroc'], label='Training AUC')
plt.plot(history_best_model.history['val_auroc'], label='Validation AUC')
plt.title(f"Training vs Validation AUC (Best Model)")
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3391470445.py in <cell line: 0>()
      9 plt.figure(figsize=(15, 5))
     10 plt.subplot(1, 3, 1)
---> 11 plt.plot(history_best_model.history['loss'], label='Training Loss')
     12 plt.plot(history_best_model.history['val_loss'], label='Validation Loss')
     13 plt.title(f"Training vs Validation Loss (Best Model)")

NameError: name 'history_best_model' is not defined
