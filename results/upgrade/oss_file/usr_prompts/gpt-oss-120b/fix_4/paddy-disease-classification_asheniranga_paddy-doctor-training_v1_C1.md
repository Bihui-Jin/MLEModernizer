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
Develop a model to classify paddy leaf images into one of the nine disease categories or normal leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
200001.jpg,normal
200002.jpg,blast
etc.
```

## Dataset
**train.csv** - The training set

- `image_id` - Unique image identifier corresponds to image file names (.jpg) found in the train_images directory.
- `label` - Type of paddy disease, also the target class. There are ten categories, including the normal leaf.
- `variety` - The name of the paddy variety.
- `age` - Age of the paddy in days.

**sample_submission.csv** - Sample submission file.

**train_images** - Training images stored under different sub-directories corresponding to ten target classes. Filename corresponds to the `image_id` column of `train.csv`.

**test_images** - Test set images.

# 2. Python version

3.10

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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        input/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        working/
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
```

-> data/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> data/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> input/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> input/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> (stopped after 10 files for performance)

# 5. Target score

0.87096

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.02998) has done: 'The fixes address the protobuf import issue, correct iterator usage, match loss weights to multiple outputs, use a proper filename for saving weights, and properly load and predict on test images while skipping sub‑directories. These changes unblock the pipeline, produce a valid CSV submission, and keep the model architecture untouched.'
- What this solution (achieved 0.0) has done: 'I fixed the mismatch between the model’s three‑output architecture and the single‑output labels supplied by the ImageDataGenerator. The model now returns only the main Inception output, compiled with a single loss/metric, so training and evaluation work correctly. I also updated the history‑plotting and test‑prediction code to use the new single‑output naming, ensuring a proper CSV submission is written.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    from google.protobuf.message_factory import MessageFactory

    if not hasattr(MessageFactory, "GetPrototype"):

        def _GetPrototype(self, descriptor):
            return self.GetMessageClass(descriptor)

        MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    GlobalAveragePooling2D,
)
from tensorflow.keras.layers import BatchNormalization, concatenate, Dropout, Flatten
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array




## === cell 1
train_meta_data = "../train.csv"
train_data_dir = "../input/paddy-disease-classification/train_images"
epochs = 100
lr = 1e-3
valid_split = 0.2
input_size = 224
batch_size = 32
classes = 10
initializer = tf.keras.initializers.HeUniform()
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
loss = tf.keras.losses.CategoricalCrossentropy()




## === cell 2
early_stop = tf.keras.callbacks.EarlyStopping(
    patience=10, monitor="val_loss", restore_best_weights=True, verbose=1
)




## === cell 3
def inception(x, filters, projection, init=initializer, name=None):
    f_1x1, f_3x3, f_3x3_reduce, f_5x5, f_5x5_reduce = filters
    x1 = Conv2D(
        filters=f_1x1,
        kernel_size=(1, 1),
        kernel_initializer=init,
        activation=relu,
        padding="same",
    )(x)
    x3_reducer = Conv2D(
        filters=f_3x3_reduce,
        kernel_size=(1, 1),
        kernel_initializer=init,
        activation=relu,
        padding="same",
    )(x)
    x5_reducer = Conv2D(
        filters=f_5x5_reduce,
        kernel_size=(1, 1),
        kernel_initializer=init,
        activation=relu,
        padding="same",
    )(x)
    pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(x)

    x3 = Conv2D(
        filters=f_3x3,
        kernel_size=(3, 3),
        kernel_initializer=init,
        activation=relu,
        padding="same",
    )(x3_reducer)
    x5 = Conv2D(
        filters=f_5x5,
        kernel_size=(5, 5),
        kernel_initializer=init,
        activation=relu,
        padding="same",
    )(x5_reducer)
    proj = Conv2D(
        filters=projection,
        kernel_size=(1, 1),
        kernel_initializer=init,
        activation=relu,
        padding="same",
    )(pool)

    return concatenate([x1, x3, x5, proj], axis=3, name=name)


def model_builder(shape, classes):
    input_layer = Input(shape=shape)
    x = Conv2D(
        64,
        (7, 7),
        strides=(2, 2),
        activation=relu,
        padding="same",
        kernel_initializer=initializer,
    )(input_layer)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (1, 1), activation=relu, padding="same")(x)
    x = Conv2D(192, (3, 3), activation=relu, padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    x = inception(x, [64, 128, 96, 32, 16], projection=32, name="inception_3a")
    x = inception(x, [128, 192, 128, 96, 32], projection=64, name="inception_3b")
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    x = inception(x, [192, 208, 96, 48, 16], projection=64, name="inception_4a")

    aux_1 = AveragePooling2D((5, 5), strides=(3, 3), padding="valid")(x)
    aux_1 = Conv2D(
        128, (1, 1), kernel_initializer=initializer, activation=relu, padding="valid"
    )(aux_1)
    aux_1 = Dense(1024, activation=relu)(aux_1)
    aux_1 = Dropout(0.7)(aux_1)
    aux_1 = GlobalAveragePooling2D()(aux_1)

    x = inception(x, [160, 224, 112, 64, 24], projection=64, name="inception_4b")
    x = inception(x, [128, 256, 128, 64, 24], projection=64, name="inception_4c")
    x = inception(x, [112, 288, 144, 64, 32], projection=64, name="inception_4d")
    x = inception(x, [256, 320, 160, 128, 32], projection=128, name="inception_4e")

    aux_2 = AveragePooling2D((5, 5), strides=(3, 3), padding="valid")(x)
    aux_2 = Conv2D(
        128, (1, 1), kernel_initializer=initializer, activation=relu, padding="valid"
    )(aux_2)
    aux_2 = Dense(1024, activation=relu)(aux_2)
    aux_2 = Dropout(0.7)(aux_2)
    aux_2 = GlobalAveragePooling2D()(aux_2)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    x = inception(x, [256, 320, 160, 128, 32], projection=128, name="inception_5a")
    x = inception(x, [384, 384, 192, 128, 48], projection=128, name="inception_5b")
    x = AveragePooling2D((7, 7), strides=(1, 1))(x)
    x = Dropout(0.4)(x)
    x = GlobalAveragePooling2D()(x)
    main_out = Dense(classes, activation=softmax, name="main_out")(x)

    model = Model(inputs=input_layer, outputs=main_out)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"],
    )
    return model




## === cell 4
class_names = sorted(pd.read_csv(train_meta_data)["label"].unique())

generator = ImageDataGenerator(rescale=1 / 255, validation_split=valid_split)

train_data = generator.flow_from_directory(
    directory=train_data_dir,
    target_size=(input_size, input_size),
    batch_size=batch_size,
    subset="training",
    class_mode="categorical",
    classes=class_names,
    shuffle=True,
)

valid_data = generator.flow_from_directory(
    directory=train_data_dir,
    target_size=(input_size, input_size),
    batch_size=batch_size,
    subset="validation",
    class_mode="categorical",
    classes=class_names,
    shuffle=False,
)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/3501949672.py in <cell line: 0>()
      1 # Create a deterministic list of class names from the CSV to avoid stray folders
----> 2 class_names = sorted(pd.read_csv(train_meta_data)["label"].unique())
      3 
      4 generator = ImageDataGenerator(rescale=1 / 255, validation_split=valid_split)
      5 

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

FileNotFoundError: [Errno 2] No such file or directory: '../train.csv'

## === cell 5
train_batch = next(train_data)
valid_batch = next(valid_data)
len(train_batch[0]), len(valid_batch[0])




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4136272988.py in <cell line: 0>()
----> 1 train_batch = next(train_data)
      2 valid_batch = next(valid_data)
      3 len(train_batch[0]), len(valid_batch[0])
      4 
      5 

NameError: name 'train_data' is not defined

## === cell 6
model = model_builder(shape=(input_size, input_size, 3), classes=classes)




## === cell 7
model.summary()




## === cell 8
tf.keras.utils.plot_model(model, "baseline_inception.png")




## === cell 9
history = model.fit(
    train_data, validation_data=valid_data, epochs=epochs, callbacks=[early_stop]
)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1463280320.py in <cell line: 0>()
      1 history = model.fit(
----> 2     train_data, validation_data=valid_data, epochs=epochs, callbacks=[early_stop]
      3 )
      4 
      5 

NameError: name 'train_data' is not defined

## === cell 10
plt.figure(figsize=[12, 6], dpi=300)
sns.lineplot(
    x=range(len(history.history["accuracy"])),
    y=history.history["accuracy"],
    label="train",
)
sns.lineplot(
    x=range(len(history.history["val_accuracy"])),
    y=history.history["val_accuracy"],
    label="validation",
)
plt.show()




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/456603417.py in <cell line: 0>()
      1 plt.figure(figsize=[12, 6], dpi=300)
      2 sns.lineplot(
----> 3     x=range(len(history.history["accuracy"])),
      4     y=history.history["accuracy"],
      5     label="train",

NameError: name 'history' is not defined

## === cell 11
plt.figure(figsize=[12, 6], dpi=300)
sns.lineplot(
    x=range(len(history.history["loss"])),
    y=history.history["loss"],
    label="train",
)
sns.lineplot(
    x=range(len(history.history["val_loss"])),
    y=history.history["val_loss"],
    label="validation",
)
plt.show()




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/152515602.py in <cell line: 0>()
      1 plt.figure(figsize=[12, 6], dpi=300)
      2 sns.lineplot(
----> 3     x=range(len(history.history["loss"])),
      4     y=history.history["loss"],
      5     label="train",

NameError: name 'history' is not defined

## === cell 12
print(
    f"train score : {model.evaluate(train_data)} -- validation : {model.evaluate(valid_data)}"
)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2111908320.py in <cell line: 0>()
      1 print(
----> 2     f"train score : {model.evaluate(train_data)} -- validation : {model.evaluate(valid_data)}"
      3 )
      4 
      5 

NameError: name 'train_data' is not defined

## === cell 13
pd.DataFrame(history.history).to_csv("history.csv", index=False)




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1445596701.py in <cell line: 0>()
----> 1 pd.DataFrame(history.history).to_csv("history.csv", index=False)
      2 
      3 

NameError: name 'history' is not defined

## === cell 14
model.save("baseline.hdf5")




## === cell 15
model.save_weights("baseline_inception_weights.weights.h5")




## === cell 16
train_data.class_indices  # just to expose mapping if needed




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2499039068.py in <cell line: 0>()
----> 1 train_data.class_indices  # just to expose mapping if needed
      2 
      3 

NameError: name 'train_data' is not defined

## === cell 17
test_dir = "../input/paddy-disease-classification/test_images"
test_files = [
    f
    for f in os.listdir(test_dir)
    if f.lower().endswith(".jpg") and not os.path.isdir(os.path.join(test_dir, f))
]
test_preds = []

for i, file in enumerate(sorted(test_files)):
    img_path = os.path.join(test_dir, file)
    img = load_img(img_path, target_size=(input_size, input_size))
    img_arr = np.expand_dims(img_to_array(img) / 255.0, axis=0)
    pred = model.predict(img_arr, verbose=0)  # shape (1, classes)
    class_idx = np.argmax(pred, axis=1)[0]
    label = list(train_data.class_indices.keys())[
        list(train_data.class_indices.values()).index(class_idx)
    ]
    test_preds.append([file, label])
    print(f"{i+1}/{len(test_files)}", end="\r")




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4198839664.py in <cell line: 0>()
     13     pred = model.predict(img_arr, verbose=0)  # shape (1, classes)
     14     class_idx = np.argmax(pred, axis=1)[0]
---> 15     label = list(train_data.class_indices.keys())[
     16         list(train_data.class_indices.values()).index(class_idx)
     17     ]

NameError: name 'train_data' is not defined

## === cell 18
submission = pd.DataFrame(test_preds, columns=["image_id", "label"])
submission.to_csv("baseline_submission.csv", index=False)
submission

## --- ERROR in outputing the csv:
Invalid submission: Expected 2602 rows, but got 0
