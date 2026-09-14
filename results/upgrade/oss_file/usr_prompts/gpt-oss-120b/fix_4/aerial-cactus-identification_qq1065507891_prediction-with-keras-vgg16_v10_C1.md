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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
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
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 3 other files
        input/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 3 other files
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 3 other files
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
Here is some information about the columns:
has_cactus (float64) has 1 unique values: [0.5]
id (object) has 2000 unique values. Some example values: ['09034a34de0e2015a8a28dfe18f423f6.jpg', '44b974fd955c4cd306c7dbae154646b9.jpg', '37cd593f40ba13982e7587a613a9a789.jpg', 'c892c865a5706d3072ba44beafbf2d1c.jpg']

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
Here is some information about the columns:
has_cactus (int64) has 2 unique values: [1, 0]
id (object) has 2000 unique values. Some example values: ['2de8f189f1dce439766637e75df0ee27.jpg', 'f94b18b4c6850fc44b54f5fbe3c5b0c6.jpg', 'fb0624c0ebc01643a8f4318537099078.jpg', '00b4dfbb267109b5f0d0dde365fa6161.jpg']

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
Here is some information about the columns:
has_cactus (float64) has 1 unique values: [0.5]
id (object) has 2000 unique values. Some example values: ['09034a34de0e2015a8a28dfe18f423f6.jpg', '44b974fd955c4cd306c7dbae154646b9.jpg', '37cd593f40ba13982e7587a613a9a789.jpg', 'c892c865a5706d3072ba44beafbf2d1c.jpg']

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
Here is some information about the columns:
has_cactus (int64) has 2 unique values: [1, 0]
id (object) has 2000 unique values. Some example values: ['2de8f189f1dce439766637e75df0ee27.jpg', 'f94b18b4c6850fc44b54f5fbe3c5b0c6.jpg', 'fb0624c0ebc01643a8f4318537099078.jpg', '00b4dfbb267109b5f0d0dde365fa6161.jpg']

-> working/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
Here is some information about the columns:
has_cactus (float64) has 1 unique values: [0.5]
id (object) has 2000 unique values. Some example values: ['09034a34de0e2015a8a28dfe18f423f6.jpg', '44b974fd955c4cd306c7dbae154646b9.jpg', '37cd593f40ba13982e7587a613a9a789.jpg', 'c892c865a5706d3072ba44beafbf2d1c.jpg']

-> working/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
Here is some information about the columns:
has_cactus (int64) has 2 unique values: [1, 0]
id (object) has 2000 unique values. Some example values: ['2de8f189f1dce439766637e75df0ee27.jpg', 'f94b18b4c6850fc44b54f5fbe3c5b0c6.jpg', 'fb0624c0ebc01643a8f4318537099078.jpg', '00b4dfbb267109b5f0d0dde365fa6161.jpg']

# 5. Target score

0.9926

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Flatten, Dropout, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import callbacks

print("Input root contents:", os.listdir("./input"))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_PATH = "./input/aerial-cactus-identification"
TRAIN_DIR = os.path.join(BASE_PATH, "train")
TEST_DIR = os.path.join(BASE_PATH, "test")
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")


def process_picture():
    data = pd.read_csv(TRAIN_CSV)
    image_files = []
    labels = []
    for img_id in data["id"].values:
        label = data.loc[data["id"] == img_id, "has_cactus"].values[0]
        labels.append(label)
        img_path = os.path.join(TRAIN_DIR, img_id)
        image_files.append(img_path)
    return image_files, labels


def _read_image(path):
    """Read image in BGR color, resize to 32x32, ensure 3 channels."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # force color read
    if img is None:
        img = np.zeros((32, 32, 3), dtype=np.uint8)
    else:
        img = cv2.resize(img, (32, 32))
    return img


def get_images_labels():
    image_files, labels = process_picture()
    images = [_read_image(f) for f in image_files]
    train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(
        images,
        labels,
        test_size=0.2,
        random_state=7,
        shuffle=True,
        stratify=labels,
    )
    train_imgs = np.array(train_imgs, dtype=np.float32) / 255.0
    val_imgs = np.array(val_imgs, dtype=np.float32) / 255.0
    print("Train shape:", train_imgs.shape)
    return train_imgs, val_imgs, np.array(train_lbls), np.array(val_lbls)




## === cell 2
train_images, test_images, train_labels_int, test_labels_int = get_images_labels()

class_weight_arr = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels_int),
    y=train_labels_int,
)
class_weight = dict(enumerate(class_weight_arr))

train_labels = to_categorical(train_labels_int, 2)
test_labels = to_categorical(test_labels_int, 2)

print("Class weights:", class_weight)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2933149350.py in <cell line: 0>()
      1 # Load data and prepare labels / class weights
----> 2 train_images, test_images, train_labels_int, test_labels_int = get_images_labels()
      3 
      4 class_weight_arr = compute_class_weight(
      5     class_weight="balanced",

/tmp/ipykernel_11/4101619013.py in get_images_labels()
     29 
     30 def get_images_labels():
---> 31     image_files, labels = process_picture()
     32     images = [_read_image(f) for f in image_files]
     33     train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(

/tmp/ipykernel_11/4101619013.py in process_picture()
      7 
      8 def process_picture():
----> 9     data = pd.read_csv(TRAIN_CSV)
     10     image_files = []
     11     labels = []

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

FileNotFoundError: [Errno 2] No such file or directory: './input/aerial-cactus-identification/train.csv'

## === cell 3
def build_model():
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(32, 32, 3))
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers:
        if "block4" in layer.name or "block5" in layer.name:
            layer.trainable = True

    add_model = Sequential()
    add_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    add_model.add(BatchNormalization())
    add_model.add(Dense(256, activation="relu", name="FC1"))
    add_model.add(BatchNormalization())
    add_model.add(Dropout(0.5))
    add_model.add(Dense(128, activation="relu", name="FC2"))
    add_model.add(BatchNormalization())
    add_model.add(Dense(2, activation="softmax", name="softmax"))

    model = Model(inputs=base_model.input, outputs=add_model(base_model.output))
    model.summary()
    return model


def train(batch_size=64, nb_epoch=50):
    model = build_model()
    optimizer = Adam(1e-5)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    early_stop = callbacks.EarlyStopping(
        monitor="val_accuracy", patience=10, mode="auto", restore_best_weights=True
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=5, mode="auto"
    )
    cb_list = [early_stop, reduce_lr]

    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=[0.9, 1.5],
        vertical_flip=True,
        horizontal_flip=True,
    )
    train_datagen.fit(train_images)

    history = model.fit(
        train_datagen.flow(train_images, train_labels, batch_size=batch_size),
        steps_per_epoch=train_images.shape[0] // batch_size,
        epochs=nb_epoch,
        validation_data=(test_images, test_labels),
        class_weight=class_weight,
        callbacks=cb_list,
        verbose=2,
    )
    score = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Validation accuracy: {score[1]*100:.2f}%")
    model.save("./test.h5")
    return model, history




## === cell 4
model, history = train()




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1474431857.py in <cell line: 0>()
      1 # Train the model
----> 2 model, history = train()
      3 
      4 

/tmp/ipykernel_11/3209007214.py in train(batch_size, nb_epoch)
     48         horizontal_flip=True,
     49     )
---> 50     train_datagen.fit(train_images)
     51 
     52     history = model.fit(

NameError: name 'train_images' is not defined

## === cell 5
def get_test_images():
    imgs = []
    ids = []
    for fname in os.listdir(TEST_DIR):
        ids.append(fname)
        fpath = os.path.join(TEST_DIR, fname)
        img = cv2.imread(fpath, cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((32, 32, 3), dtype=np.uint8)
        else:
            img = cv2.resize(img, (32, 32))
        imgs.append(img)
    imgs = np.asarray(imgs, dtype=np.float32) / 255.0
    print("Test images shape:", imgs.shape)
    return imgs, ids




## === cell 6
def predict_and_submit(model, out_path="submission.csv"):
    images, ids = get_test_images()
    probs = model.predict(images, verbose=0)[:, 1]  # probability of class 1
    sub_df = pd.DataFrame({"id": ids, "has_cactus": probs})
    sub_df.to_csv(out_path, index=False)
    print(f"Submission written to {out_path}")




## === cell 7
predict_and_submit(model)

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2233695693.py in <cell line: 0>()
      1 # Generate submission
----> 2 predict_and_submit(model)

NameError: name 'model' is not defined
