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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.5480507706255666

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'The changes add a simple image cache to the `ImageDataGenerator` so each image file is read from disk at most once, dramatically reducing I/O overhead during the three training epochs.  A small memory increase (≈2–3 GB for the whole dataset) is acceptable and does not alter the model architecture, loss, or training schedule.  Minor deterministic‑seed settings are added for reproducibility, and the generator’s signature is unchanged, preserving all downstream logic.'
- What this solution (achieved 0.61099) has done: 'The updated script removes the per‑image caching (which caused massive memory use and swapping), enlarges the batch size to cut the number of training steps, and enables multi‑worker data loading in `model.fit`. These changes keep the same preprocessing, model architecture, and training regime, but dramatically reduce I/O and Python‑loop overhead so the whole run finishes well under the 600 s limit.'
- What this solution (achieved 0.61099) has done: 'The fix adds the missing imports for numpy, pandas, random, TensorFlow, and scikit‑learn utilities, and imports the Keras layers, models, and optimizer used in the model definition. With these imports the seeds, model creation, data generator, training, and prediction run without NameError, and the script now writes a proper `submission.csv` matching the required format. No core logic is altered, preserving the original modeling approach while making the pipeline executable.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
gm_exp = tf.Variable(3.0, dtype=tf.float32)


def generalized_mean_pool_2d(X):
    pool = (
        tf.reduce_mean(tf.abs(X ** (gm_exp)), axis=[1, 2], keepdims=False) + 1e-7
    ) ** (1.0 / gm_exp)
    return pool


def create_model(input_shape):
    inp = Input(shape=input_shape)
    base = ResNet50(
        weights="imagenet", include_top=False, input_tensor=inp, pooling=None
    )
    for l in base.layers:
        l.trainable = True
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    out = Dense(5, activation="softmax", name="label")(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model




## === cell 2
class ImageDataGenerator(tf.keras.utils.Sequence):
    """
    Keras Sequence that loads images on‑the‑fly with a small LRU cache.
    Caching prevents repeated disk reads across epochs while keeping memory use bounded.
    """

    def __init__(
        self,
        df,
        img_dir,
        batch_size=128,
        img_size=(224, 224, 3),
        shuffle=True,
        augment=False,
        cache_size=4000,
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.img_h, self.img_w, self.img_c = img_size
        self.shuffle = shuffle
        self.augment = augment
        self.cache = {}
        self.cache_size = cache_size
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _load_image(self, image_id):
        """Load, resize, normalize and cache a single image using TensorFlow."""
        if image_id in self.cache:
            return self.cache[image_id]

        img_path = os.path.join(self.img_dir, image_id)
        try:
            img_bytes = tf.io.read_file(img_path)
            img = tf.image.decode_jpeg(img_bytes, channels=self.img_c)
        except Exception:
            img = tf.zeros((self.img_h, self.img_w, self.img_c), dtype=tf.float32)

        img = tf.image.resize(img, (self.img_h, self.img_w))
        img = tf.cast(img, tf.float32) / 255.0
        img_np = img.numpy()

        if len(self.cache) >= self.cache_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[image_id] = img_np
        return img_np

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_idx]

        X = np.empty(
            (len(batch_df), self.img_h, self.img_w, self.img_c), dtype=np.float32
        )
        y = np.empty(len(batch_df), dtype=np.int32)

        for i, row in enumerate(batch_df.itertuples()):
            img = self._load_image(row.image_id)

            if self.augment and (i % 2 == 0):
                img = np.flip(img, axis=1)  # horizontal flip

            X[i] = img
            if hasattr(row, "label"):
                y[i] = row.label

        if "label" in self.df.columns:
            return X, y
        else:
            return X




## === cell 3
BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")

train_df = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
sample_submission = pd.read_csv(os.path.join(BASE_PATH, "sample_submission.csv"))

train_split, val_split = train_test_split(
    train_df, test_size=0.2, stratify=train_df["label"], random_state=42
)

train_gen = ImageDataGenerator(train_split, TRAIN_IMG_DIR, augment=True, batch_size=128)
val_gen = ImageDataGenerator(val_split, TRAIN_IMG_DIR, augment=False, batch_size=128)

model = create_model((224, 224, 3))
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=2,
    verbose=2,
    workers=4,
    use_multiprocessing=True,
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2139855989.py in <cell line: 0>()
     14 
     15 model = create_model((224, 224, 3))
---> 16 model.fit(
     17     train_gen,
     18     validation_data=val_gen,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    117             return fn(*args, **kwargs)
    118         except Exception as e:
--> 119             filtered_tb = _process_traceback_frames(e.__traceback__)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`

TypeError: TensorFlowTrainer.fit() got an unexpected keyword argument 'workers'

## === cell 4
test_gen = ImageDataGenerator(
    sample_submission.drop(columns=["label"]),
    TEST_IMG_DIR,
    batch_size=128,
    augment=False,
    shuffle=False,
)

preds = model.predict(test_gen, verbose=0)
pred_labels = np.argmax(preds, axis=1)
sample_submission["label"] = pred_labels
sample_submission.to_csv("submission.csv", index=False)
