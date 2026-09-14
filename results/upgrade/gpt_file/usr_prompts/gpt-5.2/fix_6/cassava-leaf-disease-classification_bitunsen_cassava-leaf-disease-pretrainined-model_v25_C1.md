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

0.8862194016319129

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from PIL import Image

import tensorflow as tf

tf.keras.backend.clear_session()
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

print("TF version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

assert os.path.isdir(TRAIN_DIR), f"Missing TRAIN_DIR: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing TEST_DIR: {TEST_DIR}"



## === cell 2
with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as file:
    map_classes = json.loads(file.read())

print(json.dumps(map_classes, indent=2))

label_list = [int(key) for key in map_classes.keys()]
NUM_CLASSES = len(label_list)
print("NUM_CLASSES:", NUM_CLASSES)



## === cell 3
input_files = os.listdir(os.path.join(BASE_DIR, "train_images"))
print(f"Number of train images: {len(input_files)}")



## === cell 4
IMG_HEIGHT = 400
IMG_WIDTH = 400
batch_size = 32

PRE_TRAINED_MODEL = (
    "../input/inceptionresnetv2/Cassava_Best_InceptionResNet_Model_V02.hdf5"
)

try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE = Image.LANCZOS



## === cell 5
try:
    from albumentations import (
        Compose,
        HorizontalFlip,
        VerticalFlip,
        CenterCrop,
        RandomBrightnessContrast,
        ToFloat,
        ShiftScaleRotate,
    )

    AUGMENTATIONS_TRAIN = Compose(
        [
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            CenterCrop(p=1.0, height=IMG_HEIGHT, width=IMG_WIDTH),
            ShiftScaleRotate(
                p=0.5,
                shift_limit=0,
                scale_limit=(0.5, 1.50),
                rotate_limit=15,
                interpolation=0,
                border_mode=0,
            ),
            ToFloat(max_value=255),
        ]
    )
    AUGMENTATIONS_TEST = Compose([ToFloat(max_value=255)])
    _HAS_ALB = True
except Exception as e:
    print(
        "albumentations not available; using no-op float scaling augmentations. Error:",
        repr(e),
    )

    class _NoOp:
        def __call__(self, image):
            return {"image": image.astype(np.float32) / 255.0}

    AUGMENTATIONS_TRAIN = _NoOp()
    AUGMENTATIONS_TEST = _NoOp()
    _HAS_ALB = False



## === cell 6
from tensorflow.keras.utils import Sequence


def _tf_load_and_resize_uint8(image_path):
    img_bytes = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)  # uint8
    img = tf.image.resize(
        img, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.clip_by_value(tf.round(img), 0.0, 255.0)
    return tf.cast(img, tf.uint8).numpy()


def load_single_image(data_type, image_id):
    if data_type == "TEST_DATA":
        image_path = os.path.join(TEST_DIR, image_id)
    else:
        image_path = os.path.join(TRAIN_DIR, image_id)
    return _tf_load_and_resize_uint8(image_path)


class AugmentedImageSequence(Sequence):
    def __init__(self, mode, data_set_type, x_set, y_set, batch_size, augmentations):
        self.mode = mode
        self.data_type = data_set_type
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.augment = augmentations

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size : (idx + 1) * self.batch_size]

        if self.mode == "TEST":
            batch_y = []
        else:
            batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]

        img_list = []
        for x in batch_x:
            if self.data_type == "TRAIN_DATA":
                random_num = random.uniform(0, 1)
                if random_num > 0.5:
                    img_data = self.augment(image=load_single_image(self.data_type, x))[
                        "image"
                    ]
                else:
                    img_data = load_single_image(self.data_type, x)
            else:
                if self.data_type == "VALIDATE_DATA":
                    img_data = self.augment(image=load_single_image(self.data_type, x))[
                        "image"
                    ]
                else:
                    img_data = load_single_image(self.data_type, x)

            img_list.append(img_data)

        img_array = np.stack(img_list, axis=0).astype(np.float32)

        return img_array, np.array(batch_y)




## === cell 7
sample_sub_path = os.path.join(BASE_DIR, "sample_submission.csv")
sample_sub = pd.read_csv(sample_sub_path)

test_df = sample_sub[["image_id"]].copy()
test_samples = test_df.shape[0]
print("Test samples:", test_samples)

test_gen = AugmentedImageSequence(
    mode="TEST",
    data_set_type="TEST_DATA",
    x_set=test_df["image_id"].values,
    y_set=None,
    batch_size=batch_size,
    augmentations=AUGMENTATIONS_TEST,
)




## === cell 8
def build_inceptionresnetv2_model():
    base = tf.keras.applications.InceptionResNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        pooling="avg",
    )
    x = base.output
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = tf.keras.Model(inputs=base.input, outputs=out)
    return model


def get_model():
    if os.path.exists(PRE_TRAINED_MODEL):
        model = tf.keras.models.load_model(PRE_TRAINED_MODEL)
        return model, "loaded_pretrained"
    else:
        model = build_inceptionresnetv2_model()
        return model, "trained_fallback"


model, model_mode = get_model()
print("Model mode:", model_mode)



## === cell 9
if model_mode == "trained_fallback":
    train_csv_path = os.path.join(BASE_DIR, "train.csv")
    train_df = pd.read_csv(train_csv_path)

    train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    val_frac = 0.2
    n_val = int(len(train_df) * val_frac)

    val_df = train_df.iloc[:n_val].reset_index(drop=True)
    trn_df = train_df.iloc[n_val:].reset_index(drop=True)

    trn_gen = AugmentedImageSequence(
        mode="TRAIN",
        data_set_type="TRAIN_DATA",
        x_set=trn_df["image_id"].values,
        y_set=trn_df["label"].values,
        batch_size=batch_size,
        augmentations=AUGMENTATIONS_TRAIN,
    )
    val_gen = AugmentedImageSequence(
        mode="VALIDATE",
        data_set_type="VALIDATE_DATA",
        x_set=val_df["image_id"].values,
        y_set=val_df["label"].values,
        batch_size=batch_size,
        augmentations=AUGMENTATIONS_TEST,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        trn_gen,
        validation_data=val_gen,
        epochs=2,
        verbose=1,
        workers=min(8, (os.cpu_count() or 2)),
        use_multiprocessing=True,
        max_queue_size=16,
    )




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/556146044.py in <cell line: 0>()
     35     # PERF: Enable multi-worker data loading for Sequence to overlap preprocessing with training.
     36     # Correctness: identical batches/labels; only parallelism changes.
---> 37     model.fit(
     38         trn_gen,
     39         validation_data=val_gen,

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

## === cell 10
class TTATestSequence(Sequence):
    def __init__(self, image_ids, tta, batch_size):
        self.image_ids = np.asarray(image_ids)
        self.tta = int(tta)
        self.batch_size = int(batch_size)
        self.n = len(self.image_ids) * self.tta

        self._cache = {}

    def __len__(self):
        return int(np.ceil(self.n / float(self.batch_size)))

    def _get_img_uint8(self, image_id):
        img = self._cache.get(image_id)
        if img is None:
            img = load_single_image("TEST_DATA", image_id)  # uint8
            self._cache[image_id] = img
        return img

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, self.n)
        flat = np.arange(start, end, dtype=np.int64)
        img_idx = flat // self.tta
        tta_idx = flat % self.tta

        out = np.empty((end - start, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
        for j, (ii, tt) in enumerate(zip(img_idx, tta_idx)):
            image_id = self.image_ids[int(ii)]
            img = self._get_img_uint8(image_id)
            if int(tt) == 0:
                out[j] = AUGMENTATIONS_TEST(image=img)["image"]
            else:
                out[j] = AUGMENTATIONS_TRAIN(image=img)["image"]
        return out


tta = 6
tta_seq = TTATestSequence(test_df["image_id"].values, tta=tta, batch_size=64)

preds_all = model.predict(
    tta_seq,
    verbose=0,
    workers=min(8, (os.cpu_count() or 2)),
    use_multiprocessing=True,
    max_queue_size=16,
)

preds_all = preds_all.reshape(len(test_df), tta, NUM_CLASSES)
mean_preds = preds_all.mean(axis=1)
labels = np.argmax(mean_preds, axis=1).astype(int)

test_results_df = pd.DataFrame(
    {"image_id": test_df["image_id"].values, "label": labels}
)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2076341278.py in <cell line: 0>()
     43 # PERF: Multi-worker prefetching for prediction to overlap CPU preprocessing with model inference.
     44 # Correctness: predictions unchanged; only input pipeline parallelism changes.
---> 45 preds_all = model.predict(
     46     tta_seq,
     47     verbose=0,

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

TypeError: TensorFlowTrainer.predict() got an unexpected keyword argument 'workers'

## === cell 11
submission = sample_sub[["image_id"]].merge(test_results_df, on="image_id", how="left")
assert submission["label"].isna().sum() == 0, "Some test images missing predictions"

submission["label"] = submission["label"].astype(int)
submission.to_csv("submission.csv", index=False)

print(submission.head(3))
print("Wrote submission.csv with shape:", submission.shape)
print("File exists:", os.path.exists("submission.csv"))

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1472394949.py in <cell line: 0>()
----> 1 submission = sample_sub[["image_id"]].merge(test_results_df, on="image_id", how="left")
      2 assert submission["label"].isna().sum() == 0, "Some test images missing predictions"
      3 
      4 submission["label"] = submission["label"].astype(int)
      5 submission.to_csv("submission.csv", index=False)

NameError: name 'test_results_df' is not defined
