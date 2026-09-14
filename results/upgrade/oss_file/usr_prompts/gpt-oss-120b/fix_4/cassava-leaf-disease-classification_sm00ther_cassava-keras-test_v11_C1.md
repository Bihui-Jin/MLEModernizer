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

albumentations==2.0.8
geopandas==0.14.4
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
pillow==11.3.0
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

0.6128

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json, cv2




## === cell 1
WORK_DIR = "/kaggle/input/cassava-leaf-disease-classification"
os.listdir(WORK_DIR)

with open(os.path.join(WORK_DIR, "label_num_to_disease_map.json"), "r") as file:
    labels = json.load(file)




## === cell 2
data = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))
rand_idx = np.random.randint(data.shape[0])
rand_fname = data.iloc[rand_idx, 0]
train_images_folder = os.path.join(WORK_DIR, "train_images")
test_images_folder = os.path.join(WORK_DIR, "test_images")
rand_fpath = os.path.join(train_images_folder, rand_fname)

Image.open(rand_fpath)




## === cell 3
from tensorflow.keras.utils import Sequence
import math


class MyGeneratorUnWeighted(Sequence):
    """Simple Keras Sequence that loads images on‑the‑fly."""

    def __init__(self, image_ids, labels, folder, cfg, shuffle=True):
        self.image_ids = np.array(image_ids)
        self.labels = np.array(labels) if labels is not None else None
        self.folder = folder
        self.batch_size = cfg["batch_size"]
        self.img_h = cfg["img_height"]
        self.img_w = cfg["img_width"]
        self.shuffle = shuffle
        self.indices = np.arange(len(self.image_ids))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.image_ids) / self.batch_size)

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_files = self.image_ids[batch_idx]

        batch_x = np.empty(
            (len(batch_files), self.img_h, self.img_w, 3), dtype=np.float32
        )
        for i, fname in enumerate(batch_files):
            fpath = os.path.join(self.folder, fname)
            img = cv2.imread(fpath)  # BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
            img = cv2.resize(img, (self.img_w, self.img_h))
            batch_x[i] = img / 255.0  # normalise

        if self.labels is not None:
            batch_y = self.labels[batch_idx]
            return batch_x, batch_y
        else:
            return batch_x

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    data.image_id, data.label, test_size=0.25, stratify=data.label, random_state=42
)




## === cell 5
net_cfg = {
    "num_classes": 5,
    "img_height": 128,  # smaller size for faster training
    "img_width": 128,
    "img_channels": 3,
    "batch_size": 32,
    "backbone": None,
    "preproc_img": True,
    "augment_img": True,
    "num_epochs": 3,
    "class_weights": None,
    "num_folds": 2,
    "backbone_trainable": True,
    "use_dropout": True,
    "activation": None,
    "opt": None,
    "criterion": None,
    "metrics": None,
}




## === cell 6
import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers, losses, metrics


class CassavaNet:
    """Tiny wrapper that builds, trains and predicts with a MobileNetV2 backbone."""

    def __init__(self, cfg):
        self.cfg = cfg
        input_shape = (cfg["img_height"], cfg["img_width"], cfg["img_channels"])
        base_model = applications.MobileNetV2(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
        base_model.trainable = cfg["backbone_trainable"]

        x = layers.GlobalAveragePooling2D()(base_model.output)
        if cfg["use_dropout"]:
            x = layers.Dropout(0.2)(x)
        output = layers.Dense(cfg["num_classes"], activation="softmax")(x)

        self.model = models.Model(inputs=base_model.input, outputs=output)
        self.model.compile(
            optimizer=optimizers.Adam(),
            loss=losses.SparseCategoricalCrossentropy(),
            metrics=[metrics.SparseCategoricalAccuracy(name="accuracy")],
        )

    def train(self, train_gen, val_gen):
        self.model.fit(
            train_gen, validation_data=val_gen, epochs=self.cfg["num_epochs"], verbose=2
        )

    def predict(self, test_gen):
        probs = self.model.predict(test_gen, verbose=0)
        return np.argmax(probs, axis=1)




## === cell 7
train_gen = MyGeneratorUnWeighted(
    x_train, y_train, train_images_folder, net_cfg, shuffle=True
)
val_gen = MyGeneratorUnWeighted(
    x_test, y_test, train_images_folder, net_cfg, shuffle=False
)

model_wrapper = CassavaNet(net_cfg)
model_wrapper.train(train_gen, val_gen)

test_files = sorted(os.listdir(test_images_folder))
test_gen = MyGeneratorUnWeighted(
    test_files, None, test_images_folder, net_cfg, shuffle=False
)

test_preds = model_wrapper.predict(test_gen)

submission_path = "submission.csv"
with open(submission_path, "w") as f:
    f.write("image_id,label\n")
    for fname, pred in zip(test_files, test_preds):
        f.write(f"{fname},{int(pred)}\n")

print(f"Submission file written to {submission_path} with {len(test_files)} records.")
print("First few lines of the submission:")
print(pd.read_csv(submission_path).head())

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
UnknownError                              Traceback (most recent call last)
/tmp/ipykernel_55/3667054602.py in <cell line: 0>()
     17 )
     18 
---> 19 test_preds = model_wrapper.predict(test_gen)
     20 
     21 # Write submission

/tmp/ipykernel_55/2037686574.py in predict(self, test_gen)
     32 
     33     def predict(self, test_gen):
---> 34         probs = self.model.predict(test_gen, verbose=0)
     35         return np.argmax(probs, axis=1)
     36 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

UnknownError: {{function_node __wrapped__IteratorGetNext_output_types_1_device_/job:localhost/replica:0/task:0/device:CPU:0}} error: OpenCV(4.12.0) /io/opencv/modules/imgproc/src/color.cpp:199: error: (-215:Assertion failed) !_src.empty() in function 'cvtColor'

Traceback (most recent call last):

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 269, in __call__
    ret = func(*args)
          ^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py", line 643, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/from_generator_op.py", line 198, in generator_py_func
    values = next(generator_state.get_iterator(iterator_id))
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/keras/src/trainers/data_adapters/py_dataset_adapter.py", line 248, in _finite_generator
    yield self._standardize_batch(self.py_dataset[i])
                                  ~~~~~~~~~~~~~~~^^^

  File "/tmp/ipykernel_55/1800943146.py", line 32, in __getitem__
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

cv2.error: OpenCV(4.12.0) /io/opencv/modules/imgproc/src/color.cpp:199: error: (-215:Assertion failed) !_src.empty() in function 'cvtColor'



	 [[{{node PyFunc}}]] [Op:IteratorGetNext] name:
