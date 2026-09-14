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

0.8806285886974917

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, gc, random, math, re
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


print("TensorFlow version:", tf.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
class FixedDropout(tf.keras.layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        symbolic_shape = K.shape(inputs)
        noise_shape = [
            symbolic_shape[axis] if shape is None else shape
            for axis, shape in enumerate(self.noise_shape)
        ]
        return tuple(noise_shape)




## === cell 2

IMG_SIZE = (512, 512)
N_CLASSES = 5


def build_densenet201():
    base = tf.keras.applications.DenseNet201(
        include_top=True, weights="imagenet", input_shape=(*IMG_SIZE, 3), classes=1000
    )
    x = base.output
    x = tf.keras.layers.Dense(N_CLASSES, activation="softmax", name="cassava_head")(x)
    model = tf.keras.Model(
        inputs=base.input, outputs=x, name="densenet201_imagenet_to_5"
    )
    return model


def build_inceptionv3():
    base = tf.keras.applications.InceptionV3(
        include_top=True, weights="imagenet", input_shape=(*IMG_SIZE, 3), classes=1000
    )
    x = base.output
    x = tf.keras.layers.Dense(N_CLASSES, activation="softmax", name="cassava_head")(x)
    model = tf.keras.Model(
        inputs=base.input, outputs=x, name="inceptionv3_imagenet_to_5"
    )
    return model


def build_efficientnetb3():
    base = tf.keras.applications.EfficientNetB3(
        include_top=True, weights="imagenet", input_shape=(*IMG_SIZE, 3), classes=1000
    )
    x = base.output
    x = tf.keras.layers.Dense(N_CLASSES, activation="softmax", name="cassava_head")(x)
    model = tf.keras.Model(
        inputs=base.input, outputs=x, name="efficientnetb3_imagenet_to_5"
    )
    return model


dense201 = build_densenet201()
inception = build_inceptionv3()
efficient_net = build_efficientnetb3()



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/320295791.py in <cell line: 0>()
     45 
     46 
---> 47 dense201 = build_densenet201()
     48 inception = build_inceptionv3()
     49 efficient_net = build_efficientnetb3()

/tmp/ipykernel_11/320295791.py in build_densenet201()
      8 
      9 def build_densenet201():
---> 10     base = tf.keras.applications.DenseNet201(
     11         include_top=True, weights="imagenet", input_shape=(*IMG_SIZE, 3), classes=1000
     12     )

/usr/local/lib/python3.11/dist-packages/keras/src/applications/densenet.py in DenseNet201(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)
    400 ):
    401     """Instantiates the Densenet201 architecture."""
--> 402     return DenseNet(
    403         [6, 12, 48, 32],
    404         include_top,

/usr/local/lib/python3.11/dist-packages/keras/src/applications/densenet.py in DenseNet(blocks, include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)
    204 
    205     # Determine proper input shape
--> 206     input_shape = imagenet_utils.obtain_input_shape(
    207         input_shape,
    208         default_size=224,

/usr/local/lib/python3.11/dist-packages/keras/src/applications/imagenet_utils.py in obtain_input_shape(input_shape, default_size, min_size, data_format, require_flatten, weights)
    347         if input_shape is not None:
    348             if input_shape != default_shape:
--> 349                 raise ValueError(
    350                     "When setting `include_top=True` "
    351                     "and loading `imagenet` weights, "

ValueError: When setting `include_top=True` and loading `imagenet` weights, `input_shape` should be (224, 224, 3).  Received: input_shape=(512, 512, 3)

## === cell 3
JPEG_PATH = "../input/cassava-leaf-disease-classification/test_images"

import cv2


def load_image(jpeg_path, image_id):
    img = cv2.imread(os.path.join(jpeg_path, image_id))
    if img is None:
        raise FileNotFoundError(
            f"Could not read image: {os.path.join(jpeg_path, image_id)}"
        )
    img = img.astype(np.float32) / 255.0
    img = cv2.resize(img, IMG_SIZE)[:, :, ::-1]  # BGR->RGB
    return img


def generator(filepath, paths, batch_size=32):
    i = 0
    n = len(paths)
    while i < n:
        batch_paths = paths[i : i + batch_size]
        batch = [load_image(filepath, p) for p in batch_paths]
        i += batch_size
        yield np.stack(batch, axis=0)




## === cell 4
submission = pd.read_csv(
    "../input/cassava-leaf-disease-classification/sample_submission.csv"
)

test_image_ids = submission["image_id"].values




## === cell 5
def vote_in_ensemble(v1, v2, v3):
    if v1 == v2:
        return v1
    if v2 == v3:
        return v2
    if v1 == v3:
        return v3
    return v1




## === cell 6
def predict_for_pretrained(model, batch_size=16):
    ds_test = generator(JPEG_PATH, test_image_ids, batch_size=batch_size)
    probs = model.predict(ds_test, verbose=1)
    preds = np.argmax(probs, axis=-1)
    return preds


dense_preds = predict_for_pretrained(dense201)
inception_preds = predict_for_pretrained(inception)
efficient_net_preds = predict_for_pretrained(efficient_net)

n = len(submission)
dense_preds = dense_preds[:n]
inception_preds = inception_preds[:n]
efficient_net_preds = efficient_net_preds[:n]

result = []
for idx in range(n):
    result.append(
        vote_in_ensemble(
            dense_preds[idx], inception_preds[idx], efficient_net_preds[idx]
        )
    )



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2704659732.py in <cell line: 0>()
      7 
      8 
----> 9 dense_preds = predict_for_pretrained(dense201)
     10 inception_preds = predict_for_pretrained(inception)
     11 efficient_net_preds = predict_for_pretrained(efficient_net)

NameError: name 'dense201' is not defined

## === cell 7
submission["label"] = np.asarray(result, dtype=np.int64)
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2344226077.py in <cell line: 0>()
----> 1 submission["label"] = np.asarray(result, dtype=np.int64)
      2 submission.to_csv("submission.csv", index=False)
      3 print(submission.head())
      4 print("Wrote submission.csv with shape:", submission.shape)

NameError: name 'result' is not defined
