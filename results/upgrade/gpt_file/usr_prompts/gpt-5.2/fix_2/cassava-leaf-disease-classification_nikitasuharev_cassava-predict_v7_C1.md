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

0.7417648836506497

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


print("Tensorflow version " + tf.__version__)




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

NUM_CLASSES = 5
IMG_SIZE = 512


def build_pretrained_app(
    app_ctor,
    preprocess_fn,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    num_classes=NUM_CLASSES,
):
    inputs = keras.Input(shape=input_shape)
    x = preprocess_fn(inputs)
    base = app_ctor(
        include_top=False, weights="imagenet", input_tensor=x, pooling="avg"
    )
    outputs = keras.layers.Dense(num_classes, activation="softmax")(base.output)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


dense201 = build_pretrained_app(
    keras.applications.DenseNet201, keras.applications.densenet.preprocess_input
)
inception = build_pretrained_app(
    keras.applications.InceptionV3, keras.applications.inception_v3.preprocess_input
)
efficient_net = build_pretrained_app(
    keras.applications.EfficientNetB3, keras.applications.efficientnet.preprocess_input
)



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
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))[:, :, ::-1]  # BGR->RGB
    return img


def generator(filepath, paths, batch_size=32):
    n = len(paths)
    for start in range(0, n, batch_size):
        batch_paths = paths[start : start + batch_size]
        batch = [load_image(filepath, p) for p in batch_paths]
        yield np.stack(batch, axis=0)




## === cell 4
submission = pd.read_csv(
    "../input/cassava-leaf-disease-classification/sample_submission.csv"
)

test_image_ids = np.sort(submission.image_id.values)




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

n = len(test_image_ids)
dense_preds = dense_preds[:n]
inception_preds = inception_preds[:n]
efficient_net_preds = efficient_net_preds[:n]

result = []
for idx in range(n):
    result.append(
        vote_in_ensemble(
            int(dense_preds[idx]),
            int(inception_preds[idx]),
            int(efficient_net_preds[idx]),
        )
    )



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1719362994.py in <cell line: 0>()
      7 
      8 
----> 9 dense_preds = predict_for_pretrained(dense201)
     10 inception_preds = predict_for_pretrained(inception)
     11 efficient_net_preds = predict_for_pretrained(efficient_net)

/tmp/ipykernel_11/1719362994.py in predict_for_pretrained(model, batch_size)
      2     # steps is determined implicitly by the finite generator length
      3     ds_test = generator(JPEG_PATH, test_image_ids, batch_size=batch_size)
----> 4     probs = model.predict(ds_test, verbose=1)
      5     preds = np.argmax(probs, axis=-1)
      6     return preds

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/data_adapters/generator_data_adapter.py in __init__(self, generator)
     15         self._output_signature = None
     16         if not isinstance(first_batches[0], tuple):
---> 17             raise ValueError(
     18                 "When passing a Python generator to a Keras model, "
     19                 "the generator must return a tuple, either "

ValueError: When passing a Python generator to a Keras model, the generator must return a tuple, either (input,) or (inputs, targets) or (inputs, targets, sample_weights). Received: [[[[8.09886232e-02 1.00596465e-01 1.79027855e-01]
   [2.53714789e-02 5.15969694e-02 1.26719519e-01]
   [1.19504444e-02 3.94014269e-02 1.12318091e-01]
   ...
   [3.76290619e-01 5.24052203e-01 2.70465702e-01]
   [3.83700967e-01 5.41071177e-01 2.51120180e-01]
   [3.83706748e-01 5.47120154e-01 2.28860289e-01]]

  [[9.60075855e-02 1.15615427e-01 1.93035766e-01]
   [7.57017881e-02 1.01927280e-01 1.76038802e-01]
   [6.45153597e-02 9.19663385e-02 1.64282709e-01]
   ...
   [3.98131132e-01 5.39613962e-01 2.82371134e-01]
   [4.08870429e-01 5.63098431e-01 2.71599263e-01]
   [4.00437564e-01 5.59085906e-01 2.41837084e-01]]

  [[5.54946028e-02 7.51024410e-02 1.49612248e-01]
   [8.09388384e-02 1.07164331e-01 1.78365320e-01]
   [8.67034346e-02 1.14154413e-01 1.84742659e-01]
   ...
   [3.61835569e-01 4.94127214e-01 2.36242920e-01]
   [3.72340292e-01 5.14270306e-01 2.19999433e-01]
   [3.62136185e-01 5.06605208e-01 1.93952024e-01]]

  ...

  [[2.94669122e-01 3.88786763e-01 4.04473037e-01]
   [2.94669122e-01 3.88786763e-01 4.04473037e-01]
   [2.93075979e-01 3.87193620e-01 3.99693638e-01]
   ...
   [8.58834088e-01 7.66339242e-01 7.14450240e-01]
   [8.45251262e-01 7.48345613e-01 6.91207170e-01]
   [8.37422431e-01 7.40516841e-01 6.77741110e-01]]

  [[2.98039228e-01 3.92156869e-01 4.07843143e-01]
   [3.00494999e-01 3.94612640e-01 4.10298914e-01]
   [2.96446085e-01 3.90563726e-01 4.03063744e-01]
   ...
   [8.52524698e-01 7.51796126e-01 6.88762844e-01]
   [8.42943847e-01 7.43893623e-01 6.70517385e-01]
   [8.41237783e-01 7.42187500e-01 6.67357922e-01]]

  [[2.95462847e-01 3.89580488e-01 4.05266762e-01]
   [2.98323572e-01 3.92441213e-01 4.08127487e-01]
   [2.92861521e-01 3.86979163e-01 3.99479181e-01]
   ...
   [8.05500329e-01 7.05667913e-01 6.32616222e-01]
   [8.05050373e-01 7.07011163e-01 6.27459586e-01]
   [8.19276571e-01 7.21237361e-01 6.35636866e-01]]]


 [[[1.21660545e-01 1.17738970e-01 1.37346819e-01]
   [1.27787992e-01 1.23866431e-01 1.43474266e-01]
   [1.32322311e-01 1.28400743e-01 1.48008585e-01]
   ...
   [2.72055000e-01 2.68133432e-01 2.57104009e-01]
   [2.34996364e-01 2.30462044e-01 2.24457145e-01]
   [1.97457120e-01 1.89613983e-01 1.93535551e-01]]

  [[1.10906862e-01 1.06985293e-01 1.26593143e-01]
   [1.17034316e-01 1.13112748e-01 1.32720590e-01]
   [1.21568628e-01 1.17647059e-01 1.37254909e-01]
   ...
   [3.27052683e-01 3.23131114e-01 3.10490400e-01]
   [2.75586903e-01 2.71210551e-01 2.63025641e-01]
   [2.35876232e-01 2.29044124e-01 2.29932606e-01]]

  [[1.04166664e-01 1.00245096e-01 1.19852945e-01]
   [1.10294119e-01 1.06372550e-01 1.25980392e-01]
   [1.15512982e-01 1.11591414e-01 1.31199270e-01]
   ...
   [2.92442173e-01 2.88520604e-01 2.71241218e-01]
   [2.45471448e-01 2.41549879e-01 2.27089077e-01]
   [2.13482305e-01 2.09560737e-01 2.01717600e-01]]

  ...

  [[1.23161776e-02 1.23161776e-02 4.47303941e-03]
   [1.04291132e-02 1.04291132e-02 2.58597592e-03]
   [6.15808880e-03 6.15808880e-03 0.00000000e+00]
   ...
   [5.42174093e-02 4.24527042e-02 1.81879979e-02]
   [5.65870106e-02 4.48223054e-02 1.73713248e-02]
   [5.65870106e-02 4.48223054e-02 1.73713248e-02]]

  [[2.06887648e-02 2.06887648e-02 1.28456270e-02]
   [1.77437589e-02 1.77437589e-02 9.90062021e-03]
   [1.24818096e-02 1.24818096e-02 4.63867234e-03]
   ...
   [5.02632894e-02 3.84985842e-02 1.18690645e-02]
   [5.24461940e-02 4.06814888e-02 1.23209637e-02]
   [5.69939092e-02 4.52292040e-02 1.19571462e-02]]

  [[2.60110293e-02 2.60110293e-02 1.81678925e-02]
   [2.31924020e-02 2.31924020e-02 1.53492652e-02]
   [1.76776964e-02 1.76776964e-02 9.83455963e-03]
   ...
   [4.57395054e-02 3.39748003e-02 6.52382104e-03]
   [5.21532260e-02 4.03885208e-02 1.17120482e-02]
   [6.38815463e-02 5.21168448e-02 1.68227255e-02]]]


 [[[2.10873351e-01 2.22638056e-01 1.48128256e-01]
   [2.16189876e-01 2.27954581e-01 1.60062432e-01]
   [2.21738100e-01 2.30316520e-01 1.65243000e-01]
   ...
   [1.18903182e-01 4.29371558e-02 5.07802926e-02]
   [1.13672830e-01 3.91630307e-02 4.70061675e-02]
   [1.13388479e-01 3.88786793e-02 4.67218161e-02]]

  [[2.35132322e-01 2.46897027e-01 1.74409285e-01]
   [2.29390711e-01 2.41155416e-01 1.73579201e-01]
   [2.26850688e-01 2.35429108e-01 1.71177045e-01]
   ...
   [1.14307597e-01 4.08088267e-02 4.66299057e-02]
   [1.10237636e-01 3.87609154e-02 4.45819944e-02]
   [1.07781865e-01 3.63051482e-02 4.21262272e-02]]

  [[2.42111862e-01 2.53876567e-01 1.87209904e-01]
   [2.29577392e-01 2.41342098e-01 1.74675435e-01]
   [2.22752959e-01 2.30330884e-01 1.71445116e-01]
   ...
   [1.07342407e-01 4.01242748e-02 4.01242748e-02]
   [1.01960786e-01 3.92156877e-02 3.92156877e-02]
   [1.00749657e-01 4.04268168e-02 3.92156877e-02]]

  ...

  [[2.35294122e-02 5.49019612e-02 1.17647061e-02]
   [2.35294122e-02 5.49019612e-02 1.17647061e-02]
   [2.35294122e-02 5.49019612e-02 1.17647061e-02]
   ...
   [2.28448603e-02 3.54674086e-02 1.42664295e-02]
   [1.98711324e-02 2.71015242e-02 7.49368174e-03]
   [2.12928932e-02 2.52144616e-02 5.60661778e-03]]

  [[2.35294122e-02 5.49019612e-02 1.17647061e-02]
   [2.35294122e-02 5.49019612e-02 1.17647061e-02]
   [2.35294122e-02 5.49019612e-02 1.17647061e-02]
   ...
   [2.18012799e-02 3.44238281e-02 1.32228481e-02]
   [1.66973043e-02 2.39276960e-02 4.31985315e-03]
   [1.66973043e-02 2.06188727e-02 1.01102947e-03]]

  [[2.35294122e-02 5.49019612e-02 1.17647061e-02]
   [2.35294122e-02 5.49019612e-02 1.17647061e-02]
   [2.35294122e-02 5.49019612e-02 1.17647061e-02]
   ...
   [1.97447538e-02 3.23673040e-02 1.11663220e-02]
   [1.21017164e-02 1.93321090e-02 2.84352049e-04]
   [1.21017164e-02 1.60232857e-02 0.00000000e+00]]]


 ...


 [[[1.00000000e+00 1.00000000e+00 9.73223031e-01]
   [1.00000000e+00 1.00000000e+00 9.79556382e-01]
   [1.00000000e+00 1.00000000e+00 9.80729163e-01]
   ...
   [1.00000000e+00 1.00000000e+00 1.00000000e+00]
   [1.00000000e+00 1.00000000e+00 1.00000000e+00]
   [1.00000000e+00 1.00000000e+00 1.00000000e+00]]

  [[1.00000000e+00 9.98988986e-01 9.83425260e-01]
   [1.00000000e+00 9.99842048e-01 9.85881031e-01]
   [1.00000000e+00 1.00000000e+00 9.86335814e-01]
   ...
   [1.00000000e+00 1.00000000e+00 1.00000000e+00]
   [1.00000000e+00 1.00000000e+00 1.00000000e+00]
   [1.00000000e+00 1.00000000e+00 1.00000000e+00]]

  [[1.00000000e+00 9.94867325e-01 9.95526969e-01]
   [1.00000000e+00 9.97965515e-01 9.95526969e-01]
   [1.00000000e+00 9.98314977e-01 9.95526969e-01]
   ...
   [1.00000000e+00 1.00000000e+00 1.00000000e+00]
   [1.00000000e+00 1.00000000e+00 1.00000000e+00]
   [1.00000000e+00 1.00000000e+00 1.00000000e+00]]

  ...

  [[3.14664721e-01 1.89221442e-01 1.80826828e-01]
   [3.83534402e-01 2.63914108e-01 2.55544394e-01]
   [3.66103709e-01 2.55013019e-01 2.47762531e-01]
   ...
   [2.42932364e-01 1.60579428e-01 1.92503452e-01]
   [2.14620665e-01 1.32267728e-01 1.59718707e-01]
   [1.79624319e-01 9.72713679e-02 1.24722354e-01]]

  [[2.66013712e-01 1.29349545e-01 1.33056641e-01]
   [3.04992914e-01 1.74692675e-01 1.75581157e-01]
   [2.52769798e-01 1.32487938e-01 1.32604748e-01]
   ...
   [1.93161190e-01 1.10808253e-01 1.46102369e-01]
   [1.79689422e-01 9.73364785e-02 1.30608544e-01]
   [1.72391042e-01 9.00381058e-02 1.23310171e-01]]

  [[2.65218109e-01 1.24231204e-01 1.31979555e-01]
   [3.02525669e-01 1.68072149e-01 1.72553807e-01]
   [2.45790258e-01 1.20573878e-01 1.24358535e-01]
   ...
   [1.44732311e-01 6.09231405e-02 1.00585945e-01]
   [1.53842106e-01 7.14891627e-02 1.06783278e-01]
   [1.74117267e-01 9.17643234e-02 1.27058446e-01]]]


 [[[5.23917198e-01 3.86662245e-01 3.78819108e-01]
   [5.31117916e-01 3.93862963e-01 3.86019826e-01]
   [5.68997025e-01 4.31742132e-01 4.23898995e-01]
   ...
   [2.71084189e-01 4.14711654e-01 2.50939220e-01]
   [2.51807600e-01 3.82414222e-01 2.39876300e-01]
   [2.51813382e-01 3.65875870e-01 2.43106619e-01]]

  [[4.72435087e-01 3.36191207e-01 3.28348070e-01]
   [5.16544163e-01 3.80300224e-01 3.72457087e-01]
   [5.51757812e-01 4.15513963e-01 4.07670826e-01]
   ...
   [2.04281569e-01 3.48920047e-01 1.74979895e-01]
   [2.16263607e-01 3.51465821e-01 1.92851946e-01]
   [2.09536806e-01 3.28194916e-01 1.86147183e-01]]

  [[4.73919094e-01 3.42007518e-01 3.33690464e-01]
   [5.03111660e-01 3.70568126e-01 3.62461716e-01]
   [5.03298342e-01 3.72966468e-01 3.64122838e-01]
   ...
   [2.12816909e-01 3.58996838e-01 1.69546574e-01]
   [2.49356627e-01 3.83204073e-01 2.07438156e-01]
   [2.40504369e-01 3.62073004e-01 1.96233541e-01]]

  ...

  [[4.13419127e-01 3.15379918e-01 2.95725137e-01]
   [4.29473042e-01 3.34742635e-01 2.94117659e-01]
   [4.62600529e-01 3.71403962e-01 3.16685826e-01]
   ...
   [4.67738986e-01 3.94425005e-01 4.53421801e-01]
   [4.75734353e-01 4.06371593e-01 4.59435314e-01]
   [4.79952693e-01 4.17207599e-01 4.60344851e-01]]

  [[4.05147076e-01 3.13747525e-01 2.82383591e-01]
   [4.28568304e-01 3.37657988e-01 2.86932260e-01]
   [4.79001045e-01 3.88804972e-01 3.24895650e-01]
   ...
   [5.14288485e-01 4.38920826e-01 4.98602211e-01]
   [5.00315964e-01 4.30043668e-01 4.83562142e-01]
   [4.88550305e-01 4.19984132e-01 4.66031909e-01]]

  [[4.94301498e-01 4.05208349e-01 3.70526969e-01]
   [5.17257392e-01 4.27674055e-01 3.74610335e-01]
   [5.62528729e-01 4.72332656e-01 4.02144611e-01]
   ...
   [5.67146361e-01 4.91778702e-01 5.51460087e-01]
   [5.26111603e-01 4.55523312e-01 5.09199798e-01]
   [5.21543801e-01 4.50955480e-01 4.98014361e-01]]]


 [[[6.20375723e-02 8.90040994e-02 2.02971816e-01]
   [8.34865198e-02 1.10368796e-01 2.24378631e-01]
   [1.07859418e-01 1.31723925e-01 2.47242644e-01]
   ...
   [3.31480742e-01 3.27559173e-01 3.58931720e-01]
   [2.94468045e-01 2.90546477e-01 3.21919024e-01]
   [2.60594755e-01 2.56673187e-01 2.88045734e-01]]

  [[6.07450604e-02 8.05367306e-02 1.98091865e-01]
   [7.97439888e-02 9.85552594e-02 2.16600597e-01]
   [1.01807602e-01 1.19393386e-01 2.38051474e-01]
   ...
   [2.66789228e-01 2.62267351e-01 2.95440793e-01]
   [2.49345139e-01 2.44412541e-01 2.78818190e-01]
   [2.30812460e-01 2.25879863e-01 2.60285527e-01]]

  [[6.45077005e-02 7.51081929e-02 1.96047798e-01]
   [8.26593190e-02 9.22794119e-02 2.13498577e-01]
   [9.82125103e-02 1.09977216e-01 2.31545836e-01]
   ...
   [1.90411508e-01 1.84161499e-01 2.22519338e-01]
   [1.82187319e-01 1.74344182e-01 2.17481434e-01]
   [1.76991433e-01 1.69148296e-01 2.12285548e-01]]

  ...

  [[6.49816245e-02 8.85110348e-02 8.06678981e-02]
   [6.82904497e-02 9.18198600e-02 8.39767233e-02]
   [7.21813738e-02 9.57107842e-02 8.78676474e-02]
   ...
   [1.99805647e-01 3.83384079e-01 3.35283607e-01]
   [1.82947502e-01 3.65024716e-01 3.09688091e-01]
   [1.88770503e-01 3.70847702e-01 3.16070199e-01]]

  [[7.03957975e-02 9.39252079e-02 8.60820711e-02]
   [7.33408034e-02 9.68702137e-02 8.90270770e-02]
   [7.39095062e-02 9.74389166e-02 8.95957798e-02]
   ...
   [1.89657062e-01 3.72053087e-01 3.31415653e-01]
   [1.74014837e-01 3.54406983e-01 3.10018390e-01]
   [1.84926480e-01 3.65318626e-01 3.21546614e-01]]

  [[7.78521374e-02 1.01381548e-01 9.35384110e-02]
   [7.50698894e-02 9.85992998e-02 9.07561630e-02]
   [7.45098069e-02 9.80392173e-02 9.01960805e-02]
   ...
   [1.84650734e-01 3.66636038e-01 3.30332786e-01]
   [1.73161775e-01 3.53553921e-01 3.14053893e-01]
   [1.87502876e-01 3.67895037e-01 3.28437120e-01]]]]

## === cell 7
submission["label"] = result
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
assert submission.shape[0] == 2676 and list(submission.columns) == ["image_id", "label"]
assert os.path.exists("submission.csv")

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3779046076.py in <cell line: 0>()
----> 1 submission["label"] = result
      2 submission.to_csv("submission.csv", index=False)
      3 
      4 print(submission.head())
      5 print("Wrote submission.csv with shape:", submission.shape)

NameError: name 'result' is not defined
