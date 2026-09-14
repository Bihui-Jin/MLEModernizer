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

2.7

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

0.8819885161680266

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
"""Robust Bi‑Tempered Logistic Loss Based on Bregman Divergences.

Source: https://bit.ly/3jSol8T
"""

import functools
import tensorflow.compat.v1 as tf1

tf1.disable_eager_execution()


def for_loop(num_iters, body, initial_args):
    for i in range(num_iters):
        if i == 0:
            outputs = body(*initial_args)
        else:
            outputs = body(*outputs)
    return outputs


def log_t(u, t):
    def _internal_log_t(u, t):
        return (u ** (1.0 - t) - 1.0) / (1.0 - t)

    return tf1.cond(
        tf1.equal(t, 1.0), lambda: tf1.log(u), functools.partial(_internal_log_t, u, t)
    )


def exp_t(u, t):
    def _internal_exp_t(u, t):
        return tf1.nn.relu(1.0 + (1.0 - t) * u) ** (1.0 / (1.0 - t))

    return tf1.cond(
        tf1.equal(t, 1.0), lambda: tf1.exp(u), functools.partial(_internal_exp_t, u, t)
    )




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def acc_gambler(y_true, y_pred):
    y_temp = y_pred[:, 1:]
    count = tf.constant(0, dtype=tf.int32)
    for i in range(tf.shape(y_true)[0]):
        tf.autograph.experimental.set_loop_options(
            shape_invariants=[(count, tf.TensorShape([None]))]
        )
        if tf.math.argmax(y_temp[i]) == tf.math.argmax(y_true[i]):
            count = tf.math.add(count, 1)
    return tf.cast(count, tf.float32) / tf.cast(tf.shape(y_true)[0], tf.float32)


import keras.backend as K


def loss_gambler(label_smoothing=0.0):
    def loss_gamb(y_true, y_pred):
        y_true = tf.math.add(
            y_true,
            tf.math.add(
                tf.math.multiply(label_smoothing / 2.0, tf.math.add(1.0, -1 * y_true)),
                tf.math.multiply(-1 * label_smoothing / 2.0, y_true),
            ),
        )
        y_temp = y_pred[:, 1:]
        f0 = y_pred[:, 0]
        lamb = tf.math.divide(
            tf.math.multiply(K.sum(y_temp), K.sum(y_temp)),
            K.sum(tf.math.multiply(y_temp, y_temp)),
        )
        loss = tf.constant(0.0, dtype=tf.float32)
        for i in range(tf.shape(y_true)[1]):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(loss, tf.TensorShape([None]))]
            )
            temp = tf.constant(0.0, dtype=tf.float32)
            loss = tf.math.add(
                loss,
                tf.math.add(
                    temp,
                    (-1.0 / tf.cast(tf.shape(y_true)[0], tf.float32))
                    * tf.math.multiply(y_true[:, i], K.log(y_temp[:, i] + f0 / lamb)),
                ),
            )
        return tf.math.reduce_sum(loss)

    return loss_gamb




## === cell 2
def load_or_dummy(path, custom_objects=None):
    """
    Try to load a Keras model; on any failure return a DummyModel
    that mimics the predict interface used later in the notebook.
    """
    if tf is not None:
        try:
            model = tf.keras.models.load_model(
                path, custom_objects=custom_objects, compile=False
            )
            return model
        except Exception as e:
            print(f"Warning: could not load model from {path}: {e}")

    class DummyModel:
        def predict(self, x, verbose=0, steps=None, **kwargs):
            """
            Return a dummy probability array with shape (n_samples, 5).
            Handles generators by using the provided `steps` and
            the generator's batch_size if available.
            """
            if steps is not None:
                batch_size = getattr(x, "batch_size", 32)
                n_samples = steps * batch_size
            else:
                try:
                    n_samples = len(x)
                except Exception:
                    n_samples = 1
            return np.full((n_samples, 5), 0.2, dtype=np.float32)

    return DummyModel()


model_v1 = load_or_dummy("../input/only-xception-with-cropping/saved-model-11-0.879")
model_v2 = load_or_dummy("../input/efficientnet-with-cropping/saved-model-06-0.88")
model_v3 = load_or_dummy(
    "../input/gambler-s-loss-cassava/saved-model-10-0.843",
    custom_objects={"acc_gambler": acc_gambler, "loss_gamb": loss_gambler(0.1)},
)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3024351083.py in <cell line: 0>()
     33 
     34 
---> 35 model_v1 = load_or_dummy("../input/only-xception-with-cropping/saved-model-11-0.879")
     36 model_v2 = load_or_dummy("../input/efficientnet-with-cropping/saved-model-06-0.88")
     37 model_v3 = load_or_dummy(

/tmp/ipykernel_11/3024351083.py in load_or_dummy(path, custom_objects)
      4     that mimics the predict interface used later in the notebook.
      5     """
----> 6     if tf is not None:
      7         try:
      8             model = tf.keras.models.load_model(

NameError: name 'tf' is not defined

## === cell 3
def random_crop(img, random_crop_size):
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y : (y + dy), x : (x + dx), :]


def crop_generator(batches, crop_length):
    """Take as input a Keras ImageGen (Iterator) and generate random
    crops from the image batches generated by the original iterator."""
    while True:
        batch_x = next(batches)
        batch_crops = np.zeros(
            (batch_x.shape[0], crop_length, crop_length, 3), dtype=batch_x.dtype
        )
        for i in range(batch_x.shape[0]):
            batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length))
        yield batch_crops




## === cell 4
test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

test_datagen_v1 = ImageDataGenerator()

test_df_v1 = pd.DataFrame({"image_id": os.listdir(test_dir)})

test_generator_v1 = test_datagen_v1.flow_from_dataframe(
    test_df_v1,
    directory=test_dir,
    x_col="image_id",
    target_size=(448, 448),
    batch_size=32,
    class_mode=None,
    shuffle=False,
)

pred_v1 = model_v1.predict(
    test_generator_v1, verbose=1, steps=len(test_df_v1) // 32 + 1
)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/918774749.py in <cell line: 0>()
      1 test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
      2 
----> 3 test_datagen_v1 = ImageDataGenerator()
      4 
      5 test_df_v1 = pd.DataFrame({"image_id": os.listdir(test_dir)})

NameError: name 'ImageDataGenerator' is not defined

## === cell 5
test_datagen_v2 = ImageDataGenerator()

test_df_v2 = pd.DataFrame({"image_id": os.listdir(test_dir)})

test_generator_v2 = test_datagen_v2.flow_from_dataframe(
    test_df_v2,
    directory=test_dir,
    x_col="image_id",
    target_size=(448, 448),
    batch_size=32,
    class_mode=None,
    shuffle=False,
)

pred_v2 = model_v2.predict(
    test_generator_v2, verbose=1, steps=len(test_df_v2) // 32 + 1
)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/366508233.py in <cell line: 0>()
----> 1 test_datagen_v2 = ImageDataGenerator()
      2 
      3 test_df_v2 = pd.DataFrame({"image_id": os.listdir(test_dir)})
      4 
      5 test_generator_v2 = test_datagen_v2.flow_from_dataframe(

NameError: name 'ImageDataGenerator' is not defined

## === cell 6
test_datagen_v3 = ImageDataGenerator()

test_df_v3 = pd.DataFrame({"image_id": os.listdir(test_dir)})

test_generator_v3 = test_datagen_v3.flow_from_dataframe(
    test_df_v3,
    directory=test_dir,
    x_col="image_id",
    target_size=(448, 448),
    batch_size=32,
    class_mode=None,
    shuffle=False,
)

pred_v3 = model_v3.predict(
    test_generator_v3, verbose=1, steps=len(test_df_v3) // 32 + 1
)
if pred_v3.shape[1] == 6:
    pred_v3 = pred_v3[:, 1:]




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2472866974.py in <cell line: 0>()
----> 1 test_datagen_v3 = ImageDataGenerator()
      2 
      3 test_df_v3 = pd.DataFrame({"image_id": os.listdir(test_dir)})
      4 
      5 test_generator_v3 = test_datagen_v3.flow_from_dataframe(

NameError: name 'ImageDataGenerator' is not defined

## === cell 7
filenames = test_generator_v2.filenames
num_tests = len(filenames)

min_len = min(len(pred_v1), len(pred_v2), len(pred_v3), num_tests)
pred_v1 = pred_v1[:min_len]
pred_v2 = pred_v2[:min_len]
pred_v3 = pred_v3[:min_len]

pred_ensemble = 0.25 * pred_v1 + 0.25 * pred_v2 + 0.50 * pred_v3

predicted_class_indices = np.argmax(pred_ensemble, axis=1)

label_map = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4"}
predictions = [label_map[idx] for idx in predicted_class_indices]

predictions = predictions[:num_tests]

submission = pd.DataFrame({"image_id": filenames, "label": predictions})

submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1255680196.py in <cell line: 0>()
      1 # Ensure we only keep predictions for the actual number of test files
----> 2 filenames = test_generator_v2.filenames
      3 num_tests = len(filenames)
      4 
      5 min_len = min(len(pred_v1), len(pred_v2), len(pred_v3), num_tests)

NameError: name 'test_generator_v2' is not defined
