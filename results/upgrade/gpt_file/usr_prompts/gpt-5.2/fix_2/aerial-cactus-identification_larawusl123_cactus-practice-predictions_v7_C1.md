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
imutils==0.5.4
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
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.999451

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

INPUT_ROOT = "/kaggle/input/aerial-cactus-identification"
print("Input root exists:", os.path.exists(INPUT_ROOT))
print("Listing /kaggle/input:", os.listdir("/kaggle/input")[:10])



## === cell 1
train_dir = os.path.join(INPUT_ROOT, "train", "train")
test_dir = os.path.join(INPUT_ROOT, "test", "test")
train_csv_path = os.path.join(INPUT_ROOT, "train.csv")
sample_sub_path = os.path.join(INPUT_ROOT, "sample_submission.csv")

print("train_dir:", train_dir, "exists:", os.path.exists(train_dir))
print("test_dir :", test_dir, "exists:", os.path.exists(test_dir))
print("train_csv:", train_csv_path, "exists:", os.path.exists(train_csv_path))
print("sample   :", sample_sub_path, "exists:", os.path.exists(sample_sub_path))



## === cell 2
import tensorflow as tf
from tensorflow import keras

print("TensorFlow:", tf.__version__)

SEED = 42
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Determinism not fully enabled:", repr(e))




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
def focal_loss_fn(gamma=2.0, alpha=0.25):
    EPSILON = 1e-6

    def ce(y_true, y_pred, weights=None):
        mask = y_pred < EPSILON
        true_vals = tf.fill(tf.shape(y_pred), EPSILON)
        y_pred = tf.where(mask, true_vals, y_pred)
        ce_val = y_true * (-tf.math.log(y_pred)) + (1 - y_true) * (
            -tf.math.log(1 - y_pred)
        )

        if weights is not None:
            ce_val = ce_val * weights

        ce_loss = tf.reduce_mean(ce_val + EPSILON)
        return ce_loss

    def focal_loss_fixed(y_true, y_pred):
        t = y_true
        p = y_pred

        pt = p * t + (1 - p) * (1 - t)
        w = alpha * t + (1 - alpha) * (1 - t)
        w = tf.pow((1 - pt), gamma)

        fl = ce(y_true, y_pred, w)
        return fl

    return focal_loss_fixed




## === cell 4
train_df = pd.read_csv(train_csv_path)
sample_sub = pd.read_csv(sample_sub_path)

print(train_df.shape, train_df.columns.tolist())
print(sample_sub.shape, sample_sub.columns.tolist())
print(
    "Train label distribution:\n", train_df["has_cactus"].value_counts(normalize=True)
)

IMG_SIZE = 32  # dataset thumbnails are 32x32; keep native resolution
BATCH_SIZE = 256

train_paths = [os.path.join(train_dir, fname) for fname in train_df["id"].values]
train_labels = train_df["has_cactus"].astype("float32").values


def decode_image(path, label=None):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear")
    if label is None:
        return img
    return img, tf.reshape(label, (1,))


from sklearn.model_selection import train_test_split

idx = np.arange(len(train_paths))
train_idx, val_idx = train_test_split(
    idx,
    test_size=0.15,
    random_state=SEED,
    stratify=train_labels,
)

tr_paths = np.array(train_paths)[train_idx]
tr_labels = train_labels[train_idx]
va_paths = np.array(train_paths)[val_idx]
va_labels = train_labels[val_idx]

ds_train = tf.data.Dataset.from_tensor_slices((tr_paths, tr_labels))
ds_val = tf.data.Dataset.from_tensor_slices((va_paths, va_labels))

AUTOTUNE = tf.data.AUTOTUNE
ds_train = (
    ds_train.shuffle(4096, seed=SEED, reshuffle_each_iteration=True)
    .map(decode_image, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)
ds_val = (
    ds_val.map(decode_image, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)



## === cell 5
inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = keras.layers.MaxPooling2D()(x)

x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = keras.layers.MaxPooling2D()(x)

x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.25)(x)
outputs = keras.layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[keras.metrics.AUC(name="auc")],
)
model.summary()



## === cell 6
EPOCHS = 12
history = model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS, verbose=2)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NotFoundError                             Traceback (most recent call last)
/tmp/ipykernel_55/2497888025.py in <cell line: 0>()
      1 # Train (no early stopping added; fixed epochs to respect constraints)
      2 EPOCHS = 12
----> 3 history = model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS, verbose=2)
      4 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     57       e.message += " name: " + name
     58     raise core._status_to_exception(e) from None
---> 59   except TypeError as e:
     60     keras_symbolic_tensors = [x for x in inputs if _is_keras_symbolic_tensor(x)]
     61     if keras_symbolic_tensors:

NotFoundError: Graph execution error:

Detected at node ReadFile defined at (most recent call last):
<stack traces unavailable>
Detected at node ReadFile defined at (most recent call last):
<stack traces unavailable>
2 root error(s) found.
  (0) NOT_FOUND:  /kaggle/input/aerial-cactus-identification/train/train/a930a0284d619db873d7cd9722b01afd.jpg; No such file or directory
	 [[{{node ReadFile}}]]
	 [[IteratorGetNext]]
	 [[IteratorGetNext/_7]]
  (1) NOT_FOUND:  /kaggle/input/aerial-cactus-identification/train/train/a930a0284d619db873d7cd9722b01afd.jpg; No such file or directory
	 [[{{node ReadFile}}]]
	 [[IteratorGetNext]]
0 successful operations.
0 derived errors ignored. [Op:__inference_multi_step_on_iterator_2722]

## === cell 7
test_ids = sample_sub["id"].values
test_paths = [os.path.join(test_dir, fname) for fname in test_ids]

missing = [p for p in test_paths if not os.path.exists(p)]
print("Missing test files:", len(missing))
if len(missing) > 0:
    print("Example missing:", missing[0])

ds_test = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(lambda p: decode_image(p, None), num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)



## === cell 8
predictions = model.predict(ds_test, verbose=1).reshape(-1)

predictions = np.clip(predictions, 0.0, 1.0)

print(
    "Pred shape:",
    predictions.shape,
    "min/max:",
    float(predictions.min()),
    float(predictions.max()),
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NotFoundError                             Traceback (most recent call last)
/tmp/ipykernel_55/3738976427.py in <cell line: 0>()
      1 # Predict probabilities
----> 2 predictions = model.predict(ds_test, verbose=1).reshape(-1)
      3 
      4 # Safety: clip to valid probability range (score-neutral; avoids rare numeric issues)
      5 predictions = np.clip(predictions, 0.0, 1.0)

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

NotFoundError: {{function_node __wrapped__IteratorGetNext_output_types_1_device_/job:localhost/replica:0/task:0/device:CPU:0}} /kaggle/input/aerial-cactus-identification/test/test/09034a34de0e2015a8a28dfe18f423f6.jpg; No such file or directory
	 [[{{node ReadFile}}]] [Op:IteratorGetNext] name: 

## === cell 9
submit_csv = pd.DataFrame(
    {"id": test_ids, "has_cactus": predictions.astype(np.float32)}
)
submit_path = "submission_resnet50.csv"
submit_csv.to_csv(submit_path, index=False)

print("Wrote:", submit_path)
print(submit_csv.head())
print("Rows:", len(submit_csv), "Cols:", submit_csv.columns.tolist())

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/68843084.py in <cell line: 0>()
      1 # Build submission aligned to sample_submission order
      2 submit_csv = pd.DataFrame(
----> 3     {"id": test_ids, "has_cactus": predictions.astype(np.float32)}
      4 )
      5 submit_path = "submission_resnet50.csv"

NameError: name 'predictions' is not defined
