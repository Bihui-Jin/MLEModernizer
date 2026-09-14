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
Given a dataset of images of dogs, predict the breed of each image.

## Metric
Multi Class Log Loss.

## Submission Format
For each image in the test set, you must predict a probability for each of the different breeds. The file should contain a header and have the following format:
```
id,affenpinscher,afghan_hound,..,yorkshire_terrier
000621fb3cbb32d8935728e48679680e,0.0083,0.0,...,0.0083
etc.
```

## Dataset Description
- `train.zip` - the training set, you are provided the breed for these dogs
- `test.zip` - the test set, you must predict the probability of each breed for each image
- `sample_submission.csv` - a sample submission file in the correct format
- `labels.csv` - the breeds for the images in the train set

# 2. Python version

3.11

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
tf_keras==2.18.0
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
        input/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
        working/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
```

-> data/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> data/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> input/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> input/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.26695

# 6. Current score

4.89636

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 4.9186) has done: 'I fix the environment crash caused by an incompatible protobuf version by forcing the pure-Python protobuf implementation before importing TensorFlow/Keras. Then I fix feature extraction by correctly building the pretrained base model (your current code accidentally calls the model output as if it were a layer, producing `None` tensors) so training can run. Next, I fix test image loading to only read `.jpg` files and avoid subdirectories like `test/test`, and ensure test IDs align with the same sorted filename order used for prediction. Finally, I keep the same model/training approach but correct the callback monitor from `val_acc` to `val_accuracy` so LR scheduling actually works, and write a valid `submission.csv` with the exact sample-submission columns.'
- What this solution (achieved 4.89636) has done: 'I fix the environment crash by forcing TensorFlow to use the pure-Python protobuf implementation early and consistently, which avoids the `MessageFactory.GetPrototype` error in this Kaggle image. Next, I fix the `None values not supported` training failure by ensuring images are converted to real numeric arrays (not PIL objects) and by explicitly converting extracted features to `np.float32`. Finally, I keep your exact model/training approach (InceptionV3 feature extraction + dropout + softmax) but make the pipeline deterministic and ensure the submission columns and test-id ordering match `sample_submission.csv`, so the log-loss score improves from the current broken/misaligned output toward the target.'
- What this solution (achieved 4.89637) has done: 'I fix the TensorFlow/protobuf crash by pinning the protobuf runtime to the pure-Python implementation **and** forcing a compatible protobuf major version inside the notebook before importing TensorFlow (this is the root cause of the `MessageFactory.GetPrototype` error in this environment). Next, I fix the `None values not supported` training failure by making `get_features()` robust to any `None`/object outputs (and by explicitly converting/validating arrays before `model.fit`), while keeping your InceptionV3 feature-extraction + dropout + softmax training logic unchanged. I also make the train/test image directory selection unambiguous (avoid nested `test/test` or `train/train` pitfalls) and ensure the submission columns and test-id ordering exactly match `sample_submission.csv`. These are correctness/stability fixes; they should also improve the score substantially from the current broken calibration/misalignment behavior toward your target log-loss band.'
- What this solution (achieved 4.89636) has done: 'I fix the `None values not supported` crash by eliminating accidental `None` entries in the feature matrix (caused by inconsistent preprocessing/output shapes) and by making feature extraction deterministic and consistent between train/test. I also ensure labels `y` are built as a proper 2D one-hot float array (not a 3D array from `(N,1)` indices) so Keras never sees `None` during tensor conversion. These are correctness fixes that also materially improve log-loss because the model actually train on valid features/labels rather than crashing or training on malformed arrays. Finally, I keep the same core approach (InceptionV3 feature extraction + Dropout + Dense softmax) and produce a submission aligned exactly to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
import subprocess


def _ensure_protobuf_compatible():
    try:
        from importlib.metadata import version as _pkg_version

        v = _pkg_version("protobuf")
        major = int(v.split(".")[0])
        if major >= 5:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
            )
            for m in list(sys.modules.keys()):
                if m.startswith("google.protobuf"):
                    del sys.modules[m]
    except Exception as e:
        print("Warning: protobuf compatibility step encountered:", repr(e))


_ensure_protobuf_compatible()

import gc
import time
import random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.autonotebook import tqdm

import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

from keras import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam, SGD
from keras.layers import Dense, Dropout, Lambda, Input, GlobalAveragePooling2D
from keras.utils import to_categorical
from keras.preprocessing.image import load_img
from tensorflow.keras.models import Model

print("Python:", os.sys.version.split()[0])
print("TensorFlow:", tf.__version__)



## === cell 1
labels = pd.read_csv("/kaggle/input/dog-breed-identification/labels.csv")
labels.head()



## === cell 2
classes = sorted(list(set(labels["breed"])))
n_classes = len(classes)
print("Total unique breed {}".format(n_classes))

class_to_num = dict(zip(classes, range(n_classes)))



## === cell 3
input_shape = (331, 331, 3)


def images_to_array(directory, label_dataframe, target_size=input_shape):
    image_labels = label_dataframe["breed"].values
    images = np.zeros(
        [len(label_dataframe), target_size[0], target_size[1], target_size[2]],
        dtype=np.uint8,
    )
    y_idx = np.zeros([len(label_dataframe)], dtype=np.int32)

    for ix, image_name in enumerate(tqdm(label_dataframe["id"].values)):
        img_dir = os.path.join(directory, image_name + ".jpg")
        img = load_img(img_dir, target_size=target_size)
        images[ix] = np.asarray(img, dtype=np.uint8)
        del img

        dog_breed = image_labels[ix]
        y_idx[ix] = class_to_num[dog_breed]

    y = to_categorical(y_idx, num_classes=n_classes).astype(np.float32, copy=False)
    return images, y




## === cell 4
TRAIN_DIR_CANDIDATES = [
    "/kaggle/input/dog-breed-identification/train",
    "/kaggle/input/dog-breed-identification/dog-breed-identification/train",
]
TRAIN_DIR = next((p for p in TRAIN_DIR_CANDIDATES if os.path.isdir(p)), None)
if TRAIN_DIR is None:
    raise FileNotFoundError(
        f"Could not find train dir in candidates: {TRAIN_DIR_CANDIDATES}"
    )
print("Using TRAIN_DIR:", TRAIN_DIR)

t = time.time()
X, y = images_to_array(TRAIN_DIR, labels[:])
print("runtime in seconds: {}".format(time.time() - t))
print("X:", X.shape, X.dtype, "y:", y.shape, y.dtype)



## === cell 5
lrr = ReduceLROnPlateau(
    monitor="val_accuracy", factor=0.01, patience=3, min_lr=1e-5, verbose=1
)
EarlyStop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)



## === cell 6
batch_size = 128
epochs = 100
learn_rate = 0.001

sgd = SGD(learning_rate=learn_rate, momentum=0.9, nesterov=False)
adam = Adam(
    learning_rate=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False
)



## === cell 7
img_size = (331, 331, 3)


def get_features(model_name, model_preprocessor, input_size, data):
    input_layer = Input(shape=input_size)
    preprocessed = Lambda(model_preprocessor)(input_layer)

    base_model = model_name(
        weights="imagenet", include_top=False, input_shape=input_size
    )
    base_model.trainable = False

    x = base_model(preprocessed, training=False)
    avg = GlobalAveragePooling2D()(x)
    feature_extractor = Model(inputs=input_layer, outputs=avg)

    feature_maps = feature_extractor.predict(data, verbose=1, batch_size=64)
    if feature_maps is None:
        raise ValueError(
            "Feature extractor returned None. Check model graph construction."
        )

    feature_maps = np.asarray(feature_maps)
    if feature_maps.dtype == object:
        raise ValueError(
            f"Feature maps dtype=object; got invalid outputs. Shape={feature_maps.shape}"
        )
    feature_maps = feature_maps.astype(np.float32, copy=False)

    if feature_maps.ndim != 2:
        raise ValueError(
            f"Expected 2D feature matrix [N,F], got shape={feature_maps.shape}"
        )

    if np.any(~np.isfinite(feature_maps)):
        raise ValueError("Non-finite values found in extracted features.")

    print("Feature maps shape: ", feature_maps.shape, feature_maps.dtype)
    return feature_maps




## === cell 8
from keras.applications.inception_v3 import InceptionV3, preprocess_input

inception_preprocessor = preprocess_input

inception_features = get_features(InceptionV3, inception_preprocessor, img_size, X)
print("Inception feature maps shape", inception_features.shape)



## === cell 9
del X
gc.collect()



## === cell 10
y = np.asarray(y, dtype=np.float32)
inception_features = np.asarray(inception_features, dtype=np.float32)

assert inception_features.ndim == 2, "Expected [N, F] features"
assert y.ndim == 2 and y.shape[1] == n_classes, f"Expected y as [N,{n_classes}]"
assert np.isfinite(inception_features).all(), "Found NaN/Inf in features"
assert np.isfinite(y).all(), "Found NaN/Inf in labels"
assert y.shape[0] == inception_features.shape[0], "Mismatch between X features and y"

model = Sequential()
model.add(Dropout(0.7, input_shape=(inception_features.shape[1],)))
model.add(Dense(n_classes, activation="softmax"))

model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(
    inception_features,
    y,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    callbacks=[lrr, EarlyStop],
    verbose=1,
)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1487739122.py in <cell line: 0>()
     15 model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])
     16 
---> 17 history = model.fit(
     18     inception_features,
     19     y,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/core.py in convert_to_tensor(x, dtype, sparse)
    135             x = tf.convert_to_tensor(x)
    136             return tf.cast(x, dtype)
--> 137         return tf.convert_to_tensor(x, dtype=dtype)
    138     elif dtype is not None and not x.dtype == dtype:
    139         if isinstance(x, tf.SparseTensor):

ValueError: None values not supported.

## === cell 11
model.summary()



## === cell 12
acc = history.history.get("accuracy", [])
val_acc = history.history.get("val_accuracy", [])
loss = history.history.get("loss", [])
val_loss = history.history.get("val_loss", [])
epochs_range = range(1, len(acc) + 1)

plt.plot(epochs_range, acc, label="Train accuracy")
plt.plot(epochs_range, val_acc, label="Val accuracy")
plt.title("Training & validation accuracy")
plt.legend()

plt.figure()
plt.plot(epochs_range, loss, label="Train loss")
plt.plot(epochs_range, val_loss, label="Val loss")
plt.title("Training & validation loss")
plt.legend()
plt.show()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1716371872.py in <cell line: 0>()
----> 1 acc = history.history.get("accuracy", [])
      2 val_acc = history.history.get("val_accuracy", [])
      3 loss = history.history.get("loss", [])
      4 val_loss = history.history.get("val_loss", [])
      5 epochs_range = range(1, len(acc) + 1)

NameError: name 'history' is not defined

## === cell 13
del inception_features
gc.collect()




## === cell 14
def images_to_array_test(test_path, img_size=(331, 331, 3)):
    fnames = sorted([f for f in os.listdir(test_path) if f.lower().endswith(".jpg")])
    test_filenames = [os.path.join(test_path, f) for f in fnames]

    data_size = len(test_filenames)
    images = np.zeros([data_size, img_size[0], img_size[1], 3], dtype=np.uint8)

    for ix, img_dir in enumerate(tqdm(test_filenames)):
        img = load_img(img_dir, target_size=img_size)
        images[ix] = np.asarray(img, dtype=np.uint8)
        del img

    print("Output Data Size: ", images.shape)
    return images, fnames


TEST_DIR_CANDIDATES = [
    "/kaggle/input/dog-breed-identification/test",
    "/kaggle/input/dog-breed-identification/dog-breed-identification/test",
]
TEST_DIR = next((p for p in TEST_DIR_CANDIDATES if os.path.isdir(p)), None)
if TEST_DIR is None:
    raise FileNotFoundError(
        f"Could not find test dir in candidates: {TEST_DIR_CANDIDATES}"
    )
print("Using TEST_DIR:", TEST_DIR)

test_data, test_fnames = images_to_array_test(TEST_DIR, img_size)




## === cell 15
def extact_features(data):
    inception_features_local = get_features(
        InceptionV3, inception_preprocessor, img_size, data
    )
    print("Inception feature maps shape", inception_features_local.shape)
    return inception_features_local


test_features = extact_features(test_data)



## === cell 16
del test_data
gc.collect()



## === cell 17
pred = model.predict(
    np.asarray(test_features, dtype=np.float32), verbose=1, batch_size=64
)
pred = np.asarray(pred, dtype=np.float32)
print("pred:", pred.shape, pred.dtype)



## === cell 18
sample_sub = pd.read_csv("/kaggle/input/dog-breed-identification/sample_submission.csv")
breed_cols = [c for c in sample_sub.columns if c != "id"]

test_ids = [os.path.splitext(f)[0] for f in test_fnames]

preds_df = pd.DataFrame(pred, columns=classes)
preds_df.insert(0, "id", test_ids)

preds_df = preds_df.reindex(columns=["id"] + breed_cols)

missing = [c for c in breed_cols if c not in preds_df.columns]
if missing:
    for c in missing:
        preds_df[c] = 1e-12
    preds_df = preds_df.reindex(columns=["id"] + breed_cols)

probs = preds_df[breed_cols].to_numpy(dtype=np.float64)
probs = np.clip(probs, 1e-12, 1.0)
probs = probs / probs.sum(axis=1, keepdims=True)
preds_df[breed_cols] = probs

preds_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", preds_df.shape)
print(preds_df.head())
