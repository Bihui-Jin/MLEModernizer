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

3.12

# 3. Installed packages

colorama==0.4.6
geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
scikit-plot==0.3.7
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
termcolor==3.1.0
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

0.3496

# 6. Current score

0.44817

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.41222) has done: 'I fix the protobuf/Keras import crash by setting TensorFlow-compatible protobuf environment flags before importing TensorFlow/Keras, which resolves the `MessageFactory.GetPrototype` error in this Kaggle setup. I also fix the repeated `NameError: LabelEncoder is not defined` and missing `precision_recall_curve` by importing them once in a stable place. To move the log-loss score strongly toward the target (your current 4.82 is far from 0.3496), I keep your feature-extraction + softmax-head core approach but correct the submission class/probability alignment by using the exact column order from `sample_submission.csv`, and I train the final head on all extracted train features (not just a split) before predicting test. Finally, I ensure the script runs end-to-end within time by skipping the earlier exploratory CNN-on-raw-pixels block (it’s not used for submission and is too slow) while preserving the final feature-extractor logic that produces `submission.csv`.'
- What this solution (achieved 0.4366) has done: 'The crash happens before any training because TensorFlow/Keras is importing protobuf in a way that’s incompatible with the `MessageFactory.GetPrototype` API in this environment. I fix this by forcing the pure-Python protobuf implementation and a compatible TF/keras path **before** importing TensorFlow, and by preferring `tf.keras` consistently (to avoid mixed `keras`/`tf.keras` incompatibilities). I also fix a logic bug in `train_test_split(stratify=...)` (stratify must be 1D labels, not one-hot), which can silently break stratification and hurt log-loss. These changes keep your feature-extraction + softmax-head approach identical, but should run end-to-end and nudge the score toward the 0.3496 target.'
- What this solution (achieved 0.41705) has done: 'I fix the protobuf/TensorFlow import crash by forcing a compatible protobuf setting and (most importantly) importing `tf_keras` (the Kaggle-provided TF-compatible Keras) instead of the standalone `keras` package, while keeping the same `tf.keras.*` API usage everywhere else. I also correct a small but real bug in your feature-extractor construction (the base model call currently misuses `input_shape` and can build incorrectly); the fix preserves the same Inception/Xception/NASNet feature pipeline and head architecture. Finally, I keep the exact submission column order from `sample_submission.csv` and ensure the script always writes `submission.csv` end-to-end.'
- What this solution (achieved 0.42784) has done: 'I fix the protobuf/TensorFlow import crash by setting additional TF/protobuf environment flags *before* importing TensorFlow and by using the TensorFlow-bundled Keras (`tf.keras`) consistently (this avoids the `MessageFactory.GetPrototype` error seen in cell 0). I keep your exact feature-extraction + concatenation + softmax head logic unchanged, but make the image loading and input-size handling consistent (PIL loader expects `(H,W)` while the model input is `(H,W,3)`), preventing silent shape/resize issues. Finally, I keep the submission column order exactly aligned to `sample_submission.csv` and ensure the script always writes a valid `submission.csv`. These fixes are stability/correctness oriented and should also modestly improve log-loss by avoiding preprocessing/shape mismatches and framework mix-ups.'
- What this solution (achieved 0.44095) has done: 'I fix the TensorFlow/protobuf crash by forcing a safe protobuf backend *before* importing TensorFlow, which prevents the `MessageFactory.GetPrototype` error in this environment. I also eliminate the `tf_keras` vs standalone `keras` layer-mismatch that breaks `Sequential.add()` by using **one** consistent Keras implementation (`tf.keras`) everywhere, while keeping your exact feature-extractor + concatenated-features + softmax-head approach unchanged. Finally, I ensure the train/val split stratification uses the correct 1D labels (already intended), and guarantee the submission probabilities are written in the exact `sample_submission.csv` column order to produce a valid `submission.csv`.'
- What this solution (achieved 0.45771) has done: 'I fix the protobuf/TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing the C++ protobuf implementation (and unsetting the Python one) before importing TensorFlow, which is the root cause of your current runtime failure. I keep your exact feature-extraction + concatenation + softmax-head training logic unchanged, but make the environment/Keras import path consistent and stable under TF 2.18 / Python 3.12. I also add a small safety fallback to locate the dataset under either `/kaggle/input/...` or `/kaggle/data/...` without changing your intended paths. These changes are primarily correctness/stability and should allow the same training to run end-to-end and produce a valid `submission.csv`, with score expected to improve back toward your target by ensuring the pipeline actually trains/inferes reliably.'
- What this solution (achieved 0.44817) has done: 'The immediate blocker is the TensorFlow/protobuf import crash; I fix it by forcing the pure-Python protobuf implementation *before* importing TensorFlow, which is the most reliable workaround for the `MessageFactory.GetPrototype` error in this Kaggle Python 3.12 environment. Then, to move log-loss down toward your target with minimal semantic change, I fix a silent but impactful preprocessing bug: your feature extractor was feeding **raw uint8 images** into Inception/Xception/NASNet `preprocess_input` functions without casting to float, which can significantly degrade predictions and calibration. I also make sure feature extraction runs in inference mode (`training=False`) for deterministic BatchNorm behavior, and keep the submission column order exactly matching `sample_submission.csv`. No changes to the model head architecture, loss, or the overall feature-extraction + softmax-head approach.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2, random, time, shutil, csv

plt.rcParams["figure.figsize"] = (10, 6)
sns.set_style("whitegrid")
pd.set_option("display.float_format", lambda x: "%.3f" % x)
pd.set_option("display.max_columns", None)

import warnings

warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    GlobalAveragePooling2D,
    Lambda,
    Dropout,
    InputLayer,
    Input,
)

from tqdm import tqdm
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    log_loss,
)

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

print("TF version:", tf.__version__)
print("tf.keras version:", keras.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_dir = "/kaggle/input/dog-breed-identification/train"
test_dir = "/kaggle/input/dog-breed-identification/test"
labels_path = "/kaggle/input/dog-breed-identification/labels.csv"
sample_sub_path = "/kaggle/input/dog-breed-identification/sample_submission.csv"

if not os.path.exists(train_dir):
    alt_train_dir = "/kaggle/data/dog-breed-identification/train"
    if os.path.exists(alt_train_dir):
        train_dir = alt_train_dir
if not os.path.exists(test_dir):
    alt_test_dir = "/kaggle/data/dog-breed-identification/test"
    if os.path.exists(alt_test_dir):
        test_dir = alt_test_dir
if not os.path.exists(labels_path):
    alt_labels_path = "/kaggle/data/dog-breed-identification/labels.csv"
    if os.path.exists(alt_labels_path):
        labels_path = alt_labels_path
if not os.path.exists(sample_sub_path):
    alt_sample_sub_path = "/kaggle/data/dog-breed-identification/sample_submission.csv"
    if os.path.exists(alt_sample_sub_path):
        sample_sub_path = alt_sample_sub_path

labels_dataframe = pd.read_csv(labels_path)
sample_df = pd.read_csv(sample_sub_path)

print(labels_dataframe.head())
print(sample_df.head())
print("Train rows:", len(labels_dataframe), " Test rows:", len(sample_df))



## === cell 2
submission_cols = list(sample_df.columns)
assert submission_cols[0] == "id"
dog_breeds = submission_cols[1:]  # exact required order
n_classes = len(dog_breeds)

class_to_num = {b: i for i, b in enumerate(dog_breeds)}
num_to_class = {i: b for b, i in class_to_num.items()}

missing = set(labels_dataframe["breed"].unique()) - set(dog_breeds)
print("n_classes:", n_classes, "missing_from_submission_cols:", len(missing))
if missing:
    print("Missing breeds (should be empty):", list(sorted(missing))[:10])




## === cell 3
def get_num_files(path):
    """Counts the number of files in a folder."""
    if not os.path.exists(path):
        return 0
    return sum([len(files) for r, d, files in os.walk(path)])


print("Data samples size: ", get_num_files(train_dir))
print("Test samples size: ", get_num_files(test_dir))




## === cell 4
def images_to_array(data_dir, labels_dataframe, img_size=(224, 224, 3)):
    """
    1- Read image samples from certain directory.
    2- Resize it, then stack them into one big numpy array.
    3- Read sample's label from the labels dataframe.
    4- One hot encode labels array.
    5- Shuffle Data and label arrays.
    """
    images_names = labels_dataframe["id"].values
    images_labels = labels_dataframe["breed"].values
    data_size = len(images_names)

    X = np.zeros([data_size, img_size[0], img_size[1], img_size[2]], dtype=np.uint8)
    y = np.zeros([data_size, 1], dtype=np.int32)

    target_hw = (img_size[0], img_size[1])

    for i in tqdm(range(data_size), desc="Loading train images"):
        image_name = images_names[i]
        img_path = os.path.join(data_dir, image_name + ".jpg")
        img_pixels = load_img(img_path, target_size=target_hw)
        X[i] = np.asarray(img_pixels, dtype=np.uint8)

        breed = images_labels[i]
        y[i, 0] = class_to_num[breed]

    y = to_categorical(y, num_classes=n_classes)

    ind = np.random.permutation(data_size)
    X = X[ind]
    y = y[ind]

    print("Output Data Size: ", X.shape)
    print("Output Label Size: ", y.shape)
    return X, y




## === cell 5
img_size = (300, 300, 3)

X, y = images_to_array(train_dir, labels_dataframe, img_size)




## === cell 6
def get_features(model_name, data_preprocessor, input_size, data):
    """
    1- Create a feature extractor to extract features from the data.
    2- Returns the extracted features.

    Fixes:
    - Cast uint8 -> float32 BEFORE preprocess_input (critical for correct scaling).
    - Run base_model in inference mode (training=False) for deterministic BatchNorm.
    """
    input_layer = Input(shape=input_size, dtype=tf.float32)

    def _pp(x):
        x = tf.cast(x, tf.float32)
        return data_preprocessor(x)

    preprocessed = Lambda(_pp)(input_layer)

    base_model = model_name(
        weights="imagenet", include_top=False, input_tensor=preprocessed
    )
    avg = GlobalAveragePooling2D()(base_model.output)

    feature_extractor = Model(inputs=input_layer, outputs=avg)
    feature_extractor.trainable = False

    data_f = data.astype(np.float32, copy=False)
    feature_maps = feature_extractor.predict(data_f, batch_size=64, verbose=1)
    print("Feature maps shape: ", feature_maps.shape)
    return feature_maps




## === cell 7
from tensorflow.keras.applications.inception_v3 import (
    InceptionV3,
    preprocess_input as inception_preprocess_input,
)
from tensorflow.keras.applications.xception import (
    Xception,
    preprocess_input as xception_preprocess_input,
)
from tensorflow.keras.applications.nasnet import (
    NASNetLarge,
    preprocess_input as nasnet_preprocess_input,
)

inception_features = get_features(InceptionV3, inception_preprocess_input, img_size, X)
xception_features = get_features(Xception, xception_preprocess_input, img_size, X)
nasnet_features = get_features(NASNetLarge, nasnet_preprocess_input, img_size, X)

final_features_6 = np.concatenate(
    [inception_features, xception_features, nasnet_features], axis=-1
)
print("Final feature maps shape", final_features_6.shape)

del X, inception_features, xception_features, nasnet_features



## === cell 8
y_indices = np.argmax(y, axis=1)

X_train, X_val, y_train, y_val, y_train_idx, y_val_idx = train_test_split(
    final_features_6,
    y,
    y_indices,
    test_size=0.2,
    stratify=y_indices,
    random_state=SEED,
)

class_weights = compute_class_weight(
    "balanced", classes=np.arange(n_classes), y=y_train_idx
)
class_weights_dict = {i: float(w) for i, w in enumerate(class_weights)}
print("Class weights computed.")



## === cell 9
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam

batch_size = 64
epochs = 1000

EarlyStop_callback = EarlyStopping(
    monitor="val_loss", verbose=1, mode="min", patience=15, restore_best_weights=True
)
my_callback = [EarlyStop_callback]

model_6 = keras.models.Sequential(
    [
        InputLayer(input_shape=X_train.shape[1:]),
        BatchNormalization(),
        Dropout(0.5),
        Dense(
            n_classes, activation="softmax", kernel_regularizer=regularizers.l1(0.001)
        ),
    ]
)

custom_optimizer = Adam(learning_rate=0.005)

model_6.compile(
    optimizer=custom_optimizer, loss="categorical_crossentropy", metrics=["Recall"]
)

history_6 = model_6.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_val, y_val),
    callbacks=my_callback,
    class_weight=class_weights_dict,
)



## === cell 10
val_pred = model_6.predict(X_val, batch_size=256, verbose=0)
val_true = np.argmax(y_val, axis=1)
val_loss = log_loss(val_true, val_pred, labels=np.arange(n_classes))
print(f"Validation Multi Class Log Loss: {val_loss:.5f}")



## === cell 11
y_all_indices = np.argmax(y, axis=1)
class_weights_all = compute_class_weight(
    "balanced", classes=np.arange(n_classes), y=y_all_indices
)
class_weights_all_dict = {i: float(w) for i, w in enumerate(class_weights_all)}

model_final = keras.models.Sequential(
    [
        InputLayer(input_shape=final_features_6.shape[1:]),
        BatchNormalization(),
        Dropout(0.5),
        Dense(
            n_classes, activation="softmax", kernel_regularizer=regularizers.l1(0.001)
        ),
    ]
)
model_final.compile(
    optimizer=Adam(learning_rate=0.005),
    loss="categorical_crossentropy",
    metrics=["Recall"],
)

best_epoch = (
    int(np.argmin(history_6.history["val_loss"]) + 1)
    if "val_loss" in history_6.history
    else 50
)
best_epoch = max(1, best_epoch)
print("Refit on full data for epochs:", best_epoch)

model_final.fit(
    final_features_6,
    y,
    batch_size=batch_size,
    epochs=best_epoch,
    verbose=1,
    class_weight=class_weights_all_dict,
)




## === cell 12
def images_to_array2(data_dir, labels_dataframe, img_size=(224, 224, 3)):
    """Load test images into numpy array."""
    images_names = labels_dataframe["id"].values
    data_size = len(images_names)
    X = np.zeros([data_size, img_size[0], img_size[1], 3], dtype=np.uint8)

    target_hw = (img_size[0], img_size[1])

    for i in tqdm(range(data_size), desc="Loading test images"):
        image_name = images_names[i]
        img_path = os.path.join(data_dir, image_name + ".jpg")
        img_pixels = load_img(img_path, target_size=target_hw)
        X[i] = np.asarray(img_pixels, dtype=np.uint8)

    print("Output Test Data Size: ", X.shape)
    return X


test_data = images_to_array2(test_dir, sample_df, img_size)



## === cell 13
test_inception = get_features(
    InceptionV3, inception_preprocess_input, img_size, test_data
)
test_xception = get_features(Xception, xception_preprocess_input, img_size, test_data)
test_nasnet = get_features(NASNetLarge, nasnet_preprocess_input, img_size, test_data)

test_features = np.concatenate([test_inception, test_xception, test_nasnet], axis=-1)
print("Final test feature maps shape", test_features.shape)

del test_data, test_inception, test_xception, test_nasnet



## === cell 14
y_pred = model_final.predict(test_features, batch_size=128, verbose=1)

eps = 1e-7
y_pred = np.clip(y_pred, eps, 1 - eps)
y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)



## === cell 15
submission = sample_df.copy()
if y_pred.shape[1] != len(dog_breeds):
    raise ValueError(
        f"Pred shape {y_pred.shape} does not match expected classes {len(dog_breeds)}"
    )

for j, breed in enumerate(dog_breeds):
    submission[breed] = y_pred[:, j]

out_path = "submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head())
print("Submission shape:", submission.shape)
