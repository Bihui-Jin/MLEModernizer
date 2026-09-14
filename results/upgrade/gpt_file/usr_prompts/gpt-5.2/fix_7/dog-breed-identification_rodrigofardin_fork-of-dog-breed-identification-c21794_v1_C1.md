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

0.28124

# 6. Current score

4.89637

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 4.8698) has done: 'I fix the protobuf/Keras import crash by forcing the Python protobuf implementation early, so TensorFlow/Keras load reliably in this Kaggle environment. Then I fix the training crash by using the correct validation metric key (`val_accuracy` instead of the nonexistent `val_acc`) so `ReduceLROnPlateau` works and `model.fit()` receives valid callbacks. Finally, I fix test image loading and submission alignment by filtering only `.jpg` files, sorting filenames deterministically, and building the submission from `sample_submission.csv` so the column order and row count match the required format, producing a valid `submission.csv`.'
- What this solution (achieved 4.85773) has done: 'I fix the protobuf/TensorFlow import crash by forcing the pure-Python protobuf runtime and (critically) setting `protobuf` to the compatible implementation before importing TensorFlow/Keras. Then I fix the `model.fit()` crash (`None values not supported`) by making sure images are converted to real numeric arrays (via `img_to_array`) and by correcting the feature-extractor construction so we don’t accidentally call a KerasTensor like a layer. Finally, I keep the rest of the pipeline intact and ensure predictions are properly aligned to `sample_submission.csv` columns and written as a valid `submission.csv`.'
- What this solution (achieved 4.89636) has done: 'I fix the TensorFlow/Keras import crash by pinning protobuf to the pure-Python implementation early and disabling the C++ fast-path, which avoids the `MessageFactory.GetPrototype` error in this environment. Then I fix the `None values not supported` training crash by ensuring the feature extractor is run in inference mode and returns a real NumPy array with a stable float dtype (and that the preprocessing Lambda has a defined output shape). Finally, I keep your modeling/training logic intact but ensure predictions are properly normalized (sums to 1) and aligned to `sample_submission.csv` columns/row order, then write a valid `submission.csv`.'
- What this solution (achieved 4.89637) has done: 'I fix two execution blockers that prevent end-to-end training/inference: (1) the protobuf/TensorFlow import crash by setting the required environment variables before *any* TensorFlow/Keras-related import, and (2) the `None values not supported` error during `model.fit()` by ensuring the feature extractor produces a concrete NumPy array (and by feeding `y_train` with matching dtype/shape). These changes keep your core approach identical (InceptionV3 frozen feature extractor + GAP + Dropout/Dense classifier, same loss/optimizer/training loop). I also keep the submission construction aligned to `sample_submission.csv` and sorted test filenames so the output is valid and stable. With training actually running (instead of failing and leaving an effectively untrained model), the logloss should move substantially toward the target.'
- What this solution (achieved 4.89636) has done: 'I fix the protobuf/TensorFlow import crash by forcing a compatible protobuf runtime *before* importing TensorFlow/Keras, which is currently preventing any training from running. Then I fix the `None values not supported` training error by ensuring the feature extractor outputs a concrete, finite NumPy array (and by making the preprocessing Lambda explicitly produce a tensor), so `model.fit()` always receives valid numeric inputs. Finally, I keep your InceptionV3-frozen-features + GAP + Dropout/Dense classifier logic intact, but ensure the class ordering exactly matches `sample_submission.csv` so your probability columns align with the required breed order, which should significantly reduce logloss toward the target.'
- What this solution (achieved 4.89637) has done: 'I fix the TensorFlow/protobuf import crash by setting the required environment variables before any TensorFlow/Keras import and by additionally disabling TF’s C++ protobuf path, which is the root cause of the `MessageFactory.GetPrototype` error in this environment. Then I fix the `None values not supported` training failure by ensuring the feature extractor returns a concrete NumPy array (and by guarding against accidental `None`/object arrays) and by making the label array shape exactly `(n_samples, n_classes)` (not `(n_samples, 1, n_classes)`). Finally, I keep your exact modeling approach (frozen InceptionV3 features + Dropout + Dense softmax) and ensure the submission is built from `sample_submission.csv` ordering and sorted test filenames, producing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_C"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_PROTOBUF_IMPLEMENTATION"] = "python"

import gc
import time
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from tqdm.autonotebook import tqdm

import tensorflow as tf
from keras import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam, SGD
from keras.layers import Dense, Dropout, Lambda, Input, GlobalAveragePooling2D
from keras.utils import to_categorical
from keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model

print("Python:", os.sys.version.split()[0])
print("TF:", tf.__version__)

tf.keras.utils.set_random_seed(42)
np.random.seed(42)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
labels = pd.read_csv("/kaggle/input/dog-breed-identification/labels.csv")
labels.head()




## === cell 2
sample_sub = pd.read_csv("/kaggle/input/dog-breed-identification/sample_submission.csv")
classes = [c for c in sample_sub.columns if c != "id"]
n_classes = len(classes)
print("Total unique breed {}".format(n_classes))

class_to_num = dict(zip(classes, range(n_classes)))

unknown = set(labels["breed"].unique()) - set(classes)
if unknown:
    raise ValueError(
        f"Found breeds in labels not present in sample_submission columns: {sorted(list(unknown))[:10]}"
    )




## === cell 3
input_shape = (331, 331, 3)


def images_to_array(directory, label_dataframe, target_size=input_shape):
    image_labels = label_dataframe["breed"].values
    images = np.zeros(
        [len(label_dataframe), target_size[0], target_size[1], target_size[2]],
        dtype=np.uint8,
    )
    y = np.zeros((len(label_dataframe),), dtype=np.int32)

    for ix, image_name in enumerate(tqdm(label_dataframe["id"].values)):
        img_path = os.path.join(directory, image_name + ".jpg")
        img = load_img(img_path, target_size=target_size)
        img_arr = img_to_array(img, dtype="uint8")
        images[ix] = img_arr
        del img, img_arr

        dog_breed = image_labels[ix]
        y[ix] = class_to_num[dog_breed]

    y = to_categorical(y, num_classes=n_classes)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 2 or y.shape[1] != n_classes:
        raise ValueError(
            f"Unexpected y shape after one-hot: {y.shape}, expected (n,{n_classes})"
        )
    return images, y




## === cell 4
t = time.time()
X, y = images_to_array("/kaggle/input/dog-breed-identification/train", labels[:])
print("runtime in seconds: {}".format(time.time() - t))
print("X:", X.shape, X.dtype, "y:", y.shape, y.dtype)




## === cell 5
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=np.argmax(y, axis=1)
)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)




## === cell 6
lrr = ReduceLROnPlateau(
    monitor="val_accuracy", factor=0.01, patience=3, min_lr=1e-5, verbose=1
)
EarlyStop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)




## === cell 7
batch_size = 128
epochs = 25
learn_rate = 0.001
sgd = SGD(learning_rate=learn_rate, momentum=0.9, nesterov=False)
adam = Adam(
    learning_rate=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False
)




## === cell 8
img_size = (331, 331, 3)


def get_features(model_name, model_preprocessor, input_size, data):
    def _preprocess(x):
        x = tf.cast(x, tf.float32)
        return model_preprocessor(x)

    input_layer = Input(input_size)
    preprocessed = Lambda(_preprocess, name="preprocess")(input_layer)

    base_model = model_name(
        weights="imagenet", include_top=False, input_shape=input_size
    )
    base_model.trainable = False

    base_out = base_model(preprocessed, training=False)
    avg = GlobalAveragePooling2D()(base_out)
    feature_extractor = Model(inputs=input_layer, outputs=avg)

    data_np = np.asarray(data, dtype=np.float32)
    if data_np.dtype == object:
        raise ValueError(
            "Input data converted to object dtype; image loading likely failed."
        )

    feature_maps = feature_extractor.predict(data_np, verbose=1)
    if feature_maps is None:
        raise ValueError("Feature extraction returned None.")
    feature_maps = np.asarray(feature_maps, dtype=np.float32)

    if not np.isfinite(feature_maps).all():
        raise ValueError("Feature extraction produced non-finite values (NaN/Inf).")

    print("Feature maps shape: ", feature_maps.shape, feature_maps.dtype)
    return feature_maps




## === cell 9
from keras.applications.inception_v3 import InceptionV3, preprocess_input

inception_preprocessor = preprocess_input
inception_features = get_features(
    InceptionV3, inception_preprocessor, img_size, X_train
)
print("Inception feature maps shape", inception_features.shape)




## === cell 10
del X, X_train
gc.collect()




## === cell 11
y_train_fit = np.asarray(y_train, dtype=np.float32)
if y_train_fit.ndim != 2 or y_train_fit.shape[1] != n_classes:
    raise ValueError(
        f"Unexpected y_train_fit shape: {y_train_fit.shape}, expected (n,{n_classes})"
    )
if not np.isfinite(y_train_fit).all():
    raise ValueError("y_train_fit contains NaN/Inf.")

model = Sequential()
model.add(Dropout(0.7, input_shape=(inception_features.shape[1],)))
model.add(Dense(n_classes, activation="softmax"))

model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(
    inception_features,
    y_train_fit,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    callbacks=[lrr, EarlyStop],
    verbose=1,
)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/312814411.py in <cell line: 0>()
     14 model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])
     15 
---> 16 history = model.fit(
     17     inception_features,
     18     y_train_fit,

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

## === cell 12
model.summary()




## === cell 13
acc = history.history.get("accuracy", [])
val_acc = history.history.get("val_accuracy", [])
loss = history.history.get("loss", [])
val_loss = history.history.get("val_loss", [])
ep = range(1, len(loss) + 1)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(ep, acc, label="Train accuracy")
plt.plot(ep, val_acc, label="Val accuracy")
plt.title("Training & validation accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(ep, loss, label="Train loss")
plt.plot(ep, val_loss, label="Val loss")
plt.title("Training & validation loss")
plt.legend()
plt.tight_layout()
plt.show()




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1556337481.py in <cell line: 0>()
----> 1 acc = history.history.get("accuracy", [])
      2 val_acc = history.history.get("val_accuracy", [])
      3 loss = history.history.get("loss", [])
      4 val_loss = history.history.get("val_loss", [])
      5 ep = range(1, len(loss) + 1)

NameError: name 'history' is not defined

## === cell 14
del inception_features
gc.collect()




## === cell 15
def extact_features(data):
    inception_features_local = get_features(
        InceptionV3, inception_preprocessor, img_size, data
    )
    print("Inception feature maps shape", inception_features_local.shape)
    return inception_features_local


test_features_holdout = extact_features(X_test)




## === cell 16
y_pred_holdout = model.predict(test_features_holdout, verbose=1)




## === cell 17
from sklearn.metrics import accuracy_score

y_test_indices = np.argmax(y_test, axis=1)
y_pred_indices = np.argmax(y_pred_holdout, axis=1)

print("Holdout accuracy:", accuracy_score(y_test_indices, y_pred_indices))




## === cell 18
def images_to_array_test(test_path, img_size=(331, 331, 3)):
    file_names = sorted(
        [f for f in os.listdir(test_path) if f.lower().endswith(".jpg")]
    )
    test_filenames = [os.path.join(test_path, fname) for fname in file_names]

    data_size = len(test_filenames)
    images = np.zeros([data_size, img_size[0], img_size[1], 3], dtype=np.uint8)

    for ix, img_path in enumerate(tqdm(test_filenames)):
        img = load_img(img_path, target_size=img_size)
        img_arr = img_to_array(img, dtype="uint8")
        images[ix] = img_arr
        del img, img_arr

    print("Output Data Size: ", images.shape)
    return images, file_names


test_path = "/kaggle/input/dog-breed-identification/test/"
test_data, test_file_names = images_to_array_test(test_path, img_size)




## === cell 19
test_features = extact_features(test_data)




## === cell 20
del test_data
gc.collect()




## === cell 21
pred = model.predict(test_features, verbose=1)
pred = np.asarray(pred, dtype=np.float64)

pred = np.clip(pred, 1e-15, 1.0)
pred = pred / pred.sum(axis=1, keepdims=True)

print(
    "pred shape:",
    pred.shape,
    "row_sum[min,max]:",
    pred.sum(axis=1).min(),
    pred.sum(axis=1).max(),
)

sample_sub = pd.read_csv("/kaggle/input/dog-breed-identification/sample_submission.csv")
breed_cols = [c for c in sample_sub.columns if c != "id"]

test_ids = [os.path.splitext(f)[0] for f in test_file_names]

assert pred.shape[0] == len(
    test_ids
), f"Pred rows {pred.shape[0]} != test ids {len(test_ids)}"
assert pred.shape[1] == len(
    breed_cols
), f"Pred cols {pred.shape[1]} != breed cols {len(breed_cols)}"

submission = pd.DataFrame(pred, columns=breed_cols)
submission.insert(0, "id", test_ids)
submission = submission[["id"] + breed_cols]

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
