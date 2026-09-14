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

4.80775

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 4.85625) has done: 'I set the protobuf implementation flag before importing TensorFlow to avoid the `MessageFactory` error, and I fix the test‑image loader so it skips directories and builds the correct path list. This ensures the feature extraction runs on the proper 1024 test images, making the prediction array length match the submission DataFrame and allowing the notebook to finish and write a valid `submission.csv`.'
- What this solution (achieved 4.89378) has done: 'I set the protobuf flag before any imports, cast feature and label arrays to float32, lower the dropout rate (so the model can learn), and ensure the training step runs without “None values” errors. These fixes keep the original pipeline intact while allowing the model to train and produce a valid `submission.csv`, moving the score closer to the target.'
- What this solution (achieved 4.87083) has done: 'I fixed the root cause of the “None values not supported” error by correcting how labels are converted to categorical format. The label vector is now built as a 1‑D integer array before calling `to_categorical`, which yields a proper (samples, n_classes) target matrix. This change restores successful model training and allows the pipeline to finish and write a valid `submission.csv`. No other logic was altered.'
- What this solution (achieved 4.88756) has done: 'I add a safety conversion that replaces any NaN values in the extracted feature arrays and label matrices (preventing the “None values not supported” error), lower the dropout rate slightly, and insert a small hidden dense layer before the final soft‑max so the model can learn a bit richer representation. These adjustments keep the overall pipeline unchanged while fixing the runtime crash and should improve the validation log‑loss, moving the score closer to the target.'
- What this solution (achieved 4.80775) has done: 'I replace the standalone `keras` imports with the unified `tensorflow.keras` equivalents (including the InceptionV3 application) to resolve the protobuf import error and the “None values not supported” issue that arose from mismatched TensorFlow/Keras versions. No other logic is changed, preserving the original model and workflow while ensuring the script runs end‑to‑end and writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import gc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from tqdm.autonotebook import tqdm

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import (
    Dropout,
    Dense,
    GlobalAveragePooling2D,
    Input,
    Lambda,
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
    image_labels = label_dataframe["breed"]
    images = np.zeros(
        [len(label_dataframe), target_size[0], target_size[1], target_size[2]],
        dtype=np.uint8,
    )
    y_int = np.zeros(len(label_dataframe), dtype=np.int32)

    for ix, image_name in enumerate(tqdm(label_dataframe["id"].values)):
        img_path = os.path.join(directory, image_name + ".jpg")
        img = load_img(img_path, target_size=target_size)
        img = np.array(img, dtype=np.uint8)
        images[ix] = img
        del img

        dog_breed = image_labels.iloc[ix]
        y_int[ix] = class_to_num[dog_breed]

    y = to_categorical(y_int, num_classes=n_classes)
    return images, y




## === cell 4
import time

t = time.time()
X, y = images_to_array("/kaggle/input/dog-breed-identification/train", labels[:])
print("runtime in seconds: {}".format(time.time() - t))




## === cell 5
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

y_train = np.nan_to_num(y_train.astype("float32"), nan=0.0)
y_test = np.nan_to_num(y_test.astype("float32"), nan=0.0)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)




## === cell 6
lrr = ReduceLROnPlateau(
    monitor="val_accuracy", factor=0.01, patience=3, min_lr=1e-5, verbose=1
)

EarlyStop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)




## === cell 7
batch_size = 128
epochs = 30  # a few more epochs for better convergence
learn_rate = 0.001
sgd = SGD(learning_rate=learn_rate, momentum=0.9, nesterov=False)
adam = Adam(
    learning_rate=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=None, amsgrad=False
)




## === cell 8
img_size = (331, 331, 3)


def get_features(model_name, model_preprocessor, input_size, data):
    input_layer = Input(input_size)
    preprocessor = Lambda(model_preprocessor)(input_layer)
    base_model = model_name(
        weights="imagenet", include_top=False, input_shape=input_size
    )(preprocessor)
    avg = GlobalAveragePooling2D()(base_model)
    feature_extractor = Model(inputs=input_layer, outputs=avg)

    feature_maps = feature_extractor.predict(data, verbose=1)
    print("Feature maps shape: ", feature_maps.shape)
    return feature_maps




## === cell 9
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

inception_preprocessor = preprocess_input
inception_features = get_features(
    InceptionV3, inception_preprocessor, img_size, X_train
)

print("Inception feature maps shape", inception_features.shape)




## === cell 10
del X, X_train
gc.collect()

inception_features = np.nan_to_num(inception_features.astype("float32"), nan=0.0)




## === cell 11
model = Sequential()
model.add(Dropout(0.15, input_shape=(inception_features.shape[1],)))
model.add(Dense(256, activation="relu"))
model.add(Dense(n_classes, activation="softmax"))

model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])




## === cell 12
history = model.fit(
    inception_features,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    callbacks=[lrr, EarlyStop],
    verbose=2,
)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/1013819864.py in <cell line: 0>()
----> 1 history = model.fit(
      2     inception_features,
      3     y_train,
      4     batch_size=batch_size,
      5     epochs=epochs,

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

## === cell 13
model.summary()




## === cell 14
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epoch_range = range(1, len(acc) + 1)

plt.plot(epoch_range, acc, label="Train accuracy")
plt.plot(epoch_range, val_acc, label="Val accuracy")
plt.title("Training & validation accuracy")
plt.legend()
plt.show()

plt.figure()
plt.plot(epoch_range, loss, label="Train loss")
plt.plot(epoch_range, val_loss, label="Val loss")
plt.title("Training & validation loss")
plt.legend()
plt.show()




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3939918645.py in <cell line: 0>()
----> 1 acc = history.history["accuracy"]
      2 val_acc = history.history["val_accuracy"]
      3 loss = history.history["loss"]
      4 val_loss = history.history["val_loss"]
      5 epoch_range = range(1, len(acc) + 1)

NameError: name 'history' is not defined

## === cell 15
del inception_features
gc.collect()




## === cell 16
def extact_features(data):
    feats = get_features(InceptionV3, inception_preprocessor, img_size, data)
    print("Inception feature maps shape", feats.shape)
    return np.nan_to_num(feats.astype("float32"), nan=0.0)


test_features = extact_features(X_test)




## === cell 17
y_pred = model.predict(test_features)




## === cell 18
from sklearn.metrics import accuracy_score

y_test_indices = np.argmax(y_test, axis=1)
y_pred_indices = np.argmax(y_pred, axis=1)

print(
    "Validation accuracy (raw predictions):",
    accuracy_score(y_test_indices, y_pred_indices),
)




## === cell 19
def images_to_array_test(test_path, img_size=(331, 331, 3)):
    filenames = [f for f in os.listdir(test_path) if f.lower().endswith(".jpg")]
    data_size = len(filenames)
    images = np.zeros([data_size, img_size[0], img_size[1], 3], dtype=np.uint8)

    for ix, fname in enumerate(tqdm(filenames)):
        img_path = os.path.join(test_path, fname)
        img = load_img(img_path, target_size=img_size)
        img = np.array(img, dtype=np.uint8)
        images[ix] = img
        del img
    print("Output Data Size: ", images.shape)
    return images, filenames


test_data, test_filenames = images_to_array_test(
    "/kaggle/input/dog-breed-identification/test/", img_size
)




## === cell 20
test_features = extact_features(test_data)




## === cell 21
test_pred = model.predict(test_features)




## === cell 22
preds_df = pd.DataFrame(columns=["id"] + list(classes))
preds_df["id"] = [os.path.splitext(fname)[0] for fname in test_filenames]
preds_df.loc[:, list(classes)] = test_pred
preds_df.to_csv("submission.csv", index=False)
print("Submission written to submission.csv with shape", preds_df.shape)
