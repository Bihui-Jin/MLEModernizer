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

0.34011

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 4.85625) has done: 'I set the protobuf implementation flag before importing TensorFlow to avoid the `MessageFactory` error, and I fix the test‑image loader so it skips directories and builds the correct path list. This ensures the feature extraction runs on the proper 1024 test images, making the prediction array length match the submission DataFrame and allowing the notebook to finish and write a valid `submission.csv`.'
- What this solution (achieved 4.89378) has done: 'I set the protobuf flag before any imports, cast feature and label arrays to float32, lower the dropout rate (so the model can learn), and ensure the training step runs without “None values” errors. These fixes keep the original pipeline intact while allowing the model to train and produce a valid `submission.csv`, moving the score closer to the target.'
- What this solution (achieved 4.87083) has done: 'I fixed the root cause of the “None values not supported” error by correcting how labels are converted to categorical format. The label vector is now built as a 1‑D integer array before calling `to_categorical`, which yields a proper (samples, n_classes) target matrix. This change restores successful model training and allows the pipeline to finish and write a valid `submission.csv`. No other logic was altered.'
- What this solution (achieved 4.88756) has done: 'I add a safety conversion that replaces any NaN values in the extracted feature arrays and label matrices (preventing the “None values not supported” error), lower the dropout rate slightly, and insert a small hidden dense layer before the final soft‑max so the model can learn a bit richer representation. These adjustments keep the overall pipeline unchanged while fixing the runtime crash and should improve the validation log‑loss, moving the score closer to the target.'
- What this solution (achieved 4.80775) has done: 'I replace the standalone `keras` imports with the unified `tensorflow.keras` equivalents (including the InceptionV3 application) to resolve the protobuf import error and the “None values not supported” issue that arose from mismatched TensorFlow/Keras versions. No other logic is changed, preserving the original model and workflow while ensuring the script runs end‑to‑end and writes a valid `submission.csv`.'
- What this solution (achieved 4.83579) has done: 'I replace any possible `None` or `nan` values in the feature and label arrays with zeros, explicitly cast the label matrix to `float32`, and add a small sanity‑check before fitting so the model receives clean tensors. These fixes resolve the “None values not supported” error while keeping the original architecture and training logic untouched, allowing the script to run end‑to‑end and produce a valid `submission.csv`.'
- What this solution (achieved 0.31483) has done: 'I move the protobuf‑environment flag to the very top before any imports, switch the import order so TensorFlow loads first, and add a small robustness check that guarantees the feature and label arrays contain only numeric values. I also lower dropout slightly, add an extra dense layer, and reduce the learning rate to let the model train a bit better, which should improve the log‑loss toward the target while keeping the original workflow unchanged. The script now runs end‑to‑end and writes a proper `submission.csv`.'
- What this solution (achieved 0.32739) has done: 'I set the protobuf flag before any imports, adjust the learning‑rate‑schedule to monitor validation loss, stratify the train/validation split, and allow more training epochs (60) so the model can improve a bit without changing its core architecture. These small fixes keep the original pipeline intact while addressing the earlier import error and nudging the log‑loss toward the target.'
- What this solution (achieved 0.33159) has done: 'I keep the original pipeline but compute validation features ahead of training and pass them to model.fit via validation_data instead of using an internal validation_split. This gives the early‑stopping callbacks a true validation signal, which should modestly improve the log‑loss and move the score closer to the target while preserving all core logic. I also renumber the cells to start at 1 as required.'
- What this solution (achieved 0.34011) has done: 'I keep the original workflow but fix the protobuf import issue by ensuring the environment flag is set before any TensorFlow import and add deterministic seeds. To nudge the log‑loss toward the target, I lower the learning rate, allow more epochs, give early‑stopping a longer patience, and slightly enlarge the classifier (add a 512‑unit dense layer) while preserving the overall architecture.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import gc
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.autonotebook import tqdm
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
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
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

tf.random.set_seed(42)
np.random.seed(42)



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
t = time.time()
X, y = images_to_array("/kaggle/input/dog-breed-identification/train", labels[:])
print("runtime in seconds: {}".format(time.time() - t))



## === cell 5
y_int_labels = np.argmax(y, axis=1)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y_int_labels
)

y_train = np.nan_to_num(y_train.astype("float32"), nan=0.0)
y_val = np.nan_to_num(y_val.astype("float32"), nan=0.0)

print("Train/val shapes:", X_train.shape, y_train.shape, X_val.shape, y_val.shape)



## === cell 6
lrr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
)
early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)



## === cell 7
batch_size = 128
epochs = 120  # more epochs, early stopping will stop earlier if needed
learn_rate = 1e-4  # lower learning rate for finer convergence
adam = Adam(learning_rate=learn_rate)



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
inception_features = get_features(InceptionV3, preprocess_input, img_size, X_train)

inception_features_val = get_features(InceptionV3, preprocess_input, img_size, X_val)

inception_features = np.nan_to_num(inception_features.astype("float32"), nan=0.0)
inception_features_val = np.nan_to_num(
    inception_features_val.astype("float32"), nan=0.0
)



## === cell 10
del X, X_train
gc.collect()



## === cell 11
model = Sequential()
model.add(Dropout(0.10, input_shape=(inception_features.shape[1],)))
model.add(Dense(512, activation="relu"))  # added larger hidden layer
model.add(Dense(256, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(n_classes, activation="softmax"))

model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])



## === cell 12
assert not np.isnan(inception_features).any(), "NaNs found in features"
assert not np.isnan(y_train).any(), "NaNs found in labels"

history = model.fit(
    inception_features,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(inception_features_val, y_val),
    callbacks=[lrr, early_stop],
    verbose=2,
)



## === cell 13
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



## === cell 14
del inception_features, inception_features_val
gc.collect()




## === cell 15
def extract_features(data):
    feats = get_features(InceptionV3, preprocess_input, img_size, data)
    print("Inception feature maps shape", feats.shape)
    return np.nan_to_num(feats.astype("float32"), nan=0.0)




## === cell 16
test_features_val = extract_features(X_val)



## === cell 17
y_pred_val = model.predict(test_features_val)



## === cell 18
from sklearn.metrics import accuracy_score

y_val_indices = np.argmax(y_val, axis=1)
y_pred_val_indices = np.argmax(y_pred_val, axis=1)

print(
    "Validation accuracy (raw predictions):",
    accuracy_score(y_val_indices, y_pred_val_indices),
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
test_features = extract_features(test_data)



## === cell 21
test_pred = model.predict(test_features)



## === cell 22
preds_df = pd.DataFrame(columns=["id"] + list(classes))
preds_df["id"] = [os.path.splitext(fname)[0] for fname in test_filenames]
preds_df.loc[:, list(classes)] = test_pred
preds_df.to_csv("submission.csv", index=False)
print("Submission written to submission.csv with shape", preds_df.shape)
