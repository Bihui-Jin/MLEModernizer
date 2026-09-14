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

3.8

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

6.02316

# 6. Current score

4.7829

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 4.78373) has done: 'I fix the Keras/TensorFlow import/protobuf crash by using `tf.keras` consistently and removing the standalone `keras` imports that trigger the `MessageFactory.GetPrototype` issue in this environment. I also remove notebook-only magic (`%matplotlib inline`), ensure `tqdm`/`train_test_split`/layers are actually imported, and update `ModelCheckpoint` to save in the required `.keras` format so the best model file is created and can be loaded. Finally, I make prediction use the same preprocessing as training (rescale 1/255) and build the submission by starting from `sample_submission.csv` to guarantee correct column order and presence of all 120 breeds, then write a valid `.csv` submission.'
- What this solution (achieved 4.78281) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by ensuring we never import the standalone `keras` package (Keras 3) and instead force `tf.keras` usage only, plus set `TF_USE_LEGACY_KERAS=1` before TensorFlow is imported. Then I keep the exact same model/training logic, but make the checkpoint monitor `val_loss` (aligned with the log-loss metric) so the saved “best” model is the one that should score closer to your higher-logloss target (i.e., slightly worse than the current overly-good 4.78). Finally, I keep the submission creation based on `sample_submission.csv` to guarantee correct columns/order and always write a valid `submission.csv`.'
- What this solution (achieved 4.78368) has done: 'You’re hitting the protobuf `MessageFactory.GetPrototype` crash before any training runs, which is caused by an incompatibility between TensorFlow 2.18 and protobuf 6 in this environment. The minimal fix is to force protobuf to use the pure-Python implementation (and keep legacy `tf.keras`) before importing TensorFlow, which avoids the failing C++ path. I also make the submission-building cell robust to any rare missing/extra class-column mismatch by aligning predictions strictly to the `sample_submission.csv` column order (score-neutral, but prevents invalid submissions). Core model/training logic is unchanged, so your score should remain in the same neighborhood, just with the pipeline running end-to-end and writing a valid `submission.csv`.'
- What this solution (achieved 4.78393) has done: 'I fix the protobuf/TensorFlow crash by setting the required environment variables *before* any TensorFlow-related import happens (including indirect imports), and by importing `tensorflow` earlier than anything that might pull in standalone `keras`. I also make the import section robust by removing unused `keras`/plot imports that can indirectly trigger the same protobuf path, while keeping your training loop, model architecture, generators, and loss exactly the same. Finally, I keep the submission creation anchored to `sample_submission.csv` to guarantee correct column order and a valid `submission.csv` output. These changes are score-neutral in intent (they primarily unblock execution and ensure a valid CSV is written).'
- What this solution (achieved 4.78365) has done: 'I fix the protobuf/TensorFlow crash that currently prevents the notebook from running by forcing TensorFlow to use the pure-Python protobuf implementation and legacy tf.keras *before* TensorFlow (or anything that imports it indirectly) is loaded. I also remove the standalone `keras`/`tensorflow.keras.preprocessing` dependency for image loading by switching to `tf.keras.utils.load_img/img_to_array`, which avoids the code path that triggers the `MessageFactory.GetPrototype` error in this environment while keeping the same preprocessing semantics. These changes are execution/stability fixes and are intended to be score-neutral (your model, augmentation, training loop, and submission alignment remain the same). The script still write a valid `submission.csv` using `sample_submission.csv` to guarantee correct column order and presence.'
- What this solution (achieved 4.78323) has done: 'We need to stop the `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` crash that occurs during TensorFlow import in this Kaggle environment (TF 2.18 + protobuf 6). The minimal reliable fix is to ensure the pure-Python protobuf implementation is used and to force-load `google.protobuf.message_factory` before importing TensorFlow, so TensorFlow does not hit the incompatible compiled-protobuf path. I also remove the `tensorflow.keras.preprocessing.image` import (it can indirectly trigger problematic protobuf paths) and keep image loading via `tf.keras.utils.load_img/img_to_array` (same semantics). These changes are execution/stability-only and keep your model/training/submission logic intact, producing the same `submission.csv` format.'
- What this solution (achieved 4.78245) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing TensorFlow to use the pure-Python protobuf runtime before any protobuf/TensorFlow modules are loaded, and by explicitly importing `google.protobuf.internal.api_implementation` to ensure the Python implementation is selected. I also remove the `tensorflow.keras.preprocessing.image` import (it can trigger the problematic protobuf path) while keeping the exact same image loading semantics via `tf.keras.utils.load_img/img_to_array`. These changes are execution/stability-only and should be score-neutral (your model, augmentation, training loop, and submission alignment remain the same). The script still write a valid `submission.csv` using `sample_submission.csv` to guarantee correct column order.'
- What this solution (achieved 4.78391) has done: 'We need to unblock the TensorFlow import crash (`MessageFactory.GetPrototype`) while keeping your model/training/prediction logic unchanged so the score stays in the same neighborhood (and thus closer to your higher-logloss target). The most reliable minimal fix in this TF 2.18 + protobuf 6 environment is to force protobuf to use the pure-Python implementation and also force `TF_USE_LEGACY_KERAS=1`, then import TensorFlow only after those settings are applied. Additionally, `tensorflow.keras.preprocessing.image.ImageDataGenerator` can indirectly trigger problematic protobuf/keras paths here, so we keep the same augmentation semantics but switch to `tf.keras.preprocessing.image.ImageDataGenerator` via `tf.keras` (still the same API) and avoid any standalone `keras` imports. Finally, we keep the submission creation anchored to `sample_submission.csv` to guarantee the required column order and a valid `submission.csv`.'
- What this solution (achieved 4.78291) has done: 'I fix the runtime crash happening at TensorFlow import (`MessageFactory.GetPrototype`) by ensuring protobuf’s Python implementation is selected early and by avoiding any standalone `keras` imports (keeping `tf.keras` only), which is the minimal unblocker in this TF 2.18 + protobuf 6 environment. I also remove `matplotlib` imports/usage (they are non-essential and can indirectly trigger problematic import chains in some Kaggle images), while keeping your model, generators, training loop, and prediction logic unchanged. Finally, I keep submission creation anchored to `sample_submission.csv` (same as you already do) to guarantee correct column order and a valid `submission.csv`.'
- What this solution (achieved 4.7829) has done: 'We fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by enforcing protobuf’s pure-Python implementation and legacy `tf.keras` before any TensorFlow-related import, and by eagerly importing protobuf’s message factory to avoid the incompatible compiled path in this Kaggle image. This is an execution/stability fix only and does not change your model, training loop, preprocessing, or submission formatting. We also remove the `EarlyStopping` callback from the callbacks list (while keeping it defined) so training always runs the full configured `Epochs=50`, which should worsen (increase) log loss toward your higher target score since your current score is “too good” for the target band. The pipeline still write a valid `submission.csv` using `sample_submission.csv` for correct column order.'

# 9. Code solution

## === cell 0
labels_csv = "../input/dog-breed-identification/labels.csv"
sample_submission_csv = "../input/dog-breed-identification/sample_submission.csv"

jpg_train = "../input/dog-breed-identification/train/{}.jpg"
jpg_test = "../input/dog-breed-identification/test/{}.jpg"

im_resize = 64  # image size
num_class = 120
batch_size = 32
Epochs = 50



## === cell 1
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import google.protobuf.internal.api_implementation as _api_impl  # noqa: F401
import google.protobuf.message_factory as _message_factory  # noqa: F401

import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import load_img, img_to_array

ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

np.random.seed(42)
tf.random.set_seed(42)

print("TF version:", tf.__version__)
print("Using legacy tf.keras:", os.environ.get("TF_USE_LEGACY_KERAS"))




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
def gen_graph(history, title):
    return




## === cell 3
df_train = pd.read_csv(labels_csv)
df_test = pd.read_csv(sample_submission_csv)



## === cell 4
labels = df_train["breed"]
one_hot = pd.get_dummies(labels, sparse=False)
one_hot_labels = np.asarray(one_hot, dtype=np.float32)

assert (
    one_hot_labels.shape[1] == num_class
), f"Expected {num_class} classes, got {one_hot_labels.shape[1]}"



## === cell 5
x_train = []
y_train = []
x_test = []



## === cell 6
i = 0
for f, breed in tqdm(df_train.values, total=len(df_train)):
    img = load_img(jpg_train.format(f), target_size=(im_resize, im_resize))
    img_resized = img_to_array(img)
    x_train.append(img_resized)
    y_train.append(one_hot_labels[i])
    i += 1



## === cell 7
for f in tqdm(df_test["id"].values, total=len(df_test)):
    img = load_img(jpg_test.format(f), target_size=(im_resize, im_resize))
    img_resized = img_to_array(img)
    x_test.append(img_resized)



## === cell 8
X_train, X_valid, Y_train, Y_valid = train_test_split(
    x_train, y_train, shuffle=True, test_size=0.2, random_state=42
)



## === cell 9
del x_train, y_train, df_train



## === cell 10
train_datagen = ImageDataGenerator(
    rotation_range=15, rescale=1.0 / 255.0, horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_datagen.flow(
    np.array(X_train), np.array(Y_train), batch_size=batch_size
)
test_generator = test_datagen.flow(
    np.array(X_valid), np.array(Y_valid), batch_size=batch_size * 5
)



## === cell 11
model = Sequential()

model.add(
    Conv2D(
        32,
        (3, 3),
        padding="same",
        input_shape=(im_resize, im_resize, 3),
        activation="relu",
    )
)
model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.20))

model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.20))

model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.20))

model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.30))

model.add(Dense(num_class, activation="softmax"))



## === cell 12
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=[tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy")],
)



## === cell 13
print(model.summary())



## === cell 14
earlystop = EarlyStopping(
    monitor="val_loss", min_delta=0.0, patience=5, restore_best_weights=False
)

checkpoint_path = "model_best.keras"
checkpoint_callback = ModelCheckpoint(
    checkpoint_path, monitor="val_loss", mode="min", save_best_only=True, verbose=1
)



## === cell 15
history = model.fit(
    train_generator,
    callbacks=[checkpoint_callback],
    epochs=Epochs,
    steps_per_epoch=len(train_generator),
    validation_data=test_generator,
    validation_steps=len(test_generator),
    verbose=1,
)



## === cell 16
gen_graph(history, "график точности")



## === cell 17
if os.path.exists(checkpoint_path):
    model = tf.keras.models.load_model(checkpoint_path)



## === cell 18
x_test_arr = np.array(x_test, dtype=np.float32) / 255.0
preds = model.predict(x_test_arr, batch_size=batch_size, verbose=1)



## === cell 19
sub = pd.read_csv(sample_submission_csv)
breed_cols = sub.columns[1:]  # all breed columns in correct order

preds_df = pd.DataFrame(preds, columns=one_hot.columns)

preds_aligned = preds_df.reindex(columns=breed_cols, fill_value=1e-8).to_numpy(
    dtype=np.float32
)
preds_aligned = preds_aligned / np.clip(
    preds_aligned.sum(axis=1, keepdims=True), 1e-12, None
)

sub.loc[:, breed_cols] = preds_aligned
sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
