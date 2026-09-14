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

5.57999

# 6. Current score

4.66913

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 4.78552) has done: 'I fix the Keras/TensorFlow import incompatibilities causing the protobuf `GetPrototype` crash by using `tf.keras` consistently, and I remove notebook-only syntax (`%matplotlib inline`) so the script runs as a .py. I update deprecated/removed Keras 3 module paths (e.g., `keras.layers.convolutional`, `keras.utils.generic_utils`) and fix `ImageDataGenerator` usage so the generators are defined. I also switch the final activation to `softmax` (multi-class log loss expects a probability distribution) while keeping the same CNN architecture and training loop, and ensure the checkpoint saves to a `.keras` file and is loaded correctly. Finally, I build the submission by following `sample_submission.csv` column order to guarantee a valid `.csv` with the required header and aligned `id`s.'
- What this solution (achieved 4.34651) has done: 'I fix the training crash by telling `EarlyStopping` and `ModelCheckpoint` how to interpret `val_categorical_accuracy` (mode='max'), which unblocks model.fit and ensures the checkpoint file is actually created. To address the protobuf `MessageFactory.GetPrototype` error seen at import time in some Kaggle images, I force the pure-Python protobuf implementation before importing TensorFlow (a minimal environment-compatibility fix). I keep the same CNN architecture, preprocessing, generators, and training loop; no score-tuning changes are introduced beyond making it run. Finally, I make model loading conditional on the checkpoint existing (fallback to the in-memory model) so submission generation always completes and writes a valid `submission.csv`.'
- What this solution (achieved 4.47448) has done: 'I fix the import-time crash (`MessageFactory` missing `GetPrototype`) by forcing a protobuf version that is compatible with TensorFlow 2.18, without changing your model/training logic. To keep the run stable in Kaggle, I also add a small fallback that disables the “python protobuf implementation” override if it still triggers the error, then re-import TensorFlow cleanly. Finally, I ensure the submission is always written in the exact `sample_submission.csv` column order and that predictions are numerically safe probabilities (row-normalized and clipped), which is score-neutral but prevents invalid log-loss edge cases.'
- What this solution (achieved 4.66913) has done: 'I fix the import-time protobuf/TensorFlow crash by setting a compatible protobuf implementation *before* importing TensorFlow (this is the root cause of the `MessageFactory.GetPrototype` error). I keep your exact CNN, generators, training loop, and submission construction unchanged so the modeling/evaluation semantics stay the same and the score should remain in the same range (already better than the target, so no intentional score-tuning changes). I also make the TensorFlow import resilient by retrying with a fallback environment setting if the first import still fails in this Kaggle image. Finally, I keep the submission writing exactly as you already do to guarantee a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.model_selection import train_test_split

try:
    import tensorflow as tf
except Exception as e:
    os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
    os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
    import tensorflow as tf  # noqa: F401

from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array,
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

np.random.seed(42)
tf.random.set_seed(42)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def gen_graph(history, title):
    if history is None:
        return
    if (
        "categorical_accuracy" in history.history
        and "val_categorical_accuracy" in history.history
    ):
        plt.plot(history.history["categorical_accuracy"])
        plt.plot(history.history["val_categorical_accuracy"])
        plt.title("Accuracy " + title)
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["train", "validation"], loc="upper left")
        plt.show()

    if "fbeta" in history.history and "val_fbeta" in history.history:
        plt.plot(history.history["fbeta"])
        plt.plot(history.history["val_fbeta"])
        plt.title("fbeta " + title)
        plt.ylabel("fbeta")
        plt.xlabel("Epoch")
        plt.legend(["train", "validation"], loc="upper left")
        plt.show()




## === cell 2
def fbeta(y_true, y_pred, beta=2):
    y_pred = backend.clip(y_pred, 0, 1)
    tp = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)), axis=1)
    fp = backend.sum(backend.round(backend.clip(y_pred - y_true, 0, 1)), axis=1)
    fn = backend.sum(backend.round(backend.clip(y_true - y_pred, 0, 1)), axis=1)
    p = tp / (tp + fp + backend.epsilon())
    r = tp / (tp + fn + backend.epsilon())
    bb = beta**2
    fbeta_score = backend.mean((1 + bb) * (p * r) / (bb * p + r + backend.epsilon()))
    return fbeta_score




## === cell 3
DATA_DIR = "../input/dog-breed-identification"

labels_path = os.path.join(DATA_DIR, "labels.csv")
sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")
train_dir = os.path.join(DATA_DIR, "train")
test_dir = os.path.join(DATA_DIR, "test")

assert os.path.exists(labels_path), f"Missing: {labels_path}"
assert os.path.exists(sample_sub_path), f"Missing: {sample_sub_path}"
assert os.path.isdir(train_dir), f"Missing dir: {train_dir}"
assert os.path.isdir(test_dir), f"Missing dir: {test_dir}"

df_train = pd.read_csv(labels_path)
df_test = pd.read_csv(sample_sub_path)

jpg_train = os.path.join(train_dir, "{}.jpg")
jpg_test = os.path.join(test_dir, "{}.jpg")



## === cell 4
df_train.head()



## === cell 5
df_test.head()



## === cell 6
labels = df_train["breed"]
one_hot = pd.get_dummies(labels, sparse=False)



## === cell 7
one_hot_labels = np.asarray(one_hot, dtype=np.float32)
one_hot_labels.shape



## === cell 8
im_resize = 64
num_class = one_hot_labels.shape[1]
num_class



## === cell 9
x_train = []
y_train = []
x_test = []



## === cell 10
i = 0
for f, breed in tqdm(df_train.values, total=len(df_train)):
    img = load_img(jpg_train.format(f), target_size=(im_resize, im_resize))
    img_resized = img_to_array(img)
    x_train.append(img_resized)
    y_train.append(one_hot_labels[i])
    i += 1



## === cell 11
for f in tqdm(df_test["id"].values, total=len(df_test)):
    img = load_img(jpg_test.format(f), target_size=(im_resize, im_resize))
    img_resized = img_to_array(img)
    x_test.append(img_resized)



## === cell 12
X_train, X_valid, Y_train, Y_valid = train_test_split(
    np.array(x_train, dtype=np.float32),
    np.array(y_train, dtype=np.float32),
    shuffle=True,
    test_size=0.2,
    random_state=42,
    stratify=np.argmax(np.array(y_train), axis=1),
)



## === cell 13
del x_train, y_train



## === cell 14
datagen = ImageDataGenerator(rescale=1.0 / 255.0)



## === cell 15
model = Sequential()

model.add(
    Conv2D(
        32,
        (3, 3),
        padding="same",
        input_shape=(im_resize, im_resize, 3),
        activation="relu",
        kernel_initializer="he_uniform",
    )
)
model.add(
    Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_uniform",
        padding="same",
    )
)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(
    Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_uniform",
        padding="same",
    )
)
model.add(
    Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_uniform",
        padding="same",
    )
)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(
    Conv2D(
        128,
        (3, 3),
        activation="relu",
        kernel_initializer="he_uniform",
        padding="same",
    )
)
model.add(
    Conv2D(
        128,
        (3, 3),
        activation="relu",
        kernel_initializer="he_uniform",
        padding="same",
    )
)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256, activation="relu", kernel_initializer="he_uniform"))
model.add(Dropout(0.5))

model.add(Dense(num_class, activation="softmax"))



## === cell 16
print(model.summary())



## === cell 17
model.compile(
    optimizer="Adam",
    loss="categorical_crossentropy",
    metrics=[categorical_accuracy, fbeta],
)



## === cell 18
train_generator = datagen.flow(X_train, Y_train, batch_size=128, shuffle=True)
valid_generator = datagen.flow(X_valid, Y_valid, batch_size=128, shuffle=False)



## === cell 19
earlystop = EarlyStopping(
    monitor="val_categorical_accuracy",
    mode="max",
    min_delta=0.0,
    patience=5,
    restore_best_weights=False,
)

checkpoint_path = "model_best.keras"
checkpoint_callback = ModelCheckpoint(
    checkpoint_path,
    monitor="val_categorical_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1,
)



## === cell 20
batch_size = 128
Epochs = 50

history_rmsprop = model.fit(
    train_generator,
    callbacks=[earlystop, checkpoint_callback],
    epochs=Epochs,
    steps_per_epoch=len(train_generator),
    validation_data=valid_generator,
    validation_steps=len(valid_generator),
    verbose=2,
)



## === cell 21
gen_graph(history_rmsprop, "training curves")



## === cell 22
if os.path.exists(checkpoint_path):
    model = keras.models.load_model(checkpoint_path, custom_objects={"fbeta": fbeta})
else:
    print(f"Warning: checkpoint not found at {checkpoint_path}; using in-memory model.")



## === cell 23
x_test_arr = np.array(x_test, dtype=np.float32) / 255.0
preds = model.predict(x_test_arr, batch_size=128, verbose=0)



## === cell 24
sample_cols = df_test.columns.tolist()  # includes "id" then breed columns
breed_cols = sample_cols[1:]

sub = pd.DataFrame(preds, columns=breed_cols)
sub.insert(0, "id", df_test["id"].values)

probs = sub[breed_cols].to_numpy(dtype=np.float64)
probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
row_sums = probs.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1.0
probs = probs / row_sums
probs = np.clip(probs, 1e-15, 1.0)
probs = probs / probs.sum(axis=1, keepdims=True)

sub[breed_cols] = probs.astype(np.float32)
sub = sub[["id"] + breed_cols]  # enforce order

sub.head(5)



## === cell 25
out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print(f"Wrote submission: {out_path} with shape={sub.shape}")
print(
    "Columns match sample_submission:", sub.columns.tolist() == df_test.columns.tolist()
)
