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

5.50952

# 6. Current score

4.7852

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 4.78346) has done: 'I make the notebook runnable in Kaggle by fixing the Keras/TensorFlow import mix that triggers the protobuf `MessageFactory` crash, removing IPython-only `%matplotlib inline`, and ensuring all required symbols (tqdm, train_test_split, layers, ImageDataGenerator, metrics) are imported from the same `tf.keras` namespace. I keep your CNN architecture/training loop intact, but update `ModelCheckpoint` to save in the new Keras format (`.keras`) and load that same file later. I also ensure test predictions are made on properly rescaled images (matching training/validation preprocessing) and write a correctly formatted submission using `sample_submission.csv` column order so Kaggle accepts it. Finally, I keep the runtime stable by using a non-shuffled validation generator and a safe `model.predict(..., batch_size=...)` call.'
- What this solution (achieved 4.78514) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by forcing TensorFlow to use the pure-Python protobuf implementation before importing TensorFlow/Keras, which is the minimal change that unblocks runtime in Kaggle for this environment. I keep your model, augmentation, training loop, and checkpointing logic identical, only adjusting imports/seed setup to be consistent and stable. I also ensure the submission is written as a valid `.csv` with the exact column order from `sample_submission.csv` so Kaggle accepts it. No score-targeting changes are applied since your current score is already better than the target (lower is better), and the goal here is correctness and end-to-end execution.'
- What this solution (achieved 4.78418) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf runtime *and* disabling the C++ implementation before TensorFlow is imported, which is the root cause of the current runtime error in this environment. I keep your CNN, augmentation, training loop, checkpointing, and prediction logic unchanged to avoid unintended score shifts (your current score is already better than the target, and lower is better). I also make the run more robust by ensuring the output submission filename ends with `.csv` and matches `sample_submission.csv` column order exactly (so Kaggle accepts it reliably). No score-tuning changes be introduced beyond these stability fixes.'
- What this solution (achieved 4.78538) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by forcing TensorFlow to use the pure-Python protobuf runtime *and* disabling the C++ protobuf implementation before TensorFlow (or anything that may import it) is loaded; this is the root cause of the runtime error in this Kaggle environment. I also keep all modeling/training logic intact and only make minimal, score-neutral robustness tweaks: ensure `num_class` is derived from the labels (to avoid any rare class-mismatch issues) and use the exact column order from `sample_submission.csv` when writing the submission. Since your current score (4.78418, lower is better) is already better than the target (5.50952), I not apply any changes intended to improve the score—only to make the notebook run end-to-end reliably. The output be a valid `submission.csv` with the required header and probabilities for all breeds.'
- What this solution (achieved 4.7852) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by setting the protobuf environment variables *before any TensorFlow/Keras import happens*, and by also forcing the pure-Python protobuf module via `sys.modules` as a last-resort guard for this Kaggle image. I keep your CNN architecture, augmentation, training loop, and checkpoint usage intact, only making minimal stability changes: dropping unused env vars, ensuring the validation generator is non-shuffled (already), and keeping submission column order exactly matching `sample_submission.csv`. Since your current score is already better than the target (lower is better), I not introduce any score-improving changes—only runtime correctness so it runs end-to-end and writes `submission.csv`. The result be a valid `.csv` with probabilities for all breeds and the correct header/order.'

# 9. Code solution

## === cell 0
labels_csv = "../input/dog-breed-identification/labels.csv"
sample_submission_csv = "../input/dog-breed-identification/sample_submission.csv"

jpg_train = "../input/dog-breed-identification/train/{}.jpg"
jpg_test = "../input/dog-breed-identification/test/{}.jpg"

im_resize = 64  # image size
num_class = 120  # will be validated against labels later (core logic unchanged)
batch_size = 32
Epochs = 50



## === cell 1
import os
import sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

try:
    import google.protobuf as _pb  # noqa: F401
    from google.protobuf import message_factory as _mf  # noqa: F401
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.preprocessing.image import (
    load_img,
    img_to_array,
    ImageDataGenerator,
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

np.random.seed(42)
tf.random.set_seed(42)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
def gen_graph(history, title):
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("crossentropy " + title)
    plt.ylabel("crossentropy")
    plt.xlabel("Epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()

    plt.plot(history.history["categorical_accuracy"])
    plt.plot(history.history["val_categorical_accuracy"])
    plt.title("categorical_accuracy " + title)
    plt.ylabel("categorical_accuracy")
    plt.xlabel("Epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()




## === cell 3
df_train = pd.read_csv(labels_csv)
df_test = pd.read_csv(sample_submission_csv)



## === cell 4
df_train.head()



## === cell 5
df_test.head()



## === cell 6
labels = df_train["breed"]
one_hot = pd.get_dummies(labels, sparse=False)
one_hot_labels = np.asarray(one_hot)

num_class = one_hot.shape[1]



## === cell 7
x_train = []
y_train = []
x_test = []



## === cell 8
i = 0
for f, breed in tqdm(df_train.values, total=len(df_train)):
    img = load_img(jpg_train.format(f), target_size=(im_resize, im_resize))
    img_resized = img_to_array(img)
    x_train.append(img_resized)
    label = one_hot_labels[i]
    y_train.append(label)
    i += 1



## === cell 9
for f in tqdm(df_test["id"].values, total=len(df_test)):
    img = load_img(jpg_test.format(f), target_size=(im_resize, im_resize))
    img_resized = img_to_array(img)
    x_test.append(img_resized)



## === cell 10
X_train, X_valid, Y_train, Y_valid = train_test_split(
    x_train, y_train, shuffle=True, test_size=0.2, random_state=42
)



## === cell 11
del x_train, y_train, df_train



## === cell 12
train_datagen = ImageDataGenerator(
    rotation_range=15,  # rotation
    rescale=1.0 / 255,  # normalization
    horizontal_flip=True,
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow(
    np.array(X_train), np.array(Y_train), batch_size=batch_size, shuffle=True
)

test_generator = test_datagen.flow(
    np.array(X_valid), np.array(Y_valid), batch_size=batch_size * 5, shuffle=False
)



## === cell 13
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



## === cell 14
model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=[categorical_accuracy]
)



## === cell 15
print(model.summary())



## === cell 16
earlystop = EarlyStopping(
    monitor="val_loss", min_delta=0, patience=5, restore_best_weights=False
)

checkpoint_path = "model_best.keras"
checkpoint_callback = ModelCheckpoint(
    checkpoint_path, monitor="val_categorical_accuracy", save_best_only=True, verbose=1
)



## === cell 17
history = model.fit(
    train_generator,
    callbacks=[earlystop, checkpoint_callback],
    epochs=Epochs,
    steps_per_epoch=len(train_generator),
    validation_data=test_generator,
    validation_steps=len(test_generator),
)



## === cell 18
gen_graph(history, "график точности")



## === cell 19
model = load_model(checkpoint_path)



## === cell 20
x_test_arr = np.array(x_test, dtype=np.float32) / 255.0
preds = model.predict(x_test_arr, batch_size=batch_size * 5, verbose=1)

sub = pd.DataFrame(preds, columns=one_hot.columns.values)
sub.insert(0, "id", df_test["id"].values)
sub = sub.reindex(columns=df_test.columns)

out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote submission:", out_path, "shape:", sub.shape)
print(sub.head(3))
