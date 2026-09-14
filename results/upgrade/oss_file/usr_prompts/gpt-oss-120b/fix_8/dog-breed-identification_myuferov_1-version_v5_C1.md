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
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
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

32.44366

# 6. Current score

4.57268

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 4.78303) has done: 'I fixed the import mismatches (using tensorflow.keras consistently), corrected the ImageDataGenerator and Conv2D imports, changed the checkpoint filename to the required “.keras” extension, wired the validation data correctly, added simple safety checks when loading images, and ensured the test‑time predictions are built into a proper submission CSV matching the sample format. These changes unblock the script so it runs end‑to‑end and writes a valid output_rmsprop_aug.csv file.'
- What this solution (achieved 4.78568) has done: 'The update speeds up the pipeline by loading and normalising all images once (removing repeated NumPy casts), dropping the costly ZCA whitening from the image‑augmentation generator, and increasing the batch size so each epoch performs far fewer gradient steps while keeping the same architecture, loss and early‑stopping logic. All changes are confined to data‑handling and training‑configuration cells, preserving the original model and evaluation semantics.'
- What this solution (achieved 4.7037) has done: 'The fix updates the imports to use `tensorflow.keras` (which avoids the protobuf import error), adds the missing `tqdm` import, and makes sure all required libraries are loaded before any code runs. No core model logic is changed; the script now runs end‑to‑end and writes a correct `.csv` submission file.'
- What this solution (achieved 4.67704) has done: 'The fix adds the required environment variable to avoid the protobuf `MessageFactory` error that occurs when importing TensorFlow. This change is applied before any TensorFlow imports, allowing the script to run end‑to‑end and produce a valid submission CSV while preserving the original model and training logic.'
- What this solution (achieved 4.57268) has done: 'The fix keeps the existing environment‑variable work‑around that resolves the protobuf import error, ensures the model and data pipeline remain unchanged, and guarantees that a correctly‑formatted CSV submission is written at the end. No changes to the core architecture or training logic are made, preserving the current low log‑loss while making the script run end‑to‑end.'

# 9. Code solution

## === cell 0
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.metrics import categorical_accuracy, categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def gen_graph(history, title):
    plt.plot(history.history["categorical_accuracy"])
    plt.plot(history.history["val_categorical_accuracy"])
    plt.title("Accuracy " + title)
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()




## === cell 2
df_train = pd.read_csv("../input/dog-breed-identification/labels.csv")
df_test = pd.read_csv("../input/dog-breed-identification/sample_submission.csv")
jpg_train = "../input/dog-breed-identification/train/{}.jpg"
jpg_test = "../input/dog-breed-identification/test/{}.jpg"



## === cell 3
labels = df_train["breed"]
one_hot = pd.get_dummies(labels, sparse=True)



## === cell 4
one_hot_labels = np.asarray(one_hot)



## === cell 5
im_resize = 64  # image size (64x64)
num_class = 120  # number of breeds



## === cell 6
x_train = []
y_train = []
x_test = []



## === cell 7
i = 0
for f, breed in tqdm(df_train.values, desc="Loading train images"):
    img_path = jpg_train.format(f)
    img = cv2.imread(img_path)
    if img is None:
        continue
    img_resized = cv2.resize(img, (im_resize, im_resize))
    x_train.append(img_resized.astype(np.float32) / 255.0)  # normalize once
    y_train.append(one_hot_labels[i])
    i += 1



## === cell 8
for f in tqdm(df_test["id"].values, desc="Loading test images"):
    img_path = jpg_test.format(f)
    img = cv2.imread(img_path)
    if img is None:
        img = np.zeros((im_resize, im_resize, 3), dtype=np.uint8)
    img_resized = cv2.resize(img, (im_resize, im_resize))
    x_test.append(img_resized.astype(np.float32) / 255.0)  # normalize once



## === cell 9
X_train, X_valid, Y_train, Y_valid = train_test_split(
    x_train, y_train, test_size=0.1, shuffle=True, random_state=42
)



## === cell 10
del x_train, y_train, df_train  # free memory



## === cell 11
datagen = ImageDataGenerator(
    rotation_range=15,
    horizontal_flip=True,
)



## === cell 12
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
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dense(num_class, activation="softmax"))



## === cell 13
print(model.summary())



## === cell 14
model.compile(
    optimizer="Adam",
    loss="categorical_crossentropy",
    metrics=[categorical_accuracy, categorical_crossentropy],
)



## === cell 15
batch_size = 1024
train_generator = datagen.flow(
    np.array(X_train), np.array(Y_train), batch_size=batch_size, shuffle=True
)



## === cell 16
earlystop = EarlyStopping(
    monitor="val_categorical_accuracy",
    mode="max",
    patience=5,
    restore_best_weights=True,
)

checkpoint_callback = ModelCheckpoint(
    "model_best.keras",  # must end with .keras for Keras
    monitor="val_categorical_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1,
)



## === cell 17
epochs = 30
history_rmsprop = model.fit(
    train_generator,
    callbacks=[earlystop, checkpoint_callback],
    epochs=epochs,
    steps_per_epoch=max(1, len(X_train) // batch_size),
    validation_data=(np.array(X_valid), np.array(Y_valid)),
    validation_steps=max(1, len(X_valid) // batch_size),
    verbose=2,
)



## === cell 18
gen_graph(history_rmsprop, "график точности")



## === cell 19
model = load_model("model_best.keras")



## === cell 20
test_array = np.array(x_test, dtype=np.float32)  # already normalized
preds = model.predict(test_array, verbose=1)



## === cell 21
sub = pd.DataFrame(preds, columns=one_hot.columns)
sub.insert(0, "id", df_test["id"])
sub.head()



## === cell 22
sub.to_csv("output_rmsprop_aug.csv", index=False)
