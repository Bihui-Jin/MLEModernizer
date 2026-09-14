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

16.79687

# 6. Current score

5.03977

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 4.78084) has done: 'I fix the environment/runtime breakages caused by mixing `tensorflow.keras` with standalone `keras==3` (protobuf error, missing modules, moved layer paths) by using a single API (`tf.keras`) consistently. I keep the same CNN architecture and training loop, but make the output activation/loss consistent with the competition’s multiclass log loss (softmax + categorical_crossentropy), which is a minimal semantic correction that should substantially improve score from “not yielded” to a reasonable value. I also ensure the generators, checkpoint saving/loading (Keras 3 requires `.keras`), and submission columns exactly match `sample_submission.csv` ordering to avoid silent misalignment. Finally, I write a valid `.csv` submission file to the working directory.'
- What this solution (achieved 5.03977) has done: 'I fix the two runtime blockers that prevent training and checkpoint creation: the protobuf/tf import crash and the EarlyStopping “mode” ambiguity in Keras. I also ensure checkpointing always produces a loadable model by saving weights-only during training (robust across Keras/TensorFlow versions) and then loading those weights into the same architecture for prediction. These changes are execution/stability fixes and should keep the modeling logic the same while allowing the pipeline to complete and write a valid `.csv` submission. Since your current score (4.78084) is already much better than the target (16.79687, lower-is-better), I won’t make score-improving changes beyond what’s required to run end-to-end.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array,
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.metrics import categorical_accuracy
from sklearn.model_selection import train_test_split
from tqdm import tqdm

tf.keras.utils.set_random_seed(42)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def gen_graph(history, title):
    if history is None:
        return
    if "categorical_accuracy" in history.history:
        plt.plot(history.history["categorical_accuracy"])
    if "val_categorical_accuracy" in history.history:
        plt.plot(history.history["val_categorical_accuracy"])
    plt.title("Accuracy " + str(title))
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()

    if "fbeta" in history.history:
        plt.plot(history.history["fbeta"])
        if "val_fbeta" in history.history:
            plt.plot(history.history["val_fbeta"])
        plt.title("fbeta " + str(title))
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
df_train = pd.read_csv("../input/dog-breed-identification/labels.csv")
df_test = pd.read_csv("../input/dog-breed-identification/sample_submission.csv")
jpg_train = "../input/dog-breed-identification/train/{}.jpg"
jpg_test = "../input/dog-breed-identification/test/{}.jpg"



## === cell 4
df_train.head()



## === cell 5
df_test.head()



## === cell 6
labels = df_train["breed"]
one_hot = pd.get_dummies(labels, sparse=False)



## === cell 7
one_hot_labels = np.asarray(one_hot, dtype=np.float32)
one_hot_labels



## === cell 8
im_resize = 64  # image size
num_class = 120  # number of classes



## === cell 9
x_train = []
y_train = []
x_test = []



## === cell 10
pass



## === cell 11
i = 0
for f, breed in tqdm(df_train.values, total=len(df_train)):
    img = load_img(jpg_train.format(f), target_size=(im_resize, im_resize))
    img_resized = img_to_array(img)
    x_train.append(img_resized)
    y_train.append(one_hot_labels[i])
    i += 1



## === cell 12
for f in tqdm(df_test["id"].values, total=len(df_test)):
    img = load_img(jpg_test.format(f), target_size=(im_resize, im_resize))
    img_resized = img_to_array(img)
    x_test.append(img_resized)



## === cell 13
X_train, X_valid, Y_train, Y_valid = train_test_split(
    x_train, y_train, shuffle=True, test_size=0.2, random_state=42
)



## === cell 14
del x_train, y_train



## === cell 15
datagen = ImageDataGenerator(rescale=1.0 / 255.0)



## === cell 16
pass



## === cell 17
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

model.add(Flatten())
model.add(Dense(256, activation="relu", kernel_initializer="he_uniform"))
model.add(Dense(num_class, activation="softmax"))



## === cell 18
print(model.summary())



## === cell 19
model.compile(
    optimizer="Adam",
    loss="categorical_crossentropy",
    metrics=[categorical_accuracy, fbeta],
)



## === cell 20
train_generator = datagen.flow(
    np.array(X_train, dtype=np.float32),
    np.array(Y_train, dtype=np.float32),
    batch_size=128,
    shuffle=True,
)
valid_generator = datagen.flow(
    np.array(X_valid, dtype=np.float32),
    np.array(Y_valid, dtype=np.float32),
    batch_size=128,
    shuffle=False,
)



## === cell 21
earlystop = EarlyStopping(
    monitor="val_categorical_accuracy",
    mode="max",
    min_delta=0,
    patience=5,
    restore_best_weights=False,
)

checkpoint_path = "model_best.weights.h5"
checkpoint_callback = ModelCheckpoint(
    checkpoint_path,
    monitor="val_categorical_accuracy",
    mode="max",
    save_best_only=True,
    save_weights_only=True,
    verbose=1,
)



## === cell 22
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



## === cell 23
gen_graph(history_rmsprop, "график точности")



## === cell 24
pass



## === cell 25
if os.path.exists(checkpoint_path):
    model.load_weights(checkpoint_path)
else:
    print(
        f"WARNING: checkpoint not found at {checkpoint_path}; using last epoch weights."
    )



## === cell 26
x_test_arr = np.array(x_test, dtype=np.float32) / 255.0
preds = model.predict(x_test_arr, verbose=0)



## === cell 27
sub = pd.DataFrame(preds, columns=one_hot.columns.values)

sample_cols = df_test.columns.tolist()
breed_cols = sample_cols[1:]  # after 'id'
sub = sub.reindex(columns=breed_cols)
sub.insert(0, "id", df_test["id"].values)

sub.head(5)



## === cell 28
sub.to_csv("output_rmsprop_aug.csv", index=False)
print("Wrote submission to output_rmsprop_aug.csv with shape:", sub.shape)
