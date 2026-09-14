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
Use binary leaf images and extracted features to identify the species of plant.

## Metric
Multi-class log loss. 

The submitted probabilities for a given device are not required to sum to one because they are rescaled prior to being scored (each row is divided by the row sum), but they need to be in the range of [0, 1]. In order to avoid the extremes of the log function, predicted probabilities are replaced with \\(max(min(p,1-10^{-15}),10^{-15})\\).

## Submission Format
You must submit a csv file with the image id, all candidate species names, and a probability for each species. The order of the rows does not matter. The file must have a header and should look like the following:

id,Acer_Capillipes,Acer_Circinatum,Acer_Mono,...
2,0.1,0.5,0,0.2,...
5,0,0.3,0,0.4,...
6,0,0,0,0.7,...
etc.

## Dataset
The dataset consists of images of leaf specimens which have been converted to binary black leaves against white backgrounds. 

Three sets of features are also provided per image: a shape contiguous descriptor, an interior texture histogram, and a ﬁne-scale margin histogram. 

For each feature, a 64-attribute vector is given per leaf sample.

### File descriptions
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format
- **images/** - the image files (each image is named with its corresponding id)

### Data fields
- **id** - an anonymous id unique to an image
- **margin_1, margin_2, margin_3, ..., margin_64** - each of the 64 attribute vectors for the margin feature
- **shape_1, shape_2, shape_3, ..., shape_64** - each of the 64 attribute vectors for the shape feature
- **texture_1, texture_2, texture_3, ..., texture_64** - each of the 64 attribute vectors for the texture feature

# 2. Python version

3.6

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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (70 lines)
            images.zip (22.0 MB)
            sample_submission.csv (100 lines)
            sample_submission.csv.zip (2.3 kB)
            test.csv (100 lines)
            test.csv.zip (39.3 kB)
            train.csv (892 lines)
            train.csv.zip (357.1 kB)
            images/
                42.jpg (32.6 kB)
                168.jpg (16.5 kB)
                ... and 988 other files
            leaf-classification/
                description.md (70 lines)
                images.zip (22.0 MB)
                ... and 6 other files
                images/
                    42.jpg (32.6 kB)
                    168.jpg (16.5 kB)
                    ... and 988 other files
                leaf-classification/
        input/
            description.md (70 lines)
            images.zip (22.0 MB)
            sample_submission.csv (100 lines)
            sample_submission.csv.zip (2.3 kB)
            test.csv (100 lines)
            test.csv.zip (39.3 kB)
            train.csv (892 lines)
            train.csv.zip (357.1 kB)
            images/
                42.jpg (32.6 kB)
                168.jpg (16.5 kB)
                ... and 988 other files
            leaf-classification/
                description.md (70 lines)
                images.zip (22.0 MB)
                ... and 6 other files
                images/
                    42.jpg (32.6 kB)
                    168.jpg (16.5 kB)
                    ... and 988 other files
                leaf-classification/
        working/
            leaf-classification/
                description.md (70 lines)
                images.zip (22.0 MB)
                ... and 6 other files
                images/
                    42.jpg (32.6 kB)
                    168.jpg (16.5 kB)
                    ... and 988 other files
                leaf-classification/
```

-> data/leaf-classification/sample_submission.csv has 99 rows and 100 columns.
The columns are: id, Acer_Capillipes, Acer_Circinatum, Acer_Mono, Acer_Opalus, Acer_Palmatum, Acer_Pictum, Acer_Platanoids, Acer_Rubrum, Acer_Rufinerve, Acer_Saccharinum, Alnus_Cordata, Alnus_Maximowiczii, Alnus_Rubra, Alnus_Sieboldiana... and 85 more columns

-> data/leaf-classification/test.csv has 99 rows and 193 columns.
The columns are: id, margin1, margin2, margin3, margin4, margin5, margin6, margin7, margin8, margin9, margin10, margin11, margin12, margin13, margin14... and 178 more columns

-> data/leaf-classification/train.csv has 891 rows and 194 columns.
The columns are: id, species, margin1, margin2, margin3, margin4, margin5, margin6, margin7, margin8, margin9, margin10, margin11, margin12, margin13... and 179 more columns

-> data/sample_submission.csv has 99 rows and 100 columns.
The columns are: id, Acer_Capillipes, Acer_Circinatum, Acer_Mono, Acer_Opalus, Acer_Palmatum, Acer_Pictum, Acer_Platanoids, Acer_Rubrum, Acer_Rufinerve, Acer_Saccharinum, Alnus_Cordata, Alnus_Maximowiczii, Alnus_Rubra, Alnus_Sieboldiana... and 85 more columns

-> data/test.csv has 99 rows and 193 columns.
The columns are: id, margin1, margin2, margin3, margin4, margin5, margin6, margin7, margin8, margin9, margin10, margin11, margin12, margin13, margin14... and 178 more columns

-> data/train.csv has 891 rows and 194 columns.
The columns are: id, species, margin1, margin2, margin3, margin4, margin5, margin6, margin7, margin8, margin9, margin10, margin11, margin12, margin13... and 179 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.01373

# 6. Current score

0.02893

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.03179) has done: 'I update deprecated/removed imports and Keras API arguments so the notebook runs on your environment (scikit-learn 1.2 + Keras 3). I also fix a couple of runtime issues that would prevent creating a valid submission: incorrect column sorting (`sort()`), missing `id` column in the output, and incompatible `predict_proba()` calls. Finally, I ensure preprocessing is fit on train and applied to test (instead of refitting on test), which is a minimal correctness fix and should improve log-loss toward the target without changing the core model/training approach.'
- What this solution (achieved 0.02893) has done: 'The immediate blocker is an environment/runtime crash coming from protobuf incompatibilities that Keras can trigger during import (`MessageFactory.GetPrototype`). I fix this by forcing the safe pure-Python protobuf implementation before importing Keras, which is a minimal, score-neutral change that restores end-to-end execution. Then I keep your exact model/training logic, but add a tiny, metric-aligned post-processing step: renormalize each prediction row to sum to 1 (the metric does this anyway) and clip with the competition’s epsilon, which typically improves log-loss slightly without changing the model. Finally, I ensure the submission columns exactly match `sample_submission.csv` and write a valid `.csv` to the working directory.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import time

start = time.time()

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

np.random.seed(12345)

TRAIN_PATHS = [
    "/kaggle/input/train.csv",
    "/kaggle/input/leaf-classification/train.csv",
    "/kaggle/data/train.csv",
    "/kaggle/data/leaf-classification/train.csv",
]
TEST_PATHS = [
    "/kaggle/input/test.csv",
    "/kaggle/input/leaf-classification/test.csv",
    "/kaggle/data/test.csv",
    "/kaggle/data/leaf-classification/test.csv",
]
SAMPLE_SUB_PATHS = [
    "/kaggle/input/sample_submission.csv",
    "/kaggle/input/leaf-classification/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
    "/kaggle/data/leaf-classification/sample_submission.csv",
]


def _read_first_existing(paths):
    last_err = None
    for p in paths:
        try:
            return pd.read_csv(p)
        except FileNotFoundError as e:
            last_err = e
            continue
    raise FileNotFoundError(f"None of the paths exist: {paths}. Last error: {last_err}")


data = _read_first_existing(TRAIN_PATHS)
parent_data = data.copy()  # keep a copy of original data
ID = data.pop("id")

y = data.pop("species")
le = LabelEncoder()
y = le.fit_transform(y)
print("y shape:", y.shape)

X_raw = data.values.astype(np.float32)

y_cat = to_categorical(y)
print("y_cat shape:", y_cat.shape)

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=12345)
train_index, val_index = next(iter(sss.split(X_raw, y)))
x_train_raw, x_val_raw = X_raw[train_index], X_raw[val_index]
y_train, y_val = y_cat[train_index], y_cat[val_index]

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train_raw)
x_val = scaler.transform(x_val_raw)

print("x_train dim:", x_train.shape)
print("x_val dim:  ", x_val.shape)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
model1 = Sequential()
model1.add(Dense(600, input_dim=192, kernel_initializer="uniform", activation="relu"))
model1.add(Dropout(0.3))
model1.add(Dense(600, activation="sigmoid"))
model1.add(Dropout(0.3))
model1.add(Dense(99, activation="softmax"))

model1.compile(
    loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
)

early_stopping = EarlyStopping(
    monitor="val_loss", patience=300, restore_best_weights=True
)
history1 = model1.fit(
    x_train,
    y_train,
    batch_size=192,
    epochs=2500,  # was nb_epoch
    verbose=0,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping],
)

print("model1 val_acc:", max(history1.history.get("val_accuracy", [np.nan])))
print("model1 val_loss:", min(history1.history.get("val_loss", [np.nan])))
print("model1 train_acc:", max(history1.history.get("accuracy", [np.nan])))
print("model1 train_loss:", min(history1.history.get("loss", [np.nan])))

plt.semilogy(history1.history["loss"])
plt.semilogy(history1.history["val_loss"])
plt.title("model1 loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.show()

plt.plot(history1.history["accuracy"])
plt.plot(history1.history["val_accuracy"])
plt.title("model1 accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.show()



## === cell 2
model2 = Sequential()
model2.add(
    Dense(600, input_dim=192, kernel_initializer="glorot_normal", activation="relu")
)
model2.add(Dropout(0.1))
model2.add(Dense(300, activation="sigmoid"))
model2.add(Dropout(0.1))
model2.add(Dense(99, activation="softmax"))

model2.compile(
    loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
)

early_stopping = EarlyStopping(
    monitor="val_loss", patience=300, restore_best_weights=True
)
history2 = model2.fit(
    x_train,
    y_train,
    batch_size=192,
    epochs=2500,
    verbose=0,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping],
)

print("model2 val_acc:", max(history2.history.get("val_accuracy", [np.nan])))
print("model2 val_loss:", min(history2.history.get("val_loss", [np.nan])))
print("model2 train_acc:", max(history2.history.get("accuracy", [np.nan])))
print("model2 train_loss:", min(history2.history.get("loss", [np.nan])))

plt.semilogy(history2.history["loss"])
plt.semilogy(history2.history["val_loss"])
plt.title("model2 loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.show()

plt.plot(history2.history["accuracy"])
plt.plot(history2.history["val_accuracy"])
plt.title("model2 accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.show()



## === cell 3
model3 = Sequential()
model3.add(
    Dense(800, input_dim=192, kernel_initializer="glorot_normal", activation="relu")
)
model3.add(Dropout(0.1))
model3.add(Dense(400, activation="sigmoid"))
model3.add(Dropout(0.1))
model3.add(Dense(99, activation="softmax"))

model3.compile(
    loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
)

early_stopping = EarlyStopping(
    monitor="val_loss", patience=300, restore_best_weights=True
)
history3 = model3.fit(
    x_train,
    y_train,
    batch_size=192,
    epochs=2500,
    verbose=0,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping],
)

print("model3 val_acc:", max(history3.history.get("val_accuracy", [np.nan])))
print("model3 val_loss:", min(history3.history.get("val_loss", [np.nan])))
print("model3 train_acc:", max(history3.history.get("accuracy", [np.nan])))
print("model3 train_loss:", min(history3.history.get("loss", [np.nan])))

plt.semilogy(history3.history["loss"])
plt.semilogy(history3.history["val_loss"])
plt.title("model3 loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.show()

plt.plot(history3.history["accuracy"])
plt.plot(history3.history["val_accuracy"])
plt.title("model3 accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.show()



## === cell 4
test = _read_first_existing(TEST_PATHS)
index = test.pop("id").values

X_test_raw = test.values.astype(np.float32)
X_test = scaler.transform(X_test_raw)  # IMPORTANT: do not fit on test

yPred1 = model1.predict(X_test, verbose=0)
yPred2 = model2.predict(X_test, verbose=0)
yPred3 = model3.predict(X_test, verbose=0)

yPred = (yPred1 + yPred2 + yPred3) / 3.0

row_sum = yPred.sum(axis=1, keepdims=True)
row_sum[row_sum == 0.0] = 1.0
yPred = yPred / row_sum
eps = 1e-15
yPred = np.clip(yPred, eps, 1.0 - eps)

class_names = list(le.classes_)

sub = pd.DataFrame(yPred, columns=class_names)
sub.insert(0, "id", index)

sample_sub = _read_first_existing(SAMPLE_SUB_PATHS)
for c in sample_sub.columns:
    if c != "id" and c not in sub.columns:
        sub[c] = eps
sub = sub[sample_sub.columns]

SUB_PATH = "submission_nn_kernel.csv"
sub.to_csv(SUB_PATH, index=False)

end = time.time()
print(f"\nWrote {SUB_PATH} with shape {sub.shape}")
print(round((end - start), 2), "seconds")
print(
    "Submission columns match sample_submission:",
    list(sub.columns) == list(sample_sub.columns),
)
