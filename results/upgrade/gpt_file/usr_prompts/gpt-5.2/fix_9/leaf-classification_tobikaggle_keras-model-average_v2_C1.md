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

0.02857

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.03179) has done: 'I update deprecated/removed imports and Keras API arguments so the notebook runs on your environment (scikit-learn 1.2 + Keras 3). I also fix a couple of runtime issues that would prevent creating a valid submission: incorrect column sorting (`sort()`), missing `id` column in the output, and incompatible `predict_proba()` calls. Finally, I ensure preprocessing is fit on train and applied to test (instead of refitting on test), which is a minimal correctness fix and should improve log-loss toward the target without changing the core model/training approach.'
- What this solution (achieved 0.02893) has done: 'The immediate blocker is an environment/runtime crash coming from protobuf incompatibilities that Keras can trigger during import (`MessageFactory.GetPrototype`). I fix this by forcing the safe pure-Python protobuf implementation before importing Keras, which is a minimal, score-neutral change that restores end-to-end execution. Then I keep your exact model/training logic, but add a tiny, metric-aligned post-processing step: renormalize each prediction row to sum to 1 (the metric does this anyway) and clip with the competition’s epsilon, which typically improves log-loss slightly without changing the model. Finally, I ensure the submission columns exactly match `sample_submission.csv` and write a valid `.csv` to the working directory.'
- What this solution (achieved 0.02309) has done: 'I fix the protobuf/Keras import crash by setting both protobuf environment variables *before* any Keras import, which prevents the `MessageFactory.GetPrototype` failure in Kaggle’s mixed protobuf setup. Then I keep your exact preprocessing/training/inference logic, but make execution stable across runs by also seeding Python and TensorFlow deterministically (score-neutral). Finally, I ensure the submission strictly matches `sample_submission.csv` columns (including any missing class columns filled with epsilon) and is written as a `.csv` in the working directory.'
- What this solution (achieved 0.01863) has done: 'I fix the protobuf/Keras crash by forcing the pure-Python protobuf backend *before* importing TensorFlow/tf_keras and by importing `tf_keras` earlier so TensorFlow doesn’t pull in an incompatible protobuf runtime first. Then I keep your exact preprocessing, model architectures, training loops, and ensembling logic, but make one minimal, score-improving adjustment: use `StratifiedShuffleSplit` once (still the same split strategy) and then fit each model on **all training data** for inference after choosing early-stopped weights, which typically reduces log-loss vs using only the split-trained weights while preserving the same training approach. Finally, I keep the submission formatting safeguards (column alignment to `sample_submission.csv`, per-row normalization, clipping) to ensure a valid `.csv` is always produced.'
- What this solution (achieved 0.0206) has done: 'You’re currently crashing at import time with a protobuf `MessageFactory.GetPrototype` error; the most reliable minimal fix in Kaggle is to force the pure-Python protobuf implementation *and* ensure `protobuf<4` semantics by setting env vars plus importing `google.protobuf` before TensorFlow/tf_keras. Next, to move log-loss slightly toward the target without changing the model/training approach, I add a tiny probability “smoothing” step (mixing predictions with a very small uniform prior) and then renormalize+clip; this is metric-aligned calibration and tends to reduce overconfident mistakes. I also keep submission column alignment exactly matching `sample_submission.csv` and ensure the output is written as a `.csv` in the working directory. All other core logic (models, training loops, refit-on-full) remains unchanged.'
- What this solution (achieved 0.02058) has done: 'We fix the import-time protobuf crash by forcing protobuf’s pure-Python implementation and (critically) ensuring TensorFlow/tf_keras never uses the incompatible C++ protobuf backend in this environment. This is done by setting the env vars before any protobuf/TensorFlow import and explicitly importing `google.protobuf.internal.api_implementation` early to lock the implementation. The rest of the pipeline (data prep, model definitions, training loops, refit-on-full, ensembling, and the light prediction smoothing + normalization + clipping) is kept intact to preserve evaluation semantics while aiming to improve log-loss toward the target. Finally, we ensure the submission columns exactly match `sample_submission.csv` and write a valid `.csv` file.'
- What this solution (achieved 0.02063) has done: 'I fix the remaining import-time protobuf crash by avoiding the incompatible protobuf runtime entirely: we force the pure-Python protobuf backend and import TensorFlow/tf_keras only after that lock is in place, but without touching your model/training logic. Then I make the submission generation more robust by ensuring the class-probability columns are exactly aligned to `sample_submission.csv` (including any missing columns) and that probabilities are clipped and row-normalized as required by the metric. These changes are score-neutral to slightly score-improving (better-calibrated probabilities) and are aimed at moving log-loss down toward your target while keeping your architecture/training loop identical. The script still write `submission_nn_kernel.csv` in the working directory.'
- What this solution (achieved 0.02857) has done: 'We fix the import-time crash (`MessageFactory` missing `GetPrototype`) by ensuring the protobuf **pure-Python** implementation is locked in *before* any TensorFlow/tf_keras import, and by avoiding the explicit `google.protobuf` imports that can trigger the incompatible backend in this Kaggle image. This is a runtime-stability fix and should be score-neutral. Then, to move log-loss slightly toward your target without changing the model/training approach, we tune only the tiny probability-smoothing `alpha` (calibration) to a slightly larger but still minimal value, which typically reduces overconfidence and improves multiclass log loss. Everything else (data split, scaling, model architectures, training loops, refit-on-full, ensembling, normalization/clipping, and submission column alignment) is kept intact.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import time

start = time.time()

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

import tf_keras
import tensorflow as tf
from tf_keras.models import Sequential
from tf_keras.layers import Dense, Dropout
from tf_keras.utils import to_categorical
from tf_keras.callbacks import EarlyStopping

SEED = 12345
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

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

_ = data.pop("id")
y = data.pop("species")
le = LabelEncoder()
y = le.fit_transform(y)

X_raw = data.values.astype(np.float32)
y_cat = to_categorical(y)

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=SEED)
train_index, val_index = next(iter(sss.split(X_raw, y)))
x_train_raw, x_val_raw = X_raw[train_index], X_raw[val_index]
y_train, y_val = y_cat[train_index], y_cat[val_index]

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train_raw)
x_val = scaler.transform(x_val_raw)

X_full = scaler.transform(X_raw)
y_full = y_cat

print("x_train dim:", x_train.shape)
print("x_val dim:  ", x_val.shape)
print("X_full dim: ", X_full.shape)
print("num classes:", y_cat.shape[1])
print("feature dim:", X_raw.shape[1])




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def build_model_1(input_dim=192, num_classes=99):
    m = Sequential()
    m.add(
        Dense(600, input_dim=input_dim, kernel_initializer="uniform", activation="relu")
    )
    m.add(Dropout(0.3))
    m.add(Dense(600, activation="sigmoid"))
    m.add(Dropout(0.3))
    m.add(Dense(num_classes, activation="softmax"))
    m.compile(
        loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
    )
    return m


def build_model_2(input_dim=192, num_classes=99):
    m = Sequential()
    m.add(
        Dense(
            600,
            input_dim=input_dim,
            kernel_initializer="glorot_normal",
            activation="relu",
        )
    )
    m.add(Dropout(0.1))
    m.add(Dense(300, activation="sigmoid"))
    m.add(Dropout(0.1))
    m.add(Dense(num_classes, activation="softmax"))
    m.compile(
        loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
    )
    return m


def build_model_3(input_dim=192, num_classes=99):
    m = Sequential()
    m.add(
        Dense(
            800,
            input_dim=input_dim,
            kernel_initializer="glorot_normal",
            activation="relu",
        )
    )
    m.add(Dropout(0.1))
    m.add(Dense(400, activation="sigmoid"))
    m.add(Dropout(0.1))
    m.add(Dense(num_classes, activation="softmax"))
    m.compile(
        loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
    )
    return m


INPUT_DIM = x_train.shape[1]
NUM_CLASSES = y_cat.shape[1]



## === cell 2
model1 = build_model_1(input_dim=INPUT_DIM, num_classes=NUM_CLASSES)

early_stopping = EarlyStopping(
    monitor="val_loss", patience=300, restore_best_weights=True
)
history1 = model1.fit(
    x_train,
    y_train,
    batch_size=192,
    epochs=2500,
    verbose=0,
    validation_data=(x_val, y_val),
    callbacks=[early_stopping],
)

print(
    "model1 val_acc:", float(np.nanmax(history1.history.get("val_accuracy", [np.nan])))
)
print("model1 val_loss:", float(np.nanmin(history1.history.get("val_loss", [np.nan]))))
print("model1 train_acc:", float(np.nanmax(history1.history.get("accuracy", [np.nan]))))
print("model1 train_loss:", float(np.nanmin(history1.history.get("loss", [np.nan]))))

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



## === cell 3
model2 = build_model_2(input_dim=INPUT_DIM, num_classes=NUM_CLASSES)

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

print(
    "model2 val_acc:", float(np.nanmax(history2.history.get("val_accuracy", [np.nan])))
)
print("model2 val_loss:", float(np.nanmin(history2.history.get("val_loss", [np.nan]))))
print("model2 train_acc:", float(np.nanmax(history2.history.get("accuracy", [np.nan]))))
print("model2 train_loss:", float(np.nanmin(history2.history.get("loss", [np.nan]))))

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



## === cell 4
model3 = build_model_3(input_dim=INPUT_DIM, num_classes=NUM_CLASSES)

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

print(
    "model3 val_acc:", float(np.nanmax(history3.history.get("val_accuracy", [np.nan])))
)
print("model3 val_loss:", float(np.nanmin(history3.history.get("val_loss", [np.nan]))))
print("model3 train_acc:", float(np.nanmax(history3.history.get("accuracy", [np.nan]))))
print("model3 train_loss:", float(np.nanmin(history3.history.get("loss", [np.nan]))))

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




## === cell 5
def refit_on_full_data(build_fn, x_full, y_full, x_val, y_val):
    m = build_fn(input_dim=INPUT_DIM, num_classes=NUM_CLASSES)
    cb = EarlyStopping(monitor="val_loss", patience=300, restore_best_weights=True)
    m.fit(
        x_full,
        y_full,
        batch_size=192,
        epochs=2500,
        verbose=0,
        validation_data=(x_val, y_val),
        callbacks=[cb],
    )
    return m


model1_full = refit_on_full_data(build_model_1, X_full, y_full, x_val, y_val)
model2_full = refit_on_full_data(build_model_2, X_full, y_full, x_val, y_val)
model3_full = refit_on_full_data(build_model_3, X_full, y_full, x_val, y_val)



## === cell 6
test = _read_first_existing(TEST_PATHS)
index = test.pop("id").values

X_test_raw = test.values.astype(np.float32)
X_test = scaler.transform(X_test_raw)  # IMPORTANT: do not fit on test

yPred1 = model1_full.predict(X_test, verbose=0)
yPred2 = model2_full.predict(X_test, verbose=0)
yPred3 = model3_full.predict(X_test, verbose=0)

yPred = (yPred1 + yPred2 + yPred3) / 3.0

alpha = 0.01
yPred = (1.0 - alpha) * yPred + alpha * (1.0 / yPred.shape[1])

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
print(
    "Pred min/max:",
    float(sub.drop(columns=["id"]).to_numpy().min()),
    float(sub.drop(columns=["id"]).to_numpy().max()),
)
