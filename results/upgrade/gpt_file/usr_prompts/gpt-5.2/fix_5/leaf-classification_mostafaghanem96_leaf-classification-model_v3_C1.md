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

3.11

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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

0.06927

# 6. Current score

0.14685

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.11441) has done: 'I fix the environment-breaking import issue by switching from standalone `keras` to `tf.keras`, which avoids the protobuf `MessageFactory` crash in this Kaggle setup. I also fix the stratified split error by increasing the validation fraction so the validation fold has at least one sample per class, and then ensure the same scaler fit on train is used to transform test (the current code incorrectly fits a new scaler on test, which hurts logloss badly). I restore missing symbols (`np_utils`, `ModelCheckpoint`) via `tf.keras.utils.to_categorical` and `tf.keras.callbacks.ModelCheckpoint`, and update the `Convolution1D` call to the modern `Conv1D` API. Finally, I keep the model/training core intact, write a valid `submission.csv`, and make the optional TFLite export robust (skip if the saved model file isn’t present).'
- What this solution (achieved 0.1081) has done: 'We need to fix the protobuf `MessageFactory.GetPrototype` crash that happens at import time in this Kaggle environment; the minimal safe way is to force the pure‑Python protobuf implementation before importing TensorFlow. Then, to move logloss toward the target without changing the core model/training loop, we (a) add a tiny epsilon clip to predictions to respect the metric’s numerical stability and (b) average predictions from the in-memory final model and the best checkpoint when available, which is a score-improving calibration/variance-reduction step without altering architecture or training. Finally, we ensure the submission columns exactly match `sample_submission.csv` and write a valid `submission.csv`.'
- What this solution (achieved 0.12188) has done: 'We need to fix the import-time protobuf crash (`MessageFactory.GetPrototype`) so the notebook can run end-to-end; the most reliable minimal fix in this Kaggle TF 2.18 setup is to force the Python protobuf implementation *and* force protobuf version 3 API before importing TensorFlow. Then we keep your model/training exactly the same, but make the checkpoint save/load robust under Keras 3 by saving weights-only (avoids HDF5/serialization edge cases) and loading them back into the same architecture for ensembling. Finally, we keep the same scaler usage and submission column alignment, and ensure `submission.csv` is always written with the exact sample-submission columns.'
- What this solution (achieved 0.14685) has done: 'We fix the import-time protobuf crash by forcing the pure-Python protobuf implementation *and* importing TensorFlow only after that environment is set, plus adding a safe fallback that restarts TF import with the alternate implementation if the crash still occurs. Then we keep your model, training loop, scaler, and ensembling logic unchanged, but make the checkpoint cloning robust by building/compiling the cloned model before loading weights (avoids silent shape/build issues). Finally, we ensure submission columns match `sample_submission.csv` exactly and that a valid `submission.csv` is always written.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

try:
    import tensorflow as tf
except AttributeError as e:
    if "MessageFactory" in repr(e) or "GetPrototype" in repr(e):
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"
        import importlib, sys

        for m in list(sys.modules.keys()):
            if m.startswith(("tensorflow", "google.protobuf")):
                sys.modules.pop(m, None)
        tf = importlib.import_module("tensorflow")
    else:
        raise

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, Conv1D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint

np.random.seed(23)
tf.random.set_seed(23)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_data = pd.read_csv("../input/leaf-classification/train.csv.zip")
test_data = pd.read_csv("../input/leaf-classification/test.csv.zip")




## === cell 2
def encode(train, test):
    label_encoder = LabelEncoder().fit(train.species)
    labels = label_encoder.transform(train.species)
    classes = list(label_encoder.classes_)

    train = train.drop(["species", "id"], axis=1)
    test_ids = test.id.values
    test = test.drop("id", axis=1)

    return train, labels, test, classes, test_ids




## === cell 3
train, labels, test, classes, test_ids = encode(train_data, test_data)



## === cell 4
test_data.head()



## === cell 5
scaler = StandardScaler().fit(train.values)
scaled_train = scaler.transform(train.values)



## === cell 6
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=23)
for train_index, valid_index in sss.split(scaled_train, labels):
    X_train, X_valid = scaled_train[train_index], scaled_train[valid_index]
    y_train, y_valid = labels[train_index], labels[valid_index]



## === cell 7
num_features = 64  # number of features per feature type (shape, texture, margin)
num_class = len(classes)



## === cell 8
X_train_r = np.zeros((len(X_train), num_features, 3), dtype=np.float32)
X_train_r[:, :, 0] = X_train[:, :num_features]
X_train_r[:, :, 1] = X_train[:, num_features:128]
X_train_r[:, :, 2] = X_train[:, 128:]

X_valid_r = np.zeros((len(X_valid), num_features, 3), dtype=np.float32)
X_valid_r[:, :, 0] = X_valid[:, :num_features]
X_valid_r[:, :, 1] = X_valid[:, num_features:128]
X_valid_r[:, :, 2] = X_valid[:, 128:]



## === cell 9
model = Sequential()
model.add(Conv1D(512, 1, input_shape=(num_features, 3)))
model.add(Activation("relu"))
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(2048, activation="relu"))
model.add(Dense(1024, activation="relu"))
model.add(Dense(num_class))
model.add(Activation("softmax"))

y_train_oh = keras.utils.to_categorical(y_train, num_class)
y_valid_oh = keras.utils.to_categorical(y_valid, num_class)

sgd = SGD(learning_rate=0.01, nesterov=True, momentum=0.9, weight_decay=1e-6)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])



## === cell 10
best_model_file = "leafCNN.weights.h5"
best_model = ModelCheckpoint(
    best_model_file,
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
)

nb_epoch = 15
history = model.fit(
    X_train_r,
    y_train_oh,
    epochs=nb_epoch,
    validation_data=(X_valid_r, y_valid_oh),
    batch_size=16,
    callbacks=[best_model],
    verbose=2,
)

print("val_acc: ", max(history.history.get("val_accuracy", [np.nan])))
print("val_loss: ", min(history.history.get("val_loss", [np.nan])))
print("train_acc: ", max(history.history.get("accuracy", [np.nan])))
print("train_loss: ", min(history.history.get("loss", [np.nan])))

if (
    len(history.history.get("val_loss", [])) > 0
    and len(history.history.get("loss", [])) > 0
):
    print()
    print(
        "train/val loss ratio: ",
        min(history.history["loss"]) / min(history.history["val_loss"]),
    )



## === cell 11
scaled_test = scaler.transform(test.values)



## === cell 12
test_dataset = np.zeros((len(scaled_test), num_features, 3), dtype=np.float32)
test_dataset[:, :, 0] = scaled_test[:, :num_features]
test_dataset[:, :, 1] = scaled_test[:, num_features:128]
test_dataset[:, :, 2] = scaled_test[:, 128:]



## === cell 13
preds_final = model.predict(test_dataset, verbose=0)

preds_best = None
if os.path.exists(best_model_file):
    try:
        best_loaded = keras.models.clone_model(model)
        best_loaded.compile(
            loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"]
        )
        _ = best_loaded.predict(test_dataset[:1], verbose=0)  # build weights
        best_loaded.load_weights(best_model_file)
        preds_best = best_loaded.predict(test_dataset, verbose=0)
    except Exception as e:
        print(
            "Warning: could not load/predict with best model checkpoint weights. Error:",
            repr(e),
        )

if preds_best is not None and preds_best.shape == preds_final.shape:
    preds_test = 0.5 * preds_final + 0.5 * preds_best
else:
    preds_test = preds_final

eps = 1e-15
preds_test = np.clip(preds_test, eps, 1.0 - eps)

preds_test



## === cell 14
submission = pd.DataFrame(preds_test, columns=classes)
submission.insert(0, "id", test_ids)

sample_sub = pd.read_csv("../input/leaf-classification/sample_submission.csv.zip")
submission = submission.reindex(columns=sample_sub.columns)

submission



## === cell 15
submission.to_csv("submission.csv", index=False)
print("done! wrote submission.csv with shape:", submission.shape)



## === cell 16
model_weights_path = "/kaggle/working/leafCNN.weights.h5"
if os.path.exists(model_weights_path):
    try:
        export_model = keras.models.clone_model(model)
        export_model.compile(loss="categorical_crossentropy", optimizer=sgd)
        _ = export_model.predict(test_dataset[:1], verbose=0)  # build
        export_model.load_weights(model_weights_path)

        converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
        tflite_model = converter.convert()

        tflite_model_path = "leafCNN.tflite"
        with open(tflite_model_path, "wb") as f:
            f.write(tflite_model)
        print("TFLite model written to", tflite_model_path)
    except Exception as e:
        print("Skipping TFLite export due to error:", repr(e))
else:
    print("Skipping TFLite export; checkpoint not found at", model_weights_path)
