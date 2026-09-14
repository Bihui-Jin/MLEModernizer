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

cloudpathlib==0.21.1
cuda-pathfinder==1.3.2
geopandas==0.14.4
jmespath==1.0.1
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
path==17.1.1
path.py==12.5.0
pathos==0.3.2
pathspec==0.12.1
protobuf==6.33.0
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
testpath==0.6.0
tf_keras==2.18.0

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

25.91327

# 6. Current score

1.01167

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.00812) has done: 'I fix the protobuf/Keras import crash by using `tf.keras` consistently (avoiding the standalone `keras` import that triggers the `MessageFactory` error in this environment). Then I fix feature shape extraction for InceptionResNetV2 by reading `inception_bottleneck.output_shape` (the previous `layers[-1].output_shape` path is not valid in TF/Keras 2.18), which unblocks `h,w,d` and all downstream tensors. Finally, I replace the incorrect `path.Path` usage (`listdir`, `makedirs`) with `pathlib.Path`/`os` equivalents and ensure the script always writes a valid `submission.csv` matching `sample_submission.csv` columns and test id order.'
- What this solution (achieved 0.99458) has done: 'I fix the protobuf/Keras import crash by forcing a pure `tf.keras` backend and (critically) setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` failure in this Kaggle environment. I also correct the cell numbering to start at 1 (your current script starts at cell 0) while preserving the exact modeling/training logic and submission-building semantics. Finally, I keep the existing feature extraction + dense classifier pipeline unchanged, only adding small stability guards (directory creation and deterministic seeds) that do not change the evaluation logic, ensuring a valid `submission.csv` is always written.'
- What this solution (achieved 0.95704) has done: 'I fix the protobuf/TensorFlow crash by forcing the pure-Python protobuf implementation earlier and also ensuring TensorFlow uses the Python protobuf backend, which resolves the `MessageFactory.GetPrototype` error in TF 2.18 + protobuf 6.x environments. I keep the model, training loop, feature extraction, and submission-building logic unchanged, only adding a small compatibility/environment guard so the notebook runs end-to-end reliably. I also make the cell numbering start at 1 (your provided script starts at cell 0) and add a safety fallback for the input path resolution (some Kaggle setups mount the dataset at `../input/dog-breed-identification/` while others at `../input/dog-breed-identification/dog-breed-identification/`). These changes are score-neutral; they just unblock execution and ensure a valid `submission.csv` is always written in the required format.'
- What this solution (achieved 1.05398) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before* any TensorFlow-related imports and by removing the unsupported `TF_PROTOBUF_IMPLEMENTATION` env var that can trigger the `MessageFactory.GetPrototype` failure in TF 2.18 + protobuf 6.x. I keep your feature-extraction + dense-classifier training/inference pipeline exactly the same, only making minimal environment/import changes so it runs end-to-end. I also add a small path safety check for the test copy step (so it doesn’t try to copy into an existing directory in some rerun scenarios) while keeping the same directory structure and test ordering. These changes are intended to be score-neutral (or negligibly different) and primarily restore execution so a valid `submission.csv` is written.'
- What this solution (achieved 0.93593) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation early and also disabling the C++ protobuf backend via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION`, which is the missing piece causing `MessageFactory.GetPrototype` failures in TF 2.18 + protobuf 6.x. I also renumber your cells to start at 1 (Kaggle cell parser expects that) while keeping your feature extraction + dense classifier pipeline unchanged. Finally, I add a tiny safety fix in the test copy step so `copytree` always copies into the intended `no_class/` folder (and won’t silently create a nested structure if rerun), without changing ordering or prediction logic.'
- What this solution (achieved 1.01167) has done: 'We fix the immediate runtime crash (`MessageFactory.GetPrototype`) by setting the protobuf env vars *before* any TensorFlow import and by forcing TF to use the Python protobuf implementation in a way that is compatible with TF 2.18 + protobuf 6.x. We keep the exact same feature-extraction + dense classifier pipeline, training loop, and submission-building logic, only making minimal import/env ordering changes so the notebook runs end-to-end. Since your current score (0.93593, lower is better) is already far better than the target (25.91327), we not make any score-improving changes; we focus strictly on correctness/stability and producing a valid `submission.csv`. Finally, we renumber the cells to start at 1 (your provided script starts at 0) to match the required format.'

# 9. Code solution

## === cell 0
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import sys
import shutil
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(123)
tf.random.set_seed(123)

print("Python:", sys.version)
print("TF:", tf.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def resolve_dogbreed_root():
    candidates = [
        "../input/dog-breed-identification",
        "../input/dog-breed-identification/dog-breed-identification",
        "/kaggle/input/dog-breed-identification",
        "/kaggle/input/dog-breed-identification/dog-breed-identification",
        "/kaggle/data/dog-breed-identification",
        "/kaggle/data/dog-breed-identification/dog-breed-identification",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "labels.csv")) and (
            os.path.isdir(os.path.join(c, "train"))
            or os.path.isfile(os.path.join(c, "train.zip"))
        ):
            return c
    return "../input/dog-breed-identification"


ROOT = resolve_dogbreed_root()
print("Using dataset root:", ROOT)

dataset_dir = os.path.join(ROOT, "train")
labels = pd.read_csv(os.path.join(ROOT, "labels.csv"))

assert os.path.isdir(dataset_dir), f"Train directory not found: {dataset_dir}"
assert {"id", "breed"}.issubset(
    labels.columns
), "labels.csv must have columns: id, breed"
labels.head()




## === cell 2
def make_dir(x):
    if not os.path.exists(x):
        os.makedirs(x)


base_dir = "./subset"
make_dir(base_dir)



## === cell 3
n_class = len(labels.breed.unique())
n_class



## === cell 4
train_dir = os.path.join(base_dir, "train")
make_dir(train_dir)
val_dir = os.path.join(base_dir, "validation")
make_dir(val_dir)



## === cell 5
breeds = labels.breed.unique()
for breed in breeds:
    make_dir(os.path.join(train_dir, breed))
    make_dir(os.path.join(val_dir, breed))

for breed in breeds:
    images = labels[labels.breed == breed]["id"].values
    i = 0
    for img_id in images:
        source = os.path.join(dataset_dir, f"{img_id}.jpg")
        if not os.path.exists(source):
            continue
        if i % 10 < 2:
            destination = os.path.join(val_dir, breed, f"{img_id}.jpg")
        else:
            destination = os.path.join(train_dir, breed, f"{img_id}.jpg")
        if not os.path.exists(destination):
            shutil.copyfile(source, destination)
        i += 1

print("Prepared subset directories:", base_dir)



## === cell 6
batch_size = 64

datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = datagen.flow_from_directory(
    directory=train_dir,
    target_size=(299, 299),
    batch_size=batch_size,
    class_mode="sparse",
    shuffle=True,
    seed=123,
)

validation_generator = datagen.flow_from_directory(
    directory=val_dir,
    target_size=(299, 299),
    batch_size=batch_size,
    class_mode="sparse",
    shuffle=False,
    seed=123,
)



## === cell 7
from tensorflow.keras.applications import InceptionResNetV2

inception_bottleneck = InceptionResNetV2(
    weights="imagenet", include_top=False, input_shape=(299, 299, 3)
)

feature_shape = inception_bottleneck.output_shape[1:]
print(f"The shape of each feature tensor is: {feature_shape}")

h, w, d = feature_shape
(h, w, d)



## === cell 8
val_samples = validation_generator.n
X_val = np.zeros(shape=(val_samples, h, w, d), dtype=np.float32)
y_val = np.zeros(shape=(val_samples,), dtype=np.float32)

len_ = 0
for input_batch, label_batch in validation_generator:
    features_batch = inception_bottleneck.predict(input_batch, verbose=0)
    bs = features_batch.shape[0]
    X_val[len_ : len_ + bs] = features_batch
    y_val[len_ : len_ + bs] = label_batch
    len_ += bs
    if len_ >= val_samples:
        break

print("X_val:", X_val.shape, "y_val:", y_val.shape)



## === cell 9
train_samples = train_generator.n
X_train = np.zeros(shape=(train_samples, h, w, d), dtype=np.float32)
y_train = np.zeros(shape=(train_samples,), dtype=np.float32)

len_ = 0
for input_batch, label_batch in train_generator:
    features_batch = inception_bottleneck.predict(input_batch, verbose=0)
    bs = features_batch.shape[0]
    X_train[len_ : len_ + bs] = features_batch
    y_train[len_ : len_ + bs] = label_batch
    len_ += bs
    if len_ >= train_samples:
        break

print("X_train:", X_train.shape, "y_train:", y_train.shape)



## === cell 10
X_train = np.reshape(X_train, (train_samples, h * w * d))
print(f"Train Shape: {X_train.shape}")

X_val = np.reshape(X_val, (val_samples, h * w * d))
print(f"Validation Shape: {X_val.shape}")



## === cell 11
model_2 = models.Sequential()
model_2.add(layers.Dense(512, activation="relu", input_dim=h * w * d))
model_2.add(layers.Dropout(0.2))
model_2.add(layers.Dense(512, activation="relu"))
model_2.add(layers.Dropout(0.2))
model_2.add(layers.Dense(n_class, activation="softmax"))

model_2.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
model_2.summary()



## === cell 12
ckpt_dir = "../working/my_model"
make_dir(ckpt_dir)
checkpointer = ModelCheckpoint(
    filepath=os.path.join(ckpt_dir, "best_model.keras"),
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
)
early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=10)

epochs = 50
history = model_2.fit(
    X_train,
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_val, y_val),
    callbacks=[checkpointer, early_stop],
    verbose=1,
)



## === cell 13
model_dir = "../working/model"
make_dir(model_dir)
model_2.save(os.path.join(model_dir, "model_2.keras"))



## === cell 14
model_2.evaluate(X_val, y_val, verbose=1)



## === cell 15
for var in ["X_train", "y_train", "train_generator", "validation_generator"]:
    if var in globals():
        del globals()[var]



## === cell 16
test_dataset_dir = os.path.join(ROOT, "test")
assert os.path.isdir(test_dataset_dir), f"Test directory not found: {test_dataset_dir}"



## === cell 17
from pathlib import Path

test_dir = Path(base_dir) / "test" / "no_class"
test_dir.parent.mkdir(parents=True, exist_ok=True)

test_dir.mkdir(parents=True, exist_ok=True)
if not any(test_dir.glob("*.jpg")):
    for p in Path(test_dataset_dir).glob("*.jpg"):
        shutil.copyfile(p, test_dir / p.name)

test_files = sorted([p for p in test_dir.iterdir() if p.suffix.lower() == ".jpg"])
print("Test images:", len(test_files))



## === cell 18
datagen = ImageDataGenerator(rescale=1.0 / 255.0)

test_generator = datagen.flow_from_directory(
    directory=str(test_dir.parent),  # points to .../subset/test/
    target_size=(299, 299),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False,
)



## === cell 19
test_samples = test_generator.n
y_pred = np.zeros(shape=(test_samples, len(breeds)), dtype=np.float32)

len_ = 0
for input_batch in test_generator:
    features_batch = inception_bottleneck.predict(input_batch, verbose=0)
    features_batch = np.reshape(features_batch, (features_batch.shape[0], h * w * d))
    preds = model_2.predict(features_batch, verbose=0)
    bs = preds.shape[0]
    y_pred[len_ : len_ + bs] = preds
    len_ += bs
    if len_ >= test_samples:
        break

print("Pred shape:", y_pred.shape)



## === cell 20
sample_path = os.path.join(ROOT, "sample_submission.csv")
sample_sub = pd.read_csv(sample_path)

test_ids = [Path(fn).stem for fn in test_generator.filenames]

train_breeds_sorted = sorted([p.name for p in Path(train_dir).iterdir() if p.is_dir()])
assert (
    len(train_breeds_sorted) == y_pred.shape[1]
), "Mismatch between predicted columns and breed folders."

pred_df = pd.DataFrame(y_pred, columns=train_breeds_sorted)
pred_df.insert(0, "id", test_ids)

pred_df = pred_df[sample_sub.columns]

eps = 1e-7
prob_cols = sample_sub.columns[1:]
pred_df[prob_cols] = np.clip(pred_df[prob_cols].values, eps, 1.0 - eps)
pred_df[prob_cols] = pred_df[prob_cols].div(pred_df[prob_cols].sum(axis=1), axis=0)

out_path = "submission.csv"
pred_df.to_csv(out_path, index=False)
print("Wrote submission:", out_path, "shape:", pred_df.shape)



## === cell 21
assert out_path.endswith(".csv")
assert (
    pred_df.shape[0] == sample_sub.shape[0]
), "Submission row count must match sample submission."
assert list(pred_df.columns) == list(
    sample_sub.columns
), "Submission columns must exactly match sample submission."
pred_df.head()
