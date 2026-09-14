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
Given a dataset of images of dogs and cats, predict if an image is a dog or a cat.

## Metric
Log loss.

## Submission Format
For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:

```
id,label
1,0.5
2,0.5
3,0.5
...
```

## Dataset
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id.

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-image==0.25.2
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
        input/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
        working/
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

# 5. Target score

0.55889

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.71851) has done: 'I update the imports and Keras API calls to be compatible with the installed `keras==3.x` (fixing the `ImageDataGenerator`, `lr`, and `fit_generator/evaluate_generator` errors) without changing the model’s core architecture or training approach. I also fix all dataset paths to match your actual folder structure (`/kaggle/input/dogs-vs-cats-redux-kernels-edition/train/` and `.../test/unknown/`), which is the root cause of the image loading and “reading folders” errors. Finally, I generate the submission from the provided `sample_submission.csv` to guarantee correct `id,label` columns and id alignment, and write a valid `submission_file.csv`.'
- What this solution (achieved 0.17014) has done: 'I fix the runtime break by using a single Keras stack consistently: `tf_keras` for both the `ImageDataGenerator` iterators and the model, because Keras 3 can’t train on `tf_keras` iterators (that’s what caused the “Unrecognized data type” errors). I also avoid the protobuf-related import crash by importing preprocessing from `tf_keras` after setting the protobuf implementation to `python`, which is the safest workaround in Kaggle images when that specific `MessageFactory` error appears. To move the logloss score toward your target (lower is better) with minimal semantics change, I only increase the number of steps per epoch to cover the full training split instead of training on just 5 mini-batches, keeping the same architecture, loss, optimizer, and number of epochs. The submission writing remains aligned to `sample_submission.csv` to guarantee correct `id,label` format and ordering.'
- What this solution (achieved 0.67484) has done: 'I fix the protobuf crash that happens when importing `tf_keras.preprocessing.image.ImageDataGenerator` by avoiding that module entirely and switching the generators to the stable `tf.data` + `tf_keras.utils.image_dataset_from_directory` pipeline (same semantics: rescale to [0,1], binary labels, train/val split). I keep the VGG16 transfer-learning model, optimizer, loss, and training loop intact, only changing the input pipeline so the notebook runs end-to-end in this environment. I also ensure the submission IDs align exactly to `sample_submission.csv` by reading that file and predicting in the same order, writing a valid `submission_file.csv`. These changes are expected to improve logloss from the current overfit/buggy state toward your target by training on the full dataset reliably and producing correctly-aligned probabilities.'
- What this solution (achieved 0.73056) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation *before* importing TensorFlow (the env var must be set before those imports). I fix the dataset loading error by pointing `image_dataset_from_directory` at the actual leaf directory that only contains `cat/` and `dog/` subfolders (your `TRAIN_DIR` currently contains an extra nested `train/` directory that breaks `class_names`). Finally, I make the holdout split deterministic and consistent by using the same directory-based split for both training and evaluation (so `build_batches` uses the same files as the Keras validation subset), which should improve logloss toward the target without changing the model architecture, loss, optimizer, or number of epochs, and still write a valid `submission_file.csv`.'
- What this solution (achieved 0.81677) has done: 'I fix the two root causes that prevent your notebook from running: (1) the protobuf `MessageFactory.GetPrototype` crash by setting the required environment variables before importing TensorFlow, and (2) the broken dataset paths so `image_dataset_from_directory` actually sees `cat/` and `dog/` subfolders. To keep your core model/training logic unchanged, I won’t alter the VGG16-based architecture, loss, optimizer, or the 10-epoch training loop; I only correct I/O and make the train/val split deterministic. I also make the test directory robust to the extra nested `test/test/unknown` layout and ensure the submission is written as a valid `submission_file.csv` with `id,label` aligned to `sample_submission.csv`. These fixes are expected to improve log loss from the current broken/misaligned run toward your target by ensuring the model trains on the intended data and predictions map to the correct test IDs.'
- What this solution (achieved 0.81677) has done: 'I fix the protobuf-related TensorFlow import crash by setting the required environment variables (and a couple of safe TF flags) before importing TensorFlow, then importing `tensorflow`/`tf_keras` in a guarded way so the notebook always runs. I also fix the dataset path bug causing `image_dataset_from_directory` to see an unexpected extra `train/` subfolder by selecting the correct leaf directory that contains only `cat/` and `dog/`. These changes unblock `train_generator`/`validation_generator` creation so the existing VGG16 transfer-learning model trains as intended and produces predictions aligned to `sample_submission.csv`. The submission writing remain unchanged in format (`id,label`) and always produce `submission_file.csv`.'
- What this solution (achieved 0.81677) has done: 'I fix the protobuf/TensorFlow import crash by setting the environment variables before any TensorFlow-related import, and I add a robust fallback that uses `tf_keras` only if `tensorflow` import fails in this environment. I also fix the train directory resolution so it points to the true leaf folder that contains only `cat/` and `dog/` (excluding an extra nested `train/` folder), which is what caused the `class_names` mismatch and the downstream `NameError`s. These are correctness/stability fixes that unblock end-to-end training and prediction without changing your VGG16 transfer-learning architecture, optimizer, loss, or 10-epoch training loop. Finally, I keep submission creation aligned to `sample_submission.csv` and always write a valid `submission_file.csv` with `id,label`.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

BASE_PATH = "/kaggle/input/dogs-vs-cats-redux-kernels-edition"


def resolve_leaf_train_dir(base_path: str) -> str:
    """
    FIX: Resolve to the directory that contains class folders cat/ and dog/
    and does NOT contain any extra nested 'train' directory alongside them.
    This avoids class_names mismatch where Keras detects ['cat','dog','train'].
    """
    candidates = [
        os.path.join(base_path, "train"),
        os.path.join(base_path, "train", "train"),
    ]
    for c in candidates:
        if not os.path.isdir(c):
            continue
        subdirs = sorted(
            [d for d in os.listdir(c) if os.path.isdir(os.path.join(c, d))]
        )
        if ("cat" in subdirs) and ("dog" in subdirs) and ("train" not in subdirs):
            return c

    for c in candidates:
        if not os.path.isdir(c):
            continue
        for d in os.listdir(c):
            p = os.path.join(c, d)
            if not os.path.isdir(p):
                continue
            subdirs = sorted(
                [sd for sd in os.listdir(p) if os.path.isdir(os.path.join(p, sd))]
            )
            if ("cat" in subdirs) and ("dog" in subdirs) and ("train" not in subdirs):
                return p

    raise FileNotFoundError(
        "Could not resolve a valid TRAIN_DIR with only cat/ and dog/ subfolders."
    )


def resolve_test_dir(base_path: str) -> str:
    candidates = [
        os.path.join(base_path, "test", "unknown"),
        os.path.join(base_path, "test", "test", "unknown"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            try:
                if any(
                    fn.lower().endswith((".jpg", ".jpeg", ".png"))
                    for fn in os.listdir(c)
                ):
                    return c
            except Exception:
                pass
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError("Could not resolve a valid TEST_DIR.")


TRAIN_DIR = resolve_leaf_train_dir(BASE_PATH)
TEST_DIR = resolve_test_dir(BASE_PATH)
SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")

print("BASE_PATH exists:", os.path.exists(BASE_PATH))
print("Resolved TRAIN_DIR:", TRAIN_DIR, "exists:", os.path.exists(TRAIN_DIR))
print(
    "TRAIN_DIR subdirs:",
    sorted(
        [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
    ),
)
print("Resolved TEST_DIR:", TEST_DIR, "exists:", os.path.exists(TEST_DIR))
print("Sample submission exists:", os.path.exists(SAMPLE_SUB_PATH), SAMPLE_SUB_PATH)

assert os.path.isfile(SAMPLE_SUB_PATH), "sample_submission.csv not found."



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1234803896.py in <cell line: 0>()
     73 
     74 
---> 75 TRAIN_DIR = resolve_leaf_train_dir(BASE_PATH)
     76 TEST_DIR = resolve_test_dir(BASE_PATH)
     77 SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")

/tmp/ipykernel_11/1234803896.py in resolve_leaf_train_dir(base_path)
     47                 return p
     48 
---> 49     raise FileNotFoundError(
     50         "Could not resolve a valid TRAIN_DIR with only cat/ and dog/ subfolders."
     51     )

FileNotFoundError: Could not resolve a valid TRAIN_DIR with only cat/ and dog/ subfolders.

## === cell 1
import tf_keras

print("tf_keras version:", getattr(tf_keras, "__version__", "unknown"))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
from os import listdir

train_data = []
test_data = []

cat_dir = os.path.join(TRAIN_DIR, "cat")
dog_dir = os.path.join(TRAIN_DIR, "dog")

cat_files = [
    f for f in listdir(cat_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
]
dog_files = [
    f for f in listdir(dog_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

for file in cat_files:
    some_number = random.randint(1, 100)
    label = "0"
    if some_number < 85:
        train_data.append([file, label, "cat"])
    else:
        test_data.append([file, label, "cat"])

for file in dog_files:
    some_number = random.randint(1, 100)
    label = "1"
    if some_number < 85:
        train_data.append([file, label, "dog"])
    else:
        test_data.append([file, label, "dog"])

train = pd.DataFrame(train_data, columns=["filename", "class", "folder"])
test = pd.DataFrame(test_data, columns=["filename", "class", "folder"])

train.head(10)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3346116403.py in <cell line: 0>()
      4 test_data = []
      5 
----> 6 cat_dir = os.path.join(TRAIN_DIR, "cat")
      7 dog_dir = os.path.join(TRAIN_DIR, "dog")
      8 

NameError: name 'TRAIN_DIR' is not defined

## === cell 3
test.head(10)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2348734731.py in <cell line: 0>()
----> 1 test.head(10)
      2 

NameError: name 'test' is not defined

## === cell 4
print("Train size", len(train))
print("Test size", len(test))

for label in ["0", "1"]:
    print("------------")
    print("\tTrain has", len(train[train["class"] == label]), label)
    print("\tTest has", len(test[test["class"] == label]), label)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3257751539.py in <cell line: 0>()
----> 1 print("Train size", len(train))
      2 print("Test size", len(test))
      3 
      4 for label in ["0", "1"]:
      5     print("------------")

NameError: name 'train' is not defined

## === cell 5
IMAGE_WIDTH = 96
IMAGE_HEIGHT = 96
BATCH_SIZE = 32

VALIDATION_SPLIT = 0.15

train_ds = tf_keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=["cat", "dog"],
    image_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    validation_split=VALIDATION_SPLIT,
    subset="training",
)

val_ds = tf_keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="binary",
    class_names=["cat", "dog"],
    image_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
)

rescale = tf_keras.layers.Rescaling(1.0 / 255.0)
train_ds = train_ds.map(lambda x, y: (rescale(x), y))
val_ds = val_ds.map(lambda x, y: (rescale(x), y))

train_generator = train_ds
validation_generator = val_ds

print("train batches (iterator len):", len(train_ds))
print("val batches (iterator len):", len(val_ds))



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4239994084.py in <cell line: 0>()
      7 # FIX: Now TRAIN_DIR is guaranteed not to contain an extra 'train' directory, so class_names matches.
      8 train_ds = tf_keras.utils.image_dataset_from_directory(
----> 9     TRAIN_DIR,
     10     labels="inferred",
     11     label_mode="binary",

NameError: name 'TRAIN_DIR' is not defined

## === cell 6
from tf_keras.applications import vgg16

model = vgg16.VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3),
    pooling="max",
)



## === cell 7
for layer in model.layers[:-5]:
    layer.trainable = False



## === cell 8
from tf_keras.layers import Dense
from tf_keras.models import Sequential

transfer_model = Sequential()
for layer in model.layers:
    transfer_model.add(layer)
transfer_model.add(Dense(512, activation="relu"))
transfer_model.add(Dense(1, activation="sigmoid"))



## === cell 9
from tf_keras import optimizers

adam = optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
transfer_model.compile(adam, loss="binary_crossentropy", metrics=["accuracy"])



## === cell 10
model_history = transfer_model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10,
    verbose=1,
)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/705402691.py in <cell line: 0>()
      1 model_history = transfer_model.fit(
----> 2     train_generator,
      3     validation_data=validation_generator,
      4     epochs=10,
      5     verbose=1,

NameError: name 'train_generator' is not defined

## === cell 11
transfer_model.evaluate(validation_generator, verbose=1)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1064694668.py in <cell line: 0>()
----> 1 transfer_model.evaluate(validation_generator, verbose=1)
      2 

NameError: name 'validation_generator' is not defined

## === cell 12
import cv2
from skimage import io
from tf_keras.utils import image_dataset_from_directory


def dataframe_from_directory_validation_split(
    directory, validation_split=0.15, subset="validation", seed=42
):
    ds = image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="binary",
        class_names=["cat", "dog"],
        image_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=seed,
        validation_split=validation_split,
        subset=subset,
    )
    file_paths = ds.file_paths
    rows = []
    for p in file_paths:
        folder = os.path.basename(os.path.dirname(p))
        filename = os.path.basename(p)
        label = "1" if folder == "dog" else "0"
        rows.append([filename, label, folder])
    return pd.DataFrame(rows, columns=["filename", "class", "folder"])


val_df = dataframe_from_directory_validation_split(
    TRAIN_DIR, VALIDATION_SPLIT, subset="validation", seed=SEED
)
print("Recovered val_df:", val_df.shape)
val_df.head()




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/937340209.py in <cell line: 0>()
     31 
     32 val_df = dataframe_from_directory_validation_split(
---> 33     TRAIN_DIR, VALIDATION_SPLIT, subset="validation", seed=SEED
     34 )
     35 print("Recovered val_df:", val_df.shape)

NameError: name 'TRAIN_DIR' is not defined

## === cell 13
def build_batches(df, has_labels=True, limit=500):
    X = []
    y = []
    i = 0

    for _, row in df.iterrows():
        if has_labels:
            y.append(row["class"])

        if has_labels:
            raw_image_path = os.path.join(TRAIN_DIR, row["folder"], row["filename"])
        else:
            raw_image_path = os.path.join(TEST_DIR, row["filename"])

        raw_image = io.imread(raw_image_path)

        if raw_image.ndim == 2:
            raw_image = np.stack([raw_image] * 3, axis=-1)
        elif raw_image.shape[-1] == 4:
            raw_image = raw_image[:, :, :3]

        raw_image = cv2.resize(
            raw_image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_CUBIC
        )
        X.append(raw_image)

        i += 1
        if i % 500 == 0:
            print("Done", i, "images")
        if limit != -1 and i == limit:
            break

    X = np.array(X, dtype=np.float32) / 255.0
    y = np.array(y).astype(np.float32) if has_labels else None
    return X, y


X_val, y_val = build_batches(val_df, has_labels=True, limit=-1)
print(X_val.shape, y_val.shape)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1117939479.py in <cell line: 0>()
     36 
     37 
---> 38 X_val, y_val = build_batches(val_df, has_labels=True, limit=-1)
     39 print(X_val.shape, y_val.shape)
     40 

NameError: name 'val_df' is not defined

## === cell 14
y_hat = transfer_model.predict(X_val, verbose=1)
print(y_hat.shape)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/993231011.py in <cell line: 0>()
----> 1 y_hat = transfer_model.predict(X_val, verbose=1)
      2 print(y_hat.shape)
      3 

NameError: name 'X_val' is not defined

## === cell 15
from sklearn.metrics import log_loss

print(
    "Holdout (Keras val subset) log loss:",
    log_loss(y_val.astype(np.float32), y_hat.reshape(-1)),
)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2684897632.py in <cell line: 0>()
      3 print(
      4     "Holdout (Keras val subset) log loss:",
----> 5     log_loss(y_val.astype(np.float32), y_hat.reshape(-1)),
      6 )
      7 

NameError: name 'y_val' is not defined

## === cell 16
transfer_model.evaluate(X_val, y_val.astype(np.float32), verbose=1)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4236061893.py in <cell line: 0>()
----> 1 transfer_model.evaluate(X_val, y_val.astype(np.float32), verbose=1)
      2 

NameError: name 'X_val' is not defined

## === cell 17
sub = pd.read_csv(SAMPLE_SUB_PATH)
sub["filename"] = sub["id"].astype(str) + ".jpg"
sub.head()



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2623843672.py in <cell line: 0>()
----> 1 sub = pd.read_csv(SAMPLE_SUB_PATH)
      2 sub["filename"] = sub["id"].astype(str) + ".jpg"
      3 sub.head()
      4 

NameError: name 'SAMPLE_SUB_PATH' is not defined

## === cell 18
X_out, _ = build_batches(sub[["filename"]], has_labels=False, limit=-1)
print("Inference tensor:", X_out.shape)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2232682109.py in <cell line: 0>()
----> 1 X_out, _ = build_batches(sub[["filename"]], has_labels=False, limit=-1)
      2 print("Inference tensor:", X_out.shape)
      3 

NameError: name 'sub' is not defined

## === cell 19
results = transfer_model.predict(X_out, verbose=1).reshape(-1)
print("Preds:", results.shape, float(results.min()), float(results.max()))



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3498116738.py in <cell line: 0>()
----> 1 results = transfer_model.predict(X_out, verbose=1).reshape(-1)
      2 print("Preds:", results.shape, float(results.min()), float(results.max()))
      3 

NameError: name 'X_out' is not defined

## === cell 20
sub["label"] = results.astype(float)
sub["label"] = sub["label"].clip(1e-7, 1 - 1e-7)
sub[["id", "label"]].head()



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/887430437.py in <cell line: 0>()
----> 1 sub["label"] = results.astype(float)
      2 sub["label"] = sub["label"].clip(1e-7, 1 - 1e-7)
      3 sub[["id", "label"]].head()
      4 

NameError: name 'results' is not defined

## === cell 21
out_path = "submission_file.csv"
sub[["id", "label"]].to_csv(out_path, index=False)
print("Wrote", out_path, "with shape:", sub[["id", "label"]].shape)
print(sub[["id", "label"]].columns.tolist())
print(sub[["id", "label"]].head())

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2160536554.py in <cell line: 0>()
      1 out_path = "submission_file.csv"
----> 2 sub[["id", "label"]].to_csv(out_path, index=False)
      3 print("Wrote", out_path, "with shape:", sub[["id", "label"]].shape)
      4 print(sub[["id", "label"]].columns.tolist())
      5 print(sub[["id", "label"]].head())

NameError: name 'sub' is not defined
