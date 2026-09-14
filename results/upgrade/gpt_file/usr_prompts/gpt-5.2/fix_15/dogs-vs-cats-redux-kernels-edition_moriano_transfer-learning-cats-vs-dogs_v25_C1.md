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

0.22786

# 6. Current score

0.09504

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.06312) has done: 'The timeout is dominated by slow Python-side image loading/resize in `build_batches()` (skimage+iterrows) plus extra non-essential work (per-image plotting, manual evaluate/predict loops) that repeats I/O. I keep the same model, training loop, augmentations, epochs, and prediction semantics, but make data input significantly faster by (1) using Keras generators’ built-in multiprocessing/prefetch and (2) rewriting `build_batches()` to use OpenCV decoding and `itertuples()` with preallocated NumPy arrays (same resize/scale). I also disable the expensive display/plot cells by default (they don’t affect submission accuracy) and avoid redundant validation “test” generator augmentation mismatch by keeping it unchanged but making it faster. All paths remain identical; outputs and training behavior remain equivalent aside from negligible float differences.'
- What this solution (achieved 0.73718) has done: 'The timeout is dominated by Python-level image loading/augmentation in `ImageDataGenerator` and your custom OpenCV generator, which both read/resize images on the fly with minimal pipelining. I keep the same VGG16 transfer model, same train/val split, same augmentations, same epochs/optimizer/loss, but switch the input pipelines to `tf.data` with parallel decode/resize, caching, and prefetch so the GPU/CPU stays busy. I also remove redundant array copies in the batch builder (used for prediction) and make prediction use a `tf.data` pipeline too, preserving identical rescale/resize semantics. These changes are equivalent in outputs (up to negligible float differences) but cut wall time substantially by reducing Python overhead and enabling parallelism.'
- What this solution (achieved 0.0994) has done: 'I fix the TensorFlow/Keras import mismatch that is causing the `MessageFactory.GetPrototype` crash by using `tf_keras` consistently and avoiding the standalone `tensorflow` import that triggers the protobuf issue in this environment. I also fix the missing `tf.image.rotate` by switching the rotation augmentation to a Keras-native layer (`tf_keras.layers.RandomRotation`) so the augmentation still exists and stays within the same training semantics (random rotation + flip) without adding new modeling logic. These changes unblock dataset creation, training, and inference so the notebook runs end-to-end and writes a valid `submission_file.csv`. Finally, I keep the same VGG16 transfer setup, optimizer, loss, epochs, and general preprocessing; the only score-affecting change is restoring working augmentation so performance should move down (improve) from the current 0.737 logloss toward the target.'
- What this solution (achieved 0.08928) has done: 'I fix the crash in `cell 2` by avoiding the problematic `keras.utils.set_random_seed()` call (it triggers the protobuf `MessageFactory.GetPrototype` error in this Kaggle image) and by importing `tensorflow` once up front so `tf.data` and `tf_keras` can coexist safely. I keep your model, training loop, augmentations, preprocessing, and submission logic unchanged so the score impact is negligible (your current score is already better than the target band for a lower-is-better metric). I also make `AUTOTUNE` robust and ensure the script always finds the correct `TEST_DIR` and writes a valid `submission_file.csv`. Everything else remains the same to preserve evaluation semantics.'
- What this solution (achieved 0.08608) has done: 'I fix the root cause of the early crash (`MessageFactory.GetPrototype`) by avoiding the `keras.backend.tf` access pattern and instead importing TensorFlow explicitly in a protobuf-safe way before using `tf_keras`. Then I make `tf` available consistently across cells so the `@tf.function` pipelines and test-time `tf.data` prediction work, which also resolves the downstream `NameError`s for `tf`, `train_ds`, and `results`. I keep your model, augmentation, training loop, and preprocessing semantics the same (VGG16 transfer learning, binary crossentropy, 5 epochs, resize to 224, scale to [0,1]). Finally, I ensure the submission is always written as `submission_file.csv` with the required `id,label` columns.'
- What this solution (achieved 0.11151) has done: 'I fix the early crash caused by the TensorFlow/protobuf incompatibility that triggers `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` by ensuring TensorFlow is imported in a safer way and forcing the pure-Python protobuf implementation before TF loads. I keep your model, preprocessing, tf.data pipelines, epochs, optimizer, and submission logic unchanged so score impact is negligible (your current score is already better than the target for a lower-is-better metric). I also make the dataset directory resolution slightly more robust (but path semantics unchanged) so the code reliably finds `TRAIN_DIR`/`TEST_DIR` in this Kaggle layout and always writes `submission_file.csv`. Finally, I keep determinism settings as-is where possible without calling APIs known to trigger the protobuf crash in this environment.'
- What this solution (achieved 0.11078) has done: 'The crash happens before any training because TensorFlow is failing to import due to a protobuf/TensorFlow incompatibility (`MessageFactory.GetPrototype`). The smallest stable fix in Kaggle notebooks is to force TensorFlow to use the pure-Python protobuf implementation *and* disable the C++ protobuf runtime via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` and `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=3`, set **before** importing TensorFlow; then import `tf_keras` normally. This is score-neutral (it just unblocks runtime) and preserves your model, data pipelines, augmentation, epochs, and submission formatting unchanged. I also keep your directory resolution and submission-writing logic intact so it still produces `submission_file.csv` with `id,label`.'
- What this solution (achieved 0.07832) has done: 'Your notebook crashes immediately due to a known TensorFlow/protobuf incompatibility triggered by `tf = keras.backend.tf`; fixing that by importing TensorFlow directly (after forcing pure-Python protobuf) unblocks all downstream cells where `tf`, `TRAIN_DIR`, `TEST_DIR`, and datasets were undefined. I keep the exact same VGG16 transfer model, epochs, optimizer, loss, and tf.data preprocessing/augmentation logic, changing only the imports and TensorFlow handle so execution is stable. I also make the base directory resolution slightly more robust (without changing path semantics) so it consistently finds the provided Kaggle folders. Finally, I ensure a valid `submission_file.csv` is always written with `id,label` and aligned ordering.'
- What this solution (achieved 0.09504) has done: 'I fix the immediate runtime crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation *before any TensorFlow-related import happens*, and by importing `tf_keras` first (so we never trigger the problematic standalone Keras/TensorFlow import path). This is score-neutral (it only unblocks execution) and keeps your exact model, augmentation, preprocessing, training loop, and submission formatting unchanged. I also make the TensorFlow import more defensive (so the notebook doesn’t die if TF isn’t available) while still producing the required `submission_file.csv` when it is. No architectural/training changes are introduced, so your score should remain in the same regime (already better than the target band for a lower-is-better metric).'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"

os.environ.setdefault("PYTHONHASHSEED", "42")
random.seed(42)
np.random.seed(42)

import tf_keras as keras

try:
    tf = keras.backend.tf
except Exception:
    import tensorflow as tf  # fallback, but normally keras.backend.tf works here

BASE_DIR = "/kaggle/input/dogs-vs-cats-redux-kernels-edition"

TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test", "unknown")

if not os.path.exists(TRAIN_DIR):
    for cd in [
        os.path.join(BASE_DIR, "dogs-vs-cats-redux-kernels-edition", "train"),
        os.path.join("/kaggle/input", "dogs-vs-cats-redux-kernels-edition", "train"),
        os.path.join("/kaggle/data", "dogs-vs-cats-redux-kernels-edition", "train"),
    ]:
        if os.path.exists(cd):
            TRAIN_DIR = cd
            break

if not os.path.exists(TEST_DIR):
    candidate_dirs = [
        os.path.join(BASE_DIR, "test", "unknown"),
        os.path.join(BASE_DIR, "test", "test", "unknown"),
        os.path.join(BASE_DIR, "dogs-vs-cats-redux-kernels-edition", "test", "unknown"),
        os.path.join(
            BASE_DIR, "dogs-vs-cats-redux-kernels-edition", "test", "test", "unknown"
        ),
        os.path.join(
            "/kaggle/input", "dogs-vs-cats-redux-kernels-edition", "test", "unknown"
        ),
        os.path.join(
            "/kaggle/input",
            "dogs-vs-cats-redux-kernels-edition",
            "test",
            "test",
            "unknown",
        ),
        os.path.join(
            "/kaggle/data", "dogs-vs-cats-redux-kernels-edition", "test", "unknown"
        ),
        os.path.join(
            "/kaggle/data",
            "dogs-vs-cats-redux-kernels-edition",
            "test",
            "test",
            "unknown",
        ),
    ]
    for cd in candidate_dirs:
        if os.path.exists(cd):
            TEST_DIR = cd
            break

print("BASE_DIR exists:", os.path.exists(BASE_DIR))
print("TRAIN_DIR exists:", os.path.exists(TRAIN_DIR))
print("TEST_DIR exists:", os.path.exists(TEST_DIR))
print(
    "TRAIN_DIR subdirs:",
    os.listdir(TRAIN_DIR)[:10] if os.path.exists(TRAIN_DIR) else None,
)
print(
    "TEST_DIR sample:", os.listdir(TEST_DIR)[:10] if os.path.exists(TEST_DIR) else None
)

AUTOTUNE = getattr(tf.data, "AUTOTUNE", None)
if AUTOTUNE is None:
    AUTOTUNE = tf.data.experimental.AUTOTUNE



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from os import listdir

train_data = []
for cls in ["cat", "dog"]:
    cls_dir = os.path.join(TRAIN_DIR, cls)
    label = "1" if cls == "dog" else "0"
    for file in listdir(cls_dir):
        rel_path = f"{cls}/{file}"  # relative to TRAIN_DIR
        train_data.append([rel_path, label])

df_all = pd.DataFrame(train_data, columns=["filename", "class"])
df_all = df_all.sample(frac=1.0, random_state=42).reset_index(drop=True)
split_idx = int(len(df_all) * 0.85)
train = df_all.iloc[:split_idx].copy()
test = df_all.iloc[split_idx:].copy()

print("Train size", len(train))
print("Val size", len(test))
for label in ["0", "1"]:
    print("------------")
    print("\tTrain has", len(train[train["class"] == label]), label)
    print("\tVal has", len(test[test["class"] == label]), label)



## === cell 2
from tf_keras import layers as KL

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
BATCH_SIZE = 32

try:
    tf.random.set_seed(42)
except Exception as e:
    print("Warning: could not set tf random seed:", repr(e))

augmenter = keras.Sequential(
    [
        KL.RandomRotation(factor=0.25, fill_mode="nearest"),  # 0.25*2pi ~= 90 degrees
        KL.RandomFlip(mode="horizontal"),
    ],
    name="augmenter",
)


@tf.function
def _decode_and_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, (IMAGE_HEIGHT, IMAGE_WIDTH), method=tf.image.ResizeMethod.BICUBIC
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


@tf.function
def _augment(img):
    return augmenter(img, training=True)


def make_train_val_datasets(train_df, val_df, batch_size):
    train_paths = (TRAIN_DIR + "/" + train_df["filename"].astype(str)).to_numpy()
    train_labels = train_df["class"].astype(np.float32).to_numpy()

    val_paths = (TRAIN_DIR + "/" + val_df["filename"].astype(str)).to_numpy()
    val_labels = val_df["class"].astype(np.float32).to_numpy()

    ds_train = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    ds_train = ds_train.shuffle(
        len(train_paths), seed=42, reshuffle_each_iteration=True
    )
    ds_train = ds_train.map(
        lambda p, y: (_augment(_decode_and_resize(p)), y),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds_train = ds_train.batch(batch_size, drop_remainder=False)
    ds_train = ds_train.prefetch(AUTOTUNE)

    ds_val = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    ds_val = ds_val.map(
        lambda p, y: (_decode_and_resize(p), y),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds_val = ds_val.batch(batch_size, drop_remainder=False)
    ds_val = ds_val.prefetch(AUTOTUNE)

    return ds_train, ds_val


train_ds, val_ds = make_train_val_datasets(train, test, BATCH_SIZE)


class _NWrap:
    def __init__(self, n):
        self.n = n


train_generator = _NWrap(len(train))
validation_generator = _NWrap(len(test))



## === cell 3
from tf_keras.applications import vgg16

model = vgg16.VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3),
    pooling="max",
)



## === cell 4
for layer in model.layers[:-5]:
    layer.trainable = False



## === cell 5
from tf_keras.layers import Dense
from tf_keras.models import Sequential

transfer_model_vgg16 = Sequential()
for layer in model.layers:
    transfer_model_vgg16.add(layer)

transfer_model_vgg16.add(Dense(512, activation="relu"))
transfer_model_vgg16.add(Dense(1, activation="sigmoid"))

transfer_model_vgg16.summary()



## === cell 6
print("Skipping model_to_dot visualization (not available in this environment).")



## === cell 7
from tf_keras import optimizers

adam = optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

transfer_model_vgg16.compile(
    optimizer=adam,
    loss="binary_crossentropy",
    metrics=["accuracy"],
)



## === cell 8
vgg16_model_history = transfer_model_vgg16.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    verbose=1,
)



## === cell 9
PLOT_DEBUG = False

if PLOT_DEBUG:
    from IPython.display import Image, display

    def plot_prediction(image_path, label):
        display(Image(filename=image_path, width=IMAGE_WIDTH, height=IMAGE_HEIGHT))
        prediction = "dog"
        confidence = float(label)
        if confidence < 0.5:
            prediction = "cat"
            confidence = 1.0 - confidence
        legend = (
            "The image %s above is a %s with a confidence of %.2f%% (p_dog=%.6f)"
            % (image_path, prediction, confidence * 100, float(label))
        )
        print(legend)

else:

    def plot_prediction(image_path, label):
        pass




## === cell 10
import cv2


def build_batches(
    df, has_labels=True, limit=500, batch_size=BATCH_SIZE, produce="images"
):
    """
    produce: "images" -> yields (X, y) if has_labels else yields X only
             "paths"  -> yields (paths, y) if has_labels else yields paths only
    Note: For has_labels=False, expects df columns: filename (or id/filename) and uses TEST_DIR.
    """
    n_rows = len(df)
    if limit != -1:
        n_rows = min(n_rows, int(limit))

    has_filename_col = "filename" in df.columns

    Xb = np.empty((batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.float32)
    yb = np.empty((batch_size,), dtype=np.float32) if has_labels else None
    paths = [None] * batch_size

    b = 0
    i = 0

    for row in df.itertuples(index=False):
        if i >= n_rows:
            break

        if has_labels:
            yb[b] = float(getattr(row, "class"))
            raw_image_path = os.path.join(TRAIN_DIR, getattr(row, "filename"))
        else:
            if has_filename_col:
                fn = getattr(row, "filename")
                raw_image_path = os.path.join(TEST_DIR, fn)
            else:
                raw_image_path = os.path.join(
                    TEST_DIR, f"{int(getattr(row, 'id'))}.jpg"
                )

        img = cv2.imread(raw_image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {raw_image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(
            img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_CUBIC
        )

        Xb[b] = img.astype(np.float32) / 255.0
        paths[b] = raw_image_path

        b += 1
        i += 1

        if b == batch_size:
            if produce == "images":
                if has_labels:
                    yield Xb, yb
                else:
                    yield Xb
            else:
                if has_labels:
                    yield paths, yb
                else:
                    yield paths
            b = 0

    if b > 0:
        if produce == "images":
            if has_labels:
                yield Xb[:b], yb[:b]
            else:
                yield Xb[:b]
        else:
            if has_labels:
                yield paths[:b], yb[:b]
            else:
                yield paths[:b]




## === cell 11
RUN_EVAL_DEBUG = False

if RUN_EVAL_DEBUG:
    samples = 64
    eval_steps = 1
    eval_result = transfer_model_vgg16.evaluate(
        build_batches(test, limit=samples, batch_size=BATCH_SIZE),
        steps=eval_steps,
        verbose=1,
    )
    print("Eval:", eval_result)



## === cell 12
RUN_PRED_DEBUG = False

if RUN_PRED_DEBUG:
    some_predictions = transfer_model_vgg16.predict(
        build_batches(test, limit=12, batch_size=1),
        steps=12,
        verbose=1,
    )



## === cell 13
if RUN_PRED_DEBUG:
    idx = 0
    for mini_batch_files in build_batches(
        test, limit=12, batch_size=1, produce="paths", has_labels=False
    ):
        mini_batch_file = mini_batch_files[0]
        predicted_label = float(some_predictions[idx][0])
        idx += 1
        plot_prediction(mini_batch_file, predicted_label)



## === cell 14
from os import listdir

test_files = [f for f in listdir(TEST_DIR) if f.lower().endswith(".jpg")]
output = pd.DataFrame({"filename": test_files})
output["id"] = output["filename"].str.replace(".jpg", "", regex=False).astype(int)
output = output.sort_values("id").reset_index(drop=True)

print("Num test images:", len(output))
print(output.head())



## === cell 15
pred_batch_size = 64

test_paths = (TEST_DIR + "/" + output["filename"].astype(str)).to_numpy()
ds_test = tf.data.Dataset.from_tensor_slices(test_paths)
ds_test = ds_test.map(
    _decode_and_resize, num_parallel_calls=AUTOTUNE, deterministic=True
)
ds_test = ds_test.batch(pred_batch_size, drop_remainder=False).prefetch(AUTOTUNE)

results = transfer_model_vgg16.predict(ds_test, verbose=1)
results = results.reshape(-1)[: len(output)]
print("Pred shape:", results.shape)



## === cell 16
output["label"] = results.astype(float)
print(output[["id", "label"]].head(10))

if PLOT_DEBUG:
    stop = 5
    for _, row in output.head(stop).iterrows():
        path = os.path.join(TEST_DIR, f"{int(row['id'])}.jpg")
        plot_prediction(path, row["label"])

submission = output[["id", "label"]].copy()
submission["id"] = submission["id"].astype(int)
submission["label"] = submission["label"].astype(float)

if list(submission.columns) != ["id", "label"]:
    raise ValueError(
        f"Submission columns must be ['id','label'], got {submission.columns.tolist()}"
    )
if submission.isna().any().any():
    raise ValueError("Submission contains NaNs.")
if len(submission) != len(output):
    raise ValueError("Submission row count mismatch.")

submission.to_csv("submission_file.csv", index=False)
print("Wrote submission_file.csv with shape:", submission.shape)
print(submission.head())
