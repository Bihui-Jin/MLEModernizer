# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

No external packages required in the script and installed.

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

0.7623100731269614

# 6. Current score

0.62604

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 4.77119) has done: 'I fix the TensorFlow/TF-Hub runtime incompatibility that’s causing the `MessageFactory`/`GetPrototype` and `KerasTensor is symbolic` errors by removing TF-Hub usage and switching the backbone to `tf.keras.applications.MobileNetV2` (same MobileNetV2 family and same “frozen backbone + Dense softmax head” core structure). I also ensure the train/valid and full-train pipelines actually create the model objects so later cells (`fit/predict/submission`) don’t fail with `NameError`. Finally, I make the submission generation robust by strictly matching `sample_submission.csv` ids/columns and writing `/kaggle/working/submission.csv`. This should run end-to-end in the Kaggle Python 3.11 environment and produce a valid multi-class probability submission for log loss.'
- What this solution (achieved 4.84058) has done: 'The immediate failure is the `MessageFactory.GetPrototype` protobuf/TensorFlow import crash, which happens before any training; I fix it by forcing TensorFlow to use the pure-Python protobuf implementation via environment variables set *before* importing TensorFlow. Then, to move log-loss down toward the target (your 4.77 is far worse than 0.76), I make a minimal, metric-aligned correction: the labels are currently integer-encoded with `LabelEncoder` order, but submission columns must match the competition’s canonical class order from `sample_submission.csv`, so I remap labels to that order and also ensure prediction columns follow that same order. Finally, I keep the existing MobileNetV2 frozen-backbone + softmax head training logic intact and ensure the pipeline always writes `/kaggle/working/submission.csv`.'
- What this solution (achieved 4.84096) has done: 'You’re hitting a TensorFlow/protobuf compatibility crash (`MessageFactory` has no `GetPrototype`) before training starts, so the pipeline never reaches model fitting/prediction reliably. I fix that by forcing a protobuf version compatible with TF in this environment via a safe runtime install and kernel restart guard, while keeping your MobileNetV2 frozen-backbone + softmax head logic unchanged. I also make the import sequence robust (env vars set before TF import) and ensure the submission is always aligned to `sample_submission.csv` ids/columns and written to `/kaggle/working/submission.csv`. These changes are execution/stability fixes and should also reduce log-loss versus the current “effectively broken/unstable” run because the model actually train and produce valid probabilities.'
- What this solution (achieved 4.87663) has done: 'I remove the forced protobuf downgrade-and-exit logic that currently stops execution with `SystemExit: 0`, and instead rely on the already-set environment variables that make TensorFlow use the pure-Python protobuf runtime. I also remove early stopping callbacks (they violate your “no early stopping” constraint) while keeping the exact same model (frozen MobileNetV2 backbone + Dense softmax head), optimizer, loss, and data pipeline. To move log-loss down toward your target without changing the core approach, I train on the full training set (not only the first 1000) for the validation split/model sanity-check so the learned head isn’t severely underfit. Finally, I keep the submission generation aligned to `sample_submission.csv` columns/ids and write `/kaggle/working/submission.csv`.'
- What this solution (achieved 4.8767) has done: 'Your pipeline likely didn’t “yield” a score because it can fail early on import/training in this environment (protobuf pin + restart) and/or because the generated submission probabilities can contain exact zeros (or not sum to 1 after any numerical drift), which leads to very poor multi-class log loss when Kaggle clips less aggressively than expected. I keep your exact model and training approach (frozen ImageNet MobileNetV2 + Dense softmax, same datasets, same epochs), but make two minimal, score-relevant fixes: remove the pip/re-exec protobuf pin (rely on the already-set env vars) for stability, and add a tiny epsilon clipping + renormalization step before writing the submission to avoid log-loss blowups. I also make the test file ordering and id alignment strictly follow `sample_submission.csv` to ensure rows match perfectly (no accidental NaNs/filled uniform rows). The result still trains and predicts the same way, but should reliably produce a valid `submission.csv` and typically reduce log loss versus the unstable/zero-probability case.'
- What this solution (achieved 4.88906) has done: 'Your current notebook likely doesn’t “yield” a score because it exits early due to the protobuf pip-install + `os.execv` restart logic; removing that restart makes the run deterministic and guarantees a `submission.csv` is written. To move log-loss down (lower is better) without changing your model/training approach, I keep the exact same frozen MobileNetV2 + Dense softmax setup and training loop, but fix a data bug: the full-training dataset is currently shuffled, which can desync images/labels in `tf.data` and severely harms learning—so I disable shuffling only for the full-train pass. Finally, I keep your strict `sample_submission.csv` column/id alignment and probability clipping/renormalization to avoid log-loss blowups from zeros.'
- What this solution (achieved 0.62649) has done: 'I fix the TensorFlow import crash (`MessageFactory`/protobuf incompatibility) by forcing TensorFlow to use the pure-Python protobuf implementation *before* any TensorFlow-related import, and by keeping the import order clean so it reliably works on Kaggle Python 3.11. Then I make one minimal, score-relevant correction: the training pipeline currently feeds raw `[0,1]` images into MobileNetV2, but MobileNetV2’s ImageNet weights expect `preprocess_input` scaling to `[-1,1]`; applying the correct preprocessing (once, not twice) typically reduces log loss substantially without changing the model/training approach. Finally, I keep the exact same frozen MobileNetV2 + Dense softmax architecture and training loops, and ensure the submission stays strictly aligned to `sample_submission.csv` ids/columns and is written to `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.62604) has done: 'The current failure happens at TensorFlow import due to a protobuf runtime mismatch (`MessageFactory` missing `GetPrototype`). I fix this by forcing TensorFlow to use the pure-Python protobuf implementation *and* ensuring the corresponding `protobuf` Python module version is compatible, via a small pre-import install guard (only if needed) and a clean import sequence. I also fix a minor filesystem bug in `save_model()` (it currently creates the parent of the intended directory, not the directory itself), which can break saving later. These are stability/correctness fixes and should keep the same model/training logic while letting the notebook run end-to-end and write `/kaggle/working/submission.csv`.'

# 9. Code solution

## === cell 0
import os
import sys

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

print("Python:", sys.version)



## === cell 1
import subprocess
import importlib


def _ensure_compatible_protobuf():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        major = int(str(pb_ver).split(".")[0])
        print("protobuf version:", pb_ver)
        if major >= 5:
            print("Installing compatible protobuf (<5) to avoid TF import crash...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
            )
            importlib.invalidate_caches()
            import google.protobuf  # noqa: F401
            from google.protobuf import __version__ as pb_ver2

            print("protobuf version after install:", pb_ver2)
    except Exception as e:
        print("Warning: protobuf compatibility guard encountered:", repr(e))


_ensure_compatible_protobuf()



## === cell 2
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import pandas as pd
import numpy as np
import PIL
import pathlib
import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow:", tf.__version__)
print("Keras (tf.keras):", keras.__version__)

tf.random.set_seed(42)
np.random.seed(42)



## === cell 3
label_df = pd.read_csv("/kaggle/input/dog-breed-identification/labels.csv")
label_df.head()



## === cell 4
label_df.shape



## === cell 5
label_df.breed.value_counts().plot.bar(figsize=(18, 5))



## === cell 6
np.max(label_df.breed.value_counts()), np.min(label_df.breed.value_counts())



## === cell 7
label_df.breed.value_counts().plot.hist()



## === cell 8
label_df.breed.value_counts().median()



## === cell 9
im = PIL.Image.open(
    "/kaggle/input/dog-breed-identification/train/000bec180eb18c7604dcecc8fe0dba07.jpg"
)
im



## === cell 10
im = PIL.Image.open(
    f"/kaggle/input/dog-breed-identification/train/{label_df.id[6]}.jpg"
)
im



## === cell 11
img_path = []
for name in label_df["id"]:
    img_path.append(f"/kaggle/input/dog-breed-identification/train/{name}.jpg")
img_path[:2]



## === cell 12
PIL.Image.open(img_path[9])  # ok got it !!!



## === cell 13
len(os.listdir("/kaggle/input/dog-breed-identification/train")) == len(img_path)



## === cell 14
label_df.shape, len(os.listdir("/kaggle/input/dog-breed-identification/train"))



## === cell 15
labels = label_df.breed.to_numpy()
labels, len(labels)



## === cell 16
len(label_df["breed"].unique())



## === cell 17
label_df.id[60], label_df.breed[60]




## === cell 18
def showImage(path):
    plt.imshow(PIL.Image.open(path))


showImage(img_path[60])



## === cell 19
np.asarray(PIL.Image.open(img_path[60])).shape



## === cell 20
samp_i = PIL.Image.open(img_path[60])
print(samp_i.format)
print(samp_i.size)  # w (as columns) x h (as rows)



## === cell 21
X = img_path

sample_sub = pd.read_csv("/kaggle/input/dog-breed-identification/sample_submission.csv")
breed_cols = [c for c in sample_sub.columns if c != "id"]  # canonical competition order

breed_to_idx = {b: i for i, b in enumerate(breed_cols)}
label_set = set(labels.tolist())
if set(breed_to_idx.keys()) != label_set:
    missing = label_set - set(breed_to_idx.keys())
    extra = set(breed_to_idx.keys()) - label_set
    raise ValueError(
        f"Breed set mismatch between labels and sample_submission. missing={missing}, extra={extra}"
    )

y = np.asarray([breed_to_idx[b] for b in labels], dtype=np.int32)

len(X), len(y)



## === cell 22
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

len(X_train), len(X_valid)



## === cell 23
samp = PIL.Image.open(img_path[42])
np.asanyarray(samp).shape  # row, col, channel (R,G,B)



## === cell 24
image = imread(img_path[42])  # output np array (257, 350, 3)
plt.imshow(image)



## === cell 25
image.max(), image.min()  # 8 bit 0 - 255



## === cell 26
batch_size = 32
img_height = 224
img_width = 224



## === cell 27
ts = tf.io.read_file(img_path[42])
img = tf.io.decode_jpeg(ts, channels=3)
tf.image.resize(img, [img_height, img_width]).shape




## === cell 28
def preprocess_img(sel_path):
    image_raw = tf.io.read_file(sel_path)
    img_dec = tf.io.decode_jpeg(image_raw, channels=3)
    img_dec = tf.image.resize(img_dec, [img_height, img_width])
    img_dec = tf.cast(img_dec, tf.float32)
    img_pp = tf.keras.applications.mobilenet_v2.preprocess_input(img_dec)  # [-1, 1]
    return img_pp


t_img = preprocess_img(img_path[42])
t_img[:2]




## === cell 29
def process_path(file_path, labels):
    label = labels
    img = preprocess_img(file_path)
    return img, label


process_path(X[0], y[0])[0].shape, process_path(X[0], y[0])[1]




## === cell 30
def create_data_batch(X, y=None, valid_data=False, test_data=False, shuffle=True):
    if test_data:
        print("creating batchs for testing set...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X)))
        data_batch = (
            data.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        return data_batch
    elif valid_data:
        print("creating batchs for validation set...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
        data_batch = (
            data.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        return data_batch
    else:
        print("creating batchs for training set...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
        if shuffle:
            data = data.shuffle(
                buffer_size=1000, seed=42, reshuffle_each_iteration=True
            )
        data = data.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
        data_batch = data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return data_batch


train_data = create_data_batch(X_train, y_train, shuffle=True)
valid_data = create_data_batch(X_valid, y_valid, valid_data=True)

train_data.element_spec, valid_data.element_spec




## === cell 31
def batch_img_show(data_set):
    image_batch, label_batch = next(iter(data_set))

    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow((image_batch[i] + 1.0) / 2.0)
        label = int(label_batch[i])

        plt.title(breed_cols[label])
        plt.axis("off")


batch_img_show(train_data)  # the img will be shuffled when you run it again



## === cell 32
INPUT_SHAPE = [None, img_height, img_width, 3]
OUTPUT_SHAPE = len(breed_cols)

MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/5"




## === cell 33
def create_model(model_url=None):
    print("Building the model ...")
    print("Using tf.keras.applications.MobileNetV2 backbone (frozen)")

    inputs = keras.Input(shape=(img_height, img_width, 3), name="image")

    x = inputs

    backbone = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(img_height, img_width, 3),
        pooling="avg",
    )
    backbone.trainable = False
    x = backbone(x)

    outputs = layers.Dense(
        units=OUTPUT_SHAPE, activation="softmax", name="breed_softmax"
    )(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="dog_breed_model")

    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model


model = create_model(MODEL_URL)
model.summary()




## === cell 34
def create_tensorboard_callback():
    log_dir = "/kaggle/working/log/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )
    return tensorboard_callback


tensorboard = create_tensorboard_callback()

"Yes" if tf.config.list_physical_devices("GPU") else "No GPU available"



## === cell 35
_ = model.fit(
    x=train_data,
    validation_data=valid_data,
    epochs=5,  # keep runtime reasonable; core training loop unchanged (fit over epochs).
    validation_freq=1,
    callbacks=[tensorboard],
    verbose=2,
)



## === cell 36
pred = model.predict(valid_data, verbose=1)
pred.shape



## === cell 37
model.evaluate(valid_data, verbose=2)




## === cell 38
def save_model(model, suffix=None):
    model_dir = os.path.join(
        "/kaggle/working/models", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    os.makedirs(model_dir, exist_ok=True)
    model_path = model_dir + "-" + (suffix if suffix else "model") + ".h5"
    print(f"Saving model to: {model_path}")
    model.save(model_path)
    return model_path


def load_model(model_path):
    print(f'Loading model from: "{model_path}"')
    model = keras.models.load_model(model_path)
    return model




## === cell 39
print(f"Full datasets X,y = {len(X)}, {len(y)}")

full_data_batch = create_data_batch(X, y, shuffle=False)

full_model = create_model(MODEL_URL)



## === cell 40
_ = full_model.fit(
    x=full_data_batch,
    epochs=5,  # keep runtime reasonable; no early stopping.
    verbose=2,
)



## === cell 41
test_path = "/kaggle/input/dog-breed-identification/test"

sample_sub = pd.read_csv("/kaggle/input/dog-breed-identification/sample_submission.csv")
test_ids = sample_sub["id"].astype(str).tolist()
test_img_path = [os.path.join(test_path, f"{i}.jpg") for i in test_ids]

missing = [p for p in test_img_path if not os.path.exists(p)]
if missing:
    raise FileNotFoundError(
        f"Missing {len(missing)} test images. Example: {missing[0]}"
    )

print("Num test images:", len(test_ids))
sample_id = min(499, len(test_ids) - 1)
plt.imshow(imread(os.path.join(test_path, f"{test_ids[sample_id]}.jpg")))
plt.axis("off")



## === cell 42
test_data = create_data_batch(test_img_path, test_data=True)
test_data



## === cell 43
y_pred = full_model.predict(test_data, verbose=1)
y_pred.shape



## === cell 44
eps = 1e-7
y_pred = np.asarray(y_pred, dtype=np.float64)
y_pred = np.clip(y_pred, eps, 1.0 - eps)
y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)

sample_sub = pd.read_csv("/kaggle/input/dog-breed-identification/sample_submission.csv")
breed_cols = [c for c in sample_sub.columns if c != "id"]

preds_submit = pd.DataFrame(y_pred, columns=breed_cols)
preds_submit.insert(0, "id", test_ids)

preds_submit = preds_submit.set_index("id").reindex(sample_sub.set_index("id").index)
preds_submit = preds_submit[breed_cols].reset_index()

submission = preds_submit
submission.head(), submission.shape



## === cell 45
out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(
    "Columns:", submission.columns[:5].tolist(), "... total:", len(submission.columns)
)
print("Shape:", submission.shape)
probs = submission.drop(columns=["id"]).to_numpy(dtype=np.float64)
print("Min/Max prob (sanity):", float(probs.min()), float(probs.max()))
print(
    "Row sums (sanity):",
    float(np.min(probs.sum(axis=1))),
    float(np.max(probs.sum(axis=1))),
)
