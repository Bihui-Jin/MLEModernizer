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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

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
scikit-image==0.25.2
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
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.8993

# 6. Current score

0.99957

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.99349) has done: 'I fix the TensorFlow import crash by switching to the already-installed `tf_keras` backend (so protobuf incompatibilities don’t break runtime), while keeping your model/training logic intact. Then I fix the dataset pipeline errors by using valid `tf.image.resize` method constants (instead of the unsupported `"BILINEAR"` string) and by replacing `tf.image.rotate` with `tensorflow_addons`-free augmentation (via KerasCV layers) so the map() graph can build. Finally, I ensure the test pipeline outputs image tensors (not strings) and that `submission.csv` is written with the correct columns and row alignment.'
- What this solution (achieved 0.99349) has done: 'I fix the immediate runtime crash caused by an incompatibility between TensorFlow’s bundled protobuf code and the installed `protobuf==6.33.0` by force-setting the pure-Python protobuf implementation before importing TensorFlow/tf_keras. This is a correctness/stability-only change (no model/training logic changes), so it should keep your score effectively the same while making the notebook run end-to-end reliably. I also make the checkpoint path writable in Kaggle (`/kaggle/working/...`) to avoid occasional `/tmp` filesystem quirks and ensure a submission CSV is always produced. No changes are made to the model architecture, data split, augmentations, or training loop, so performance should not be intentionally altered (your current score is already above target).'
- What this solution (achieved 0.9992) has done: 'The crash happens before any training because TensorFlow 2.18 is incompatible with `protobuf==6.x` in this environment, and setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` no longer prevents the `MessageFactory.GetPrototype` failure. The minimal, score-neutral fix is to avoid importing TensorFlow entirely and instead run the same high-level pipeline using Keras 3 (which is installed) on the JAX backend; this preserves the core CNN architecture, loss, training loop, and output semantics while restoring end-to-end execution. I also keep all paths and the submission format identical, and ensure the model checkpointing works with Keras 3. No intentional performance improvements are added (your current score is already above target), this is primarily a stability/compatibility repair so a valid `submission.csv` is reliably produced.'
- What this solution (achieved 0.99893) has done: 'I fix the runtime crash happening at `import keras` by forcing the pure-Python protobuf implementation early (before any TensorFlow/Keras-related import), which avoids the `MessageFactory.GetPrototype` incompatibility in this Kaggle environment. I also make the Keras backend selection robust by preferring `KERAS_BACKEND=jax` but safely falling back to `tensorflow` if JAX backend isn’t available, without changing the model/training logic. Finally, I keep the existing data pipeline and submission writing intact, only adding small stability checks so the notebook always finishes and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.99828) has done: 'The crash happens before training because `import keras` triggers TensorFlow’s protobuf bindings, and in this environment that combination raises `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'`. The minimal, score-neutral fix is to avoid TF-backed Keras entirely and run the exact same model/training loop using Keras 3 on the JAX backend (falling back to NumPy backend if JAX isn’t available), which keeps your architecture/loss/training semantics intact while restoring end-to-end execution. I also keep paths and the submission format identical and add a small backend-selection guard so the notebook reliably runs on Kaggle. Since your current score (0.99893) is already above the target (0.8993), I do not make any changes intended to further improve performance.'
- What this solution (achieved 0.99938) has done: 'The crash happens at `import keras` due to a TensorFlow/protobuf incompatibility in this environment (the `MessageFactory.GetPrototype` error). The minimal fix is to force Keras 3 to use the JAX backend (which avoids importing TensorFlow) and to set the pure-Python protobuf implementation *before any* keras/tensorflow-related import as a safety guard. I also make the backend selection explicit (set, not setdefault) and add a small fallback to NumPy backend if JAX is unavailable, without changing your model, data processing, training loop, or submission formatting (so score should remain in the same ballpark, i.e., still above target). Finally, I keep all paths the same and ensure `submission.csv` is always written.'
- What this solution (achieved 0.99941) has done: 'I fix the crash in the first import cell by ensuring we never trigger the TensorFlow/protobuf path: instead of `import keras` (which can dispatch into TF in this environment), we explicitly use the already-installed `keras_core` with the JAX backend, which preserves the Keras 3-style API and keeps your model/training logic intact. I keep all data paths, preprocessing, model architecture, training loop, and submission formatting the same, only adjusting the imports and small compatibility points so it runs end-to-end. Since your current score (0.99938) is already well above the target (0.8993), I not add any score-improving changes; this is strictly a stability/runtime fix. The script still write `submission.csv` with `id,has_cactus` in the correct order.'
- What this solution (achieved 0.99957) has done: 'The crash is happening before training because the protobuf C++ runtime used by TensorFlow/tf_keras is being imported indirectly (even when you intend to use JAX), and it’s incompatible with `protobuf==6.33.0` in this environment. The minimal fix is to force Keras to use the JAX backend *and* set the protobuf environment variables before any Keras/TensorFlow-related import, so the TensorFlow/protobuf path is never triggered. I also make one small robustness tweak to avoid building the p2/p98 cache for all 14k images when it already exists (still same logic, just safer), and keep submission writing unchanged. Since your current score is already above the target, no score-improving changes are introduced—this is strictly a runtime/stability fix to ensure an end-to-end run and a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd



## === cell 1
os.environ["KERAS_BACKEND"] = "jax"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

SEED = 1337
random.seed(SEED)
np.random.seed(SEED)

import keras_core as keras
from keras_core import layers, models

try:
    keras.utils.set_random_seed(SEED)
except Exception:
    pass

from zipfile import ZipFile

print("keras_core version:", getattr(keras, "__version__", "unknown"))
try:
    print("keras_core backend:", keras.backend.backend())
except Exception:
    pass



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
path = "/kaggle/input/aerial-cactus-identification/"
files_dataframe = pd.read_csv(path + "train.csv", dtype={"id": str, "has_cactus": int})
files_dataframe.head()



## === cell 3
os.makedirs("./train", exist_ok=True)
os.makedirs("./test", exist_ok=True)


def _maybe_extract(zip_path, out_dir):
    marker = os.path.join(out_dir, ".extracted")
    if os.path.exists(marker):
        return
    with ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    with open(marker, "w") as f:
        f.write("ok")


_maybe_extract(path + "train.zip", "./train")
_maybe_extract(path + "test.zip", "./test")

training_files = "train/" + files_dataframe["id"]
print("Training sample:")
print(training_files.head(2))

print("Sanity check exists:", os.path.exists("./" + training_files.iloc[0]))



## === cell 4
class_reparts = files_dataframe["has_cactus"].value_counts()
ax = class_reparts.plot.bar()



## === cell 5
total_samples = files_dataframe["has_cactus"].size
print("Total number of samples: ", total_samples)
has_cactus_weight = total_samples / (2 * class_reparts[1])
no_cactus_weight = total_samples / (2 * class_reparts[0])
class_weights = {0: no_cactus_weight, 1: has_cactus_weight}
print("Class weights: ", class_weights)



## === cell 6
import skimage.exposure as exposure


def preprocess(img):
    p2, p98 = np.percentile(img, (3, 97))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    return img_rescale




## === cell 7
import imageio.v2 as imageio
from skimage.transform import resize as sk_resize

BATCH_SIZE = 32
IMG_SIZE = (32, 32)

files_df = files_dataframe.copy()
files_df["id"] = files_df["id"].astype(str)

perm = np.random.RandomState(SEED).permutation(len(files_df))
files_df = files_df.iloc[perm].reset_index(drop=True)

val_frac = 0.25
val_n = int(round(len(files_df) * val_frac))
val_df = files_df.iloc[:val_n].reset_index(drop=True)
train_df = files_df.iloc[val_n:].reset_index(drop=True)

print("Train size:", len(train_df), "Val size:", len(val_df))

_p_cache_path = "/kaggle/working/p2p98_cache.npz"


def _build_p2p98_cache(df, root_dir="./train/"):
    ids = df["id"].to_numpy()
    p2 = np.empty(len(ids), dtype=np.float32)
    p98 = np.empty(len(ids), dtype=np.float32)
    for i, fn in enumerate(ids):
        fpath = os.path.join(root_dir, fn)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing image: {fpath}")
        img = imageio.imread(fpath).astype(np.float32)
        p2[i], p98[i] = np.percentile(img, (3, 97))
    return ids.astype(str), p2, p98


cache_ok = False
if os.path.exists(_p_cache_path):
    try:
        cache = np.load(_p_cache_path, allow_pickle=False)
        cache_ids = cache["id"].astype(str)
        cache_p2 = cache["p2"].astype(np.float32)
        cache_p98 = cache["p98"].astype(np.float32)
        cache_ok = (
            (len(cache_ids) == len(files_df))
            and (len(cache_p2) == len(files_df))
            and (len(cache_p98) == len(files_df))
        )
    except Exception:
        cache_ok = False

if not cache_ok:
    cache_ids, cache_p2, cache_p98 = _build_p2p98_cache(files_df, root_dir="./train/")
    np.savez_compressed(_p_cache_path, id=cache_ids, p2=cache_p2, p98=cache_p98)

p2p98_map = {k: (float(a), float(b)) for k, a, b in zip(cache_ids, cache_p2, cache_p98)}


def _rescale_intensity_np(img, p2, p98):
    p2 = max(p2, 0.0)
    p98 = min(p98, 255.0)
    denom = max(p98 - p2, 1e-6)
    img = (img - p2) * (255.0 / denom)
    img = np.clip(img, 0.0, 255.0)
    return img


def _samplewise_center_std_np(img):
    mean = float(np.mean(img))
    std = float(np.std(img))
    img = img - mean
    img = img / max(std, 1e-6)
    return img


def _read_and_process_train_image(img_id, do_augment=False):
    fpath = os.path.join("./train", img_id)
    img = imageio.imread(fpath).astype(np.float32)  # 0..255
    p2, p98 = p2p98_map[img_id]
    img = _rescale_intensity_np(img, p2, p98)

    img = sk_resize(img, IMG_SIZE, preserve_range=True, anti_aliasing=True).astype(
        np.float32
    )
    img = _samplewise_center_std_np(img)
    return img


def _read_and_process_test_image(img_id):
    fpath = os.path.join("./test", img_id)
    img = imageio.imread(fpath).astype(np.float32)  # 0..255

    flat = img.reshape(-1)
    flat_sorted = np.sort(flat)
    n = flat_sorted.shape[0]
    i2 = int(round(0.03 * (n - 1)))
    i98 = int(round(0.97 * (n - 1)))
    p2 = float(flat_sorted[i2])
    p98 = float(flat_sorted[i98])

    img = sk_resize(img, IMG_SIZE, preserve_range=True, anti_aliasing=True).astype(
        np.float32
    )
    img = _rescale_intensity_np(img, p2, p98)
    img = _samplewise_center_std_np(img)
    return img


class DataSequence(keras.utils.Sequence):
    def __init__(self, df, training=True, batch_size=32, seed=1337):
        self.df = df.reset_index(drop=True)
        self.training = training
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.training:
            self.rng.shuffle(self.indexes)

    def __getitem__(self, idx):
        batch_idx = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch = self.df.iloc[batch_idx]
        x = np.empty((len(batch), 32, 32, 3), dtype=np.float32)
        y = np.empty((len(batch), 2), dtype=np.float32)

        for i, (img_id, label) in enumerate(
            zip(batch["id"].values, batch["has_cactus"].values)
        ):
            x[i] = _read_and_process_train_image(img_id, do_augment=False)
            lab = int(label)
            y[i, 0] = 1.0 - lab
            y[i, 1] = float(lab)

        return x, y


training_seq = DataSequence(train_df, training=True, batch_size=BATCH_SIZE, seed=SEED)
validation_seq = DataSequence(val_df, training=False, batch_size=BATCH_SIZE, seed=SEED)

training_n = len(train_df)
validation_n = len(val_df)



## === cell 8
print("training samples:", training_n)
print("validation samples:", validation_n)



## === cell 9
pass



## === cell 10
model = models.Sequential()

model.add(
    layers.Conv2D(
        32, (5, 5), padding="valid", activation="relu", input_shape=(32, 32, 3)
    )
)
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(64, (3, 3), padding="valid", activation="relu"))
model.add(layers.Conv2D(64, (3, 3), padding="valid", activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(128, (3, 3), padding="valid", activation="relu"))
model.add(layers.Conv2D(128, (3, 3), padding="valid", activation="relu"))
model.add(layers.Dropout(0.2))

model.add(layers.Flatten())
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(2, activation="softmax"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()



## === cell 11
from keras_core.callbacks import ReduceLROnPlateau, ModelCheckpoint

reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=0.001)

checkpoint_path = "/kaggle/working/checkpoint.keras"
save_best_model = ModelCheckpoint(
    checkpoint_path,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
)



## === cell 12
history = model.fit(
    training_seq,
    validation_data=validation_seq,
    verbose=1,
    epochs=10,
    callbacks=[reduce_lr, save_best_model],
    class_weight=class_weights,
)



## === cell 13
sample_sub = pd.read_csv(path + "sample_submission.csv", dtype={"id": str})
test_df = sample_sub.copy()

test_x = np.empty((len(test_df), 32, 32, 3), dtype=np.float32)
for i, img_id in enumerate(test_df["id"].values):
    test_x[i] = _read_and_process_test_image(img_id)

print("test samples:", len(test_df))



## === cell 14
from keras_core.models import load_model

if os.path.exists(checkpoint_path):
    model = load_model(checkpoint_path)



## === cell 15
proba = model.predict(test_x, batch_size=BATCH_SIZE, verbose=0)
if proba.ndim == 2 and proba.shape[1] == 2:
    has_cactus_proba = proba[:, 1]
else:
    has_cactus_proba = proba.reshape(-1)

output = sample_sub.copy()
output["has_cactus"] = has_cactus_proba.astype(float)
output["has_cactus"] = output["has_cactus"].clip(0.0, 1.0)

output.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", output.shape)
print(output.head())



## === cell 16
import shutil

for d in ["test", "train"]:
    try:
        shutil.rmtree(d)
    except OSError:
        print(f"{d} files already erased or not present")
