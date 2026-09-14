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

3.11

# 3. Installed packages



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

0.84975

# 6. Current score

0.5572

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.58769) has done: 'I fix the TensorFlow/protobuf crash by removing the environment override that forces the pure-Python protobuf implementation, which is what triggers the `MessageFactory.GetPrototype` error in this Kaggle image. Then I fix the data unzipping/organization so the test images are actually located under the directory that `flow_from_directory(test_dir)` scans, preventing the “PyDataset has length 0” failure. Finally, I keep your model/training loop intact but make the submission-building robust by using the sample submission to guarantee correct ids/order and by clipping probabilities slightly to avoid logloss infinities. These changes are execution/stability fixes and should also improve score versus producing no valid submission.'
- What this solution (achieved 0.54644) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing TensorFlow to use the C++ protobuf implementation (the pure-Python one triggers this error in this Kaggle image), and I do it before importing TensorFlow. Then I make the unzip + file-moving idempotent so reruns don’t break when files are already moved, while keeping your exact data organization and training loop intact. Finally, I keep the same submission-building logic but make it robust to generator filename formats and ensure the output is a valid `submission.csv` with `id,label` aligned to `sample_submission.csv` order. These changes are execution/stability fixes and should also improve logloss versus failing/crashing runs.'
- What this solution (achieved 0.55941) has done: 'I fix the TensorFlow import crash by removing the incompatible protobuf environment override and ensuring the runtime uses the default protobuf implementation that works in this Kaggle image. Because TensorFlow currently fails to import, all downstream `NameError`s (missing Keras symbols, generators, model, history) happen; those resolve once TF imports successfully. I also make the unzip/move steps idempotent and ensure the test directory structure matches what `flow_from_directory` expects so the test generator is never empty. Finally, I keep your exact model and training loop intact, and make submission creation robust by aligning to `sample_submission.csv` order and clipping probabilities to avoid log-loss infinities, producing a valid `submission.csv`.'
- What this solution (achieved 0.57316) has done: 'We need to fix the TensorFlow import crash caused by forcing protobuf to use the C++ implementation, which fails in this environment (`cannot import name '_message'`). I remove the protobuf environment overrides so TensorFlow can import, which also resolve the downstream `NameError`s for Keras objects and generators. I keep your data unzip/move logic and your exact model/training loop intact, but make the unzip check robust and ensure the test directory has the expected `test/unknown/*.jpg` structure for `flow_from_directory`. Finally, I keep the same submission-building logic while ensuring predictions align to `sample_submission.csv` order and clipping probabilities to avoid logloss infinities, producing a valid `submission.csv`.'
- What this solution (achieved 0.57103) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing the pure‑Python protobuf implementation before importing TensorFlow, which is the stable setting for this Kaggle image and avoids the current runtime error. I keep your exact model, augmentation, and training loop unchanged, but I make the test directory discovery robust to the nested `test/test/unknown` structure so `flow_from_directory()` always finds images and predictions align to ids. I also make the submission id parsing tolerate different filename formats and guarantee we write a valid `submission.csv` with `id,label` in the exact `sample_submission.csv` order. These changes are execution/stability fixes and should move logloss back toward the target by producing correct, aligned probabilities rather than crashing.'
- What this solution (achieved 0.55935) has done: 'I fix the TensorFlow import crash by removing the protobuf environment override that triggers the `MessageFactory.GetPrototype` error in this Kaggle runtime, while keeping your model/training loop unchanged. I also make the dataset unzip step robust to both possible folder layouts (`/kaggle/input/...` and `/kaggle/data/...`) without changing the subsequent file-moving logic. Finally, I keep your submission-building semantics but add one stability tweak: explicitly map predictions to the “dog probability” expected by the competition using the generator’s `class_indices` (score-improving if the current run had label inversion), and keep clipping to avoid logloss infinities.'
- What this solution (achieved 0.57835) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing TensorFlow to use the pure‑Python protobuf implementation *before* importing TensorFlow, which is the most reliable workaround on Kaggle when protobuf/TensorFlow wheels are mismatched. Then I make the environment-seeding order consistent (set `PYTHONHASHSEED` before Python starts hashing isn’t possible here, but we can keep the rest deterministic) without changing your model or training loop. Finally, I keep your exact data prep/training/inference logic, but make submission writing more robust by ensuring prediction length matches the test generator length and by mapping to dog-probability using `class_indices` reliably (score-impacting only if labels were inverted).'
- What this solution (achieved 0.64614) has done: 'I fix the TensorFlow/protobuf crash in the first cell by removing the forced pure-Python protobuf environment variables that trigger `MessageFactory.GetPrototype` in this Kaggle runtime, while keeping the rest of the pipeline unchanged. Then I keep your existing unzip/move logic and generators intact, but add a small safeguard to ensure the train/valid class-index mapping is consistent and the submission is always “probability of dog” (this is score-relevant but does not change the model/training core logic). Finally, I keep the submission-building logic but make the “dog-probability” mapping robust for both possible directory names (dog/dogs, cat/cats) and keep the probability clipping for logloss stability, producing a valid `submission.csv`. These are minimal changes aimed at unblocking execution and nudging logloss down toward the target band by preventing label inversion.'
- What this solution (achieved 0.5891) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by switching to the `tf.keras`-bundled Keras preprocessing API (instead of relying on the standalone `keras_preprocessing`/legacy path that can trigger protobuf-related breakage in this Kaggle image), without changing your model/training loop semantics. I also make the protobuf environment handling explicit and safe before importing TensorFlow, but avoid any other refactors. Finally, I keep your submission logic intact while ensuring the test generator always yields the same ordering as `sample_submission.csv` and still clips probabilities to avoid logloss infinities.'
- What this solution (achieved 0.5572) has done: 'We fix the immediate blocker: TensorFlow fails to import due to a protobuf/TensorFlow incompatibility in this Kaggle runtime (“MessageFactory has no attribute GetPrototype”). The minimal, most reliable workaround here is to force TensorFlow to use the pure-Python protobuf implementation *before* importing TensorFlow, and (optionally) disable the C++ protobuf fast-path, which prevents that crash without changing your model/training logic. Everything else (data unzip/organization, generators, model architecture, training loop, and submission creation) be kept the same, aside from tiny safety checks to ensure a `submission.csv` is always written correctly. This should restore end-to-end execution and get your logloss back near the previously achieved band (and closer to the target) by producing a valid, correctly-aligned submission.'

# 9. Code solution

## === cell 0
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_DISABLE_C_DESCRIPTOR", "1")

import random
import numpy as np
import pandas as pd

os.environ["PYTHONHASHSEED"] = "42"
random.seed(42)
np.random.seed(42)

import tensorflow as tf

tf.random.set_seed(42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

print("TensorFlow version:", tf.__version__)
print("GPU devices:", tf.config.list_logical_devices("GPU"))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import zipfile
from pathlib import Path

CANDIDATES = [
    Path("/kaggle/input/dogs-vs-cats-redux-kernels-edition"),
    Path("/kaggle/input"),
    Path("/kaggle/data/dogs-vs-cats-redux-kernels-edition"),
    Path("/kaggle/data"),
]

BASE_INPUT = None
for c in CANDIDATES:
    if (c / "train.zip").exists() and (c / "test.zip").exists():
        BASE_INPUT = c
        break

if BASE_INPUT is None:
    raise FileNotFoundError(
        "Could not find train.zip/test.zip under expected Kaggle paths. "
        "Tried: " + ", ".join(str(c) for c in CANDIDATES)
    )

train_zip = BASE_INPUT / "train.zip"
test_zip = BASE_INPUT / "test.zip"


def safe_unzip(zip_path: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    if any(dest.rglob("*.jpg")):
        return
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)


RAW_DIR = Path("raw")
RAW_TRAIN_DIR = RAW_DIR / "train"
RAW_TEST_DIR = RAW_DIR / "test"

safe_unzip(train_zip, RAW_TRAIN_DIR)
safe_unzip(test_zip, RAW_TEST_DIR)

print("Base input:", BASE_INPUT)
print("Raw train jpgs:", len(list(RAW_TRAIN_DIR.rglob("*.jpg"))))
print("Raw test jpgs:", len(list(RAW_TEST_DIR.rglob("*.jpg"))))



## === cell 2
from pathlib import Path

Path("train/cats").mkdir(parents=True, exist_ok=True)
Path("train/dogs").mkdir(parents=True, exist_ok=True)
Path("valid/cats").mkdir(parents=True, exist_ok=True)
Path("valid/dogs").mkdir(parents=True, exist_ok=True)

Path("test/unknown").mkdir(parents=True, exist_ok=True)



## === cell 3
import shutil
import glob
from pathlib import Path


def move_files(src_glob, dest_dir):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for fp in glob.glob(str(src_glob)):
        p = Path(fp)
        if p.is_file():
            target = dest_dir / p.name
            if target.exists():
                try:
                    p.unlink()
                except Exception:
                    pass
                continue
            shutil.move(str(p), str(target))
            moved += 1
    return moved


moved_cats = move_files("raw/train/cat.*.jpg", "train/cats")
moved_dogs = move_files("raw/train/dog.*.jpg", "train/dogs")

moved_test = 0
moved_test += move_files("raw/test/*.jpg", "test/unknown")
moved_test += move_files("raw/test/test/*.jpg", "test/unknown")
moved_test += move_files("raw/test/unknown/*.jpg", "test/unknown")
moved_test += move_files("raw/test/test/unknown/*.jpg", "test/unknown")

print("Moved cats:", moved_cats, "Moved dogs:", moved_dogs, "Moved test:", moved_test)
print("Train cats:", len(list(Path("train/cats").glob("*.jpg"))))
print("Train dogs:", len(list(Path("train/dogs").glob("*.jpg"))))
print("Test files:", len(list(Path("test/unknown").glob("*.jpg"))))



## === cell 4
import random
from pathlib import Path
import shutil

random.seed(42)


def move_random_files(src_dir, dst_dir, n):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = [p for p in src_dir.iterdir() if p.is_file()]
    if len(files) < n:
        raise ValueError(
            f"Not enough files in {src_dir} to move {n}; found {len(files)}"
        )
    chosen = random.sample(files, n)
    for p in chosen:
        target = dst_dir / p.name
        if target.exists():
            try:
                p.unlink()
            except Exception:
                pass
            continue
        shutil.move(str(p), str(target))


if (len(list(Path("valid/cats").glob("*.jpg"))) == 0) and (
    len(list(Path("valid/dogs").glob("*.jpg"))) == 0
):
    move_random_files("train/cats", "valid/cats", 400)
    move_random_files("train/dogs", "valid/dogs", 400)

print(
    "After split - Train cats:",
    len(list(Path("train/cats").glob("*.jpg"))),
    "Train dogs:",
    len(list(Path("train/dogs").glob("*.jpg"))),
)
print(
    "After split - Valid cats:",
    len(list(Path("valid/cats").glob("*.jpg"))),
    "Valid dogs:",
    len(list(Path("valid/dogs").glob("*.jpg"))),
)



## === cell 5
train_dir = "train/"
test_dir = "test/"
valid_dir = "valid/"

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=256,
    class_mode="binary",
    shuffle=True,
    seed=42,
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=128,
    class_mode=None,
    shuffle=False,
)

valid_generator = test_datagen.flow_from_directory(
    valid_dir,
    target_size=(128, 128),
    batch_size=128,
    class_mode="binary",
    shuffle=False,
)

print(
    "train batches:",
    len(train_generator),
    "valid batches:",
    len(valid_generator),
    "test batches:",
    len(test_generator),
)

if test_generator.n == 0:
    raise RuntimeError(
        "Test generator has 0 images. Expected images under test/<subdir>/*.jpg. "
        "Check that test/unknown contains jpg files."
    )

print("Train class_indices:", train_generator.class_indices)
print("Valid class_indices:", valid_generator.class_indices)

if train_generator.class_indices != valid_generator.class_indices:
    raise RuntimeError(
        f"class_indices mismatch between train and valid generators.\n"
        f"train: {train_generator.class_indices}\nvalid: {valid_generator.class_indices}\n"
        "This can silently corrupt validation semantics; please ensure identical folder names."
    )



## === cell 6
model = Sequential(
    [
        Conv2D(16, 3, activation="relu", input_shape=(128, 128, 3)),
        MaxPooling2D(2),
        Conv2D(32, 3, activation="relu"),
        MaxPooling2D(2),
        Conv2D(64, 3, activation="relu"),
        MaxPooling2D(2),
        Conv2D(128, 3, activation="relu"),
        MaxPooling2D(2),
        Flatten(),
        Dense(512, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ]
)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()



## === cell 7
history = model.fit(
    train_generator,
    steps_per_epoch=10,
    epochs=15,
    validation_data=valid_generator,
    verbose=0,
)



## === cell 8
train_accuracy = history.history["accuracy"]
train_loss = history.history["loss"]
val_accuracy = history.history["val_accuracy"]
val_loss = history.history["val_loss"]

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(train_accuracy, label="Training Accuracy")
plt.plot(val_accuracy, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")

plt.subplot(2, 1, 2)
plt.plot(train_loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.legend(loc="lower right")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.xlabel("epoch")
plt.show()



## === cell 9
model.save("model_cat_vs_dogs.h5")



## === cell 10
from tensorflow.keras.models import load_model

model = load_model("model_cat_vs_dogs.h5")



## === cell 11
from pathlib import Path

sample_path = Path(
    "/kaggle/input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv"
)
if not sample_path.exists():
    sample_path = Path("/kaggle/input/sample_submission.csv")
if not sample_path.exists():
    sample_path = Path(
        "/kaggle/data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv"
    )
if not sample_path.exists():
    sample_path = Path("/kaggle/data/sample_submission.csv")

sample = pd.read_csv(sample_path)

pred_list = model.predict(test_generator, verbose=0).reshape(-1)

if len(pred_list) != test_generator.n:
    raise RuntimeError(
        f"Prediction length mismatch: got {len(pred_list)} preds but test_generator.n={test_generator.n}"
    )

ci = train_generator.class_indices

dog_keys = [k for k in ci.keys() if k.lower() in ("dog", "dogs")]
cat_keys = [k for k in ci.keys() if k.lower() in ("cat", "cats")]

if len(dog_keys) == 1 and len(cat_keys) == 1:
    dog_key = dog_keys[0]
    cat_key = cat_keys[0]
    if ci[dog_key] == 0 and ci[cat_key] == 1:
        pred_list = 1.0 - pred_list
else:
    raise RuntimeError(
        f"Unexpected class_indices keys: {ci}. Expected dog/cat (or dogs/cats)."
    )

filenames = test_generator.filenames

id_list = []
for fn in filenames:
    stem = Path(fn).stem
    try:
        id_list.append(int(stem))
    except ValueError:
        toks = [t for t in stem.replace("-", "_").split("_") if t.isdigit()]
        if not toks:
            raise RuntimeError(f"Cannot parse numeric id from filename: {fn}")
        id_list.append(int(toks[-1]))

pred_df = pd.DataFrame({"id": id_list, "label": pred_list})

res = sample[["id"]].merge(pred_df, on="id", how="left")
if res["label"].isna().any():
    missing = int(res["label"].isna().sum())
    bad_ids = res.loc[res["label"].isna(), "id"].head(10).tolist()
    raise RuntimeError(
        f"Missing predictions for {missing} test ids; example missing ids: {bad_ids}. "
        "Check filename/id parsing and generator file discovery."
    )

res["label"] = res["label"].clip(1e-6, 1 - 1e-6)

print("Submission rows:", len(res), "Columns:", list(res.columns))
res.to_csv("submission.csv", index=False)
print("Wrote submission.csv ->", Path("submission.csv").resolve())



## === cell 12
import matplotlib.pyplot as plt

batch = next(iter(test_generator))
imgs = batch if isinstance(batch, np.ndarray) else batch[0]

n_show = min(8, imgs.shape[0])
for i in range(n_show):
    img = imgs[i]
    predict = float(model.predict(np.expand_dims(img, axis=0), verbose=0)[0][0])

    if predict >= 0.5:
        print("It seems like it's a dog! Estimation :", np.round(predict, 2) * 100, "%")
    else:
        print(
            "It seems like it's a cat! Estimation :",
            np.round(1 - predict, 2) * 100,
            "%",
        )
    plt.imshow(img)
    plt.axis("off")
    plt.show()
