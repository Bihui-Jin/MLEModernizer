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

3.10

# 3. Installed packages

geopandas==0.14.4
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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124

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

0.56091

# 6. Current score

0.69603

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.69315) has done: 'I fix the TensorFlow import crash by removing the unnecessary `torch` import and setting stable TF runtime options, so the notebook can start. Then I correct the train/test file paths after extracting zips, fix the label parsing and the train/val split to avoid index errors, and ensure the `tf.data` pipeline keeps file paths as strings (preventing the `ReadFile` float32 error). Finally, I generate predictions as dog probabilities (not argmax class IDs), keep all test IDs (no dropped remainder), sort by ID, and write a valid `submission.csv` matching the required `id,label` format.'
- What this solution (achieved 0.69315) has done: 'I fix the TensorFlow/protobuf import crash by setting the environment variables *before* importing TensorFlow, which is the root cause of the `MessageFactory.GetPrototype` error in this environment. Then I fix the training `StopIteration` by ensuring the training dataset is repeated and by providing an explicit `steps_per_epoch` (this preserves the same training semantics while making Keras+MirroredStrategy iteration stable). I also correct the extracted folder paths for train/test images (the zip extracts into nested folders), so you actually load images instead of an empty dataset. These changes should both unblock end-to-end execution and improve score from ~0.693 (near-random) toward the target by enabling real training and correct inference.'
- What this solution (achieved 0.69315) has done: 'I fix the protobuf/TensorFlow import crash by setting the required environment variables before any TensorFlow-related import and by forcing protobuf to use the pure-Python implementation (this directly addresses the `MessageFactory.GetPrototype` error). Then I fix the `StopIteration` during `model.fit()` by ensuring the distributed input pipeline is fully repeatable and by setting Keras distribution options that avoid premature iterator exhaustion under `MirroredStrategy` while keeping the same training semantics (same data, epochs, steps). Finally, I keep the submission logic intact but add a couple of safety checks so `submission.csv` is always produced with the correct `id,label` format and alignment to the sample submission IDs.'
- What this solution (achieved 0.69315) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation before importing TensorFlow, and (as an extra safeguard) downgrading protobuf in-process if the environment still exposes the `MessageFactory.GetPrototype` issue. Then I correct the extracted train/test image directory discovery to reliably locate the actual `.jpg` files under the nested zip structure so you don’t train on an empty dataset. Finally, I fix the `StopIteration` during `model.fit()` under `MirroredStrategy` by ensuring the distributed dataset uses an explicit `AutoShardPolicy` and `steps_per_epoch` that match the repeated dataset, while keeping your model, loss, and training loop semantics intact so the score can improve toward the target.'
- What this solution (achieved 0.69325) has done: 'The runtime errors come from incorrect assumptions about where the ZIPs extract (the images end up under nested `train/` and `test/` folders), so the script trains on an empty dataset and later can’t find any test images. I fix the path discovery to reliably locate the actual `.jpg` files for both train and test by searching recursively and preferring the leaf directory with images, which unblocks end-to-end execution and should improve log loss toward the target by enabling real training/prediction. I also fix the `StopIteration` during `model.fit()` by using Keras’ `steps_per_execution=1` and ensuring the distributed datasets have a stable distribution/sharding policy while keeping the same model, loss, and training loop semantics. Finally, I keep the submission format identical but add a hard check that we produce exactly the sample submission IDs in the right order.'
- What this solution (achieved 0.69603) has done: 'The `StopIteration` happens because under `MirroredStrategy` your input pipeline is not safely sharded/repeated across replicas, so the distributed iterator exhausts even though the dataset has `.repeat()`. I fix this by setting `AutoShardPolicy.DATA` (more stable for in-memory tensor-slice datasets) and by explicitly distributing the datasets via `strategy.experimental_distribute_dataset`, while keeping your model, loss, optimizer, epochs, and steps semantics unchanged. I also make `train_steps` use the exact number of full batches (since you use `drop_remainder=True`) to avoid any mismatch, and guard plotting so it doesn’t crash if training fails. These changes are score-neutral logically (they just ensure the same intended training actually runs) and should improve log loss from ~0.693 (near-random due to failed training) toward your 0.56091 target.'

# 9. Code solution

## === cell 0
import os, glob, zipfile, sys, subprocess

os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.environ.get("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")


def _ensure_compatible_protobuf():
    """
    Bug fix: Some Kaggle TF builds can crash with newer protobuf C++ bindings.
    Keep minimal: prefer pure-python protobuf; if protobuf major>=5, install protobuf<5.
    """
    try:
        import google.protobuf  # noqa: F401
        from importlib.metadata import version

        v = version("protobuf")
        major = int(v.split(".")[0])
        if major >= 5:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
            )
            import importlib

            importlib.invalidate_caches()
            for m in list(sys.modules):
                if m.startswith("google.protobuf"):
                    del sys.modules[m]
    except Exception as e:
        print("Protobuf compatibility check/install skipped due to:", repr(e))


_ensure_compatible_protobuf()

import numpy as np
import pandas as pd

import tensorflow as tf
import matplotlib.pyplot as plt

AUTOTUNE = tf.data.AUTOTUNE

print("TensorFlow:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))

tf.random.set_seed(42)
np.random.seed(42)



## === cell 1
train_zip_path = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip"
test_zip_path = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip"

with zipfile.ZipFile(train_zip_path, "r") as z:
    z.extractall("/kaggle/working/")
with zipfile.ZipFile(test_zip_path, "r") as z:
    z.extractall("/kaggle/working/")

print("Extracted train exists:", os.path.isdir("/kaggle/working/train"))
print("Extracted test exists:", os.path.isdir("/kaggle/working/test"))

print("Top-level /kaggle/working entries:", sorted(os.listdir("/kaggle/working"))[:20])



## === cell 2
dataset_path = "./dataset"
if not os.path.isdir(dataset_path):
    os.mkdir(dataset_path)

train_path = os.path.join(dataset_path, "train")
val_path = os.path.join(dataset_path, "val")
test_path = os.path.join(dataset_path, "test")

for p in [train_path, val_path, test_path]:
    if not os.path.isdir(p):
        os.mkdir(p)




## === cell 3
def get_path(path, ext):
    return glob.glob(os.path.join(path, f"*.{ext}"))


def check(path):
    base = os.path.basename(path)  # dog.1234.jpg
    animal = base.split(".")[0]  # dog or cat
    return 1 if animal == "dog" else 0


def find_leaf_dir_with_jpg(root_dir):
    """
    Bug fix: ZIPs can extract into nested subfolders; reliably discover jpg files.
    Returns (leaf_dir, list_of_jpg_paths).
    """
    candidates = sorted(
        glob.glob(os.path.join(root_dir, "**", "*.jpg"), recursive=True)
    )
    if not candidates:
        return None, []
    parents = [os.path.dirname(p) for p in candidates]
    parent_counts = {}
    for p in parents:
        parent_counts[p] = parent_counts.get(p, 0) + 1
    leaf_dir = max(parent_counts.items(), key=lambda kv: kv[1])[0]
    leaf_candidates = sorted(glob.glob(os.path.join(leaf_dir, "*.jpg")))
    return leaf_dir, leaf_candidates if leaf_candidates else candidates


def find_train_images():
    """
    Bug fix: handle all known extraction layouts:
    - /kaggle/working/train/train/*.jpg (original kernels)
    - /kaggle/working/train/*.jpg
    - /kaggle/working/dogs-vs-cats-redux-kernels-edition/train/... (nested)
    """
    preferred = [
        "/kaggle/working/train/train",
        "/kaggle/working/train",
        "/kaggle/working/dogs-vs-cats-redux-kernels-edition/train/train",
        "/kaggle/working/dogs-vs-cats-redux-kernels-edition/train",
    ]
    for d in preferred:
        if os.path.isdir(d):
            lst = sorted(get_path(d, "jpg"))
            if lst:
                return d, lst
    for root in [
        "/kaggle/working/train",
        "/kaggle/working/dogs-vs-cats-redux-kernels-edition/train",
        "/kaggle/working",
    ]:
        if os.path.isdir(root):
            found_dir, candidates = find_leaf_dir_with_jpg(root)
            if candidates:
                return found_dir, sorted(candidates)
    return None, []


def find_test_images():
    """
    Bug fix: test zip often extracts to /kaggle/working/test/test/*.jpg
    but sometimes to nested unknown/ folders. Discover recursively.
    """
    preferred = [
        "/kaggle/working/test/test",
        "/kaggle/working/test",
        "/kaggle/working/dogs-vs-cats-redux-kernels-edition/test/test",
        "/kaggle/working/dogs-vs-cats-redux-kernels-edition/test",
    ]
    for d in preferred:
        if os.path.isdir(d):
            lst = sorted(get_path(d, "jpg"))
            if lst:
                return d, lst
    for root in [
        "/kaggle/working/test",
        "/kaggle/working/dogs-vs-cats-redux-kernels-edition/test",
        "/kaggle/working",
    ]:
        if os.path.isdir(root):
            found_dir, candidates = find_leaf_dir_with_jpg(root)
            if candidates:
                return found_dir, sorted(candidates)
    return None, []




## === cell 4
train_img_dir, data_list = find_train_images()
if not data_list:
    raise RuntimeError(
        "No training images found. Checked common locations and recursive search under /kaggle/working."
    )

result = list(map(check, data_list))

print("Train image dir:", train_img_dir)
print("Total train images found:", len(data_list))
print("dogs:", result.count(1), "cats:", result.count(0))



## === cell 5
dogs_list = [i for i in data_list if check(i) == 1]
cats_list = [i for i in data_list if check(i) == 0]
print("dogs_list:", len(dogs_list), "cats_list:", len(cats_list))



## === cell 6
split_ratio = 0.8
rng = np.random.RandomState(42)



## === cell 7
dogs_list_shuf = dogs_list.copy()
cats_list_shuf = cats_list.copy()
rng.shuffle(dogs_list_shuf)
rng.shuffle(cats_list_shuf)

n = min(len(dogs_list_shuf), len(cats_list_shuf))
dogs_list_shuf = dogs_list_shuf[:n]
cats_list_shuf = cats_list_shuf[:n]

split_idx = int(n * split_ratio)

train_data = dogs_list_shuf[:split_idx] + cats_list_shuf[:split_idx]
val_data = dogs_list_shuf[split_idx:] + cats_list_shuf[split_idx:]

rng.shuffle(train_data)
rng.shuffle(val_data)

train_label = list(map(check, train_data))
val_label = list(map(check, val_data))

print("Train size:", len(train_data), "Val size:", len(val_data))



## === cell 8
class_label = ["cat", "dog"]  # not used directly; kept for compatibility



## === cell 9
img_size = 224


def preprocess_image(image_bytes):
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, [img_size, img_size])
    image = tf.cast(image, tf.float32) / 255.0
    return image




## === cell 10
def load_and_preprocess_image(path):
    path = tf.convert_to_tensor(path, dtype=tf.string)
    image = tf.io.read_file(path)
    return preprocess_image(image)




## === cell 11
train_paths = tf.constant(train_data, dtype=tf.string)
val_paths = tf.constant(val_data, dtype=tf.string)
train_labels = tf.constant(train_label, dtype=tf.int32)
val_labels = tf.constant(val_label, dtype=tf.int32)

ds_train = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
ds_val = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))


def load_and_preprocess_from_path_label(path, label):
    img = load_and_preprocess_image(path)
    return img, tf.one_hot(label, 2)


ds_train = ds_train.map(
    load_and_preprocess_from_path_label, num_parallel_calls=AUTOTUNE
)
ds_val = ds_val.map(load_and_preprocess_from_path_label, num_parallel_calls=AUTOTUNE)



## === cell 12
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers



## === cell 13
batch_size = 64

train_steps = max(1, len(train_data) // batch_size)
val_steps = max(1, int(np.ceil(len(val_data) / batch_size)))

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = (
    tf.data.experimental.AutoShardPolicy.DATA
)

dsb_train = (
    ds_train.shuffle(2048, seed=42, reshuffle_each_iteration=True)
    .batch(batch_size=batch_size, drop_remainder=True)
    .repeat()
    .with_options(options)
    .prefetch(AUTOTUNE)
)
dsb_val = (
    ds_val.batch(batch_size=batch_size, drop_remainder=False)
    .with_options(options)
    .prefetch(AUTOTUNE)
)

print("train_steps:", train_steps, "val_steps:", val_steps)



## === cell 14
_ = EfficientNetB0(
    weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3)
)
print("EfficientNetB0 backbone loaded OK")



## === cell 15
img_augmentation = Sequential(
    [
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(),
        layers.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)




## === cell 16
def build_model(num_classes):
    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = img_augmentation(inputs)
    backbone = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")

    backbone.trainable = False

    x = layers.GlobalAveragePooling2D(name="avg_pool")(backbone.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    model = tf.keras.Model(inputs, outputs, name="EfficientNet")
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        steps_per_execution=1,
    )
    return model




## === cell 17
try:
    strategy = tf.distribute.MirroredStrategy()
except Exception as e:
    print("MirroredStrategy failed, falling back to default strategy:", repr(e))
    strategy = tf.distribute.get_strategy()

print("Num replicas in strategy:", strategy.num_replicas_in_sync)



## === cell 18
with strategy.scope():
    new_model = build_model(num_classes=2)

dist_train = strategy.experimental_distribute_dataset(dsb_train)
dist_val = strategy.experimental_distribute_dataset(dsb_val)

epochs = 10
hist = new_model.fit(
    dist_train,
    epochs=epochs,
    steps_per_epoch=train_steps,
    validation_data=dist_val,
    validation_steps=val_steps,
    verbose=2,
)




## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
StopIteration                             Traceback (most recent call last)
/tmp/ipykernel_12/4133685917.py in <cell line: 0>()
      7 
      8 epochs = 10
----> 9 hist = new_model.fit(
     10     dist_train,
     11     epochs=epochs,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/distribute/input_lib.py in __next__(self)
    264       return self.get_next()
    265     except errors.OutOfRangeError:
--> 266       raise StopIteration
    267 
    268   def __iter__(self):

StopIteration: 

## === cell 19
def plot_hist(hist):
    plt.plot(hist.history.get("accuracy", []))
    plt.plot(hist.history.get("val_accuracy", []))
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


if "hist" in globals() and hasattr(hist, "history"):
    plot_hist(hist)



## === cell 20
test_img_dir, test_list = find_test_images()
if not test_list:
    raise RuntimeError(
        "No test images found. Checked common locations and recursive search under /kaggle/working."
    )


def id_load(x):
    return int(os.path.basename(x).split(".")[0])


id_list = list(map(id_load, test_list))
print("Test image dir:", test_img_dir)
print("Total test images found:", len(test_list), "Example ids:", id_list[:5])



## === cell 21
test_paths = tf.constant(test_list, dtype=tf.string)
test_ids = tf.constant(id_list, dtype=tf.int32)

ds_test = tf.data.Dataset.from_tensor_slices((test_paths, test_ids))


def test_map(path, id_):
    return load_and_preprocess_image(path), id_


ds_test = ds_test.map(test_map, num_parallel_calls=AUTOTUNE).with_options(options)
dsb_test = ds_test.batch(batch_size=100, drop_remainder=False).prefetch(AUTOTUNE)



## === cell 22
for imgs, ids in dsb_test.take(1):
    print("Batch imgs:", imgs.shape, imgs.dtype)
    print("Batch ids:", ids.shape, ids.dtype, ids[:10].numpy())



## === cell 23
all_ids = []
all_probs = []

for imgs, ids in dsb_test:
    probs = new_model.predict(imgs, verbose=0)
    dog_probs = probs[:, 1]  # probability of dog
    all_ids.extend(ids.numpy().tolist())
    all_probs.extend(dog_probs.tolist())

submission_df = pd.DataFrame({"id": all_ids, "label": all_probs})
submission_df["id"] = submission_df["id"].astype(int)
submission_df["label"] = submission_df["label"].astype(float)
submission_df = submission_df.sort_values("id").reset_index(drop=True)

print(submission_df.head())
print(
    "Submission rows:", len(submission_df), "Unique ids:", submission_df["id"].nunique()
)



## === cell 24
sample_path = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv"
sample = pd.read_csv(sample_path)

merged = sample[["id"]].merge(submission_df, on="id", how="left")
merged["label"] = merged["label"].astype(float).fillna(0.5).clip(1e-7, 1 - 1e-7)
merged["id"] = merged["id"].astype(int)
merged = merged.sort_values("id").reset_index(drop=True)

assert len(merged) == len(sample), (len(merged), len(sample))
assert merged["id"].tolist() == sample["id"].astype(int).sort_values().tolist()

merged.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", merged.shape)
print(merged.head())
print("CSV path:", os.path.abspath("submission.csv"))
