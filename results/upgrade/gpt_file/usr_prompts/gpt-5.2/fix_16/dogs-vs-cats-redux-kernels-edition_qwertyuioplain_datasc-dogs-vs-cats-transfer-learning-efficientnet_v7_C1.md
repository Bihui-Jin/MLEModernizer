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

0.02736

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.69315) has done: 'I fix the TensorFlow import crash by removing the unnecessary `torch` import and setting stable TF runtime options, so the notebook can start. Then I correct the train/test file paths after extracting zips, fix the label parsing and the train/val split to avoid index errors, and ensure the `tf.data` pipeline keeps file paths as strings (preventing the `ReadFile` float32 error). Finally, I generate predictions as dog probabilities (not argmax class IDs), keep all test IDs (no dropped remainder), sort by ID, and write a valid `submission.csv` matching the required `id,label` format.'
- What this solution (achieved 0.69315) has done: 'I fix the TensorFlow/protobuf import crash by setting the environment variables *before* importing TensorFlow, which is the root cause of the `MessageFactory.GetPrototype` error in this environment. Then I fix the training `StopIteration` by ensuring the training dataset is repeated and by providing an explicit `steps_per_epoch` (this preserves the same training semantics while making Keras+MirroredStrategy iteration stable). I also correct the extracted folder paths for train/test images (the zip extracts into nested folders), so you actually load images instead of an empty dataset. These changes should both unblock end-to-end execution and improve score from ~0.693 (near-random) toward the target by enabling real training and correct inference.'
- What this solution (achieved 0.69315) has done: 'I fix the protobuf/TensorFlow import crash by setting the required environment variables before any TensorFlow-related import and by forcing protobuf to use the pure-Python implementation (this directly addresses the `MessageFactory.GetPrototype` error). Then I fix the `StopIteration` during `model.fit()` by ensuring the distributed input pipeline is fully repeatable and by setting Keras distribution options that avoid premature iterator exhaustion under `MirroredStrategy` while keeping the same training semantics (same data, epochs, steps). Finally, I keep the submission logic intact but add a couple of safety checks so `submission.csv` is always produced with the correct `id,label` format and alignment to the sample submission IDs.'
- What this solution (achieved 0.69315) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation before importing TensorFlow, and (as an extra safeguard) downgrading protobuf in-process if the environment still exposes the `MessageFactory.GetPrototype` issue. Then I correct the extracted train/test image directory discovery to reliably locate the actual `.jpg` files under the nested zip structure so you don’t train on an empty dataset. Finally, I fix the `StopIteration` during `model.fit()` under `MirroredStrategy` by ensuring the distributed dataset uses an explicit `AutoShardPolicy` and `steps_per_epoch` that match the repeated dataset, while keeping your model, loss, and training loop semantics intact so the score can improve toward the target.'
- What this solution (achieved 0.69325) has done: 'The runtime errors come from incorrect assumptions about where the ZIPs extract (the images end up under nested `train/` and `test/` folders), so the script trains on an empty dataset and later can’t find any test images. I fix the path discovery to reliably locate the actual `.jpg` files for both train and test by searching recursively and preferring the leaf directory with images, which unblocks end-to-end execution and should improve log loss toward the target by enabling real training/prediction. I also fix the `StopIteration` during `model.fit()` by using Keras’ `steps_per_execution=1` and ensuring the distributed datasets have a stable distribution/sharding policy while keeping the same model, loss, and training loop semantics. Finally, I keep the submission format identical but add a hard check that we produce exactly the sample submission IDs in the right order.'
- What this solution (achieved 0.69603) has done: 'The `StopIteration` happens because under `MirroredStrategy` your input pipeline is not safely sharded/repeated across replicas, so the distributed iterator exhausts even though the dataset has `.repeat()`. I fix this by setting `AutoShardPolicy.DATA` (more stable for in-memory tensor-slice datasets) and by explicitly distributing the datasets via `strategy.experimental_distribute_dataset`, while keeping your model, loss, optimizer, epochs, and steps semantics unchanged. I also make `train_steps` use the exact number of full batches (since you use `drop_remainder=True`) to avoid any mismatch, and guard plotting so it doesn’t crash if training fails. These changes are score-neutral logically (they just ensure the same intended training actually runs) and should improve log loss from ~0.693 (near-random due to failed training) toward your 0.56091 target.'
- What this solution (achieved 0.70251) has done: 'The `StopIteration` is coming from distributing an already batched/repeated dataset via `strategy.experimental_distribute_dataset` and then feeding that distributed iterator into `model.fit`, which can exhaust unexpectedly in recent TF/Keras builds. The minimal/stable fix is to let Keras handle distribution by passing the regular `tf.data.Dataset` objects directly to `fit`, while keeping the same `.repeat()` and explicit `steps_per_epoch` (so the training semantics stay the same). I also set the dataset `auto_shard_policy` to `OFF` (instead of `DATA`) because this is a tensor-slices pipeline with file I/O and tends to be the most robust under `MirroredStrategy` in Kaggle kernels. Everything else (model, optimizer, loss, epochs, steps, and submission creation) is kept intact so the score should move down from near-random toward your 0.56091 target.'
- What this solution (achieved 0.70917) has done: 'I fix the `StopIteration` crash by making the training and validation datasets explicitly infinite and correctly sharded for `MirroredStrategy`, which prevents the distributed iterator from exhausting unexpectedly while keeping the same model, loss, optimizer, epochs, and steps. Concretely, I set `drop_remainder=True` for validation batching (so each replica always gets the same per-step shapes) and add `.repeat()` to the validation dataset since you also pass `validation_steps`. I also compute `val_steps` consistently with `drop_remainder=True` to avoid any step/batch mismatch. These are execution-stability fixes (not architecture changes) and should also improve log loss vs the current partially/incorrectly trained run by ensuring training actually completes.'
- What this solution (achieved 0.69706) has done: 'I fix the root cause of the cascade of `NameError`s by ensuring the training image discovery does not accidentally pick up non-training JPGs (e.g., test/unknown) during recursive search, which currently can lead to an empty `train_data` after label parsing. Then I make the dataset options object defined once (globally) so it’s available for both train/val and test pipelines even if you later re-order or re-run cells. Finally, I keep the model/training logic intact but ensure the pipelines are always built and that a valid `submission.csv` is written aligned to `sample_submission.csv` IDs.'
- What this solution (achieved 0.69384) has done: 'The crash in cell 13 is because `train_data` ends up empty: the current `find_train_images()` only accepts `cat.*`/`dog.*` filenames, but in your extracted folder layout the training images are under `train/cat/*.jpg` and `train/dog/*.jpg` (no `cat.`/`dog.` prefix), so everything gets filtered out. I minimally extend the label parsing to support both filename styles (either `cat.123.jpg`/`dog.123.jpg` or `.../cat/xxx.jpg`/`.../dog/xxx.jpg`) and update the train image discovery to correctly pick up the class-subfolder layout without changing the model/training core logic. This should both fix the runtime error and improve score toward the 0.56091 target because the model actually train on the real labeled images instead of failing/degenerating. Submission writing remains the same `id,label` format and still be aligned to `sample_submission.csv`.'
- What this solution (achieved 0.69266) has done: 'Your current log loss (0.69384) is close to random, and the most likely reason is underfitting from keeping the ImageNet backbone fully frozen while using a very large learning rate (1e-2) on only the tiny classification head. To move toward the target (0.56091) with minimal disruption to your core approach, I (1) unfreeze only the last small portion of the EfficientNet backbone after the initial head training and run a short fine-tuning phase with a lower learning rate, and (2) keep the same architecture, loss, data pipeline, and submission logic intact. This is a standard, minimal extension of the same training loop (still `model.fit`) that typically yields a meaningful log-loss drop without changing evaluation semantics. I also clip predictions before writing to CSV to avoid any potential `log(0)` issues (tiny, safe improvement for log loss).'
- What this solution (achieved 0.69231) has done: 'I fix the fine-tuning crash by giving the EfficientNet backbone an explicit, stable layer name inside `build_model()` and then retrieving it robustly during fine-tuning (with a safe fallback), so cell 19 no longer errors. I also make the validation pipeline infinite when `validation_steps` is provided, matching the training pipeline and preventing distributed `StopIteration` edge cases, without changing the model or training semantics. Finally, I keep submission generation unchanged but add a couple of safety casts/clips to ensure the `id,label` CSV is always valid and aligned to the sample submission, so you can submit and (with fine-tuning now actually running) move log loss down toward the 0.56091 target.'
- What this solution (achieved 0.6928) has done: 'I fix the fine-tuning crash by reliably locating the EfficientNet backbone even when Keras inlines its layers into the top-level model (so there is no `efficientnetb0` layer name to fetch). This is a minimal change confined to the fine-tuning cell: it doesn’t alter the model architecture, training loop structure, loss, or data pipeline—only how we reference the backbone for unfreezing. With fine-tuning now actually running, the score should move down from ~0.692 toward your 0.56091 target, while the submission-writing logic remains unchanged and still produces a valid `submission.csv`. I also keep everything compatible with the Kaggle TF runtime by preserving the existing protobuf/TF environment guards.'
- What this solution (achieved 0.02736) has done: 'Your current score (0.6928) is still close to random log loss, so the smallest likely lever to move toward 0.56091 is to fix probability calibration/underfitting without changing the architecture or data pipeline. I keep the same EfficientNetB0+GAP+BN+Dropout+Dense(softmax) model and the same `model.fit` flow, but make two minimal training adjustments that usually reduce log loss: (1) use a sane head-training learning rate (1e-3 instead of 1e-2) to avoid unstable/poorly calibrated softmax outputs, and (2) unfreeze by targeting the actual EfficientNet backbone model object (instead of “guessing” a layer index in the flattened layer list), then fine-tune the last ~20 layers of that backbone at 1e-4. I also apply EfficientNet’s expected `preprocess_input` normalization (this is not a feature change; it matches the pretrained weights’ assumptions and typically improves log loss with minimal risk). The submission format and ID alignment remain unchanged.'

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

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = (
    tf.data.experimental.AutoShardPolicy.OFF
)



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
    base = os.path.basename(path).lower()  # dog.1234.jpg or 1234.jpg
    if base.startswith("dog."):
        return 1
    if base.startswith("cat."):
        return 0

    parts = os.path.normpath(path).lower().split(os.sep)
    if "dog" in parts:
        return 1
    if "cat" in parts:
        return 0

    raise ValueError(f"Could not infer label from path: {path}")


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


def _filter_train_like(paths):
    """
    Bug fix: accept either filename-based labels (cat.* / dog.*) OR folder-based labels (.../cat/... , .../dog/...).
    Also avoid test-only numeric ids (e.g., 1234.jpg in test) by requiring either label in name or label folder.
    """
    out = []
    for p in paths:
        b = os.path.basename(p).lower()
        if b.startswith("cat.") or b.startswith("dog."):
            out.append(p)
            continue
        parts = os.path.normpath(p).lower().split(os.sep)
        if "cat" in parts or "dog" in parts:
            out.append(p)
    return out


def find_train_images():
    """
    Bug fix: handle known extraction layouts including class-subfolder format:
    - /kaggle/working/train/cat/*.jpg and /kaggle/working/train/dog/*.jpg
    - /kaggle/working/train/train/*.jpg (old kernels)
    - /kaggle/working/train/*.jpg
    - nested under /kaggle/working/dogs-vs-cats-redux-kernels-edition/...
    """
    class_folder_roots = [
        "/kaggle/working/train",
        "/kaggle/working/dogs-vs-cats-redux-kernels-edition/train",
    ]
    for root in class_folder_roots:
        cat_dir = os.path.join(root, "cat")
        dog_dir = os.path.join(root, "dog")
        if os.path.isdir(cat_dir) and os.path.isdir(dog_dir):
            cats = sorted(glob.glob(os.path.join(cat_dir, "*.jpg")))
            dogs = sorted(glob.glob(os.path.join(dog_dir, "*.jpg")))
            combined = dogs + cats
            combined = _filter_train_like(combined)
            if combined:
                return root, combined

    preferred = [
        "/kaggle/working/train/train",
        "/kaggle/working/train",
        "/kaggle/working/dogs-vs-cats-redux-kernels-edition/train/train",
        "/kaggle/working/dogs-vs-cats-redux-kernels-edition/train",
    ]
    for d in preferred:
        if os.path.isdir(d):
            lst = sorted(get_path(d, "jpg"))
            lst = _filter_train_like(lst)
            if lst:
                return d, lst

    for root in [
        "/kaggle/working/train",
        "/kaggle/working/dogs-vs-cats-redux-kernels-edition/train",
    ]:
        if os.path.isdir(root):
            found_dir, candidates = find_leaf_dir_with_jpg(root)
            candidates = _filter_train_like(candidates)
            if candidates:
                return found_dir, sorted(candidates)

    if os.path.isdir("/kaggle/working"):
        found_dir, candidates = find_leaf_dir_with_jpg("/kaggle/working")
        candidates = _filter_train_like(candidates)
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
                numeric = []
                for p in candidates:
                    b = os.path.basename(p)
                    stem = b.split(".")[0]
                    if stem.isdigit():
                        numeric.append(p)
                return found_dir, sorted(numeric if numeric else candidates)
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


from tensorflow.keras.applications.efficientnet import (
    preprocess_input as effnet_preprocess_input,
)


def preprocess_image(image_bytes):
    image = tf.image.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(image, [img_size, img_size])
    image = tf.cast(image, tf.float32)
    image = effnet_preprocess_input(image)
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

train_steps = len(train_data) // batch_size
val_steps = len(val_data) // batch_size

if train_steps < 1:
    raise RuntimeError(
        f"Not enough training samples ({len(train_data)}) for batch_size={batch_size} with drop_remainder=True."
    )
if val_steps < 1:
    print(
        f"Warning: Not enough validation samples ({len(val_data)}) for batch_size={batch_size}. "
        "Validation will be disabled to avoid StopIteration."
    )

dsb_train = (
    ds_train.shuffle(2048, seed=42, reshuffle_each_iteration=True)
    .batch(batch_size=batch_size, drop_remainder=True)
    .repeat()
    .with_options(options)
    .prefetch(AUTOTUNE)
)

if val_steps >= 1:
    dsb_val = (
        ds_val.batch(batch_size=batch_size, drop_remainder=True)
        .repeat()
        .with_options(options)
        .prefetch(AUTOTUNE)
    )
else:
    dsb_val = (
        ds_val.batch(batch_size=batch_size, drop_remainder=True)
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

    backbone = EfficientNetB0(
        include_top=False, input_tensor=x, weights="imagenet", name="efficientnetb0"
    )
    backbone.trainable = False

    x = layers.GlobalAveragePooling2D(name="avg_pool")(backbone.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    model = tf.keras.Model(inputs, outputs, name="EfficientNet")

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

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

epochs = 10

if val_steps >= 1:
    hist = new_model.fit(
        dsb_train,
        epochs=epochs,
        steps_per_epoch=train_steps,
        validation_data=dsb_val,
        validation_steps=val_steps,
        verbose=2,
    )
else:
    hist = new_model.fit(
        dsb_train,
        epochs=epochs,
        steps_per_epoch=train_steps,
        verbose=2,
    )




## === cell 19
def _get_backbone_model(m):
    for l in m.layers:
        if isinstance(l, tf.keras.Model) and l.name.startswith("efficientnet"):
            return l
    return None


with strategy.scope():
    backbone = _get_backbone_model(new_model)
    if backbone is None:
        raise ValueError(
            "Could not find EfficientNet backbone submodel for fine-tuning."
        )

    backbone.trainable = True

    fine_tune_at = max(0, len(backbone.layers) - 20)
    for l in backbone.layers[:fine_tune_at]:
        l.trainable = False

    new_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        steps_per_execution=1,
    )

fine_tune_epochs = 2
if val_steps >= 1:
    hist_ft = new_model.fit(
        dsb_train,
        epochs=epochs + fine_tune_epochs,
        initial_epoch=epochs,
        steps_per_epoch=train_steps,
        validation_data=dsb_val,
        validation_steps=val_steps,
        verbose=2,
    )
else:
    hist_ft = new_model.fit(
        dsb_train,
        epochs=epochs + fine_tune_epochs,
        initial_epoch=epochs,
        steps_per_epoch=train_steps,
        verbose=2,
    )




## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3974872118.py in <cell line: 0>()
     11     backbone = _get_backbone_model(new_model)
     12     if backbone is None:
---> 13         raise ValueError(
     14             "Could not find EfficientNet backbone submodel for fine-tuning."
     15         )

ValueError: Could not find EfficientNet backbone submodel for fine-tuning.

## === cell 20
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



## === cell 21
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



## === cell 22
test_paths = tf.constant(test_list, dtype=tf.string)
test_ids = tf.constant(id_list, dtype=tf.int32)

ds_test = tf.data.Dataset.from_tensor_slices((test_paths, test_ids))


def test_map(path, id_):
    return load_and_preprocess_image(path), id_


ds_test = ds_test.map(test_map, num_parallel_calls=AUTOTUNE).with_options(options)
dsb_test = ds_test.batch(batch_size=100, drop_remainder=False).prefetch(AUTOTUNE)



## === cell 23
for imgs, ids in dsb_test.take(1):
    print("Batch imgs:", imgs.shape, imgs.dtype)
    print("Batch ids:", ids.shape, ids.dtype, ids[:10].numpy())



## === cell 24
all_ids = []
all_probs = []

for imgs, ids in dsb_test:
    probs = new_model.predict(imgs, verbose=0)
    probs = np.asarray(probs)
    dog_probs = probs[:, 1]  # probability of dog
    dog_probs = np.clip(dog_probs, 1e-7, 1.0 - 1e-7)
    all_ids.extend(np.asarray(ids).astype(int).tolist())
    all_probs.extend(dog_probs.astype(float).tolist())

submission_df = pd.DataFrame({"id": all_ids, "label": all_probs})
submission_df["id"] = submission_df["id"].astype(int)
submission_df["label"] = submission_df["label"].astype(float)
submission_df = submission_df.sort_values("id").reset_index(drop=True)

print(submission_df.head())
print(
    "Submission rows:", len(submission_df), "Unique ids:", submission_df["id"].nunique()
)



## === cell 25
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
