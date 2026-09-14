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

3.13

# 3. Installed packages

No external packages required in the script and installed.

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

0.9666001666666668

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing

import os
import random

try:
    import tensorflow as tf
except Exception as e:
    tf = None
    print("WARNING: TensorFlow import failed; cannot train CNN in this environment.")
    print("Import error:", repr(e))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
if tf is not None:
    try:
        tf.random.set_seed(SEED)
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    except Exception as e:
        print("WARNING: tf.random.set_seed failed:", repr(e))

if tf is not None:
    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
    except Exception:
        pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import zipfile

extract_dir = "/kaggle/working"

train_marker = os.path.join(extract_dir, "__train_extracted__")
test_marker = os.path.join(extract_dir, "__test_extracted__")

if not os.path.exists(train_marker):
    with zipfile.ZipFile(
        "/kaggle/input/aerial-cactus-identification/train.zip", "r"
    ) as zip_ref:
        zip_ref.extractall(extract_dir)
    with open(train_marker, "w") as f:
        f.write("ok")

if not os.path.exists(test_marker):
    with zipfile.ZipFile(
        "/kaggle/input/aerial-cactus-identification/test.zip", "r"
    ) as zip_ref:
        zip_ref.extractall(extract_dir)
    with open(test_marker, "w") as f:
        f.write("ok")




## === cell 2
def find_image_dir_fast(root, target_name):
    candidates = [
        os.path.join(root, target_name),
        os.path.join(root, "aerial-cactus-identification", target_name),
        os.path.join(
            root,
            "aerial-cactus-identification",
            "aerial-cactus-identification",
            target_name,
        ),
    ]
    for p in candidates:
        if os.path.isdir(p):
            try:
                with os.scandir(p) as it:
                    for e in it:
                        if e.is_file() and e.name.lower().endswith(".jpg"):
                            return p
            except FileNotFoundError:
                pass

    level1 = []
    try:
        with os.scandir(root) as it:
            for e in it:
                if e.is_dir():
                    level1.append(e.path)
    except FileNotFoundError:
        pass

    for base in level1:
        p = os.path.join(base, target_name)
        if os.path.isdir(p):
            with os.scandir(p) as it:
                for e in it:
                    if e.is_file() and e.name.lower().endswith(".jpg"):
                        return p

        try:
            with os.scandir(base) as it:
                for e in it:
                    if e.is_dir():
                        p2 = os.path.join(e.path, target_name)
                        if os.path.isdir(p2):
                            with os.scandir(p2) as it2:
                                for f in it2:
                                    if f.is_file() and f.name.lower().endswith(".jpg"):
                                        return p2
        except FileNotFoundError:
            pass

    raise FileNotFoundError(
        f"Could not find extracted '{target_name}' directory with jpgs under: {root}"
    )


train_dir = find_image_dir_fast("/kaggle/working", "train")
test_dir = find_image_dir_fast("/kaggle/working", "test")

print("Detected train_dir:", train_dir)
print("Detected test_dir:", test_dir)

train_df = pd.read_csv("/kaggle/input/aerial-cactus-identification/train.csv")
train_df.head(20)




## === cell 3
def count_files(directory):
    with os.scandir(directory) as it:
        return sum(1 for e in it if e.is_file())


train_count = count_files(train_dir)
test_count = count_files(test_dir)

print(f"Train images: {train_count}")
print(f"Test images: {test_count}")



## === cell 4
class_ratio = train_df["has_cactus"].value_counts(normalize=True) * 100
print(class_ratio)



## === cell 5
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_df["has_cactus"]),
    y=train_df["has_cactus"],
)
class_weights_dict = dict(enumerate(class_weights))
print(class_weights_dict)



## === cell 6
train_df["has_cactus"] = train_df["has_cactus"].astype("str")



## === cell 7
if tf is not None:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    def custom_preprocessing(image):
        s = int(np.uint32(image.sum()))
        rng = np.random.default_rng(s ^ SEED)

        k = int(rng.integers(0, 4))
        if k:
            image = np.rot90(image, k)

        if int(rng.integers(0, 2)):
            image = np.fliplr(image)

        if int(rng.integers(0, 2)):
            image = np.flipud(image)

        factor = float(rng.uniform(0.8, 1.2))
        image = image.astype(np.float32)
        image *= factor * (1.0 / 255.0)
        np.clip(image, 0.0, 1.0, out=image)
        return image

    train_datagen = ImageDataGenerator(
        validation_split=0.10,
        preprocessing_function=custom_preprocessing,
    )

    df_for_gen = train_df[["id", "has_cactus"]].copy()

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=df_for_gen,
        directory=train_dir,
        x_col="id",
        y_col="has_cactus",
        target_size=(32, 32),
        subset="training",
        batch_size=512,
        shuffle=True,
        class_mode="binary",
        seed=SEED,
    )

    val_generator = train_datagen.flow_from_dataframe(
        dataframe=df_for_gen,
        directory=train_dir,
        x_col="id",
        y_col="has_cactus",
        target_size=(32, 32),
        subset="validation",
        batch_size=256,
        shuffle=True,
        class_mode="binary",
        seed=SEED,
    )



## === cell 8
if tf is not None:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    test_datagen = ImageDataGenerator(rescale=1 / 255)

    test_parent = os.path.dirname(test_dir)

    test_generator = test_datagen.flow_from_directory(
        directory=test_parent,
        classes=["test"],
        target_size=(32, 32),
        batch_size=512,
        shuffle=False,
        class_mode=None,
    )



## === cell 9
if tf is not None:
    from tensorflow.keras.applications import EfficientNetB3
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.optimizers import Adam

    efficient_net = EfficientNetB3(
        weights="imagenet", input_shape=(32, 32, 3), include_top=False, pooling="max"
    )

    efficient_net.trainable = False

    model = Sequential()
    model.add(efficient_net)
    model.add(Dense(units=120, activation="relu"))
    model.add(Dense(units=120, activation="relu"))
    model.add(Dense(units=1, activation="sigmoid"))
    model.summary()



## === cell 10
if tf is not None:
    model.compile(
        optimizer=Adam(learning_rate=0.00005),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )



## === cell 11
if tf is not None:
    history = model.fit(
        train_generator,
        epochs=70,
        steps_per_epoch=30,
        validation_data=val_generator,
        validation_steps=7,
        class_weight=class_weights_dict,
        verbose=2,
        workers=2,
        use_multiprocessing=True,
        max_queue_size=16,
    )



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1281563289.py in <cell line: 0>()
      3     # Correctness preserved: generator order/shuffle is still controlled by seed; no change to steps/epochs.
      4     # Use a small worker count to avoid overhead/oversubscription in Kaggle.
----> 5     history = model.fit(
      6         train_generator,
      7         epochs=70,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    117             return fn(*args, **kwargs)
    118         except Exception as e:
--> 119             filtered_tb = _process_traceback_frames(e.__traceback__)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`

TypeError: TensorFlowTrainer.fit() got an unexpected keyword argument 'workers'

## === cell 12
if tf is not None:
    preds = model.predict(
        test_generator, verbose=1, workers=2, use_multiprocessing=True
    )



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1702921405.py in <cell line: 0>()
      1 if tf is not None:
      2     # --- Speed: prediction can also pipeline input with workers.
----> 3     preds = model.predict(
      4         test_generator, verbose=1, workers=2, use_multiprocessing=True
      5     )

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    117             return fn(*args, **kwargs)
    118         except Exception as e:
--> 119             filtered_tb = _process_traceback_frames(e.__traceback__)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`

TypeError: TensorFlowTrainer.predict() got an unexpected keyword argument 'workers'

## === cell 13
sample_sub = pd.read_csv(
    "/kaggle/input/aerial-cactus-identification/sample_submission.csv"
)

if tf is not None:
    test_files = [os.path.basename(fn) for fn in test_generator.filenames]
    pred_df = pd.DataFrame(
        {"id": test_files, "has_cactus": preds.reshape(-1).astype(float)}
    )
    submission = sample_sub.merge(pred_df, on="id", how="left", suffixes=("", "_pred"))[
        ["id", "has_cactus"]
    ]

    missing = submission["has_cactus"].isna().sum()
    if missing:
        raise ValueError(
            f"{missing} test ids in sample_submission were not found in generated predictions mapping."
        )
else:
    submission = sample_sub.copy()
    submission["has_cactus"] = 0.5

print(submission.head(20))
print(submission.shape)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/382821859.py in <cell line: 0>()
      6     test_files = [os.path.basename(fn) for fn in test_generator.filenames]
      7     pred_df = pd.DataFrame(
----> 8         {"id": test_files, "has_cactus": preds.reshape(-1).astype(float)}
      9     )
     10     submission = sample_sub.merge(pred_df, on="id", how="left", suffixes=("", "_pred"))[

NameError: name 'preds' is not defined

## === cell 14
submission.to_csv("/kaggle/working/submission.csv", index=False)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/482570615.py in <cell line: 0>()
----> 1 submission.to_csv("/kaggle/working/submission.csv", index=False)
      2 

NameError: name 'submission' is not defined

## === cell 15
print(os.listdir("/kaggle/working"))
print("Wrote:", "/kaggle/working/submission.csv")
