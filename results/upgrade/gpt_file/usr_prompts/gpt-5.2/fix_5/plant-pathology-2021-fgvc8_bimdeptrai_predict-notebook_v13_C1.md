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
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.7467036011080335

# 6. Current score

0.24507

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.25674) has done: 'I remove the `tensorflow_addons` import that triggers the protobuf `MessageFactory` error, since it isn’t used anywhere in your pipeline. I also fix the missing pretrained model file by switching to a built-in Keras `MobileNetV2` backbone (same model family as your intended `.h5`) so the notebook can run end-to-end without external inputs. Finally, I correct the submission-writing logic bugs (`==` vs `=`, chained assignment, and the always-true `or` condition) and ensure predictions are mapped to the correct label columns with a safe fallback so every row gets at least one label, producing a valid `submission.csv`.'
- What this solution (achieved 0.24507) has done: 'The timeout is dominated by CPU-side image loading/augmentation and large 512×512 inputs flowing through MobileNetV2, plus extra overhead from `ImageDataGenerator` not prefetching/parallelizing efficiently. To keep the exact same model/training logic and semantics, the main speedups are: (1) enable Keras generator multiprocessing workers with deterministic seeding, (2) cache and reuse computed step counts to avoid generator-length overhead, (3) vectorize the per-class threshold search (removing Python loops), and (4) vectorize label assembly for the submission to reduce per-row Python work. These changes are equivalent in results (same data, same augmentations, same epochs/loss/model), but cut wall time by reducing input pipeline stalls and Python overhead.'

# 9. Code solution

## === cell 0
import os
import random

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras

from sklearn.preprocessing import MultiLabelBinarizer

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTHONHASHSEED", "42")
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
TRAIN_CSV = "../input/plant-pathology-2021-fgvc8/train.csv"
SAMPLE_SUB = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"
TRAIN_DIR = "../input/plant-pathology-2021-fgvc8/train_images"
TEST_DIR = "../input/plant-pathology-2021-fgvc8/test_images"

train = pd.read_csv(TRAIN_CSV)
submissions = pd.read_csv(SAMPLE_SUB)

print("train:", train.shape, "submissions:", submissions.shape)
print(train.head())
print(submissions.head())



## === cell 2
h_target = 512
w_target = 512
batch_size = 32

N_WORKERS = max(2, (os.cpu_count() or 4) - 1)
USE_MULTIPROCESSING = True
MAX_QUEUE_SIZE = 16



## === cell 3
label_split = train["labels"].apply(lambda x: x.split())
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(label_split)
class_names = list(mlb.classes_)
num_classes = len(class_names)

print("num_classes:", num_classes)
print("classes:", class_names)

train_df = train.copy()
for i, c in enumerate(class_names):
    train_df[c] = y[:, i].astype(np.float32)

idx = np.arange(len(train_df))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
split = int(0.9 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
va_df = train_df.iloc[va_idx].reset_index(drop=True)

print("tr_df:", tr_df.shape, "va_df:", va_df.shape)



## === cell 4
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.10,
    horizontal_flip=True,
)
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_datagen.flow_from_dataframe(
    tr_df,
    directory=TRAIN_DIR,
    x_col="image",
    y_col=class_names,
    target_size=(h_target, w_target),
    color_mode="rgb",
    class_mode="raw",
    shuffle=True,
    seed=SEED,
    batch_size=batch_size,
)

valid_generator = valid_datagen.flow_from_dataframe(
    va_df,
    directory=TRAIN_DIR,
    x_col="image",
    y_col=class_names,
    target_size=(h_target, w_target),
    color_mode="rgb",
    class_mode="raw",
    shuffle=False,
    batch_size=batch_size,
)

test_generator = test_datagen.flow_from_dataframe(
    submissions,
    directory=TEST_DIR,
    x_col="image",
    y_col=None,
    target_size=(h_target, w_target),
    color_mode="rgb",
    class_mode=None,
    shuffle=False,
    batch_size=batch_size,
)

train_steps = int(np.ceil(train_generator.n / train_generator.batch_size))
valid_steps = int(np.ceil(valid_generator.n / valid_generator.batch_size))
test_steps = int(np.ceil(test_generator.n / test_generator.batch_size))

print("steps:", {"train": train_steps, "valid": valid_steps, "test": test_steps})



## === cell 5
base = tf.keras.applications.MobileNetV2(
    input_shape=(h_target, w_target, 3),
    include_top=False,
    weights="imagenet",
    pooling="avg",
)
x = base.output
outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)
model = tf.keras.Model(inputs=base.input, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
)

history = model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    validation_data=valid_generator,
    validation_steps=valid_steps,
    epochs=3,
    verbose=1,
    workers=N_WORKERS,
    use_multiprocessing=USE_MULTIPROCESSING,
    max_queue_size=MAX_QUEUE_SIZE,
)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2975931643.py in <cell line: 0>()
     16 # Enable parallel batch generation/augmentation to eliminate input bottleneck.
     17 # This preserves training loop/epochs and data exactly; only improves throughput.
---> 18 history = model.fit(
     19     train_generator,
     20     steps_per_epoch=train_steps,

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

## === cell 6
def _f1_binary_vec(y_true_01, y_pred_01, eps=1e-12):
    tp = np.sum((y_true_01 == 1) & (y_pred_01 == 1))
    fp = np.sum((y_true_01 == 0) & (y_pred_01 == 1))
    fn = np.sum((y_true_01 == 1) & (y_pred_01 == 0))
    return (2.0 * tp) / (2.0 * tp + fp + fn + eps)


y_val_true = va_df[class_names].values.astype(np.int32)

val_preds = model.predict(
    valid_generator,
    steps=valid_steps,
    verbose=1,
    workers=N_WORKERS,
    use_multiprocessing=USE_MULTIPROCESSING,
    max_queue_size=MAX_QUEUE_SIZE,
)
print("val_preds shape:", val_preds.shape, "y_val_true shape:", y_val_true.shape)

th_grid = np.round(np.linspace(0.10, 0.60, 11), 2).astype(np.float32)

best_th = np.full(num_classes, 0.30, dtype=np.float32)
best_f1 = np.full(num_classes, -1.0, dtype=np.float32)

pred_ge = val_preds[None, :, :] >= th_grid[:, None, None]

for j in range(num_classes):
    yj = y_val_true[:, j].astype(bool)
    if yj.sum() < 3:
        continue

    pj = pred_ge[:, :, j]
    tp = np.sum(pj & yj[None, :], axis=1).astype(np.float64)
    fp = np.sum(pj & (~yj[None, :]), axis=1).astype(np.float64)
    fn = np.sum((~pj) & yj[None, :], axis=1).astype(np.float64)

    f1s = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-12)
    k = int(np.argmax(f1s))
    best_f1[j] = f1s[k].astype(np.float32)
    best_th[j] = th_grid[k]

print("Per-class thresholds (first 10):", list(zip(class_names[:10], best_th[:10])))
print("Per-class val F1 (first 10):", list(zip(class_names[:10], best_f1[:10])))



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/887784868.py in <cell line: 0>()
     10 
     11 # Use multiprocessing workers for predict to speed up image decoding/preprocessing.
---> 12 val_preds = model.predict(
     13     valid_generator,
     14     steps=valid_steps,

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

## === cell 7
preds = model.predict(
    test_generator,
    steps=test_steps,
    verbose=1,
    workers=N_WORKERS,
    use_multiprocessing=USE_MULTIPROCESSING,
    max_queue_size=MAX_QUEUE_SIZE,
)
print("preds shape:", preds.shape)

pred_df = pd.DataFrame(preds, columns=class_names)
print(pred_df.head())



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3106913953.py in <cell line: 0>()
      1 # Use multiprocessing workers for test prediction as well.
----> 2 preds = model.predict(
      3     test_generator,
      4     steps=test_steps,
      5     verbose=1,

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

## === cell 8
best_th_arr = best_th.astype(np.float32)
preds_arr = pred_df.values.astype(np.float32)

mask = preds_arr >= best_th_arr[None, :]

row_counts = mask.sum(axis=1)
need_argmax = row_counts == 0
if np.any(need_argmax):
    argm = np.argmax(preds_arr[need_argmax], axis=1)
    mask[need_argmax, :] = False
    mask[need_argmax, argm] = True

if "healthy" in class_names:
    healthy_idx = class_names.index("healthy")
    has_healthy = mask[:, healthy_idx]
    has_other = mask.sum(axis=1) > 1
    drop_healthy = has_healthy & has_other
    if np.any(drop_healthy):
        mask[drop_healthy, healthy_idx] = False

out_labels = []
for i in range(mask.shape[0]):
    idxs = np.flatnonzero(mask[i])
    out_labels.append(" ".join([class_names[j] for j in idxs]))

submissions.loc[:, "labels"] = out_labels
print(submissions.head())



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3574728450.py in <cell line: 0>()
      1 # Vectorize submission label construction to reduce Python overhead on 3,7k+ rows.
      2 # This preserves the exact selection logic and tie-break behavior.
----> 3 best_th_arr = best_th.astype(np.float32)
      4 preds_arr = pred_df.values.astype(np.float32)
      5 

NameError: name 'best_th' is not defined

## === cell 9
submissions.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submissions.shape)
print(submissions.head(10).to_string(index=False))
