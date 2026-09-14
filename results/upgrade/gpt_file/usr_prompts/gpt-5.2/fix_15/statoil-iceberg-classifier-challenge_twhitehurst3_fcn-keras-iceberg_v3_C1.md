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
Predict whether an image contains a ship or an iceberg.

## Metric
Log loss.

## Submission Format
For each id in the test set, you must predict the probability that the image contains an iceberg (a number between 0 and 1). The file should contain a header and have the following format:

```
id,is_iceberg
809385f7,0.5
7535f0cd,0.4
3aa99a38,0.9
etc.
```

## Dataset
The labels are provided by human experts and geographic knowledge on the target. All the images are 75x75 images with two bands.

The data (`train.json`, `test.json`) is presented in `json` format.

The files consist of a list of images, and for each image, you can find the following fields:

- **id** - the id of the image
- **band_1, band_2** - the [flattened](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.flatten.html) image data. Each band has 75x75 pixel values in the list, so the list has 5625 elements. Note that these values are not the normal non-negative integers in image files since they have physical meanings - these are **float** numbers with unit being [dB](https://en.wikipedia.org/wiki/Decibel). Band 1 and Band 2 are signals characterized by radar backscatter produced from different polarizations at a particular incidence angle. The polarizations correspond to HH (transmit/receive horizontally) and HV (transmit horizontally and receive vertically).
- **inc_angle** - the incidence angle of which the image was taken. Note that this field has missing data marked as "na", and those images with "na" incidence angles are all in the training data to prevent leakage.
- **is_iceberg** - the target variable, set to 1 if it is an iceberg, and 0 if it is a ship. This field only exists in `train.json`.

Please note that we have included machine-generated images in the test set to prevent hand labeling. They are excluded in scoring.

sample_submission.csv: The submission file in the correct format:

# 2. Python version

3.7

# 3. Installed packages



# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (99 lines)
            sample_submission.csv (322 lines)
            sample_submission.csv.7z (2.0 kB)
            sample_submission.csv.zip (2.0 kB)
            test.json (1 lines)
            test.json.7z (8.9 MB)
            train.json (1 lines)
            train.json.7z (35.8 MB)
            statoil-iceberg-classifier-challenge/
                description.md (99 lines)
                sample_submission.csv (322 lines)
                ... and 6 other files
                statoil-iceberg-classifier-challenge/
        input/
            description.md (99 lines)
            sample_submission.csv (322 lines)
            sample_submission.csv.7z (2.0 kB)
            sample_submission.csv.zip (2.0 kB)
            test.json (1 lines)
            test.json.7z (8.9 MB)
            train.json (1 lines)
            train.json.7z (35.8 MB)
            statoil-iceberg-classifier-challenge/
                description.md (99 lines)
                sample_submission.csv (322 lines)
                ... and 6 other files
                statoil-iceberg-classifier-challenge/
        working/
            statoil-iceberg-classifier-challenge/
                description.md (99 lines)
                sample_submission.csv (322 lines)
                ... and 6 other files
                statoil-iceberg-classifier-challenge/
```

-> data/sample_submission.csv has 321 rows and 2 columns.
The columns are: id, is_iceberg

-> data/statoil-iceberg-classifier-challenge/sample_submission.csv has 321 rows and 2 columns.
The columns are: id, is_iceberg

-> data/statoil-iceberg-classifier-challenge/test.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "id": {
        "type": "string"
      },
      "band_1": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "band_2": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "inc_angle": {
        "type": [
          "number",
          "string"
        ]
      }
    },
    "required": [
      "band_1",
      "band_2",
      "id",
      "inc_angle"
    ]
  }
}

-> data/statoil-iceberg-classifier-challenge/train.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "id": {
        "type": "string"
      },
      "band_1": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "band_2": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "inc_angle": {
        "type": [
          "number",
          "string"
        ]
      },
      "is_iceberg": {
        "type": "integer"
      }
    },
    "required": [
      "band_1",
      "band_2",
      "id",
      "inc_angle",
      "is_iceberg"
    ]
  }
}

-> data/test.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "id": {
        "type": "string"
      },
      "band_1": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "band_2": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "inc_angle": {
        "type": [
          "number",
          "string"
        ]
      }
    },
    "required": [
      "band_1",
      "band_2",
      "id",
      "inc_angle"
    ]
  }
}

-> data/train.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "id": {
        "type": "string"
      },
      "band_1": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "band_2": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "inc_angle": {
        "type": [
          "number",
          "string"
        ]
      },
      "is_iceberg": {
        "type": "integer"
      }
    },
    "required": [
      "band_1",
      "band_2",
      "id",
      "inc_angle",
      "is_iceberg"
    ]
  }
}

-> input/sample_submission.csv has 321 rows and 2 columns.
The columns are: id, is_iceberg

-> (stopped after 10 files for performance)

# 5. Target score

0.3247

# 6. Current score

0.69324

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.69334) has done: 'We fix the environment/runtime crash that happens on `import tensorflow` by forcing the pure-Python protobuf implementation before TensorFlow loads. Then we fix the model save/load failure caused by the `Lambda(lambda x: x, ...)` layer under newer Keras safe deserialization by replacing it with an equivalent safe layer (`InputLayer`) while keeping the same architecture semantics. Finally, we ensure the best model is available for inference (fallback to the in-memory trained model if needed) and always write a valid `submission.csv` with the required `id,is_iceberg` columns.'
- What this solution (achieved 0.43616) has done: 'The immediate blocker is the TensorFlow import crash caused by an incompatible protobuf runtime; fixing that cleanly requires forcing a compatible protobuf version before importing TensorFlow (the current env var workaround isn’t sufficient in this Kaggle image). After it runs, the 0.693 logloss indicates the model is effectively untrained/mis-calibrated; the smallest legitimate improvement (without changing architecture or training loop) is to apply the standard per-image normalization used for this competition (zero-mean/unit-std per band) and to correctly use `inc_angle` as an additional input feature by folding it into the existing 3rd channel (keeping input shape and architecture unchanged). Finally, we keep the same submission format but add safety checks to ensure predictions align with test ids and always write a valid `submission.csv`.'
- What this solution (achieved 0.69316) has done: 'We fix the TensorFlow import crash by pinning protobuf to the pure-Python implementation *and* patching the specific `MessageFactory.GetPrototype` incompatibility that triggers the error in this Kaggle image. This is a runtime-only compatibility shim and does not change your model/training logic. Then we keep everything else the same so training/inference run end-to-end and a valid `submission.csv` is always written. With TF loading correctly again, your existing preprocessing (per-image standardization + incidence-angle folding) and training should execute and is expected to improve logloss versus the current broken run.'
- What this solution (achieved 0.45327) has done: 'I fix the protobuf shim that currently throws an AttributeError by safely patching both the class and instance method cases (and making it a no-op when not needed), so TensorFlow can import reliably. I also add a small, score-neutral safety fallback to load the best checkpoint with `compile=False` if Keras deserialization/compile metadata causes issues, while keeping the same model/training logic. Finally, I keep all paths and submission formatting the same, ensuring `submission.csv` is always written with `id,is_iceberg` aligned to the test ids.'
- What this solution (achieved 0.46294) has done: 'Your current score (0.45327 logloss; lower is better) is still far from the target 0.3247, so we should make a small, legitimate improvement without changing the model or training loop. The most likely issue is a train/test preprocessing mismatch: you compute `inc_median` on the full training set before the split, which leaks validation information and can hurt generalization; we compute the incidence-angle fill/scaling stats on the training split only and apply them consistently to val/test. We also keep the same per-image standardization and the same “inc_angle folded into 3rd channel” idea, but fit the scaling (median) only on `x_train` ids to avoid leakage. Finally, we keep the same submission writing, but ensure the ids align and the output is always valid.'
- What this solution (achieved 0.69311) has done: 'We need to move logloss down from 0.46294 toward 0.3247 (lower is better), so the smallest safe improvement is to fix preprocessing consistency rather than changing the model/training loop. Right now each band is standardized per-image, but the 3rd channel mixes a *non-standardized* incidence-angle offset into a standardized image channel, which can distort scale; we keep the exact “avg_band + inc_map” idea but standardize the third channel per-image as well (same semantics, just consistent scaling). To reduce overfitting without changing architecture or training approach, we also add a tiny amount of label smoothing in the existing binary cross-entropy loss (still logloss-aligned) and ensure determinism stays the same. Everything else (data loading, split, model, fit call, callbacks, submission writing) stays intact and still produces `submission.csv`.'
- What this solution (achieved 0.69326) has done: 'Your logloss (0.69311) is essentially random, so the smallest change likely to move it down toward 0.3247 is to fix a train/test preprocessing mismatch in how the third channel is constructed: you currently standardize `band_1`, `band_2`, and the mixed third channel separately, but the third channel also contains an incidence-angle constant that gets “washed out” by per-image standardization. I keep your exact “avg_band + inc_map” core idea and input shape, but change only the incidence-angle injection to be scale-compatible by adding it as a small offset after the third-channel standardization (so the model can still “see” inc_angle without destabilizing the pixel scale). I also remove label smoothing (it can hurt logloss calibration here) while keeping the same BinaryCrossentropy and training loop/architecture. Everything else (TF/protobuf shim, split, callbacks, saving/loading, and submission writing) stays the same and still produces a valid `submission.csv`.'
- What this solution (achieved 0.69316) has done: 'Your current logloss (0.69326) is essentially random, so we should make a minimal change that most plausibly fixes a train/test mismatch without touching the model or training loop. The most likely culprit is the third-channel construction: right now you per-image standardize the third channel and then add an incidence-angle constant, but the scaling of that injected signal is arbitrary and can destabilize the learned distribution. I keep the exact same “avg_band + inc_map” idea and input shape, but scale the injected incidence-angle by the third channel’s per-image standard deviation so it becomes a small, consistent perturbation across images (instead of an uncalibrated offset). This preserves your core preprocessing semantics while making the `inc_angle` feature usable and should move logloss down toward the target band.'
- What this solution (achieved 0.40003) has done: 'Your current logloss (0.69316) is close to random, so the smallest likely-to-help change (without touching your model or training loop) is to fix the scale/semantics of how `inc_angle` is injected into the 3rd channel. Right now the injected term is multiplied by the per-image third-channel std, which makes the incidence signal *larger* exactly when the image is more variable, destabilizing distributions and often leading to poor generalization. I keep the exact “avg_band + inc_map” core idea and 3-channel input shape, but instead inject a bounded, dimensionless offset after standardizing the third channel (so it remains a small, consistent perturbation). Everything else (TF/protobuf shim, split, callbacks, compile/fit, and submission writing) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 0.69312) has done: 'We need to move logloss down from 0.40003 toward 0.3247 (lower is better), so I keep your model and training loop intact and focus on a minimal, metric-aligned preprocessing fix. The smallest likely gain is improving the `inc_angle` injection so it is on a stable, consistent scale across images: instead of a fixed `0.15` strength, we normalize incidence using the training-split mean/std and inject a small scaled offset (still into the 3rd channel after standardizing it), which typically improves calibration and generalization without changing the architecture. I also add a tiny epsilon clip during prediction (logloss-safe) to avoid extreme probabilities hurting logloss on unseen data. Everything else (data paths, split, model, callbacks, saving/loading, and submission writing) remains the same and still produces a valid `submission.csv`.'
- What this solution (achieved 0.39541) has done: 'To move logloss down from ~0.693 toward the 0.3247 target (lower is better) without changing your model/training loop, the most likely issue is that your Conv2D stack is currently operating on shrinking feature maps because you never set `padding="same"`, which can collapse spatial information too quickly and yield near-random predictions. I make the minimal architecture-equivalent fix of adding `padding="same"` to the existing `SeparableConv2D` layers (same blocks/filters/activations/loop, just padding), which is a standard correction for this competition and typically produces a large, legitimate gain. I also switch `EarlyStopping(mode="min")` explicitly (it currently uses `mode="auto"`) to ensure it reliably tracks val_loss direction, and keep everything else identical (preprocessing, optimizer, loss, callbacks, submission format). This should improve validation loss and public LB logloss while remaining within your constraints and still writing a valid `submission.csv`.'
- What this solution (achieved 0.48818) has done: 'Your current logloss (0.39541) is still above the 0.3247 target (lower is better), so we should make a small, low-risk generalization improvement without changing the model architecture or training loop. The most likely remaining issue is that the per-image standardization makes the injected `inc_angle` signal too weak/noisy; we keep the exact same “inject into 3rd channel after standardization” idea, but tune only the injection strength slightly upward based on typical scales (a calibration change, not a modeling change). We also make training more stable (and often slightly better calibrated for logloss) by ensuring `ReduceLROnPlateau` uses `mode="min"` (it currently uses `"auto"`, which can be inconsistent) and by evaluating/predicting with the same loaded best checkpoint. Everything else (data loading, split, preprocessing structure, model layers, loss, optimizer, epochs, callbacks, submission format) stays the same and still writes a valid `submission.csv`.'
- What this solution (achieved 0.69324) has done: 'We need to move logloss down from 0.48818 toward 0.3247 (lower is better), so I keep your exact model and training loop intact and focus on one minimal, high-impact generalization fix: ensure the training/validation split is grouped by `inc_angle` availability. In this competition, all `"na"` incidence angles are in the training set; mixing those into validation makes the validation distribution mismatched to test (which has valid angles), and the model can learn artifacts that hurt LB. By filtering out `"na"` rows before splitting (still training on them afterward, but validating only on “test-like” examples), we improve model selection via `ModelCheckpoint` without changing architecture, loss, or optimization. Everything else (preprocessing, callbacks, prediction clipping, submission format) stays the same and still produces a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import math
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import google.protobuf  # noqa: F401
from google.protobuf import message_factory as _message_factory


def _ensure_getprototype():
    MF = _message_factory.MessageFactory

    def _GetPrototype(self, descriptor):
        if hasattr(self, "GetMessageClass"):
            return self.GetMessageClass(descriptor)
        if hasattr(_message_factory, "GetMessageClass"):
            return _message_factory.GetMessageClass(descriptor)
        raise AttributeError("No GetMessageClass available to emulate GetPrototype")

    if not hasattr(MF, "GetPrototype"):
        setattr(MF, "GetPrototype", _GetPrototype)

    try:
        inst = MF()
        if not hasattr(inst, "GetPrototype"):
            setattr(inst, "GetPrototype", _GetPrototype.__get__(inst, MF))
    except Exception:
        pass


_ensure_getprototype()

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, InputLayer
from tensorflow.keras.layers import SeparableConv2D, BatchNormalization, MaxPooling2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    TensorBoard,
    CSVLogger,
    ReduceLROnPlateau,
)

np.random.seed(1)
tf.random.set_seed(1)

print("TensorFlow version:", tf.__version__)




## === cell 1
def show_final_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].set_title("loss")
    ax[0].plot(history.epoch, history.history.get("loss", []), label="Train loss")
    ax[0].plot(
        history.epoch, history.history.get("val_loss", []), label="Validation loss"
    )
    ax[1].set_title("acc")
    ax[1].plot(
        history.epoch,
        history.history.get("accuracy", history.history.get("acc", [])),
        label="Train acc",
    )
    ax[1].plot(
        history.epoch,
        history.history.get("val_accuracy", history.history.get("val_acc", [])),
        label="Validation acc",
    )
    ax[0].legend()
    ax[1].legend()
    plt.show()




## === cell 2
train_path = "/kaggle/input/train.json"
test_path = "/kaggle/input/test.json"
if not os.path.exists(train_path) or not os.path.exists(test_path):
    train_path = "/kaggle/input/statoil-iceberg-classifier-challenge/train.json"
    test_path = "/kaggle/input/statoil-iceberg-classifier-challenge/test.json"

train = pd.read_json(train_path)
test = pd.read_json(test_path)


def _to_band_array(series):
    return np.array(
        [np.array(band, dtype=np.float32).reshape(75, 75) for band in series]
    )


def _per_image_standardize(x):
    mean = x.mean(axis=(1, 2), keepdims=True)
    std = x.std(axis=(1, 2), keepdims=True)
    return (x - mean) / (std + 1e-6)


def _clean_inc_angle(s):
    s = s.replace("na", np.nan)
    ang = pd.to_numeric(s, errors="coerce").astype(np.float32)
    return ang.values


def _build_X_from_parts(
    b1,
    b2,
    inc_angle_raw,
    inc_median_value,
    inc_mean_value,
    inc_std_value,
):
    """
    Keep core preprocessing idea identical:
      - per-image standardize band_1 and band_2
      - 3rd channel uses avg of bands plus an incidence-angle derived constant map
      - inject incidence-angle AFTER third-channel standardization
    """
    b1 = _per_image_standardize(b1)
    b2 = _per_image_standardize(b2)

    inc = np.where(np.isfinite(inc_angle_raw), inc_angle_raw, inc_median_value).astype(
        np.float32
    )

    inc_norm = (inc - inc_mean_value) / (inc_std_value + 1e-6)
    inc_norm = np.clip(inc_norm, -3.0, 3.0).astype(np.float32)
    inc_map = inc_norm.reshape(-1, 1, 1)

    avg_band = (b1 + b2) / 2.0

    third_mean = avg_band.mean(axis=(1, 2), keepdims=True)
    third_std = avg_band.std(axis=(1, 2), keepdims=True)
    third_channel = (avg_band - third_mean) / (third_std + 1e-6)

    inj_strength = np.float32(0.12)
    third_channel = (third_channel + inj_strength * inc_map).astype(np.float32)

    X = np.concatenate(
        [
            b1[:, :, :, np.newaxis],
            b2[:, :, :, np.newaxis],
            third_channel[:, :, :, np.newaxis],
        ],
        axis=-1,
    ).astype(np.float32)
    return X


X_band_1_all = _to_band_array(train["band_1"])
X_band_2_all = _to_band_array(train["band_2"])
inc_all = _clean_inc_angle(train["inc_angle"])
target_train = train["is_iceberg"].astype(np.float32).values

idx_all = np.arange(len(train), dtype=np.int64)

finite_mask = np.isfinite(inc_all)
idx_finite = idx_all[finite_mask]
y_finite = target_train[finite_mask]

idx_train_finite, idx_val, y_train_finite, y_val = train_test_split(
    idx_finite, y_finite, random_state=1, train_size=0.80, stratify=y_finite
)

idx_train = np.setdiff1d(idx_all, idx_val, assume_unique=False)
y_train = target_train[idx_train]

inc_train_raw = inc_all[idx_train]
inc_median = np.nanmedian(inc_train_raw).astype(np.float32)
inc_train_filled = np.where(
    np.isfinite(inc_train_raw), inc_train_raw, inc_median
).astype(np.float32)
inc_mean = np.mean(inc_train_filled).astype(np.float32)
inc_std = np.std(inc_train_filled).astype(np.float32)

x_train = _build_X_from_parts(
    X_band_1_all[idx_train],
    X_band_2_all[idx_train],
    inc_all[idx_train],
    inc_median,
    inc_mean,
    inc_std,
)
x_val = _build_X_from_parts(
    X_band_1_all[idx_val],
    X_band_2_all[idx_val],
    inc_all[idx_val],
    inc_median,
    inc_mean,
    inc_std,
)

print("Train/Val shapes:", x_train.shape, x_val.shape, y_train.shape, y_val.shape)
print(
    "Val inc_angle finite ratio:",
    float(np.mean(np.isfinite(inc_all[idx_val]))),
)
print(
    "Inc angle median/mean/std (train split):",
    float(inc_median),
    float(inc_mean),
    float(inc_std),
)




## === cell 3
def ConvBlock(model, layers, filters):
    for _ in range(layers):
        model.add(SeparableConv2D(filters, (3, 3), activation="relu", padding="same"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))


def FCN():
    model = Sequential()
    model.add(InputLayer(input_shape=(75, 75, 3)))
    ConvBlock(model, 1, 64)
    ConvBlock(model, 1, 128)
    ConvBlock(model, 1, 128)
    ConvBlock(model, 1, 64)
    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))
    return model


model = FCN()
model.summary()



## === cell 4
best_model_weights = "./base.model.keras"  # standard keras format filename
checkpoint = ModelCheckpoint(
    best_model_weights,
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    mode="min",
    save_weights_only=False,
)

earlystop = EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,
    patience=10,
    verbose=1,
    mode="min",
)

tensorboard = TensorBoard(
    log_dir="./logs",
    histogram_freq=0,
    write_graph=True,
    write_images=False,
)

csvlogger = CSVLogger(
    filename="training_csv.log",
    separator=",",
    append=False,
)

reduce = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=40,
    verbose=1,
    mode="min",
    cooldown=1,
)

callbacks = [checkpoint, tensorboard, csvlogger, reduce, earlystop]



## === cell 5
opt = SGD(learning_rate=1e-4, momentum=0.95)
opt1 = Adam(learning_rate=2e-4)

loss_fn = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.0)

model.compile(
    loss=loss_fn,
    optimizer=opt1,
    metrics=["accuracy"],
)

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    batch_size=32,
    verbose=1,
    epochs=200,
    callbacks=callbacks,
)



## === cell 6
show_final_history(history)



## === cell 7
if os.path.exists(best_model_weights):
    try:
        best_model = tf.keras.models.load_model(best_model_weights)
    except Exception as e:
        print("load_model failed, retrying with compile=False. Error:", repr(e))
        best_model = tf.keras.models.load_model(best_model_weights, compile=False)
else:
    best_model = model

model_score = best_model.evaluate(x_val, y_val, verbose=1)
print("Model Val Loss:", model_score[0])
print("Model Val Accuracy:", model_score[1])



## === cell 8
band1_test = _to_band_array(test["band_1"])
band2_test = _to_band_array(test["band_2"])
inc_test_raw = _clean_inc_angle(test["inc_angle"])

x_test = _build_X_from_parts(
    band1_test,
    band2_test,
    inc_test_raw,
    inc_median,
    inc_mean,
    inc_std,
)

predictions = best_model.predict(x_test, batch_size=32, verbose=1)

preds = predictions.reshape((-1,)).astype(np.float64)
preds = np.clip(preds, 1e-6, 1.0 - 1e-6)

test_ids = test["id"].values
if len(test_ids) != len(preds):
    raise RuntimeError(
        f"Prediction length mismatch: ids={len(test_ids)} preds={len(preds)}"
    )

submission = pd.DataFrame({"id": test_ids, "is_iceberg": preds})

out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote submission:", out_path)
print(submission.head())
print("Submission shape:", submission.shape)
