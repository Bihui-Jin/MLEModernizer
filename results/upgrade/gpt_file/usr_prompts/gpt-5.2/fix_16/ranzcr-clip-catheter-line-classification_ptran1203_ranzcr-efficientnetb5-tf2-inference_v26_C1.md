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
Detect the presence and position of catheters and lines on chest x-rays.

## Metric
Area under the ROC curve for each label, with the final score being the average of the individual AUCs of each predicted column.

## Submission Format
For each ID in the test set, you must predict a probability for all target variables. The file should contain a header and have the following format:
```
StudyInstanceUID,ETT - Abnormal,ETT - Borderline,ETT - Normal,NGT - Abnormal,NGT - Borderline,NGT - Incompletely Imaged,NGT - Normal,CVC - Abnormal,CVC - Borderline,CVC - Normal,Swan Ganz Catheter Present
1.2.826.0.1.3680043.8.498.62451881164053375557257228990443168843,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.83721761279899623084220697845011427274,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.12732270010839808189235995393981377825,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.11769539755086084996287023095028033598,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.87838627504097587943394933987052577153,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.53211840524738036417560823327351887819,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.93555795394184819372299157360228027866,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.52241894131170494723503100795076463919,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.36500167484503936720548852591033878284,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.86199852603457900780565655267977637728,0,0,0,0,0,0,0,0,0,0,0
```

## Dataset
`train.csv` contains image IDs, binary labels, and patient IDs.

TFRecords are available for both train and test.

We've also included `train_annotations.csv`. These are segmentation annotations for training samples that have them. They are included solely as additional information for competitors.

- train.csv - contains image IDs, binary labels, and patient IDs.
- sample_submission.csv - a sample submission file in the correct format
- test - test images
- train - training images

### Columns
- `StudyInstanceUID` - unique ID for each image
- `ETT - Abnormal` - endotracheal tube placement abnormal
- `ETT - Borderline` - endotracheal tube placement borderline abnormal
- `ETT - Normal` - endotracheal tube placement normal
- `NGT - Abnormal` - nasogastric tube placement abnormal
- `NGT - Borderline` - nasogastric tube placement borderline abnormal
- `NGT - Incompletely Imaged` - nasogastric tube placement inconclusive due to imaging
- `NGT - Normal` - nasogastric tube placement borderline normal
- `CVC - Abnormal` - central venous catheter placement abnormal
- `CVC - Borderline` - central venous catheter placement borderline abnormal
- `CVC - Normal` - central venous catheter placement normal
- `Swan Ganz Catheter Present`
- `PatientID` - unique ID for each patient in the dataset

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
        input/
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
        working/
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
```

-> data/ranzcr-clip-catheter-line-classification/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/ranzcr-clip-catheter-line-classification/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/ranzcr-clip-catheter-line-classification/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> data/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> (stopped after 10 files for performance)

# 5. Target score

0.9008278922660714

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.48226) has done: 'I fix the TensorFlow import/runtime crash caused by an incompatible `protobuf` version by force-downgrading protobuf to a TF-compatible version at runtime before importing TensorFlow. Then I fix the submission-time error by removing the unnecessary `.numpy()` call on the already-numpy output of `predict_on_batch`. Finally, I make the submission columns robust to the fact that your provided `sample_submission.csv` is missing some required target columns by constructing the full 11-label column set and writing a correctly formatted `submission.csv`.'
- What this solution (achieved 0.52972) has done: 'Your current score is far below the target, so the most likely issue is not the architecture but a data/weight mismatch: you are normalizing with ImageNet mean/std while EfficientNet models expect their own `preprocess_input`, and you may also be failing to actually load the intended competition weights (the path looks external and often missing), causing near-random predictions. I keep the model architecture and prediction loop intact, but switch preprocessing to the correct EfficientNet preprocessing for the chosen backbone (a small but meaningful correctness fix) and make the weight-loading failure explicit so you don’t unknowingly submit an untrained model. I also enforce deterministic ordering/alignment between `sample_submission` and predictions and add a safety check that the submission has the full 11 required targets, filling only truly missing columns with 0.5.'
- What this solution (achieved 0.45512) has done: 'Your score is far below the target, and the dominant issue is that you are almost certainly submitting near-untrained predictions because the external weights path doesn’t exist; the smallest legitimate way to move toward the target is to actually train the existing model briefly on the provided `train.csv` + images. I keep your model architecture (EfficientNet + Dropout + Dense sigmoid) and loss unchanged, and add a minimal patient-level train/validation split plus a short training phase before inference. I also switch your TF `AUTOTUNE` constant to the stable `tf.data.AUTOTUNE` and keep preprocessing consistent with EfficientNet. Finally, the submission writing stays the same schema enforcement, but now uses the trained weights to produce meaningfully better AUC.'

# 9. Code solution

## === cell 0
import os, sys
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob

from datetime import datetime as dt
from sklearn.model_selection import GroupShuffleSplit

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
tf.random.set_seed(42)
np.random.seed(42)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

W = H = 456
N_CLASSES = 11
autotune = tf.data.AUTOTUNE

target_cols = [
    "CVC - Abnormal",
    "CVC - Borderline",
    "CVC - Normal",
    "ETT - Abnormal",
    "ETT - Borderline",
    "ETT - Normal",
    "NGT - Abnormal",
    "NGT - Borderline",
    "NGT - Incompletely Imaged",
    "NGT - Normal",
    "Swan Ganz Catheter Present",
]

DATA_ROOT = "../input/ranzcr-clip-catheter-line-classification"
test_dir = os.path.join(DATA_ROOT, "test")
train_dir = os.path.join(DATA_ROOT, "train")
sample_path = os.path.join(DATA_ROOT, "sample_submission.csv")
train_csv_path = os.path.join(DATA_ROOT, "train.csv")

weight_dir = "../input/cassava2020weights"

model_map = {
    "efficientb3": [
        tf.keras.applications.EfficientNetB3,
        os.path.join(weight_dir, "ranzcr_efficientb3.h5"),
    ],
    "efficientb5": [
        tf.keras.applications.EfficientNetB5,
        os.path.join(weight_dir, "ranzcr_efficientb5.h5"),
    ],
    "efficientb7": [
        tf.keras.applications.EfficientNetB7,
        os.path.join(weight_dir, "ranzcr_efficientb7.h5"),
    ],
}

print("TF version:", tf.__version__)
print(
    "Train dir exists:",
    os.path.isdir(train_dir),
    "| Test dir exists:",
    os.path.isdir(test_dir),
    "| sample_submission exists:",
    os.path.isfile(sample_path),
    "| train.csv exists:",
    os.path.isfile(train_csv_path),
)

if os.path.isdir(test_dir):
    approx_n = len(glob(os.path.join(test_dir, "*.jpg")))
else:
    approx_n = 0
print("Num test images (jpg):", approx_n)

try:
    cpu_count = os.cpu_count() or 8
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(max(1, min(8, cpu_count // 2)))
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
if not os.path.isdir(test_dir):
    raise FileNotFoundError(f"Test image directory not found: {test_dir}")

test_files = np.array(sorted(glob(os.path.join(test_dir, "*.jpg"))), dtype=object)
if len(test_files) == 0:
    raise FileNotFoundError(f"No .jpg files found in {test_dir}")

test_uids = np.array([os.path.basename(p)[:-4] for p in test_files], dtype=object)


@tf.function(jit_compile=True)
def parse_and_preprocess_jpeg(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, (H, W), method="bilinear", antialias=False)
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img


BATCH_TEST = 256

test_data = tf.data.Dataset.from_tensor_slices(test_files)

options = tf.data.Options()
options.experimental_deterministic = False
options.experimental_optimization.apply_default_optimizations = True
options.experimental_optimization.map_parallelization = True
options.experimental_optimization.parallel_batch = True
options.experimental_optimization.autotune_buffers = True
options.experimental_optimization.autotune_cpu_budget = 0  # let TF choose
options.experimental_optimization.autotune_ram_budget = 0  # let TF choose
try:
    options.threading.private_threadpool_size = min(32, max(8, (os.cpu_count() or 8)))
    options.threading.max_intra_op_parallelism = 1
except Exception:
    pass
test_data = test_data.with_options(options)

test_data = test_data.map(parse_and_preprocess_jpeg, num_parallel_calls=autotune)
test_data = test_data.cache()
test_data = test_data.batch(BATCH_TEST, drop_remainder=False)
test_data = test_data.prefetch(autotune)

print("Built test_data batches:", int(np.ceil(len(test_files) / BATCH_TEST)))




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/627829716.py in <cell line: 0>()
     30 options.experimental_optimization.map_parallelization = True
     31 options.experimental_optimization.parallel_batch = True
---> 32 options.experimental_optimization.autotune_buffers = True
     33 options.experimental_optimization.autotune_cpu_budget = 0  # let TF choose
     34 options.experimental_optimization.autotune_ram_budget = 0  # let TF choose

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune_buffers on OptimizationOptions.

## === cell 2
def show_samples(dataset):
    rows = cols = 3
    fig = plt.figure(figsize=(12, 12))
    i = 0
    for img, uid in dataset.unbatch().shuffle(200).take(rows * cols):
        i += 1
        fig.add_subplot(rows, cols, i)

        x = (img + 1.0) * 127.5
        x = tf.clip_by_value(x, 0.0, 255.0)
        plt.imshow(tf.cast(x, tf.uint8).numpy())
        plt.title(uid.numpy().decode("utf-8")[:28], fontsize=8)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


print("Skipping sample visualization to meet 600s timeout.")




## === cell 3
def get_model(
    base_model,
    baseline_weight=None,
    init_weight=None,
    lr=0.001,
    optimizer=tf.optimizers.Adam,
):
    base_model = base_model(
        include_top=False, input_shape=(H, W, 3), pooling="avg", weights=baseline_weight
    )
    base_out = base_model.output
    out = tf.keras.layers.Dropout(0.3)(base_out)
    out = tf.keras.layers.Dense(N_CLASSES, activation="sigmoid")(out)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=out)

    model.compile(
        optimizer=optimizer(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC()],
    )

    if init_weight and os.path.isfile(init_weight):
        try:
            model.load_weights(init_weight)
            print(f"Weight loaded from {init_weight}")
        except Exception as e:
            print(f"Load weight from {init_weight} failed: {e}")
            print(
                "WARNING: Proceeding without loaded weights will likely score ~0.5 AUC."
            )
    else:
        print(f"Init weight not found, using default weights. Path={init_weight}")
        print("WARNING: Proceeding without loaded weights will likely score ~0.5 AUC.")
    return model


base_mode, weight_path = model_map["efficientb5"]
model = get_model(base_mode, init_weight=weight_path)




## === cell 4
did_load_external = os.path.isfile(weight_path)

if not did_load_external:
    if not os.path.isfile(train_csv_path):
        raise FileNotFoundError(f"train.csv not found at: {train_csv_path}")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train image directory not found: {train_dir}")

    train_df = pd.read_csv(train_csv_path)
    missing = [
        c
        for c in (["StudyInstanceUID", "PatientID"] + target_cols)
        if c not in train_df.columns
    ]
    if missing:
        raise ValueError(f"train.csv is missing required columns: {missing}")

    train_jpg_uids = np.array(
        [os.path.basename(p)[:-4] for p in glob(os.path.join(train_dir, "*.jpg"))],
        dtype=object,
    )
    train_jpg_set = set(train_jpg_uids.tolist())

    uid_series = train_df["StudyInstanceUID"].astype(str)
    mask = uid_series.isin(train_jpg_set)
    train_df = train_df.loc[mask].copy()
    train_df["path"] = (
        train_dir + "/" + train_df["StudyInstanceUID"].astype(str) + ".jpg"
    )
    train_df = train_df.reset_index(drop=True)
    print("Train rows with existing images:", len(train_df))

    X_paths = train_df["path"].values
    Y = train_df[target_cols].values.astype(np.float32)
    groups = train_df["PatientID"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    tr_idx, va_idx = next(gss.split(X_paths, Y, groups=groups))

    tr_paths, tr_y = X_paths[tr_idx], Y[tr_idx]
    va_paths, va_y = X_paths[va_idx], Y[va_idx]

    print("Train/Val sizes:", len(tr_paths), len(va_paths))

    @tf.function(jit_compile=True)
    def parse_jpeg_with_label(path, y):
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
        img = tf.image.resize(img, (H, W), method="bilinear", antialias=False)
        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return img, y

    BATCH = 16

    train_ds = tf.data.Dataset.from_tensor_slices((tr_paths, tr_y))
    train_ds = train_ds.shuffle(
        min(len(tr_paths), 2048), seed=42, reshuffle_each_iteration=True
    )
    train_ds = train_ds.map(parse_jpeg_with_label, num_parallel_calls=autotune)
    train_ds = train_ds.cache().batch(BATCH, drop_remainder=False).prefetch(autotune)

    val_ds = tf.data.Dataset.from_tensor_slices((va_paths, va_y))
    val_ds = val_ds.map(parse_jpeg_with_label, num_parallel_calls=autotune)
    val_ds = val_ds.cache().batch(BATCH, drop_remainder=False).prefetch(autotune)

    def _set_lr_safely(opt, lr_value: float):
        try:
            lr_obj = opt.learning_rate
        except Exception:
            lr_obj = None

        try:
            if hasattr(lr_obj, "assign"):
                lr_obj.assign(float(lr_value))
                return
        except Exception:
            pass

        try:
            opt.learning_rate = float(lr_value)
            return
        except Exception:
            pass

        try:
            tf.keras.backend.set_value(opt.learning_rate, float(lr_value))
            return
        except Exception as e:
            raise RuntimeError(f"Failed to set learning rate safely: {e}") from e

    _set_lr_safely(model.optimizer, 1e-4)

    EPOCHS = 2  # keep unchanged
    print(f"Training for {EPOCHS} epochs because external weights were not found...")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)
else:
    if not os.path.isfile(sample_path):
        raise FileNotFoundError(f"sample_submission.csv not found at: {sample_path}")
    print(
        "External weights exist; skipping training to preserve expected scoring behavior."
    )

sub = pd.read_csv(sample_path)
id_col = "StudyInstanceUID"
required_out_cols = [id_col] + target_cols  # enforce full set and stable order

steps = int(np.ceil(len(test_files) / BATCH_TEST))
preds = model.predict(test_data, verbose=0, steps=steps)

pred_df = pd.DataFrame(preds, columns=target_cols)
pred_df[id_col] = test_uids

pred_df_indexed = pred_df.set_index(id_col)
out = sub[[id_col]].copy()
out = out.join(pred_df_indexed, on=id_col)

for c in required_out_cols:
    if c == id_col:
        continue
    if c not in out.columns:
        out[c] = 0.5

out = out[required_out_cols]
prob_cols = [c for c in required_out_cols if c != id_col]
out[prob_cols] = out[prob_cols].astype(np.float32).fillna(0.5).clip(0.0, 1.0)

out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", out.shape)
print("Columns:", list(out.columns))
print(out.head())
print("Any NaNs in probs:", bool(out[prob_cols].isna().any().any()))
print(
    "Min/Max prob:",
    float(out[prob_cols].min().min()),
    float(out[prob_cols].max().max()),
)

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_55/1464092093.py in <cell line: 0>()
     99     EPOCHS = 2  # keep unchanged
    100     print(f"Training for {EPOCHS} epochs because external weights were not found...")
--> 101     model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)
    102 else:
    103     if not os.path.isfile(sample_path):

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     57       e.message += " name: " + name
     58     raise core._status_to_exception(e) from None
---> 59   except TypeError as e:
     60     keras_symbolic_tensors = [x for x in inputs if _is_keras_symbolic_tensor(x)]
     61     if keras_symbolic_tensors:

InvalidArgumentError: Graph execution error:

Detected at node path defined at (most recent call last):
<stack traces unavailable>
Detected at node path defined at (most recent call last):
<stack traces unavailable>
Detected at node path defined at (most recent call last):
<stack traces unavailable>
Detected at node path defined at (most recent call last):
<stack traces unavailable>
2 root error(s) found.
  (0) INVALID_ARGUMENT:  Error in user-defined function passed to ParallelMapDatasetV2:3 transformation with iterator: Iterator::Root::Prefetch::BatchV2::MemoryCacheImpl::ParallelMapV2: Detected unsupported operations when trying to compile graph __inference_parse_jpeg_with_label_6059[] on XLA_CPU_JIT: _Arg (No registered '_Arg' OpKernel for XLA_CPU_JIT devices compatible with node {{node path}}
	 (OpKernel was found, but attributes didn't match) Requested Attributes: T=DT_STRING, _output_shapes=[[]], _user_specified_name="path", index=0){{node path}}
The op is created at: 
dummy_file_name:10:dummy_function_name
	tf2xla conversion failed while converting __inference_parse_jpeg_with_label_6059[]. Run with TF_DUMP_GRAPH_PREFIX=/path/to/dump/dir and --vmodule=xla_compiler=2 to obtain a dump of the compiled functions.
	 [[PartitionedCall]]
	 [[IteratorGetNext]]
	 [[IteratorGetNext/_4]]
  (1) INVALID_ARGUMENT:  Error in user-defined function passed to ParallelMapDatasetV2:3 transformation with iterator: Iterator::Root::Prefetch::BatchV2::MemoryCacheImpl::ParallelMapV2: Detected unsupported operations when trying to compile graph __inference_parse_jpeg_with_label_6059[] on XLA_CPU_JIT: _Arg (No registered '_Arg' OpKernel for XLA_CPU_JIT devices compatible with node {{node path}}
	 (OpKernel was found, but attributes didn't match) Requested Attributes: T=DT_STRING, _output_shapes=[[]], _user_specified_name="path", index=0){{node path}}
The op is created at: 
dummy_file_name:10:dummy_function_name
	tf2xla conversion failed while converting __inference_parse_jpeg_with_label_6059[]. Run with TF_DUMP_GRAPH_PREFIX=/path/to/dump/dir and --vmodule=xla_compiler=2 to obtain a dump of the compiled functions.
	 [[PartitionedCall]]
	 [[IteratorGetNext]]
0 successful operations.
0 derived errors ignored. [Op:__inference_multi_step_on_iterator_117599]
