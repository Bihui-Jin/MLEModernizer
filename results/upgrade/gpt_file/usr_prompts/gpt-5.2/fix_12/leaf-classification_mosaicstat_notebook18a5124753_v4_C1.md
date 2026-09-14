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
Use binary leaf images and extracted features to identify the species of plant.

## Metric
Multi-class log loss. 

The submitted probabilities for a given device are not required to sum to one because they are rescaled prior to being scored (each row is divided by the row sum), but they need to be in the range of [0, 1]. In order to avoid the extremes of the log function, predicted probabilities are replaced with \\(max(min(p,1-10^{-15}),10^{-15})\\).

## Submission Format
You must submit a csv file with the image id, all candidate species names, and a probability for each species. The order of the rows does not matter. The file must have a header and should look like the following:

id,Acer_Capillipes,Acer_Circinatum,Acer_Mono,...
2,0.1,0.5,0,0.2,...
5,0,0.3,0,0.4,...
6,0,0,0,0.7,...
etc.

## Dataset
The dataset consists of images of leaf specimens which have been converted to binary black leaves against white backgrounds. 

Three sets of features are also provided per image: a shape contiguous descriptor, an interior texture histogram, and a ﬁne-scale margin histogram. 

For each feature, a 64-attribute vector is given per leaf sample.

### File descriptions
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format
- **images/** - the image files (each image is named with its corresponding id)

### Data fields
- **id** - an anonymous id unique to an image
- **margin_1, margin_2, margin_3, ..., margin_64** - each of the 64 attribute vectors for the margin feature
- **shape_1, shape_2, shape_3, ..., shape_64** - each of the 64 attribute vectors for the shape feature
- **texture_1, texture_2, texture_3, ..., texture_64** - each of the 64 attribute vectors for the texture feature

# 2. Python version

3.13

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (70 lines)
            images.zip (22.0 MB)
            sample_submission.csv (100 lines)
            sample_submission.csv.zip (2.3 kB)
            test.csv (100 lines)
            test.csv.zip (39.3 kB)
            train.csv (892 lines)
            train.csv.zip (357.1 kB)
            images/
                42.jpg (32.6 kB)
                168.jpg (16.5 kB)
                ... and 988 other files
            leaf-classification/
                description.md (70 lines)
                images.zip (22.0 MB)
                ... and 6 other files
                images/
                    42.jpg (32.6 kB)
                    168.jpg (16.5 kB)
                    ... and 988 other files
                leaf-classification/
        input/
            description.md (70 lines)
            images.zip (22.0 MB)
            sample_submission.csv (100 lines)
            sample_submission.csv.zip (2.3 kB)
            test.csv (100 lines)
            test.csv.zip (39.3 kB)
            train.csv (892 lines)
            train.csv.zip (357.1 kB)
            images/
                42.jpg (32.6 kB)
                168.jpg (16.5 kB)
                ... and 988 other files
            leaf-classification/
                description.md (70 lines)
                images.zip (22.0 MB)
                ... and 6 other files
                images/
                    42.jpg (32.6 kB)
                    168.jpg (16.5 kB)
                    ... and 988 other files
                leaf-classification/
        working/
            leaf-classification/
                description.md (70 lines)
                images.zip (22.0 MB)
                ... and 6 other files
                images/
                    42.jpg (32.6 kB)
                    168.jpg (16.5 kB)
                    ... and 988 other files
                leaf-classification/
```

-> data/leaf-classification/sample_submission.csv has 99 rows and 100 columns.
The columns are: id, Acer_Capillipes, Acer_Circinatum, Acer_Mono, Acer_Opalus, Acer_Palmatum, Acer_Pictum, Acer_Platanoids, Acer_Rubrum, Acer_Rufinerve, Acer_Saccharinum, Alnus_Cordata, Alnus_Maximowiczii, Alnus_Rubra, Alnus_Sieboldiana... and 85 more columns

-> data/leaf-classification/test.csv has 99 rows and 193 columns.
The columns are: id, margin1, margin2, margin3, margin4, margin5, margin6, margin7, margin8, margin9, margin10, margin11, margin12, margin13, margin14... and 178 more columns

-> data/leaf-classification/train.csv has 891 rows and 194 columns.
The columns are: id, species, margin1, margin2, margin3, margin4, margin5, margin6, margin7, margin8, margin9, margin10, margin11, margin12, margin13... and 179 more columns

-> data/sample_submission.csv has 99 rows and 100 columns.
The columns are: id, Acer_Capillipes, Acer_Circinatum, Acer_Mono, Acer_Opalus, Acer_Palmatum, Acer_Pictum, Acer_Platanoids, Acer_Rubrum, Acer_Rufinerve, Acer_Saccharinum, Alnus_Cordata, Alnus_Maximowiczii, Alnus_Rubra, Alnus_Sieboldiana... and 85 more columns

-> data/test.csv has 99 rows and 193 columns.
The columns are: id, margin1, margin2, margin3, margin4, margin5, margin6, margin7, margin8, margin9, margin10, margin11, margin12, margin13, margin14... and 178 more columns

-> data/train.csv has 891 rows and 194 columns.
The columns are: id, species, margin1, margin2, margin3, margin4, margin5, margin6, margin7, margin8, margin9, margin10, margin11, margin12, margin13... and 179 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.3703026208665841

# 6. Current score

0.19594

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.22871) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf implementation (this avoids the `MessageFactory.GetPrototype` error in some Kaggle TF/protobuf combinations) and by making the TF strategy creation robust. Then I fix the train/validation split bug by ensuring the validation size is at least the number of classes when using stratification, which removes the `test_size should be >= n_classes` runtime error without changing the model logic. Finally, I ensure the submission columns align exactly to `sample_submission.csv` (correct class column order) and that a valid `.csv` is always written to `/kaggle/working/`.'
- What this solution (achieved 0.25396) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by forcing the Python protobuf implementation *before* any TensorFlow import and by avoiding importing TensorFlow in multiple cells (which can re-trigger the issue in some Kaggle images). I also remove duplicated/redundant TensorFlow initialization so the runtime state is consistent, while keeping the same model architectures/training semantics and ensuring the pipeline runs end-to-end. Since your current score (0.22871) is already much better than the target (0.3703, lower-is-better), I not make any score-improving changes; the goal is stability/correctness and producing a valid submission CSV with the correct column order. The final script writes `/kaggle/working/submission_deep.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.1773) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before any TensorFlow-related import happens*, and by also forcing TensorFlow to use the legacy Keras implementation to avoid a known TF/Keras/protobuf incompatibility in some Kaggle images. I keep your model/training logic intact, but remove the built-in EarlyStopping usage (which changes training semantics and score) and ensure the script trains for the full configured EPOCHS as originally defined. I also eliminate the duplicate EDA cell (cell 5) that calls `plt.show()` (unnecessary in Kaggle batch runs) to reduce runtime risk, while keeping outputs saved. Finally, I keep the submission column alignment exactly matching `sample_submission.csv` and always write a valid `.csv` to `/kaggle/working/submission_deep.csv`.'
- What this solution (achieved 0.19337) has done: 'I fix the TensorFlow/protobuf crash that prevents the notebook from running by ensuring the protobuf runtime is pinned to the pure-Python implementation and by applying a safe monkey-patch for the missing `MessageFactory.GetPrototype` API before importing TensorFlow. I also make the Kaggle data-path resolution robust by falling back to `/kaggle/input/leaf-classification` if the initial `/kaggle/input/leaf-classification` path isn’t present as expected, without changing any modeling/training logic. Since your current score (0.1773, lower-is-better) is already far better than the target (0.3703), I not make any score-improving changes; the goal is to restore end-to-end execution and a valid submission CSV with correct column order. The rest of the code (models, CV, final training, and submission creation) is kept intact.'
- What this solution (achieved 0.19856) has done: 'I fix the TensorFlow/protobuf crash by applying the `MessageFactory.GetPrototype` monkey-patch earlier and more robustly, including patching the internal `_message_factory.MessageFactory` class that TensorFlow often uses (your current patch targets only `google.protobuf.message_factory`). I also wrap the TensorFlow import in a safety fallback that retries after forcing the pure-Python protobuf runtime, so the notebook runs end-to-end reliably in this Kaggle image. These changes are score-neutral (they don’t alter the model/training logic) and only address the runtime failure. The rest of the pipeline is kept intact and still writes a correctly ordered `/kaggle/working/submission_deep.csv`.'
- What this solution (achieved 0.19823) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by expanding the monkey-patch to also patch the *instance* method (not just the class attribute), because the error is raised on an already-created `MessageFactory` object during TensorFlow import. I keep your TensorFlow setup/model/training logic intact; the changes are only to make TF import reliable under this Kaggle image. Since your current score (0.19856, lower-is-better) is already much better than the target (0.3703), I avoid any score-improving changes and focus on stability and producing the same valid submission CSV with correct column order. The pipeline still write `/kaggle/working/submission_deep.csv`.'
- What this solution (achieved 0.19672) has done: 'You’re failing before any training starts because TensorFlow import still triggers the protobuf `MessageFactory.GetPrototype` AttributeError; the current patch misses some factory objects and some protobuf versions don’t expose `default_factory` as expected. I make the protobuf patch more robust by patching both the Python and internal C++/upb-backed message factory classes/instances (where available) *before* importing TensorFlow, and I wrap TF import in a small retry that applies the patch again if needed. These changes are runtime/stability-only and should not affect model logic or score. Everything else (data loading, models, CV, final training, and writing `/kaggle/working/submission_deep.csv` with sample_submission column order) is kept the same.'
- What this solution (achieved 0.20199) has done: 'I fix the TensorFlow import crash by strengthening the protobuf `MessageFactory.GetPrototype` monkey-patch so it also patches already-instantiated factory objects that TensorFlow may use during import (the current patch misses a common instance path). This is a runtime-only stability fix and not change your model/training logic or score behavior. I also align the cell numbering to start at 1 (your provided script starts at cell 0), so the notebook/script runner can execute cleanly. Everything else (data loading, scaling, CNN architectures, CV, final training, and submission column alignment to `sample_submission.csv`) is preserved.'
- What this solution (achieved 0.19396) has done: 'I fix the TensorFlow import crash by making the protobuf `MessageFactory.GetPrototype` monkey-patch robust for both class- and instance-level factories, including factories created before patching, and by retrying the TF import once after patching. This is a runtime/stability-only change and should be score-neutral (no changes to model/training logic). I also renumber the notebook cells to start at 1 (your script starts at cell 0), because the runner expects sequential cell numbers. Everything else (data loading, scaling, model architectures, CV/training loops, and submission column alignment to `sample_submission.csv`) is preserved.'
- What this solution (achieved 0.20126) has done: 'I fix the TensorFlow import crash by making the protobuf `MessageFactory.GetPrototype` patch actually apply to already-instantiated factory objects (the current patch exits early when `GetPrototype` exists as an attribute placeholder but is not callable), and by retrying the TF import once after patching. These changes are runtime/stability-only and do not touch your model architectures, training loops, or post-processing, so they should be score-neutral. I also renumber the cells to start at 1 (your script starts at cell 0) so the provided “cells” format runs cleanly end-to-end. The rest of the pipeline (data loading, scaling, CV, final train, and writing `/kaggle/working/submission_deep.csv` aligned to `sample_submission.csv`) is preserved.'
- What this solution (achieved 0.19594) has done: 'I fix the TensorFlow import crash by replacing the brittle protobuf `MessageFactory.GetPrototype` monkey-patch with a safe, version-agnostic environment setup that prevents the incompatible C++ protobuf path from being used (the root cause of the AttributeError in this Kaggle image). This is a runtime-only change and does not alter your model architectures, training loops, losses, or prediction logic, so it should be score-neutral. I also renumber the cells to start at 1 (your runner expects sequential numbering) while preserving the original cell order/content, and keep the submission column alignment exactly matching `sample_submission.csv` so a valid `.csv` is always produced.'

# 9. Code solution

## === cell 0
import os, random
from pathlib import Path

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

SEED = 42


def set_py_seed(seed=SEED):
    random.seed(seed)
    import numpy as np

    np.random.seed(seed)


set_py_seed()

DATA_DIR = Path("/kaggle/input/leaf-classification")
if not DATA_DIR.exists():
    alt = Path("/kaggle/input/leaf-classification/leaf-classification")
    if alt.exists():
        DATA_DIR = alt

TRAIN_ZIP = DATA_DIR / "train.csv.zip"
TEST_ZIP = DATA_DIR / "test.csv.zip"
SAMPLE_ZIP = DATA_DIR / "sample_submission.csv.zip"
IMAGES_ZIP = DATA_DIR / "images.zip"  # optional

WORK_DIR = Path("/kaggle/working")
WORK_DIR.mkdir(parents=True, exist_ok=True)

print("Setup OK | DATA_DIR =", str(DATA_DIR))



## === cell 1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)

print("Imported sklearn, numpy, pandas, matplotlib")



## === cell 2
import tensorflow as tf

tf.random.set_seed(SEED)

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("Memory growth not set:", e)

try:
    strategy = tf.distribute.MirroredStrategy()
except Exception as e:
    print("MirroredStrategy failed, falling back to OneDeviceStrategy. Error:", repr(e))
    device = "/GPU:0" if gpus else "/CPU:0"
    strategy = tf.distribute.OneDeviceStrategy(device=device)

NUM_REPLICAS = strategy.num_replicas_in_sync
print("TensorFlow:", tf.__version__)
print("GPUs:", gpus)
print("Replicas in sync:", NUM_REPLICAS)

EPOCHS = 50
PATIENCE = 8
BASE_BATCH = 64
BATCH = min(BASE_BATCH * max(1, NUM_REPLICAS), 256)
print("Batch size:", BATCH)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
train_df = pd.read_csv(TRAIN_ZIP, compression="zip")
test_df = pd.read_csv(TEST_ZIP, compression="zip")
sample_sub = pd.read_csv(SAMPLE_ZIP, compression="zip")

id_col = "id"
target_col = "species"
feature_cols = [c for c in train_df.columns if c not in [id_col, target_col]]

X = train_df[feature_cols].values
y_labels = train_df[target_col].values
X_test = test_df[feature_cols].values
test_ids = test_df[id_col].values

le = LabelEncoder()
y_int = le.fit_transform(y_labels)
num_classes = len(le.classes_)

print(f"Rows={len(train_df)}  Features={len(feature_cols)}  Classes={num_classes}")
train_df.head(3)



## === cell 4
cls_counts = pd.Series(y_labels).value_counts().sort_values(ascending=False)
plt.figure(figsize=(10, 4))
cls_counts.head(20).plot(kind="bar")
plt.title("Top 20 species counts")
plt.tight_layout()
plt.savefig(WORK_DIR / "eda_class_balance_top20.png")
plt.close()

X_std_for_pca = StandardScaler().fit_transform(X)
pc = PCA(n_components=2, random_state=SEED).fit_transform(X_std_for_pca)
plt.figure(figsize=(6, 5))
plt.scatter(pc[:, 0], pc[:, 1], s=6, c=y_int, cmap="tab20")
plt.title("PCA on standardized features")
plt.tight_layout()
plt.savefig(WORK_DIR / "eda_pca.png")
plt.close()

print(
    "Saved EDA plots:",
    (WORK_DIR / "eda_class_balance_top20.png").name,
    (WORK_DIR / "eda_pca.png").name,
)



## === cell 5
pass



## === cell 6
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_std = scaler.fit_transform(X)
X_test_std = scaler.transform(X_test)

X_1d = X_std[..., None]  # (n_samples, n_features, 1)
X_test_1d = X_test_std[..., None]
input_shape = (X_1d.shape[1], 1)

print("Train shape:", X_1d.shape, " Test shape:", X_test_1d.shape)



## === cell 7
print("Reusing existing TensorFlow import and distribution strategy.")
print("TF:", tf.__version__, "| Replicas:", NUM_REPLICAS)
print("BATCH =", BATCH)



## === cell 8
from tensorflow.keras import layers, models, optimizers


def build_cnn_baseline(input_shape, num_classes):
    m = models.Sequential(name="cnn_baseline")
    m.add(layers.Conv1D(32, 5, activation="relu", input_shape=input_shape))
    m.add(layers.MaxPooling1D(2))
    m.add(layers.Flatten())
    m.add(layers.Dense(64, activation="relu"))
    m.add(layers.Dense(num_classes, activation="softmax"))
    m.compile(
        optimizer=optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return m


from tensorflow.keras import layers, models, optimizers, initializers


def build_cnn_deep(input_shape, num_classes):
    he = initializers.HeNormal()

    inp = layers.Input(shape=input_shape)

    x = layers.Conv1D(64, 7, padding="same", kernel_initializer=he)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.15)(x)

    x = layers.Conv1D(128, 5, padding="same", kernel_initializer=he)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.20)(x)

    x = layers.Conv1D(128, 3, padding="same", kernel_initializer=he)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    gap = layers.GlobalAveragePooling1D()(x)
    flat = layers.Flatten()(x)
    x = layers.Concatenate()([gap, flat])

    x = layers.Dense(128, activation="relu", kernel_initializer=he)(x)
    x = layers.Dropout(0.25)(x)

    out = layers.Dense(num_classes, activation="softmax")(x)

    m = models.Model(inp, out, name="cnn_deep_fixed")
    m.compile(
        optimizer=optimizers.Adam(learning_rate=3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return m


def build_cnn_tunable(
    input_shape, num_classes, filters=64, ksize=5, dense_units=128, dr=0.25, lr=1e-3
):
    he = tf.keras.initializers.HeNormal()
    inp = layers.Input(shape=input_shape)

    x = layers.Conv1D(filters, ksize, padding="same", kernel_initializer=he)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D()(x)
    x = layers.Dropout(dr)(x)

    x = layers.Conv1D(filters * 2, ksize, padding="same", kernel_initializer=he)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D()(x)
    x = layers.Dropout(dr)(x)

    x = layers.Conv1D(filters * 2, 3, padding="same", kernel_initializer=he)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    gap = layers.GlobalAveragePooling1D()(x)
    flat = layers.Flatten()(x)
    x = layers.Concatenate()([gap, flat])

    x = layers.Dense(dense_units, activation="relu", kernel_initializer=he)(x)
    x = layers.Dropout(dr)(x)

    out = layers.Dense(num_classes, activation="softmax")(x)

    m = models.Model(inp, out, name="cnn_tunable_fixed")
    m.compile(
        optimizer=optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return m




## === cell 9
from tensorflow.keras import callbacks
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

tf.keras.backend.clear_session()

_cv_device = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
cv_strategy = tf.distribute.OneDeviceStrategy(device=_cv_device)


def with_strategy(builder):
    def _wrapped(_shape, _classes, *args, **kwargs):
        with cv_strategy.scope():
            return builder(_shape, _classes, *args, **kwargs)

    return _wrapped


def train_with_val(model, X_tr, y_tr, X_va, y_va):
    cbs = [
        callbacks.ReduceLROnPlateau(
            patience=max(PATIENCE // 2, 3), factor=0.5, min_lr=1e-5, monitor="val_loss"
        ),
    ]
    hist = model.fit(
        X_tr,
        y_tr,
        validation_data=(X_va, y_va),
        epochs=EPOCHS,
        batch_size=BATCH,
        verbose=0,
        callbacks=cbs,
    )
    proba = model.predict(X_va, verbose=0)
    acc = accuracy_score(y_va, np.argmax(proba, axis=1))
    return proba, acc, hist.history


def run_cv(model_builder, X_data, y_data, folds=5, name="model"):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    oof = np.zeros((len(y_data), num_classes), dtype=np.float32)
    accs = []
    for fold, (tr, va) in enumerate(skf.split(X_data, y_data), start=1):
        print(f"[{name}] Fold {fold}/{folds}")
        X_tr, X_va = X_data[tr], X_data[va]
        y_tr, y_va = y_data[tr], y_data[va]
        model = model_builder(input_shape, num_classes)
        proba, acc, _ = train_with_val(model, X_tr, y_tr, X_va, y_va)
        oof[va] = proba
        accs.append(acc)
        print(f"  val_acc={acc:.4f}")
    oof_acc = accuracy_score(y_data, np.argmax(oof, axis=1))
    print(
        f"[{name}] mean_acc={np.mean(accs):.4f}  std={np.std(accs):.4f}  oof_acc={oof_acc:.4f}"
    )
    return oof, accs




## === cell 10
oof_baseline, accs_baseline = run_cv(
    with_strategy(build_cnn_baseline), X_1d, y_int, folds=5, name="CNN_baseline"
)

oof_deep, accs_deep = run_cv(
    with_strategy(build_cnn_deep), X_1d, y_int, folds=5, name="CNN_deep"
)



## === cell 11
cfg = {"filters": 64, "ksize": 5, "dense_units": 192, "dr": 0.25, "lr": 1e-3}
oof_tuned, accs_tuned = run_cv(
    with_strategy(lambda s, c: build_cnn_tunable(s, c, **cfg)),
    X_1d,
    y_int,
    folds=5,
    name="CNN_tuned_fixed",
)



## === cell 12
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)


def plot_multiclass_roc_pr(y_true_int, y_proba, title_prefix):
    y_true = tf.keras.utils.to_categorical(y_true_int, num_classes=num_classes)

    try:
        auc_macro = roc_auc_score(y_true, y_proba, average="macro", multi_class="ovr")
        auc_weighted = roc_auc_score(
            y_true, y_proba, average="weighted", multi_class="ovr"
        )
    except Exception:
        auc_macro = float("nan")
        auc_weighted = float("nan")

    fpr, tpr, _ = roc_curve(y_true.ravel(), y_proba.ravel())
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label="micro ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", label="chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(
        f"{title_prefix} — ROC (micro)\nmacro AUC={auc_macro:.3f} | weighted AUC={auc_weighted:.3f}"
    )
    plt.legend()
    plt.tight_layout()
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true.ravel(), y_proba.ravel())
    ap_macro = average_precision_score(y_true, y_proba, average="macro")
    ap_weighted = average_precision_score(y_true, y_proba, average="weighted")
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label="micro PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(
        f"{title_prefix} — PR (micro)\nmacro AP={ap_macro:.3f} | weighted AP={ap_weighted:.3f}"
    )
    plt.legend()
    plt.tight_layout()
    plt.close()


plot_multiclass_roc_pr(y_int, oof_baseline, "CNN Baseline")



## === cell 13
pass



## === cell 14
plot_multiclass_roc_pr(y_int, oof_deep, "CNN Deep")



## === cell 15
pass



## === cell 16
plot_multiclass_roc_pr(y_int, oof_tuned, "CNN Tuned (fixed)")



## === cell 17
pass



## === cell 18
import seaborn as sns
from sklearn.metrics import confusion_matrix

va_pred = np.argmax(oof_tuned, axis=1)
cm = confusion_matrix(y_int, va_pred)

plt.figure(figsize=(14, 12))
sns.heatmap(
    cm, cmap="Blues", cbar=True, square=True, xticklabels=False, yticklabels=False
)
plt.title("Confusion Matrix (Tuned CNN, 99 classes)", fontsize=16)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.tight_layout()
plt.close()



## === cell 19
pass



## === cell 20
pass



## === cell 21
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras import callbacks

n_samples = len(y_int)
min_val = num_classes
val_frac = max(0.10, min_val / n_samples)
val_frac = min(val_frac, 0.5)

X_tr, X_va, y_tr, y_va = train_test_split(
    X_1d, y_int, test_size=val_frac, stratify=y_int, random_state=SEED
)
print(
    f"Train/Val sizes: {len(y_tr)}/{len(y_va)} (val_frac={val_frac:.3f}, classes={num_classes})"
)


def choose_safe_batch(val_size, num_replicas, target=64):
    """Largest batch ≤ target and ≤ val_size that is divisible by replicas."""
    b = min(target, val_size)
    b = (b // num_replicas) * num_replicas
    if b < num_replicas:
        b = num_replicas
    return int(b)


BATCH_FINAL = choose_safe_batch(len(y_va), NUM_REPLICAS, target=64)
print(f"Replicas: {NUM_REPLICAS} | Val size: {len(y_va)} | Using BATCH={BATCH_FINAL}")


def make_ds(X, y=None, batch_size=32, training=False, drop=True):
    ds = (
        tf.data.Dataset.from_tensor_slices((X, y))
        if y is not None
        else tf.data.Dataset.from_tensor_slices(X)
    )
    if training:
        ds = ds.shuffle(min(len(X), 2048), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=drop)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_ds(X_tr, y_tr, batch_size=BATCH_FINAL, training=True, drop=True)
val_ds = make_ds(X_va, y_va, batch_size=BATCH_FINAL, training=False, drop=True)

with strategy.scope():
    model = build_cnn_deep(input_shape, num_classes)

cbs = [
    callbacks.ReduceLROnPlateau(
        patience=max(PATIENCE // 2, 3), factor=0.5, min_lr=1e-5, monitor="val_loss"
    ),
]

print(
    "Steps/epoch:",
    int(tf.data.experimental.cardinality(train_ds).numpy()),
    "| Val steps:",
    int(tf.data.experimental.cardinality(val_ds).numpy()),
)

history = model.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1, callbacks=cbs
)

val_proba = model.predict(
    tf.data.Dataset.from_tensor_slices(X_va).batch(BATCH_FINAL), verbose=0
)
val_acc = accuracy_score(y_va, val_proba.argmax(axis=1))
print(f"Validation accuracy (full val): {val_acc:.4f}")

test_proba = model.predict(
    tf.data.Dataset.from_tensor_slices(X_test_1d).batch(BATCH_FINAL), verbose=0
)
print("Test proba shape:", test_proba.shape)

class_cols = [c for c in sample_sub.columns if c != id_col]

submission = pd.DataFrame(test_proba, columns=le.classes_)
submission.insert(0, id_col, test_ids)

for c in class_cols:
    if c not in submission.columns:
        submission[c] = 0.0
submission = submission[[id_col] + class_cols]

out_path = "/kaggle/working/submission_deep.csv"
submission.to_csv(out_path, index=False)
print("Wrote submission to:", out_path)
print(submission.head(3))
