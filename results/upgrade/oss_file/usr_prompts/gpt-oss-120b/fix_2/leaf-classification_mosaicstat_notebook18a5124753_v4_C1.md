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

0.27169

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 0.27169) has done: 'I fix the TensorFlow import error by disabling GPU visibility before loading TF, and adjust the stratified validation split so the validation set contains at least as many samples as there are classes (99). These changes resolve the runtime crashes and allow the script to generate a proper submission CSV while keeping the original modeling logic intact.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
import os, random, json
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

SEED = 42


def set_py_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)


set_py_seed()

DATA_DIR = Path("/kaggle/input/leaf-classification")
TRAIN_ZIP = DATA_DIR / "train.csv.zip"
TEST_ZIP = DATA_DIR / "test.csv.zip"
SAMPLE_ZIP = DATA_DIR / "sample_submission.csv.zip"
IMAGES_ZIP = DATA_DIR / "images.zip"  # optional
WORK_DIR = Path("/kaggle/working")
WORK_DIR.mkdir(parents=True, exist_ok=True)

print("Setup OK")




## === cell 2
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




## === cell 3
import tensorflow as tf

tf.random.set_seed(SEED)

strategy = tf.distribute.MirroredStrategy()
NUM_REPLICAS = strategy.num_replicas_in_sync
print("TensorFlow:", tf.__version__)
print("Replicas in sync:", NUM_REPLICAS)

EPOCHS = 50
PATIENCE = 8
BASE_BATCH = 64
BATCH = min(BASE_BATCH * max(1, NUM_REPLICAS), 256)
print("Batch size:", BATCH)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
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




## === cell 5
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




## === cell 6
cls_counts = pd.Series(y_labels).value_counts().sort_values(ascending=False)
plt.figure(figsize=(10, 4))
cls_counts.head(20).plot(kind="bar")
plt.title("Top 20 species counts")
plt.tight_layout()
plt.savefig(WORK_DIR / "eda_class_balance_top20.png")
plt.show()  # show inline

X_std_for_pca = StandardScaler().fit_transform(X)
pc = PCA(n_components=2, random_state=SEED).fit_transform(X_std_for_pca)
plt.figure(figsize=(6, 5))
plt.scatter(pc[:, 0], pc[:, 1], s=6, c=y_int, cmap="tab20")
plt.title("PCA on standardized features")
plt.tight_layout()
plt.savefig(WORK_DIR / "eda_pca.png")
plt.show()  # show inline




## === cell 7
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_std = scaler.fit_transform(X)
X_test_std = scaler.transform(X_test)

X_1d = X_std[..., None]  # (n_samples, n_features, 1)
X_test_1d = X_test_std[..., None]
input_shape = (X_1d.shape[1], 1)

print("Train shape:", X_1d.shape, " Test shape:", X_test_1d.shape)




## === cell 8
import tensorflow as tf

tf.random.set_seed(SEED)

for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print("Memory growth not set:", e)

strategy = tf.distribute.MirroredStrategy()
NUM_REPLICAS = strategy.num_replicas_in_sync
print("TF:", tf.__version__, "| Replicas:", NUM_REPLICAS)

EPOCHS = 50
PATIENCE = 8
BASE_BATCH = 64
BATCH = min(BASE_BATCH * max(1, NUM_REPLICAS), 256)
print("BATCH =", BATCH)




## === cell 9
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




## === cell 10
from tensorflow.keras import callbacks
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf

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
        callbacks.EarlyStopping(
            patience=PATIENCE, restore_best_weights=True, monitor="val_accuracy"
        ),
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




## === cell 11
oof_baseline, accs_baseline = run_cv(
    with_strategy(build_cnn_baseline), X_1d, y_int, folds=5, name="CNN_baseline"
)

oof_deep, accs_deep = run_cv(
    with_strategy(build_cnn_deep), X_1d, y_int, folds=5, name="CNN_deep"
)




## === cell 12
import random
from sklearn.metrics import accuracy_score

cfg = {"filters": 64, "ksize": 5, "dense_units": 192, "dr": 0.25, "lr": 1e-3}
oof_tuned, accs_tuned = run_cv(
    with_strategy(lambda s, c: build_cnn_tunable(s, c, **cfg)),
    X_1d,
    y_int,
    folds=5,
    name="CNN_tuned_fixed",
)




## === cell 13
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
import tensorflow as tf


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
        f"{title_prefix} — ROC (micro)\\nmacro AUC={auc_macro:.3f} | weighted AUC={auc_weighted:.3f}"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()

    precision, recall, _ = precision_recall_curve(y_true.ravel(), y_proba.ravel())
    ap_macro = average_precision_score(y_true, y_proba, average="macro")
    ap_weighted = average_precision_score(y_true, y_proba, average="weighted")
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label="micro PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(
        f"{title_prefix} — PR (micro)\\nmacro AP={ap_macro:.3f} | weighted AP={ap_weighted:.3f}"
    )
    plt.legend()
    plt.tight_layout()
    plt.show()


plot_multiclass_roc_pr(y_int, oof_baseline, "CNN Baseline")




## === cell 14
plot_multiclass_roc_pr(y_int, oof_deep, "CNN Deep")




## === cell 15
plot_multiclass_roc_pr(y_int, oof_tuned, "CNN Tuned (fixed)")




## === cell 16
import seaborn as sns
import matplotlib.pyplot as plt
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
plt.show()




## === cell 17
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras import callbacks

X_tr, X_va, y_tr, y_va = train_test_split(
    X_1d, y_int, test_size=0.20, stratify=y_int, random_state=SEED
)


def choose_safe_batch(val_size, num_replicas, target=64):
    """Largest batch ≤ target and ≤ val_size that is divisible by replicas."""
    b = min(target, val_size)
    b = (b // num_replicas) * num_replicas
    if b < num_replicas:
        b = num_replicas
    return int(b)


BATCH = choose_safe_batch(len(y_va), NUM_REPLICAS, target=64)
print(f"Replicas: {NUM_REPLICAS} | Val size: {len(y_va)} | Using BATCH={BATCH}")


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


train_ds = make_ds(X_tr, y_tr, batch_size=BATCH, training=True, drop=True)
val_ds = make_ds(X_va, y_va, batch_size=BATCH, training=False, drop=True)

with strategy.scope():
    model = build_cnn_deep(input_shape, num_classes)

cbs = [
    callbacks.EarlyStopping(
        patience=PATIENCE, restore_best_weights=True, monitor="val_accuracy"
    ),
    callbacks.ReduceLROnPlateau(
        patience=max(PATIENCE // 2, 3), factor=0.5, min_lr=1e-5, monitor="val_loss"
    ),
]

print(
    "Steps/epoch:",
    tf.data.experimental.cardinality(train_ds).numpy(),
    "| Val steps:",
    tf.data.experimental.cardinality(val_ds).numpy(),
)

history = model.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1, callbacks=cbs
)

val_proba = model.predict(
    tf.data.Dataset.from_tensor_slices(X_va).batch(BATCH), verbose=0
)
val_acc = accuracy_score(y_va, val_proba.argmax(axis=1))
print(f"Validation accuracy (full val): {val_acc:.4f}")

test_proba = model.predict(
    tf.data.Dataset.from_tensor_slices(X_test_1d).batch(BATCH), verbose=0
)
print("Test proba shape:", test_proba.shape)

id_col = "id"
class_cols = list(sample_sub.columns)
class_cols.remove(id_col)
sub_probs = pd.DataFrame(0.0, index=np.arange(len(test_ids)), columns=class_cols)
sub_probs[le.classes_] = test_proba
submission = pd.concat([pd.Series(test_ids, name=id_col), sub_probs], axis=1)

out_path = "/kaggle/working/submission_deep.csv"
submission.to_csv(out_path, index=False)
print("Wrote submission to:", out_path)
display(submission.head(3))
