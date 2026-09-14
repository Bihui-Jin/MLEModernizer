# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Classify plant seedlings into their respective species.

## Metric
Micro-averaged F1-score.

## Submission Format
For each `file` in the test set, you must predict a probability for the `species` variable. The file should contain a header and have the following format:

```
file,species
0021e90e4.png,Maize
003d61042.png,Sugar beet
007b3da8b.png,Common wheat
etc.
```

## Dataset
The list of species is as follows:

```
Black-grass
Charlock
Cleavers
Common Chickweed
Common wheat
Fat Hen
Loose Silky-bent
Maize
Scentless Mayweed
Shepherds Purse
Small-flowered Cranesbill
Sugar beet
```

- **train.csv** - the training set, with plant species organized by folder
- **test.csv** - the test set, you need to predict the species of each image
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.14

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        input/
            description.md (84 lines)
            sample_submission.csv (667 lines)
            sample_submission.csv.zip (4.6 kB)
            test.zip (259.0 MB)
            train.zip (1.5 GB)
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
            test/
                5db43df54.png (177.5 kB)
                09d34fe5b.png (156.0 kB)
                ... and 664 other files
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
            train/
                Black-grass/
                    2ed589264.png (44.6 kB)
                    840a7ed59.png (708.1 kB)
                    ... and 219 other files
                Charlock/
                    ee4a02bf9.png (229.3 kB)
                    e795c53c9.png (354.4 kB)
                    ... and 322 other files
                ... and 11 other folders
        working/
            plant-seedlings-classification/
                description.md (84 lines)
                sample_submission.csv (667 lines)
                ... and 3 other files
                plant-seedlings-classification/
                test/
                    5db43df54.png (177.5 kB)
                    09d34fe5b.png (156.0 kB)
                    ... and 664 other files
                    test/
                train/
                    Black-grass/
                        2ed589264.png (44.6 kB)
                        840a7ed59.png (708.1 kB)
                        ... and 219 other files
                    Charlock/
                        ee4a02bf9.png (229.3 kB)
                        e795c53c9.png (354.4 kB)
                        ... and 322 other files
                    ... and 11 other folders
```

-> data/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> data/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> input/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

-> working/plant-seedlings-classification/sample_submission.csv has 666 rows and 2 columns.
The columns are: file, species

# 5. Code solution

## === cell 0
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import tensorflow as tf

tf.keras.utils.set_random_seed(123)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("TensorFlow version:", tf.__version__)




## === cell 1
def _pick_existing_root(candidates):
    for p in candidates:
        if os.path.isdir(p):
            return p
    return None


if os.path.exists("/kaggle/input"):
    DATA_ROOT = _pick_existing_root(
        [
            "/kaggle/input/plant-seedlings-classification",
            "/kaggle/input/plant-seedlings-classification/plant-seedlings-classification",
            "/kaggle/input",
        ]
    )
    print("Estamos en Kaggle")
else:
    DATA_ROOT = _pick_existing_root(
        [
            "./data/plant-seedlings-classification",
            "./data",
        ]
    )
    print("Estamos en Colab/local")

if DATA_ROOT is None:
    raise FileNotFoundError("Could not resolve DATA_ROOT from expected candidates.")

TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

if (not os.path.isdir(TRAIN_DIR)) or (not os.path.isdir(TEST_DIR)):
    alt_root = _pick_existing_root(
        [
            os.path.join(DATA_ROOT, "plant-seedlings-classification"),
            os.path.join(
                DATA_ROOT,
                "plant-seedlings-classification",
                "plant-seedlings-classification",
            ),
        ]
    )
    if alt_root is not None:
        DATA_ROOT = alt_root
        TRAIN_DIR = os.path.join(DATA_ROOT, "train")
        TEST_DIR = os.path.join(DATA_ROOT, "test")
        SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

print("DATA_ROOT:", DATA_ROOT)
print("TRAIN_DIR:", TRAIN_DIR, "exists:", os.path.isdir(TRAIN_DIR))
print("TEST_DIR:", TEST_DIR, "exists:", os.path.isdir(TEST_DIR))
print("SAMPLE_SUB:", SAMPLE_SUB_PATH, "exists:", os.path.isfile(SAMPLE_SUB_PATH))

if not os.path.isdir(TRAIN_DIR):
    raise FileNotFoundError(f"TRAIN_DIR not found: {TRAIN_DIR}")
if not os.path.isdir(TEST_DIR):
    raise FileNotFoundError(f"TEST_DIR not found: {TEST_DIR}")




## === cell 2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from PIL import Image
import matplotlib.image as mpimg

RUN_EDA = os.environ.get("RUN_EDA", "0") == "1"




## === cell 3
train_dir = TRAIN_DIR
categories = sorted(
    [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
)
print("Num categories found:", len(categories))
print(categories)




## === cell 4
if RUN_EDA:
    plt.figure(figsize=(9, 12))
    shown = 0
    for category in categories:
        files = [
            f
            for f in os.listdir(os.path.join(train_dir, category))
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if not files:
            continue
        img_name = random.choice(files)
        img_path = os.path.join(train_dir, category, img_name)

        shown += 1
        plt.subplot(4, 3, shown)
        plt.imshow(Image.open(img_path))
        plt.title(category)
        plt.axis("off")
        if shown >= 12:
            break
    plt.tight_layout()
    plt.show()




## === cell 5
if RUN_EDA:
    data_stats = []
    for category in categories:
        cat_path = os.path.join(train_dir, category)
        files = [
            f
            for f in os.listdir(cat_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if not files:
            continue

        sample_img_path = os.path.join(cat_path, files[random.randrange(len(files))])
        with Image.open(sample_img_path) as img:
            width, height = img.size

        data_stats.append(
            {
                "Category": category,
                "Count": len(files),
                "Sample Resolution": f"{width}x{height}",
            }
        )

    df_stats = pd.DataFrame(data_stats)
    print(df_stats)
    print(f"\nNumero total de imagenes: {df_stats['Count'].sum()}")




## === cell 6
if RUN_EDA:
    primera_foto = [
        f
        for f in os.listdir(os.path.join(TRAIN_DIR, "Black-grass"))
        if f.lower().endswith(".png")
    ][0]
    full_path = os.path.join(TRAIN_DIR, "Black-grass", primera_foto)

    img = mpimg.imread(full_path)
    plt.imshow(img)
    plt.axis("off")
    plt.show()

    print("dimensiones de la foto:", np.shape(img))
    print("Tipo de dato de la foto:", img.dtype)
    print("valor minimo de los pixeles:", img.min())
    print("valor maximo de los pixeles", img.max())




## === cell 7
if RUN_EDA:
    metadata_table = []
    for category in categories:
        files = [
            f
            for f in os.listdir(os.path.join(train_dir, category))
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if not files:
            continue
        img_name = files[0]
        img_path = os.path.join(train_dir, category, img_name)
        img = mpimg.imread(img_path)

        metadata_table.append(
            {
                "Category": category,
                "Dimensiones primera foto": np.shape(img),
                "tipo de datos": img.dtype,
                "valor minimo de los pixeles": float(img.min()),
                "valor maximo de los pixeles": float(img.max()),
            }
        )

    datos = pd.DataFrame(metadata_table)
    print(datos)




## === cell 8
from tensorflow.keras import layers, models




## === cell 9
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.15,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.15,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
)

NUM_CLASSES = len(train_ds.class_names)
print("Detected class names:", train_ds.class_names)
print("NUM_CLASSES:", NUM_CLASSES)




## === cell 10
AUTOTUNE = tf.data.AUTOTUNE

options = tf.data.Options()
options.experimental_deterministic = True
try:
    options.autotune.enabled = True
except Exception:
    pass
try:
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
except Exception:
    pass

train_ds = train_ds.with_options(options)
val_ds = val_ds.with_options(options)

USE_DISK_CACHE = os.environ.get("USE_DISK_CACHE", "1") == "1"
cache_dir = "/kaggle/working/tf_cache"
os.makedirs(cache_dir, exist_ok=True)

if USE_DISK_CACHE:
    train_cache_path = os.path.join(cache_dir, "train_cache")
    val_cache_path = os.path.join(cache_dir, "val_cache")
    train_ds = train_ds.cache(train_cache_path)
    val_ds = val_ds.cache(val_cache_path)
else:
    train_ds = train_ds.cache()
    val_ds = val_ds.cache()

try:
    train_ds = train_ds.shuffle(
        buffer_size=2048, seed=123, reshuffle_each_iteration=True
    )
except Exception:
    pass

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)




## === cell 11
model_cnn = models.Sequential(
    [
        layers.Input(shape=(224, 224, 3)),
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.1),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ]
)

model_cnn.summary()




## === cell 12
model_cnn.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=True,
    steps_per_execution=16,
)




## === cell 13
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=4, restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6, verbose=1
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_seedling_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1,
)




## === cell 14
history = model_cnn.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=[early_stop, reduce_lr, checkpoint],
)




## === cell 15
test_loss, test_acc = model_cnn.evaluate(val_ds, verbose=0)
print("Accuracy de la evaluacion:", float(test_acc))




## === cell 16
print("Skipping unused val_ds predictions to save time.")




## === cell 17
def visualize_predictions(model, dataset, class_names, n=18):
    images, labels = next(iter(dataset))
    preds = model.predict(images, verbose=0)

    plt.figure(figsize=(12, 25))
    n = min(n, images.shape[0])
    for i in range(n):
        ax = plt.subplot(6, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))

        actual_idx = int(np.argmax(labels[i]))
        predict_idx = int(np.argmax(preds[i]))
        color = "green" if actual_idx == predict_idx else "red"

        plt.title(
            f"Real: {class_names[actual_idx]}\nPredicho: {class_names[predict_idx]}",
            color=color,
            fontsize=10,
        )
        plt.axis("off")
    plt.tight_layout()
    plt.show()


if RUN_EDA:
    visualize_predictions(model_cnn, val_ds, train_ds.class_names)




## === cell 18
test_dir = TEST_DIR
test_files = sorted(
    [f for f in os.listdir(test_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
)

class_names = train_ds.class_names
print(f"Preprocesando {len(test_files)} imagenes...")

test_paths = [os.path.join(test_dir, f) for f in test_files]
path_ds = tf.data.Dataset.from_tensor_slices(test_paths)


def _load_and_resize(path):
    img_bytes = tf.io.read_file(path)
    p = tf.strings.lower(path)
    is_png = tf.strings.regex_full_match(p, ".*\\.png")
    img = tf.cond(
        is_png,
        lambda: tf.io.decode_png(img_bytes, channels=3),
        lambda: tf.io.decode_jpeg(img_bytes, channels=3),
    )
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    return img


path_ds = path_ds.map(_load_and_resize, num_parallel_calls=AUTOTUNE, deterministic=True)
test_ds2 = path_ds.with_options(options).batch(BATCH_SIZE)

test_cache_path = os.path.join(cache_dir, "test_cache")
USE_DISK_CACHE = os.environ.get("USE_DISK_CACHE", "1") == "1"
if USE_DISK_CACHE:
    test_ds2 = test_ds2.cache(test_cache_path)
else:
    test_ds2 = test_ds2.cache()

test_ds2 = test_ds2.prefetch(AUTOTUNE)

probs = model_cnn.predict(test_ds2, verbose=0)
pred_idx = np.argmax(probs, axis=1)
pred_labels = [class_names[int(i)] for i in pred_idx]

submission_df = pd.DataFrame({"file": test_files, "species": pred_labels})

if os.path.isfile(SAMPLE_SUB_PATH):
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    submission_df = sample_sub[["file"]].merge(submission_df, on="file", how="left")
    if submission_df["species"].isna().any():
        submission_df["species"] = submission_df["species"].fillna(class_names[0])

out_path = "/kaggle/working/submission.csv"
submission_df.to_csv(out_path, index=False)
print(f"Fichero guardado como: {out_path}")
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", list(submission_df.columns))
print("Submission columns OK:", list(submission_df.columns) == ["file", "species"])
print("Any NA species:", bool(submission_df["species"].isna().any()))
