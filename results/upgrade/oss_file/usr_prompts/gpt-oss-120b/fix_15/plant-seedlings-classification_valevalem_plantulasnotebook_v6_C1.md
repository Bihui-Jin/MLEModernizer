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
import tensorflow as tf

tf.config.optimizer.set_jit(True)

from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

cpu_count = os.cpu_count() or 1
tf.config.threading.set_inter_op_parallelism_threads(cpu_count)
tf.config.threading.set_intra_op_parallelism_threads(cpu_count)

import numpy as np, random

np.random.seed(123)
random.seed(123)
tf.random.set_seed(123)



## === cell 1
if os.path.exists("/kaggle/input"):
    DATA_ROOT = "/kaggle/input/plant-seedlings-classification"
    print("Estamos en Kaggle")
else:
    DATA_ROOT = "./data"
    print("Estamos en Colab")



## === cell 2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from PIL import Image



## === cell 3
train_dir = os.path.join(DATA_ROOT, "train")
categories = sorted(
    [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
)
categories



## === cell 4
plt.figure(figsize=(9, 12))
for i, category in enumerate(categories):
    img_name = random.choice(os.listdir(os.path.join(train_dir, category)))
    img_path = os.path.join(train_dir, category, img_name)

    plt.subplot(4, 3, i + 1)
    plt.imshow(Image.open(img_path))
    plt.title(category)
    plt.axis("off")
plt.close()  # avoid blocking display



## === cell 5
data_stats = []

for category in categories:
    cat_path = os.path.join(train_dir, category)
    files = [
        f for f in os.listdir(cat_path) if os.path.isfile(os.path.join(cat_path, f))
    ]

    sample_img_path = os.path.join(cat_path, random.choice(files))
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
import matplotlib.image as mpimg

primera_foto = os.listdir(os.path.join(train_dir, "Black-grass"))[0]
full_path = os.path.join(train_dir, "Black-grass", primera_foto)

img = mpimg.imread(full_path)
plt.imshow(img)
plt.axis("off")
plt.close()  # avoid blocking display

print("dimensiones de la foto:", np.shape(img))
print("Tipo de dato de la foto:", img.dtype)
print("valor minimo de los pixeles:", img.min())
print("valor maximo de los pixeles", img.max())



## === cell 7
metadata_table = []

for category in categories:
    img_name = os.listdir(os.path.join(train_dir, category))[0]
    img_path = os.path.join(train_dir, category, img_name)
    img = mpimg.imread(img_path)

    metadata_table.append(
        {
            "Category": category,
            "Dimensiones primera foto": np.shape(img),
            "tipo de datos": img.dtype,
            "valor minimo de los pixeles": img.min(),
            "valor maximo de los pixeles": img.max(),
        }
    )

datos = pd.DataFrame(metadata_table)
print(datos)



## === cell 8
from tensorflow.keras import layers, models

NUM_CLASSES = 13

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



## === cell 9
model_cnn.summary()



## === cell 10
model_cnn.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)



## === cell 11
IMG_SIZE = (224, 224)
BATCH_SIZE = 1024  # increased batch size to cut training steps
AUTOTUNE = tf.data.AUTOTUNE

full_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
    seed=123,
)

ds_size = tf.data.experimental.cardinality(full_ds).numpy()
train_size = int(0.85 * ds_size)

train_ds = full_ds.take(train_size)
val_ds = full_ds.skip(train_size)

class_names = full_ds.class_names

val_ds = val_ds.cache().prefetch(AUTOTUNE)

train_ds = train_ds.shuffle(buffer_size=1000, seed=123).prefetch(AUTOTUNE)

options = tf.data.Options()
options.experimental_deterministic = False
train_ds = train_ds.with_options(options)
val_ds = val_ds.with_options(options)



## === cell 12
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



## === cell 13
history = model_cnn.fit(
    train_ds,
    validation_data=val_ds,
    epochs=12,
    callbacks=[early_stop, reduce_lr, checkpoint],
)



## === cell 14
test_loss, test_acc = model_cnn.evaluate(val_ds, verbose=0)
print(f"Accuracy de la evaluacion:", test_acc)



## === cell 15
sample_images, _ = next(iter(val_ds))
predicciones = model_cnn.predict(sample_images, verbose=0)




## === cell 16
def visualize_predictions(model, dataset, class_names):
    images, labels = next(iter(dataset))

    plt.figure(figsize=(12, 25))
    for i in range(18):
        ax = plt.subplot(6, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))

        actual_idx = np.argmax(labels[i])
        predict_idx = np.argmax(predicciones[i])

        color = "green" if actual_idx == predict_idx else "red"

        plt.title(
            f"Real: {class_names[actual_idx]}\nPredicho: {class_names[predict_idx]}",
            color=color,
            fontsize=10,
        )
        plt.axis("off")
    plt.tight_layout()
    plt.close()  # avoid blocking display




## === cell 17
from tensorflow.keras.preprocessing import image

test_dir = os.path.join(DATA_ROOT, "test")
test_files = sorted(
    [f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]
)

print(f"Preprocesando {len(test_files)} imágenes en lotes...")

test_paths = [os.path.join(test_dir, fname) for fname in test_files]

test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.map(
    lambda x: tf.image.decode_jpeg(tf.io.read_file(x), channels=3),
    num_parallel_calls=AUTOTUNE,
)
test_ds = test_ds.map(
    lambda img: tf.image.resize(img, IMG_SIZE),
    num_parallel_calls=AUTOTUNE,
)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

batch_preds = model_cnn.predict(test_ds, verbose=0)

predictions = [
    {"file": fname, "species": class_names[np.argmax(pred)]}
    for fname, pred in zip(test_files, batch_preds)
]

submission_df = pd.DataFrame(predictions)
submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)

print(f"Fichero guardado como '{submission_path}'")
