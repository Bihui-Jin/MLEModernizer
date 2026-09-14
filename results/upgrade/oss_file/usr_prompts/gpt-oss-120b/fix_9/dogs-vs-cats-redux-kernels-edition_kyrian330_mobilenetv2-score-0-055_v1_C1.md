# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

3.13

# 3. Installed packages

No external packages required in the script and installed.

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

# 5. Code solution

## === cell 0
import os, glob, random, zipfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, regularizers, callbacks

    TF_AVAILABLE = True
except Exception as e:
    print("TensorFlow import failed; proceeding with fallback model.")
    TF_AVAILABLE = False

if TF_AVAILABLE:
    tf.config.threading.set_inter_op_parallelism_threads(
        tf.config.threading.cpu_count()
    )
    tf.config.threading.set_intra_op_parallelism_threads(
        tf.config.threading.cpu_count()
    )

print("✅ Environment ready")




## === cell 1
extract_dir = "/kaggle/input"
train_dir = os.path.join(extract_dir, "dogs-vs-cats-redux-kernels-edition", "train")
test_dir = os.path.join(extract_dir, "dogs-vs-cats-redux-kernels-edition", "test")

if not os.path.isdir(train_dir):
    raise RuntimeError(f"Training directory not found at {train_dir}")
if not os.path.isdir(test_dir):
    raise RuntimeError(f"Test directory not found at {test_dir}")

DATA_DIR = train_dir
TEST_DIR = test_dir

IMG_SIZE = 256
BATCH_SIZE = 32
SEED = 42
print("✅ Paths verified")




## === cell 2
all_images = []
for cls in ("cat", "dog"):
    all_images.extend(glob.glob(os.path.join(DATA_DIR, cls, "*.jpg")))
labels = [1 if "dog" in os.path.basename(p).lower() else 0 for p in all_images]

if len(all_images) == 0:
    raise RuntimeError("No training images found. Check DATA_DIR path.")

train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_images, labels, test_size=0.15, stratify=labels, random_state=SEED
)


def decode_img(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return img, label


def build_dataset(paths, labels, is_train=True):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(decode_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()
    if is_train:
        ds = ds.shuffle(1024, seed=SEED)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


if TF_AVAILABLE:
    tf.random.set_seed(SEED)  # ensure deterministic behavior
    train_ds = build_dataset(train_paths, train_labels, is_train=True)
    val_ds = build_dataset(val_paths, val_labels, is_train=False)
else:
    train_ds = val_ds = None
print("✅ Dataset preparation complete")




## === cell 3
if TF_AVAILABLE:
    from tensorflow.keras import mixed_precision

    policy = mixed_precision.Policy("mixed_float16")
    mixed_precision.set_global_policy(policy)

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False
    for layer in base_model.layers[-30:]:
        layer.trainable = True

    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="MobileNetV2_GAP")(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(
        x
    )
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(2e-4))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid", dtype="float32")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
else:
    model = None
print("✅ Model definition complete")




## === cell 4
if TF_AVAILABLE and model is not None:
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True
    )
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=20, callbacks=[early_stop], verbose=2
    )
    print("✅ Training finished")
else:
    history = None
    print("✅ Skipping training (fallback mode)")




## === cell 5
test_paths = sorted(
    glob.glob(os.path.join(TEST_DIR, "*.jpg"))
    + glob.glob(os.path.join(TEST_DIR, "*/*.jpg")),
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
)


def build_test_ds(paths):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(lambda p: decode_img(p, 0)[0], num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


if TF_AVAILABLE and model is not None:
    test_ds = build_test_ds(test_paths)
    preds = model.predict(test_ds).ravel()
    preds = np.clip(preds, 0.005, 0.995)  # Clip extreme values
else:
    preds = np.full(len(test_paths), 0.5, dtype=np.float32)

print("✅ Predictions ready")




## === cell 6
print(f"预测最大值：{preds.max():.4f}")
print(f"预测最小值：{preds.min():.4f}")
print(f"预测均值：{preds.mean():.4f}")

plt.figure(figsize=(8, 4))
plt.hist(preds, bins=100)
plt.title("Test Prediction Distribution")
plt.xlabel("Probability")
plt.ylabel("Count")
plt.savefig("/kaggle/working/prediction_histogram.png")
plt.close()
print("✅ Prediction histogram saved at /kaggle/working/prediction_histogram.png")




## === cell 7
submission = pd.DataFrame(
    {
        "id": [int(os.path.splitext(os.path.basename(p))[0]) for p in test_paths],
        "label": preds,
    }
)
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("✅ submission saved at /kaggle/working/submission.csv")
