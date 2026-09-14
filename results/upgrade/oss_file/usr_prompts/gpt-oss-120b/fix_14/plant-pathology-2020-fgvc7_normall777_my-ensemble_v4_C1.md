# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Detect apple diseases from images.

## Metric
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os, random, numpy as np, pandas as pd

BASE_PATH = "/kaggle/input/plant-pathology-2020-fgvc7"
IMG_SIZE = 224
EPOCHS = 5
BATCH_SIZE = 256  # will be overridden if TF strategy is available

random.seed(42)
np.random.seed(42)

try:
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers, callbacks
    from tensorflow.keras.applications import EfficientNetB0, DenseNet121
    from sklearn.model_selection import train_test_split

    tf.random.set_seed(42)
    tf.config.threading.set_intra_op_parallelism_threads(
        tf.config.threading.cpu_count()
    )
    tf.config.threading.set_inter_op_parallelism_threads(
        tf.config.threading.cpu_count()
    )

    AUTO = tf.data.AUTOTUNE
    strategy = tf.distribute.get_strategy()
    BATCH_SIZE = 256 * strategy.num_replicas_in_sync
    TF_AVAILABLE = False
    print("Tensorflow imported, but using fallback models to meet runtime constraints.")
except Exception as e:
    print("TensorFlow import failed – falling back to a simple baseline.")
    print("Error:", e)
    TF_AVAILABLE = False




## === cell 1
train_df = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
test_df = pd.read_csv(os.path.join(BASE_PATH, "test.csv"))
sub_df = pd.read_csv(os.path.join(BASE_PATH, "sample_submission.csv"))


def format_path(img_id):
    return os.path.join(BASE_PATH, "images", f"{img_id}.jpg")


train_paths = train_df["image_id"].apply(format_path).values
test_paths = test_df["image_id"].apply(format_path).values
train_labels = train_df.loc[:, "healthy":].values.astype(np.float32)

if TF_AVAILABLE:
    train_paths, valid_paths, train_labels, valid_labels = train_test_split(
        train_paths,
        train_labels,
        test_size=0.15,
        random_state=2020,
        stratify=train_labels.argmax(axis=1),
    )




## === cell 2
if TF_AVAILABLE:

    def decode_image(path, label=None, img_size=(IMG_SIZE, IMG_SIZE)):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, img_size)
        img = tf.cast(img, tf.float32) / 255.0
        return (img, label) if label is not None else img

    def augment(image, label=None):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        return (image, label) if label is not None else image

    train_ds = (
        tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
        .map(decode_image, num_parallel_calls=AUTO)
        .cache()
        .map(augment, num_parallel_calls=AUTO)
        .shuffle(buffer=len(train_paths), seed=42)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )

    valid_ds = (
        tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))
        .map(decode_image, num_parallel_calls=AUTO)
        .cache()
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )

    test_ds = (
        tf.data.Dataset.from_tensor_slices(test_paths)
        .map(lambda p: decode_image(p), num_parallel_calls=AUTO)
        .cache()
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )

    def build_model(base_cls, num_classes):
        base = base_cls(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
        )
        x = base.output
        outputs = layers.Dense(num_classes, activation="sigmoid")(x)
        model = models.Model(inputs=base.input, outputs=outputs)
        model.compile(
            optimizer=optimizers.Nadam(),
            loss="binary_crossentropy",
            metrics=["binary_accuracy"],
        )
        return model




## === cell 3
if TF_AVAILABLE:
    with strategy.scope():
        model1 = build_model(EfficientNetB0, train_labels.shape[1])
    print("Training EfficientNetB0...")
    model1.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS,
        callbacks=[callbacks.EarlyStopping(patience=2, restore_best_weights=True)],
        verbose=2,
    )
    tf.keras.backend.clear_session()

    with strategy.scope():
        model2 = build_model(DenseNet121, train_labels.shape[1])
    print("Training DenseNet121...")
    model2.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS,
        callbacks=[callbacks.EarlyStopping(patience=2, restore_best_weights=True)],
        verbose=2,
    )
    tf.keras.backend.clear_session()
else:
    print("Running enhanced pixel‑based fallback model...")
    from PIL import Image
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.multiclass import OneVsRestClassifier
    import concurrent.futures

    SMALL_SIZE = 128  # larger than before for richer features
    RANDOM_STATE = 42

    def load_and_resize(paths):
        def process(p):
            try:
                img = Image.open(p).convert("RGB")
                img = img.resize((SMALL_SIZE, SMALL_SIZE), Image.BILINEAR)
                arr = np.asarray(img, dtype=np.float32) / 255.0
            except Exception as e:
                print(f"Warning: could not process {p}: {e}")
                arr = np.zeros((SMALL_SIZE, SMALL_SIZE, 3), dtype=np.float32)
            return arr.ravel()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=os.cpu_count()
        ) as executor:
            flattened = list(executor.map(process, paths))
        return np.vstack(flattened)

    X_train = load_and_resize(train_paths)
    X_test = load_and_resize(test_paths)

    clf_lr = OneVsRestClassifier(
        LogisticRegression(
            solver="saga",
            max_iter=500,
            n_jobs=-1,
            class_weight="balanced",
            penalty="l2",
            random_state=RANDOM_STATE,
        )
    )
    clf_gb = OneVsRestClassifier(
        GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=3,
            random_state=RANDOM_STATE,
        )
    )
    clf_rf = OneVsRestClassifier(
        RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    )

    clf_lr.fit(X_train, train_labels)
    clf_gb.fit(X_train, train_labels)
    clf_rf.fit(X_train, train_labels)

    prob_lr = clf_lr.predict_proba(X_test).astype(np.float32)
    prob_gb = clf_gb.predict_proba(X_test).astype(np.float32)
    prob_rf = clf_rf.predict_proba(X_test).astype(np.float32)

    ensemble_pred = (prob_lr + prob_gb + prob_rf) / 3.0

print("Generating predictions...")
if TF_AVAILABLE and "model1" in globals() and "model2" in globals():
    pred1 = model1.predict(test_ds, verbose=1)
    pred2 = model2.predict(test_ds, verbose=1)
    alpha = 0.5
    ensemble_pred = alpha * pred1 + (1 - alpha) * pred2

sub_df.loc[:, "healthy":] = ensemble_pred
sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print(f"Submission saved to {sub_path}")
print(sub_df.head())
