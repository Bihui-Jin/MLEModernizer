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
Given a dataset of images of dogs, predict the breed of each image.

## Metric
Multi Class Log Loss.

## Submission Format
For each image in the test set, you must predict a probability for each of the different breeds. The file should contain a header and have the following format:
```
id,affenpinscher,afghan_hound,..,yorkshire_terrier
000621fb3cbb32d8935728e48679680e,0.0083,0.0,...,0.0083
etc.
```

## Dataset Description
- `train.zip` - the training set, you are provided the breed for these dogs
- `test.zip` - the test set, you must predict the probability of each breed for each image
- `sample_submission.csv` - a sample submission file in the correct format
- `labels.csv` - the breeds for the images in the train set

# 2. Python version

3.13

# 3. Installed packages



# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
        input/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
        working/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
```

-> data/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> data/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> input/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> input/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.99749

# 6. Current score

0.80115

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.80633) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf implementation before importing TensorFlow, which resolves the `MessageFactory.GetPrototype` error in this environment. Then I fix test file enumeration so we only pass `.jpg` files to the `tf.io.read_file` pipeline (your current `test_files` accidentally includes a `test/` directory entry, causing `Is a directory`). Finally, I keep your modeling/training logic intact but make the inference/submission robust: ensure predictions are aligned to `sample_submission.csv` columns, renormalized to sum to 1 per row (safe for log-loss), and always write a valid `submission.csv`.'
- What this solution (achieved 0.78404) has done: 'I fix the TensorFlow import crash by forcing protobuf to use the pure-Python implementation *and* ensuring that environment flag is set before any TensorFlow/protobuf-related import happens (and by pinning the Python protobuf runtime to the compatible “python” backend). Then I keep your exact data pipeline/model/training logic, but make the label construction a proper NumPy array (instead of a Python list of arrays) so `tf.data.Dataset.from_tensor_slices` gets a clean, consistent shape/dtype. Finally, I keep your submission alignment/renormalization (good for log-loss) and ensure test file enumeration and `id` ordering matches `sample_submission.csv` to avoid silent misalignment that can hurt score.'
- What this solution (achieved 0.80115) has done: 'You’re currently crashing on TensorFlow import due to an incompatibility between the protobuf runtime and the TF build in this environment; I fix that by forcing the pure-Python protobuf implementation *and* ensuring the environment variables are set before any protobuf/TensorFlow import happens. After TF successfully imports, I keep your exact model/training/inference logic, but make the TFHub usage offline-safe by first trying to load a local SavedModel copy from the Kaggle dataset (if present) and only falling back to the URL; this avoids hard failures when internet is unavailable. Finally, I keep your submission alignment/renormalization and ensure the output is always written as `submission.csv` with correct row order and columns.'

# 9. Code solution

## === cell 0
import os, timeit, datetime

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

import tensorflow as tf

print("TF version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def cpu():
    with tf.device("/cpu:0"):
        random_image_cpu = tf.random.normal((10, 100, 100, 3))
        net_cpu = tf.keras.layers.Conv2D(8, 7)(random_image_cpu)
        return tf.math.reduce_sum(net_cpu)


def gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return None
    with tf.device("/device:GPU:0"):
        random_image_gpu = tf.random.normal((10, 100, 100, 3))
        net_gpu = tf.keras.layers.Conv2D(8, 7)(random_image_gpu)
        return tf.math.reduce_sum(net_gpu)


_ = cpu()
_ = gpu()



## === cell 2
import matplotlib.pyplot as plt
from IPython.display import Image

DATA_DIR = "/kaggle/input/dog-breed-identification"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
LABELS_CSV = os.path.join(DATA_DIR, "labels.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")

labels = pd.read_csv(LABELS_CSV)
labels.head()



## === cell 3
labels.info()



## === cell 4
print("Train images:", len(os.listdir(TRAIN_DIR)))
print("Test images:", len(os.listdir(TEST_DIR)))
print("Labels rows:", len(labels))
print("Unique breeds:", labels["breed"].nunique())



## === cell 5
files = [os.path.join(TRAIN_DIR, f"{fname}.jpg") for fname in labels["id"].astype(str)]
assert len(files) == len(labels), "Mismatch between files list and labels rows."
Image(files[0])



## === cell 6
breeds = labels["breed"].to_numpy()
unique_breeds = np.unique(breeds)
print("Num unique breeds:", len(unique_breeds))
print("First 5 breeds:", unique_breeds[:5])



## === cell 7
boolean_labels = (breeds[:, None] == unique_breeds[None, :]).astype(np.float32)
X = files
y = boolean_labels

print("X:", len(X), "y shape:", y.shape, "y dtype:", y.dtype)



## === cell 8
from sklearn.model_selection import train_test_split

NUM_IMAGES = 1000  # keep original core setup
X_train, X_val, y_train, y_val = train_test_split(
    X[:NUM_IMAGES], y[:NUM_IMAGES], train_size=0.8, random_state=42, shuffle=True
)
len(X_train), len(X_val), len(y_train), len(y_val)



## === cell 9
IMG_SIZE = 224


def process_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
    return image


def get_label(path, label):
    return process_image(path), label


BATCH_SIZE = 32


def create_batch(X, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):
    if test_data:
        print("Creating test batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X)))
        data_batch = (
            data.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        return data_batch
    elif valid_data:
        print("Creating validation batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
        data_batch = (
            data.map(get_label, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        return data_batch
    else:
        print("Creating train batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
        data = data.shuffle(len(X))
        data_batch = (
            data.map(get_label, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        return data_batch


train_data = create_batch(X_train, y_train)
val_data = create_batch(X_val, y_val, valid_data=True)




## === cell 10
def show25img(images, labels_):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(images[i])
        plt.title(unique_breeds[labels_[i].argmax()])
        plt.axis("off")


train_img, train_label = next(train_data.as_numpy_iterator())
show25img(train_img, train_label)



## === cell 11
import tensorflow_hub as hub

INPUT_SHAPE = [None, IMG_SIZE, IMG_SIZE, 3]
OUTPUT_SHAPE = len(unique_breeds)
MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"


def _try_load_hub_model():
    candidate_paths = [
        "/kaggle/input/tfhub-mobilenet-v2-130-224",  # if user added a dataset
        "/kaggle/input/tfhub-models/mobilenet_v2_130_224_classification_4",
        "/kaggle/input/mobilenet-v2-130-224",
        "/kaggle/working/tfhub_mobilenet_v2_130_224_classification_4",
    ]
    for p in candidate_paths:
        if os.path.isdir(p):
            try:
                m = hub.load(p)
                print(f"Loaded TFHub model from local path: {p}")
                return m
            except Exception as e:
                print(
                    f"Found local candidate but failed to load ({p}): {type(e).__name__}: {e}"
                )
                continue
    print(f"Loading TFHub model from URL: {MODEL_URL}")
    return hub.load(MODEL_URL)


hub_model = _try_load_hub_model()


def hub_forward(images):
    outputs = hub_model(images)
    if isinstance(outputs, dict):
        for k in ("default", "logits", "predictions"):
            if k in outputs:
                return outputs[k]
        return next(iter(outputs.values()))
    return outputs


def create_model(
    input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, model_url=MODEL_URL
):
    print("Building model with", model_url)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(IMG_SIZE, IMG_SIZE, 3)),
            tf.keras.layers.Lambda(hub_forward, name="tfhub_mobilenet_v2_130_224"),
            tf.keras.layers.Dense(units=output_shape, activation="softmax"),
        ]
    )
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )
    return model


model = create_model()
model.summary()



## === cell 12
os.makedirs("/kaggle/working/logs", exist_ok=True)


def tensorboard_callback():
    logdir = os.path.join(
        "/kaggle/working/logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    return tf.keras.callbacks.TensorBoard(logdir)


early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=3, restore_best_weights=True
)

NUM_EPOCHS = 100


def train_model():
    m = create_model()
    tb = tensorboard_callback()
    m.fit(
        x=train_data,
        epochs=NUM_EPOCHS,
        validation_data=val_data,
        validation_freq=1,
        callbacks=[tb, early_stop],
        verbose=2,
    )
    return m


model = train_model()



## === cell 13
predictions = model.predict(val_data, verbose=1)
print("Val preds shape:", predictions.shape)

idx = 0
print(predictions[idx])
print(f"Max value is: {np.max(predictions[idx])}")
print(f"index of max value is: {np.argmax(predictions[idx])}")
print(f"predicted breed is {unique_breeds[np.argmax(predictions[idx])]}")




## === cell 14
def unbatchify(data):
    images = []
    labels_ = []
    for image, label in data.unbatch().as_numpy_iterator():
        images.append(image)
        labels_.append(label)
    return images, labels_


val_img, val_label = unbatchify(val_data)


def get_y_val(index):
    return unique_breeds[np.argmax(predictions[index])]


def plot_pred(index):
    y_val = get_y_val(index)
    y_actual = unique_breeds[np.argmax(val_label[index])]
    img = val_img[index]
    y_prob = np.max(predictions[index])

    plt.imshow(img)
    plt.axis("off")
    color = "green" if y_val == y_actual else "red"
    plt.title(f"{y_val} {(y_prob*100):.2f}% ({y_actual})", color=color)


plot_pred(13)



## === cell 15
full_data = create_batch(X, y)
full_model = create_model()

full_model_tb = tensorboard_callback()
full_model_stop = tf.keras.callbacks.EarlyStopping(
    monitor="accuracy", patience=5, restore_best_weights=True
)

full_model.fit(
    x=full_data,
    epochs=NUM_EPOCHS,
    callbacks=[full_model_tb, full_model_stop],
    verbose=2,
)

os.makedirs("/kaggle/working/models", exist_ok=True)
full_model.save(
    "/kaggle/working/models/full_trained_mobilenetV2-Adam.keras", overwrite=True
)



## === cell 16
test_filenames = sorted(
    [
        f
        for f in os.listdir(TEST_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and os.path.isfile(os.path.join(TEST_DIR, f))
    ]
)
assert len(test_filenames) > 0, f"No test images found in {TEST_DIR}"

test_files = [os.path.join(TEST_DIR, fname) for fname in test_filenames]
test_ids = [os.path.splitext(fname)[0] for fname in test_filenames]

test_data = create_batch(test_files, test_data=True)
test_predictions = full_model.predict(test_data, verbose=1)
print("Test preds shape:", test_predictions.shape)



## === cell 17
sample = pd.read_csv(SAMPLE_SUB)
breed_cols = [c for c in sample.columns if c != "id"]

class_to_idx = {b: i for i, b in enumerate(unique_breeds)}
missing = [b for b in breed_cols if b not in class_to_idx]
if missing:
    raise ValueError(
        f"Breeds in sample submission not found in training labels: {missing[:5]}"
    )

reindex = [class_to_idx[b] for b in breed_cols]
preds_aligned = test_predictions[:, reindex]

preds_aligned = np.clip(preds_aligned, 1e-12, 1.0)
preds_aligned = preds_aligned / preds_aligned.sum(axis=1, keepdims=True)

pred_df = pd.DataFrame(preds_aligned, columns=breed_cols)
pred_df.insert(0, "id", test_ids)

pred_df = sample[["id"]].merge(pred_df, on="id", how="left")
assert (
    pred_df.isna().sum().sum() == 0
), "Some test ids in sample submission were not predicted."
assert len(pred_df) == len(
    sample
), f"Submission length {len(pred_df)} != expected {len(sample)}"

pred_df.to_csv("submission.csv", index=False)

print(pred_df.head())
print("Wrote submission.csv with shape:", pred_df.shape)
print("Saved to:", os.path.abspath("submission.csv"))
