# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Identify hotels from images.

## Metric
Mean Average Precision @ 5 (MAP@5)

## Submission Format
For each image in the test set, you must predict a space-delimited list of hotel IDs that could match that image. The first ID should be the most relevant one and the last the least relevant one. The file should contain a header and have the following format:

```
image,hotel_id
99e91ad5f2870678.jpg,36363 53586 18807 64314 60181
b5cc62ab665591a9.jpg,36363 53586 18807 64314 60181
d5664a972d5a644b.jpg,36363 53586 18807 64314 60181
```

## Dataset
**train.csv** - The training set metadata.

- `image` - The image ID.

- `chain` - An ID code for the hotel chain. A `chain` of zero (0) indicates that the hotel is either not part of a chain or the chain is not known. This field is not available for the test set. The number of hotels per chain varies widely.

- `hotel_id` - The hotel ID. The target class.

- `timestamp` - When the image was taken. Provided for the training set only.

**sample_submission.csv** - A sample submission file in the correct format.

- `image` The image ID

- `hotel_id` The hotel ID. The target class.

**train_images** - The training set contains 97000+ images from around 7700 hotels from across the globe. All of the images for each hotel chain are in a dedicated subfolder for that chain.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 13,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
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
            description.md (120 lines)
            sample_submission.csv (9757 lines)
            sample_submission.csv.zip (106.8 kB)
            test.zip (160 Bytes)
            test_images.zip (2.6 GB)
            train.csv (87799 lines)
            train.csv.zip (1.9 MB)
            train.zip (162 Bytes)
            train_images.zip (23.5 GB)
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
            test/
                test/
            test_images/
                ccc436fc41bf402f.jpg (72.8 kB)
                fb9d48b39c614c32.jpg (91.3 kB)
                ... and 9754 other files
                test_images/
            train/
                train/
            train_images/
                0/
                    b5bd0a0a2de05bb5.jpg (73.8 kB)
                    c242bcf0719f9d61.jpg (71.9 kB)
                    ... and 18211 other files
                1/
                    a7ad6a44813b77c8.jpg (81.1 kB)
                    9b89db65b496490d.jpg (630.0 kB)
                    ... and 1116 other files
                ... and 87 other folders
        input/
            description.md (120 lines)
            sample_submission.csv (9757 lines)
            sample_submission.csv.zip (106.8 kB)
            test.zip (160 Bytes)
            test_images.zip (2.6 GB)
            train.csv (87799 lines)
            train.csv.zip (1.9 MB)
            train.zip (162 Bytes)
            train_images.zip (23.5 GB)
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
            test/
                test/
                    test/
            test_images/
                ccc436fc41bf402f.jpg (72.8 kB)
                fb9d48b39c614c32.jpg (91.3 kB)
                ... and 9754 other files
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
            train/
                train/
                    train/
            train_images/
                0/
                    b5bd0a0a2de05bb5.jpg (73.8 kB)
                    c242bcf0719f9d61.jpg (71.9 kB)
                    ... and 18211 other files
                1/
                    a7ad6a44813b77c8.jpg (81.1 kB)
                    9b89db65b496490d.jpg (630.0 kB)
                    ... and 1116 other files
                ... and 87 other folders
        working/
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
```

-> data/hotel-id-2021-fgvc8/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> data/hotel-id-2021-fgvc8/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> data/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> data/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> input/hotel-id-2021-fgvc8/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> input/hotel-id-2021-fgvc8/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing

print("TF:", tf.__version__)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)




## === cell 1
def intialize_accel(hardware):
    """
    input:
    str: GPU or TPU for hardware accelerator

    output:
    strategy -- used later for model definition and fitting
    """
    if hardware == "TPU":
        try:
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            strategy = tf.distribute.TPUStrategy(resolver)
            print("TPU Initialized")
            print("TPU Units:", strategy.num_replicas_in_sync)
            return strategy
        except Exception as e:
            print("TPU Initialization Failed:", repr(e))
            print("Falling back to default strategy.")
            return tf.distribute.get_strategy()

    elif hardware == "GPU":
        print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
        try:
            strategy = tf.distribute.MirroredStrategy()
        except Exception as e:
            print("MirroredStrategy init failed:", repr(e))
            strategy = tf.distribute.get_strategy()
        return strategy

    print("Unknown hardware option; using default strategy.")
    return tf.distribute.get_strategy()


strategy = intialize_accel("TPU")
AUTO = tf.data.AUTOTUNE



## === cell 2
DIR = "../input/hotel-id-2021-fgvc8"

Train_PATH = os.path.join(DIR, "train_images")
Test_PATH = os.path.join(DIR, "test_images")

train_csv_path = os.path.join(DIR, "train.csv")
sample_sub_path = os.path.join(DIR, "sample_submission.csv")

train_df = pd.read_csv(train_csv_path)
train_df = train_df.drop_duplicates(subset=["image"]).reset_index(drop=True)

print("Number of unique hotel chains: ", train_df.chain.nunique())
print("Number of unique hotels: ", train_df.hotel_id.nunique())
print("Number of Training Samples: ", train_df.shape[0])
print(train_df.head())



## === cell 3
Classes = train_df.hotel_id.nunique()
Channels = 3

size = (200, 200)
Split = int(0.9 * train_df.shape[0])



## === cell 4
le = preprocessing.LabelEncoder()
train_df["label"] = le.fit_transform(train_df["hotel_id"])

train_df["image_path"] = train_df.apply(
    lambda r: os.path.join(Train_PATH, str(r["chain"]), r["image"]), axis=1
)

exists_mask = train_df["image_path"].map(os.path.exists)
missing = (~exists_mask).sum()
if missing:
    print(f"Warning: {missing} train image paths missing; dropping them.")
train_df = train_df.loc[exists_mask].reset_index(drop=True)

print(train_df[["hotel_id", "label", "image_path"]].head())




## === cell 5
def image_proces(path, labels):
    """
    Reads, decodes, resizes and scales image to [0,1].
    """
    data = tf.io.read_file(path)
    data = tf.image.decode_jpeg(data, channels=3)
    data = tf.image.resize(data, size)
    data = tf.cast(data, tf.float32) / 255.0
    return data, labels


def import_image(paths, labels):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(image_proces, num_parallel_calls=AUTO)
    return dataset


def data_augment(image, labels):
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image, labels


Paths_Test = tf.io.gfile.glob(os.path.join(Test_PATH, "*.jpg"))
print("Found test jpgs:", len(Paths_Test))

dataset_Test = import_image(Paths_Test, np.arange(len(Paths_Test)))



## === cell 6
print("Number of Test Samples:", dataset_Test.cardinality().numpy())



## === cell 7
paths = train_df["image_path"].values
labels = train_df["label"].values.astype(np.int32)

train_paths, val_paths = paths[:Split], paths[Split:]
train_labels, val_labels = labels[:Split], labels[Split:]

BATCH_SIZE = 32

train_dataset = import_image(train_paths, train_labels)
train_dataset = train_dataset.map(data_augment, num_parallel_calls=AUTO)
train_dataset = train_dataset.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(AUTO)

val_dataset = import_image(val_paths, val_labels)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(AUTO)

print("Train batches:", tf.data.experimental.cardinality(train_dataset).numpy())
print("Val batches:", tf.data.experimental.cardinality(val_dataset).numpy())




## === cell 8
def create_model(Base, input_shape):
    inputs = tf.keras.Input(shape=tuple(input_shape))

    norm = tf.keras.layers.Normalization()
    x = norm(inputs)

    x = Base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(50, activation="relu", dtype="float32")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    outputs = tf.keras.layers.Dense(Classes, activation="softmax", dtype="float32")(x)

    model = tf.keras.Model(inputs, outputs)
    model._norm_layer = norm  # keep reference for later adapt
    return model




## === cell 9
def compile_model(model, lr):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]

    model.compile(
        optimizer=optimizer, loss=loss, metrics=metrics, steps_per_execution=8
    )
    return model




## === cell 10
EPOCHS = 1
VERBOSE = 1

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_accuracy",
    factor=0.1,
    patience=3,
    mode="max",
    min_delta=0.0001,
    verbose=1,
)

checkpoint_filepath = "./best_model.weights.h5"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1,
)

callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=10, mode="max", min_delta=0.0001, verbose=1
)



## === cell 11
input_shape = [200, 200, Channels]

with strategy.scope():
    Base = tf.keras.applications.ResNet50(
        weights=None, include_top=False, input_shape=tuple(input_shape)
    )
    Base.trainable = False
    model = create_model(Base, input_shape)
    model = compile_model(model, lr=0.001)

try:
    adapt_ds = train_dataset.unbatch().map(lambda x, y: x).take(512)
    model._norm_layer.adapt(adapt_ds)
    print("Normalization layer adapted.")
except Exception as e:
    print("Normalization adapt skipped due to:", repr(e))

print("Fitting")
History = model.fit(
    train_dataset,
    epochs=EPOCHS,
    callbacks=[reduce_lr, model_checkpoint_callback, callback],
    validation_data=val_dataset,
    verbose=VERBOSE,
)



## === cell 12
if os.path.exists(checkpoint_filepath):
    model.load_weights(checkpoint_filepath)
    print("Loaded best weights from:", checkpoint_filepath)
else:
    print("Best weights file not found; using last epoch weights.")

best_model = model



## === cell 13
dataset_Test_batched = dataset_Test.batch(32).prefetch(AUTO)
predictions = best_model.predict(dataset_Test_batched, verbose=1)
print("Pred shape:", predictions.shape)



## === cell 14
topk = 5
topk_idx = np.argsort(-predictions, axis=1)[:, :topk]  # descending
topk_hotel_ids = le.inverse_transform(topk_idx.reshape(-1)).reshape(-1, topk)

images_names = [os.path.basename(p) for p in Paths_Test]
pred_strings = [" ".join(map(str, row)) for row in topk_hotel_ids]

submission = pd.DataFrame({"image": images_names, "hotel_id": pred_strings})

if os.path.exists(sample_sub_path):
    sample = pd.read_csv(sample_sub_path)
    submission = sample[["image"]].merge(submission, on="image", how="left")
    if submission["hotel_id"].isna().any():
        fallback = " ".join(
            map(str, le.inverse_transform(np.arange(min(topk, Classes))))
        )
        submission["hotel_id"] = submission["hotel_id"].fillna(fallback)

print(submission.head())
print("Submission rows:", len(submission))



## === cell 15
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv")
