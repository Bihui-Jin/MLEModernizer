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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.6644001208824418

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.10762) has done: 'The update speeds up data loading by enabling multiprocessing workers during model fitting, which allows image preprocessing to run in parallel and reduces overall training time while keeping the model architecture, training epochs, and evaluation unchanged.'
- What this solution (achieved 0.45852) has done: 'The changes replace the slower `ImageDataGenerator` pipelines with efficient `tf.data` pipelines that parallelize image loading, decoding, resizing, and augmentation, while keeping the same model architecture, training schedule, and augmentation logic. The new pipelines use `tf.keras` preprocessing layers, deterministic shuffling with the original seed, caching, and prefetching, which dramatically cuts I/O and preprocessing overhead and brings total runtime under the 600‑second limit without affecting prediction accuracy. All other logic, including model definition, freezing/unfreezing, and submission creation, remains unchanged.'

# 9. Code solution

## === cell 0
def _apply_options(ds):
    opt = tf.data.Options()
    opt.experimental_deterministic = False  # enable nondeterministic order for speed
    return ds.with_options(opt)


train_split, val_split = train_test_split(
    train_df,
    test_size=0.2,
    random_state=SEED,
    stratify=train_df["label"],
)

augmentation_layer = tf.keras.Sequential(
    [
        layers.RandomFlip(mode="horizontal", seed=SEED),
        layers.RandomRotation(factor=0.111, seed=SEED),  # ~20 degrees
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1, seed=SEED),
        layers.RandomZoom(height_factor=0.1, width_factor=0.1, seed=SEED),
    ]
)


def decode_image(path):
    """Read and resize image; returns a float32 tensor in [0,1]."""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def augment_image(image, label):
    """Apply stochastic augmentations (only for training)."""
    image = augmentation_layer(image, training=True)
    return image, label


def prepare_train_dataset(paths, labels):
    """Training pipeline: decode → disk cache → shuffle → augment → batch → prefetch."""
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, l: (decode_image(p), l), num_parallel_calls=AUTOTUNE)
    ds = ds.cache("/tmp/train_cache")  # keep disk cache to avoid RAM overflow
    ds = ds.shuffle(buffer_size=1000, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(augment_image, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    ds = _apply_options(ds)
    return ds


def prepare_val_dataset(paths, labels):
    """Validation pipeline: decode → in‑memory cache → batch → prefetch."""
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, l: (decode_image(p), l), num_parallel_calls=AUTOTUNE)
    ds = ds.cache()  # small enough for RAM, faster than disk
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    ds = _apply_options(ds)
    return ds


train_ds = prepare_train_dataset(
    train_split["filepath"].values, train_split["label"].values
)

val_ds = prepare_val_dataset(val_split["filepath"].values, val_split["label"].values)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/265690913.py in <cell line: 0>()
      7 
      8 
----> 9 train_split, val_split = train_test_split(
     10     train_df,
     11     test_size=0.2,

NameError: name 'train_test_split' is not defined

## === cell 1
from tensorflow.keras.applications import MobileNetV2

base_model = MobileNetV2(
    weights="imagenet", include_top=False, input_shape=IMG_SIZE + (3,)
)
base_model.trainable = False  # freeze initially

x = base_model.output
x = GlobalAveragePooling2D()(x)
output = Dense(5, activation="softmax")(x)  # five disease classes

model = Model(inputs=base_model.input, outputs=output)
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

EPOCHS_INITIAL = 5
model.fit(
    train_ds,
    epochs=EPOCHS_INITIAL,
    validation_data=val_ds,
    verbose=1,
)

base_model.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),  # lower LR for fine‑tuning
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

EPOCHS_FINE = 7
model.fit(
    train_ds,
    epochs=EPOCHS_FINE,
    validation_data=val_ds,
    verbose=1,
)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
test_images = glob.glob(os.path.join(BASE_PATH, "test_images", "*.jpg"))
df_test = pd.DataFrame(test_images, columns=["path"])


def preprocess_test(path):
    return decode_image(path)  # reuse same decode function


test_ds = (
    tf.data.Dataset.from_tensor_slices(df_test["path"].values)
    .map(preprocess_test, num_parallel_calls=AUTOTUNE)
    .batch(128)
    .prefetch(buffer_size=AUTOTUNE)
)
test_ds = _apply_options(test_ds)

pred_test = model.predict(test_ds, verbose=1)
pred_test_labels = np.argmax(pred_test, axis=-1)

final_submission = df_test.copy()
final_submission["image_id"] = final_submission["path"].str.split("/").str[-1]
final_submission["label"] = pred_test_labels
submission_df = final_submission[["image_id", "label"]]
submission_df.to_csv("submission.csv", index=False)

print("Submission file saved as submission.csv")

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2802411223.py in <cell line: 0>()
      1 # Streamline the test pipeline with the same non‑deterministic options.
----> 2 test_images = glob.glob(os.path.join(BASE_PATH, "test_images", "*.jpg"))
      3 df_test = pd.DataFrame(test_images, columns=["path"])
      4 
      5 

NameError: name 'glob' is not defined
