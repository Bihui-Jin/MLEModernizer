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

0.752190994258084

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.13864) has done: 'I replace the failing keras imports with tensorflow.keras, drop the nonexistent model loading, add a simple training pipeline using a pretrained MobileNetV2 model, and keep the original test‑generation logic. This fixes the import and file‑not‑found errors, creates a valid model that can be trained quickly, and finally writes a proper submission.csv with the required columns.'
- What this solution (achieved 0.09791) has done: 'The changes replace the Python‑level ImageDataGenerator pipelines with efficient tf.data datasets and a Keras preprocessing layer for augmentation. This removes costly per‑image Python overhead, enables GPU‑accelerated augmentations, adds caching and prefetching, and keeps the same model architecture, loss, optimizer, and training loop, preserving accuracy while fitting well inside the 600‑second limit.'

# 9. Code solution

## === cell 0

BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification"
if not os.path.isdir(BASE_PATH):
    BASE_PATH = "../input/cassava-leaf-disease-classification"

TRAIN_IMG_PATH = os.path.join(BASE_PATH, "train_images")
TEST_IMG_PATH = os.path.join(BASE_PATH, "test_images")
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")

df_train = pd.read_csv(TRAIN_CSV)
df_train["path"] = df_train["image_id"].apply(lambda x: os.path.join(TRAIN_IMG_PATH, x))
df_train["label"] = df_train["label"].astype(str)

df_tr, df_val = train_test_split(
    df_train,
    test_size=0.1,
    stratify=df_train["label"],
    random_state=42,
)

IMG_SIZE = 224
SIZE = (IMG_SIZE, IMG_SIZE)

BATCH_SIZE = 256

AUTOTUNE = tf.data.AUTOTUNE

tf.config.threading.set_intra_op_parallelism_threads(
    tf.config.threading.get_physical_device_count("CPU")
)
tf.config.threading.set_inter_op_parallelism_threads(
    tf.config.threading.get_physical_device_count("CPU")
)


def decode_and_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, SIZE)  # default bilinear; unchanged semantics
    return img


def preprocess_train(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    if tfa is not None:
        angle = tf.random.uniform([], -20.0, 20.0) * (3.14159265 / 180.0)
        img = tfa.image.rotate(img, angle, fill_mode="nearest")
    img = img / 255.0
    label = tf.one_hot(tf.cast(label, tf.int32), depth=5)
    return img, label


def preprocess_val(img, label):
    img = img / 255.0
    label = tf.one_hot(tf.cast(label, tf.int32), depth=5)
    return img, label


def preprocess_test(img):
    img = img / 255.0
    return img


label_to_int = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4}
df_tr["label_int"] = df_tr["label"].map(label_to_int)
df_val["label_int"] = df_val["label"].map(label_to_int)

train_base = tf.data.Dataset.from_tensor_slices(
    (df_tr["path"].values, df_tr["label_int"].values)
)
train_base = train_base.map(
    lambda p, l: (decode_and_resize(p), l), num_parallel_calls=AUTOTUNE
).cache()
train_ds = train_base.shuffle(buffer_size=1000, seed=42)
train_ds = train_ds.map(preprocess_train, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

val_base = tf.data.Dataset.from_tensor_slices(
    (df_val["path"].values, df_val["label_int"].values)
)
val_base = val_base.map(
    lambda p, l: (decode_and_resize(p), l), num_parallel_calls=AUTOTUNE
).cache()
val_ds = val_base.map(preprocess_val, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1028974828.py in <cell line: 0>()
      5 
      6 BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification"
----> 7 if not os.path.isdir(BASE_PATH):
      8     BASE_PATH = "../input/cassava-leaf-disease-classification"
      9 

NameError: name 'os' is not defined

## === cell 1
base_model = MobileNetV2(input_shape=SIZE + (3,), include_top=False, weights="imagenet")
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
output = Dense(5, activation="softmax", dtype="float32")(x)  # keep logits in float32
model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

early_stop = EarlyStopping(
    monitor="val_accuracy", patience=2, restore_best_weights=True, mode="max"
)

model.fit(
    train_ds,
    epochs=10,
    validation_data=val_ds,
    callbacks=[early_stop],
    verbose=1,
)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2876352430.py in <cell line: 0>()
----> 1 base_model = MobileNetV2(input_shape=SIZE + (3,), include_top=False, weights="imagenet")
      2 base_model.trainable = False
      3 
      4 x = GlobalAveragePooling2D()(base_model.output)
      5 output = Dense(5, activation="softmax", dtype="float32")(x)  # keep logits in float32

NameError: name 'MobileNetV2' is not defined

## === cell 2
test_images = glob.glob(os.path.join(TEST_IMG_PATH, "*.jpg"))
df_test = pd.DataFrame(test_images, columns=["path"])

test_ds = tf.data.Dataset.from_tensor_slices(df_test["path"].values)
test_ds = test_ds.map(
    lambda p: decode_and_resize(p), num_parallel_calls=AUTOTUNE
).cache()
test_ds = test_ds.map(preprocess_test, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

pred_test = model.predict(test_ds, verbose=1)
pred_test_labels = np.argmax(pred_test, axis=1)

final_submission = df_test.copy()
final_submission["image_id"] = final_submission["path"].apply(
    lambda p: os.path.basename(p)
)
final_submission["label"] = pred_test_labels
submission_csv = final_submission[["image_id", "label"]]

submission_path = "submission.csv"
submission_csv.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3468973792.py in <cell line: 0>()
----> 1 test_images = glob.glob(os.path.join(TEST_IMG_PATH, "*.jpg"))
      2 df_test = pd.DataFrame(test_images, columns=["path"])
      3 
      4 test_ds = tf.data.Dataset.from_tensor_slices(df_test["path"].values)
      5 test_ds = test_ds.map(

NameError: name 'glob' is not defined
