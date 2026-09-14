# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from PIL import Image

print("TF version:", tf.__version__)

np.random.seed(42)
tf.random.set_seed(42)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

tf.config.run_functions_eagerly(False)

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass



## === cell 1
training_mode = False
previously_trained_model_file = "../input/best-model/best_model.h5"
answer_to_life = 42
first_time_train = False

if (not training_mode) and (not os.path.exists(previously_trained_model_file)):
    if os.path.exists("best_model_final.h5"):
        previously_trained_model_file = "best_model_final.h5"
        print(
            "External pretrained model not found; using local:",
            previously_trained_model_file,
        )
    elif os.path.exists("best_model.h5"):
        previously_trained_model_file = "best_model.h5"
        print(
            "External pretrained model not found; using local:",
            previously_trained_model_file,
        )
    else:
        print(
            f"Pretrained model not found at {previously_trained_model_file}. Switching to training_mode=True."
        )
        training_mode = True
        first_time_train = True




## === cell 2
def resolve_project_folder():
    candidates = [
        "../input/cassava-leaf-disease-classification",
        "/kaggle/input/cassava-leaf-disease-classification",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train.csv")):
            return c
    for root in ["../input", "/kaggle/input"]:
        if os.path.isdir(root):
            for dirpath, dirnames, filenames in os.walk(root):
                if "train.csv" in filenames and "sample_submission.csv" in filenames:
                    return dirpath
    raise FileNotFoundError(
        "Could not locate cassava-leaf-disease-classification dataset folder."
    )


project_folder = resolve_project_folder()
print("Using project_folder:", project_folder)

df = None
if training_mode:
    df = pd.read_csv(f"{project_folder}/train.csv")
    df["image_path"] = df["image_id"].apply(
        lambda x: f"{project_folder}/train_images/{x}"
    )



## === cell 3
if False:
    df.head()



## === cell 4
if False:
    image1 = Image.open(df["image_path"].tolist()[0])
    image1




## === cell 5
def get_random_crops(img):
    imgs = []
    img = tf.convert_to_tensor(img)
    for i in range(5):
        cropped_img = tf.image.random_crop(img, size=[512, 512, 3])
        imgs.append(cropped_img)
    return imgs


if False:
    images = get_random_crops(np.array(image1))




## === cell 6
def get_augmented_images(img):
    img = np.array(img)
    imgs = []
    cropped_images = get_random_crops(img)
    for cropped_img in cropped_images:
        horizontal_flip = np.random.rand() > 0.5
        vertical_flip = np.random.rand() > 0.5

        aug_img = tf.keras.preprocessing.image.random_shear(cropped_img, 0.20)
        if horizontal_flip:
            aug_img = tf.image.flip_left_right(aug_img)
        if vertical_flip:
            aug_img = tf.image.flip_up_down(aug_img)

        imgs.append(aug_img)
    return imgs




## === cell 7
if False:
    images = get_augmented_images(image1)
    tmp_img = image1.resize((512, 512))
    tmp_img = np.array(tmp_img)
    images.append(tmp_img)



## === cell 8
if False:
    np.array(images).shape



## === cell 9
if False:
    import matplotlib.pyplot as plt

    f, axarr = plt.subplots(1, 6, figsize=(20, 20))
    for i in range(6):
        axarr[i].imshow(images[i])
    plt.show()



## === cell 10
datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    horizontal_flip=True,
    shear_range=0.2,
    rotation_range=25,
    channel_shift_range=0.2,
    zoom_range=0.2,
    height_shift_range=0.2,
    vertical_flip=True,
    validation_split=0.2,
)



## === cell 11
img_size = 512
train_datagen = None
if training_mode:
    df["label"] = df["label"].astype(str)
    train_datagen = datagen.flow_from_dataframe(
        df,
        x_col="image_path",
        y_col="label",
        batch_size=16,
        class_mode="categorical",
        target_size=(img_size, img_size),
        seed=answer_to_life,
        subset="training",
    )



## === cell 12
val_datagen = None
if training_mode:
    val_datagen = datagen.flow_from_dataframe(
        df,
        x_col="image_path",
        y_col="label",
        batch_size=16,
        class_mode="categorical",
        target_size=(img_size, img_size),
        seed=answer_to_life,
        subset="validation",
    )



## === cell 13
CosineDecay = tf.keras.optimizers.schedules.CosineDecay

if training_mode:

    def create_pretrained_model():
        custom_weights_path = "../input/efficientnet-model-file/efficientnetb3_notop.h5"
        if os.path.exists(custom_weights_path):
            weights_arg = custom_weights_path
            print("Loading EfficientNetB3 weights from:", custom_weights_path)
        else:
            weights_arg = "imagenet"
            print(
                "Custom EfficientNet weights not found; falling back to weights='imagenet'."
            )

        pretrained_model = tf.keras.applications.EfficientNetB3(
            weights=weights_arg,
            include_top=False,
            input_shape=(img_size, img_size, 3),
        )
        x = layers.GlobalAveragePooling2D()(pretrained_model.output)

        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(5, activation="softmax")(x)

        model_local = tf.keras.models.Model(inputs=pretrained_model.input, outputs=x)

        decay_steps = int(round(17118.0 / 16.0)) * 3
        cosine_decay = CosineDecay(
            initial_learning_rate=1e-4, decay_steps=decay_steps, alpha=0.3
        )

        callbacks_local = [
            ModelCheckpoint(
                filepath="best_model.h5", monitor="val_loss", save_best_only=True
            )
        ]

        model_local.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=cosine_decay),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model_local, callbacks_local

    model, callbacks = create_pretrained_model()
    callbacks = [
        ModelCheckpoint(
            filepath="best_model_final.h5", monitor="val_loss", save_best_only=True
        )
    ]



## === cell 14
if not training_mode:
    model = tf.keras.models.load_model(previously_trained_model_file)

if training_mode:
    history = model.fit(
        train_datagen, epochs=6, validation_data=val_datagen, callbacks=callbacks
    )



## === cell 15
if training_mode:
    if os.path.exists("best_model_final.h5"):
        model = tf.keras.models.load_model("best_model_final.h5")
    elif os.path.exists("best_model.h5"):
        model = tf.keras.models.load_model("best_model.h5")



## === cell 16
_ = model.summary()



## === cell 17
if False:
    test_image_path = df["image_path"].tolist()[4]
    img = Image.open(test_image_path)
    img = img.resize((512, 512))



## === cell 18
if False:
    img = np.array(img)



## === cell 19
if False:
    img = img / 255.0



## === cell 20
if False:
    imgs = get_random_crops(img)
    imgs = np.array(imgs)



## === cell 21
if False:
    preds = model.predict(imgs, verbose=0)



## === cell 22
if False:
    preds = np.argmax(np.sum(preds, axis=0))



## === cell 23
if False:
    preds



## === cell 24
if False:
    df.head()



## === cell 25
test_folder = f"{project_folder}/test_images"



## === cell 26
sample_path = f"{project_folder}/sample_submission.csv"
sample_submission = pd.read_csv(sample_path)
test_files = sample_submission["image_id"].tolist()
print("Num test files (from sample_submission):", len(test_files))
assert len(test_files) == sample_submission.shape[0]



## === cell 27
if False:
    test_files[:5]



## === cell 28
AUTOTUNE = tf.data.AUTOTUNE
BATCH_ORIG = 16  # batch size over original images (each produces 6 views)

BASE_SEED = tf.constant([42, 12345], dtype=tf.int32)


@tf.function
def _decode_and_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    return img


@tf.function
def _shear_x_affine_batch(imgs, shear=0.20):
    a0 = tf.constant(1.0, tf.float32)
    a1 = tf.constant(shear, tf.float32)
    a2 = tf.constant(0.0, tf.float32)
    b0 = tf.constant(0.0, tf.float32)
    b1 = tf.constant(1.0, tf.float32)
    b2 = tf.constant(0.0, tf.float32)
    c0 = tf.constant(0.0, tf.float32)
    c1 = tf.constant(0.0, tf.float32)
    transform = tf.stack([a0, a1, a2, b0, b1, b2, c0, c1])[tf.newaxis, :]  # [1,8]
    transforms = tf.repeat(transform, repeats=tf.shape(imgs)[0], axis=0)  # [N,8]
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=imgs,
        transforms=transforms,
        output_shape=tf.shape(imgs)[1:3],
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )
    return out


@tf.function
def _make_tta_six_stateless(img, idx):
    img_big = tf.image.resize(img, [img_size, img_size], method="bilinear")
    img_big = tf.clip_by_value(img_big, 0.0, 1.0)

    k = tf.range(5, dtype=tf.int32)  # [5]
    seeds_k = tf.stack(
        [tf.fill([5], BASE_SEED[0] + tf.cast(idx, tf.int32)), BASE_SEED[1] + k],
        axis=1,
    )  # [5,2]

    crops5 = tf.image.stateless_random_crop(
        img_big, size=[5, img_size, img_size, 3], seed=seeds_k
    )  # [5,H,W,3]

    aug5 = _shear_x_affine_batch(crops5, shear=0.20)

    seeds_h = seeds_k + tf.constant([11, 101], tf.int32)
    seeds_v = seeds_k + tf.constant([17, 303], tf.int32)
    h = tf.random.stateless_uniform([5], seed=seeds_h, dtype=tf.float32) > 0.5
    v = tf.random.stateless_uniform([5], seed=seeds_v, dtype=tf.float32) > 0.5

    aug5 = tf.where(h[:, None, None, None], tf.image.flip_left_right(aug5), aug5)
    aug5 = tf.where(v[:, None, None, None], tf.image.flip_up_down(aug5), aug5)

    aug5 = tf.clip_by_value(aug5, 0.0, 1.0)

    views6 = tf.concat([aug5, img_big[None, ...]], axis=0)  # [6,H,W,3]
    return views6


paths = [os.path.join(test_folder, f) for f in test_files]
ds = tf.data.Dataset.from_tensor_slices(paths)

options = tf.data.Options()
options.experimental_deterministic = True
ds = ds.with_options(options)

ds = ds.enumerate()


def _process(i, p):
    img = _decode_and_resize(p)
    views6 = _make_tta_six_stateless(img, i)
    return views6


ds = ds.map(_process, num_parallel_calls=AUTOTUNE).cache()
ds = ds.batch(BATCH_ORIG, drop_remainder=False)


@tf.function
def _flatten_batch(x):
    shp = tf.shape(x)  # [B,6,H,W,3]
    return tf.reshape(x, [shp[0] * shp[1], shp[2], shp[3], shp[4]])


ds_flat = ds.map(_flatten_batch, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

preds_aug = model.predict(ds_flat, verbose=0)
preds_aug = preds_aug.reshape(len(test_files), 6, -1)
preds_sum = preds_aug.sum(axis=1)
predictions = preds_sum.argmax(axis=1).astype(int).tolist()
image_names = test_files

submission_df = pd.DataFrame({"image_id": image_names, "label": predictions})



## === cell 29
print(submission_df.shape)
print(submission_df.columns.tolist())
print("Matches sample rows:", len(submission_df) == len(sample_submission))
assert (
    submission_df.shape[0] == sample_submission.shape[0]
), "Submission length mismatch."
assert (
    submission_df["image_id"].tolist() == sample_submission["image_id"].tolist()
), "Image order mismatch."



## === cell 30
submission_df.head()



## === cell 31
submission_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with rows:", len(submission_df))
print(submission_df.dtypes)
