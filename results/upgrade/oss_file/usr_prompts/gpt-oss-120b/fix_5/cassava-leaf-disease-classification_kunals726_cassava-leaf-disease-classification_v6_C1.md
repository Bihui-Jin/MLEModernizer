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

0.8768510123904503

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.layers as layers
import pandas as pd
from sklearn.model_selection import train_test_split

from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.callbacks import ModelCheckpoint

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(4)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
training_mode = False
previously_trained_model_file = "../input/best-model/best_model.h5"
first_time_train = True
answer_to_life = 42



## === cell 2
project_folder = "../input/cassava-leaf-disease-classification"
df = pd.read_csv(f"{project_folder}/train.csv")
df["image_path"] = df["image_id"].apply(lambda x: f"{project_folder}/train_images/{x}")



## === cell 3
df.head()



## === cell 4
from PIL import Image

image1 = Image.open(df["image_path"].tolist()[0])



## === cell 5
import numpy as np


def get_random_crops(img):
    tf_img = tf.convert_to_tensor(img, dtype=tf.float32)
    imgs = []
    for _ in range(5):
        cropped = tf.image.random_crop(tf_img, size=[512, 512, 3])
        imgs.append(cropped.numpy())
    return imgs


images = get_random_crops(np.array(image1))




## === cell 6
def get_augmented_images(img):
    img_np = np.array(img)
    aug_imgs = []
    cropped_images = get_random_crops(img_np)
    for cropped_img in cropped_images:
        if np.random.rand() > 0.5:
            cropped_img = tf.image.flip_left_right(cropped_img).numpy()
        if np.random.rand() > 0.5:
            cropped_img = tf.image.flip_up_down(cropped_img).numpy()
        aug_imgs.append(cropped_img)
    return aug_imgs




## === cell 7
images = get_augmented_images(image1)
tmp_img = image1.resize((512, 512))
tmp_img = np.array(tmp_img)
images.append(tmp_img)



## === cell 8
np.array(images).shape



## === cell 9
import matplotlib.pyplot as plt

f, axarr = plt.subplots(1, 6, figsize=(20, 20))
for i in range(6):
    axarr[i].imshow(images[i])



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
df["label"] = df["label"].astype(str)
img_size = 512
train_datagen = datagen.flow_from_dataframe(
    df,
    x_col="image_path",
    y_col="label",
    batch_size=16,
    class_mode="categorical",
    target_size=(img_size, img_size),
    seed=answer_to_life,
    subset="training",
    workers=4,  # parallel image loading
    use_multiprocessing=True,
)



## === cell 12
val_datagen = datagen.flow_from_dataframe(
    df,
    x_col="image_path",
    y_col="label",
    batch_size=16,
    class_mode="categorical",
    target_size=(img_size, img_size),
    seed=answer_to_life,
    subset="validation",
    workers=4,  # parallel image loading
    use_multiprocessing=True,
)




## === cell 13
def create_simple_cnn():
    inputs = tf.keras.Input(shape=(img_size, img_size, 3))
    x = layers.Conv2D(32, (3, 3), activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, (3, 3), activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(5, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    decay_steps = int(round(train_datagen.n / 16.0)) * 3
    cosine_decay = CosineDecay(
        initial_learning_rate=1e-4, decay_steps=decay_steps, alpha=0.3
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cosine_decay),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model




## === cell 14
if training_mode:
    model = create_simple_cnn()
    callbacks = [
        ModelCheckpoint(
            filepath="best_model.h5", monitor="val_loss", save_best_only=True, verbose=1
        )
    ]
else:
    model = tf.keras.models.load_model(previously_trained_model_file)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/292493003.py in <cell line: 0>()
      7     ]
      8 else:
----> 9     model = tf.keras.models.load_model(previously_trained_model_file)
     10 

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    194         )
    195     if str(filepath).endswith((".h5", ".hdf5")):
--> 196         return legacy_h5_format.load_model_from_hdf5(
    197             filepath, custom_objects=custom_objects, compile=compile
    198         )

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/saving/legacy_h5_format.py in load_model_from_hdf5(filepath, custom_objects, compile)
    114     opened_new_file = not isinstance(filepath, h5py.File)
    115     if opened_new_file:
--> 116         f = h5py.File(filepath, mode="r")
    117     else:
    118         f = filepath

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in __init__(self, name, mode, driver, libver, userblock_size, swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order, fs_strategy, fs_persist, fs_threshold, fs_page_size, page_buf_size, min_meta_keep, min_raw_keep, locking, alignment_threshold, alignment_interval, meta_block_size, **kwds)
    562                                  fs_persist=fs_persist, fs_threshold=fs_threshold,
    563                                  fs_page_size=fs_page_size)
--> 564                 fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)
    565 
    566             if isinstance(libver, tuple):

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in make_fid(name, mode, userblock_size, fapl, fcpl, swmr)
    236         if swmr and swmr_support:
    237             flags |= h5f.ACC_SWMR_READ
--> 238         fid = h5f.open(name, flags, fapl=fapl)
    239     elif mode == 'r+':
    240         fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/h5f.pyx in h5py.h5f.open()

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = '../input/best-model/best_model.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 15
if training_mode:
    history = model.fit(
        train_datagen,
        epochs=6,
        validation_data=val_datagen,
        callbacks=callbacks,
        verbose=2,
        workers=4,  # parallel data loading during fit
        use_multiprocessing=True,
    )
    model = tf.keras.models.load_model("best_model.h5")



## === cell 16
from PIL import Image



## === cell 17
test_folder = f"{project_folder}/test_images"



## === cell 18
import os

test_files = sorted(os.listdir(test_folder))



## === cell 19
batch_size = 8  # number of original images per prediction batch
image_names = []
predictions = []

for i in range(0, len(test_files), batch_size):
    batch_files = test_files[i : i + batch_size]
    batch_aug_images = []  # will hold all augmentations for the batch
    batch_image_names = []  # corresponding image identifiers

    for img_name in batch_files:
        img_path = os.path.join(test_folder, img_name)
        img = Image.open(img_path).convert("RGB")
        aug_imgs = get_augmented_images(img)  # 5 random crops + flips
        base_img = img.resize((img_size, img_size))
        base_img = np.array(base_img)
        aug_imgs.append(base_img)  # add the resized original
        batch_aug_images.extend(aug_imgs)  # 6 images per original
        batch_image_names.append(img_name)

    imgs_array = np.array(batch_aug_images).astype("float32") / 255.0
    preds = model.predict(imgs_array, batch_size=32, verbose=0)
    preds = preds.reshape(len(batch_files), 6, -1)
    avg_preds = np.mean(preds, axis=1)
    pred_labels = np.argmax(avg_preds, axis=1)

    image_names.extend(batch_image_names)
    predictions.extend(pred_labels.tolist())

submission_df = pd.DataFrame({"image_id": image_names, "label": predictions})



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/649657922.py in <cell line: 0>()
     19 
     20     imgs_array = np.array(batch_aug_images).astype("float32") / 255.0
---> 21     preds = model.predict(imgs_array, batch_size=32, verbose=0)
     22     preds = preds.reshape(len(batch_files), 6, -1)
     23     avg_preds = np.mean(preds, axis=1)

NameError: name 'model' is not defined

## === cell 20
submission_df.head()



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3942055985.py in <cell line: 0>()
----> 1 submission_df.head()
      2 

NameError: name 'submission_df' is not defined

## === cell 21
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Saved submission to {submission_path} with {len(submission_df)} rows.")

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2661912159.py in <cell line: 0>()
      1 submission_path = "submission.csv"
----> 2 submission_df.to_csv(submission_path, index=False)
      3 print(f"Saved submission to {submission_path} with {len(submission_df)} rows.")

NameError: name 'submission_df' is not defined
