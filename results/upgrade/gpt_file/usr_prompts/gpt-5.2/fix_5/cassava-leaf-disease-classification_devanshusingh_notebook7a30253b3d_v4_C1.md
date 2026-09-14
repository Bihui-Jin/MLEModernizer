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

0.053

# 6. Current score

0.11958

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11958) has done: 'The timeout is driven mainly by per-image Python I/O and `model.predict()` calls inside a loop (2676 separate predicts) plus relatively slow CPU image augmentation/feeding during training. To fix this without changing the model or training semantics, I batch the test inference using a `tf.data` pipeline (vectorized decode/resize/rescale + batched `model.predict`) and I speed up training input by enabling generator shuffling (no semantic change), using `workers`/`use_multiprocessing` for background preprocessing, and avoiding unnecessary plotting/prediction work. I also switch protobuf to the faster default implementation (the current forced “python” implementation is slower) while keeping determinism controls so results remain stable aside from negligible float noise.'
- What this solution (achieved 0.11958) has done: 'The timeout is most likely coming from slow Python-side image loading/augmentation in `ImageDataGenerator.flow_from_dataframe` and from not using worker multiprocessing; the model itself is small and prediction is already using a fast `tf.data` pipeline. I keep the exact same model and training loop semantics, but speed up input throughput by enabling multi-worker prefetching for `.fit()` and by ensuring we don’t do any unnecessary generator work. I also make `steps_per_epoch`/`validation_steps` cover the full dataset (same semantics as “one pass”), and add a deterministic tf.data-based training pipeline only if the pretrained model is missing is **not allowed** (would change core approach), so we stay with the generator but optimize its execution. Finally, I keep determinism and paths unchanged.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SEED = 42
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
label_json_path = (
    "../input/cassava-leaf-disease-classification/label_num_to_disease_map.json"
)
images_dir_path = "../input/cassava-leaf-disease-classification/train_images"




## === cell 2
train_csv = pd.read_csv(train_csv_path)
train_csv["label"] = train_csv["label"].astype("string")

label_class = pd.read_json(label_json_path, orient="index")
label_class = label_class.values.flatten().tolist()




## === cell 3
print(train_csv.head())
print(train_csv.shape)




## === cell 4
train_csv = train_csv.reset_index(drop=True)




## === cell 5
print(train_csv.dtypes)




## === cell 6
print("Label names :")
for i, label in enumerate(label_class):
    print(f" {i}. {label}")




## === cell 7
BATCH_SIZE = 50
IMG_SIZE = 200




## === cell 8
train_gen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    rescale=1 / 255.0,
    validation_split=0.2,
)

valid_gen = ImageDataGenerator(
    rescale=1 / 255.0,
    validation_split=0.2,
)




## === cell 9
train_generator = train_gen.flow_from_dataframe(
    dataframe=train_csv,
    directory=images_dir_path,
    x_col="image_id",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    subset="training",
)

valid_generator = valid_gen.flow_from_dataframe(
    dataframe=train_csv,
    directory=images_dir_path,
    x_col="image_id",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=False,
    subset="validation",
)




## === cell 10
SHOW_BATCH = False

if SHOW_BATCH:
    batch = next(train_generator)
    images = batch[0]
    labels = batch[1]

    plt.figure(figsize=(12, 9))
    for i, (img, label) in enumerate(zip(images, labels)):
        plt.subplot(2, 3, i % 6 + 1)
        plt.axis("off")
        plt.imshow(img)
        plt.title(label_class[int(np.argmax(label))])
        if i == 15:
            break




## === cell 11
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Activation("softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])




## === cell 12
def scheduler(epoch, lr):
    if epoch > 6 and epoch % 2 == 0:
        lr = lr / 1.5
        return lr
    else:
        return lr


callback0 = tf.keras.callbacks.ModelCheckpoint(
    "./CasavaLeafDiseaseModel.h5", monitor="val_loss", save_best_only=True
)
callback1 = tf.keras.callbacks.LearningRateScheduler(scheduler)




## === cell 13
PRETRAINED_PATH = (
    "../input/casavaleafdiseasemodel-tf/CasavaLeafDiseaseModel_epoch_12_acc_85.h5"
)
loaded_pretrained = False
try:
    model = tf.keras.models.load_model(PRETRAINED_PATH)
    loaded_pretrained = True
    print("Loaded saved model:", PRETRAINED_PATH)
except Exception as e:
    print("No saved model. So Train the model !")
    print("Load error:", repr(e))




## === cell 14
his = None
if not loaded_pretrained:
    import math

    steps_per_epoch = max(1, int(math.ceil(train_generator.n / BATCH_SIZE)))
    validation_steps = max(1, int(math.ceil(valid_generator.n / BATCH_SIZE)))

    WORKERS = max(2, (os.cpu_count() or 2) // 2)

    his = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=10,
        validation_data=valid_generator,
        validation_steps=validation_steps,
        callbacks=[callback0, callback1],
        workers=WORKERS,
        use_multiprocessing=True,
        max_queue_size=WORKERS * 4,
        verbose=2,
    )




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3824295822.py in <cell line: 0>()
     13     WORKERS = max(2, (os.cpu_count() or 2) // 2)
     14 
---> 15     his = model.fit(
     16         train_generator,
     17         steps_per_epoch=steps_per_epoch,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    117             return fn(*args, **kwargs)
    118         except Exception as e:
--> 119             filtered_tb = _process_traceback_frames(e.__traceback__)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`

TypeError: TensorFlowTrainer.fit() got an unexpected keyword argument 'workers'

## === cell 15
RUN_VALID_PRED_DEBUG = False
if RUN_VALID_PRED_DEBUG:
    print(model.predict(next(valid_generator)[0]))




## === cell 16
if his is not None:
    stats = pd.DataFrame(his.history)
    print(stats.tail())
else:
    print("Training skipped (pretrained model loaded).")




## === cell 17
SHOW_TEST_EXAMPLE = False

test_dir = "../input/cassava-leaf-disease-classification/test_images"
if SHOW_TEST_EXAMPLE:
    test_images = sorted(
        [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
    )
    test_img_path = os.path.join(test_dir, test_images[0]) if test_images else None
    print("Example test image:", test_img_path)

    if test_img_path is not None:
        img = cv2.imread(test_img_path)
        if img is not None:
            resized_img = (
                cv2.resize(img, (IMG_SIZE, IMG_SIZE)).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
                / 255.0
            )
            plt.figure(figsize=(8, 4))
            plt.title("TEST IMAGE")
            plt.imshow(resized_img[0][:, :, ::-1])  # BGR->RGB for display
            plt.axis("off")
        else:
            print("cv2.imread failed for:", test_img_path)




## === cell 18
ss = pd.read_csv("../input/cassava-leaf-disease-classification/sample_submission.csv")

test_paths = (
    ("../input/cassava-leaf-disease-classification/test_images/") + ss["image_id"]
).to_numpy()

AUTOTUNE = tf.data.AUTOTUNE


def _load_and_preprocess(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize_with_pad(
        img, IMG_SIZE, IMG_SIZE, method="bilinear", antialias=False
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


ds = tf.data.Dataset.from_tensor_slices(test_paths)
opts = tf.data.Options()
opts.experimental_deterministic = True
ds = ds.with_options(opts)
ds = ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE)
ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

probs = model.predict(ds, verbose=0)
preds = np.argmax(probs, axis=1).astype(int).tolist()

my_submission = pd.DataFrame({"image_id": ss["image_id"].values, "label": preds})
my_submission.to_csv("submission.csv", index=False)




## === cell 19
print("Submission File: \n---------------\n")
print(my_submission.head())
print("\nWrote:", os.path.abspath("submission.csv"), "rows:", len(my_submission))
