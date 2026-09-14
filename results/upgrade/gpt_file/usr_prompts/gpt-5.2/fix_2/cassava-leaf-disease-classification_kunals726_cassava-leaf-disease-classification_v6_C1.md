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
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from PIL import Image


print("TF version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
training_mode = False
previously_trained_model_file = "../input/best-model/best_model.h5"
answer_to_life = 42
first_time_train = False

if (not training_mode) and (not os.path.exists(previously_trained_model_file)):
    print(
        f"Pretrained model not found at {previously_trained_model_file}. Switching to training_mode=True."
    )
    training_mode = True
    first_time_train = True



## === cell 2
project_folder = "../input/cassava-leaf-disease-classification"
df = pd.read_csv(f"{project_folder}/train.csv")
df["image_path"] = df["image_id"].apply(lambda x: f"{project_folder}/train_images/{x}")



## === cell 3
df.head()



## === cell 4
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


images = get_random_crops(np.array(image1))



## === cell 7
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




## === cell 8
images = get_augmented_images(image1)
tmp_img = image1.resize((512, 512))
tmp_img = np.array(tmp_img)
images.append(tmp_img)



## === cell 9
np.array(images).shape



## === cell 10
import matplotlib.pyplot as plt

f, axarr = plt.subplots(1, 6, figsize=(20, 20))
for i in range(6):
    axarr[i].imshow(images[i])
plt.show()



## === cell 12
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



## === cell 13
df["label"] = df["label"].astype(str)
img_size = 512

train_datagen = datagen.flow_from_dataframe(
    df,
    x_col="image_path",
    y_col="label",
    batch_size=8,
    class_mode="categorical",
    target_size=(img_size, img_size),
    seed=answer_to_life,
    subset="training",
)



## === cell 14
val_datagen = datagen.flow_from_dataframe(
    df,
    x_col="image_path",
    y_col="label",
    batch_size=8,
    class_mode="categorical",
    target_size=(img_size, img_size),
    seed=answer_to_life,
    subset="validation",
)



## === cell 15
CosineDecay = tf.keras.optimizers.schedules.CosineDecay

if training_mode:

    def create_pretrained_model():
        pretrained_model = tf.keras.applications.EfficientNetB3(
            weights="../input/efficientnet-model-file/efficientnetb3_notop.h5",
            include_top=False,
            drop_connect_rate=0.4,
            input_shape=(img_size, img_size, 3),
        )
        x = layers.GlobalAveragePooling2D()(pretrained_model.output)

        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(5, activation="softmax")(x)

        model = tf.keras.models.Model(inputs=pretrained_model.input, outputs=x)

        decay_steps = int(round(17118.0 / 8.0)) * 3
        cosine_decay = CosineDecay(
            initial_learning_rate=1e-4, decay_steps=decay_steps, alpha=0.3
        )

        callbacks_local = [
            ModelCheckpoint(
                filepath="best_model.h5", monitor="val_loss", save_best_only=True
            )
        ]

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=cosine_decay),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model, callbacks_local

    model, callbacks = create_pretrained_model()
    callbacks = [
        ModelCheckpoint(
            filepath="best_model_final.h5", monitor="val_loss", save_best_only=True
        )
    ]



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2265027552.py in <cell line: 0>()
     38         return model, callbacks_local
     39 
---> 40     model, callbacks = create_pretrained_model()
     41     # Original code overwrote callbacks; preserve behavior but keep it valid
     42     callbacks = [

/tmp/ipykernel_11/2265027552.py in create_pretrained_model()
      5 
      6     def create_pretrained_model():
----> 7         pretrained_model = tf.keras.applications.EfficientNetB3(
      8             weights="../input/efficientnet-model-file/efficientnetb3_notop.h5",
      9             include_top=False,

TypeError: EfficientNetB3() got an unexpected keyword argument 'drop_connect_rate'

## === cell 16
if not first_time_train and (not training_mode):
    model = tf.keras.models.load_model(previously_trained_model_file)

if training_mode:
    history = model.fit(
        train_datagen, epochs=6, validation_data=val_datagen, callbacks=callbacks
    )



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/113701509.py in <cell line: 0>()
      4 
      5 if training_mode:
----> 6     history = model.fit(
      7         train_datagen, epochs=6, validation_data=val_datagen, callbacks=callbacks
      8     )

NameError: name 'model' is not defined

## === cell 17
if training_mode:
    if os.path.exists("best_model_final.h5"):
        model = tf.keras.models.load_model("best_model_final.h5")
    elif os.path.exists("best_model.h5"):
        model = tf.keras.models.load_model("best_model.h5")



## === cell 18
_ = model.summary()



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/737846948.py in <cell line: 0>()
      1 # Sanity check: model is defined
----> 2 _ = model.summary()
      3 

NameError: name 'model' is not defined

## === cell 19
from PIL import Image



## === cell 20
test_image_path = df["image_path"].tolist()[4]
img = Image.open(test_image_path)
img = img.resize((512, 512))



## === cell 21
img = np.array(img)



## === cell 22
img = img / 255.0



## === cell 23
imgs = get_random_crops(img)
imgs = np.array(imgs)



## === cell 24
preds = model.predict(imgs, verbose=0)



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3815951247.py in <cell line: 0>()
----> 1 preds = model.predict(imgs, verbose=0)
      2 

NameError: name 'model' is not defined

## === cell 25
preds = np.argmax(np.sum(preds, axis=0))



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1302158700.py in <cell line: 0>()
----> 1 preds = np.argmax(np.sum(preds, axis=0))
      2 

NameError: name 'preds' is not defined

## === cell 26
preds



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4214700377.py in <cell line: 0>()
----> 1 preds
      2 

NameError: name 'preds' is not defined

## === cell 28
df.head()



## === cell 29
test_folder = "../input/cassava-leaf-disease-classification/test_images"



## === cell 30
sample_path = f"{project_folder}/sample_submission.csv"
sample_submission = pd.read_csv(sample_path)
test_files = sample_submission["image_id"].tolist()
print("Num test files (from sample_submission):", len(test_files))



## === cell 31
test_files[:5]



## === cell 32
submission_df = pd.DataFrame()
image_names = []
predictions = []

for img_name in test_files:
    tmp_img = Image.open(f"{test_folder}/{img_name}")

    aug_imgs = get_augmented_images(tmp_img)

    tmp_img_resized = tmp_img.resize((512, 512))
    tmp_img_resized = np.array(tmp_img_resized)

    aug_imgs.append(tmp_img_resized)
    imgs = np.array(aug_imgs)
    imgs = imgs / 255.0

    preds = model.predict(imgs, verbose=0)
    prediction = int(np.argmax(np.sum(preds, axis=0)))

    image_names.append(img_name)
    predictions.append(prediction)

submission_df = pd.DataFrame(data={"image_id": image_names, "label": predictions})



## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3696213530.py in <cell line: 0>()
     15     imgs = imgs / 255.0
     16 
---> 17     preds = model.predict(imgs, verbose=0)
     18     prediction = int(np.argmax(np.sum(preds, axis=0)))
     19 

NameError: name 'model' is not defined

## === cell 33
print(submission_df.shape)
print(submission_df.columns.tolist())
print("Matches sample rows:", len(submission_df) == len(sample_submission))



## === cell 34
submission_df.head()



## === cell 35
submission_df = sample_submission[["image_id"]].merge(
    submission_df, on="image_id", how="left"
)
assert submission_df["label"].isna().sum() == 0, "Some test images missing predictions."



## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/3309444786.py in <cell line: 0>()
      1 # Ensure exact same ordering as sample_submission (already iterated in that order, but keep safe merge)
----> 2 submission_df = sample_submission[["image_id"]].merge(
      3     submission_df, on="image_id", how="left"
      4 )
      5 assert submission_df["label"].isna().sum() == 0, "Some test images missing predictions."

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in merge(self, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
  10830         from pandas.core.reshape.merge import merge
  10831 
> 10832         return merge(
  10833             self,
  10834             right,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in merge(left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
    168         )
    169     else:
--> 170         op = _MergeOperation(
    171             left_df,
    172             right_df,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in __init__(self, left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, indicator, validate)
    792             left_drop,
    793             right_drop,
--> 794         ) = self._get_merge_keys()
    795 
    796         if left_drop:

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in _get_merge_keys(self)
   1295                         rk = cast(Hashable, rk)
   1296                         if rk is not None:
-> 1297                             right_keys.append(right._get_label_or_level_values(rk))
   1298                         else:
   1299                             # work-around for merge_asof(right_index=True)

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _get_label_or_level_values(self, key, axis)
   1909             values = self.axes[axis].get_level_values(key)._values
   1910         else:
-> 1911             raise KeyError(key)
   1912 
   1913         # Check for duplicates

KeyError: 'image_id'

## === cell 36
submission_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with rows:", len(submission_df))

## --- ERROR in outputing the csv:
Invalid submission: Submission must have the same length as the answers.
