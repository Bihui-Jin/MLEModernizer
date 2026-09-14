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

3.12

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

0.8319734058627984

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import random
import shutil
import os, cv2, json, glob
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras import models, layers
from keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import efficientnet_v2
from keras.optimizers import Adam
from kerastuner.tuners import Hyperband
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image as kimage

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
df = pd.read_csv('../input/cassava-leaf-disease-classification/train.csv')
WORK_DIR = '../input/cassava-leaf-disease-classification/'
OUTPUT_DIR = './'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)  
target_folder = "dataset"
path = "../input/cassava-leaf-disease-classification/train_images"
if not os.path.exists(target_folder):
    os.mkdir(target_folder)

for label in df['label'].unique():
    label_folder = os.path.join(target_folder, str(label))
    if not os.path.exists(label_folder):
        os.mkdir(label_folder)
    mask = df['label'] == label
    rows = df.loc[mask]
    
    for _, row in rows.iterrows():
        image_name = row['image_id']
        src_path = os.path.join(path, image_name)
        dst_path = os.path.join(label_folder, image_name)
        shutil.copy(src_path, dst_path)

## === cell 2
def img_train(directory, batch_size=32, image_size=(224, 224), validation_split=0.1, seed=123):
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        class_names=None,
        color_mode="rgb",
        batch_size=batch_size,  
        image_size=image_size,  
        shuffle=True,   
        seed=seed,
        validation_split=validation_split,
        subset="training",
        interpolation="bilinear",
        crop_to_aspect_ratio=False
    )
    if validation_split != None:
        valid_dataset = tf.keras.utils.image_dataset_from_directory(
            directory,
            labels="inferred",
            label_mode="categorical",
            class_names=None,
            color_mode="rgb",
            batch_size=batch_size,  
            image_size=image_size,  
            shuffle=True,   
            seed=seed,
            validation_split=validation_split,
            subset="validation",
            interpolation="bilinear",
            crop_to_aspect_ratio=False
        )
        return train_dataset, valid_dataset
    else:
        return train_dataset

## === cell 3
dataset = '/kaggle/working/dataset'
train_dataset, test_dataset = img_train(dataset, 16)
split = len(train_dataset)
size = int(split * 0.1)
train_dataset = train_dataset.skip(size)
valid_dataset = train_dataset.take(size)

## === cell 4
def data_augmentation():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=123),
        tf.keras.layers.RandomTranslation(0.2, 0.2, fill_mode="reflect", seed=123),
        tf.keras.layers.RandomRotation(0.2, fill_mode="reflect", seed=123),
        tf.keras.layers.RandomZoom(0.2, seed=123),
        tf.keras.layers.RandomContrast(0.2, seed=123),
    ])
data_aug = data_augmentation()
augmented_dataset = train_dataset.repeat(2).map(lambda x, y: (data_aug(x), y))
train_dataset = train_dataset.concatenate(augmented_dataset)
print("augmentation train dataset size:", train_dataset.cardinality().numpy())

## === cell 5
class SigmoidFocalCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, from_logits=False, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        if self.from_logits:
            y_pred = tf.sigmoid(y_pred)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        weight = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
        focal_loss = weight * ((1 - y_pred) ** self.gamma) * cross_entropy
        return tf.reduce_sum(focal_loss, axis=-1)

## === cell 6
def build_model(hp):
    efficientweight = '/kaggle/input/test55/efficientnetv2s.h5'
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    dropout_rate = hp.Float('dropout_rate', min_value=0, max_value=0.5)
    dense_units = hp.Int('dense_units', min_value=128, max_value=512, step=32)
    model = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
        weights=efficientweight,
        include_top=False,
        input_shape=(224, 224, 3)
    )
    x = tf.keras.layers.GlobalAveragePooling2D()(model.output)
    x = tf.keras.layers.Dense(dense_units, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(5, activation='softmax')(x)
    model = tf.keras.Model(inputs=model.input, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=SigmoidFocalCrossEntropy(alpha=0.25, gamma=2, from_logits=False),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")]
    )
    return model

## === cell 7
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=5,
    hyperband_iterations=1,
    factor=3,
    directory='my_dir',
    project_name='keras_tuner_efficientnet'
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0.001,
    patience=3,
    mode='min',
    verbose=1,
    restore_best_weights=True
)

tuner.search(
    train_dataset, 
    validation_data=valid_dataset, 
    epochs=2,
    callbacks=[early_stop]
)

best_hps = tuner.get_best_hyperparameters()[0]

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3749411520.py in <cell line: 0>()
----> 1 tuner = kt.Hyperband(
      2     build_model,
      3     objective='val_accuracy',
      4     max_epochs=5,
      5     hyperband_iterations=1,

/usr/local/lib/python3.11/dist-packages/keras_tuner/src/tuners/hyperband.py in __init__(self, hypermodel, objective, max_epochs, factor, hyperband_iterations, seed, hyperparameters, tune_new_entries, allow_new_entries, max_retries_per_trial, max_consecutive_failed_trials, **kwargs)
    418             max_consecutive_failed_trials=max_consecutive_failed_trials,
    419         )
--> 420         super().__init__(oracle=oracle, hypermodel=hypermodel, **kwargs)
    421 
    422     def run_trial(self, trial, *fit_args, **fit_kwargs):

/usr/local/lib/python3.11/dist-packages/keras_tuner/src/engine/tuner.py in __init__(self, oracle, hypermodel, max_model_size, optimizer, loss, metrics, distribution_strategy, directory, project_name, logger, tuner_id, overwrite, executions_per_trial, **kwargs)
    120             )
    121 
--> 122         super().__init__(
    123             oracle=oracle,
    124             hypermodel=hypermodel,

/usr/local/lib/python3.11/dist-packages/keras_tuner/src/engine/base_tuner.py in __init__(self, oracle, hypermodel, directory, project_name, overwrite, **kwargs)
    130         else:
    131             # Only populate initial space if not reloading.
--> 132             self._populate_initial_space()
    133 
    134         # Run in distributed mode.

/usr/local/lib/python3.11/dist-packages/keras_tuner/src/engine/base_tuner.py in _populate_initial_space(self)
    190         self.hypermodel.declare_hyperparameters(hp)
    191         self.oracle.update_space(hp)
--> 192         self._activate_all_conditions()
    193 
    194     def search(self, *fit_args, **fit_kwargs):

/usr/local/lib/python3.11/dist-packages/keras_tuner/src/engine/base_tuner.py in _activate_all_conditions(self)
    147         hp = self.oracle.get_space()
    148         while True:
--> 149             self.hypermodel.build(hp)
    150             self.oracle.update_space(hp)
    151 

/tmp/ipykernel_11/3433182558.py in build_model(hp)
      4     dropout_rate = hp.Float('dropout_rate', min_value=0, max_value=0.5)
      5     dense_units = hp.Int('dense_units', min_value=128, max_value=512, step=32)
----> 6     model = tf.keras.applications.efficientnet_v2.EfficientNetV2S(
      7         weights=efficientweight,
      8         include_top=False,

/usr/local/lib/python3.11/dist-packages/keras/src/applications/efficientnet_v2.py in EfficientNetV2S(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, include_preprocessing, name)
   1237     name="efficientnetv2-s",
   1238 ):
-> 1239     return EfficientNetV2(
   1240         width_coefficient=1.0,
   1241         depth_coefficient=1.0,

/usr/local/lib/python3.11/dist-packages/keras/src/applications/efficientnet_v2.py in EfficientNetV2(width_coefficient, depth_coefficient, default_size, dropout_rate, drop_connect_rate, depth_divisor, min_depth, bn_momentum, activation, blocks_args, name, include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, include_preprocessing, weights_name)
    894 
    895     if not (weights in {"imagenet", None} or file_utils.exists(weights)):
--> 896         raise ValueError(
    897             "The `weights` argument should be either "
    898             "`None` (random initialization), `imagenet` "

ValueError: The `weights` argument should be either `None` (random initialization), `imagenet` (pre-training on ImageNet), or the path to the weights file to be loaded.Received: weights=/kaggle/input/test55/efficientnetv2s.h5

## === cell 8
def Model_Training(tuner, train_dataset, valid_dataset, epochs=10):
    best_hps = tuner.get_best_hyperparameters()[0]
    model = tuner.hypermodel.build(best_hps)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0.001, patience=5, mode='min', verbose=1, restore_best_weights=True
    )
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=valid_dataset,
        callbacks=[early_stop]
    )
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    plt.clf()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    plt.clf()

    return model

## === cell 9
model = Model_Training(tuner, train_dataset, valid_dataset, epochs=10)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1121401390.py in <cell line: 0>()
----> 1 model = Model_Training(tuner, train_dataset, valid_dataset, epochs=10)

NameError: name 'tuner' is not defined

## === cell 10
submission = pd.DataFrame(columns=['image_id','label'])
for image_name in os.listdir(WORK_DIR + 'test_images'):
    image_path = os.path.join(WORK_DIR + 'test_images', image_name)
    image = tf.keras.preprocessing.image.load_img(image_path)
    resized_image = image.resize((224, 224))
    numpied_image = np.expand_dims(resized_image, 0)
    tensored_image = tf.cast(numpied_image, tf.float32)
    y_pred = model.predict(tensored_image)
    y_pred = np.argmax(y_pred, axis=-1)[0]
    submission.loc[len(submission)] = [image_name, int(y_pred)]
submission.to_csv('submission.csv', index=False)
folder = '/kaggle/working/dataset'
try:
    shutil.rmtree(folder)
    print("Folder Deleted")
except OSError as e:
    print(f"Error: {e}")

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1207200553.py in <cell line: 0>()
      6     numpied_image = np.expand_dims(resized_image, 0)
      7     tensored_image = tf.cast(numpied_image, tf.float32)
----> 8     y_pred = model.predict(tensored_image)
      9     y_pred = np.argmax(y_pred, axis=-1)[0]
     10     submission.loc[len(submission)] = [image_name, int(y_pred)]

NameError: name 'model' is not defined
