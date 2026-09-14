# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.9

# 2. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
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

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
%config Completer.use_jedi = False


## === cell 1

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 2
import zipfile

train_zip = '/kaggle/input/aerial-cactus-identification/train.zip'
with zipfile.ZipFile(train_zip, 'r') as zip_ref:
    zip_ref.extractall('/kaggle/tmp/')

test_zip = '/kaggle/input/aerial-cactus-identification/test.zip'
with zipfile.ZipFile(test_zip, 'r') as zip_ref:
    zip_ref.extractall('/kaggle/tmp/')
    
for dirname, _, filenames in os.walk('/kaggle/tmp'):
    for filename in filenames[:10]:
        print(os.path.join(dirname, filename))    


## === cell 3
import tensorflow as tf
from tensorflow import keras
import datetime


## --- ERROR in cell 3, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mAttributeError[0m                            Traceback (most recent call last)
[0;31mAttributeError[0m: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
class Cnn_Model:

    def __init__(self):
        self.BATCH_SZ = 32
        self.LR_RATE = 0.0001
        self.EPOCHS = 5
        self.MODEL_CKP_PATH = '/kaggle/tmp/model_ckpoint'
        self.MODEL_TRAIN_DATA = '/kaggle/tmp/train'
        self.MODEL_TEST_DATA = '/kaggle/tmp/test'
        self.model = None
        
    def get_model(self):
        return self.model
            
    def load_dataframe(self):
        self.df = pd.read_csv('../input/aerial-cactus-identification/train.csv')
        self.df.has_cactus = self.df.has_cactus.astype(str)
        
    def make_train_gen(self):

        self.train_gen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            zoom_range=0.2,
            width_shift_range=0.4,
            height_shift_range=0.4,
            horizontal_flip=True,
            vertical_flip=True,
            rotation_range=60,
            brightness_range=[0.8,1.1],
        )

        self.train_direcIter = self.train_gen.flow_from_dataframe(
            dataframe=self.df,
            directory=self.MODEL_TRAIN_DATA,
            x_col='id',
            y_col='has_cactus',
            target_size=(32,32),
            batch_size=32,
            shuffle=True,
            class_mode='binary'
        )

    def build_model(self):

        self.model = keras.models.Sequential([
            keras.layers.Conv2D(64, (3, 3), input_shape=(32, 32, 3)),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.2),
            keras.layers.Conv2D(64, (3, 3)),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Dropout(0.1),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.2),            
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(1, activation="sigmoid")
        ])

        self.model.summary()

    def compile_model(self):

        optz = keras.optimizers.Adam(learning_rate=self.LR_RATE)

        self.model.compile(
            optimizer=optz,
            loss=keras.losses.binary_crossentropy,
            metrics=["acc"]
        )

    def train_model(self):

        def scheduler(epoch, lr):
            return lr * tf.math.exp(-0.01)

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.MODEL_CKP_PATH,
            save_weights_only=False,
            monitor='acc',
            mode='max',
            save_best_only=True,
        )

        log_dir = "logs\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            update_freq="epoch",
            profile_batch=0,
            embeddings_freq=0
        )

        callback_list = [
            keras.callbacks.LearningRateScheduler(scheduler),
        ]

        self.model.fit(
            x=self.train_direcIter,
            epochs=self.EPOCHS,
            callbacks=callback_list,
            steps_per_epoch=(self.train_direcIter.samples//self.BATCH_SZ),
        )

    def load_model(self):
        self.model = tf.keras.models.load_model(self.MODEL_CKP_PATH)
        print("> Model load done...")

    def predict_values(self):
        
        for file_name in os.listdir(self.MODEL_TEST_DATA)[:10]:
            fullpath_img = os.path.join(self.MODEL_TEST_DATA, file_name)
            image = tf.keras.preprocessing.image.load_img(fullpath_img)
            input_arr = keras.preprocessing.image.img_to_array(image)
            input_arr = input_arr/255
            print(type(input_arr))
            print(input_arr.shape)
            input_arr = np.array([input_arr])  # Convert single image to a batch.
            predictions = self.model.predict(input_arr)
            print(predictions)
        
    def evaluate_model(self, steps=100):

        result = self.model.evaluate(self.train_direcIter, steps=steps)
        print(result)
