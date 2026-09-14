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
Develop a model to classify paddy leaf images into one of the nine disease categories or normal leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
200001.jpg,normal
200002.jpg,blast
etc.
```

## Dataset
**train.csv** - The training set

- `image_id` - Unique image identifier corresponds to image file names (.jpg) found in the train_images directory.
- `label` - Type of paddy disease, also the target class. There are ten categories, including the normal leaf.
- `variety` - The name of the paddy variety.
- `age` - Age of the paddy in days.

**sample_submission.csv** - Sample submission file.

**train_images** - Training images stored under different sub-directories corresponding to ten target classes. Filename corresponds to the `image_id` column of `train.csv`.

**test_images** - Test set images.

# 2. Python version

3.10

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
seaborn==0.12.2
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
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        input/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        working/
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
```

-> data/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> data/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> input/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> input/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> (stopped after 10 files for performance)

# 5. Target score

0.87096

# 6. Current score

0.02998

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.04074) has done: 'I fix the environment-breaking `protobuf`/TFDF import issue by removing the unnecessary TFDF-related import that triggers `MessageFactory.GetPrototype`. Then I update the Keras data iterator call (`next(train_data)` instead of `train_data.next()`) and correctly compile the multi-output model by providing one loss per output so `loss_weights` matches. Finally, I ensure inference only reads actual image files (skipping nested directories), uses a deterministic sorted file list aligned to `sample_submission.csv`, and writes a valid `submission.csv` with exactly the required rows/columns.'
- What this solution (achieved 0.08916) has done: 'I fix the multi-output training crash by making the data generator provide three identical label targets (one for each model output) so `y_true` structure matches `y_pred`. I keep the architecture and losses the same, only wrapping the existing `flow_from_directory` iterators with a lightweight `tf.data.Dataset` mapping. I also fix the initial import error by removing the unnecessary seaborn/matplotlib/sklearn imports (the current environment error is triggered before training starts), and replace the plotting cells with safe no-ops so the notebook runs end-to-end. These changes should both make the pipeline run and substantially improve accuracy (your current 0.04074 is consistent with effectively-untrained / mis-trained model due to the training failure).'
- What this solution (achieved 0.02998) has done: 'I fix the environment-breaking protobuf/TensorFlow import issue by ensuring no TFDF/protobuf-dependent imports are pulled in (keeping the imports minimal and compatible with TF 2.18). Then I fix the runtime errors in `model.fit()` and `model.predict()` by removing unsupported `workers/use_multiprocessing/max_queue_size` arguments in this Keras/TensorFlow version, while keeping the same data generators, steps, epochs, and model architecture. Finally, I ensure the submission generation runs end-to-end and writes a valid `submission.csv` with exactly the `image_id,label` columns aligned to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D
from tensorflow.keras.layers import (
    MaxPooling2D,
    AveragePooling2D,
    GlobalAveragePooling2D,
)
from tensorflow.keras.layers import BatchNormalization, Dropout, concatenate
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.preprocessing.image import ImageDataGenerator


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "0")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "0")

tf.random.set_seed(42)
np.random.seed(42)

print("TF version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_meta_data = "../input/paddy-disease-classification/train.csv"
train_data_dir = "../input/paddy-disease-classification/train_images"
test_data_dir = "../input/paddy-disease-classification/test_images"
sample_sub_path = "../input/paddy-disease-classification/sample_submission.csv"

epochs = 100
lr = 1e-3
valid_split = 0.2
input_size = 224
batch_size = 32
classes = None

initializer = tf.keras.initializers.HeUniform()
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)



## === cell 2
early_stop = tf.keras.callbacks.EarlyStopping(
    patience=10, monitor="val_loss", restore_best_weights=True, verbose=1
)




## === cell 3
def inception(x, filters, projection, init=initializer, name=None):
    f_1x1, f_3x3, f_3x3_reduce, f_5x5, f_5x5_reduce = filters
    x1 = Conv2D(
        filters=f_1x1,
        kernel_size=(1, 1),
        kernel_initializer=init,
        strides=(1, 1),
        activation=relu,
        padding="same",
    )(x)
    x3_reducer = Conv2D(
        filters=f_3x3_reduce,
        kernel_size=(1, 1),
        kernel_initializer=init,
        strides=(1, 1),
        activation=relu,
        padding="same",
    )(x)
    x5_reducer = Conv2D(
        filters=f_5x5_reduce,
        kernel_size=(1, 1),
        kernel_initializer=init,
        strides=(1, 1),
        activation=relu,
        padding="same",
    )(x)
    pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding="same")(x)

    x3 = Conv2D(
        filters=f_3x3,
        kernel_size=(3, 3),
        kernel_initializer=init,
        strides=(1, 1),
        activation=relu,
        padding="same",
    )(x3_reducer)
    x5 = Conv2D(
        filters=f_5x5,
        kernel_size=(5, 5),
        kernel_initializer=init,
        strides=(1, 1),
        activation=relu,
        padding="same",
    )(x5_reducer)
    proj = Conv2D(
        filters=projection,
        kernel_size=(1, 1),
        kernel_initializer=init,
        strides=(1, 1),
        activation=relu,
        padding="same",
    )(pool)

    x = concatenate([x1, x3, x5, proj], axis=3, name=name)
    return x


def model_builder(shape, classes):
    input_layer = Input(shape=shape)
    x = Conv2D(
        filters=64,
        kernel_size=(7, 7),
        kernel_initializer=initializer,
        strides=(2, 2),
        activation=relu,
        padding="same",
    )(input_layer)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv2D(
        filters=64, kernel_size=(1, 1), strides=(1, 1), activation=relu, padding="same"
    )(x)
    x = Conv2D(
        filters=192, kernel_size=(3, 3), strides=(1, 1), activation=relu, padding="same"
    )(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)
    x = inception(x, [64, 128, 96, 32, 16], projection=32, name="inception_3a")
    x = inception(x, [128, 192, 128, 96, 32], projection=64, name="inception_3b")
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)
    x = inception(x, [192, 208, 96, 48, 16], projection=64, name="inception_4a")

    aux_1 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding="valid")(x)
    aux_1 = Conv2D(
        filters=128,
        kernel_size=(1, 1),
        kernel_initializer=initializer,
        strides=(1, 1),
        activation=relu,
        padding="valid",
    )(aux_1)
    aux_1 = Dense(units=1024, activation=relu)(aux_1)
    aux_1 = Dropout(rate=0.7)(aux_1)
    aux_1 = GlobalAveragePooling2D()(aux_1)
    aux_out1 = Dense(units=classes, activation=softmax, name="aux_out1")(aux_1)

    x = inception(x, [160, 224, 112, 64, 24], projection=64, name="inception_4b")
    x = inception(x, [128, 256, 128, 64, 24], projection=64, name="inception_4c")
    x = inception(x, [112, 288, 144, 64, 32], projection=64, name="inception_4d")
    x = inception(x, [256, 320, 160, 128, 32], projection=128, name="inception_4e")

    aux_2 = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding="valid")(x)
    aux_2 = Conv2D(
        filters=128,
        kernel_size=(1, 1),
        kernel_initializer=initializer,
        strides=(1, 1),
        activation=relu,
        padding="valid",
    )(aux_2)
    aux_2 = Dense(units=1024, activation=relu)(aux_2)
    aux_2 = Dropout(rate=0.7)(aux_2)
    aux_2 = GlobalAveragePooling2D()(aux_2)
    aux_out2 = Dense(units=classes, activation=softmax, name="aux_out2")(aux_2)

    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(x)
    x = inception(x, [256, 320, 160, 128, 32], projection=128, name="inception_5a")
    x = inception(x, [384, 384, 192, 128, 48], projection=128, name="inception_5b")
    x = AveragePooling2D(pool_size=(7, 7), strides=(1, 1))(x)
    x = Dropout(rate=0.4)(x)
    x = GlobalAveragePooling2D()(x)
    output_layer = Dense(units=classes, activation=softmax, name="main_out")(x)

    model = Model(input_layer, [output_layer, aux_out1, aux_out2])

    model.compile(
        optimizer=optimizer,
        loss={
            "main_out": "categorical_crossentropy",
            "aux_out1": "categorical_crossentropy",
            "aux_out2": "categorical_crossentropy",
        },
        loss_weights={"main_out": 1.0, "aux_out1": 0.3, "aux_out2": 0.3},
        metrics={
            "main_out": "accuracy",
            "aux_out1": "accuracy",
            "aux_out2": "accuracy",
        },
    )
    return model




## === cell 4
generator = ImageDataGenerator(rescale=1 / 255.0, validation_split=valid_split)

train_data = generator.flow_from_directory(
    directory=train_data_dir,
    target_size=(input_size, input_size),
    batch_size=batch_size,
    subset="training",
    class_mode="categorical",
    shuffle=True,
    seed=42,
)

valid_data = generator.flow_from_directory(
    directory=train_data_dir,
    target_size=(input_size, input_size),
    batch_size=batch_size,
    subset="validation",
    class_mode="categorical",
    shuffle=False,
)

classes = int(train_data.num_classes)
print("Detected classes:", classes)
print("Class indices:", train_data.class_indices)

steps_per_epoch = len(train_data)
validation_steps = len(valid_data)
print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)



## === cell 5
xb_tr, yb_tr = train_data[0]
xb_va, yb_va = valid_data[0]
print("Train batch x:", xb_tr.shape, "y:", yb_tr.shape)
print("Valid batch x:", xb_va.shape, "y:", yb_va.shape)



## === cell 6
model = model_builder(shape=(input_size, input_size, 3), classes=classes)



## === cell 7
model.summary()



## === cell 8
try:
    tf.keras.utils.plot_model(model, "baseline_inception.png", show_shapes=True)
except Exception as e:
    print("plot_model skipped:", repr(e))



## === cell 9
history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[early_stop],
    verbose=1,
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2994569621.py in <cell line: 0>()
      2 # workers/use_multiprocessing/max_queue_size for these inputs.
      3 # Removing these args is input-pipeline-only and keeps core training semantics.
----> 4 history = model.fit(
      5     train_data,
      6     validation_data=valid_data,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/compile_utils.py in call(self, y_true, y_pred, sample_weight)
    753                             lambda _: "*", y_pred
    754                         )
--> 755                         raise ValueError(
    756                             "y_true and y_pred have different structures.\n"
    757                             f"y_true: {y_true_struct}\n"

ValueError: y_true and y_pred have different structures.
y_true: *
y_pred: ['*', '*', '*']


## === cell 10
print(
    "Last epoch metrics snapshot:",
    {k: v[-1] for k, v in history.history.items() if len(v) > 0},
)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1301138724.py in <cell line: 0>()
      1 print(
      2     "Last epoch metrics snapshot:",
----> 3     {k: v[-1] for k, v in history.history.items() if len(v) > 0},
      4 )
      5 

NameError: name 'history' is not defined

## === cell 11
print("Skipped full-dataset evaluate() to save time.")



## === cell 12
temp_hist = pd.DataFrame(history.history)
temp_hist.to_csv("history.csv", index=False)
temp_hist.head()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3530065703.py in <cell line: 0>()
----> 1 temp_hist = pd.DataFrame(history.history)
      2 temp_hist.to_csv("history.csv", index=False)
      3 temp_hist.head()
      4 

NameError: name 'history' is not defined

## === cell 13
model.save("baseline.hdf5")



## === cell 14
model.save_weights("baseline_inception_weights.weights.h5")



## === cell 15
train_data.class_indices



## === cell 16
sub = pd.read_csv(sample_sub_path)
needed_ids = sub["image_id"].astype(str).tolist()

idx_to_label = {v: k for k, v in train_data.class_indices.items()}

test_paths = [os.path.join(test_data_dir, image_id) for image_id in needed_ids]
exists_mask = [os.path.isfile(p) for p in test_paths]

missing = int(np.sum(np.logical_not(exists_mask)))
first_idx = int(min(idx_to_label.keys()))
missing_label = idx_to_label[first_idx]

test_df = pd.DataFrame(
    {
        "image_id": needed_ids,
        "filepath": test_paths,
        "exists": exists_mask,
    }
)

test_present_df = test_df[test_df["exists"]].copy().reset_index(drop=True)

test_gen = ImageDataGenerator(rescale=1 / 255.0).flow_from_dataframe(
    dataframe=test_present_df,
    x_col="filepath",
    y_col=None,
    target_size=(input_size, input_size),
    batch_size=batch_size,
    shuffle=False,  # keep order aligned to test_present_df
    class_mode=None,
)

pred = model.predict(
    test_gen,
    steps=len(test_gen),
    verbose=1,
)

main_pred = pred[0]  # [N, classes]
pred_idx = np.argmax(main_pred, axis=1).astype(np.int32)
pred_labels_present = [idx_to_label[int(i)] for i in pred_idx]

out_labels = np.full(shape=(len(needed_ids),), fill_value=missing_label, dtype=object)
present_positions = np.flatnonzero(test_df["exists"].to_numpy())
out_labels[present_positions] = np.array(pred_labels_present, dtype=object)

print("Missing test files:", missing)

submission = pd.DataFrame({"image_id": needed_ids, "label": out_labels})
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
