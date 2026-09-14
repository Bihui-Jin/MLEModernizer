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
Given a dataset of images from digital pathology scans, predict if the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. 

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability that center 32x32px region of a patch contains at least one pixel of tumor tissue. The file should contain a header and have the following format:

```
id,label
0b2ea2a822ad23fdb1b5dd26653da899fbd2c0d5,0
95596b92e5066c5c52466c90b69ff089b39f2737,0
248e6738860e2ebcf6258cdc1f32f299e0c76914,0
etc.
```

## Dataset
Files are named with an image `id`. The `train_labels.csv` file provides the ground truth for the images in the `train` folder. You are predicting the labels for the images in the `test` folder.

# 2. Python version

3.13

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
        input/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
        working/
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
```

-> data/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> (stopped after 10 files for performance)

# 5. Target score

0.753736212726282

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf
import random
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Dense,
    Conv2D,
)
from tensorflow.keras.initializers import lecun_normal
from tensorflow.keras.utils import register_keras_serializable

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
print("Python:", os.sys.version)
print("TF:", tf.__version__)



## === cell 2
BASE = "/kaggle/input/histopathologic-cancer-detection"
train_images = f"{BASE}/train/"
test_images = f"{BASE}/test/"
train_csv = f"{BASE}/train_labels.csv"
sample_sub_csv = f"{BASE}/sample_submission.csv"

assert os.path.exists(train_images), f"Missing path: {train_images}"
assert os.path.exists(test_images), f"Missing path: {test_images}"
assert os.path.exists(train_csv), f"Missing path: {train_csv}"
assert os.path.exists(sample_sub_csv), f"Missing path: {sample_sub_csv}"



## === cell 3
test_df = pd.read_csv(sample_sub_csv)
test_df["id"] = test_df["id"].astype(str) + ".tif"

print("Test Set Size:", test_df.shape)
test_df.head()



## === cell 4
train_df = pd.read_csv(train_csv)
train_df["id"] = train_df["id"].astype(str) + ".tif"
train_df["label"] = train_df["label"].astype(str)  # for categorical generator

print("Train Set Size:", train_df.shape)
train_df.head()



## === cell 5
pass




## === cell 6
@tf.function
def standardize(image):
    image = tf.convert_to_tensor(image)
    mean, var = tf.nn.moments(image, axes=(0, 1, 2), keepdims=True)
    std = tf.sqrt(var)
    return (image - mean) / (std + 1e-7)




## === cell 7
BATCH_SIZE = 64
TARGET_SIZE = (96, 96)

test_datagen = ImageDataGenerator(rescale=1 / 255, preprocessing_function=standardize)

test_loader = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=test_images,
    x_col="id",
    y_col=None,
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=False,
    class_mode=None,
    target_size=TARGET_SIZE,
)



## === cell 8
from sklearn.model_selection import train_test_split

train_split_df, val_split_df = train_test_split(
    train_df, test_size=0.10, random_state=SEED, stratify=train_df["label"]
)

train_datagen = ImageDataGenerator(rescale=1 / 255, preprocessing_function=standardize)
val_datagen = ImageDataGenerator(rescale=1 / 255, preprocessing_function=standardize)

train_loader = train_datagen.flow_from_dataframe(
    dataframe=train_split_df,
    directory=train_images,
    x_col="id",
    y_col="label",
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=True,
    class_mode="categorical",
    target_size=TARGET_SIZE,
)

val_loader = val_datagen.flow_from_dataframe(
    dataframe=val_split_df,
    directory=train_images,
    x_col="id",
    y_col="label",
    batch_size=BATCH_SIZE,
    seed=SEED,
    shuffle=False,
    class_mode="categorical",
    target_size=TARGET_SIZE,
)




## === cell 9
@register_keras_serializable(package="Custom")
class CBAM(Layer):
    def __init__(self, channels, reduction_ratio=16, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.channels = channels
        self.reduction_ratio = reduction_ratio

        self.global_avg_pool = GlobalAveragePooling2D()
        self.global_max_pool = GlobalMaxPooling2D()
        self.fc1 = Dense(
            units=channels // reduction_ratio,
            activation="selu",
            kernel_initializer=lecun_normal(),
        )
        self.fc2 = Dense(
            units=channels,
            activation="sigmoid",
            kernel_initializer=lecun_normal(),
        )

        self.conv = Conv2D(
            filters=1,
            kernel_size=7,
            padding="same",
            activation="sigmoid",
            kernel_initializer=lecun_normal(),
        )

    def call(self, inputs, **kwargs):
        avg_pooled = self.global_avg_pool(inputs)
        max_pooled = self.global_max_pool(inputs)
        avg_fc = self.fc2(self.fc1(avg_pooled))
        max_fc = self.fc2(self.fc1(max_pooled))
        channel_attention = avg_fc + max_fc
        channel_attention = tf.expand_dims(channel_attention, axis=1)
        channel_attention = tf.expand_dims(channel_attention, axis=1)
        channel_refined = inputs * channel_attention

        avg_spatial = tf.reduce_mean(channel_refined, axis=-1, keepdims=True)
        max_spatial = tf.reduce_max(channel_refined, axis=-1, keepdims=True)
        spatial_attention = self.conv(tf.concat([avg_spatial, max_spatial], axis=-1))
        spatial_refined = channel_refined * spatial_attention
        return spatial_refined

    def get_config(self):
        config = super(CBAM, self).get_config()
        config.update(
            {"channels": self.channels, "reduction_ratio": self.reduction_ratio}
        )
        return config


@register_keras_serializable(package="Custom")
class CustomAlphaDropout(Layer):
    def __init__(self, rate, **kwargs):
        super(CustomAlphaDropout, self).__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, training=None):
        if training is None:
            training = tf.constant(False)
        if not training:
            return inputs

        noise_shape = tf.shape(inputs)
        keep_prob = 1 - self.rate
        random_tensor = keep_prob + tf.random.uniform(noise_shape, 0, 1)
        binary_tensor = tf.floor(random_tensor)
        outputs = inputs * binary_tensor / keep_prob
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(CustomAlphaDropout, self).get_config()
        config.update({"rate": self.rate})
        return config




## === cell 10
from tensorflow.keras import layers


def build_model(input_shape=(96, 96, 3)):
    inputs = keras.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, padding="same", kernel_initializer=lecun_normal())(inputs)
    x = layers.Activation("selu")(x)
    x = CBAM(32)(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, padding="same", kernel_initializer=lecun_normal())(x)
    x = layers.Activation("selu")(x)
    x = CBAM(64)(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, padding="same", kernel_initializer=lecun_normal())(x)
    x = layers.Activation("selu")(x)
    x = CBAM(128)(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(256, 3, padding="same", kernel_initializer=lecun_normal())(x)
    x = layers.Activation("selu")(x)
    x = CBAM(256)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="selu", kernel_initializer=lecun_normal())(x)
    x = CustomAlphaDropout(0.2)(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model


cnn = build_model(input_shape=(96, 96, 3))
cnn.summary()



## === cell 11
cnn.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=[keras.metrics.AUC(name="auc")],
)

EPOCHS = 2  # unchanged

fit_kwargs = dict(
    x=train_loader,
    validation_data=val_loader,
    epochs=EPOCHS,
    verbose=1,
)
try:
    fit_kwargs.update(
        dict(workers=os.cpu_count() or 4, use_multiprocessing=True, max_queue_size=32)
    )
except Exception:
    pass

history = cnn.fit(**fit_kwargs)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2546357184.py in <cell line: 0>()
     23     pass
     24 
---> 25 history = cnn.fit(**fit_kwargs)
     26 

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

## === cell 12
pred_kwargs = dict(
    x=test_loader,
    verbose=1,
)
try:
    pred_kwargs.update(
        dict(workers=os.cpu_count() or 4, use_multiprocessing=True, max_queue_size=32)
    )
except Exception:
    pass

test_preds = cnn.predict(**pred_kwargs)

print(
    "test_preds shape:",
    test_preds.shape,
    "min/max:",
    float(test_preds.min()),
    float(test_preds.max()),
)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2824030259.py in <cell line: 0>()
     12     pass
     13 
---> 14 test_preds = cnn.predict(**pred_kwargs)
     15 
     16 print(

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

TypeError: TensorFlowTrainer.predict() got an unexpected keyword argument 'workers'

## === cell 13
class_indices = train_loader.class_indices
print("class_indices:", class_indices)

tumor_class_index = class_indices.get("1", 1)
tumor_probs = test_preds[:, tumor_class_index].astype(np.float32)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2196736844.py in <cell line: 0>()
      3 
      4 tumor_class_index = class_indices.get("1", 1)
----> 5 tumor_probs = test_preds[:, tumor_class_index].astype(np.float32)
      6 

NameError: name 'test_preds' is not defined

## === cell 14
submission_df = pd.read_csv(sample_sub_csv)

submission = pd.DataFrame(
    {
        "id": submission_df["id"].astype(str),
        "label": tumor_probs,
    }
)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
submission.head()



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2803532793.py in <cell line: 0>()
      4     {
      5         "id": submission_df["id"].astype(str),
----> 6         "label": tumor_probs,
      7     }
      8 )

NameError: name 'tumor_probs' is not defined

## === cell 15
frequency_distribution = (submission["label"].round(3).describe()).to_frame(
    name="label_stats"
)
print(frequency_distribution)
pass

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3274553659.py in <cell line: 0>()
----> 1 frequency_distribution = (submission["label"].round(3).describe()).to_frame(
      2     name="label_stats"
      3 )
      4 print(frequency_distribution)
      5 pass

NameError: name 'submission' is not defined
