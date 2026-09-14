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

## === cell 1
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import random
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Conv2D
from tensorflow.keras.initializers import lecun_normal
from tensorflow.keras.utils import register_keras_serializable


SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
print(tf.__version__)  # Should output 2.15.1

## === cell 4
test_images = '/kaggle/input/histopathologic-cancer-detection/test/'

test_df = pd.read_csv('/kaggle/input/histopathologic-cancer-detection/sample_submission.csv')

test_df['id'] = test_df['id'] + '.tif'
test_df['label'] = test_df['label'].astype(str)

print('Test Set Size:', test_df.shape)
test_df.head()

## === cell 6
sample_images = test_df.sample(16)

fig, axes = plt.subplots(4, 4, figsize=(6, 6))
fig.tight_layout(pad=1.0)

for i, ax in enumerate(axes.flat):
    id = sample_images.iloc[i]['id']  
    label = sample_images.iloc[i]['label']  

    img = mpimg.imread(os.path.join(test_images, id))

    ax.imshow(img, cmap='gray')
    ax.set_title(f"Label: {label}")
    ax.axis('off')

plt.show()

## === cell 8
def standardize(image):
    mean = np.mean(image, axis=(0, 1, 2), keepdims=True)  # Compute mean across all channels
    std = np.std(image, axis=(0, 1, 2), keepdims=True)    # Compute std across all channels
    return (image - mean) / (std + 1e-7)                  # Subtract mean and divide by std

test_datagen = ImageDataGenerator(
    rescale=1/255,
    preprocessing_function=standardize  # Apply standardization
) # Normalize pixel values

test_loader = test_datagen.flow_from_dataframe(
    dataframe = test_df,
    directory = test_images,
    x_col = 'id',
    y_col = 'label',
    batch_size = 64,
    seed = 1,
    shuffle = False,
    class_mode = 'categorical',
    target_size = (96,96)
)


## === cell 10
from tensorflow.keras.utils import register_keras_serializable

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
            activation='selu',  # Use SELU for scaled exponential units
            kernel_initializer=lecun_normal()
        )
        self.fc2 = Dense(
            units=channels,
            activation='sigmoid',  # Scale attention values
            kernel_initializer=lecun_normal()
        )

        self.conv = Conv2D(
            filters=1,
            kernel_size=7,
            padding='same',
            activation='sigmoid',  # Scale attention values
            kernel_initializer=lecun_normal()
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
        config.update({
            "channels": self.channels,
            "reduction_ratio": self.reduction_ratio
        })
        return config


@register_keras_serializable(package="Custom")
class CustomAlphaDropout(Layer):
    def __init__(self, rate, **kwargs):
        super(CustomAlphaDropout, self).__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, training=None):
        if training is None:
            training = tf.constant(False)  # Default to inference mode if not specified

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

## === cell 12
cnn = keras.models.load_model('/kaggle/input/4conv2d-w-selu-v2/keras/default/1/cbam_selu (2).keras', custom_objects={'CBAM': CBAM, "CustomAlphaDropout": CustomAlphaDropout})
cnn.summary()


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/779924987.py in <cell line: 0>()
----> 1 cnn = keras.models.load_model('/kaggle/input/4conv2d-w-selu-v2/keras/default/1/cbam_selu (2).keras', custom_objects={'CBAM': CBAM, "CustomAlphaDropout": CustomAlphaDropout})
      2 cnn.summary()

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    198         )
    199     elif str(filepath).endswith(".keras"):
--> 200         raise ValueError(
    201             f"File not found: filepath={filepath}. "
    202             "Please ensure the file is an accessible `.keras` "

ValueError: File not found: filepath=/kaggle/input/4conv2d-w-selu-v2/keras/default/1/cbam_selu (2).keras. Please ensure the file is an accessible `.keras` zip file.

## === cell 14
test_preds = cnn.predict(test_loader)

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2213367151.py in <cell line: 0>()
----> 1 test_preds = cnn.predict(test_loader)

NameError: name 'cnn' is not defined

## === cell 16
predicted_labels = np.argmax(test_preds, axis=1)

submission_df = pd.read_csv('/kaggle/input/histopathologic-cancer-detection/sample_submission.csv')

submission = pd.DataFrame({'id': submission_df['id'], 'label': predicted_labels})

submission.to_csv('submission.csv', index=False)

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3008275620.py in <cell line: 0>()
      1 # Convert probabilities to predicted class labels
----> 2 predicted_labels = np.argmax(test_preds, axis=1)
      3 
      4 submission_df = pd.read_csv('/kaggle/input/histopathologic-cancer-detection/sample_submission.csv')
      5 

NameError: name 'test_preds' is not defined

## === cell 17
submission.head()

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3365464162.py in <cell line: 0>()
----> 1 submission.head()

NameError: name 'submission' is not defined

## === cell 18
frequency_distribution = (submission.label.value_counts() / len(submission)).to_frame()

plt.figure(figsize=(6, 4))
colors = ['lightgreen', 'lightcoral']  # light green for benign, light red for malignant

frequency_distribution.iloc[:, 0].plot(kind='bar', color=colors)

plt.title('Frequency Distribution of Labels')
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.xticks([0, 1], ['Benign', 'Malignant'], rotation=0)

plt.show()

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/52911631.py in <cell line: 0>()
      1 # Calculate frequency distribution
----> 2 frequency_distribution = (submission.label.value_counts() / len(submission)).to_frame()
      3 
      4 # Plotting the frequency distribution as a bar chart
      5 plt.figure(figsize=(6, 4))

NameError: name 'submission' is not defined
