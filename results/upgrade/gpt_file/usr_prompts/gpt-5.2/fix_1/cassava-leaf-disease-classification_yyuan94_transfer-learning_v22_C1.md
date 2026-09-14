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

0.2311876699909338

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
import seaborn as sns
import json
from scipy import stats

import os



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
train_csv = pd.read_csv('/kaggle/input/cassava-leaf-disease-classification/train.csv')
print("Number of train images: {}".format(len(train_csv)))

## === cell 3
train_csv.head()

## === cell 4
with open('/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json', 'r') as fp:
    class_map = json.load(fp)
class_map

## === cell 5
ax=train_csv.pivot_table(columns='label',aggfunc='size').plot(kind='barh')
ax.set_yticklabels(class_map.values()) 
ax.set_xlabel('count')

## === cell 8
@tf.function
def process_data(path, label):
    path = '/kaggle/input/cassava-leaf-disease-classification/train_images/' + path
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3) 
    return img, tf.one_hot(label, 5) 

## === cell 9
train, val=train_test_split(train_csv, test_size=0.1, random_state=42,stratify=train_csv['label'])

## === cell 10
oversampled_df = []
target_count = int(train.pivot_table(columns='label',aggfunc='size').values[3]*0.8)
for i in range(len(class_map)):
    class_i = train[train.label==i]
    oversampled_df.append(class_i.sample(target_count, replace=True))


## === cell 11
resampled_train = pd.concat(oversampled_df, axis=0)
resampled_train.pivot_table(columns='label',aggfunc='size')

## === cell 12
resampled_train = resampled_train.sample(frac=1).reset_index(drop=True)
resampled_train.head()

## === cell 13
batch_size = 8

## === cell 14
len(train_ds)//8

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1629334655.py in <cell line: 0>()
----> 1 len(train_ds)//8

NameError: name 'train_ds' is not defined

## === cell 15
train_ds = tf.data.Dataset.from_tensor_slices((train.image_id.values, train.label.values))\
                .map(process_data, num_parallel_calls = tf.data.experimental.AUTOTUNE)\
                .shuffle(buffer_size = 2000)\
                .batch(batch_size)\
                .prefetch(tf.data.experimental.AUTOTUNE) 

val_ds = tf.data.Dataset.from_tensor_slices((val.image_id.values, val.label.values))\
                .map(process_data, num_parallel_calls = tf.data.experimental.AUTOTUNE)\
                .batch(batch_size)\
                .prefetch(tf.data.experimental.AUTOTUNE) 

## === cell 16
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.RandomCrop(height=512, width=512),
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.25),
        tf.keras.layers.experimental.preprocessing.RandomZoom((-0.2, 0)),
        tf.keras.layers.experimental.preprocessing.RandomContrast((0,0.2))
    ]
)

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/503765686.py in <cell line: 0>()
      1 data_augmentation = tf.keras.Sequential(
      2     [
----> 3         tf.keras.layers.experimental.preprocessing.RandomCrop(height=512, width=512),
      4         tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
      5         tf.keras.layers.experimental.preprocessing.RandomRotation(0.25),

AttributeError: module 'keras._tf_keras.keras.layers' has no attribute 'experimental'

## === cell 18
for images, labels in train_ds.take(1):
    images = data_augmentation(images, training=True)
    print(images.shape, labels.shape)
    plt.figure(figsize=(10, 10))
    labels = np.argmax(labels.numpy(), -1)
    for i in range(8):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images.numpy()[i])
        plt.title(class_map[str(labels[i])])
        plt.axis("off")

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1390419422.py in <cell line: 0>()
      1 # test if datset is fetching and decoding images correctly
      2 for images, labels in train_ds.take(1):
----> 3     images = data_augmentation(images, training=True)
      4     print(images.shape, labels.shape)
      5     plt.figure(figsize=(10, 10))

NameError: name 'data_augmentation' is not defined

## === cell 20
def plot_metrics(history, metrics= ['loss', 'accuracy']):
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch, history.history[metric],  label='Train')
        plt.plot(history.epoch, history.history['val_'+metric], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        else:
            plt.ylim([0, 1])

    plt.legend()

## === cell 21
def plot_cm(labels, predictions):
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    print('True Negatives: ', cm[0][0])
    print('False Positives: ', cm[0][1])
    print('False Negatives: ', cm[1][0])
    print('True Positives: ', cm[1][1])
    print('Total: ', np.sum(cm[1]))

## === cell 22
import tensorflow.keras.backend as K
def sigmoid_focal_crossentropy(y_true, y_pred, alpha=0.25, gamma=2.0, from_logits=False):
    """Implements the focal loss function.
    Focal loss was first introduced in the RetinaNet paper
    (https://arxiv.org/pdf/1708.02002.pdf). Focal loss is extremely useful for
    classification when you have highly imbalanced classes. It down-weights
    well-classified examples and focuses on hard examples. The loss value is
    much high for a sample which is misclassified by the classifier as compared
    to the loss value corresponding to a well-classified example. One of the
    best use-cases of focal loss is its usage in object detection where the
    imbalance between the background class and other classes is extremely high.
    Args:
        y_true: true targets tensor.
        y_pred: predictions tensor.
        alpha: balancing factor.
        gamma: modulating factor.
    Returns:
        Weighted loss float `Tensor`. If `reduction` is `NONE`,this has the
        same shape as `y_true`; otherwise, it is scalar.
    """
    if gamma and gamma < 0:
        raise ValueError("Value of gamma should be greater than or equal to zero")

    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.convert_to_tensor(y_true, dtype=y_pred.dtype)

    ce = K.binary_crossentropy(y_true, y_pred, from_logits=from_logits)

    if from_logits:
        pred_prob = tf.sigmoid(y_pred)
    else:
        pred_prob = y_pred

    p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
    alpha_factor = 1.0
    modulating_factor = 1.0

    if alpha:
        alpha = tf.convert_to_tensor(alpha, dtype=K.floatx())
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

    if gamma:
        gamma = tf.convert_to_tensor(gamma, dtype=K.floatx())
        modulating_factor = tf.pow((1.0 - p_t), gamma)

    return tf.reduce_sum(alpha_factor * modulating_factor * ce, axis=-1)

## === cell 23
def build_efficient_model(input_layer, input_shape, model_inputs, num_classes, dropout_rate=0.2):

    model = tf.keras.applications.EfficientNetB3(weights='/kaggle/input/efficientnetb3notop/efficientnetb3_notop.h5', 
                              include_top=False, 
                                input_shape=input_shape, 
                              drop_connect_rate=dropout_rate)
    
    model.trainable = False
    model_output = model(model_inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(model_output)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="pred")(x)

    model = tf.keras.Model(input_layer, outputs, name="EfficientNet")
    return model

## === cell 24
input_shape = (300, 300, 3)
dropout_rate=0.2
num_classes = len(class_map)

## === cell 25

input_layer = tf.keras.layers.Input([None, None, 3], dtype = tf.uint8)
x = tf.cast(input_layer, tf.float32)
x = data_augmentation(x, training=False)
x = tf.keras.layers.experimental.preprocessing.Resizing(input_shape[0], input_shape[1])(x)

base_model = tf.keras.applications.EfficientNetB3(weights='/kaggle/input/efficientnetb3notop/efficientnetb3_notop.h5', 
                              include_top=False, 
                                input_shape=input_shape, 
                              drop_connect_rate=dropout_rate)
    
base_model.trainable = False
model_output = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D(name="avg_pool")(model_output)

outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="pred")(x)

model = tf.keras.Model(input_layer, outputs, name="EfficientNet")
    


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2561759169.py in <cell line: 0>()
      2 
      3 input_layer = tf.keras.layers.Input([None, None, 3], dtype = tf.uint8)
----> 4 x = tf.cast(input_layer, tf.float32)
      5 x = data_augmentation(x, training=False)
      6 x = tf.keras.layers.experimental.preprocessing.Resizing(input_shape[0], input_shape[1])(x)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/keras_tensor.py in __tf_tensor__(self, dtype, name)
    136 
    137     def __tf_tensor__(self, dtype=None, name=None):
--> 138         raise ValueError(
    139             "A KerasTensor cannot be used as input to a TensorFlow function. "
    140             "A KerasTensor is a symbolic placeholder for a shape and dtype, "

ValueError: A KerasTensor cannot be used as input to a TensorFlow function. A KerasTensor is a symbolic placeholder for a shape and dtype, used when constructing Keras Functional models or Keras Functions. You can only use it as input to a Keras layer or a Keras operation (from the namespaces `keras.layers` and `keras.operations`). You are likely doing something like:

```
x = Input(...)
...
tf_fn(x)  # Invalid.
```

What you should do instead is wrap `tf_fn` in a layer:

```
class MyLayer(Layer):
    def call(self, x):
        return tf_fn(x)

x = MyLayer()(x)
```


## === cell 26
model.summary()

## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3470139634.py in <cell line: 0>()
----> 1 model.summary()

NameError: name 'model' is not defined

## === cell 27
layers = [layer.name for layer in base_model.layers]
layers.index('block7a_expand_conv'), len(layers)

## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3025202261.py in <cell line: 0>()
----> 1 layers = [layer.name for layer in base_model.layers]
      2 layers.index('block7a_expand_conv'), len(layers)

NameError: name 'base_model' is not defined

## === cell 28
METRICS = [
      tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc')
]

optimizer = tf.keras.optimizers.Adam()



## === cell 29
model.compile(
    optimizer=optimizer, loss='categorical_crossentropy', metrics=METRICS
)

## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3317927513.py in <cell line: 0>()
----> 1 model.compile(
      2     optimizer=optimizer, loss='categorical_crossentropy', metrics=METRICS
      3 )

NameError: name 'model' is not defined

## === cell 30
hist = model.fit(train_ds, epochs=5, validation_data=val_ds, callbacks=callbacks)

## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1708001413.py in <cell line: 0>()
----> 1 hist = model.fit(train_ds, epochs=5, validation_data=val_ds, callbacks=callbacks)

NameError: name 'model' is not defined

## === cell 31
plot_metrics(hist, ['loss', 'auc', 'precision', 'recall'])

## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2333113305.py in <cell line: 0>()
----> 1 plot_metrics(hist, ['loss', 'auc', 'precision', 'recall'])

NameError: name 'hist' is not defined

## === cell 32
val_preds = model.predict(val_ds)
val_preds = tf.math.argmax(val_preds, -1)

## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3907500812.py in <cell line: 0>()
----> 1 val_preds = model.predict(val_ds)
      2 val_preds = tf.math.argmax(val_preds, -1)

NameError: name 'model' is not defined

## === cell 33
plot_cm(val.label.values, val_preds.numpy())

## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4262013335.py in <cell line: 0>()
----> 1 plot_cm(val.label.values, val_preds.numpy())

NameError: name 'val_preds' is not defined

## === cell 34
np.mean(np.where(val.label.values==val_preds, 1, 0))

## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/177811116.py in <cell line: 0>()
----> 1 np.mean(np.where(val.label.values==val_preds, 1, 0))

NameError: name 'val_preds' is not defined

## === cell 35
fine_tune_at = layers.index('block7a_expand_conv')
base_model.trainable = True
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable =  False

model.compile(
    optimizer=tf.keras.optimizers.Adam(tf.keras.experimental.CosineDecay(1e-4, 300*10)), 
    loss='categorical_crossentropy', metrics=METRICS
)

## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2583744942.py in <cell line: 0>()
----> 1 fine_tune_at = layers.index('block7a_expand_conv')
      2 base_model.trainable = True
      3 for layer in base_model.layers[:fine_tune_at]:
      4     layer.trainable =  False
      5 

NameError: name 'layers' is not defined

## === cell 36
fine_tune_epochs = 5
total_epochs =  5 + fine_tune_epochs

history_fine = model.fit(train_ds,
                         epochs=total_epochs,
                         initial_epoch=hist.epoch[-1], 
                         validation_data=val_ds, callbacks=callbacks)

## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3355142980.py in <cell line: 0>()
      2 total_epochs =  5 + fine_tune_epochs
      3 
----> 4 history_fine = model.fit(train_ds,
      5                          epochs=total_epochs,
      6                          initial_epoch=hist.epoch[-1],

NameError: name 'model' is not defined

## === cell 42
@tf.function
def process_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3) 
    return img

## === cell 43
TEST_FILENAMES = tf.io.gfile.glob('../input/cassava-leaf-disease-classification/test_images/*.jpg')


## === cell 44
test_ds = tf.data.Dataset.from_tensor_slices((TEST_FILENAMES))\
                .map(process_img, num_parallel_calls = tf.data.experimental.AUTOTUNE)\
                .batch(batch_size)\
                .prefetch(tf.data.experimental.AUTOTUNE) 

## === cell 45
probabilities = model.predict(test_ds)
predictions = np.argmax(probabilities, axis=-1)


## --- ERROR in cell 45, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2529608989.py in <cell line: 0>()
----> 1 probabilities = model.predict(test_ds)
      2 predictions = np.argmax(probabilities, axis=-1)
      3 # print(predictions)

NameError: name 'model' is not defined

## === cell 46
test_ids = [os.path.split(path)[1] for path in TEST_FILENAMES]
submission = pd.DataFrame({'image_id': test_ids, 'label': predictions})

submission.to_csv('submission.csv', index = False)
submission

## --- ERROR in cell 46, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1278110710.py in <cell line: 0>()
      1 test_ids = [os.path.split(path)[1] for path in TEST_FILENAMES]
      2 # test_ids
----> 3 submission = pd.DataFrame({'image_id': test_ids, 'label': predictions})
      4 
      5 submission.to_csv('submission.csv', index = False)

NameError: name 'predictions' is not defined
