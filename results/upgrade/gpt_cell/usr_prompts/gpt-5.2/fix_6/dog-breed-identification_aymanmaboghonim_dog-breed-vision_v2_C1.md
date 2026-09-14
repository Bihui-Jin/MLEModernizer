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

No external packages required in the script and installed.

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
        input/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
        working/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
```

-> data/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> data/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> input/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> input/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from pandas_profiling import ProfileReport
except Exception:
    ProfileReport = None

import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_ver

    _major = int(str(_pb_ver).split(".", 1)[0])
except Exception:
    _major = None

if _major is not None and _major >= 4:
    import sys
    import subprocess

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.3"]
    )

import tensorflow as tf
import tensorflow_hub as hub

from IPython.display import display, Image
from matplotlib.pyplot import imread

from sklearn.model_selection import train_test_split


## === cell 1
print("TF version:", tf.__version__)
print("GPU", "available (YESS!!!!)" if tf.config.list_physical_devices("GPU") else "not available :(")


## === cell 2
labels_csv = pd.read_csv("../input/dog-breed-identification/labels.csv")


## === cell 3
Labels_Profile = ProfileReport(labels_csv, progress_bar= False)
Labels_Profile.to_notebook_iframe()


## === cell 4
labels_csv["breed"].value_counts().plot.bar(figsize=(20, 10));


## === cell 5
filenames = ["../input/dog-breed-identification/train/" + fname + ".jpg" for fname in labels_csv["id"]]

filenames[:5]


## === cell 6
if len(os.listdir("../input/dog-breed-identification/train/")) == len(filenames):
  print("Filenames match actual amount of files!")
else:
  print("Filenames do not match actual amount of files, check the target directory.")


## === cell 7
Image(filenames[42])


## === cell 8
labels = labels_csv["breed"].to_numpy() 
labels[:10]


## === cell 9
if len(labels) == len(filenames):
  print("Number of labels matches number of filenames!")
else:
  print("Number of labels does not match number of filenames, check data directories.")


## === cell 10
unique_breeds = np.unique(labels)
len(unique_breeds)


## === cell 11
boolean_labels = [label == np.array(unique_breeds) for label in labels]
boolean_labels[:2]


## === cell 12
X = filenames
y = boolean_labels


## === cell 13
NUM_IMAGES = 1000 

X_train, X_val, y_train, y_val = train_test_split(X[:NUM_IMAGES],
                                                  y[:NUM_IMAGES], 
                                                  test_size=0.2,
                                                  random_state=42)

len(X_train), len(y_train), len(X_val), len(y_val)


## === cell 14
IMG_SIZE = 224

def process_image(image_path):
  """
  Takes an image file path and turns it into a Tensor.
  """
  image = tf.io.read_file(image_path)
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
  return image


## === cell 15
def get_image_label(image_path, label):
  """
  Takes an image file path name and the associated label,
  processes the image and returns a tuple of (image, label).
  """
  image = process_image(image_path)
  return image, label


## === cell 16
BATCH_SIZE = 32

def create_data_batches(x, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):
  """
  Creates batches of data out of image (x) and label (y) pairs.
  Shuffles the data if it's training data but doesn't shuffle it if it's validation data.
  Also accepts test data as input (no labels).
  """
  if test_data:
    print("Creating test data batches...")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x))) # only filepaths
    data_batch = data.map(process_image).batch(BATCH_SIZE)
    return data_batch
  
  elif valid_data:
    print("Creating validation data batches...")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x), # filepaths
                                               tf.constant(y))) # labels
    data_batch = data.map(get_image_label).batch(BATCH_SIZE)
    return data_batch

  else:
    print("Creating training data batches...")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x), # filepaths
                                              tf.constant(y))) # labels
    
    data = data.shuffle(buffer_size=len(x))

    data = data.map(get_image_label)

    data_batch = data.batch(BATCH_SIZE)
  return data_batch


## === cell 17
train_data = create_data_batches(X_train, y_train)
val_data = create_data_batches(X_val, y_val, valid_data=True)


## === cell 18
train_data.element_spec, val_data.element_spec


## === cell 19
def show_25_images(images, labels):
  """
  Displays 25 images from a data batch.
  """
  plt.figure(figsize=(10, 10))
  for i in range(25):
    ax = plt.subplot(5, 5, i+1)
    plt.imshow(images[i])
    plt.title(unique_breeds[labels[i].argmax()])
    plt.axis("off")


## === cell 20
train_images, train_labels = next(train_data.as_numpy_iterator())
show_25_images(train_images, train_labels)


## === cell 21
INPUT_SHAPE = [None, IMG_SIZE, IMG_SIZE, 3] # batch, height, width, colour channel ass

OUTPUT_SHAPE = len(unique_breeds) # number of unique labels

MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"


## === cell 22
def create_model(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, model_url=MODEL_URL):
  print("Building model with:", MODEL_URL)

  model = tf.keras.Sequential([
    hub.KerasLayer(MODEL_URL), # Layer 1 (the pre-trained model )
    tf.keras.layers.Dense(units=OUTPUT_SHAPE, 
                          activation="softmax") # Layer 2 (output layer)
  ])

  model.compile(
      loss=tf.keras.losses.CategoricalCrossentropy(), 
      optimizer=tf.keras.optimizers.Adam(), 
      metrics=["accuracy"] 
  )

  model.build(INPUT_SHAPE) 
  
  return model


## === cell 23
def create_model(
    input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, model_url=MODEL_URL
):
    print("Building model with:", MODEL_URL)

    inputs = tf.keras.Input(shape=input_shape[1:], name="input_image")
    x = hub.KerasLayer(model_url, name="hub_layer")(inputs)
    outputs = tf.keras.layers.Dense(
        units=output_shape, activation="softmax", name="predictions"
    )(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="hub_classifier")

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )

    return model


model = create_model()
model.summary()


## --- ERROR in cell 23, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/4261207462.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     23[0m [0;34m[0m[0m
[1;32m     24[0m [0;34m[0m[0m
[0;32m---> 25[0;31m [0mmodel[0m [0;34m=[0m [0mcreate_model[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     26[0m [0mmodel[0m[0;34m.[0m[0msummary[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/4261207462.py[0m in [0;36mcreate_model[0;34m(input_shape, output_shape, model_url)[0m
[1;32m      8[0m     [0;31m# (Hub -> Dense softmax) while avoiding Sequential's strict layer instance restriction.[0m[0;34m[0m[0;34m[0m[0m
[1;32m      9[0m     [0minputs[0m [0;34m=[0m [0mtf[0m[0;34m.[0m[0mkeras[0m[0;34m.[0m[0mInput[0m[0;34m([0m[0mshape[0m[0;34m=[0m[0minput_shape[0m[0;34m[[0m[0;36m1[0m[0;34m:[0m[0;34m][0m[0;34m,[0m [0mname[0m[0;34m=[0m[0;34m"input_image"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 10[0;31m     [0mx[0m [0;34m=[0m [0mhub[0m[0;34m.[0m[0mKerasLayer[0m[0;34m([0m[0mmodel_url[0m[0;34m,[0m [0mname[0m[0;34m=[0m[0;34m"hub_layer"[0m[0;34m)[0m[0;34m([0m[0minputs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     11[0m     outputs = tf.keras.layers.Dense(
[1;32m     12[0m         [0munits[0m[0;34m=[0m[0moutput_shape[0m[0;34m,[0m [0mactivation[0m[0;34m=[0m[0;34m"softmax"[0m[0;34m,[0m [0mname[0m[0;34m=[0m[0;34m"predictions"[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py[0m in [0;36merror_handler[0;34m(*args, **kwargs)[0m
[1;32m     68[0m             [0;31m# To get the full stack trace, call:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     69[0m             [0;31m# `tf.debugging.disable_traceback_filtering()`[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 70[0;31m             [0;32mraise[0m [0me[0m[0;34m.[0m[0mwith_traceback[0m[0;34m([0m[0mfiltered_tb[0m[0;34m)[0m [0;32mfrom[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     71[0m         [0;32mfinally[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     72[0m             [0;32mdel[0m [0mfiltered_tb[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow_hub/keras_layer.py[0m in [0;36mcall[0;34m(self, inputs, training)[0m
[1;32m    248[0m         [0;31m# Behave like BatchNormalization. (Dropout is different, b/181839368.)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    249[0m         [0mtraining[0m [0;34m=[0m [0;32mFalse[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 250[0;31m       result = smart_cond.smart_cond(training,
[0m[1;32m    251[0m                                      [0;32mlambda[0m[0;34m:[0m [0mf[0m[0;34m([0m[0mtraining[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    252[0m                                      lambda: f(training=False))

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow_hub/keras_layer.py[0m in [0;36m<lambda>[0;34m()[0m
[1;32m    250[0m       result = smart_cond.smart_cond(training,
[1;32m    251[0m                                      [0;32mlambda[0m[0;34m:[0m [0mf[0m[0;34m([0m[0mtraining[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 252[0;31m                                      lambda: f(training=False))
[0m[1;32m    253[0m [0;34m[0m[0m
[1;32m    254[0m     [0;31m# Unwrap dicts returned by signatures.[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/core/function/polymorphism/function_type.py[0m in [0;36mcanonicalize_to_monomorphic[0;34m(args, kwargs, default_values, capture_types, polymorphic_type)[0m
[1;32m    581[0m     [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    582[0m       parameters.append(
[0;32m--> 583[0;31m           _make_validated_mono_param(name, arg, poly_parameter.kind,
[0m[1;32m    584[0m                                      [0mtype_context[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    585[0m                                      poly_parameter.type_constraint))

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/core/function/polymorphism/function_type.py[0m in [0;36m_make_validated_mono_param[0;34m(name, value, kind, type_context, poly_type)[0m
[1;32m    520[0m ) -> Parameter:
[1;32m    521[0m   [0;34m"""Generates and validates a parameter for Monomorphic FunctionType."""[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 522[0;31m   [0mmono_type[0m [0;34m=[0m [0mtrace_type[0m[0;34m.[0m[0mfrom_value[0m[0;34m([0m[0mvalue[0m[0;34m,[0m [0mtype_context[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    523[0m [0;34m[0m[0m
[1;32m    524[0m   [0;32mif[0m [0mpoly_type[0m [0;32mand[0m [0;32mnot[0m [0mmono_type[0m[0;34m.[0m[0mis_subtype_of[0m[0;34m([0m[0mpoly_type[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/tensorflow/core/function/trace_type/trace_type_builder.py[0m in [0;36mfrom_value[0;34m(value, context)[0m
[1;32m    183[0m [0;34m[0m[0m
[1;32m    184[0m   [0;32mif[0m [0mutil[0m[0;34m.[0m[0mis_np_ndarray[0m[0;34m([0m[0mvalue[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 185[0;31m     [0mndarray[0m [0;34m=[0m [0mvalue[0m[0;34m.[0m[0m__array__[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    186[0m     [0;32mreturn[0m [0mdefault_types[0m[0;34m.[0m[0mTENSOR[0m[0;34m([0m[0mndarray[0m[0;34m.[0m[0mshape[0m[0;34m,[0m [0mndarray[0m[0;34m.[0m[0mdtype[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    187[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/keras_tensor.py[0m in [0;36m__array__[0;34m(self)[0m
[1;32m    106[0m [0;34m[0m[0m
[1;32m    107[0m     [0;32mdef[0m [0m__array__[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 108[0;31m         raise ValueError(
[0m[1;32m    109[0m             [0;34m"A KerasTensor is symbolic: it's a placeholder for a shape "[0m[0;34m[0m[0;34m[0m[0m
[1;32m    110[0m             [0;34m"an a dtype. It doesn't have any actual numerical value. "[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: Exception encountered when calling layer 'hub_layer' (type KerasLayer).

A KerasTensor is symbolic: it's a placeholder for a shape an a dtype. It doesn't have any actual numerical value. You cannot convert it to a NumPy array.

Call arguments received by layer 'hub_layer' (type KerasLayer):
  • inputs=<KerasTensor shape=(None, 224, 224, 3), dtype=float32, sparse=False, name=input_image>
  • training=None

## === cell 24
%load_ext tensorboard
