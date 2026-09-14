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
Given a dataset of images of dogs, predict the breed of each image.

## Metric
Multi Class Log Loss.

## Submission Format
For each image in the test set, you must predict a probability for each of the different breeds. The file should contain a header and have the following format:
```
id,affenpinscher,afghan_hound,..,yorkshire_terrier
000621fb3cbb32d8935728e48679680e,0.0083,0.0,...,0.0083
etc.
```

## Dataset Description
- `train.zip` - the training set, you are provided the breed for these dogs
- `test.zip` - the test set, you must predict the probability of each breed for each image
- `sample_submission.csv` - a sample submission file in the correct format
- `labels.csv` - the breeds for the images in the train set

# 2. Python version

3.11

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

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

# 5. Target score

0.7623100731269614

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")



## === cell 1
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import pandas as pd
import numpy as np
import PIL
import pathlib, os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow:", tf.__version__)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
label_df = pd.read_csv("/kaggle/input/dog-breed-identification/labels.csv")
label_df.head()



## === cell 3
label_df.shape



## === cell 4
label_df.breed.value_counts().plot.bar(figsize=(18, 5))



## === cell 5
np.max(label_df.breed.value_counts()), np.min(label_df.breed.value_counts())



## === cell 6
label_df.breed.value_counts().plot.hist()



## === cell 7
label_df.breed.value_counts().median()



## === cell 8
im = PIL.Image.open(
    "/kaggle/input/dog-breed-identification/train/000bec180eb18c7604dcecc8fe0dba07.jpg"
)
im



## === cell 9
im = PIL.Image.open(
    f"/kaggle/input/dog-breed-identification/train/{label_df.id[6]}.jpg"
)
im



## === cell 10
img_path = []
for name in label_df["id"]:
    img_path.append(f"/kaggle/input/dog-breed-identification/train/{name}.jpg")
img_path[:2]



## === cell 11
PIL.Image.open(img_path[9])  # ok got it !!!



## === cell 12
len(os.listdir("/kaggle/input/dog-breed-identification/train")) == len(img_path)



## === cell 13
label_df.shape, len(os.listdir("/kaggle/input/dog-breed-identification/train"))



## === cell 14
labels = label_df.breed.to_numpy()
labels, len(labels)



## === cell 15
len(label_df["breed"].unique())



## === cell 16
label_df.id[60], label_df.breed[60]




## === cell 17
def showImage(path):
    plt.imshow(PIL.Image.open(path))


showImage(img_path[60])



## === cell 18
np.asarray(PIL.Image.open(img_path[60])).shape



## === cell 19
samp_i = PIL.Image.open(img_path[60])
print(samp_i.format)
print(samp_i.size)  # w (as columns) x h (as rows)



## === cell 20
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(labels)



## === cell 21
len(le.classes_)



## === cell 22
label_class = le.classes_
label_tf = le.transform(labels)

label_class[:5], label_tf[:5]



## === cell 23
X = img_path
y = label_tf

len(X), len(y)



## === cell 24
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    X[:1000],
    y[:1000],
    test_size=0.2,
    random_state=42,
)

len(X_train), len(X_valid)



## === cell 25
samp = PIL.Image.open(img_path[42])
np.asanyarray(samp).shape  # row, col, channel (R,G,B)



## === cell 26
image = imread(img_path[42])  # output np array (257, 350, 3)
plt.imshow(image)



## === cell 27
image.max(), image.min()  # 8 bit 0 - 255



## === cell 28
batch_size = 32
img_height = 224
img_width = 224



## === cell 29
ts = tf.io.read_file(img_path[42])
img = tf.io.decode_jpeg(ts, channels=3)
tf.image.resize(img, [img_height, img_width]).shape




## === cell 30
def preprocess_img(sel_path):
    image_raw = tf.io.read_file(sel_path)
    img_dec = tf.io.decode_jpeg(image_raw, channels=3)
    img_nor = tf.image.convert_image_dtype(img_dec, dtype=tf.float32)
    img_re = tf.image.resize(img_nor, [img_height, img_width])
    return img_re


t_img = preprocess_img(img_path[42])
t_img[:2]




## === cell 31
def process_path(file_path, labels):
    label = labels
    img = preprocess_img(file_path)
    return img, label


process_path(X[0], y[0])[0].shape, process_path(X[0], y[0])[1]




## === cell 32
def create_data_batch(X, y=None, valid_data=False, test_data=False):
    if test_data:
        print("creating batchs for testing set...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X)))
        data_batch = (
            data.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        return data_batch
    elif valid_data:
        print("creating batchs for validation set...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
        data_batch = (
            data.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        return data_batch
    else:
        print("creating batchs for training set...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
        data = data.shuffle(buffer_size=1000)  # for training set I'll shuffle
        data = data.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
        data_batch = data.batch(batch_size).prefetch(
            tf.data.AUTOTUNE
        )  # return tuple (img, label)
        return data_batch


train_data = create_data_batch(X_train, y_train)
valid_data = create_data_batch(X_valid, y_valid, valid_data=True)

train_data.element_spec, valid_data.element_spec




## === cell 33
def batch_img_show(data_set):
    image_batch, label_batch = next(iter(data_set))

    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image_batch[i])
        label = int(label_batch[i])
        l_tf = le.inverse_transform([label])
        plt.title(l_tf[0])
        plt.axis("off")


batch_img_show(train_data)  # the img will be shuffled when you run it again



## === cell 34
import tensorflow_hub as hub

INPUT_SHAPE = [None, img_height, img_width, 3]
OUTPUT_SHAPE = len(label_df.breed.unique())
MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/5"




## === cell 35
def create_model(model_url):
    print("Building the model ...")
    print(f"with {model_url}")

    inputs = tf.keras.Input(shape=(img_height, img_width, 3), name="image")
    x = hub.KerasLayer(model_url, trainable=False, name="tfhub_backbone")(inputs)
    outputs = tf.keras.layers.Dense(
        units=OUTPUT_SHAPE, activation="softmax", name="breed_softmax"
    )(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="dog_breed_model")

    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model


model = create_model(MODEL_URL)
model.summary()



## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3338806074.py in <cell line: 0>()
     19 
     20 
---> 21 model = create_model(MODEL_URL)
     22 model.summary()
     23 

/tmp/ipykernel_55/3338806074.py in create_model(model_url)
      7 
      8     inputs = tf.keras.Input(shape=(img_height, img_width, 3), name="image")
----> 9     x = hub.KerasLayer(model_url, trainable=False, name="tfhub_backbone")(inputs)
     10     outputs = tf.keras.layers.Dense(
     11         units=OUTPUT_SHAPE, activation="softmax", name="breed_softmax"

/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
     68             # To get the full stack trace, call:
     69             # `tf.debugging.disable_traceback_filtering()`
---> 70             raise e.with_traceback(filtered_tb) from None
     71         finally:
     72             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow_hub/keras_layer.py in call(self, inputs, training)
    248         # Behave like BatchNormalization. (Dropout is different, b/181839368.)
    249         training = False
--> 250       result = smart_cond.smart_cond(training,
    251                                      lambda: f(training=True),
    252                                      lambda: f(training=False))

/usr/local/lib/python3.11/dist-packages/tensorflow_hub/keras_layer.py in <lambda>()
    250       result = smart_cond.smart_cond(training,
    251                                      lambda: f(training=True),
--> 252                                      lambda: f(training=False))
    253 
    254     # Unwrap dicts returned by signatures.

/usr/local/lib/python3.11/dist-packages/tensorflow/core/function/polymorphism/function_type.py in canonicalize_to_monomorphic(args, kwargs, default_values, capture_types, polymorphic_type)
    581     else:
    582       parameters.append(
--> 583           _make_validated_mono_param(name, arg, poly_parameter.kind,
    584                                      type_context,
    585                                      poly_parameter.type_constraint))

/usr/local/lib/python3.11/dist-packages/tensorflow/core/function/polymorphism/function_type.py in _make_validated_mono_param(name, value, kind, type_context, poly_type)
    520 ) -> Parameter:
    521   """Generates and validates a parameter for Monomorphic FunctionType."""
--> 522   mono_type = trace_type.from_value(value, type_context)
    523 
    524   if poly_type and not mono_type.is_subtype_of(poly_type):

/usr/local/lib/python3.11/dist-packages/tensorflow/core/function/trace_type/trace_type_builder.py in from_value(value, context)
    183 
    184   if util.is_np_ndarray(value):
--> 185     ndarray = value.__array__()
    186     return default_types.TENSOR(ndarray.shape, ndarray.dtype)
    187 

/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/keras_tensor.py in __array__(self)
    106 
    107     def __array__(self):
--> 108         raise ValueError(
    109             "A KerasTensor is symbolic: it's a placeholder for a shape "
    110             "an a dtype. It doesn't have any actual numerical value. "

ValueError: Exception encountered when calling layer 'tfhub_backbone' (type KerasLayer).

A KerasTensor is symbolic: it's a placeholder for a shape an a dtype. It doesn't have any actual numerical value. You cannot convert it to a NumPy array.

Call arguments received by layer 'tfhub_backbone' (type KerasLayer):
  • inputs=<KerasTensor shape=(None, 224, 224, 3), dtype=float32, sparse=False, name=image>
  • training=None

## === cell 36
import datetime


def create_tensorboard_callback():
    log_dir = "/kaggle/working/log/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )
    return tensorboard_callback


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=4, verbose=1, restore_best_weights=True
)

"Yes" if tf.config.list_physical_devices("GPU") else "No GPU available"



## === cell 37
tensorboard = create_tensorboard_callback()

_ = model.fit(
    x=train_data,
    validation_data=valid_data,
    epochs=100,
    validation_freq=1,
    callbacks=[tensorboard, early_stopping],
    verbose=2,
)



## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2194926110.py in <cell line: 0>()
      2 
      3 # Quick sanity training on 1000 images (original intent); this is not used for final submission.
----> 4 _ = model.fit(
      5     x=train_data,
      6     validation_data=valid_data,

NameError: name 'model' is not defined

## === cell 38
pred = model.predict(valid_data, verbose=1)
pred.shape



## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3522407625.py in <cell line: 0>()
----> 1 pred = model.predict(valid_data, verbose=1)
      2 pred.shape
      3 

NameError: name 'model' is not defined

## === cell 39
model.evaluate(valid_data, verbose=2)




## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/24012152.py in <cell line: 0>()
      1 # Evaluate sanity-check model
----> 2 model.evaluate(valid_data, verbose=2)
      3 
      4 

NameError: name 'model' is not defined

## === cell 40
def save_model(model, suffix=None):
    model_dir = os.path.join(
        "/kaggle/working/models", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    os.makedirs(os.path.dirname(model_dir), exist_ok=True)
    model_path = model_dir + "-" + (suffix if suffix else "model") + ".h5"
    print(f"Saving model to: {model_path}")
    model.save(model_path)
    return model_path


def load_model(model_path):
    print(f'Loading model from: "{model_path}"')
    model = tf.keras.models.load_model(
        model_path, custom_objects={"KerasLayer": hub.KerasLayer}
    )
    return model




## === cell 41
print(f"Full datasets X,y = {len(X)}, {len(y)}")

full_data_batch = create_data_batch(X, y)

full_model = create_model(MODEL_URL)

full_model_early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="accuracy", patience=3, verbose=1, restore_best_weights=True
)



## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3771069510.py in <cell line: 0>()
      4 full_data_batch = create_data_batch(X, y)
      5 
----> 6 full_model = create_model(MODEL_URL)
      7 
      8 full_model_early_stopping = tf.keras.callbacks.EarlyStopping(

/tmp/ipykernel_55/3338806074.py in create_model(model_url)
      7 
      8     inputs = tf.keras.Input(shape=(img_height, img_width, 3), name="image")
----> 9     x = hub.KerasLayer(model_url, trainable=False, name="tfhub_backbone")(inputs)
     10     outputs = tf.keras.layers.Dense(
     11         units=OUTPUT_SHAPE, activation="softmax", name="breed_softmax"

/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
     68             # To get the full stack trace, call:
     69             # `tf.debugging.disable_traceback_filtering()`
---> 70             raise e.with_traceback(filtered_tb) from None
     71         finally:
     72             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow_hub/keras_layer.py in call(self, inputs, training)
    248         # Behave like BatchNormalization. (Dropout is different, b/181839368.)
    249         training = False
--> 250       result = smart_cond.smart_cond(training,
    251                                      lambda: f(training=True),
    252                                      lambda: f(training=False))

/usr/local/lib/python3.11/dist-packages/tensorflow_hub/keras_layer.py in <lambda>()
    250       result = smart_cond.smart_cond(training,
    251                                      lambda: f(training=True),
--> 252                                      lambda: f(training=False))
    253 
    254     # Unwrap dicts returned by signatures.

/usr/local/lib/python3.11/dist-packages/tensorflow/core/function/polymorphism/function_type.py in canonicalize_to_monomorphic(args, kwargs, default_values, capture_types, polymorphic_type)
    581     else:
    582       parameters.append(
--> 583           _make_validated_mono_param(name, arg, poly_parameter.kind,
    584                                      type_context,
    585                                      poly_parameter.type_constraint))

/usr/local/lib/python3.11/dist-packages/tensorflow/core/function/polymorphism/function_type.py in _make_validated_mono_param(name, value, kind, type_context, poly_type)
    520 ) -> Parameter:
    521   """Generates and validates a parameter for Monomorphic FunctionType."""
--> 522   mono_type = trace_type.from_value(value, type_context)
    523 
    524   if poly_type and not mono_type.is_subtype_of(poly_type):

/usr/local/lib/python3.11/dist-packages/tensorflow/core/function/trace_type/trace_type_builder.py in from_value(value, context)
    183 
    184   if util.is_np_ndarray(value):
--> 185     ndarray = value.__array__()
    186     return default_types.TENSOR(ndarray.shape, ndarray.dtype)
    187 

/usr/local/lib/python3.11/dist-packages/keras/src/backend/common/keras_tensor.py in __array__(self)
    106 
    107     def __array__(self):
--> 108         raise ValueError(
    109             "A KerasTensor is symbolic: it's a placeholder for a shape "
    110             "an a dtype. It doesn't have any actual numerical value. "

ValueError: Exception encountered when calling layer 'tfhub_backbone' (type KerasLayer).

A KerasTensor is symbolic: it's a placeholder for a shape an a dtype. It doesn't have any actual numerical value. You cannot convert it to a NumPy array.

Call arguments received by layer 'tfhub_backbone' (type KerasLayer):
  • inputs=<KerasTensor shape=(None, 224, 224, 3), dtype=float32, sparse=False, name=image>
  • training=None

## === cell 42
_ = full_model.fit(
    x=full_data_batch, epochs=20, callbacks=[full_model_early_stopping], verbose=2
)



## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1551078471.py in <cell line: 0>()
      1 # NOTE: Keeping the same core training approach (fit on dataset batches).
      2 # No early stopping relaxation introduced beyond what was already present in the notebook.
----> 3 _ = full_model.fit(
      4     x=full_data_batch, epochs=20, callbacks=[full_model_early_stopping], verbose=2
      5 )

NameError: name 'full_model' is not defined

## === cell 43
test_path = "/kaggle/input/dog-breed-identification/test"

dir_list = os.listdir(test_path)
file_name = [os.path.splitext(name)[0] for name in dir_list]
print("Num test images:", len(file_name))

sample_id = 499
plt.imshow(imread(f"{test_path}/{dir_list[sample_id]}"))
plt.axis("off")



## === cell 44
test_img_path = [f"{test_path}/{name}" for name in dir_list]
test_data = create_data_batch(test_img_path, test_data=True)
test_data



## === cell 45
y_pred = full_model.predict(test_data, verbose=1)
y_pred.shape



## --- ERROR in cell 45, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3736815924.py in <cell line: 0>()
      1 # Predict probabilities for all classes (already softmax).
----> 2 y_pred = full_model.predict(test_data, verbose=1)
      3 y_pred.shape
      4 

NameError: name 'full_model' is not defined

## === cell 46
sample_sub = pd.read_csv("/kaggle/input/dog-breed-identification/sample_submission.csv")

preds_submit = pd.DataFrame({"id": file_name})
breed_pred = pd.DataFrame(y_pred, columns=le.classes_)
preds_submit = pd.concat([preds_submit, breed_pred], axis=1)

preds_submit = preds_submit.set_index("id")
sample_sub = sample_sub.set_index("id")

preds_submit = preds_submit.reindex(sample_sub.index)
if preds_submit.isnull().values.any():
    fill = np.full(
        (preds_submit.shape[0], preds_submit.shape[1]),
        1.0 / preds_submit.shape[1],
        dtype=np.float32,
    )
    preds_submit = preds_submit.fillna(
        pd.DataFrame(fill, index=preds_submit.index, columns=preds_submit.columns)
    )

preds_submit = preds_submit[sample_sub.columns]

submission = preds_submit.reset_index()
submission.head(), submission.shape



## --- ERROR in cell 46, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2298223059.py in <cell line: 0>()
      4 
      5 preds_submit = pd.DataFrame({"id": file_name})
----> 6 breed_pred = pd.DataFrame(y_pred, columns=le.classes_)
      7 preds_submit = pd.concat([preds_submit, breed_pred], axis=1)
      8 

NameError: name 'y_pred' is not defined

## === cell 47
out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(
    "Columns:", submission.columns[:5].tolist(), "... total:", len(submission.columns)
)
print("Shape:", submission.shape)

## --- ERROR in cell 47, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3199655450.py in <cell line: 0>()
      1 # Write valid Kaggle submission
      2 out_path = "/kaggle/working/submission.csv"
----> 3 submission.to_csv(out_path, index=False)
      4 print("Wrote:", out_path)
      5 print(

NameError: name 'submission' is not defined
