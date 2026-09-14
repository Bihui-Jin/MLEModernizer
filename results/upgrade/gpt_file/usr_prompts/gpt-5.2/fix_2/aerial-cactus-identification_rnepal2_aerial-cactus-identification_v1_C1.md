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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
protobuf==6.33.0
scikit-image==0.25.2
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
tf_keras==2.18.0

# 4. Data file paths

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

# 5. Target score

0.7993

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, math, time, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2
from glob import glob
import tensorflow as tf
from sklearn.utils import shuffle

from skimage.io import imread
from skimage import color
from skimage.transform import resize

from PIL import Image as pil_image

import warnings

warnings.filterwarnings("ignore")

from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
IMG_SIZE = 32  # in the given original size



## === cell 2
BASE_PATH = "/kaggle/input/aerial-cactus-identification"
train_folder = os.path.join(BASE_PATH, "train", "train")
test_folder = os.path.join(BASE_PATH, "test", "test")
train_csv_path = os.path.join(BASE_PATH, "train.csv")
sample_sub_path = os.path.join(BASE_PATH, "sample_submission.csv")

print("BASE_PATH exists:", os.path.exists(BASE_PATH))
print("train_folder:", train_folder, "exists:", os.path.exists(train_folder))
print("test_folder :", test_folder, "exists:", os.path.exists(test_folder))
print("train.csv   :", train_csv_path, "exists:", os.path.exists(train_csv_path))
print("sample_sub  :", sample_sub_path, "exists:", os.path.exists(sample_sub_path))

print("train images:", len(os.listdir(train_folder)))
print("test images :", len(os.listdir(test_folder)))



## === cell 3
train_df = pd.read_csv(train_csv_path)
train_df.head()



## === cell 4
train_images_path = glob(os.path.join(train_folder, "*.jpg"))
test_images_path = glob(os.path.join(test_folder, "*.jpg"))
len(train_images_path), len(test_images_path)




## === cell 5
def expand_path(filename):
    p_train = os.path.join(train_folder, filename)
    if os.path.isfile(p_train):
        return p_train
    p_test = os.path.join(test_folder, filename)
    if os.path.isfile(p_test):
        return p_test
    return filename


def pil_image_load(image):
    image_path = expand_path(image)
    image = pil_image.open(image_path)
    return image.resize((IMG_SIZE, IMG_SIZE))




## === cell 6
pil_image_load(os.path.basename(train_images_path[10]))



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/2179213116.py in <cell line: 0>()
      1 # Quick sanity check
----> 2 pil_image_load(os.path.basename(train_images_path[10]))
      3 

IndexError: list index out of range

## === cell 7
train_df.head()




## === cell 8
def read_image(img_path, resized_shape=None):
    img_path = expand_path(img_path)
    image = imread(img_path)  # RGB
    gray_image = color.rgb2gray(image)
    rgb_image = color.gray2rgb(gray_image)

    target = resized_shape if resized_shape is not None else IMG_SIZE
    image_resized = resize(rgb_image, (target, target, 3), anti_aliasing=True)

    return image_resized.astype(np.float32)  # keep [0,1] as resize returns floats




## === cell 9
train_df.head()



## === cell 10
train_df["image"] = train_df["id"].apply(lambda path: read_image(path))
train_df.head()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4046890.py in <cell line: 0>()
      1 # Load training images into dataframe
----> 2 train_df["image"] = train_df["id"].apply(lambda path: read_image(path))
      3 train_df.head()
      4 

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in apply(self, func, convert_dtype, args, by_row, **kwargs)
   4922             args=args,
   4923             kwargs=kwargs,
-> 4924         ).apply()
   4925 
   4926     def _reindex_indexer(

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply(self)
   1425 
   1426         # self.func is Callable
-> 1427         return self.apply_standard()
   1428 
   1429     def agg(self):

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_standard(self)
   1505         #  Categorical (GH51645).
   1506         action = "ignore" if isinstance(obj.dtype, CategoricalDtype) else None
-> 1507         mapped = obj._map_values(
   1508             mapper=curried, na_action=action, convert=self.convert_dtype
   1509         )

/usr/local/lib/python3.11/dist-packages/pandas/core/base.py in _map_values(self, mapper, na_action, convert)
    919             return arr.map(mapper, na_action=na_action)
    920 
--> 921         return algorithms.map_array(arr, mapper, na_action=na_action, convert=convert)
    922 
    923     @final

/usr/local/lib/python3.11/dist-packages/pandas/core/algorithms.py in map_array(arr, mapper, na_action, convert)
   1741     values = arr.astype(object, copy=False)
   1742     if na_action is None:
-> 1743         return lib.map_infer(values, mapper, convert=convert)
   1744     else:
   1745         return lib.map_infer_mask(

lib.pyx in pandas._libs.lib.map_infer()

/tmp/ipykernel_11/4046890.py in <lambda>(path)
      1 # Load training images into dataframe
----> 2 train_df["image"] = train_df["id"].apply(lambda path: read_image(path))
      3 train_df.head()
      4 

/tmp/ipykernel_11/1906141039.py in read_image(img_path, resized_shape)
      1 def read_image(img_path, resized_shape=None):
      2     img_path = expand_path(img_path)
----> 3     image = imread(img_path)  # RGB
      4     # keep original preprocessing logic: rgb -> gray -> rgb (3ch)
      5     gray_image = color.rgb2gray(image)

/usr/local/lib/python3.11/dist-packages/skimage/_shared/utils.py in fixed_func(*args, **kwargs)
    326                     kwargs[self.new_name] = deprecated_value
    327 
--> 328             return func(*args, **kwargs)
    329 
    330         if self.modify_docstring and func.__doc__ is not None:

/usr/local/lib/python3.11/dist-packages/skimage/io/_io.py in imread(fname, as_gray, plugin, **plugin_args)
     80 
     81     with file_or_url_context(fname) as fname, _hide_plugin_deprecation_warnings():
---> 82         img = call_plugin('imread', fname, plugin=plugin, **plugin_args)
     83 
     84     if not hasattr(img, 'ndim'):

/usr/local/lib/python3.11/dist-packages/skimage/_shared/utils.py in wrapped(*args, **kwargs)
    536             stacklevel = 1 + self.get_stack_length(func) - stack_rank
    537             warnings.warn(message, category=FutureWarning, stacklevel=stacklevel)
--> 538             return func(*args, **kwargs)
    539 
    540         # modify docstring to display deprecation warning

/usr/local/lib/python3.11/dist-packages/skimage/io/manage_plugins.py in call_plugin(kind, *args, **kwargs)
    252             raise RuntimeError(f'Could not find the plugin "{plugin}" for {kind}.')
    253 
--> 254     return func(*args, **kwargs)
    255 
    256 

/usr/local/lib/python3.11/dist-packages/skimage/io/_plugins/imageio_plugin.py in imread(*args, **kwargs)
      9 @wraps(imageio_imread)
     10 def imread(*args, **kwargs):
---> 11     out = np.asarray(imageio_imread(*args, **kwargs))
     12     if not out.flags['WRITEABLE']:
     13         out = out.copy()

/usr/local/lib/python3.11/dist-packages/imageio/v3.py in imread(uri, index, plugin, extension, format_hint, **kwargs)
     51         call_kwargs["index"] = index
     52 
---> 53     with imopen(uri, "r", **plugin_kwargs) as img_file:
     54         return np.asarray(img_file.read(**call_kwargs))
     55 

/usr/local/lib/python3.11/dist-packages/imageio/core/imopen.py in imopen(uri, io_mode, plugin, extension, format_hint, legacy_mode, **kwargs)
    111         request.format_hint = format_hint
    112     else:
--> 113         request = Request(uri, io_mode, format_hint=format_hint, extension=extension)
    114 
    115     source = "<bytes>" if isinstance(uri, bytes) else uri

/usr/local/lib/python3.11/dist-packages/imageio/core/request.py in __init__(self, uri, mode, extension, format_hint, **kwargs)
    247 
    248         # Parse what was given
--> 249         self._parse_uri(uri)
    250 
    251         # Set extension

/usr/local/lib/python3.11/dist-packages/imageio/core/request.py in _parse_uri(self, uri)
    407                 # Reading: check that the file exists (but is allowed a dir)
    408                 if not os.path.exists(fn):
--> 409                     raise FileNotFoundError("No such file: '%s'" % fn)
    410             else:
    411                 # Writing: check that the directory to write to does exist

FileNotFoundError: No such file: '/kaggle/working/2de8f189f1dce439766637e75df0ee27.jpg'

## === cell 11
test_df = pd.DataFrame({"id": sorted(os.listdir(test_folder))})
test_df["image"] = test_df["id"].apply(lambda path: read_image(path))
test_df.head()



## === cell 12
test_df.head()



## === cell 13
random.shuffle(train_images_path)
fig, ax = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle("Some aerial images", fontsize=16)

df_vis = shuffle(train_df, random_state=SEED)
for i, item in enumerate(df_vis[["id", "has_cactus"]].values[15:20]):
    image = pil_image.open(expand_path(item[0]))
    ax[0, i].imshow(image)
    ax[0, i].set_title("Has Cactus = %d" % (item[1]))
ax[0, 0].set_ylabel("train images", size="large")

for i, path in enumerate(test_images_path[:5]):
    image = pil_image.open(path)
    ax[1, i].imshow(image)
ax[1, 0].set_ylabel("test images", size="large")
plt.tight_layout()
plt.show()




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1339970655.py in <cell line: 0>()
      6 df_vis = shuffle(train_df, random_state=SEED)
      7 for i, item in enumerate(df_vis[["id", "has_cactus"]].values[15:20]):
----> 8     image = pil_image.open(expand_path(item[0]))
      9     ax[0, i].imshow(image)
     10     ax[0, i].set_title("Has Cactus = %d" % (item[1]))

/usr/local/lib/python3.11/dist-packages/PIL/Image.py in open(fp, mode, formats)
   3511     if is_path(fp):
   3512         filename = os.fspath(fp)
-> 3513         fp = builtins.open(filename, "rb")
   3514         exclusive_fp = True
   3515     else:

FileNotFoundError: [Errno 2] No such file or directory: '27cfa7c7e7169a5e736734f2c45e79a5.jpg'

## === cell 14
def CNN():
    model = Sequential()
    model.add(Conv2D(256, (3, 3), strides=(1, 1), input_shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), strides=(1, 1)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(256, (3, 3), strides=(1, 1)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy", optimizer=optimizers.RMSprop(), metrics=["accuracy"]
    )
    return model




## === cell 15
def train_batch(train_df):
    images = train_df.image.values
    x_train = np.array([img for img in images], dtype=np.float32)
    y_train = train_df.has_cactus.values.astype(np.float32).reshape(-1, 1)
    return x_train, y_train




## === cell 16
model = CNN()
model.summary()



## === cell 17
X_train, y_train = train_batch(train_df)
X_train.shape, y_train.shape




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2690648954.py in <cell line: 0>()
----> 1 X_train, y_train = train_batch(train_df)
      2 X_train.shape, y_train.shape
      3 
      4 

/tmp/ipykernel_11/3564993383.py in train_batch(train_df)
      1 def train_batch(train_df):
----> 2     images = train_df.image.values
      3     x_train = np.array([img for img in images], dtype=np.float32)
      4     y_train = train_df.has_cactus.values.astype(np.float32).reshape(-1, 1)
      5     return x_train, y_train

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'image'

## === cell 18
def train_model(model, X_train, y_train, epochs=5, verbose=1):
    begin = time.time()
    checkpointer = ModelCheckpoint(
        filepath="weights.hdf5",
        monitor="val_accuracy",
        verbose=0,
        save_best_only=True,
        mode="max",
    )
    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        verbose=1,
        patience=5,
        mode="max",
        restore_best_weights=True,
    )

    for i in range(1, epochs + 1):
        print("************************************")
        print("Epoch: ", i, "/", epochs)
        print("************************************")
        model.fit(
            X_train,
            y_train,
            verbose=verbose,
            callbacks=[checkpointer, early_stopping],
            validation_split=0.05,
            shuffle=True,
        )
    elapsed = time.time() - begin
    print("total training time: ", elapsed)
    return model




## === cell 19
train_model(model, X_train, y_train, epochs=20, verbose=1)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2771772222.py in <cell line: 0>()
----> 1 train_model(model, X_train, y_train, epochs=20, verbose=1)
      2 

NameError: name 'X_train' is not defined

## === cell 20
X_test = np.array([img for img in test_df.image.values], dtype=np.float32)
X_test.shape



## === cell 21
y_pred = model.predict(X_test, batch_size=256, verbose=1).reshape(-1)
y_pred[:5], y_pred.min(), y_pred.max()



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3697855829.py in <cell line: 0>()
      1 # Prediction on test data: keep probabilities for ROC-AUC metric (submission expects probabilities)
----> 2 y_pred = model.predict(X_test, batch_size=256, verbose=1).reshape(-1)
      3 y_pred[:5], y_pred.min(), y_pred.max()
      4 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/progbar.py in update(self, current, values, finalize)
    117 
    118             if self.target is not None:
--> 119                 numdigits = int(math.log10(self.target)) + 1
    120                 bar = ("%" + str(numdigits) + "d/%d") % (current, self.target)
    121                 bar = f"\x1b[1m{bar}\x1b[0m "

ValueError: math domain error

## === cell 22
submission = pd.DataFrame(
    {"id": test_df["id"].values, "has_cactus": y_pred.astype(np.float32)}
)
submission.head()



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2681519382.py in <cell line: 0>()
      1 # Create submission with required columns and probabilities
      2 submission = pd.DataFrame(
----> 3     {"id": test_df["id"].values, "has_cactus": y_pred.astype(np.float32)}
      4 )
      5 submission.head()

NameError: name 'y_pred' is not defined

## === cell 23
assert list(submission.columns) == ["id", "has_cactus"]
assert submission.shape[0] == test_df.shape[0]
assert submission["id"].dtype == object
assert np.isfinite(submission["has_cactus"]).all()
submission.describe()



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3624307140.py in <cell line: 0>()
      1 # Basic format checks
----> 2 assert list(submission.columns) == ["id", "has_cactus"]
      3 assert submission.shape[0] == test_df.shape[0]
      4 assert submission["id"].dtype == object
      5 assert np.isfinite(submission["has_cactus"]).all()

NameError: name 'submission' is not defined

## === cell 24
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/433951654.py in <cell line: 0>()
      1 # Write valid Kaggle submission
----> 2 submission.to_csv("submission.csv", index=False)
      3 print("Wrote submission.csv with shape:", submission.shape)
      4 print(submission.head())

NameError: name 'submission' is not defined
