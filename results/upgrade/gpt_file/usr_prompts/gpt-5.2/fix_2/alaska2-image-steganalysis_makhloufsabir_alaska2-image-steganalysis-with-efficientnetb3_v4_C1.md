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
Determine which of the images have hidden messages embedded using one of three steganography algorithms (JMiPOD, JUNIWARD, UERD).

## Metric
Weighted AUC. Each region of the ROC curve is weighted according to these chosen parameters:

```
tpr_thresholds = [0.0, 0.4, 1.0]
weights = [2, 1]
```

In other words, the area between the true positive rate of 0 and 0.4 is weighted 2X, the area between 0.4 and 1 is now weighed (1X). The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.

## Submission Format
For each `Id` (image) in the test set, you must provide a score that indicates how likely this image contains hidden data: the higher the score, the more it is assumed that image contains secret data. The file should contain a header and have the following format:

```
Id,Label
0001.jpg,0.1
0002.jpg,0.99
0003.jpg,1.2
0004.jpg,-2.2
etc.
```
## Dataset
The only available information on the test set is:

1. Each embedding algorithm is used with the same probability.
2. The payload (message length) is adjusted such that the "difficulty" is approximately the same regardless the content of the image. Images with smooth content are used to hide shorter messages while highly textured images will be used to hide more secret bits. The payload is adjusted in the same manner for testing and training sets.
3. The average message length is 0.4 bit per non-zero AC DCT coefficient.
4. The images are all compressed with one of the three following JPEG quality factors: 95, 90 or 75.

### Files
- `Cover/` contains 75k unaltered images meant for use in training.
- `JMiPOD/` contains 75k examples of the JMiPOD algorithm applied to the cover images.
- `JUNIWARD/`contains 75k examples of the JUNIWARD algorithm applied to the cover images.
- `UERD/` contains 75k examples of the UERD algorithm applied to the cover images.
- `Test/` contains 5k test set images. These are the images for which you are predicting.
- `sample_submission.csv` contains an example submission in the correct format.

# 2. Python version

3.8

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        input/
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        working/
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
```

-> data/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> data/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> working/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

# 5. Target score

0.7185074836498475

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, sys, subprocess, textwrap


def _pip_install(pkg):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
    except Exception as e:
        print(f"WARNING: pip install failed for {pkg}: {e}")


_pip_install("efficientnet")



## === cell 1
print("Skipping transformers installation (not used).")



## === cell 2
import numpy as np
import pandas as pd
import os

import tensorflow as tf
import tensorflow.keras.layers as l
from keras.optimizers import Adam
import efficientnet.tfkeras as efn
from sklearn.model_selection import train_test_split

SEED = 10
tf.random.set_seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
try:
    tpu = (
        tf.distribute.cluster_resolver.TPUClusterResolver()
    )  # will work only if TPU is enabled
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Using TPU:", tpu.master())
except Exception as e:
    strategy = tf.distribute.get_strategy()
    print("Using default strategy (GPU/CPU). TPU not available:", repr(e))

print("Replicas:", strategy.num_replicas_in_sync)



## === cell 4
AUTO = tf.data.AUTOTUNE

BASE_PATH = "/kaggle/input/alaska2-image-steganalysis"
assert os.path.exists(BASE_PATH), f"Missing expected dataset path: {BASE_PATH}"



## === cell 5
sample = pd.read_csv(f"{BASE_PATH}/sample_submission.csv")

BATCH_SIZE = 8 * strategy.num_replicas_in_sync
EPOCHS = 1

dir_name = ["Test", "JUNIWARD", "JMiPOD", "Cover", "UERD"]
lists = []
cate = []

for dir_ in dir_name:
    folder = os.path.join(BASE_PATH, dir_)
    files = os.listdir(folder)
    lists.extend(files)
    cate.extend([dir_] * len(files))

df = pd.DataFrame({"cate": cate, "name": lists})



## === cell 6
df["path"] = [os.path.join(BASE_PATH, c, n) for c, n in zip(df["cate"], df["name"])]


def cate_label(c):
    return 0 if c == "Cover" else 1


Test_df = df.query("cate=='Test'").copy()
Test_df = Test_df.sort_values(by="name").reset_index(drop=True)

Train_df = df.query("cate!='Test'").copy()
Train_df["labled"] = Train_df["cate"].map(cate_label).astype(np.int32)

print("Training set counts:\n", Train_df["cate"].value_counts())
print("\nTrain sample:\n", Train_df.head(2))
print("\nTest sample:\n", Test_df.head(2))



## === cell 7
X = Train_df["path"].values
y = Train_df["labled"].values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

X_test = Test_df["path"].values

print("Shapes:", X_train.shape, X_val.shape, X_test.shape)




## === cell 8
def decode_image(filename, label=None, image_size=(512, 512)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    if label is None:
        return image
    return image, tf.cast(label, tf.float32)




## === cell 9
train_dataset = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .map(decode_image, num_parallel_calls=AUTO)
    .repeat()
    .shuffle(1024, seed=SEED, reshuffle_each_iteration=True)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset.from_tensor_slices((X_val, y_val))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset.from_tensor_slices(X_test)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)



## === cell 10
with strategy.scope():
    model = tf.keras.Sequential(
        [
            efn.EfficientNetB3(
                input_shape=(512, 512, 3),
                weights="imagenet",
                include_top=False,
            ),
            l.GlobalAveragePooling2D(),
            l.Dropout(0.1),
            l.Dense(1, activation="sigmoid"),
        ]
    )
    opt = Adam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, decay=0.01, amsgrad=False)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

model.summary()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
HTTPError                                 Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/keras/src/utils/file_utils.py in get_file(fname, origin, untar, md5_hash, file_hash, cache_subdir, hash_algorithm, extract, archive_format, cache_dir, force_download)
    310             try:
--> 311                 urlretrieve(origin, download_target, DLProgbar())
    312             except urllib.error.HTTPError as e:

/usr/lib/python3.11/urllib/request.py in urlretrieve(url, filename, reporthook, data)
    240 
--> 241     with contextlib.closing(urlopen(url, data)) as fp:
    242         headers = fp.info()

/usr/lib/python3.11/urllib/request.py in urlopen(url, data, timeout, cafile, capath, cadefault, context)
    215         opener = _opener
--> 216     return opener.open(url, data, timeout)
    217 

/usr/lib/python3.11/urllib/request.py in open(self, fullurl, data, timeout)
    524             meth = getattr(processor, meth_name)
--> 525             response = meth(req, response)
    526 

/usr/lib/python3.11/urllib/request.py in http_response(self, request, response)
    633         if not (200 <= code < 300):
--> 634             response = self.parent.error(
    635                 'http', request, response, code, msg, hdrs)

/usr/lib/python3.11/urllib/request.py in error(self, proto, *args)
    562             args = (dict, 'default', 'http_error_default') + orig_args
--> 563             return self._call_chain(*args)
    564 

/usr/lib/python3.11/urllib/request.py in _call_chain(self, chain, kind, meth_name, *args)
    495             func = getattr(handler, meth_name)
--> 496             result = func(*args)
    497             if result is not None:

/usr/lib/python3.11/urllib/request.py in http_error_default(self, req, fp, code, msg, hdrs)
    642     def http_error_default(self, req, fp, code, msg, hdrs):
--> 643         raise HTTPError(req.full_url, code, msg, hdrs, fp)
    644 

HTTPError: HTTP Error 404: Not Found

During handling of the above exception, another exception occurred:

Exception                                 Traceback (most recent call last)
/tmp/ipykernel_11/2078107109.py in <cell line: 0>()
      2     model = tf.keras.Sequential(
      3         [
----> 4             efn.EfficientNetB3(
      5                 input_shape=(512, 512, 3),
      6                 weights="imagenet",

/usr/local/lib/python3.11/dist-packages/efficientnet/__init__.py in wrapper(*args, **kwargs)
     55         kwargs['models'] = tfkeras.models
     56         kwargs['utils'] = tfkeras.utils
---> 57         return func(*args, **kwargs)
     58 
     59     return wrapper

/usr/local/lib/python3.11/dist-packages/efficientnet/model.py in EfficientNetB3(include_top, weights, input_tensor, input_shape, pooling, classes, **kwargs)
    522                    classes=1000,
    523                    **kwargs):
--> 524     return EfficientNet(
    525         1.2, 1.4, 300, 0.3,
    526         model_name='efficientnet-b3',

/usr/local/lib/python3.11/dist-packages/efficientnet/model.py in EfficientNet(width_coefficient, depth_coefficient, default_resolution, dropout_rate, drop_connect_rate, depth_divisor, blocks_args, model_name, include_top, weights, input_tensor, input_shape, pooling, classes, **kwargs)
    430             file_name = model_name + '_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
    431             file_hash = IMAGENET_WEIGHTS_HASHES[model_name][1]
--> 432         weights_path = keras_utils.get_file(
    433             file_name,
    434             IMAGENET_WEIGHTS_PATH + file_name,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/file_utils.py in get_file(fname, origin, untar, md5_hash, file_hash, cache_subdir, hash_algorithm, extract, archive_format, cache_dir, force_download)
    311                 urlretrieve(origin, download_target, DLProgbar())
    312             except urllib.error.HTTPError as e:
--> 313                 raise Exception(error_msg.format(origin, e.code, e.msg))
    314             except urllib.error.URLError as e:
    315                 raise Exception(error_msg.format(origin, e.errno, e.reason))

Exception: URL fetch failure on https://github.com/Callidior/keras-applications/releases/download/efficientnet/efficientnet-b3_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5: 404 -- Not Found

## === cell 11
STEPS_PER_EPOCH = X_train.shape[0] // BATCH_SIZE
if STEPS_PER_EPOCH < 1:
    STEPS_PER_EPOCH = 1

history = model.fit(
    train_dataset,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=valid_dataset,
    verbose=1,
)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3436876416.py in <cell line: 0>()
      4     STEPS_PER_EPOCH = 1
      5 
----> 6 history = model.fit(
      7     train_dataset,
      8     steps_per_epoch=STEPS_PER_EPOCH,

NameError: name 'model' is not defined

## === cell 12
model.save("Mymodel.h5")



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2725523508.py in <cell line: 0>()
----> 1 model.save("Mymodel.h5")
      2 

NameError: name 'model' is not defined

## === cell 13
pred = model.predict(test_dataset, verbose=1)

pred = np.asarray(pred).reshape(-1)[: len(Test_df)]
print("Pred shape:", pred.shape, "min/max:", float(pred.min()), float(pred.max()))



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4144951022.py in <cell line: 0>()
----> 1 pred = model.predict(test_dataset, verbose=1)
      2 
      3 # Ensure predictions are 1D float array aligned to Test_df sorted by name
      4 pred = np.asarray(pred).reshape(-1)[: len(Test_df)]
      5 print("Pred shape:", pred.shape, "min/max:", float(pred.min()), float(pred.max()))

NameError: name 'model' is not defined

## === cell 14
submission = pd.DataFrame(
    {"Id": Test_df["name"].values, "Label": pred.astype(np.float32)}
)
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with rows:", len(submission))
assert (
    submission.shape[0] == sample.shape[0]
), "Submission row count must match sample_submission.csv"
assert list(submission.columns) == [
    "Id",
    "Label",
], "Submission columns must be Id,Label"
assert submission["Id"].is_unique, "Ids should be unique"
assert submission["Id"].iloc[0].endswith(".jpg"), "Id should look like a jpg filename"

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2993237732.py in <cell line: 0>()
      1 # Construct submission with Ids matching the sorted test filenames.
      2 submission = pd.DataFrame(
----> 3     {"Id": Test_df["name"].values, "Label": pred.astype(np.float32)}
      4 )
      5 submission.to_csv("submission.csv", index=False)

NameError: name 'pred' is not defined
