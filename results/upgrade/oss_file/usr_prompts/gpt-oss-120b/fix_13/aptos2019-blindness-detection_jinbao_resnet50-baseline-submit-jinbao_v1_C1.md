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
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.8723692104941825

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.63043) has done: 'The update keeps the same preprocessing, dataset handling, and model architecture, but speeds up execution by (1) freezing the pretrained ResNet‑50 backbone so only the new classifier layers are trained, dramatically reducing compute per epoch, (2) limiting TensorFlow’s intra‑ and inter‑op thread counts to a reasonable pool to avoid oversubscription, and (3) minor cleanup such as removing unused early‑stopping patience that isn’t triggered. These changes are functionally equivalent for inference and preserve the original logic while fitting comfortably inside the 600 s limit.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import tensorflow as tf

tf.config.optimizer.set_jit(True)

policy = tf.keras.mixed_precision.Policy("mixed_float16")
tf.keras.mixed_precision.set_global_policy(policy)

try:
    num_threads = min(8, os.cpu_count() or 1)
    tf.config.threading.set_intra_op_parallelism_threads(num_threads)
    tf.config.threading.set_inter_op_parallelism_threads(num_threads)
except Exception as e:
    print("TensorFlow threading config skipped:", e)

print("TensorFlow version:", tf.__version__)
print("Input directory listing:", os.listdir("../input"))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:  # image too dark
            return img
        img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
        img = np.stack([img1, img2, img3], axis=-1)
        return img
    return img




## === cell 2
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 8




## === cell 3
def preprocess_image(img_path):
    image = cv2.imread(img_path)
    if image is None:
        image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = crop_image_from_gray(image)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 30), -4, 128)
    return image




## === cell 4
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, PReLU
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50


def getResNet50(input_shape=(IMG_SIZE, IMG_SIZE, 3), classes=5, weights="imagenet"):
    input_layer = Input(shape=input_shape)
    base_model = ResNet50(include_top=False, weights=weights, input_tensor=input_layer)
    base_model.trainable = True
    for layer in base_model.layers[:-10]:
        layer.trainable = False
    x = GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    x = Dense(1024, name="fc1")(x)
    x = PReLU()(x)
    x = Dropout(0.5)(x)
    output = Dense(classes, activation="softmax", name="output")(x)
    model = Model(inputs=input_layer, outputs=output)
    return model




## === cell 5
import pandas as pd, numpy as np, cv2, concurrent.futures, os
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

cv2.setNumThreads(0)

train_df = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
train_ids = train_df["id_code"].values
train_labels = train_df["diagnosis"].values

train_ids_split, val_ids_split, train_lbls_split, val_lbls_split = train_test_split(
    train_ids, train_labels, test_size=0.1, random_state=42, stratify=train_labels
)


def tf_preprocess(path):
    img_path = path.numpy().decode()
    img = preprocess_image(img_path)
    img = img.astype(np.float32) / 255.0
    return img


def load_and_label(id_bytes, label):
    img_path = tf.strings.join(
        [
            tf.constant("../input/aptos2019-blindness-detection/train_images/"),
            tf.strings.regex_replace(tf.strings.as_string(id_bytes), "\s+", ""),
            tf.constant(".png"),
        ]
    )
    img = tf.numpy_function(tf_preprocess, [img_path], tf.float32)
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    label_onehot = tf.one_hot(label, depth=5)
    return img, label_onehot


train_ds = tf.data.Dataset.from_tensor_slices((train_ids_split, train_lbls_split))
train_ds = train_ds.map(load_and_label, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = (
    train_ds.shuffle(buffer_size=len(train_ids_split))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = tf.data.Dataset.from_tensor_slices((val_ids_split, val_lbls_split))
val_ds = val_ds.map(load_and_label, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)

model = getResNet50(weights="imagenet")
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

early_stop = tf.keras.callbacks.EarlyStopping(
    patience=4, restore_best_weights=True, monitor="val_loss"
)
lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(
    patience=2, factor=0.5, monitor="val_loss"
)

print("Starting training...")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop, lr_reduce],
    verbose=2,
)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
UnknownError                              Traceback (most recent call last)
/tmp/ipykernel_11/1901166223.py in <cell line: 0>()
     70 
     71 print("Starting training...")
---> 72 model.fit(
     73     train_ds,
     74     validation_data=val_ds,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     57       e.message += " name: " + name
     58     raise core._status_to_exception(e) from None
---> 59   except TypeError as e:
     60     keras_symbolic_tensors = [x for x in inputs if _is_keras_symbolic_tensor(x)]
     61     if keras_symbolic_tensors:

UnknownError: Graph execution error:

Detected at node PyFunc defined at (most recent call last):
<stack traces unavailable>
Error in user-defined function passed to ParallelMapDatasetV2:1 transformation with iterator: Iterator::Root::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::ParallelMapV2: AttributeError: 'bytes' object has no attribute 'numpy'
Traceback (most recent call last):

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 269, in __call__
    ret = func(*args)
          ^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py", line 643, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/1901166223.py", line 20, in tf_preprocess
    img_path = path.numpy().decode()
               ^^^^^^^^^^

AttributeError: 'bytes' object has no attribute 'numpy'


	 [[{{node PyFunc}}]]
	 [[IteratorGetNext]] [Op:__inference_multi_step_on_iterator_15317]

## === cell 6
submit = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")
test_ids = submit["id_code"].values


def tf_preprocess_test(path):
    img_path = path.numpy().decode()
    img = preprocess_image(img_path)
    img = img.astype(np.float32) / 255.0
    return img


def load_test_image(id_bytes):
    img_path = tf.strings.join(
        [
            tf.constant("../input/aptos2019-blindness-detection/test_images/"),
            tf.strings.regex_replace(tf.strings.as_string(id_bytes), "\s+", ""),
            tf.constant(".png"),
        ]
    )
    img = tf.numpy_function(tf_preprocess_test, [img_path], tf.float32)
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return img


test_ds = tf.data.Dataset.from_tensor_slices(test_ids)
test_ds = test_ds.map(load_test_image, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print("Running batch inference on test set...")
preds = model.predict(test_ds, verbose=0)
ans = [int(np.argmax(p)) for p in preds]

submit["diagnosis"] = ans
submit.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
UnknownError                              Traceback (most recent call last)
/tmp/ipykernel_11/4044328673.py in <cell line: 0>()
     28 
     29 print("Running batch inference on test set...")
---> 30 preds = model.predict(test_ds, verbose=0)
     31 ans = [int(np.argmax(p)) for p in preds]
     32 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

UnknownError: {{function_node __wrapped__IteratorGetNext_output_types_1_device_/job:localhost/replica:0/task:0/device:CPU:0}} AttributeError: 'bytes' object has no attribute 'numpy'
Traceback (most recent call last):

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 269, in __call__
    ret = func(*args)
          ^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py", line 643, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/4044328673.py", line 6, in tf_preprocess_test
    img_path = path.numpy().decode()
               ^^^^^^^^^^

AttributeError: 'bytes' object has no attribute 'numpy'


	 [[{{node PyFunc}}]] [Op:IteratorGetNext] name:
