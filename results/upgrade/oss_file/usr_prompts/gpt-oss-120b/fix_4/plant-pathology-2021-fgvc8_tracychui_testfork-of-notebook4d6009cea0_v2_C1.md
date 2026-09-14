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
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.3428175702413932

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import gc

print("TensorFlow version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_INPUT = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(BASE_INPUT, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE_INPUT, "train_images")
TEST_IMG_DIR = os.path.join(BASE_INPUT, "test_images")
SUBMISSION_PATH = "./submission.csv"

IMG_SIZE = (224, 224)  # ResNet‑50 default
BATCH_SIZE = 256



## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
train_images = train_df["image"].values
train_labels_raw = train_df["labels"].fillna("").values

train_labels = [lbl.split() for lbl in train_labels_raw]



## === cell 3
_resnet_model = ResNet50(
    weights="imagenet",
    include_top=False,
    pooling="avg",
    input_shape=IMG_SIZE + (3,),
)


def _load_and_preprocess(batch_paths):
    """
    Load JPEG files with tf.io, decode, resize, and apply ResNet‑50 preprocessing.
    This replaces the slower Pillow‐based loading while producing identical
    numerical inputs (differences are negligible floating‑point variations).
    """
    img_bytes = tf.map_fn(tf.io.read_file, batch_paths, fn_output_signature=tf.string)
    imgs = tf.map_fn(
        lambda x: tf.image.decode_jpeg(x, channels=3),
        img_bytes,
        fn_output_signature=tf.uint8,
    )
    imgs = tf.image.resize(imgs, IMG_SIZE)
    imgs = preprocess_input(tf.cast(imgs, tf.float32))
    return imgs


def extract_features(img_paths, img_dir):
    """Extract ResNet‑50 avg‑pooled features for a list of image filenames."""
    n_samples = len(img_paths)
    features = np.empty((n_samples, 2048), dtype=np.float32)

    for start_idx in range(0, n_samples, BATCH_SIZE):
        batch_fnames = img_paths[start_idx : start_idx + BATCH_SIZE]
        batch_paths = [os.path.join(img_dir, fname) for fname in batch_fnames]
        batch_tensor = tf.constant(batch_paths)
        batch_images = _load_and_preprocess(batch_tensor)
        batch_feat = _resnet_model.predict(batch_images, verbose=0)
        batch_size_actual = batch_feat.shape[0]
        features[start_idx : start_idx + batch_size_actual] = batch_feat

    return features




## === cell 4
print("Extracting training features …")
X_train = extract_features(train_images, TRAIN_IMG_DIR)
print("Training features shape:", X_train.shape)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1872115145.py in <cell line: 0>()
      1 print("Extracting training features …")
----> 2 X_train = extract_features(train_images, TRAIN_IMG_DIR)
      3 print("Training features shape:", X_train.shape)
      4 

/tmp/ipykernel_11/1692361171.py in extract_features(img_paths, img_dir)
     38         # TensorFlow pipeline for fast batch loading
     39         batch_tensor = tf.constant(batch_paths)
---> 40         batch_images = _load_and_preprocess(batch_tensor)
     41         batch_feat = _resnet_model.predict(batch_images, verbose=0)
     42         batch_size_actual = batch_feat.shape[0]

/tmp/ipykernel_11/1692361171.py in _load_and_preprocess(batch_paths)
     16     img_bytes = tf.map_fn(tf.io.read_file, batch_paths, fn_output_signature=tf.string)
     17     # decode JPEG to uint8 tensor
---> 18     imgs = tf.map_fn(
     19         lambda x: tf.image.decode_jpeg(x, channels=3),
     20         img_bytes,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/deprecation.py in new_func(*args, **kwargs)
    658                   'in a future version' if date is None else
    659                   ('after %s' % date), instructions)
--> 660       return func(*args, **kwargs)
    661 
    662     doc = _add_deprecated_arg_value_notice_to_docstring(func.__doc__, date,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/deprecation.py in new_func(*args, **kwargs)
    586                 'in a future version' if date is None else ('after %s' % date),
    587                 instructions)
--> 588       return func(*args, **kwargs)
    589 
    590     doc = _add_deprecated_arg_notice_to_docstring(

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/map_fn.py in map_fn_v2(fn, elems, dtype, parallel_iterations, back_prop, swap_memory, infer_shape, name, fn_output_signature)
    635   if fn_output_signature is None:
    636     fn_output_signature = dtype
--> 637   return map_fn(
    638       fn=fn,
    639       elems=elems,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/deprecation.py in new_func(*args, **kwargs)
    586                 'in a future version' if date is None else ('after %s' % date),
    587                 instructions)
--> 588       return func(*args, **kwargs)
    589 
    590     doc = _add_deprecated_arg_notice_to_docstring(

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/map_fn.py in map_fn(fn, elems, dtype, parallel_iterations, back_prop, swap_memory, infer_shape, name, fn_output_signature)
    495       return (i + 1, tas)
    496 
--> 497     _, r_a = while_loop.while_loop(
    498         lambda i, _: i < n,
    499         compute, (i, result_batchable_ta),

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/while_loop.py in while_loop(cond, body, loop_vars, shape_invariants, parallel_iterations, back_prop, swap_memory, name, maximum_iterations, return_same_structure)
    486                                               list(loop_vars))
    487       while cond(*loop_vars):
--> 488         loop_vars = body(*loop_vars)
    489         if try_to_pack and not isinstance(loop_vars, (list, tuple)):
    490           packed = True

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/while_loop.py in <lambda>(i, lv)
    477         cond = lambda i, lv: (  # pylint: disable=g-long-lambda
    478             math_ops.logical_and(i < maximum_iterations, orig_cond(*lv)))
--> 479         body = lambda i, lv: (i + 1, orig_body(*lv))
    480       try_to_pack = False
    481 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/map_fn.py in compute(i, tas)
    490       result_value_batchable = _result_value_flat_to_batchable(
    491           result_value_flat, result_flat_signature)
--> 492       tas = [
    493           ta.write(i, value) for (ta, value) in zip(tas, result_value_batchable)
    494       ]

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/map_fn.py in <listcomp>(.0)
    491           result_value_flat, result_flat_signature)
    492       tas = [
--> 493           ta.write(i, value) for (ta, value) in zip(tas, result_value_batchable)
    494       ]
    495       return (i + 1, tas)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/tf_should_use.py in wrapped(*args, **kwargs)
    286     """Decorates the input function."""
    287     def wrapped(*args, **kwargs):
--> 288       return _add_should_use_warning(fn(*args, **kwargs),
    289                                      warn_in_eager=warn_in_eager,
    290                                      error_in_function=error_in_function)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/tensor_array_ops.py in write(self, index, value, name)
   1186       ValueError: if there are more writers than specified.
   1187     """
-> 1188     return self._implementation.write(index, value, name=name)
   1189 
   1190   def stack(self, name=None):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/tensor_array_ops.py in write(***failed resolving arguments***)
    856     """See TensorArray."""
    857     del name  # not meaningful when executing eagerly.
--> 858     self._write(index, value)
    859     return self.parent()
    860 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/tensor_array_ops.py in _write(self, index, value)
    845 
    846     if not self._element_shape.is_compatible_with(value.shape):
--> 847       raise ValueError("Incompatible shape for value (%s), expected (%s)" %
    848                        (value.shape, self._element_shape))
    849 

ValueError: Incompatible shape for value ((3000, 4000, 3)), expected ((2672, 4000, 3))

## === cell 5
mlb = MultiLabelBinarizer()
Y_train = mlb.fit_transform(train_labels)

classifier = OneVsRestClassifier(LogisticRegression(max_iter=200, n_jobs=-1))
print("Training One‑Vs‑Rest classifier …")
classifier.fit(X_train, Y_train)
print("Training completed.")
del X_train
gc.collect()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1474755565.py in <cell line: 0>()
      4 classifier = OneVsRestClassifier(LogisticRegression(max_iter=200, n_jobs=-1))
      5 print("Training One‑Vs‑Rest classifier …")
----> 6 classifier.fit(X_train, Y_train)
      7 print("Training completed.")
      8 # free memory no longer needed

NameError: name 'X_train' is not defined

## === cell 6
test_images = sorted(
    [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(".jpg")]
)
print(f"Found {len(test_images)} test images.")

print("Extracting test features …")
X_test = extract_features(test_images, TEST_IMG_DIR)
print("Test features shape:", X_test.shape)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1108372785.py in <cell line: 0>()
      5 
      6 print("Extracting test features …")
----> 7 X_test = extract_features(test_images, TEST_IMG_DIR)
      8 print("Test features shape:", X_test.shape)
      9 

/tmp/ipykernel_11/1692361171.py in extract_features(img_paths, img_dir)
     38         # TensorFlow pipeline for fast batch loading
     39         batch_tensor = tf.constant(batch_paths)
---> 40         batch_images = _load_and_preprocess(batch_tensor)
     41         batch_feat = _resnet_model.predict(batch_images, verbose=0)
     42         batch_size_actual = batch_feat.shape[0]

/tmp/ipykernel_11/1692361171.py in _load_and_preprocess(batch_paths)
     16     img_bytes = tf.map_fn(tf.io.read_file, batch_paths, fn_output_signature=tf.string)
     17     # decode JPEG to uint8 tensor
---> 18     imgs = tf.map_fn(
     19         lambda x: tf.image.decode_jpeg(x, channels=3),
     20         img_bytes,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/deprecation.py in new_func(*args, **kwargs)
    658                   'in a future version' if date is None else
    659                   ('after %s' % date), instructions)
--> 660       return func(*args, **kwargs)
    661 
    662     doc = _add_deprecated_arg_value_notice_to_docstring(func.__doc__, date,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/deprecation.py in new_func(*args, **kwargs)
    586                 'in a future version' if date is None else ('after %s' % date),
    587                 instructions)
--> 588       return func(*args, **kwargs)
    589 
    590     doc = _add_deprecated_arg_notice_to_docstring(

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/map_fn.py in map_fn_v2(fn, elems, dtype, parallel_iterations, back_prop, swap_memory, infer_shape, name, fn_output_signature)
    635   if fn_output_signature is None:
    636     fn_output_signature = dtype
--> 637   return map_fn(
    638       fn=fn,
    639       elems=elems,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/deprecation.py in new_func(*args, **kwargs)
    586                 'in a future version' if date is None else ('after %s' % date),
    587                 instructions)
--> 588       return func(*args, **kwargs)
    589 
    590     doc = _add_deprecated_arg_notice_to_docstring(

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/map_fn.py in map_fn(fn, elems, dtype, parallel_iterations, back_prop, swap_memory, infer_shape, name, fn_output_signature)
    495       return (i + 1, tas)
    496 
--> 497     _, r_a = while_loop.while_loop(
    498         lambda i, _: i < n,
    499         compute, (i, result_batchable_ta),

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/while_loop.py in while_loop(cond, body, loop_vars, shape_invariants, parallel_iterations, back_prop, swap_memory, name, maximum_iterations, return_same_structure)
    486                                               list(loop_vars))
    487       while cond(*loop_vars):
--> 488         loop_vars = body(*loop_vars)
    489         if try_to_pack and not isinstance(loop_vars, (list, tuple)):
    490           packed = True

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/while_loop.py in <lambda>(i, lv)
    477         cond = lambda i, lv: (  # pylint: disable=g-long-lambda
    478             math_ops.logical_and(i < maximum_iterations, orig_cond(*lv)))
--> 479         body = lambda i, lv: (i + 1, orig_body(*lv))
    480       try_to_pack = False
    481 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/map_fn.py in compute(i, tas)
    490       result_value_batchable = _result_value_flat_to_batchable(
    491           result_value_flat, result_flat_signature)
--> 492       tas = [
    493           ta.write(i, value) for (ta, value) in zip(tas, result_value_batchable)
    494       ]

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/map_fn.py in <listcomp>(.0)
    491           result_value_flat, result_flat_signature)
    492       tas = [
--> 493           ta.write(i, value) for (ta, value) in zip(tas, result_value_batchable)
    494       ]
    495       return (i + 1, tas)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/tf_should_use.py in wrapped(*args, **kwargs)
    286     """Decorates the input function."""
    287     def wrapped(*args, **kwargs):
--> 288       return _add_should_use_warning(fn(*args, **kwargs),
    289                                      warn_in_eager=warn_in_eager,
    290                                      error_in_function=error_in_function)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/tensor_array_ops.py in write(self, index, value, name)
   1186       ValueError: if there are more writers than specified.
   1187     """
-> 1188     return self._implementation.write(index, value, name=name)
   1189 
   1190   def stack(self, name=None):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/tensor_array_ops.py in write(***failed resolving arguments***)
    856     """See TensorArray."""
    857     del name  # not meaningful when executing eagerly.
--> 858     self._write(index, value)
    859     return self.parent()
    860 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/tensor_array_ops.py in _write(self, index, value)
    845 
    846     if not self._element_shape.is_compatible_with(value.shape):
--> 847       raise ValueError("Incompatible shape for value (%s), expected (%s)" %
    848                        (value.shape, self._element_shape))
    849 

ValueError: Incompatible shape for value ((1728, 2592, 3)), expected ((2672, 4000, 3))

## === cell 7
test_proba = classifier.predict_proba(X_test)
test_pred_binary = (test_proba >= 0.5).astype(int)


def binary_to_tags(binary_array, tag_names):
    tags_list = []
    for row in binary_array:
        tags = [tag_names[i] for i, val in enumerate(row) if val == 1]
        tags_str = " ".join(tags) if tags else "healthy"  # fallback label
        tags_list.append(tags_str)
    return tags_list


test_pred_tags = binary_to_tags(test_pred_binary, mlb.classes_)
print("Sample predictions:", test_pred_tags[:5])



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/112302196.py in <cell line: 0>()
----> 1 test_proba = classifier.predict_proba(X_test)
      2 test_pred_binary = (test_proba >= 0.5).astype(int)
      3 
      4 
      5 def binary_to_tags(binary_array, tag_names):

NameError: name 'X_test' is not defined

## === cell 8
submission_df = pd.DataFrame({"image": test_images, "labels": test_pred_tags})
print("Submission preview:")
print(submission_df.head())



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1700382791.py in <cell line: 0>()
----> 1 submission_df = pd.DataFrame({"image": test_images, "labels": test_pred_tags})
      2 print("Submission preview:")
      3 print(submission_df.head())
      4 

NameError: name 'test_pred_tags' is not defined

## === cell 9
submission_df.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission file written to {SUBMISSION_PATH}")

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1138758640.py in <cell line: 0>()
----> 1 submission_df.to_csv(SUBMISSION_PATH, index=False)
      2 print(f"Submission file written to {SUBMISSION_PATH}")

NameError: name 'submission_df' is not defined
