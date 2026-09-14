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

0.7607940904893831

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

try:
    import tensorflow as tf

    _TF_AVAILABLE = True
except Exception:
    tf = None
    _TF_AVAILABLE = False

if _TF_AVAILABLE:
    try:
        _ = tf.constant(0)
    except Exception:
        tf = None
        _TF_AVAILABLE = False




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
output_dir = "./"
test_dir = "input/plant-pathology-2021-fgvc8/test_images/"
model_dir = (
    "input/plant-pathology-2021-fgvc8/effb7-e8/epoch-8"  # optional checkpoint directory
)
image_dims = (300, 300, 3)

data_set = pd.read_csv("input/plant-pathology-2021-fgvc8/train.csv")
df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.tolist()
num_classes = len(dataset_labels)

class_counts = one_hot.sum(axis=0).values
class_probs = class_counts / len(df_labels)  # probability for each class




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4245111800.py in <cell line: 0>()
      8 
      9 # Load training metadata to build label vocab and class probabilities.
---> 10 data_set = pd.read_csv("input/plant-pathology-2021-fgvc8/train.csv")
     11 df_labels = data_set["labels"]
     12 one_hot = df_labels.str.get_dummies(sep=" ")

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: 'input/plant-pathology-2021-fgvc8/train.csv'

## === cell 2
if _TF_AVAILABLE:
    from tensorflow.keras.applications import EfficientNetB7
    from tensorflow.keras import Model
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

    class MultiLabel(Model):
        def __init__(self, num_classes, **kwargs):
            super().__init__(**kwargs)
            self.backbone = EfficientNetB7(
                include_top=False,
                weights="imagenet",
                input_shape=image_dims,
            )
            self.pool = GlobalAveragePooling2D()
            self.dropout = Dropout(0.2)
            self.classifier = Dense(num_classes, activation="sigmoid")

        def call(self, inputs, training=False):
            x = self.backbone(inputs, training=training)
            x = self.pool(x)
            x = self.dropout(x, training=training)
            return self.classifier(x)

else:
    class MultiLabel:
        def __init__(self, num_classes, **kwargs):
            self.num_classes = num_classes

        def build(self, input_shape):
            pass

        def __call__(self, inputs, training=False):
            return np.zeros((inputs.shape[0], self.num_classes), dtype=np.float32)




## === cell 3
if __name__ == "__main__":
    model = MultiLabel(num_classes=num_classes)
    model.build(input_shape=[None, *image_dims])

    if os.path.isdir(model_dir) or os.path.isfile(model_dir):
        if _TF_AVAILABLE:
            try:
                model.load_weights(model_dir)
            except Exception as e:
                print(f"Warning: could not load weights from {model_dir}: {e}")

    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    images_path_list = sorted(
        [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
    )

    results = []

    if _TF_AVAILABLE:
        batch_size = 32
        test_paths = [os.path.join(test_dir, fname) for fname in images_path_list]

        def _load_path(path):
            """Load, decode, resize, and scale a single image; also return its filename."""
            name = tf.strings.split(path, os.sep)[-1]
            img = tf.io.read_file(path)
            img = tf.io.decode_image(img, channels=3, dtype=tf.dtypes.float32)
            img = tf.image.resize(img, [image_dims[0], image_dims[1]])
            img = img * 255.0
            return name, img

        ds = tf.data.Dataset.from_tensor_slices(test_paths)
        ds = ds.map(_load_path, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size)

        for batch_names, batch_imgs in ds:
            preds_raw = model(batch_imgs, training=False)
            preds = preds_raw.numpy()

            for name_tensor, pred_vec in zip(batch_names, preds):
                name = name_tensor.numpy().decode()
                selected = [
                    dataset_labels[idx] for idx, p in enumerate(pred_vec) if p > 0.5
                ]
                label_str = " ".join(selected)
                results.append([name, label_str])
    else:
        common_class_indices = [i for i, prob in enumerate(class_probs) if prob > 0.5]
        common_labels = [dataset_labels[i] for i in common_class_indices]

        for name in images_path_list:
            if not common_labels:
                most_common_idx = int(np.argmax(class_probs))
                predicted = [dataset_labels[most_common_idx]]
            else:
                predicted = common_labels
            label_str = " ".join(predicted)
            results.append([name, label_str])

    submission_df = pd.DataFrame(results, columns=["image", "labels"])
    submission_path = os.path.join(output_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1920024100.py in <cell line: 0>()
      1 if __name__ == "__main__":
----> 2     model = MultiLabel(num_classes=num_classes)
      3     model.build(input_shape=[None, *image_dims])
      4 
      5     # Attempt to load pretrained weights only when TF is available and the path exists.

NameError: name 'num_classes' is not defined
