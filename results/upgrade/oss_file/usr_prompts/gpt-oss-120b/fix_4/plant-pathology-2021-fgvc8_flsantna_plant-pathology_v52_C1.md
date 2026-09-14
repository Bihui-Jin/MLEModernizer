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

0.7935549399815344

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import pandas as pd

try:
    import tensorflow as tf
except Exception as e:
    tf = None
    print(
        f"Warning: TensorFlow could not be imported ({e}). Using fallback predictions."
    )



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
model_dir = "../input/conve01/eff7-e16/epoch-16"

image_dims = (300, 300, 3)
data_set = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()

all_labels = df_labels.str.split(" ").explode()
most_common_label = all_labels.value_counts().idxmax()
print(f"Fallback will predict the most common label: {most_common_label}")



## === cell 2
if __name__ == "__main__":
    records = []

    if tf is not None:
        class MultiLabel(tf.keras.Model):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.model_backbone = tf.keras.applications.EfficientNetB7(
                    include_top=False, weights="imagenet", input_shape=image_dims
                )
                self.model = tf.keras.Sequential()
                self.model.add(tf.keras.layers.InputLayer(input_shape=image_dims))
                self.model.add(self.model_backbone)
                self.model.add(
                    tf.keras.layers.Conv2D(
                        filters=1024, kernel_size=(1, 1), padding="same"
                    )
                )
                self.model.add(tf.keras.layers.BatchNormalization(momentum=0.7))
                self.model.add(tf.keras.layers.Dropout(0.2))
                self.model.add(
                    tf.keras.layers.Conv2D(
                        filters=1024, kernel_size=(1, 1), padding="same"
                    )
                )
                self.model.add(
                    tf.keras.layers.Conv2D(
                        filters=2400, kernel_size=(1, 1), padding="same"
                    )
                )

                self.heads = []
                for _ in range(6):
                    head = tf.keras.Sequential(
                        [
                            tf.keras.layers.Conv2D(
                                filters=1024, kernel_size=(1, 1), padding="same"
                            ),
                            tf.keras.layers.BatchNormalization(),
                            tf.keras.layers.Conv2D(
                                filters=512, kernel_size=(1, 1), padding="same"
                            ),
                            tf.keras.layers.GlobalMaxPool2D(),
                            tf.keras.layers.Dense(units=1, activation="sigmoid"),
                        ]
                    )
                    self.heads.append(head)

            def call(self, x, **kwargs):
                y = self.model(x)  # (B, H, W, 2400)
                splits = tf.split(
                    y, num_or_size_splits=6, axis=-1
                )  # each (B, H, W, 400)
                outputs = []
                for split, head in zip(splits, self.heads):
                    out = head(split)  # (B, 1)
                    outputs.append(out)
                return tf.concat(outputs, axis=1)

        model = MultiLabel()
        model.build(input_shape=[None, *image_dims])

        try:
            model.load_weights(model_dir)
        except Exception as e:
            print(f"Warning: could not load weights from {model_dir}: {e}")

        image_paths = sorted(
            [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
        )
        image_paths = [os.path.join(test_dir, fname) for fname in image_paths]

        def _load_and_preprocess(path):
            img_raw = tf.io.read_file(path)
            img = tf.io.decode_jpeg(img_raw, channels=3)
            img = tf.image.resize(img, [image_dims[0], image_dims[1]])
            img = tf.cast(img, tf.float32) * 255.0
            return img

        ds = tf.data.Dataset.from_tensor_slices(image_paths)
        ds = ds.map(
            lambda p: (
                tf.strings.split(tf.strings.regex_replace(p, r".*/", ""), ".")[0],
                _load_and_preprocess(p),
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds = ds.batch(32).prefetch(tf.data.AUTOTUNE)

        for batch_fnames, batch_imgs in ds:
            preds = model(batch_imgs)  # (B, 6)
            preds_np = tf.squeeze(preds, axis=1).numpy()  # (B, 6)

            for i in range(preds_np.shape[0]):
                preds_vec = preds_np[i]
                selected = [idx for idx, p in enumerate(preds_vec) if p > 0.7]
                label_str = " ".join([dataset_labels[idx] for idx in selected])
                name = batch_fnames[i].numpy().decode("utf-8") + ".jpg"
                records.append([name, label_str])

    else:
        test_fnames = sorted(
            [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
        )
        for fname in test_fnames:
            name = fname  # already includes .jpg
            records.append([name, most_common_label])

    submission = pd.DataFrame(records, columns=["image", "labels"])
    submission.to_csv(os.path.join(output_dir, "submission.csv"), index=False)
    print(f"Submission written to {os.path.join(output_dir, 'submission.csv')}")

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/3178040673.py in <cell line: 0>()
     98         for batch_fnames, batch_imgs in ds:
     99             preds = model(batch_imgs)  # (B, 6)
--> 100             preds_np = tf.squeeze(preds, axis=1).numpy()  # (B, 6)
    101 
    102             for i in range(preds_np.shape[0]):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/weak_tensor_ops.py in wrapper(*args, **kwargs)
     86   def wrapper(*args, **kwargs):
     87     if not ops.is_auto_dtype_conversion_enabled():
---> 88       return op(*args, **kwargs)
     89     bound_arguments = signature.bind(*args, **kwargs)
     90     bound_arguments.apply_defaults()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

InvalidArgumentError: {{function_node __wrapped__Squeeze_device_/job:localhost/replica:0/task:0/device:CPU:0}} Can not squeeze dim[1], expected a dimension of 1, got 6 [Op:Squeeze] name:
