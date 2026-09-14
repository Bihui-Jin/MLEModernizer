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

0.8146999076638982

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import pandas as pd
import tensorflow as tf

output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
model_dir = "../input/conve01/effb7-e12/epoch-12"
efficientB7 = "../input/efficientb7/effb7"

image_dims = (300, 300, 3)

data_set = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()


def _find_savedmodel_dir(base_path: str) -> str:
    """Return a directory that contains saved_model.pb(.txt); try common Kaggle layouts."""

    def is_savedmodel_dir(p: str) -> bool:
        return (
            p
            and os.path.isdir(p)
            and (
                os.path.exists(os.path.join(p, "saved_model.pb"))
                or os.path.exists(os.path.join(p, "saved_model.pbtxt"))
            )
        )

    if is_savedmodel_dir(base_path):
        return base_path

    roots = []
    if base_path:
        roots.append(base_path)
        roots.append(os.path.dirname(base_path))
    roots.append("../input/efficientb7")

    seen = set()
    roots = [
        r for r in roots if r and not (r in seen or seen.add(r)) and os.path.exists(r)
    ]

    candidates = []
    for root in roots:
        if is_savedmodel_dir(root):
            candidates.append(root)
            continue
        if os.path.isdir(root):
            for cur, dirs, files in os.walk(root):
                if "saved_model.pb" in files or "saved_model.pbtxt" in files:
                    candidates.append(cur)

    if candidates:
        candidates = sorted(candidates, key=lambda p: (p.count(os.sep), len(p)))
        return candidates[0]

    return base_path


efficientB7 = _find_savedmodel_dir(efficientB7)
print("Resolved EfficientNetB7 SavedModel path candidate:", efficientB7)
has_savedmodel = os.path.exists(
    os.path.join(efficientB7, "saved_model.pb")
) or os.path.exists(os.path.join(efficientB7, "saved_model.pbtxt"))
print("EfficientNetB7 SavedModel exists:", has_savedmodel)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (
    Dense,
    BatchNormalization,
    Dropout,
    GlobalMaxPool2D,
    Conv2D,
    InputLayer,
)
from tensorflow import concat

try:
    from keras.layers import TFSMLayer  # Keras 3
except Exception:
    TFSMLayer = None


def _make_backbone_layer(backbone_path: str, image_dims_local):
    """
    Minimal, robust backbone loader:
    1) If a TensorFlow SavedModel is present, use Keras TFSMLayer (original intent).
    2) Else try TF Hub KerasLayer (common for EfficientNet assets).
    3) Else fall back to tf.keras.applications.EfficientNetB7(include_top=False).
    """
    if backbone_path and (
        os.path.exists(os.path.join(backbone_path, "saved_model.pb"))
        or os.path.exists(os.path.join(backbone_path, "saved_model.pbtxt"))
    ):
        if TFSMLayer is None:
            raise RuntimeError(
                "Found a SavedModel for the backbone, but keras.layers.TFSMLayer is unavailable."
            )
        call_endpoint = "serving_default"
        try:
            return TFSMLayer(backbone_path, call_endpoint=call_endpoint)
        except Exception:
            call_endpoint = "serve"
            return TFSMLayer(backbone_path, call_endpoint=call_endpoint)

    try:
        import tensorflow_hub as hub  # noqa: F401

        return hub.KerasLayer(backbone_path, trainable=False)
    except Exception:
        pass

    backbone = tf.keras.applications.EfficientNetB7(
        include_top=False,
        weights="imagenet",
        input_shape=image_dims_local,
    )
    return backbone


class MultiLabel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_backbone = _make_backbone_layer(efficientB7, image_dims)

        self.model = Sequential()
        self.model.add(InputLayer(input_shape=image_dims))
        self.model.add(self.model_backbone)
        self.model.add(Conv2D(filters=1024, kernel_size=(1, 1), padding="same"))
        self.model.add(BatchNormalization(momentum=0.7))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(filters=1024, kernel_size=(1, 1), padding="same"))
        self.model.add(Conv2D(filters=2400, kernel_size=(1, 1), padding="same"))

        self.model_pred1_2 = Conv2D(filters=1024, kernel_size=(1, 1), padding="same")
        self.model_pred1_3 = BatchNormalization()
        self.model_pred1_4 = Conv2D(filters=512, kernel_size=(1, 1), padding="same")
        self.model_pred1_5 = GlobalMaxPool2D()
        self.model_pred1_6 = Dense(units=1, activation="sigmoid")

        self.model_pred2_2 = Conv2D(filters=1024, kernel_size=(1, 1), padding="same")
        self.model_pred2_3 = BatchNormalization()
        self.model_pred2_4 = Conv2D(filters=512, kernel_size=(1, 1), padding="same")
        self.model_pred2_5 = GlobalMaxPool2D()
        self.model_pred2_6 = Dense(units=1, activation="sigmoid")

        self.model_pred3_2 = Conv2D(filters=1024, kernel_size=(1, 1), padding="same")
        self.model_pred3_3 = BatchNormalization()
        self.model_pred3_4 = Conv2D(filters=512, kernel_size=(1, 1), padding="same")
        self.model_pred3_5 = GlobalMaxPool2D()
        self.model_pred3_6 = Dense(units=1, activation="sigmoid")

        self.model_pred4_2 = Conv2D(filters=1024, kernel_size=(1, 1), padding="same")
        self.model_pred4_3 = BatchNormalization()
        self.model_pred4_4 = Conv2D(filters=512, kernel_size=(1, 1), padding="same")
        self.model_pred4_5 = GlobalMaxPool2D()
        self.model_pred4_6 = Dense(units=1, activation="sigmoid")

        self.model_pred5_2 = Conv2D(filters=1024, kernel_size=(1, 1), padding="same")
        self.model_pred5_3 = BatchNormalization()
        self.model_pred5_4 = Conv2D(filters=512, kernel_size=(1, 1), padding="same")
        self.model_pred5_5 = GlobalMaxPool2D()
        self.model_pred5_6 = Dense(units=1, activation="sigmoid")

        self.model_pred6_2 = Conv2D(filters=1024, kernel_size=(1, 1), padding="same")
        self.model_pred6_3 = BatchNormalization()
        self.model_pred6_4 = Conv2D(filters=512, kernel_size=(1, 1), padding="same")
        self.model_pred6_5 = GlobalMaxPool2D()
        self.model_pred6_6 = Dense(units=1, activation="sigmoid")

    def call(self, x, **kwargs):
        y = self.model(x)
        pred1 = y[:, :, :, :400]
        pred2 = y[:, :, :, 400:800]
        pred3 = y[:, :, :, 800:1200]
        pred4 = y[:, :, :, 1200:1600]
        pred5 = y[:, :, :, 1600:2000]
        pred6 = y[:, :, :, 2000:2400]

        pred1 = self.model_pred1_2(pred1)
        pred1 = self.model_pred1_3(pred1)
        pred1 = self.model_pred1_4(pred1)
        pred1 = self.model_pred1_5(pred1)
        pred1 = self.model_pred1_6(pred1)

        pred2 = self.model_pred2_2(pred2)
        pred2 = self.model_pred2_3(pred2)
        pred2 = self.model_pred2_4(pred2)
        pred2 = self.model_pred2_5(pred2)
        pred2 = self.model_pred2_6(pred2)

        pred3 = self.model_pred3_2(pred3)
        pred3 = self.model_pred3_3(pred3)
        pred3 = self.model_pred3_4(pred3)
        pred3 = self.model_pred3_5(pred3)
        pred3 = self.model_pred3_6(pred3)

        pred4 = self.model_pred4_2(pred4)
        pred4 = self.model_pred4_3(pred4)
        pred4 = self.model_pred4_4(pred4)
        pred4 = self.model_pred4_5(pred4)
        pred4 = self.model_pred4_6(pred4)

        pred5 = self.model_pred5_2(pred5)
        pred5 = self.model_pred5_3(pred5)
        pred5 = self.model_pred5_4(pred5)
        pred5 = self.model_pred5_5(pred5)
        pred5 = self.model_pred5_6(pred5)

        pred6 = self.model_pred6_2(pred6)
        pred6 = self.model_pred6_3(pred6)
        pred6 = self.model_pred6_4(pred6)
        pred6 = self.model_pred6_5(pred6)
        pred6 = self.model_pred6_6(pred6)

        return concat([pred1, pred2, pred3, pred4, pred5, pred6], axis=1)

    def create_model(self):
        return self.model


def _resolve_ckpt_prefix(path: str) -> str:
    """
    BUGFIX: Keras 3 no longer loads TF checkpoints via model.load_weights(directory/prefix).
    We resolve a checkpoint prefix to use with tf.train.Checkpoint instead.
    """
    if not path or not tf.io.gfile.exists(path):
        return path

    if tf.io.gfile.isdir(path):
        ckpt = tf.train.latest_checkpoint(path)
        if ckpt is not None:
            return ckpt
        for f in tf.io.gfile.listdir(path):
            if f.endswith(".index"):
                return os.path.join(path, f[:-6])
        return path

    if path.endswith(".index"):
        return path[:-6]

    return path




## === cell 2
if __name__ == "__main__":
    model = MultiLabel()
    model.build(input_shape=[None, image_dims[0], image_dims[1], image_dims[2]])

    resolved_ckpt = _resolve_ckpt_prefix(model_dir)
    print("Resolved checkpoint prefix/path:", resolved_ckpt)

    try:
        ckpt = tf.train.Checkpoint(model=model)
        status = ckpt.restore(resolved_ckpt)
        status.expect_partial()
        print("Restored checkpoint OK (expect_partial).")
    except Exception as e:
        raise RuntimeError(
            f"Failed to restore weights from '{resolved_ckpt}'. Original model_dir was '{model_dir}'."
        ) from e

    images_path_list = sorted(
        [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
    )

    def test_on_sub(index):
        img_path = os.path.join(test_dir, images_path_list[index])
        input_img = tf.io.read_file(img_path)
        image = tf.io.decode_jpeg(input_img, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)  # [0,1]
        tensor_image = tf.image.resize(image, [image_dims[0], image_dims[1]])
        name_jpg = images_path_list[index].split(os.path.sep)[-1]
        return name_jpg, tf.expand_dims(tensor_image, axis=0)

    values = []
    classes = dataset_labels

    for idx in range(len(images_path_list)):
        name, images = test_on_sub(index=idx)
        images = images * 255.0  # keep original scaling

        test_values = model.call(images)

        index_values = [
            i for i, v in enumerate(test_values[0].numpy().tolist()) if v > 0.7
        ]

        classes_img_list = [str(classes[i]) for i in index_values]
        if len(classes_img_list) == 0:
            classes_img = "healthy"
        else:
            classes_img = " ".join(classes_img_list)

        values.append([name, classes_img])

    csv_pd = pd.DataFrame(values, columns=["image", "labels"], index=None)

    out_path = os.path.join(output_dir, "submission.csv")
    csv_pd.to_csv(out_path, index=False)
    print("Wrote submission.csv with shape:", csv_pd.shape, "to:", out_path)
    print(csv_pd.head())

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/tensorflow/python/training/py_checkpoint_reader.py in NewCheckpointReader(filepattern)
     91   try:
---> 92     return CheckpointReader(compat.as_bytes(filepattern))
     93   # TODO(b/143319754): Remove the RuntimeError casting logic once we resolve the

RuntimeError: Unsuccessful TensorSliceReader constructor: Failed to find any matching files for ../input/conve01/effb7-e12/epoch-12

During handling of the above exception, another exception occurred:

NotFoundError                             Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/tensorflow/python/checkpoint/checkpoint.py in restore(self, save_path, options)
   2731     try:
-> 2732       status = self.read(save_path, options=options)
   2733       if context.executing_eagerly():

/usr/local/lib/python3.11/dist-packages/tensorflow/python/checkpoint/checkpoint.py in read(self, save_path, options)
   2594     options = options or checkpoint_options.CheckpointOptions()
-> 2595     result = self._saver.restore(save_path=save_path, options=options)
   2596     metrics.AddCheckpointReadDuration(

/usr/local/lib/python3.11/dist-packages/tensorflow/python/checkpoint/checkpoint.py in restore(self, save_path, options)
   1455       _ASYNC_CHECKPOINT_THREAD.join()
-> 1456     reader = py_checkpoint_reader.NewCheckpointReader(save_path)
   1457     graph_building = not context.executing_eagerly()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/training/py_checkpoint_reader.py in NewCheckpointReader(filepattern)
     95   except RuntimeError as e:
---> 96     error_translator(e)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/training/py_checkpoint_reader.py in error_translator(e)
     30       'matching files for') in error_message:
---> 31     raise errors_impl.NotFoundError(None, None, error_message)
     32   elif 'Sliced checkpoints are not supported' in error_message or (

NotFoundError: Unsuccessful TensorSliceReader constructor: Failed to find any matching files for ../input/conve01/effb7-e12/epoch-12

During handling of the above exception, another exception occurred:

NotFoundError                             Traceback (most recent call last)
/tmp/ipykernel_11/3337066655.py in <cell line: 0>()
     10         ckpt = tf.train.Checkpoint(model=model)
---> 11         status = ckpt.restore(resolved_ckpt)
     12         # Expect partial is OK if optimizer slots aren't present; variables should still restore.

/usr/local/lib/python3.11/dist-packages/tensorflow/python/checkpoint/checkpoint.py in restore(self, save_path, options)
   2735     except errors_impl.NotFoundError as e:
-> 2736       raise errors_impl.NotFoundError(
   2737           None, None,

NotFoundError: Error when restoring from checkpoint or SavedModel at ../input/conve01/effb7-e12/epoch-12: Unsuccessful TensorSliceReader constructor: Failed to find any matching files for ../input/conve01/effb7-e12/epoch-12
Please double-check that the path is correct. You may be missing the checkpoint suffix (e.g. the '-1' in 'path/to/ckpt-1').

The above exception was the direct cause of the following exception:

RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/3337066655.py in <cell line: 0>()
     14         print("Restored checkpoint OK (expect_partial).")
     15     except Exception as e:
---> 16         raise RuntimeError(
     17             f"Failed to restore weights from '{resolved_ckpt}'. Original model_dir was '{model_dir}'."
     18         ) from e

RuntimeError: Failed to restore weights from '../input/conve01/effb7-e12/epoch-12'. Original model_dir was '../input/conve01/effb7-e12/epoch-12'.
