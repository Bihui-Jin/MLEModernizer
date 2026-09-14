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

geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
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

0.7845060018467225

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
import tensorflow as tf



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
model_dir = "../input/conv2de01/epoch-1"

image_dims = (300, 300, 3)

data_set = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()



## === cell 2
if __name__ == "__main__":
    model = tf.keras.layers.TFSMLayer(model_dir, call_endpoint="serving_default")

    images_path_list = sorted(os.listdir(test_dir))

    def load_image(index):
        img_path = os.path.join(test_dir, images_path_list[index])
        img_bytes = tf.io.read_file(img_path)
        img = tf.io.decode_image(img_bytes, channels=3, dtype=tf.dtypes.float32)
        img_resized = tf.image.resize(img, [image_dims[0], image_dims[1]])
        return images_path_list[index], tf.expand_dims(img_resized, axis=0)

    predictions = []
    for idx in range(len(images_path_list)):
        name, img_tensor = load_image(idx)
        img_tensor = img_tensor * 255.0
        logits = model(img_tensor)
        if isinstance(logits, dict):
            logits = list(logits.values())[0]
        probs = tf.squeeze(logits).numpy()
        selected_idx = [i for i, p in enumerate(probs) if p > 0.7]
        label_str = " ".join(dataset_labels[i] for i in selected_idx).strip()
        predictions.append([name, label_str])

    submission_df = pd.DataFrame(predictions, columns=["image", "labels"])
    submission_df.to_csv(os.path.join(output_dir, "submission.csv"), index=False)

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_55/95176067.py in <cell line: 0>()
      1 if __name__ == "__main__":
      2     # Load SavedModel as a Keras layer (compatible with TF 2.18 / Keras 3)
----> 3     model = tf.keras.layers.TFSMLayer(model_dir, call_endpoint="serving_default")
      4 
      5     images_path_list = sorted(os.listdir(test_dir))

/usr/local/lib/python3.11/dist-packages/keras/src/export/tfsm_layer.py in __init__(self, filepath, call_endpoint, call_training_endpoint, trainable, name, dtype)
     64         super().__init__(trainable=trainable, name=name, dtype=dtype)
     65 
---> 66         self._reloaded_obj = tf.saved_model.load(filepath)
     67 
     68         self.filepath = filepath

/usr/local/lib/python3.11/dist-packages/tensorflow/python/saved_model/load.py in load(export_dir, tags, options)
    910   if isinstance(export_dir, os.PathLike):
    911     export_dir = os.fspath(export_dir)
--> 912   result = load_partial(export_dir, None, tags, options)["root"]
    913   return result
    914 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/saved_model/load.py in load_partial(export_dir, filters, tags, options)
   1014     tags = nest.flatten(tags)
   1015   saved_model_proto, debug_info = (
-> 1016       loader_impl.parse_saved_model_with_debug_info(export_dir))
   1017 
   1018   loader = None

/usr/local/lib/python3.11/dist-packages/tensorflow/python/saved_model/loader_impl.py in parse_saved_model_with_debug_info(export_dir)
     57     parsed. Missing graph debug info file is fine.
     58   """
---> 59   saved_model = parse_saved_model(export_dir)
     60 
     61   debug_info_path = file_io.join(

/usr/local/lib/python3.11/dist-packages/tensorflow/python/saved_model/loader_impl.py in parse_saved_model(export_dir)
    117       raise IOError(f"Cannot parse file {path_to_pbtxt}: {str(e)}.") from e
    118   else:
--> 119     raise IOError(
    120         f"SavedModel file does not exist at: {export_dir}{os.path.sep}"
    121         f"{{{constants.SAVED_MODEL_FILENAME_PBTXT}|"

OSError: SavedModel file does not exist at: ../input/conv2de01/epoch-1/{saved_model.pbtxt|saved_model.pb}
