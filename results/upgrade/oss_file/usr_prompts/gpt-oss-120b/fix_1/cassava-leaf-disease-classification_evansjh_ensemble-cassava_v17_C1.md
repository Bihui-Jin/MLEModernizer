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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.13

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.8964944091870656

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from collections import Counter
import shutil

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 5
source_dir = '/kaggle/input/cp-model'
dest_dir = '/kaggle/working/cp-model'

if os.path.exists(dest_dir):
    shutil.rmtree(dest_dir)

shutil.copytree(source_dir, dest_dir)

os.environ["TFHUB_CACHE_DIR"] = dest_dir

model_url = "https://tfhub.dev/google/cropnet/classifier/cassava_disease_V1/2"
classifier = hub.load(model_url)

print("Model loaded successfully from writable cache directory!")

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4096318692.py in <cell line: 0>()
      8 
      9 # Copy the cached model to a writable location
---> 10 shutil.copytree(source_dir, dest_dir)
     11 
     12 # Set up the cache directory to point to the writable location

/usr/lib/python3.11/shutil.py in copytree(src, dst, symlinks, ignore, copy_function, ignore_dangling_symlinks, dirs_exist_ok)
    569     """
    570     sys.audit("shutil.copytree", src, dst)
--> 571     with os.scandir(src) as itr:
    572         entries = list(itr)
    573     return _copytree(entries=entries, src=src, dst=dst, symlinks=symlinks,

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/cp-model'

## === cell 6
model_path_1 = '/kaggle/input/combinedmodel3/tensorflow2/default/1/BestModel_3454_8937.h5'
model_path_2 = '/kaggle/input/combinedmodel3/tensorflow2/default/1/best_model_0.37458707.h5'
model_path_3 = '/kaggle/input/combinedmodel3/tensorflow2/default/1/googlenet_inceptionv3.h5'
model_path_4 = '/kaggle/input/bestmodel_550_2/tensorflow2/default/1/BestModel_3577_8940.h5'
model_path_5 = '/kaggle/input/bestmodel_8878/tensorflow2/default/1/BestModel_8878_0358.h5'
model_path_6 = '/kaggle/input/bestmodel_8875/tensorflow2/default/1/BestModel_8875.h5'
model_path_7 = '/kaggle/input/googlenet_512/tensorflow2/default/1/GOOG1.h5'
model_path_8 = classifier
test_image_dir = '/kaggle/input/cassava-leaf-disease-classification/test_images'
sample = '/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv'

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3825230640.py in <cell line: 0>()
      7 model_path_6 = '/kaggle/input/bestmodel_8875/tensorflow2/default/1/BestModel_8875.h5'
      8 model_path_7 = '/kaggle/input/googlenet_512/tensorflow2/default/1/GOOG1.h5'
----> 9 model_path_8 = classifier
     10 test_image_dir = '/kaggle/input/cassava-leaf-disease-classification/test_images'
     11 sample = '/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv'

NameError: name 'classifier' is not defined

## === cell 7
sample_csv = pd.read_csv(sample)
sample_csv

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3793615243.py in <cell line: 0>()
----> 1 sample_csv = pd.read_csv(sample)
      2 sample_csv

NameError: name 'sample' is not defined

## === cell 9
models_info = [
    (model_path_1, (550, 550)),
    (model_path_6, (512, 512)),
]

models = [(load_model(path), input_size) for path, input_size in models_info]
models.append((model_path_8, (224, 224)))

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1834239231.py in <cell line: 0>()
      6 
      7 # Load models with specified input sizes
----> 8 models = [(load_model(path), input_size) for path, input_size in models_info]
      9 models.append((model_path_8, (224, 224)))

/tmp/ipykernel_11/1834239231.py in <listcomp>(.0)
      6 
      7 # Load models with specified input sizes
----> 8 models = [(load_model(path), input_size) for path, input_size in models_info]
      9 models.append((model_path_8, (224, 224)))

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    194         )
    195     if str(filepath).endswith((".h5", ".hdf5")):
--> 196         return legacy_h5_format.load_model_from_hdf5(
    197             filepath, custom_objects=custom_objects, compile=compile
    198         )

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/saving/legacy_h5_format.py in load_model_from_hdf5(filepath, custom_objects, compile)
    114     opened_new_file = not isinstance(filepath, h5py.File)
    115     if opened_new_file:
--> 116         f = h5py.File(filepath, mode="r")
    117     else:
    118         f = filepath

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in __init__(self, name, mode, driver, libver, userblock_size, swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order, fs_strategy, fs_persist, fs_threshold, fs_page_size, page_buf_size, min_meta_keep, min_raw_keep, locking, alignment_threshold, alignment_interval, meta_block_size, **kwds)
    562                                  fs_persist=fs_persist, fs_threshold=fs_threshold,
    563                                  fs_page_size=fs_page_size)
--> 564                 fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)
    565 
    566             if isinstance(libver, tuple):

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in make_fid(name, mode, userblock_size, fapl, fcpl, swmr)
    236         if swmr and swmr_support:
    237             flags |= h5f.ACC_SWMR_READ
--> 238         fid = h5f.open(name, flags, fapl=fapl)
    239     elif mode == 'r+':
    240         fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/h5f.pyx in h5py.h5f.open()

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = '/kaggle/input/combinedmodel3/tensorflow2/default/1/BestModel_3454_8937.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 10
class_labels = {
    0: 'Cassava Bacterial Blight (CBB)',
    1: 'Cassava Brown Streak Disease (CBSD)',
    2: 'Cassava Green Mottle (CGM)',
    3: 'Cassava Mosaic Disease (CMD)',
    4: 'Healthy'
}

image_predictions = []

for image_id in os.listdir(test_image_dir):
    if image_id.endswith(('.jpg', '.jpeg', '.png')):
        model_predictions = []
        confidence_scores = {}

        for model, input_size in models:
            img = load_img(os.path.join(test_image_dir, image_id), target_size=input_size)
            img_array = np.expand_dims(img_to_array(img) / 255.0, axis=0)
            
            if (model == model_path_8):
                predictions = classifier(img_array, training=False)  # Add training=False
                predicted_class = tf.math.argmax(predictions, axis=-1).numpy()[0]  # Get the predicted class index
            else:
                predictions = model.predict(img_array)
                predicted_class = np.argmax(predictions, axis=1)[0]
                
            confidence_score = predictions[0][predicted_class]
            
            model_predictions.append(predicted_class)
            if predicted_class not in confidence_scores:
                confidence_scores[predicted_class] = []
            confidence_scores[predicted_class].append(confidence_score)

        class_votes = Counter(model_predictions)
        most_common = class_votes.most_common()
        final_predicted_class = most_common[0][0]
        
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            tied_classes = [cls for cls, count in most_common if count == most_common[0][1]]
            for cls in tied_classes:
                avg_confidence = sum(confidence_scores[cls]) / len(confidence_scores[cls])
            
            final_predicted_class = max(
                tied_classes,
                key=lambda cls: sum(confidence_scores[cls]) / len(confidence_scores[cls])
            )

        image_predictions.append({'image_id': image_id, 'label': final_predicted_class})

submission_df = pd.DataFrame(image_predictions)
submission_df.to_csv('/kaggle/working/submission.csv', index=False)


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1079165703.py in <cell line: 0>()
      9 image_predictions = []
     10 
---> 11 for image_id in os.listdir(test_image_dir):
     12     if image_id.endswith(('.jpg', '.jpeg', '.png')):
     13         model_predictions = []

NameError: name 'test_image_dir' is not defined

## === cell 11
submission_df

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2788716369.py in <cell line: 0>()
----> 1 submission_df

NameError: name 'submission_df' is not defined
