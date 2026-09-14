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

3.9

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
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

0.8679359323058325

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

import numpy as np
import pandas as pd
import tensorflow as tf
from functools import partial

print("TensorFlow version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
AUTOTUNE = tf.data.experimental.AUTOTUNE

FILENAMES = tf.io.gfile.glob(os.path.join(DATA_DIR, "train_tfrecords", "*.tfrec"))
split_ind = int(0.9 * len(FILENAMES))
TRAINING_FILENAMES, VALID_FILENAMES = FILENAMES[:split_ind], FILENAMES[split_ind:]

CLASS_NAMES = pd.read_json(
    os.path.join(DATA_DIR, "label_num_to_disease_map.json"), typ="series"
)

BATCH_SIZE = 16
IMG_HEIGHT = 400
IMG_WIDTH = 300
NUM_CLASSES = 5



## === cell 2
import re


def count_data_items(filenames):
    counts = [int(re.search(r"-([0-9]+)\.tfrec", fn).group(1)) for fn in filenames]
    return sum(counts)


TEST_FILENAMES = tf.io.gfile.glob(os.path.join(DATA_DIR, "test_tfrecords", "*.tfrec"))
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)




## === cell 3
def decode_image(image_bytes):
    img = tf.image.decode_jpeg(image_bytes, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    return img


def read_tfrecord(example, labeled):
    if labeled:
        feature_desc = {
            "target": tf.io.FixedLenFeature([], tf.int64),
            "image_name": tf.io.FixedLenFeature([], tf.string),
            "image": tf.io.FixedLenFeature([], tf.string),
        }
    else:
        feature_desc = {
            "image_name": tf.io.FixedLenFeature([], tf.string),
            "image": tf.io.FixedLenFeature([], tf.string),
        }
    example = tf.io.parse_single_example(example, feature_desc)
    image = decode_image(example["image"])

    image = tf.image.adjust_saturation(image, 0.3)
    image = tf.image.adjust_hue(image, -0.2)  # hue delta must be in [-0.5, 0.5]
    image = tf.image.adjust_contrast(image, 1.0)
    image = tf.image.adjust_brightness(image, 0.3)

    if labeled:
        label = tf.cast(
            example["target"], tf.int32
        )  # keep as integer for SparseCategoricalCrossentropy
        return image, label
    else:
        idnum = example["image_name"]
        return image, idnum


def load_dataset(filenames, labeled=True, ordered=False):
    options = tf.data.Options()
    if not ordered:
        options.experimental_deterministic = False
    ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE)
    ds = ds.with_options(options)
    ds = ds.map(partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE)
    return ds


def get_dataset(filenames, labeled=True, ordered=False, shuffle=False):
    ds = load_dataset(filenames, labeled=labeled, ordered=ordered)
    if shuffle:
        ds = ds.shuffle(1024)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


train_dataset = get_dataset(TRAINING_FILENAMES, labeled=True, shuffle=True)
valid_dataset = get_dataset(VALID_FILENAMES, labeled=True, ordered=True)



## === cell 4
base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights="imagenet"
)
base_model.trainable = False  # freeze base

inputs = tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = tf.keras.applications.efficientnet.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)

model.summary()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
gaierror                                  Traceback (most recent call last)
/usr/lib/python3.11/urllib/request.py in do_open(self, http_class, req, **http_conn_args)
   1347             try:
-> 1348                 h.request(req.get_method(), req.selector, req.data, headers,
   1349                           encode_chunked=req.has_header('Transfer-encoding'))

/usr/lib/python3.11/http/client.py in request(self, method, url, body, headers, encode_chunked)
   1302         """Send a complete request to the server."""
-> 1303         self._send_request(method, url, body, headers, encode_chunked)
   1304 

/usr/lib/python3.11/http/client.py in _send_request(self, method, url, body, headers, encode_chunked)
   1348             body = _encode(body, 'body')
-> 1349         self.endheaders(body, encode_chunked=encode_chunked)
   1350 

/usr/lib/python3.11/http/client.py in endheaders(self, message_body, encode_chunked)
   1297             raise CannotSendHeader()
-> 1298         self._send_output(message_body, encode_chunked=encode_chunked)
   1299 

/usr/lib/python3.11/http/client.py in _send_output(self, message_body, encode_chunked)
   1057         del self._buffer[:]
-> 1058         self.send(msg)
   1059 

/usr/lib/python3.11/http/client.py in send(self, data)
    995             if self.auto_open:
--> 996                 self.connect()
    997             else:

/usr/lib/python3.11/http/client.py in connect(self)
   1467 
-> 1468             super().connect()
   1469 

/usr/lib/python3.11/http/client.py in connect(self)
    961         sys.audit("http.client.connect", self, self.host, self.port)
--> 962         self.sock = self._create_connection(
    963             (self.host,self.port), self.timeout, self.source_address)

/usr/lib/python3.11/socket.py in create_connection(address, timeout, source_address, all_errors)
    838     exceptions = []
--> 839     for res in getaddrinfo(host, port, 0, SOCK_STREAM):
    840         af, socktype, proto, canonname, sa = res

/usr/lib/python3.11/socket.py in getaddrinfo(host, port, family, type, proto, flags)
    973     addrlist = []
--> 974     for res in _socket.getaddrinfo(host, port, family, type, proto, flags):
    975         af, socktype, proto, canonname, sa = res

gaierror: [Errno -3] Temporary failure in name resolution

During handling of the above exception, another exception occurred:

URLError                                  Traceback (most recent call last)
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
    518         sys.audit('urllib.Request', req.full_url, req.data, req.headers, req.get_method())
--> 519         response = self._open(req, data)
    520 

/usr/lib/python3.11/urllib/request.py in _open(self, req, data)
    535         protocol = req.type
--> 536         result = self._call_chain(self.handle_open, protocol, protocol +
    537                                   '_open', req)

/usr/lib/python3.11/urllib/request.py in _call_chain(self, chain, kind, meth_name, *args)
    495             func = getattr(handler, meth_name)
--> 496             result = func(*args)
    497             if result is not None:

/usr/lib/python3.11/urllib/request.py in https_open(self, req)
   1390         def https_open(self, req):
-> 1391             return self.do_open(http.client.HTTPSConnection, req,
   1392                 context=self._context, check_hostname=self._check_hostname)

/usr/lib/python3.11/urllib/request.py in do_open(self, http_class, req, **http_conn_args)
   1350             except OSError as err: # timeout error
-> 1351                 raise URLError(err)
   1352             r = h.getresponse()

URLError: <urlopen error [Errno -3] Temporary failure in name resolution>

During handling of the above exception, another exception occurred:

Exception                                 Traceback (most recent call last)
/tmp/ipykernel_55/1849466454.py in <cell line: 0>()
      1 # Build a lightweight transfer‑learning model
----> 2 base_model = tf.keras.applications.EfficientNetB0(
      3     input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights="imagenet"
      4 )
      5 base_model.trainable = False  # freeze base

/usr/local/lib/python3.11/dist-packages/keras/src/applications/efficientnet.py in EfficientNetB0(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)
    569     name="efficientnetb0",
    570 ):
--> 571     return EfficientNet(
    572         1.0,
    573         1.0,

/usr/local/lib/python3.11/dist-packages/keras/src/applications/efficientnet.py in EfficientNet(width_coefficient, depth_coefficient, default_size, dropout_rate, drop_connect_rate, depth_divisor, activation, blocks_args, name, include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, weights_name)
    426             file_hash = WEIGHTS_HASHES[weights_name][1]
    427         file_name = name + file_suffix
--> 428         weights_path = file_utils.get_file(
    429             file_name,
    430             BASE_WEIGHTS_PATH + file_name,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/file_utils.py in get_file(fname, origin, untar, md5_hash, file_hash, cache_subdir, hash_algorithm, extract, archive_format, cache_dir, force_download)
    313                 raise Exception(error_msg.format(origin, e.code, e.msg))
    314             except urllib.error.URLError as e:
--> 315                 raise Exception(error_msg.format(origin, e.errno, e.reason))
    316         except (Exception, KeyboardInterrupt):
    317             if os.path.exists(download_target):

Exception: URL fetch failure on https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5: None -- [Errno -3] Temporary failure in name resolution

## === cell 5
model.fit(train_dataset, validation_data=valid_dataset, epochs=2, verbose=2)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3303924362.py in <cell line: 0>()
      1 # Train briefly – enough to generate predictions without long runtime
----> 2 model.fit(train_dataset, validation_data=valid_dataset, epochs=2, verbose=2)
      3 

NameError: name 'model' is not defined

## === cell 6
test_dataset = get_dataset(TEST_FILENAMES, labeled=False, ordered=True)

probabilities = model.predict(test_dataset, verbose=0)
predictions = tf.argmax(probabilities, axis=1).numpy()

test_ids_dataset = test_dataset.map(lambda img, idnum: idnum).unbatch()
test_ids = np.array(list(test_ids_dataset.as_numpy_iterator())).astype(str)

assert len(test_ids) == len(predictions), "Mismatch between ids and predictions"



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/349894447.py in <cell line: 0>()
      3 
      4 # Predict probabilities and class indices
----> 5 probabilities = model.predict(test_dataset, verbose=0)
      6 predictions = tf.argmax(probabilities, axis=1).numpy()
      7 

NameError: name 'model' is not defined

## === cell 7
submission_path = "submission.csv"
submission_df = pd.DataFrame({"image_id": test_ids, "label": predictions})
submission_df.to_csv(submission_path, index=False)

print(submission_df.head())

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3878105917.py in <cell line: 0>()
      1 # Save submission file matching Kaggle format
      2 submission_path = "submission.csv"
----> 3 submission_df = pd.DataFrame({"image_id": test_ids, "label": predictions})
      4 submission_df.to_csv(submission_path, index=False)
      5 

NameError: name 'test_ids' is not defined
