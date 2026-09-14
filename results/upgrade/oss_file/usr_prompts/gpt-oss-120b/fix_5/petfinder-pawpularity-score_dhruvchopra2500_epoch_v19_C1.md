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
Predict engagement with a pet's profile based on the photograph for that profile.

## Metric
Root mean squared error.

## Submission Format
For each `Id` in the test set, you must predict a probability for the target variable, `Pawpularity`. The file should contain a header and have the following format:

```
Id, Pawpularity
0008dbfb52aa1dc6ee51ee02adf13537, 99.24
0014a7b528f1682f0cf3b73a991c17a0, 61.71
0019c1388dfcd30ac8b112fb4250c251, 6.23
00307b779c82716b240a24f028b0031b, 9.43
00320c6dd5b4223c62a9670110d47911, 70.89
etc.
```

## Dataset
- **train/** - Folder containing training set photos of the form **{id}.jpg**, where **{id}** is a unique Pet Profile ID.
- **train.csv** - Metadata (described below) for each photo in the training set as well as the target, the photo's Pawpularity score. The Id column gives the photo's unique Pet Profile ID corresponding the photo's file name.

The train.csv and test.csv files contain metadata for photos in the training set and test set, respectively. Each pet photo is labeled with the value of 1 (Yes) or 0 (No) for each of the following features:

- **Focus** - Pet stands out against uncluttered background, not too close / far.
- **Eyes** - Both eyes are facing front or near-front, with at least 1 eye / pupil decently clear.
- **Face** - Decently clear face, facing front or near-front.
- **Near** - Single pet taking up significant portion of photo (roughly over 50% of photo width or height).
- **Action** - Pet in the middle of an action (e.g., jumping).
- **Accessory** - Accompanying physical or digital accessory / prop (i.e. toy, digital sticker), excluding collar and leash.
- **Group** - More than 1 pet in the photo.
- **Collage** - Digitally-retouched photo (i.e. with digital photo frame, combination of multiple photos).
- **Human** - Human in the photo.
- **Occlusion** - Specific undesirable objects blocking part of the pet (i.e. human, cage or fence). Note that not all blocking objects are considered occlusion.
- **Info** - Custom-added text or labels (i.e. pet name, description).
- **Blur** - Noticeably out of focus or noisy, especially for the pet's eyes and face. For Blur entries, "Eyes" column is always set to 0.

# 2. Python version

3.12

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
        input/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
        working/
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
```

-> data/petfinder-pawpularity-score/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/petfinder-pawpularity-score/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/petfinder-pawpularity-score/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> data/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> (stopped after 10 files for performance)

# 5. Target score

20.89973

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import tensorflow as tf




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def pythonic_loader_train():
    IMAGES_PATH = "/kaggle/input/petfinder-pawpularity-score/train/"
    CSV_PATH = "/kaggle/input/petfinder-pawpularity-score/train.csv"

    relevant_columns = [
        "Subject Focus",
        "Eyes",
        "Face",
        "Near",
        "Action",
        "Accessory",
        "Group",
        "Collage",
        "Human",
        "Occlusion",
        "Info",
        "Blur",
    ]
    target = "Pawpularity"

    image_names = os.listdir(IMAGES_PATH)
    image_names = image_names[: int(len(image_names) * 0.95)]
    np.random.shuffle(image_names)
    metadata_csv = pd.read_csv(CSV_PATH)

    for i in image_names:
        img = tf.keras.utils.load_img(
            path=IMAGES_PATH + i, color_mode="rgb", target_size=(256, 256)
        )
        img = np.array(img).astype(np.float32) / 255.0  # normalize

        metadata = metadata_csv[metadata_csv["Id"] == i[:-4]]
        features = metadata[relevant_columns].values[0].astype(np.float32)

        y = metadata[target].values[0].astype(np.float32)

        yield ({"Image": img, "Feature": features}, y)




## === cell 2
def pythonic_loader_test():
    IMAGES_PATH = "/kaggle/input/petfinder-pawpularity-score/test/"
    CSV_PATH = "/kaggle/input/petfinder-pawpularity-score/test.csv"

    relevant_columns = [
        "Subject Focus",
        "Eyes",
        "Face",
        "Near",
        "Action",
        "Accessory",
        "Group",
        "Collage",
        "Human",
        "Occlusion",
        "Info",
        "Blur",
    ]
    metadata_csv = pd.read_csv(CSV_PATH)

    for i in sorted(os.listdir(IMAGES_PATH)):
        img = tf.keras.utils.load_img(
            path=IMAGES_PATH + i, color_mode="rgb", target_size=(256, 256)
        )
        img = np.array(img).astype(np.float32) / 255.0

        metadata = metadata_csv[metadata_csv["Id"] == i[:-4]]
        features = metadata[relevant_columns].values[0].astype(np.float32)

        yield {"Image": img, "Feature": features}




## === cell 3
train_loader = tf.data.Dataset.from_generator(
    pythonic_loader_train,
    output_signature=(
        {
            "Image": tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
            "Feature": tf.TensorSpec(shape=(12,), dtype=tf.float32),
        },
        tf.TensorSpec(shape=(), dtype=tf.float32),
    ),
)



## === cell 4
input_image = tf.keras.Input(shape=(256, 256, 3), name="Image")
input_meta = tf.keras.Input(shape=(12,), name="Feature")

l2 = tf.keras.layers.MaxPool2D((2, 2))(input_image)
l3 = tf.keras.layers.Conv2D(8, (3, 3), activation="relu")(l2)
l4 = tf.keras.layers.MaxPool2D((2, 2))(l3)
l5 = tf.keras.layers.Conv2D(16, (3, 3), activation="relu")(l4)
l6 = tf.keras.layers.MaxPool2D((2, 2))(l5)
l7 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu")(l6)
l_mid1 = tf.keras.layers.MaxPool2D((2, 2))(l7)
l_mid2 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu")(l_mid1)
l8 = tf.keras.layers.Flatten()(l_mid2)
l10 = tf.keras.layers.Dense(1024, activation="gelu")(l8)

combined = tf.keras.layers.concatenate([l10, input_meta])
l12 = tf.keras.layers.Dense(512, activation="gelu")(combined)
bn1 = tf.keras.layers.BatchNormalization()(l12)
le1 = tf.keras.layers.Dense(512, activation="gelu")(bn1)
bn2 = tf.keras.layers.BatchNormalization()(le1)
le2 = tf.keras.layers.Dense(512, activation="gelu")(bn2)
bn3 = tf.keras.layers.BatchNormalization()(le2)
le3 = tf.keras.layers.Dense(512, activation="gelu")(bn3)
bn4 = tf.keras.layers.BatchNormalization()(le3)
le4 = tf.keras.layers.Dense(256, activation="gelu")(bn4)
bn_5 = tf.keras.layers.BatchNormalization()(le4)
le5 = tf.keras.layers.Dense(256, activation="gelu")(bn_5)
bn_6 = tf.keras.layers.BatchNormalization()(le5)

layer1 = tf.keras.layers.Dense(256, activation="gelu")(bn_6)
layer2 = tf.keras.layers.BatchNormalization()(layer1)
layer3 = tf.keras.layers.Dense(256, activation="gelu")(layer2)
layer4 = tf.keras.layers.BatchNormalization()(layer3)
layer5 = tf.keras.layers.Dense(256, activation="gelu")(layer4)
layer6 = tf.keras.layers.BatchNormalization()(layer5)

le6 = tf.keras.layers.Dense(128, activation="gelu")(layer6)
bn5 = tf.keras.layers.BatchNormalization()(le6)
le9 = tf.keras.layers.Dense(64, activation="gelu")(bn5)
bn6 = tf.keras.layers.BatchNormalization()(le9)
l13 = tf.keras.layers.Dense(16, activation="gelu")(bn6)
bn7 = tf.keras.layers.BatchNormalization()(l13)
l14 = tf.keras.layers.Dense(1, activation="sigmoid")(bn7)
output = l14 * tf.constant([100.0], dtype=tf.float32)

model = tf.keras.Model(
    inputs={"Image": input_image, "Feature": input_meta},
    outputs=output,
)



## === cell 5
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=1e-3, decay_steps=200
        )
    ),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.RootMeanSquaredError()],
)



## === cell 6
gen_train = train_loader.batch(32).prefetch(tf.data.AUTOTUNE).repeat()

steps_per_epoch = int(
    len(os.listdir("/kaggle/input/petfinder-pawpularity-score/train/")) * 0.95 // 32
)

model.fit(
    gen_train,
    steps_per_epoch=steps_per_epoch,
    epochs=4,
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
UnknownError                              Traceback (most recent call last)
/tmp/ipykernel_56/4281252166.py in <cell line: 0>()
      5 )
      6 
----> 7 model.fit(
      8     gen_train,
      9     steps_per_epoch=steps_per_epoch,

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
Detected at node PyFunc defined at (most recent call last):
<stack traces unavailable>
2 root error(s) found.
  (0) UNKNOWN:  IsADirectoryError: [Errno 21] Is a directory: '/kaggle/input/petfinder-pawpularity-score/train/train'
Traceback (most recent call last):

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 269, in __call__
    ret = func(*args)
          ^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py", line 643, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/from_generator_op.py", line 198, in generator_py_func
    values = next(generator_state.get_iterator(iterator_id))
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_56/3750622442.py", line 27, in pythonic_loader_train
    img = tf.keras.utils.load_img(
          ^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/image_utils.py", line 235, in load_img
    with open(path, "rb") as f:
         ^^^^^^^^^^^^^^^^

IsADirectoryError: [Errno 21] Is a directory: '/kaggle/input/petfinder-pawpularity-score/train/train'


	 [[{{node PyFunc}}]]
	 [[IteratorGetNext]]
	 [[IteratorGetNext/_2]]
  (1) UNKNOWN:  IsADirectoryError: [Errno 21] Is a directory: '/kaggle/input/petfinder-pawpularity-score/train/train'
Traceback (most recent call last):

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 269, in __call__
    ret = func(*args)
          ^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py", line 643, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/from_generator_op.py", line 198, in generator_py_func
    values = next(generator_state.get_iterator(iterator_id))
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_56/3750622442.py", line 27, in pythonic_loader_train
    img = tf.keras.utils.load_img(
          ^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/keras/src/utils/image_utils.py", line 235, in load_img
    with open(path, "rb") as f:
         ^^^^^^^^^^^^^^^^

IsADirectoryError: [Errno 21] Is a directory: '/kaggle/input/petfinder-pawpularity-score/train/train'


	 [[{{node PyFunc}}]]
	 [[IteratorGetNext]]
0 successful operations.
0 derived errors ignored. [Op:__inference_multi_step_on_iterator_12403]

## === cell 7
test_loader = tf.data.Dataset.from_generator(
    pythonic_loader_test,
    output_signature=(
        {
            "Image": tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
            "Feature": tf.TensorSpec(shape=(12,), dtype=tf.float32),
        },
    ),
)

gen_test = test_loader.batch(32).prefetch(tf.data.AUTOTUNE)



## === cell 8
preds = model.predict(gen_test)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_56/4218511252.py in <cell line: 0>()
----> 1 preds = model.predict(gen_test)
      2 

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

InvalidArgumentError: {{function_node __wrapped__IteratorGetNext_output_types_2_device_/job:localhost/replica:0/task:0/device:CPU:0}} TypeError: `generator` yielded an element that did not match the expected structure. The expected structure was ({'Image': tf.float32, 'Feature': tf.float32},), but the yielded element was {'Image': array([[[0.13725491, 0.13725491, 0.14509805],
        [0.14901961, 0.14901961, 0.15686275],
        [0.14117648, 0.14117648, 0.14901961],
        ...,
        [0.01960784, 0.01960784, 0.02745098],
        [0.01960784, 0.01960784, 0.02745098],
        [0.01176471, 0.01176471, 0.01960784]],

       [[0.15686275, 0.15686275, 0.16470589],
        [0.13725491, 0.13725491, 0.14509805],
        [0.14901961, 0.14901961, 0.15686275],
        ...,
        [0.01960784, 0.01960784, 0.02745098],
        [0.01960784, 0.01960784, 0.02745098],
        [0.01568628, 0.01568628, 0.02352941]],

       [[0.16470589, 0.16470589, 0.17254902],
        [0.15686275, 0.15686275, 0.16470589],
        [0.14117648, 0.14117648, 0.14901961],
        ...,
        [0.01960784, 0.01960784, 0.02745098],
        [0.01568628, 0.01568628, 0.02352941],
        [0.00784314, 0.00784314, 0.01568628]],

       ...,

       [[0.6117647 , 0.6117647 , 0.61960787],
        [0.6       , 0.6       , 0.60784316],
        [0.54901963, 0.54901963, 0.5568628 ],
        ...,
        [0.06666667, 0.0627451 , 0.08235294],
        [0.04313726, 0.03921569, 0.05882353],
        [0.0627451 , 0.05882353, 0.07843138]],

       [[0.59607846, 0.59607846, 0.6039216 ],
        [0.5921569 , 0.5921569 , 0.6       ],
        [0.53333336, 0.53333336, 0.5411765 ],
        ...,
        [0.0627451 , 0.05882353, 0.07843138],
        [0.0627451 , 0.05882353, 0.07843138],
        [0.05490196, 0.05098039, 0.07058824]],

       [[0.59607846, 0.59607846, 0.6039216 ],
        [0.5921569 , 0.5921569 , 0.6       ],
        [0.4745098 , 0.4745098 , 0.48235294],
        ...,
        [0.04705882, 0.04313726, 0.0627451 ],
        [0.04705882, 0.04313726, 0.0627451 ],
        [0.05490196, 0.05098039, 0.07058824]]], dtype=float32), 'Feature': array([0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)}.
Traceback (most recent call last):

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/from_generator_op.py", line 204, in generator_py_func
    flattened_values = nest.flatten_up_to(output_types, values)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/nest.py", line 237, in flatten_up_to
    return nest_util.flatten_up_to(
           ^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/nest_util.py", line 1541, in flatten_up_to
    return _tf_data_flatten_up_to(shallow_tree, input_tree)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/nest_util.py", line 1570, in _tf_data_flatten_up_to
    _tf_data_assert_shallow_structure(shallow_tree, input_tree)

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/nest_util.py", line 1420, in _tf_data_assert_shallow_structure
    raise TypeError(

TypeError: The two structures don't have the same sequence type. Input structure has type 'dict', while shallow structure has type 'tuple'.


The above exception was the direct cause of the following exception:


Traceback (most recent call last):

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 269, in __call__
    ret = func(*args)
          ^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py", line 643, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/from_generator_op.py", line 206, in generator_py_func
    raise TypeError(

TypeError: `generator` yielded an element that did not match the expected structure. The expected structure was ({'Image': tf.float32, 'Feature': tf.float32},), but the yielded element was {'Image': array([[[0.13725491, 0.13725491, 0.14509805],
        [0.14901961, 0.14901961, 0.15686275],
        [0.14117648, 0.14117648, 0.14901961],
        ...,
        [0.01960784, 0.01960784, 0.02745098],
        [0.01960784, 0.01960784, 0.02745098],
        [0.01176471, 0.01176471, 0.01960784]],

       [[0.15686275, 0.15686275, 0.16470589],
        [0.13725491, 0.13725491, 0.14509805],
        [0.14901961, 0.14901961, 0.15686275],
        ...,
        [0.01960784, 0.01960784, 0.02745098],
        [0.01960784, 0.01960784, 0.02745098],
        [0.01568628, 0.01568628, 0.02352941]],

       [[0.16470589, 0.16470589, 0.17254902],
        [0.15686275, 0.15686275, 0.16470589],
        [0.14117648, 0.14117648, 0.14901961],
        ...,
        [0.01960784, 0.01960784, 0.02745098],
        [0.01568628, 0.01568628, 0.02352941],
        [0.00784314, 0.00784314, 0.01568628]],

       ...,

       [[0.6117647 , 0.6117647 , 0.61960787],
        [0.6       , 0.6       , 0.60784316],
        [0.54901963, 0.54901963, 0.5568628 ],
        ...,
        [0.06666667, 0.0627451 , 0.08235294],
        [0.04313726, 0.03921569, 0.05882353],
        [0.0627451 , 0.05882353, 0.07843138]],

       [[0.59607846, 0.59607846, 0.6039216 ],
        [0.5921569 , 0.5921569 , 0.6       ],
        [0.53333336, 0.53333336, 0.5411765 ],
        ...,
        [0.0627451 , 0.05882353, 0.07843138],
        [0.0627451 , 0.05882353, 0.07843138],
        [0.05490196, 0.05098039, 0.07058824]],

       [[0.59607846, 0.59607846, 0.6039216 ],
        [0.5921569 , 0.5921569 , 0.6       ],
        [0.4745098 , 0.4745098 , 0.48235294],
        ...,
        [0.04705882, 0.04313726, 0.0627451 ],
        [0.04705882, 0.04313726, 0.0627451 ],
        [0.05490196, 0.05098039, 0.07058824]]], dtype=float32), 'Feature': array([0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)}.


	 [[{{node PyFunc}}]] [Op:IteratorGetNext] name: 

## === cell 9
ser = pd.Series(preds.squeeze(), name="Pawpularity")

id_ = pd.Series(
    sorted(
        [i[:-4] for i in os.listdir("/kaggle/input/petfinder-pawpularity-score/test/")]
    ),
    name="Id",
)

ans = pd.DataFrame(id_).join(ser)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/218744874.py in <cell line: 0>()
----> 1 ser = pd.Series(preds.squeeze(), name="Pawpularity")
      2 
      3 id_ = pd.Series(
      4     sorted(
      5         [i[:-4] for i in os.listdir("/kaggle/input/petfinder-pawpularity-score/test/")]

NameError: name 'preds' is not defined

## === cell 10
ans.to_csv("submission.csv", index=False)

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/899919599.py in <cell line: 0>()
----> 1 ans.to_csv("submission.csv", index=False)

NameError: name 'ans' is not defined
