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

22.05008

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re




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
        img = np.array(img).astype(np.float32) / 255.0

        metadata = metadata_csv[metadata_csv["Id"] == i[:-4]]
        features = metadata[relevant_columns].values[0].astype(np.float32)

        y = float(metadata[target].values[0])

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

    image_names = os.listdir(IMAGES_PATH)
    metadata_csv = pd.read_csv(CSV_PATH)

    for i in image_names:
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
test_loader = tf.data.Dataset.from_generator(
    pythonic_loader_test,
    output_signature=(
        {
            "Image": tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
            "Feature": tf.TensorSpec(shape=(12,), dtype=tf.float32),
        },
    ),
)



## === cell 5
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
    inputs={"Image": input_image, "Feature": input_meta}, outputs={"Label": output}
)



## === cell 6
model.summary()



## === cell 7
initial_learning_rate = 1e-3
first_decay_steps = 200
lr_warmup_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=initial_learning_rate, decay_steps=first_decay_steps
)



## === cell 8
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_warmup_decayed_fn),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.RootMeanSquaredError()],
)



## === cell 9
gen_train = train_loader.batch(32).prefetch(tf.data.AUTOTUNE).repeat()
gen_test = test_loader.batch(32).prefetch(tf.data.AUTOTUNE)



## === cell 10
print(tf.config.list_logical_devices("GPU"))



## === cell 11
steps_per_epoch = int(
    len(os.listdir("/kaggle/input/petfinder-pawpularity-score/train/")) * 0.95 // 32
)
model.fit(
    gen_train,
    steps_per_epoch=steps_per_epoch,
    epochs=3,
    validation_data=gen_test,
    validation_steps=1,
)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_55/121878190.py in <cell line: 0>()
      2     len(os.listdir("/kaggle/input/petfinder-pawpularity-score/train/")) * 0.95 // 32
      3 )
----> 4 model.fit(
      5     gen_train,
      6     steps_per_epoch=steps_per_epoch,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/compile_utils.py in <listcomp>(.0)
    243                     MetricsList(
    244                         [
--> 245                             get_metric(m, y_true[0], y_pred[0])
    246                             for m in metrics
    247                             if m is not None

IndexError: list index out of range

## === cell 12
test_csv = pd.read_csv("/kaggle/input/petfinder-pawpularity-score/test.csv")
test_ids = test_csv["Id"].tolist()

ordered_test_loader = tf.data.Dataset.from_tensor_slices(test_ids).map(
    lambda id_str: tf.py_function(
        func=lambda x: pythonic_loader_test_single(x.numpy().decode()),
        inp=[id_str],
        Tout={"Image": tf.float32, "Feature": tf.float32},
    ),
    num_parallel_calls=tf.data.AUTOTUNE,
)


def pythonic_loader_test_single(id_str):
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
    img = tf.keras.utils.load_img(
        path=os.path.join(IMAGES_PATH, f"{id_str}.jpg"),
        color_mode="rgb",
        target_size=(256, 256),
    )
    img = np.array(img).astype(np.float32) / 255.0
    metadata = pd.read_csv(CSV_PATH)
    meta_row = metadata[metadata["Id"] == id_str]
    features = meta_row[relevant_columns].values[0].astype(np.float32)
    return {"Image": img, "Feature": features}


preds_dict = model.predict(ordered_test_loader.batch(1), verbose=0)
preds = preds_dict["Label"].squeeze()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1211349270.py in <cell line: 0>()
      4 
      5 # Build a dataset that yields examples in the same order as test_csv
----> 6 ordered_test_loader = tf.data.Dataset.from_tensor_slices(test_ids).map(
      7     lambda id_str: tf.py_function(
      8         func=lambda x: pythonic_loader_test_single(x.numpy().decode()),

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in map(self, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
   2339     from tensorflow.python.data.ops import map_op
   2340 
-> 2341     return map_op._map_v2(
   2342         self,
   2343         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in _map_v2(input_dataset, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
     55           num_parallel_calls,
     56       )
---> 57     return _ParallelMapDataset(
     58         input_dataset,
     59         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in __init__(self, input_dataset, map_func, num_parallel_calls, deterministic, use_inter_op_parallelism, preserve_cardinality, use_legacy_function, use_unbounded_threadpool, name)
    200     self._input_dataset = input_dataset
    201     self._use_inter_op_parallelism = use_inter_op_parallelism
--> 202     self._map_func = structured_function.StructuredFunctionWrapper(
    203         map_func,
    204         self._transformation_name(),

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in __init__(self, func, transformation_name, dataset, input_classes, input_shapes, input_types, input_structure, add_to_graph, use_legacy_function, defun_kwargs)
    263         fn_factory = trace_tf_function(defun_kwargs)
    264 
--> 265     self._function = fn_factory()
    266     # There is no graph to add in eager mode.
    267     add_to_graph &= not context.executing_eagerly()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in get_concrete_function(self, *args, **kwargs)
   1249   def get_concrete_function(self, *args, **kwargs):
   1250     # Implements PolymorphicFunction.get_concrete_function.
-> 1251     concrete = self._get_concrete_function_garbage_collected(*args, **kwargs)
   1252     concrete._garbage_collector.release()  # pylint: disable=protected-access
   1253     return concrete

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _get_concrete_function_garbage_collected(self, *args, **kwargs)
   1219       if self._variable_creation_config is None:
   1220         initializers = []
-> 1221         self._initialize(args, kwargs, add_initializers_to=initializers)
   1222         self._initialize_uninitialized_variables(initializers)
   1223 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _initialize(self, args, kwds, add_initializers_to)
    694     )
    695     # Force the definition of the function for these arguments
--> 696     self._concrete_variable_creation_fn = tracing_compilation.trace_function(
    697         args, kwds, self._variable_creation_config
    698     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in trace_function(args, kwargs, tracing_options)
    176       kwargs = {}
    177 
--> 178     concrete_function = _maybe_define_function(
    179         args, kwargs, tracing_options
    180     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _maybe_define_function(args, kwargs, tracing_options)
    281         else:
    282           target_func_type = lookup_func_type
--> 283         concrete_function = _create_concrete_function(
    284             target_func_type, lookup_func_context, func_graph, tracing_options
    285         )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _create_concrete_function(function_type, type_context, func_graph, tracing_options)
    308       attributes_lib.DISABLE_ACD, False
    309   )
--> 310   traced_func_graph = func_graph_module.func_graph_from_py_func(
    311       tracing_options.name,
    312       tracing_options.python_function,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/func_graph.py in func_graph_from_py_func(name, python_func, args, kwargs, signature, func_graph, add_control_dependencies, arg_names, op_return_value, collections, capture_by_value, create_placeholders)
   1057 
   1058     _, original_func = tf_decorator.unwrap(python_func)
-> 1059     func_outputs = python_func(*func_args, **func_kwargs)
   1060 
   1061     # invariant: `func_outputs` contains only Tensors, CompositeTensors,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in wrapped_fn(*args, **kwds)
    597         # the function a weak reference to itself to avoid a reference cycle.
    598         with OptionalXlaContext(compile_with_xla):
--> 599           out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    600         return out
    601 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapped_fn(*args)
    229       # Note: wrapper_helper will apply autograph based on context.
    230       def wrapped_fn(*args):  # pylint: disable=missing-docstring
--> 231         ret = wrapper_helper(*args)
    232         ret = structure.to_tensor_list(self._output_structure, ret)
    233         return [ops.convert_to_tensor(t) for t in ret]

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapper_helper(*args)
    159       if not _should_unpack(nested_args):
    160         nested_args = (nested_args,)
--> 161       ret = autograph.tf_convert(self._func, ag_ctx)(*nested_args)
    162       ret = variable_utils.convert_variables_to_tensors(ret)
    163       if _should_pack(ret):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in wrapper(*args, **kwargs)
    691       except Exception as e:  # pylint:disable=broad-except
    692         if hasattr(e, 'ag_error_metadata'):
--> 693           raise e.ag_error_metadata.to_exception(e)
    694         else:
    695           raise

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in wrapper(*args, **kwargs)
    688       try:
    689         with conversion_ctx:
--> 690           return converted_call(f, args, kwargs, options=options)
    691       except Exception as e:  # pylint:disable=broad-except
    692         if hasattr(e, 'ag_error_metadata'):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    437     try:
    438       if kwargs is not None:
--> 439         result = converted_f(*effective_args, **kwargs)
    440       else:
    441         result = converted_f(*effective_args)

/tmp/__autograph_generated_filew_bps0jm.py in <lambda>(id_str)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda id_str: ag__.with_function_scope(lambda lscope: ag__.converted_call(tf.py_function, (), dict(func=ag__.autograph_artifact(lambda x: ag__.converted_call(pythonic_loader_test_single, (ag__.converted_call(ag__.converted_call(x.numpy, (), None, lscope).decode, (), None, lscope),), None, lscope)), inp=[id_str], Tout={'Image': tf.float32, 'Feature': tf.float32}), lscope), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/function_wrappers.py in with_function_scope(thunk, scope_name, options)
    111   """Inline version of the FunctionScope context manager."""
    112   with FunctionScope('lambda_', scope_name, options) as scope:
--> 113     return thunk(scope)

/tmp/__autograph_generated_filew_bps0jm.py in <lambda>(lscope)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda id_str: ag__.with_function_scope(lambda lscope: ag__.converted_call(tf.py_function, (), dict(func=ag__.autograph_artifact(lambda x: ag__.converted_call(pythonic_loader_test_single, (ag__.converted_call(ag__.converted_call(x.numpy, (), None, lscope).decode, (), None, lscope),), None, lscope)), inp=[id_str], Tout={'Image': tf.float32, 'Feature': tf.float32}), lscope), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    375 
    376   if not options.user_requested and conversion.is_allowlisted(f):
--> 377     return _call_unconverted(f, args, kwargs, options)
    378 
    379   # internal_convert_user_code is for example turned off when issuing a dynamic

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in _call_unconverted(f, args, kwargs, options, update_cache)
    457 
    458   if kwargs is not None:
--> 459     return f(*args, **kwargs)
    460   return f(*args)
    461 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/dtypes.py in as_dtype(type_value)
    850     return _INTERN_TABLE[type_value.as_datatype_enum]
    851 
--> 852   raise TypeError(f"Cannot convert the argument `type_value`: {type_value!r} "
    853                   "to a TensorFlow DType.")

TypeError: in user code:

    File "/tmp/ipykernel_55/1211349270.py", line 7, in None  *
        )

    TypeError: Cannot convert the argument `type_value`: {'Image': tf.float32, 'Feature': tf.float32} to a TensorFlow DType.


## === cell 13
submission = pd.DataFrame({"Id": test_ids, "Pawpularity": preds})
submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1037704573.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"Id": test_ids, "Pawpularity": preds})
      2 submission.to_csv("submission.csv", index=False)

NameError: name 'preds' is not defined
