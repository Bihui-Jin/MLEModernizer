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
Given simulated manufacturing control data, predict whether the machine is in state `0` or state `1`.

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability for the `target` variable. The file should contain a header and have the following format:

```
id,target
900000,0.65
900001,0.97
900002,0.02
etc.
```

## Dataset
- **train.csv** - the training data, which includes normalized continuous data and categorical data
- **test.csv** - the test set; your task is to predict binary `target` variable which represents the state of a manufacturing process
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.10

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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
            description.md (79 lines)
            sample_submission.csv (100001 lines)
            sample_submission.csv.zip (224.9 kB)
            test.csv (100001 lines)
            test.csv.zip (16.6 MB)
            train.csv (800001 lines)
            train.csv.zip (133.3 MB)
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
        input/
            description.md (79 lines)
            sample_submission.csv (100001 lines)
            sample_submission.csv.zip (224.9 kB)
            test.csv (100001 lines)
            test.csv.zip (16.6 MB)
            train.csv (800001 lines)
            train.csv.zip (133.3 MB)
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
        working/
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
```

-> data/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> data/tabular-playground-series-may-2022/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> data/tabular-playground-series-may-2022/test.csv has 100000 rows and 32 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 17 more columns

-> data/tabular-playground-series-may-2022/train.csv has 800000 rows and 33 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 18 more columns

-> data/test.csv has 100000 rows and 32 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 17 more columns

-> data/train.csv has 800000 rows and 33 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 18 more columns

-> input/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> (stopped after 10 files for performance)

# 5. Target score

0.99648

# 6. Current score

0.49242

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.49242) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation before importing TensorFlow, which resolves the `MessageFactory.GetPrototype` error in this environment. I also fix the correlation calls to use only numeric columns so the EDA cells don’t crash on the string `f_27` column. The model definition be corrected to use a valid Keras input shape `Input(shape=(41,))` (same architecture otherwise), enabling training and prediction to run. Finally, I ensure the submission is written with the required `.csv` suffix and correct filename (`submission.csv`) so Kaggle accepts it.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
df_train = pd.read_csv("../input/tabular-playground-series-may-2022/train.csv")
df_train.head()



## === cell 2
df_train.select_dtypes(include=[np.number]).corr().tail(1)



## === cell 3
num_corr = df_train.select_dtypes(include=[np.number]).corr()
f = plt.figure(figsize=(19, 15))
plt.matshow(num_corr, fignum=f.number)
plt.xticks(range(num_corr.shape[1]), num_corr.columns, fontsize=14, rotation=45)
plt.yticks(range(num_corr.shape[0]), num_corr.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title("Correlation Matrix", fontsize=16)



## === cell 4
df_train.info()



## === cell 5
df_test = pd.read_csv("../input/tabular-playground-series-may-2022/test.csv")



## === cell 6
for df in [df_train, df_test]:
    for i in range(10):
        df[f"ch{i}"] = df.f_27.str.get(i).apply(ord) - ord("A")

    df["unique_characters"] = df.f_27.apply(lambda s: len(set(s)))

features = [f for f in df_test.columns if f != "id" and f != "f_27"]



## === cell 7
X_train = df_train.drop(["target"], axis=1)[features]
Y_train = df_train["target"].to_numpy()
X_test = df_test[features].copy()



## === cell 8
X_test.head()



## === cell 9
StSc = StandardScaler()
X_train = StSc.fit_transform(X_train)
X_test = StSc.transform(X_test)



## === cell 10
print(X_train[0:5])
print(Y_train[0:5])



## === cell 11
X_train.shape



## === cell 12
X_test.shape



## === cell 13
L2 = 0.000003
model_class = tf.keras.models.Sequential(
    [
        tf.keras.layers.Input(shape=(41,)),
        tf.keras.layers.Dense(
            82, kernel_regularizer=tf.keras.regularizers.l2(L2), activation="swish"
        ),
        tf.keras.layers.Dense(
            82, kernel_regularizer=tf.keras.regularizers.l2(L2), activation="swish"
        ),
        tf.keras.layers.Dense(
            82, kernel_regularizer=tf.keras.regularizers.l2(L2), activation="swish"
        ),
        tf.keras.layers.Dense(
            41, kernel_regularizer=tf.keras.regularizers.l2(L2), activation="swish"
        ),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)



## === cell 14
loss = tf.keras.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam()
model_class.compile(
    optimizer=opt,
    loss=loss,
    metrics=[
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
    ],
)



## === cell 15
earlystopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=6,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)

LR = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=2, verbose=0, mode="auto"
)



## === cell 16
history = model_class.fit(
    x=X_train,
    y=Y_train,
    batch_size=500,
    epochs=300,
    verbose=1,
    callbacks=[LR, earlystopping],
    validation_split=0.1,
    validation_data=None,
    shuffle=False,
    class_weight=None,
    sample_weight=None,
    initial_epoch=0,
    steps_per_epoch=1800,
    validation_steps=None,
    validation_batch_size=None,
    validation_freq=1,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False,
)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3768808381.py in <cell line: 0>()
----> 1 history = model_class.fit(
      2     x=X_train,
      3     y=Y_train,
      4     batch_size=500,
      5     epochs=300,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    117             return fn(*args, **kwargs)
    118         except Exception as e:
--> 119             filtered_tb = _process_traceback_frames(e.__traceback__)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`

TypeError: TensorFlowTrainer.fit() got an unexpected keyword argument 'max_queue_size'

## === cell 17
acc_train = history.history["binary_accuracy"]
acc_val = history.history["val_binary_accuracy"]

epochs = range(len(acc_train))
plt.plot(epochs, acc_train, "r", label="Training accuracy")
plt.plot(epochs, acc_val, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend(loc=0)
plt.figure()
plt.show()



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2338263733.py in <cell line: 0>()
----> 1 acc_train = history.history["binary_accuracy"]
      2 acc_val = history.history["val_binary_accuracy"]
      3 
      4 epochs = range(len(acc_train))
      5 plt.plot(epochs, acc_train, "r", label="Training accuracy")

NameError: name 'history' is not defined

## === cell 18
pred = model_class.predict(X_test, verbose=0).reshape(-1)
df_test["target"] = pred
submit = df_test[["id", "target"]]

submit.to_csv("submission.csv", index=False)

print(submit.head())
print("Wrote submission.csv with shape:", submit.shape)
