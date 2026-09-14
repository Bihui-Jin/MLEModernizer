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
scipy==1.15.3
seaborn==0.12.2
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

0.9969

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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train = pd.read_csv("../input/tabular-playground-series-may-2022/train.csv")
test = pd.read_csv("../input/tabular-playground-series-may-2022/test.csv")
sample_submission = pd.read_csv(
    "../input/tabular-playground-series-may-2022/sample_submission.csv"
)



## === cell 2
for df in [train, test]:
    for i in range(10):
        df[f"ch{i}"] = df.f_27.str.get(i).apply(ord) - ord("A")
    df["unique_characters"] = df.f_27.apply(lambda s: len(set(s)))
features = [f for f in test.columns if f != "id" and f != "f_27"]
test[features].head(2)



## === cell 3
train.f_00[0:100].plot()
train.f_00[0:100].ewm(com=0.8, min_periods=1).mean().fillna(0).plot()



## === cell 4
from sklearn.preprocessing import StandardScaler



## === cell 5
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Running on TPU ", tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)



## === cell 6
stat = ["mean", "std", "sum", "var", "euc", "ptp", "logsumexp"]



## === cell 7
cols = train.dtypes[train.dtypes == float].index.to_list()


def new_feats(df):
    df["mean"] = df[cols].mean(axis=1).values
    df["std"] = df[cols].std(axis=1).values
    df["sum"] = df[cols].sum(axis=1).values
    df["var"] = df[cols].var(axis=1).values
    df["euc"] = tf.math.reduce_euclidean_norm(df[cols], axis=1).numpy()
    df["ptp"] = df[cols].values.ptp(axis=1)
    df["logsumexp"] = tf.math.reduce_logsumexp(df[cols], axis=1).numpy()
    for col in cols:
        df[f"roll_{col}"] = df[col].ewm(com=0.8, min_periods=1).mean().fillna(0)


new_feats(train)
new_feats(test)



## === cell 8
rolls = [f for f in test.columns if f[0:4] == "roll"]



## === cell 9
scale_cols = [f for f in test.columns if train[f].dtype == float]



## === cell 10
sc = StandardScaler()
sc.fit(train[scale_cols].values)
scaled_st_train = sc.transform(train[scale_cols].values)
scaled_st_test = sc.transform(test[scale_cols].values)
train[scale_cols] = scaled_st_train
test[scale_cols] = scaled_st_test



## === cell 11
train[features + stat + rolls].values.reshape(8, 8, train.shape[0], 1).shape



## === cell 12
traincon = train[features + stat + rolls].values.reshape(train.shape[0], 8, 8)
testcon = test[features + stat + rolls].values.reshape(test.shape[0], 8, 8)



## === cell 13
plt.imshow(traincon[0])



## === cell 14
labels = train.target



## === cell 15
import random



## === cell 16
total_sample = 20
randInt = np.array(
    [random.choice(labels[labels == 1].index) for x in range(int(total_sample / 2))]
    + [
        random.choice(labels[labels == 0].index)
        for x in range(int(total_sample / 2) + 1)
    ]
)
fig, axs = plt.subplots(nrows=5, ncols=4, figsize=(20, 20))
plt.subplots_adjust(wspace=-0.2, hspace=0.6)
for i, ax in enumerate(axs.flat):
    pcm = ax.imshow(traincon[randInt[i]].squeeze(), cmap=plt.cm.magma_r)
    if labels[i] == 0:
        ax.set_title(f"loc:{randInt[i]} state :{labels[i]}", fontdict={"color": "blue"})
    else:
        ax.set_title(
            f"loc:{randInt[i]} state :{labels[i]}", fontdict={"color": "green"}
        )
    ax.set_xlabel("features", fontdict={"color": "green"})
    ax.set_ylabel("features", fontdict={"color": "green"})
    fig.colorbar(pcm, ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()



## === cell 17
train_cols = test.columns.to_list()
train_cols.remove("id")



## === cell 18
len(features + stat + rolls)



## === cell 19
import warnings

warnings.filterwarnings("ignore")



## === cell 20
from sklearn.metrics import roc_curve


def plot_loss_auc(history, y_true, prediction):
    """
    history: history = model.fit()
    y_true: true validation set or test set labels
    prediction: prediction probabilities on val set or test set
    """
    fp, tp, _ = roc_curve(y_true, prediction)
    _, ax = plt.subplots(ncols=4, nrows=1, figsize=(20, 3))
    ax[1].set_xlabel("epochs")
    ax[1].set_ylabel("loss")
    ax[1].set_title("final val_loss %1.4f" % (history.history["val_loss"][-1:][0]))
    ax[2].set_xlabel("epochs")
    ax[2].set_ylabel("auc")
    ax[2].set_title("final val_auc %1.4f" % (history.history["val_auc"][-1:][0]))
    ax[0].set_xlabel("learning rate")
    ax[0].set_ylabel("loss")
    ax[0].set_title("semilogx lr vs loss")
    ax[3].plot(fp, tp, label="ROC", linewidth=2)
    ax[3].vlines(x=0, ymin=0.0, ymax=1.0, linewidth=0.5, color="r", linestyles="--")
    ax[3].hlines(y=1, xmin=0.0, xmax=1.0, linewidth=0.5, color="r", linestyles="--")
    ax[3].set_xlabel("False positives")
    ax[3].set_ylabel("True positives")
    ax[3].set_title("ROC")
    if "lr" in history.history:
        ax[0].semilogx(history.history["lr"], history.history["loss"])
    ax[0].set_ylim(ymax=0.11)
    pd.DataFrame(
        [history.history["auc"], history.history["val_auc"]], index=["auc", "val_auc"]
    ).T.plot(ax=ax[2])
    pd.DataFrame(
        [history.history["loss"], history.history["val_loss"]],
        index=["loss", "val_loss"],
    ).T.plot(ax=ax[1])
    plt.show()




## === cell 21
from sklearn.metrics import confusion_matrix, classification_report


def plot_cm(y_true, prediction, p=0.5):
    """
    y_true: true validation set or test set labels
    prediction: prediction probabilities on val set or test set
    """
    cm = confusion_matrix(y_true, prediction > p)
    plt.figure(figsize=(3, 3))
    sns.heatmap(cm, annot=True, fmt="d", cbar=False)
    plt.title("Confusion matrix @{:.2f}".format(p))
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")

    print("\nState 0 Detected (True Negatives): ", cm[0][0])
    print("State 1 Incorrectly Detected (False Positives): ", cm[0][1])
    print("State 1 Missed (False Negatives): ", cm[1][0])
    print("State 1 Detected (True Positives): ", cm[1][1])
    print("Total States : ", np.sum(cm[1]))
    plt.show()




## === cell 22
class reps(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        def _fmt(v):
            return "NA" if v is None else f"{v:.4f}"

        print(
            f"epoch : {epoch}, "
            f"loss : {_fmt(logs.get('loss'))}, "
            f"val loss : {_fmt(logs.get('val_loss'))}, "
            f"auc : {_fmt(logs.get('auc'))}, "
            f"val auc : {_fmt(logs.get('val_auc'))}"
        )


report_callback = reps()



## === cell 23
epochs = 50
exp = -0.05
lr = 0.001  # lr at begin
start = 10  # epoch
mid = 17  # epoch


def lr_decay(epoch, lr):
    lr_in = float(lr)
    if epoch < start:
        out = lr_in
    elif epoch < mid:
        out = lr_in * float((1 + np.cos(epoch / (epochs - 1.5) * np.pi)) / 1.8)
    else:
        out = lr_in * float(np.exp(exp))
    return float(out)


def plot_lr_decay(epochs, lr):
    x = np.arange(0, epochs)
    lrs = []
    lr2 = lr
    for epoch in x:
        lr = lr_decay(epoch, lr)
        lrs.append(lr)
    y = np.array(lrs)
    plt.figure(figsize=(8, 4))
    plt.plot(x, y)
    plt.vlines(
        x=start - 1, linestyles="--", colors="g", ymin=y[-1], ymax=lr2, linewidth=0.95
    )
    plt.vlines(
        x=mid - 1,
        linestyles="--",
        colors="orange",
        ymin=y[-1],
        ymax=lr2,
        linewidth=0.95,
    )
    plt.vlines(x=35, linestyles="--", colors="r", ymin=y[-1], ymax=lr2, linewidth=0.95)
    plt.hlines(
        y=0, linestyles="--", colors="r", xmin=start - 1, xmax=50, linewidth=0.95
    )
    plt.xlabel("epochs")
    plt.ylabel("learning rate")
    plt.title("learning rate decay")


plot_lr_decay(epochs, lr)
lrDecay = tf.keras.callbacks.LearningRateScheduler(lr_decay)
callbacks = [lrDecay, report_callback]



## === cell 24
from tensorflow.keras.layers import (
    Conv1D,
    Conv2D,
    Flatten,
    Dropout,
    Conv1DTranspose,
    Conv2DTranspose,
    Dense,
    Reshape,
    GlobalAveragePooling1D,
)
from tensorflow.keras import Input
from tensorflow import keras



## === cell 25
lr = 0.001
input_shape = (8, 8)


def cnn2():
    model = keras.Sequential(
        [
            Input(shape=input_shape),
            Reshape(target_shape=(8, 8, 1)),
            Conv2D(
                filters=144,
                kernel_size=(7, 7),
                padding="same",
                strides=2,
                activation="relu",
            ),
            Dropout(rate=0.1),
            Reshape(target_shape=(4 * 12, 4 * 12)),
            Conv1D(
                filters=64, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            Conv1DTranspose(
                filters=36, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            Conv1DTranspose(
                filters=64, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            Dropout(rate=0.1),
            Conv1DTranspose(
                filters=128, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            Flatten(),
            Dropout(rate=0.4),
            Dense(1, activation="sigmoid"),
        ]
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model




## === cell 26
lr = 0.001
input_shape = (8, 8)


def cnn():
    model = keras.Sequential(
        [
            Input(shape=input_shape),
            Conv1D(
                filters=128, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            Dropout(rate=0.2),
            Conv1D(
                filters=64, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            Conv1DTranspose(
                filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            Conv1DTranspose(
                filters=64, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            Dropout(rate=0.2),
            Conv1DTranspose(
                filters=128, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            Flatten(),
            Dense(1, activation="sigmoid"),
        ]
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model




## === cell 27
model = cnn2()
model.summary()



## === cell 28
import scipy.stats
from sklearn.model_selection import KFold, train_test_split



## === cell 29
epochs = 45
batch_size = 1024
pred_list = []
cv = 2  # min value 2 for fold split
verbose = 0
kf = KFold(n_splits=cv)
fold_split = False  # {True: KFold split | False: train_test_split}
if fold_split:
    for fold, (split_train, split_test) in enumerate(kf.split(train)):
        print("\n", "*=" * 10 + "*", f"fold {fold+1}", "*=" * 10 + "*", "\n")
        X_train = traincon[split_train]
        X_test = traincon[split_test]
        y_train = labels.iloc[split_train].values
        y_test = labels.iloc[split_test].values

        with strategy.scope():
            model = cnn2()
            history = model.fit(
                X_train,
                y_train,
                batch_size=batch_size,
                epochs=epochs,
                callbacks=callbacks,
                validation_data=(X_test, y_test),
                verbose=verbose,
                shuffle=True,
                steps_per_epoch=X_train.shape[0] // batch_size,
            )

        test_pred = model.predict(testcon, batch_size=batch_size, verbose=0).reshape(-1)
        pred_list.append(scipy.stats.rankdata(test_pred))

        val_pred = model.predict(X_test, batch_size=batch_size, verbose=0).reshape(-1)
        plot_loss_auc(history, y_test, val_pred)
        plot_cm(y_test, val_pred, p=0.5)

else:  # train_test_split
    random_states = np.linspace(789, 9876, cv).astype(int)
    print("random_states:", random_states)
    for fold, random_state in zip(range(cv), random_states):
        print("\n", "*=" * 10 + "*", f"fold {fold+1}", "*=" * 10 + "*", "\n")
        X_train, X_test, y_train, y_test = train_test_split(
            traincon, labels.values, test_size=0.06, random_state=random_state
        )
        with strategy.scope():
            model = cnn2()
            history = model.fit(
                X_train,
                y_train,
                batch_size=batch_size,
                epochs=45,
                callbacks=callbacks,
                validation_data=(X_test, y_test),
                verbose=verbose,
                shuffle=True,
                steps_per_epoch=X_train.shape[0] // batch_size,
            )

        test_pred = model.predict(testcon, batch_size=batch_size, verbose=0).reshape(-1)
        pred_list.append(scipy.stats.rankdata(test_pred))

        val_pred = model.predict(X_test, batch_size=batch_size, verbose=0).reshape(-1)
        plot_loss_auc(history, y_test, val_pred)
        plot_cm(y_test, val_pred, p=0.5)



## === cell 30
mean_cnn = np.mean(np.vstack(pred_list), axis=0).reshape(-1)



## === cell 31
sample_submission["target"] = mean_cnn
sample_submission.to_csv("submission.csv", index=False)
sample_submission.head()

## --- ERROR in outputing the csv:
Invalid submission: Submission target column should contain probabilities, and therefore contain values between 0 and 1 inclusive
