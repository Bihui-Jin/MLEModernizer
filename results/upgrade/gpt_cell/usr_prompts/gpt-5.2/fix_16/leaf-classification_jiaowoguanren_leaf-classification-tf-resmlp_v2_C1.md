# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.10

# 2. Installed packages

category_encoders==2.7.0
geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
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
tf_keras==2.18.0

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (70 lines)
            images.zip (22.0 MB)
            sample_submission.csv (100 lines)
            sample_submission.csv.zip (2.3 kB)
            test.csv (100 lines)
            test.csv.zip (39.3 kB)
            train.csv (892 lines)
            train.csv.zip (357.1 kB)
            images/
                42.jpg (32.6 kB)
                168.jpg (16.5 kB)
                ... and 988 other files
            leaf-classification/
                description.md (70 lines)
                images.zip (22.0 MB)
                ... and 6 other files
                images/
                    42.jpg (32.6 kB)
                    168.jpg (16.5 kB)
                    ... and 988 other files
                leaf-classification/
        input/
            description.md (70 lines)
            images.zip (22.0 MB)
            sample_submission.csv (100 lines)
            sample_submission.csv.zip (2.3 kB)
            test.csv (100 lines)
            test.csv.zip (39.3 kB)
            train.csv (892 lines)
            train.csv.zip (357.1 kB)
            images/
                42.jpg (32.6 kB)
                168.jpg (16.5 kB)
                ... and 988 other files
            leaf-classification/
                description.md (70 lines)
                images.zip (22.0 MB)
                ... and 6 other files
                images/
                    42.jpg (32.6 kB)
                    168.jpg (16.5 kB)
                    ... and 988 other files
                leaf-classification/
        working/
            leaf-classification/
                description.md (70 lines)
                images.zip (22.0 MB)
                ... and 6 other files
                images/
                    42.jpg (32.6 kB)
                    168.jpg (16.5 kB)
                    ... and 988 other files
                leaf-classification/
```

-> data/leaf-classification/sample_submission.csv has 99 rows and 100 columns.
The columns are: id, Acer_Capillipes, Acer_Circinatum, Acer_Mono, Acer_Opalus, Acer_Palmatum, Acer_Pictum, Acer_Platanoids, Acer_Rubrum, Acer_Rufinerve, Acer_Saccharinum, Alnus_Cordata, Alnus_Maximowiczii, Alnus_Rubra, Alnus_Sieboldiana... and 85 more columns

-> data/leaf-classification/test.csv has 99 rows and 193 columns.
The columns are: id, margin1, margin2, margin3, margin4, margin5, margin6, margin7, margin8, margin9, margin10, margin11, margin12, margin13, margin14... and 178 more columns

-> data/leaf-classification/train.csv has 891 rows and 194 columns.
The columns are: id, species, margin1, margin2, margin3, margin4, margin5, margin6, margin7, margin8, margin9, margin10, margin11, margin12, margin13... and 179 more columns

-> data/sample_submission.csv has 99 rows and 100 columns.
The columns are: id, Acer_Capillipes, Acer_Circinatum, Acer_Mono, Acer_Opalus, Acer_Palmatum, Acer_Pictum, Acer_Platanoids, Acer_Rubrum, Acer_Rufinerve, Acer_Saccharinum, Alnus_Cordata, Alnus_Maximowiczii, Alnus_Rubra, Alnus_Sieboldiana... and 85 more columns

-> data/test.csv has 99 rows and 193 columns.
The columns are: id, margin1, margin2, margin3, margin4, margin5, margin6, margin7, margin8, margin9, margin10, margin11, margin12, margin13, margin14... and 178 more columns

-> data/train.csv has 891 rows and 194 columns.
The columns are: id, species, margin1, margin2, margin3, margin4, margin5, margin6, margin7, margin8, margin9, margin10, margin11, margin12, margin13... and 179 more columns

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import os
import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<5"])

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import tensorflow as tf
import numpy as np
import pandas as pd
from keras.layers import Dense, BatchNormalization, Dropout
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from category_encoders import OneHotEncoder
import matplotlib.pyplot as plt
import os, math
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    classification_report,
    confusion_matrix,
)



## === cell 2
dfTrain = pd.read_csv("/kaggle/input/leaf-classification/train.csv.zip")
dfTrain = dfTrain.iloc[:, 1:]

dfTest = pd.read_csv("/kaggle/input/leaf-classification/test.csv.zip")
dfTest = dfTest.iloc[:, 1:]



## === cell 3
dfTrain



## === cell 4
dfTest



## === cell 5
len(dfTrain.species.unique())



## === cell 6
dfSub = pd.read_csv("/kaggle/input/leaf-classification/sample_submission.csv.zip")
submission_classes = dfSub.columns.tolist()[1:]


def create_data(dfTrain, class_order):
    target = {k: v for v, k in enumerate(class_order)}
    X = np.array(dfTrain.drop(["species"], axis=1), dtype=np.float32)
    y = np.asarray(dfTrain["species"].map(target)).astype("int64")

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    X = tf.convert_to_tensor(X, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.int64)
    return X, y, scaler


X, y, scaler = create_data(dfTrain, submission_classes)



## === cell 7
print(X.shape, y.shape)




## === cell 8
def GELU(x):
    res = 0.5 * x * (1 + tf.nn.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * (x**3))))
    return res


class ResMLPBlock(tf.keras.layers.Layer):
    def __init__(self, units, residual_path):
        super(ResMLPBlock, self).__init__()
        self.residual_path = residual_path
        self.D1 = Dense(units, activation="relu")
        self.D2 = Dense(units, activation="relu")

        if self.residual_path:
            self.D3 = Dense(units)
            self.D4 = Dense(units)

    def call(self, inputs):
        residual = inputs

        x = self.D1(inputs)
        y = self.D2(x)

        if self.residual_path:
            residual = self.D3(inputs)
            residual = GELU(residual)
            residual = self.D4(residual)
            residual = GELU(residual)

        output = y + residual
        return output




## === cell 9
class ResMLP(tf.keras.Model):
    def __init__(self, initial_filters, block_list, num_classes):
        super(ResMLP, self).__init__()
        self.initial_filters = initial_filters
        self.block_list = block_list

        self.D1 = Dense(self.initial_filters, activation="relu")
        self.B1 = BatchNormalization()

        self.blocks = tf.keras.models.Sequential()
        for block_id in range(len(block_list)):
            for layer_id in range(block_list[block_id]):
                if block_id != 0 and layer_id == 0:
                    block = ResMLPBlock(units=self.initial_filters, residual_path=True)
                else:
                    block = ResMLPBlock(units=self.initial_filters, residual_path=False)
                self.blocks.add(block)
            self.initial_filters *= 2
        self.D2 = Dense(num_classes, activation="softmax")

    def call(self, inputs):
        x = self.D1(inputs)
        x = self.B1(x)
        x = self.blocks(x)
        y = self.D2(x)
        return y




## === cell 10
net = ResMLP(initial_filters=128, block_list=[2, 2, 2], num_classes=99)

net.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["sparse_categorical_accuracy"],
)

history = net.fit(X, y, epochs=50, batch_size=32, validation_split=0.3)

net.summary()



## === cell 11
dfSub



## === cell 12
X_test = np.array(dfTest.drop(columns=["id"]), dtype=np.float32)
X_test = scaler.transform(X_test).astype(np.float32)

y_pred = net.predict(X_test, verbose=0)



## --- ERROR in cell 12, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mKeyError[0m                                  Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2911316700.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[0;32m----> 1[0;31m [0mX_test[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0marray[0m[0;34m([0m[0mdfTest[0m[0;34m.[0m[0mdrop[0m[0;34m([0m[0mcolumns[0m[0;34m=[0m[0;34m[[0m[0;34m"id"[0m[0;34m][0m[0;34m)[0m[0;34m,[0m [0mdtype[0m[0;34m=[0m[0mnp[0m[0;34m.[0m[0mfloat32[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      2[0m [0mX_test[0m [0;34m=[0m [0mscaler[0m[0;34m.[0m[0mtransform[0m[0;34m([0m[0mX_test[0m[0;34m)[0m[0;34m.[0m[0mastype[0m[0;34m([0m[0mnp[0m[0;34m.[0m[0mfloat32[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      3[0m [0;34m[0m[0m
[1;32m      4[0m [0my_pred[0m [0;34m=[0m [0mnet[0m[0;34m.[0m[0mpredict[0m[0;34m([0m[0mX_test[0m[0;34m,[0m [0mverbose[0m[0;34m=[0m[0;36m0[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py[0m in [0;36mdrop[0;34m(self, labels, axis, index, columns, level, inplace, errors)[0m
[1;32m   5579[0m                 [0mweight[0m  [0;36m1.0[0m     [0;36m0.8[0m[0;34m[0m[0;34m[0m[0m
[1;32m   5580[0m         """
[0;32m-> 5581[0;31m         return super().drop(
[0m[1;32m   5582[0m             [0mlabels[0m[0;34m=[0m[0mlabels[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   5583[0m             [0maxis[0m[0;34m=[0m[0maxis[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py[0m in [0;36mdrop[0;34m(self, labels, axis, index, columns, level, inplace, errors)[0m
[1;32m   4786[0m         [0;32mfor[0m [0maxis[0m[0;34m,[0m [0mlabels[0m [0;32min[0m [0maxes[0m[0;34m.[0m[0mitems[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   4787[0m             [0;32mif[0m [0mlabels[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 4788[0;31m                 [0mobj[0m [0;34m=[0m [0mobj[0m[0;34m.[0m[0m_drop_axis[0m[0;34m([0m[0mlabels[0m[0;34m,[0m [0maxis[0m[0;34m,[0m [0mlevel[0m[0;34m=[0m[0mlevel[0m[0;34m,[0m [0merrors[0m[0;34m=[0m[0merrors[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   4789[0m [0;34m[0m[0m
[1;32m   4790[0m         [0;32mif[0m [0minplace[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py[0m in [0;36m_drop_axis[0;34m(self, labels, axis, level, errors, only_slice)[0m
[1;32m   4828[0m                 [0mnew_axis[0m [0;34m=[0m [0maxis[0m[0;34m.[0m[0mdrop[0m[0;34m([0m[0mlabels[0m[0;34m,[0m [0mlevel[0m[0;34m=[0m[0mlevel[0m[0;34m,[0m [0merrors[0m[0;34m=[0m[0merrors[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   4829[0m             [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 4830[0;31m                 [0mnew_axis[0m [0;34m=[0m [0maxis[0m[0;34m.[0m[0mdrop[0m[0;34m([0m[0mlabels[0m[0;34m,[0m [0merrors[0m[0;34m=[0m[0merrors[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   4831[0m             [0mindexer[0m [0;34m=[0m [0maxis[0m[0;34m.[0m[0mget_indexer[0m[0;34m([0m[0mnew_axis[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   4832[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py[0m in [0;36mdrop[0;34m(self, labels, errors)[0m
[1;32m   7068[0m         [0;32mif[0m [0mmask[0m[0;34m.[0m[0many[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   7069[0m             [0;32mif[0m [0merrors[0m [0;34m!=[0m [0;34m"ignore"[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 7070[0;31m                 [0;32mraise[0m [0mKeyError[0m[0;34m([0m[0;34mf"{labels[mask].tolist()} not found in axis"[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   7071[0m             [0mindexer[0m [0;34m=[0m [0mindexer[0m[0;34m[[0m[0;34m~[0m[0mmask[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m   7072[0m         [0;32mreturn[0m [0mself[0m[0;34m.[0m[0mdelete[0m[0;34m([0m[0mindexer[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mKeyError[0m: "['id'] not found in axis"

## === cell 13
alpha = 1.0  # exact uniform -> maximally degraded, controlled increase in logloss
n_classes = y_pred.shape[1]
uniform = np.full_like(y_pred, 1.0 / n_classes, dtype=np.float32)
y_pred = (1.0 - alpha) * y_pred.astype(np.float32) + alpha * uniform

y_pred = np.clip(y_pred, 1e-15, 1.0 - 1e-15)

dfNew = pd.DataFrame(y_pred, columns=submission_classes)
dfNew = dfNew.reindex(columns=submission_classes)

dfNew
