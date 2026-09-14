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

3.6

# 2. Installed packages

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tf_keras==2.18.0

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (99 lines)
            sample_submission.csv (322 lines)
            sample_submission.csv.7z (2.0 kB)
            sample_submission.csv.zip (2.0 kB)
            test.json (1 lines)
            test.json.7z (8.9 MB)
            train.json (1 lines)
            train.json.7z (35.8 MB)
            statoil-iceberg-classifier-challenge/
                description.md (99 lines)
                sample_submission.csv (322 lines)
                ... and 6 other files
                statoil-iceberg-classifier-challenge/
        input/
            description.md (99 lines)
            sample_submission.csv (322 lines)
            sample_submission.csv.7z (2.0 kB)
            sample_submission.csv.zip (2.0 kB)
            test.json (1 lines)
            test.json.7z (8.9 MB)
            train.json (1 lines)
            train.json.7z (35.8 MB)
            statoil-iceberg-classifier-challenge/
                description.md (99 lines)
                sample_submission.csv (322 lines)
                ... and 6 other files
                statoil-iceberg-classifier-challenge/
        working/
            statoil-iceberg-classifier-challenge/
                description.md (99 lines)
                sample_submission.csv (322 lines)
                ... and 6 other files
                statoil-iceberg-classifier-challenge/
```

-> data/sample_submission.csv has 321 rows and 2 columns.
The columns are: id, is_iceberg

-> data/statoil-iceberg-classifier-challenge/sample_submission.csv has 321 rows and 2 columns.
The columns are: id, is_iceberg

-> data/statoil-iceberg-classifier-challenge/test.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "id": {
        "type": "string"
      },
      "band_1": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "band_2": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "inc_angle": {
        "type": [
          "number",
          "string"
        ]
      }
    },
    "required": [
      "band_1",
      "band_2",
      "id",
      "inc_angle"
    ]
  }
}

-> data/statoil-iceberg-classifier-challenge/train.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "id": {
        "type": "string"
      },
      "band_1": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "band_2": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "inc_angle": {
        "type": [
          "number",
          "string"
        ]
      },
      "is_iceberg": {
        "type": "integer"
      }
    },
    "required": [
      "band_1",
      "band_2",
      "id",
      "inc_angle",
      "is_iceberg"
    ]
  }
}

-> data/test.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "id": {
        "type": "string"
      },
      "band_1": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "band_2": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "inc_angle": {
        "type": [
          "number",
          "string"
        ]
      }
    },
    "required": [
      "band_1",
      "band_2",
      "id",
      "inc_angle"
    ]
  }
}

-> data/train.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "id": {
        "type": "string"
      },
      "band_1": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "band_2": {
        "type": "array",
        "items": {
          "type": "number"
        }
      },
      "inc_angle": {
        "type": [
          "number",
          "string"
        ]
      },
      "is_iceberg": {
        "type": "integer"
      }
    },
    "required": [
      "band_1",
      "band_2",
      "id",
      "inc_angle",
      "is_iceberg"
    ]
  }
}

-> input/sample_submission.csv has 321 rows and 2 columns.
The columns are: id, is_iceberg

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))



## === cell 1
train_df = pd.read_json('../input/train.json')
test_df = pd.read_json('../input/test.json')
train_df.head()


## === cell 2
X_band_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train_df["band_1"]])
X_band_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train_df["band_2"]])


## === cell 3
n_samples = X_band_1.shape[0]

X_band = np.zeros([n_samples, 75, 75, 3], dtype=np.float32)
for t in range(n_samples):
    X_band[t, :, :, 0] = X_band_1[t]
    X_band[t, :, :, 1] = X_band_2[t]
    X_band[t, :, :, 2] = (X_band_1[t] + X_band_2[t]) / 2.0


## === cell 4
import os
import sys
import subprocess

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    import google.protobuf  # noqa: F401
except Exception:
    pass

try:
    from google.protobuf import __version__ as _pb_ver

    _pb_major = int(_pb_ver.split(".")[0])
except Exception:
    _pb_major = None

if _pb_major is None or _pb_major >= 5:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "protobuf<5"])

from tf_keras import layers
from tf_keras.layers import (
    Input,
    Dense,
    Activation,
    ZeroPadding2D,
    BatchNormalization,
    Flatten,
    Conv2D,
)
from tf_keras.layers import (
    AveragePooling2D,
    MaxPooling2D,
    Dropout,
    GlobalMaxPooling2D,
    GlobalAveragePooling2D,
)
from tf_keras.models import Model
from tf_keras import backend as K

K.set_image_data_format("channels_last")

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


## === cell 5
def Iceberg_model(input_shape):
    X_in = Input(input_shape)
    
    X = Conv2D(16,kernel_size=(5,5),input_shape=(75,75,3))(X_in)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2,2))(X)
    
    X = Conv2D(32,kernel_size=(5,5))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2,2))(X)
    
    X = Conv2D(64,kernel_size=(5,5))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2,2))(X)
    
    X = Flatten()(X)
    
    X = Dense(128)(X)
    X = Activation('relu')(X)
    
    X = Dense(64)(X)
    X = Activation('relu')(X)
    
    X = Dense(1)(X)
    X = Activation('sigmoid')(X)
    
    model = Model(inputs=X_in,outputs=X,name='Iceberg_model')
    return model


## === cell 6
IcebergModel = Iceberg_model((75,75,3))


## === cell 7
IcebergModel.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


## === cell 8
target = train_df['is_iceberg'].values
IcebergModel.fit(x=X_band,y=target,epochs=20,batch_size=128)


## === cell 9
IcebergModel.evaluate(x=X_band,y=target)


## === cell 10
from sklearn.metrics import classification_report
pred_label = IcebergModel.predict(x=X_band)
pred_label[pred_label>0.5]=1
pred_label[pred_label<=0.5]=0
print(classification_report(target,pred_label))


## === cell 11
X_band_test_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test_df["band_1"]])
X_band_test_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test_df["band_2"]])
X_test = np.zeros([8424,75,75,3])
for t in range(8424):
    X_test[t,:,:,0] = X_band_test_1[t]
    X_test[t,:,:,1] = X_band_test_2[t]
    X_test[t,:,:,2] = (X_band_test_1[t]+X_band_test_2[t])/2


## --- ERROR in cell 11, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mIndexError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/3731687949.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      3[0m [0mX_test[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mzeros[0m[0;34m([0m[0;34m[[0m[0;36m8424[0m[0;34m,[0m[0;36m75[0m[0;34m,[0m[0;36m75[0m[0;34m,[0m[0;36m3[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;32mfor[0m [0mt[0m [0;32min[0m [0mrange[0m[0;34m([0m[0;36m8424[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 5[0;31m     [0mX_test[0m[0;34m[[0m[0mt[0m[0;34m,[0m[0;34m:[0m[0;34m,[0m[0;34m:[0m[0;34m,[0m[0;36m0[0m[0;34m][0m [0;34m=[0m [0mX_band_test_1[0m[0;34m[[0m[0mt[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      6[0m     [0mX_test[0m[0;34m[[0m[0mt[0m[0;34m,[0m[0;34m:[0m[0;34m,[0m[0;34m:[0m[0;34m,[0m[0;36m1[0m[0;34m][0m [0;34m=[0m [0mX_band_test_2[0m[0;34m[[0m[0mt[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m      7[0m     [0mX_test[0m[0;34m[[0m[0mt[0m[0;34m,[0m[0;34m:[0m[0;34m,[0m[0;34m:[0m[0;34m,[0m[0;36m2[0m[0;34m][0m [0;34m=[0m [0;34m([0m[0mX_band_test_1[0m[0;34m[[0m[0mt[0m[0;34m][0m[0;34m+[0m[0mX_band_test_2[0m[0;34m[[0m[0mt[0m[0;34m][0m[0;34m)[0m[0;34m/[0m[0;36m2[0m[0;34m[0m[0;34m[0m[0m

[0;31mIndexError[0m: index 321 is out of bounds for axis 0 with size 321

## === cell 12
pred = IcebergModel.predict(x=X_test)

sub_df = pd.DataFrame()
sub_df['id'] = test_df['id']
sub_df['is_iceberg'] = pred
sub_df.to_csv('output.csv',index=False)
