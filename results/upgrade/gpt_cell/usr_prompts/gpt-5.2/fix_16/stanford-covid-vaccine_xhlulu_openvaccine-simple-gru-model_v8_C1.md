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

3.8

# 2. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
plotly==5.24.1
plotly-express==0.4.1
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

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (125 lines)
            sample_submission.csv (25681 lines)
            sample_submission.csv.zip (74.8 kB)
            test.json (240 lines)
            train.json (2160 lines)
            stanford-covid-vaccine/
                description.md (125 lines)
                sample_submission.csv (25681 lines)
                ... and 3 other files
                stanford-covid-vaccine/
        input/
            description.md (125 lines)
            sample_submission.csv (25681 lines)
            sample_submission.csv.zip (74.8 kB)
            test.json (240 lines)
            train.json (2160 lines)
            stanford-covid-vaccine/
                description.md (125 lines)
                sample_submission.csv (25681 lines)
                ... and 3 other files
                stanford-covid-vaccine/
        working/
            stanford-covid-vaccine/
                description.md (125 lines)
                sample_submission.csv (25681 lines)
                ... and 3 other files
                stanford-covid-vaccine/
```

-> data/sample_submission.csv has 25680 rows and 6 columns.
The columns are: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

-> data/stanford-covid-vaccine/sample_submission.csv has 25680 rows and 6 columns.
The columns are: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

-> data/stanford-covid-vaccine/test.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer"
    },
    "id": {
      "type": "string"
    },
    "sequence": {
      "type": "string"
    },
    "structure": {
      "type": "string"
    },
    "predicted_loop_type": {
      "type": "string"
    },
    "seq_length": {
      "type": "integer"
    },
    "seq_scored": {
      "type": "integer"
    }
  },
  "required": [
    "id",
    "index",
    "predicted_loop_type",
    "seq_length",
    "seq_scored",
    "sequence",
    "structure"
  ]
}

-> data/stanford-covid-vaccine/train.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer"
    },
    "id": {
      "type": "string"
    },
    "sequence": {
      "type": "string"
    },
    "structure": {
      "type": "string"
    },
    "predicted_loop_type": {
      "type": "string"
    },
    "signal_to_noise": {
      "type": "number"
    },
    "SN_filter": {
      "type": "integer"
    },
    "seq_length": {
      "type": "integer"
    },
    "seq_scored": {
      "type": "integer"
    },
    "reactivity_error": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_Mg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_Mg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "reactivity": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_Mg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_Mg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    }
  },
  "required": [
    "SN_filter",
    "deg_50C",
    "deg_Mg_50C",
    "deg_Mg_pH10",
    "deg_error_50C",
    "deg_error_Mg_50C",
    "deg_error_Mg_pH10",
    "deg_error_pH10",
    "deg_pH10",
    "id",
    "index",
    "predicted_loop_type",
    "reactivity",
    "reactivity_error",
    "seq_length",
    "seq_scored",
    "sequence",
    "signal_to_noise",
    "structure"
  ]
}

-> data/test.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer"
    },
    "id": {
      "type": "string"
    },
    "sequence": {
      "type": "string"
    },
    "structure": {
      "type": "string"
    },
    "predicted_loop_type": {
      "type": "string"
    },
    "seq_length": {
      "type": "integer"
    },
    "seq_scored": {
      "type": "integer"
    }
  },
  "required": [
    "id",
    "index",
    "predicted_loop_type",
    "seq_length",
    "seq_scored",
    "sequence",
    "structure"
  ]
}

-> data/train.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "index": {
      "type": "integer"
    },
    "id": {
      "type": "string"
    },
    "sequence": {
      "type": "string"
    },
    "structure": {
      "type": "string"
    },
    "predicted_loop_type": {
      "type": "string"
    },
    "signal_to_noise": {
      "type": "number"
    },
    "SN_filter": {
      "type": "integer"
    },
    "seq_length": {
      "type": "integer"
    },
    "seq_scored": {
      "type": "integer"
    },
    "reactivity_error": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_Mg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_Mg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_error_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "reactivity": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_Mg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_pH10": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_Mg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    },
    "deg_50C": {
      "type": "array",
      "items": {
        "type": "number"
      }
    }
  },
  "required": [
    "SN_filter",
    "deg_50C",
    "deg_Mg_50C",
    "deg_Mg_pH10",
    "deg_error_50C",
    "deg_error_Mg_50C",
    "deg_error_Mg_pH10",
    "deg_error_pH10",
    "deg_pH10",
    "id",
    "index",
    "predicted_loop_type",
    "reactivity",
    "reactivity_error",
    "seq_length",
    "seq_scored",
    "sequence",
    "signal_to_noise",
    "structure"
  ]
}

-> input/sample_submission.csv has 25680 rows and 6 columns.
The columns are: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

try:
    from google.protobuf import message_factory as _message_factory

    if hasattr(_message_factory, "MessageFactory"):
        _mf_cls = _message_factory.MessageFactory
        if not hasattr(_mf_cls, "GetPrototype"):
            _get_msg_cls = getattr(_message_factory, "GetMessageClass", None)
            if _get_msg_cls is not None:

                def _GetPrototype(self, descriptor):
                    return _get_msg_cls(descriptor)

                _mf_cls.GetPrototype = _GetPrototype
except Exception:
    pass

import json

import pandas as pd
import numpy as np
import plotly.express as px
import tensorflow.keras.layers as L
import tensorflow as tf
from sklearn.model_selection import train_test_split


## === cell 1
tf.random.set_seed(2020)
np.random.seed(2020)


## === cell 2
pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']


## === cell 3
def MCRMSE(y_true, y_pred):
    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)
    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)


## === cell 4
def gru_layer(hidden_dim, dropout):
    return L.Bidirectional(
        L.GRU(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer='orthogonal')
    )


## === cell 5
def build_model(embed_size, seq_len=107, pred_len=68, dropout=0.5, 
                sp_dropout=0.2, embed_dim=75, hidden_dim=128, n_layers=2):
    inputs = L.Input(shape=(seq_len, 3))
    embed = L.Embedding(input_dim=embed_size, output_dim=embed_dim)(inputs)
    
    reshaped = tf.reshape(
        embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3])
    )
    hidden = L.SpatialDropout1D(sp_dropout)(reshaped)
    
    for x in range(n_layers):
        hidden = gru_layer(hidden_dim, dropout)(hidden)
    
    truncated = hidden[:, :pred_len]
    out = L.Dense(5, activation='linear')(truncated)
    
    model = tf.keras.Model(inputs=inputs, outputs=out)
    model.compile(tf.optimizers.Adam(), loss='mse', metrics=[MCRMSE])
    
    return model


## === cell 6
def pandas_list_to_array(df):
    """
    Input: dataframe of shape (x, y), containing list of length l
    Return: np.array of shape (x, l, y)
    """
    
    return np.transpose(
        np.array(df.values.tolist()),
        (0, 2, 1)
    )


## === cell 7
def preprocess_inputs(df, token2int, cols=['sequence', 'structure', 'predicted_loop_type']):
    return pandas_list_to_array(
        df[cols].applymap(lambda seq: [token2int[x] for x in seq])
    )


## === cell 8
data_dir = '/kaggle/input/stanford-covid-vaccine/'
train = pd.read_json(data_dir + 'train.json', lines=True)
test = pd.read_json(data_dir + 'test.json', lines=True)
sample_df = pd.read_csv(data_dir + 'sample_submission.csv')


## === cell 9
token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}

train_inputs = preprocess_inputs(train, token2int)
train_labels = pandas_list_to_array(train[pred_cols])


## === cell 10
x_train, x_val, y_train, y_val = train_test_split(
    train_inputs, train_labels, test_size=.1, random_state=34
)


## === cell 11
public_df = test.query("seq_length == 107")
private_df = test.query("seq_length == 130")

SEQ_LEN = 107


def _safe_preprocess(df):
    if df is None or len(df) == 0:
        return np.zeros((0, SEQ_LEN, 3), dtype=np.int32)
    return preprocess_inputs(df, token2int)[:, :SEQ_LEN, :]


public_inputs = _safe_preprocess(public_df)
private_inputs = _safe_preprocess(private_df)


## === cell 12
def build_model(
    embed_size,
    seq_len=107,
    pred_len=68,
    dropout=0.5,
    sp_dropout=0.2,
    embed_dim=75,
    hidden_dim=128,
    n_layers=2,
):
    inputs = L.Input(shape=(seq_len, 3))
    embed = L.Embedding(input_dim=embed_size, output_dim=embed_dim)(inputs)

    reshaped = L.Reshape((seq_len, 3 * embed_dim))(embed)

    hidden = L.SpatialDropout1D(sp_dropout)(reshaped)

    for x in range(n_layers):
        hidden = gru_layer(hidden_dim, dropout)(hidden)

    truncated = hidden[:, :pred_len]
    out = L.Dense(5, activation="linear")(truncated)

    model = tf.keras.Model(inputs=inputs, outputs=out)
    model.compile(tf.optimizers.Adam(), loss="mse", metrics=[MCRMSE])

    return model


model = build_model(embed_size=len(token2int))
model.summary()


## === cell 13
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    batch_size=64,
    epochs=70,
    verbose=2,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_MCRMSE', patience=5),
        tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='val_MCRMSE')
    ]
)


## === cell 14
hist_df = pd.DataFrame(history.history)

fig = px.line(
    hist_df,
    y=["loss", "val_loss", "MCRMSE", "val_MCRMSE"],
    labels={"index": "epoch", "value": "Mean Squared Error"},
    title="Training History",
)
fig.show()


## --- ERROR in cell 14, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/2562909837.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      3[0m [0mhist_df[0m [0;34m=[0m [0mpd[0m[0;34m.[0m[0mDataFrame[0m[0;34m([0m[0mhistory[0m[0;34m.[0m[0mhistory[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m [0;34m[0m[0m
[0;32m----> 5[0;31m fig = px.line(
[0m[1;32m      6[0m     [0mhist_df[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m      7[0m     [0my[0m[0;34m=[0m[0;34m[[0m[0;34m"loss"[0m[0;34m,[0m [0;34m"val_loss"[0m[0;34m,[0m [0;34m"MCRMSE"[0m[0;34m,[0m [0;34m"val_MCRMSE"[0m[0;34m][0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/plotly/express/_chart_types.py[0m in [0;36mline[0;34m(data_frame, x, y, line_group, color, line_dash, symbol, hover_name, hover_data, custom_data, text, facet_row, facet_col, facet_col_wrap, facet_row_spacing, facet_col_spacing, error_x, error_x_minus, error_y, error_y_minus, animation_frame, animation_group, category_orders, labels, orientation, color_discrete_sequence, color_discrete_map, line_dash_sequence, line_dash_map, symbol_sequence, symbol_map, markers, log_x, log_y, range_x, range_y, line_shape, render_mode, title, template, width, height)[0m
[1;32m    262[0m     [0ma[0m [0mpolyline[0m [0mmark[0m [0;32min[0m [0;36m2[0m[0mD[0m [0mspace[0m[0;34m.[0m[0;34m[0m[0;34m[0m[0m
[1;32m    263[0m     """
[0;32m--> 264[0;31m     [0;32mreturn[0m [0mmake_figure[0m[0;34m([0m[0margs[0m[0;34m=[0m[0mlocals[0m[0;34m([0m[0;34m)[0m[0;34m,[0m [0mconstructor[0m[0;34m=[0m[0mgo[0m[0;34m.[0m[0mScatter[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    265[0m [0;34m[0m[0m
[1;32m    266[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/plotly/express/_core.py[0m in [0;36mmake_figure[0;34m(args, constructor, trace_patch, layout_patch)[0m
[1;32m   2115[0m     [0mapply_default_cascade[0m[0;34m([0m[0margs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2116[0m [0;34m[0m[0m
[0;32m-> 2117[0;31m     [0margs[0m [0;34m=[0m [0mbuild_dataframe[0m[0;34m([0m[0margs[0m[0;34m,[0m [0mconstructor[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   2118[0m     [0;32mif[0m [0mconstructor[0m [0;32min[0m [0;34m[[0m[0mgo[0m[0;34m.[0m[0mTreemap[0m[0;34m,[0m [0mgo[0m[0;34m.[0m[0mSunburst[0m[0;34m,[0m [0mgo[0m[0;34m.[0m[0mIcicle[0m[0;34m][0m [0;32mand[0m [0margs[0m[0;34m[[0m[0;34m"path"[0m[0;34m][0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2119[0m         [0margs[0m [0;34m=[0m [0mprocess_dataframe_hierarchy[0m[0;34m([0m[0margs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/plotly/express/_core.py[0m in [0;36mbuild_dataframe[0;34m(args, constructor)[0m
[1;32m   1511[0m     [0;31m# now that things have been prepped, we do the systematic rewriting of `args`[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1512[0m [0;34m[0m[0m
[0;32m-> 1513[0;31m     df_output, wide_id_vars = process_args_into_dataframe(
[0m[1;32m   1514[0m         [0margs[0m[0;34m,[0m [0mwide_mode[0m[0;34m,[0m [0mvar_name[0m[0;34m,[0m [0mvalue_name[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1515[0m     )

[0;32m/usr/local/lib/python3.11/dist-packages/plotly/express/_core.py[0m in [0;36mprocess_args_into_dataframe[0;34m(args, wide_mode, var_name, value_name)[0m
[1;32m   1272[0m [0;34m[0m[0m
[1;32m   1273[0m                 [0;32mif[0m [0mlength[0m [0;32mand[0m [0mlen[0m[0;34m([0m[0margument[0m[0;34m)[0m [0;34m!=[0m [0mlength[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1274[0;31m                     raise ValueError(
[0m[1;32m   1275[0m                         [0;34m"All arguments should have the same length. "[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1276[0m                         [0;34m"The length of argument `%s` is %d, whereas the "[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: All arguments should have the same length. The length of argument `y` is 4, whereas the length of  previously-processed arguments ['index'] is 70

## === cell 15
model_public = build_model(seq_len=107, pred_len=107, embed_size=len(token2int))
model_private = build_model(seq_len=130, pred_len=130, embed_size=len(token2int))

model_public.load_weights('model.h5')
model_private.load_weights('model.h5')
