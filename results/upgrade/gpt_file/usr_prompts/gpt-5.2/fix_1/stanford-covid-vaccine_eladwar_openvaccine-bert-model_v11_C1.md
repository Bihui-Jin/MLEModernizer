# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Overview
Predict likely degradation rates at each base of an RNA molecule.

## Metric
Mean columnwise root mean squared error:

$\textrm{MCRMSE} = \frac{1}{N_{t}}\sum_{j=1}^{N_{t}}\sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_{ij} - \hat{y}_{ij})^2}$

where $N_{t}$ is the number of scored ground truth target columns, and $y$ and $\hat{y}$ are the actual and predicted values, respectively.

There are multiple ground truth values provided in the training data. While the submission format requires all 5 to be predicted, only the following are scored: reactivity, deg_Mg_pH10, and deg_Mg_50C.

## Submission Formats
For each sample `id` in the test set, you must predict targets for *each* sequence position (`seqpos`), one per row. If the length of the `sequence` of an `id` is, e.g., 107, then you should make 107 predictions. Positions greater than the `seq_scored` value of a sample are not scored, but still need a value in the solution file.

```csv
id_seqpos,reactivity,deg_Mg_pH10,deg_pH10,deg_Mg_50C,deg_50C
id_d190610e8_0,0.1,0.3,0.2,0.5,0.4
id_d190610e8_1,0.3,0.2,0.5,0.4,0.2
id_d190610e8_2,0.5,0.4,0.2,0.1,0.2
etc.
```

## Dataset 
- **train.json** - the training data
- **test.json** - the test set, without any columns associated with the ground truth.
- **sample_submission.csv** - a sample submission file in the correct format

#### Columns
- `id` - An arbitrary identifier for each sample.
- `seq_scored` - (68 in Train and Public Test, 68 in Private Test) Integer value denoting the number of positions used in scoring with predicted values. This should match the length of `reactivity`, `deg_*` and `*_error_*` columns.
- `seq_length` - (107 in Train and Public Test, 107 in Private Test) Integer values, denotes the length of `sequence`.
- `sequence` - (1x107 string in Train and Public Test, 107 in Private Test) Describes the RNA sequence, a combination of `A`, `G`, `U`, and `C` for each sample. Should be 107 characters long, and the first 68 bases should correspond to the 68 positions specified in `seq_scored` (note: indexed starting at 0).
- `structure` - (1x107 string in Train and Public Test, 107 in Private Test) An array of `(`, `)`, and `.` characters that describe whether a base is estimated to be paired or unpaired. Paired bases are denoted by opening and closing parentheses e.g. (....) means that base 0 is paired to base 5, and bases 1-4 are unpaired.
- `reactivity` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likely secondary structure of the RNA sample.
- `deg_pH10` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating without magnesium at high pH (pH 10).
- `deg_Mg_pH10` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating with magnesium in high pH (pH 10).
- `deg_50C` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating without magnesium at high temperature (50 degrees Celsius).
- `deg_Mg_50C` - (1x68 vector in Train and Public Test, 1x68 in Private Test) An array of floating point numbers, should have the same length as `seq_scored`. These numbers are reactivity values for the first 68 bases as denoted in `sequence`, and used to determine the likelihood of degradation at the base/linkage after incubating with magnesium at high temperature (50 degrees Celsius).
- `*_error_*` - An array of floating point numbers, should have the same length as the corresponding `reactivity` or `deg_*` columns, calculated errors in experimental values obtained in `reactivity` and `deg_*` columns.
- `predicted_loop_type` - (1x107 string) Describes the structural context (also referred to as 'loop type')of each character in `sequence`. Loop types assigned by bpRNA from Vienna RNAfold 2 structure. From the bpRNA_documentation: S: paired "Stem" M: Multiloop I: Internal loop B: Bulge H: Hairpin loop E: dangling End X: eXternal loop
    - `S/N filter` Indicates if the sample passed filters described below in `Additional Notes`.

#### Additional Notes
At the beginning of the competition, Stanford scientists have data on 2400 RNA sequences of length 107. For technical reasons, measurements cannot be carried out on the final bases of these RNA sequences, so we have experimental data (ground truth) in 5 conditions for the first 68 bases.

We have split out 240 of these 2400 sequences for a public test set to allow for continuous evaluation through the competition, on the public leaderboard. These sequences, in `test.json`, have been additionally filtered based on three criteria detailed below to ensure that this subset is not dominated by any large cluster of RNA molecules with poor data, which might bias the public leaderboard. The remaining 2160 sequences for which we have data are in `train.json`.

For our final and most important scoring (the Private Leaderbooard), Stanford scientists are carrying out measurements on 240 new RNAs. For these data, we expect to have measurements for the first 68 bases, again missing the ends of the RNA. These sequences constitute the 240 sequences in `test.json`.

For those interested in how the sequences in `test.json` were filtered, here were the steps to ensure a diverse and high quality test set for public leaderboard scoring:

1. Minimum value across all 5 conditions must be greater than -0.5.
2. Mean signal/noise across all 5 conditions must be greater than 1.0. [Signal/noise is defined as mean( measurement value over 68 nts )/mean( statistical error in measurement value over 68 nts)]
3. To help ensure sequence diversity, the resulting sequences were clustered into clusters with less than 50% sequence similarity, and the 240 test set sequences were chosen from clusters with 3 or fewer members. That is, any sequence in the test set should be sequence similar to at most 2 other sequences.

Note that these filters have not been applied to the 2160 RNAs in the public training data `train.json` -- some of those measurements have negative values or poor signal-to-noise, or some RNA sequences have near-identical sequences in that set. But we are providing all those data in case competitors can squeeze out more signal.

# 2. Python version

3.8

# 3. Installed packages

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
sentence-transformers==4.1.0
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
transformers==4.53.3

# 4. Data file paths

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

# 5. Target score

0.55896

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import json
import tensorflow.keras.layers as L
import tensorflow as tf
import plotly.express as px
from sklearn.preprocessing import quantile_transform,StandardScaler
from transformers import BertConfig,TFBertModel,BertModel


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']


## === cell 2
def gru_layer(hidden_dim, dropout):
    return L.Bidirectional(L.GRU(hidden_dim, dropout=dropout, return_sequences=True))

def build_model(seq_len=107, pred_len=68, dropout=0.5, embed_dim=100, hidden_dim=128):
    ids = L.Input(shape=(seq_len,3), dtype=tf.int32)
    config = BertConfig() 
    config.vocab_size = len(token2int)
    config.num_hidden_layers = 3
    config.attention_probs_dropout_prob  = 0.133
    config.hidden_act='swish'
    bert_model = TFBertModel(config=config)

    embed = bert_model(ids[:,0])[0]
    embed1 = bert_model(ids[:,1])[0]
    embed2 = bert_model(ids[:,2])[0]
    bert_embeddings = L.Concatenate()([embed,embed1,embed2])
    reshaped = tf.reshape(
        bert_embeddings, shape=(-1, bert_embeddings.shape[2],  bert_embeddings.shape[1]))

    hidden = L.AveragePooling1D(pool_size=16)(reshaped)
    truncated = hidden[:,:pred_len, :]
    out = L.Dense(5, activation='linear')(truncated)

    model = tf.keras.Model(inputs=ids, outputs=out)

    model.compile(tf.keras.optimizers.Adam(), loss='mse')
    
    return model


## === cell 3
init_checkpoint


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4230592513.py in <cell line: 0>()
----> 1 init_checkpoint

NameError: name 'init_checkpoint' is not defined

## === cell 4
token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}

def preprocess_inputs(df, cols=['sequence', 'structure', 'predicted_loop_type']):
    return np.transpose(
        np.array(
            df[cols]
            .applymap(lambda seq: [token2int[x] for x in seq])
            .values
            .tolist()
        ),
        (0, 2, 1)
    )


## === cell 5
train = pd.read_json('/kaggle/input/stanford-covid-vaccine/train.json', lines=True)
test = pd.read_json('/kaggle/input/stanford-covid-vaccine/test.json', lines=True)
sample_df = pd.read_csv('/kaggle/input/stanford-covid-vaccine/sample_submission.csv')


## === cell 6
train_inputs = preprocess_inputs(train)
train_labels = np.array(train[pred_cols].values.tolist()).transpose((0, 2, 1))


## === cell 7
train_inputs.shape


## === cell 8
for df in [train,test]:
    df['Paired']=[sum([i=='(' or i==')' for i in j]) for j in df['structure']]
    df['Unpaired']=[sum([i=='.' for i in j]) for j in df['structure']]
    for col in ['E','S','H','I','G','A','U']:
        if col in ['E','S','H','I']:
            df[col]=[sum([i==col for i in j])/len(j) for j in df['predicted_loop_type']]
        else:
            df[col]=[sum([i==col for i in j])/len(j) for j in df['sequence']]
for a in [ 'G', 'A', 'C', 'U']:
    train[a+'_position']=[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in train['sequence']]
    test[a+'_position']=[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in test['sequence']]
for a in [ 'E', 'S', 'H',]:
    train[a+'_position']=[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in train['predicted_loop_type']]
    test[a+'_position']=[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in test['predicted_loop_type']]
for a in [ 'E', 'S', 'H',]:
    train[a+'']=[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in train['predicted_loop_type']]
    test[a+'_position']=[np.sum([i for i in range(len(j)) if j[i]==a])/len([i for i in range(len(j)) if j[i]==a]) for j in test['predicted_loop_type']]


## === cell 9
target_columns = ['reactivity', 'deg_Mg_pH10','deg_pH10', 'deg_Mg_50C', 'deg_50C']
target_columns.extend(['SN_filter', 'signal_to_noise'])
target_columns.extend(['deg_error_pH10', 'deg_error_Mg_50C', 'deg_error_50C', 'reactivity_error', 'deg_error_Mg_pH10'] )
train.drop(target_columns,axis=1,inplace=True)


## === cell 10
SC = StandardScaler()
train_measurements = SC.fit_transform(pd.concat((train.select_dtypes('float64'),train.select_dtypes('int64')),axis=1))


## === cell 11
train_measurements.shape


## === cell 12
np.min(train_measurements),np.max(train_measurements)


## === cell 13
model = build_model()
model.summary()


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2178120478.py in <cell line: 0>()
----> 1 model = build_model()
      2 model.summary()

/tmp/ipykernel_11/2704921152.py in build_model(seq_len, pred_len, dropout, embed_dim, hidden_dim)
     11     bert_model = TFBertModel(config=config)
     12 
---> 13     embed = bert_model(ids[:,0])[0]
     14     embed1 = bert_model(ids[:,1])[0]
     15     embed2 = bert_model(ids[:,2])[0]

/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
     68             # To get the full stack trace, call:
     69             # `tf.debugging.disable_traceback_filtering()`
---> 70             raise e.with_traceback(filtered_tb) from None
     71         finally:
     72             del filtered_tb

/usr/local/lib/python3.11/dist-packages/transformers/modeling_tf_utils.py in run_call_with_unpacked_inputs(self, *args, **kwargs)
    434             config = self.config
    435 
--> 436         unpacked_inputs = input_processing(func, config, **fn_args_and_kwargs)
    437         return func(self, **unpacked_inputs)
    438 

/usr/local/lib/python3.11/dist-packages/transformers/modeling_tf_utils.py in input_processing(func, config, **kwargs)
    564             output[main_input_name] = main_input
    565         else:
--> 566             raise ValueError(
    567                 f"Data of type {type(main_input)} is not allowed only {allowed_types} is accepted for"
    568                 f" {main_input_name}."

ValueError: Exception encountered when calling layer 'tf_bert_model' (type TFBertModel).

Data of type <class 'keras.src.backend.common.keras_tensor.KerasTensor'> is not allowed only (<class 'tensorflow.python.framework.tensor.Tensor'>, <class 'bool'>, <class 'int'>, <class 'transformers.utils.generic.ModelOutput'>, <class 'tuple'>, <class 'list'>, <class 'dict'>, <class 'numpy.ndarray'>) is accepted for input_ids.

Call arguments received by layer 'tf_bert_model' (type TFBertModel):
  • input_ids=<KerasTensor shape=(None, 3), dtype=int32, sparse=False, name=keras_tensor_1>
  • attention_mask=None
  • token_type_ids=None
  • position_ids=None
  • head_mask=None
  • inputs_embeds=None
  • encoder_hidden_states=None
  • encoder_attention_mask=None
  • past_key_values=None
  • use_cache=None
  • output_attentions=None
  • output_hidden_states=None
  • return_dict=None
  • training=False

## === cell 14
train_inputs.shape,train_labels.shape


## === cell 15
with tf.device('/gpu'):
    history = model.fit(
        train_inputs, train_labels, 
        batch_size=64,
        epochs=100,
        validation_split=0.05,
            callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau()
    ]
    )


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2155643642.py in <cell line: 0>()
      1 with tf.device('/gpu'):
----> 2     history = model.fit(
      3         train_inputs, train_labels,
      4         batch_size=64,
      5         epochs=100,

NameError: name 'model' is not defined

## === cell 16
model.save_weights('model.h5')


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3602248015.py in <cell line: 0>()
      1 #Bert's a bitch with weights
----> 2 model.save_weights('model.h5')

NameError: name 'model' is not defined

## === cell 18
fig = px.line(
    history.history, y=['loss', 'val_loss'], 
    labels={'index': 'epoch', 'value': 'Mean Squared Error'}, 
    title='Training History')
fig.show()


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1068241897.py in <cell line: 0>()
      1 fig = px.line(
----> 2     history.history, y=['loss', 'val_loss'],
      3     labels={'index': 'epoch', 'value': 'Mean Squared Error'},
      4     title='Training History')
      5 fig.show()

NameError: name 'history' is not defined

## === cell 19
public_df = test.query("seq_length == 107").copy()
private_df = test.query("seq_length == 130").copy()
public_test_measurements  = SC.fit_transform(pd.concat((public_df.select_dtypes('float64'),public_df.select_dtypes('int64')),axis=1))
private_test_measurements  = SC.fit_transform(pd.concat((private_df.select_dtypes('float64'),private_df.select_dtypes('int64')),axis=1))
public_inputs = preprocess_inputs(public_df)
private_inputs = preprocess_inputs(private_df)


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/188013814.py in <cell line: 0>()
      2 private_df = test.query("seq_length == 130").copy()
      3 public_test_measurements  = SC.fit_transform(pd.concat((public_df.select_dtypes('float64'),public_df.select_dtypes('int64')),axis=1))
----> 4 private_test_measurements  = SC.fit_transform(pd.concat((private_df.select_dtypes('float64'),private_df.select_dtypes('int64')),axis=1))
      5 public_inputs = preprocess_inputs(public_df)
      6 private_inputs = preprocess_inputs(private_df)

/usr/local/lib/python3.11/dist-packages/sklearn/utils/_set_output.py in wrapped(self, X, *args, **kwargs)
    138     @wraps(f)
    139     def wrapped(self, X, *args, **kwargs):
--> 140         data_to_wrap = f(self, X, *args, **kwargs)
    141         if isinstance(data_to_wrap, tuple):
    142             # only wrap the first output for cross decomposition

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in fit_transform(self, X, y, **fit_params)
    876         if y is None:
    877             # fit method of arity 1 (unsupervised transformation)
--> 878             return self.fit(X, **fit_params).transform(X)
    879         else:
    880             # fit method of arity 2 (supervised transformation)

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_data.py in fit(self, X, y, sample_weight)
    822         # Reset internal state before fitting
    823         self._reset()
--> 824         return self.partial_fit(X, y, sample_weight)
    825 
    826     def partial_fit(self, X, y=None, sample_weight=None):

/usr/local/lib/python3.11/dist-packages/sklearn/preprocessing/_data.py in partial_fit(self, X, y, sample_weight)
    859 
    860         first_call = not hasattr(self, "n_samples_seen_")
--> 861         X = self._validate_data(
    862             X,
    863             accept_sparse=("csr", "csc"),

/usr/local/lib/python3.11/dist-packages/sklearn/base.py in _validate_data(self, X, y, reset, validate_separately, **check_params)
    563             raise ValueError("Validation should be done on X, y or both.")
    564         elif not no_val_X and no_val_y:
--> 565             X = check_array(X, input_name="X", **check_params)
    566             out = X
    567         elif no_val_X and not no_val_y:

/usr/local/lib/python3.11/dist-packages/sklearn/utils/validation.py in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
    929         n_samples = _num_samples(array)
    930         if n_samples < ensure_min_samples:
--> 931             raise ValueError(
    932                 "Found array with %d sample(s) (shape=%s) while a"
    933                 " minimum of %d is required%s."

ValueError: Found array with 0 sample(s) (shape=(0, 19)) while a minimum of 1 is required by StandardScaler.

## === cell 20
model_short = build_model(seq_len=107, pred_len=107)
model_long = build_model(seq_len=130, pred_len=130)

model_short.load_weights('model.h5')
model_long.load_weights('model.h5')

public_preds = model_short.predict([public_inputs])
private_preds = model_long.predict([private_inputs])


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3669277551.py in <cell line: 0>()
      1 # Caveat: The prediction format requires the output to be the same length as the input,
      2 # although it's not the case for the training data.
----> 3 model_short = build_model(seq_len=107, pred_len=107)
      4 model_long = build_model(seq_len=130, pred_len=130)
      5 

/tmp/ipykernel_11/2704921152.py in build_model(seq_len, pred_len, dropout, embed_dim, hidden_dim)
     11     bert_model = TFBertModel(config=config)
     12 
---> 13     embed = bert_model(ids[:,0])[0]
     14     embed1 = bert_model(ids[:,1])[0]
     15     embed2 = bert_model(ids[:,2])[0]

/usr/local/lib/python3.11/dist-packages/tf_keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
     68             # To get the full stack trace, call:
     69             # `tf.debugging.disable_traceback_filtering()`
---> 70             raise e.with_traceback(filtered_tb) from None
     71         finally:
     72             del filtered_tb

/usr/local/lib/python3.11/dist-packages/transformers/modeling_tf_utils.py in run_call_with_unpacked_inputs(self, *args, **kwargs)
    434             config = self.config
    435 
--> 436         unpacked_inputs = input_processing(func, config, **fn_args_and_kwargs)
    437         return func(self, **unpacked_inputs)
    438 

/usr/local/lib/python3.11/dist-packages/transformers/modeling_tf_utils.py in input_processing(func, config, **kwargs)
    564             output[main_input_name] = main_input
    565         else:
--> 566             raise ValueError(
    567                 f"Data of type {type(main_input)} is not allowed only {allowed_types} is accepted for"
    568                 f" {main_input_name}."

ValueError: Exception encountered when calling layer 'tf_bert_model_1' (type TFBertModel).

Data of type <class 'keras.src.backend.common.keras_tensor.KerasTensor'> is not allowed only (<class 'tensorflow.python.framework.tensor.Tensor'>, <class 'bool'>, <class 'int'>, <class 'transformers.utils.generic.ModelOutput'>, <class 'tuple'>, <class 'list'>, <class 'dict'>, <class 'numpy.ndarray'>) is accepted for input_ids.

Call arguments received by layer 'tf_bert_model_1' (type TFBertModel):
  • input_ids=<KerasTensor shape=(None, 3), dtype=int32, sparse=False, name=keras_tensor_3>
  • attention_mask=None
  • token_type_ids=None
  • position_ids=None
  • head_mask=None
  • inputs_embeds=None
  • encoder_hidden_states=None
  • encoder_attention_mask=None
  • past_key_values=None
  • use_cache=None
  • output_attentions=None
  • output_hidden_states=None
  • return_dict=None
  • training=False

## === cell 21
print(public_preds.shape, private_preds.shape)


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3862167184.py in <cell line: 0>()
----> 1 print(public_preds.shape, private_preds.shape)

NameError: name 'public_preds' is not defined

## === cell 22
preds_ls = []

for df, preds in [(public_df, public_preds), (private_df, private_preds)]:
    for i, uid in enumerate(df.id):
        single_pred = preds[i]

        single_df = pd.DataFrame(single_pred, columns=pred_cols)
        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]

        preds_ls.append(single_df)

preds_df = pd.concat(preds_ls)


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3747906997.py in <cell line: 0>()
      1 preds_ls = []
      2 
----> 3 for df, preds in [(public_df, public_preds), (private_df, private_preds)]:
      4     for i, uid in enumerate(df.id):
      5         single_pred = preds[i]

NameError: name 'public_preds' is not defined

## === cell 23
submission = sample_df[['id_seqpos']].merge(preds_df, on=['id_seqpos'])
submission.to_csv('submission.csv', index=False)


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2529884991.py in <cell line: 0>()
----> 1 submission = sample_df[['id_seqpos']].merge(preds_df, on=['id_seqpos'])
      2 submission.to_csv('submission.csv', index=False)

NameError: name 'preds_df' is not defined
