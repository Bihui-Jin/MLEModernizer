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
Predict which chatbot response a user will prefer in a competition between two chatbots.

## Metric
Log loss with "eps=auto"

## Submission Format
For each id in the test set, you must predict the probability for each target class. The file should contain a header and have the following format:

```
 id,winner_model_a,winner_model_b,winner_tie
 136060,0.33,0,33,0.33
 211333,0.33,0,33,0.33
 1233961,0.33,0,33,0.33
 etc
```

## Dataset
**train.csv**

- `id` - A unique identifier for the row.
- `model_[a/b]` - The identity of model_[a/b]. Included in train.csv but not test.csv.
- `prompt` - The prompt that was given as an input (to both models).
- `response_[a/b]` - The response from model_[a/b] to the given prompt.
- `winner_model_[a/b/tie]` - Binary columns marking the judge's selection. The ground truth target column.

**test.csv**

- `id`
- `prompt`
- `response_[a/b]`

**sample_submission.csv** A submission file in the correct format.

- `id`
- `winner_model_[a/b/tie]` - This is what is predicted from the test set.

# 2. Python version

3.14

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (98 lines)
            sample_submission.csv (5749 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (5749 lines)
            test.csv.zip (5.9 MB)
            train.csv (51730 lines)
            train.csv.zip (53.4 MB)
            lmsys-chatbot-arena/
                description.md (98 lines)
                sample_submission.csv (5749 lines)
                ... and 5 other files
                lmsys-chatbot-arena/
        input/
            description.md (98 lines)
            sample_submission.csv (5749 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (5749 lines)
            test.csv.zip (5.9 MB)
            train.csv (51730 lines)
            train.csv.zip (53.4 MB)
            lmsys-chatbot-arena/
                description.md (98 lines)
                sample_submission.csv (5749 lines)
                ... and 5 other files
                lmsys-chatbot-arena/
        working/
            lmsys-chatbot-arena/
                description.md (98 lines)
                sample_submission.csv (5749 lines)
                ... and 5 other files
                lmsys-chatbot-arena/
```

-> data/lmsys-chatbot-arena/sample_submission.csv has 5748 rows and 4 columns.
The columns are: id, winner_model_a, winner_model_b, winner_tie

-> data/lmsys-chatbot-arena/test.csv has 5748 rows and 4 columns.
The columns are: id, prompt, response_a, response_b

-> data/lmsys-chatbot-arena/train.csv has 51729 rows and 9 columns.
The columns are: id, model_a, model_b, prompt, response_a, response_b, winner_model_a, winner_model_b, winner_tie

-> data/sample_submission.csv has 5748 rows and 4 columns.
The columns are: id, winner_model_a, winner_model_b, winner_tie

-> data/test.csv has 5748 rows and 4 columns.
The columns are: id, prompt, response_a, response_b

-> data/train.csv has 51729 rows and 9 columns.
The columns are: id, model_a, model_b, prompt, response_a, response_b, winner_model_a, winner_model_b, winner_tie

-> (stopped after 10 files for performance)

# 5. Target score

1.0934370701962233

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
!pip install --upgrade protobuf==3.20.3

## === cell 1
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import warnings
warnings.filterwarnings('ignore')


## === cell 2
print("TensorFlow Version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

test = pd.read_csv('/kaggle/input/lmsys-chatbot-arena/test.csv')
sample_submission = pd.read_csv('/kaggle/input/lmsys-chatbot-arena/sample_submission.csv')

print("Test Shape:", test.shape)
print("Sample Submission Shape:", sample_submission.shape)


## === cell 3
test['combined_text'] = test['prompt'] + ' [SEP] ' + test['response_a'] + ' [SEP] ' + test['response_b']


## === cell 4
MAX_LEN = 512

print("\n" + "="*80)
print("LOADING GRU MODELS AND TOKENIZER")
print("="*80)

with open('/kaggle/input/lmsys-gru/tokenizer_gru.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

X_test_seq = tokenizer.texts_to_sequences(test['combined_text'])
X_test = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post', truncating='post')

print("X_test shape:", X_test.shape)


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3131210540.py in <cell line: 0>()
      5 print("="*80)
      6 
----> 7 with open('/kaggle/input/lmsys-gru/tokenizer_gru.pkl', 'rb') as f:
      8     tokenizer = pickle.load(f)
      9 

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/lmsys-gru/tokenizer_gru.pkl'

## === cell 5
import tensorflow as tf
from tensorflow import keras

class NotEqual(keras.layers.Layer):
    def __init__(self, cast_to_float=True, **kwargs):
        super().__init__(**kwargs)
        self.cast_to_float = cast_to_float
    
    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            inputs = inputs[0]
        
        out = tf.math.not_equal(inputs, 0)
        if self.cast_to_float:
            return tf.cast(out, tf.float32)
        return out
    
    def get_config(self):
        config = super().get_config()
        config.update({"cast_to_float": self.cast_to_float})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

def custom_load_model(filepath):
    import json
    import h5py
    
    with h5py.File(filepath, 'r') as f:
        model_config = json.loads(f.attrs['model_config'])
        
        for layer in model_config['config']['layers']:
            if layer['class_name'] == 'NotEqual':
                for node in layer.get('inbound_nodes', []):
                    if len(node['args']) > 1 and isinstance(node['args'][1], int):
                        node['args'] = [node['args'][0]]
    
    import tempfile
    import shutil
    
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        tmp_path = tmp.name
    
    shutil.copy2(filepath, tmp_path)
    
    with h5py.File(tmp_path, 'r+') as f:
        del f.attrs['model_config']
        f.attrs['model_config'] = json.dumps(model_config)
    
    model = keras.models.load_model(
        tmp_path,
        custom_objects={'NotEqual': NotEqual},
        compile=False
    )
    
    import os
    os.remove(tmp_path)
    
    return model

n_folds = 5
fold_predictions = []

for fold in range(n_folds):
    model_path = f'/kaggle/input/lmsys-gru/gru_model_fold_{fold}.h5'
    print(f"Loading fold {fold} from {model_path}")
    try:
        model = custom_load_model(model_path)
        predictions = model.predict(X_test, batch_size=32, verbose=0)
        fold_predictions.append(predictions)
        print(f"Fold {fold} predictions shape:", predictions.shape)
    except Exception as e:
        print(f"Failed to load fold {fold}: {repr(e)}")
        import traceback
        traceback.print_exc()

if fold_predictions:
    avg_predictions = tf.reduce_mean(fold_predictions, axis=0).numpy()
    print(f"\nFinal averaged predictions shape: {avg_predictions.shape}")
else:
    print("\nNo models loaded successfully!")

## === cell 6
print("\n" + "="*80)
print("AVERAGING PREDICTIONS")
print("="*80)

avg_predictions = np.mean(fold_predictions, axis=0)
print("Average predictions shape:", avg_predictions.shape)


## === cell 7
print("\n" + "="*80)
print("CREATING SUBMISSION")
print("="*80)

submission = sample_submission.copy()
submission['winner_model_a'] = avg_predictions[:, 0]
submission['winner_model_b'] = avg_predictions[:, 1]
submission['winner_tie'] = avg_predictions[:, 2]

output_path = '/kaggle/working/submission.csv'
submission.to_csv(output_path, index=False)
print("Saved:", output_path)
print("Shape:", submission.shape)


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/1453528320.py in <cell line: 0>()
      4 
      5 submission = sample_submission.copy()
----> 6 submission['winner_model_a'] = avg_predictions[:, 0]
      7 submission['winner_model_b'] = avg_predictions[:, 1]
      8 submission['winner_tie'] = avg_predictions[:, 2]

IndexError: invalid index to scalar variable.

## === cell 8
print("\n" + "="*80)
print("SUBMISSION PREVIEW")
print("="*80)
print(submission.head(20))


## === cell 9
print("\n" + "="*80)
print("PREDICTION STATISTICS")
print("="*80)

print(f"Model A - Mean: {avg_predictions[:, 0].mean():.4f}, Std: {avg_predictions[:, 0].std():.4f}, Min: {avg_predictions[:, 0].min():.4f}, Max: {avg_predictions[:, 0].max():.4f}")
print(f"Model B - Mean: {avg_predictions[:, 1].mean():.4f}, Std: {avg_predictions[:, 1].std():.4f}, Min: {avg_predictions[:, 1].min():.4f}, Max: {avg_predictions[:, 1].max():.4f}")
print(f"Tie     - Mean: {avg_predictions[:, 2].mean():.4f}, Std: {avg_predictions[:, 2].std():.4f}, Min: {avg_predictions[:, 2].min():.4f}, Max: {avg_predictions[:, 2].max():.4f}")


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/2619693006.py in <cell line: 0>()
      3 print("="*80)
      4 
----> 5 print(f"Model A - Mean: {avg_predictions[:, 0].mean():.4f}, Std: {avg_predictions[:, 0].std():.4f}, Min: {avg_predictions[:, 0].min():.4f}, Max: {avg_predictions[:, 0].max():.4f}")
      6 print(f"Model B - Mean: {avg_predictions[:, 1].mean():.4f}, Std: {avg_predictions[:, 1].std():.4f}, Min: {avg_predictions[:, 1].min():.4f}, Max: {avg_predictions[:, 1].max():.4f}")
      7 print(f"Tie     - Mean: {avg_predictions[:, 2].mean():.4f}, Std: {avg_predictions[:, 2].std():.4f}, Min: {avg_predictions[:, 2].min():.4f}, Max: {avg_predictions[:, 2].max():.4f}")

IndexError: invalid index to scalar variable.

## === cell 10
print("\n" + "="*80)
print("VERIFYING PROBABILITIES SUM TO 1")
print("="*80)

prob_sums = avg_predictions.sum(axis=1)
print("Min sum:", float(prob_sums.min()))
print("Max sum:", float(prob_sums.max()))
print("Mean sum:", float(prob_sums.mean()))


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
AxisError                                 Traceback (most recent call last)
/tmp/ipykernel_11/471232877.py in <cell line: 0>()
      3 print("="*80)
      4 
----> 5 prob_sums = avg_predictions.sum(axis=1)
      6 print("Min sum:", float(prob_sums.min()))
      7 print("Max sum:", float(prob_sums.max()))

/usr/local/lib/python3.11/dist-packages/numpy/core/_methods.py in _sum(a, axis, dtype, out, keepdims, initial, where)
     47 def _sum(a, axis=None, dtype=None, out=None, keepdims=False,
     48          initial=_NoValue, where=True):
---> 49     return umr_sum(a, axis, dtype, out, keepdims, initial, where)
     50 
     51 def _prod(a, axis=None, dtype=None, out=None, keepdims=False,

AxisError: axis 1 is out of bounds for array of dimension 0

## === cell 11
print("\n" + "="*80)
print("INFERENCE COMPLETED")
print("="*80)
print("Submission file saved:", output_path)

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2917646465.py in <cell line: 0>()
      2 print("INFERENCE COMPLETED")
      3 print("="*80)
----> 4 print("Submission file saved:", output_path)

NameError: name 'output_path' is not defined
