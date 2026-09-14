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

1.092333390084535

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

df_test_data = pd.read_csv('/kaggle/input/lmsys-chatbot-arena/test.csv')

df_sample_submission = pd.read_csv('/kaggle/input/lmsys-chatbot-arena/sample_submission.csv')

print("Test Shape:", df_test_data.shape)
print("Sample Submission Shape:", df_sample_submission.shape)

## === cell 3
df_test_data['text_concatenated'] = df_test_data['prompt'] + ' [SEP] ' + df_test_data['response_a'] + ' [SEP] ' + df_test_data['response_b']

## === cell 4
SEQUENCE_LENGTH = 512

print("\n" + "="*80)
print("LOADING GRU MODELS AND TOKENIZER")
print("="*80)

with open('/kaggle/input/lmsys-hemkesh-10040-gru/tokenizer_gru.pkl', 'rb') as file_handler:
    text_tokenizer = pickle.load(file_handler)

test_sequences = text_tokenizer.texts_to_sequences(df_test_data['text_concatenated'])
test_padded_inputs = pad_sequences(test_sequences, maxlen=SEQUENCE_LENGTH, padding='post', truncating='post')

print("X_test shape:", test_padded_inputs.shape)

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2293328271.py in <cell line: 0>()
      5 print("="*80)
      6 
----> 7 with open('/kaggle/input/lmsys-hemkesh-10040-gru/tokenizer_gru.pkl', 'rb') as file_handler:
      8     text_tokenizer = pickle.load(file_handler)
      9 

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/lmsys-hemkesh-10040-gru/tokenizer_gru.pkl'

## === cell 5
import tensorflow as tf
from tensorflow import keras
import json
import h5py
import tempfile
import shutil
import os
import traceback

class NonZeroMask(keras.layers.Layer):
    def __init__(self, convert_to_float=True, **kwargs):
        super().__init__(**kwargs)
        self.convert_to_float = convert_to_float
    
    def call(self, input_tensor):
        if isinstance(input_tensor, (list, tuple)):
            input_tensor = input_tensor[0]
        
        output_tensor = tf.math.not_equal(input_tensor, 0)
        if self.convert_to_float:
            return tf.cast(output_tensor, tf.float32)
        return output_tensor
    
    def get_config(self):
        config = super().get_config()
        config.update({"cast_to_float": self.convert_to_float})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

def load_model_with_custom_config(model_file_path):
    
    with h5py.File(model_file_path, 'r') as file_handler:
        architecture_config = json.loads(file_handler.attrs['model_config'])
        
        for network_layer in architecture_config['config']['layers']:
            if network_layer['class_name'] == 'NotEqual': # Checking against original class name in saved file
                for graph_node in network_layer.get('inbound_nodes', []):
                    if len(graph_node['args']) > 1 and isinstance(graph_node['args'][1], int):
                        graph_node['args'] = [graph_node['args'][0]]
    
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file_obj:
        temp_file_path = temp_file_obj.name
    
    shutil.copy2(model_file_path, temp_file_path)
    
    with h5py.File(temp_file_path, 'r+') as file_handler:
        del file_handler.attrs['model_config']
        file_handler.attrs['model_config'] = json.dumps(architecture_config)
    
    loaded_model = keras.models.load_model(
        temp_file_path,
        custom_objects={'NotEqual': NonZeroMask}, 
        compile=False
    )
    
    os.remove(temp_file_path)
    
    return loaded_model

k_fold_count = 5
ensemble_predictions_list = []

for current_fold_index in range(k_fold_count):
    input_model_path = f'/kaggle/input/lmsys-hemkesh-10040-gru/gru_model_fold_{current_fold_index}.h5'
    print(f"Loading fold {current_fold_index} from {input_model_path}")
    try:
        current_gru_model = load_model_with_custom_config(input_model_path)
        
        fold_output_probs = current_gru_model.predict(test_padded_inputs, batch_size=32, verbose=0)
        ensemble_predictions_list.append(fold_output_probs)
        print(f"Fold {current_fold_index} predictions shape:", fold_output_probs.shape)
        
    except Exception as error_msg:
        print(f"Failed to load fold {current_fold_index}: {repr(error_msg)}")
        traceback.print_exc()

if ensemble_predictions_list:
    final_ensemble_probabilities = tf.reduce_mean(ensemble_predictions_list, axis=0).numpy()
    print(f"\nFinal averaged predictions shape: {final_ensemble_probabilities.shape}")
else:
    print("\nNo models loaded successfully!")

## === cell 6
print("\n" + "="*80)
print("AVERAGING PREDICTIONS")
print("="*80)

final_ensemble_probabilities = np.mean(ensemble_predictions_list, axis=0)

print("Average predictions shape:", final_ensemble_probabilities.shape)

## === cell 7
print("\n" + "="*80)
print("CREATING SUBMISSION")
print("="*80)

df_final_submission = df_sample_submission.copy()

df_final_submission['winner_model_a'] = final_ensemble_probabilities[:, 0]
df_final_submission['winner_model_b'] = final_ensemble_probabilities[:, 1]
df_final_submission['winner_tie'] = final_ensemble_probabilities[:, 2]

submission_file_path = '/kaggle/working/submission.csv'
df_final_submission.to_csv(submission_file_path, index=False)

print("Saved:", submission_file_path)
print("Shape:", df_final_submission.shape)

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/745836255.py in <cell line: 0>()
      5 df_final_submission = df_sample_submission.copy()
      6 
----> 7 df_final_submission['winner_model_a'] = final_ensemble_probabilities[:, 0]
      8 df_final_submission['winner_model_b'] = final_ensemble_probabilities[:, 1]
      9 df_final_submission['winner_tie'] = final_ensemble_probabilities[:, 2]

IndexError: invalid index to scalar variable.

## === cell 8
print("\n" + "="*80)
print("SUBMISSION PREVIEW")
print("="*80)
print(df_final_submission.head(20))

## === cell 9
print("\n" + "="*80)
print("PREDICTION STATISTICS")
print("="*80)
print(f"Model A - Mean: {final_ensemble_probabilities[:, 0].mean():.4f}, Std: {final_ensemble_probabilities[:, 0].std():.4f}, Min: {final_ensemble_probabilities[:, 0].min():.4f}, Max: {final_ensemble_probabilities[:, 0].max():.4f}")
print(f"Model B - Mean: {final_ensemble_probabilities[:, 1].mean():.4f}, Std: {final_ensemble_probabilities[:, 1].std():.4f}, Min: {final_ensemble_probabilities[:, 1].min():.4f}, Max: {final_ensemble_probabilities[:, 1].max():.4f}")
print(f"Tie     - Mean: {final_ensemble_probabilities[:, 2].mean():.4f}, Std: {final_ensemble_probabilities[:, 2].std():.4f}, Min: {final_ensemble_probabilities[:, 2].min():.4f}, Max: {final_ensemble_probabilities[:, 2].max():.4f}")

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/3477495526.py in <cell line: 0>()
      2 print("PREDICTION STATISTICS")
      3 print("="*80)
----> 4 print(f"Model A - Mean: {final_ensemble_probabilities[:, 0].mean():.4f}, Std: {final_ensemble_probabilities[:, 0].std():.4f}, Min: {final_ensemble_probabilities[:, 0].min():.4f}, Max: {final_ensemble_probabilities[:, 0].max():.4f}")
      5 print(f"Model B - Mean: {final_ensemble_probabilities[:, 1].mean():.4f}, Std: {final_ensemble_probabilities[:, 1].std():.4f}, Min: {final_ensemble_probabilities[:, 1].min():.4f}, Max: {final_ensemble_probabilities[:, 1].max():.4f}")
      6 print(f"Tie     - Mean: {final_ensemble_probabilities[:, 2].mean():.4f}, Std: {final_ensemble_probabilities[:, 2].std():.4f}, Min: {final_ensemble_probabilities[:, 2].min():.4f}, Max: {final_ensemble_probabilities[:, 2].max():.4f}")

IndexError: invalid index to scalar variable.

## === cell 10
print("\n" + "="*80)
print("VERIFYING PROBABILITIES SUM TO 1")
print("="*80)

probability_row_sums = final_ensemble_probabilities.sum(axis=1)

print("Min sum:", float(probability_row_sums.min()))
print("Max sum:", float(probability_row_sums.max()))
print("Mean sum:", float(probability_row_sums.mean()))

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
AxisError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4027724137.py in <cell line: 0>()
      3 print("="*80)
      4 
----> 5 probability_row_sums = final_ensemble_probabilities.sum(axis=1)
      6 
      7 print("Min sum:", float(probability_row_sums.min()))

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
print("Submission file saved:", submission_file_path)

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2552874089.py in <cell line: 0>()
      2 print("INFERENCE COMPLETED")
      3 print("="*80)
----> 4 print("Submission file saved:", submission_file_path)

NameError: name 'submission_file_path' is not defined
