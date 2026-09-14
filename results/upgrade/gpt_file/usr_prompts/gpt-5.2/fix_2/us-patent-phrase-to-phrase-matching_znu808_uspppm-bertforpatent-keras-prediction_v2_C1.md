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
Given pairs of phrases (an `anchor` and a `target` phrase), build a model to rate how similar they are.  

## Metric
Pearson correlation coefficient.

## Submission Format
For each `id` (representing a pair of phrases) in the test set, you must predict the similarity `score`. The file should contain a header and have the following format:

```
id,score
4112d61851461f60,0
09e418c93a776564,0.25
36baf228038e314b,1
etc.

```

## Dataset
The scores are in the 0-1 range with increments of 0.25 with the following meanings:

- **1.0** - Very close match. This is typically an exact match except possibly for differences in conjugation, quantity (e.g. singular vs. plural), and addition or removal of stopwords (e.g. "the", "and", "or").
- **0.75** - Close synonym, e.g. "mobile phone" vs. "cellphone". This also includes abbreviations, e.g. "TCP" -> "transmission control protocol".
- **0.5** - Synonyms which don't have the same meaning (same function, same properties). This includes broad-narrow (hyponym) and narrow-broad (hypernym) matches.
- **0.25** - Somewhat related, e.g. the two phrases are in the same high level domain but are not synonyms. This also includes antonyms.
- **0.0** - Unrelated.

Files
-----

- **train.csv** - the training set, containing phrases, contexts, and their similarity scores
- **test.csv** - the test set set, identical in structure to the training set but without the score
- **sample_submission.csv** - a sample submission file in the correct format

Columns
-------

- `id` - a unique identifier for a pair of phrases
- `anchor` - the first phrase
- `target` - the second phrase
- `context` - the [CPC classification (version 2021.05)](https://en.wikipedia.org/wiki/Cooperative_Patent_Classification), which indicates the subject within which the similarity is to be scored
- `score` - the similarity. This is sourced from a combination of one or more manual expert ratings.

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        input/
            description.md (118 lines)
            sample_submission.csv (3649 lines)
            sample_submission.csv.zip (38.3 kB)
            test.csv (3649 lines)
            test.csv.zip (86.4 kB)
            train.csv (32826 lines)
            train.csv.zip (790.5 kB)
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
        working/
            us-patent-phrase-to-phrase-matching/
                description.md (118 lines)
                sample_submission.csv (3649 lines)
                ... and 5 other files
                us-patent-phrase-to-phrase-matching/
```

-> data/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> data/us-patent-phrase-to-phrase-matching/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> data/us-patent-phrase-to-phrase-matching/test.csv has 3648 rows and 4 columns.
The columns are: id, anchor, target, context

-> data/us-patent-phrase-to-phrase-matching/train.csv has 32825 rows and 5 columns.
The columns are: id, anchor, target, context, score

-> input/sample_submission.csv has 3648 rows and 2 columns.
The columns are: id, score

-> (stopped after 10 files for performance)

# 5. Target score

0.8046283730655472

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from transformers import BertTokenizerFast

from tqdm import tqdm

np.random.seed(42)
tf.random.set_seed(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
vocab_path = "/kaggle/input/google-bfp-pt/bert_for_patents_vocab_39k.txt"
ft_model_dir = "/kaggle/input/usppm-bft-ft-v1/"
test_data_path = "/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv"
sample_sub_path = (
    "/kaggle/input/us-patent-phrase-to-phrase-matching/sample_submission.csv"
)

MAX_SEQ_LEN = 40

print("Exists vocab_path:", os.path.exists(vocab_path), vocab_path)
print("Exists ft_model_dir:", os.path.exists(ft_model_dir), ft_model_dir)
print("Exists test_data_path:", os.path.exists(test_data_path), test_data_path)
print("Exists sample_sub_path:", os.path.exists(sample_sub_path), sample_sub_path)



## === cell 2
if os.path.isfile(vocab_path):
    tokenizer = BertTokenizerFast(vocab_file=vocab_path, do_lower_case=True)
    print("Loaded tokenizer from vocab_file:", vocab_path)
else:
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    print("Fallback tokenizer: bert-base-uncased")

pad_idx = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
print("Padding token:", tokenizer.pad_token, "->", pad_idx)



## === cell 3
test_data = pd.read_csv(test_data_path, sep=",")
print(test_data.head())
print("Test rows:", len(test_data))



## === cell 4
test_x_tokens = []
test_x_indices = []
test_x_segments = []

for row in tqdm(
    test_data.itertuples(index=False), total=len(test_data), desc="Test-data"
):
    anchor = row.anchor
    target = row.target

    token_anchor = tokenizer.encode(anchor, add_special_tokens=False)
    token_target = tokenizer.encode(target, add_special_tokens=False)

    ids = tokenizer.build_inputs_with_special_tokens(token_anchor, token_target)
    segments = tokenizer.create_token_type_ids_from_sequences(
        token_anchor, token_target
    )

    test_x_tokens.append(tokenizer.decode(ids))
    test_x_indices.append(ids)
    test_x_segments.append(segments)

test_x_indices = tf.keras.preprocessing.sequence.pad_sequences(
    test_x_indices, padding="post", truncating="post", maxlen=MAX_SEQ_LEN, value=pad_idx
)
test_x_segments = tf.keras.preprocessing.sequence.pad_sequences(
    test_x_segments, padding="post", truncating="post", maxlen=MAX_SEQ_LEN, value=0
)

test_x_indices = np.asarray(test_x_indices, dtype=np.int32)
test_x_segments = np.asarray(test_x_segments, dtype=np.int32)
test_x = [test_x_indices, test_x_segments]

print("Inputs shapes:", test_x[0].shape, test_x[1].shape)
print("Example decoded:", test_x_tokens[0])




## === cell 5
def _load_model_keras3_compatible(model_dir: str):
    if os.path.isfile(model_dir) and (
        model_dir.endswith(".keras") or model_dir.endswith(".h5")
    ):
        return keras.models.load_model(model_dir)

    if os.path.isdir(model_dir):
        try:
            layer = keras.layers.TFSMLayer(model_dir, call_endpoint="serving_default")
        except Exception as e:
            loaded = tf.saved_model.load(model_dir)
            sigs = list(loaded.signatures.keys())
            raise RuntimeError(
                f"Could not create TFSMLayer with call_endpoint='serving_default'. "
                f"Available signatures: {sigs}. Original error: {e}"
            )

        input_ids = keras.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="input_ids")
        token_type_ids = keras.Input(
            shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="token_type_ids"
        )

        try:
            outputs = layer({"input_ids": input_ids, "token_type_ids": token_type_ids})
        except Exception:
            try:
                outputs = layer(
                    {"input_word_ids": input_ids, "input_type_ids": token_type_ids}
                )
            except Exception:
                outputs = layer([input_ids, token_type_ids])

        if isinstance(outputs, dict):
            for k in ["outputs", "logits", "predictions", "output_0"]:
                if k in outputs:
                    outputs = outputs[k]
                    break
            else:
                outputs = outputs[sorted(outputs.keys())[0]]

        return keras.Model(inputs=[input_ids, token_type_ids], outputs=outputs)

    raise ValueError(f"Model path not found or unsupported: {model_dir}")


usppm_bfp_ft_model = _load_model_keras3_compatible(ft_model_dir)
usppm_bfp_ft_model.summary()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3213738305.py in <cell line: 0>()
     55 
     56 
---> 57 usppm_bfp_ft_model = _load_model_keras3_compatible(ft_model_dir)
     58 usppm_bfp_ft_model.summary()
     59 

/tmp/ipykernel_11/3213738305.py in _load_model_keras3_compatible(model_dir)
     52         return keras.Model(inputs=[input_ids, token_type_ids], outputs=outputs)
     53 
---> 54     raise ValueError(f"Model path not found or unsupported: {model_dir}")
     55 
     56 

ValueError: Model path not found or unsupported: /kaggle/input/usppm-bft-ft-v1/

## === cell 6
pred = usppm_bfp_ft_model.predict(test_x, batch_size=256, verbose=1)

pred = np.asarray(pred)
if pred.ndim > 1:
    pred = pred.reshape(pred.shape[0], -1)
    pred = pred[:, 0]
pred = pred.astype(np.float32)

print("Pred shape:", pred.shape, "min/max:", float(pred.min()), float(pred.max()))



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3672419321.py in <cell line: 0>()
      1 # Predict
----> 2 pred = usppm_bfp_ft_model.predict(test_x, batch_size=256, verbose=1)
      3 
      4 # Ensure 1D float array
      5 pred = np.asarray(pred)

NameError: name 'usppm_bfp_ft_model' is not defined

## === cell 7
submission = pd.read_csv(sample_sub_path)
submission["score"] = np.clip(pred, 0.0, 1.0)

if len(submission) != len(test_data):
    raise RuntimeError(
        f"Row mismatch: submission has {len(submission)} rows but test has {len(test_data)} rows"
    )

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3226971335.py in <cell line: 0>()
      1 # Create submission
      2 submission = pd.read_csv(sample_sub_path)
----> 3 submission["score"] = np.clip(pred, 0.0, 1.0)
      4 
      5 # Safety: align lengths

NameError: name 'pred' is not defined
