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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from transformers import BertTokenizer
from tqdm import tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
vocab_path = "/kaggle/input/google-bfp-pt/bert_for_patents_vocab_39k.txt"
ft_model_dir = "/kaggle/input/usppm-bft-ft-v1/"
test_data_path = "/kaggle/input/us-patent-phrase-to-phrase-matching/test.csv"

MAX_SEQ_LEN = 40

## === cell 4
tokenizer = BertTokenizer(vocab_file=vocab_path, do_lower_case=True)
pad_idx = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
print(tokenizer)
print("Padding token index : ", pad_idx)

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1283219157.py in <cell line: 0>()
----> 1 tokenizer = BertTokenizer(vocab_file=vocab_path, do_lower_case=True)
      2 pad_idx = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
      3 print(tokenizer)
      4 print("Padding token index : ", pad_idx)

/usr/local/lib/python3.11/dist-packages/transformers/models/bert/tokenization_bert.py in __init__(self, vocab_file, do_lower_case, do_basic_tokenize, never_split, unk_token, sep_token, pad_token, cls_token, mask_token, tokenize_chinese_chars, strip_accents, clean_up_tokenization_spaces, **kwargs)
    113     ):
    114         if not os.path.isfile(vocab_file):
--> 115             raise ValueError(
    116                 f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a Google pretrained"
    117                 " model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"

ValueError: Can't find a vocabulary file at path '/kaggle/input/google-bfp-pt/bert_for_patents_vocab_39k.txt'. To load the vocabulary from a Google pretrained model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`

## === cell 6
test_data = pd.read_csv(test_data_path, sep=',')
print(test_data[:5])

## === cell 8
test_x_tokens = []
test_x_indices = []
test_x_segments = []

for data in tqdm(test_data.values, desc="Test-data"):
    anchor = data[1]
    target = data[2]
    
    token_anchor = tokenizer.encode(anchor,add_special_tokens=False)
    token_target = tokenizer.encode(target,add_special_tokens=False)
    
    ids = tokenizer.build_inputs_with_special_tokens(token_anchor, token_target)
    segments = tokenizer.create_token_type_ids_from_sequences(token_anchor, token_target)

    test_x_tokens.append(tokenizer.decode(ids))
    test_x_indices.append(ids)
    test_x_segments.append(segments)
    
test_x_indices = tf.keras.preprocessing.sequence.pad_sequences(test_x_indices, padding="post", maxlen=MAX_SEQ_LEN, value=pad_idx)    
test_x_segments = tf.keras.preprocessing.sequence.pad_sequences(test_x_segments, padding="post", maxlen=MAX_SEQ_LEN, value=pad_idx)    
test_x_indices = np.array(test_x_indices)
test_x_segments = np.array(test_x_segments)
test_x = [test_x_indices, test_x_segments]

print(test_x[0].shape, test_x[1].shape)
print(test_x_tokens[:3])
print(test_x[0][:3])
print(test_x[1][:3])

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1672836837.py in <cell line: 0>()
      7     target = data[2]
      8 
----> 9     token_anchor = tokenizer.encode(anchor,add_special_tokens=False)
     10     token_target = tokenizer.encode(target,add_special_tokens=False)
     11 

NameError: name 'tokenizer' is not defined

## === cell 10
usppm_bfp_ft_model = keras.models.load_model(ft_model_dir)
usppm_bfp_ft_model.summary()

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3759289808.py in <cell line: 0>()
----> 1 usppm_bfp_ft_model = keras.models.load_model(ft_model_dir)
      2 usppm_bfp_ft_model.summary()

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    204         )
    205     else:
--> 206         raise ValueError(
    207             f"File format not supported: filepath={filepath}. "
    208             "Keras 3 only supports V3 `.keras` files and "

ValueError: File format not supported: filepath=/kaggle/input/usppm-bft-ft-v1/. Keras 3 only supports V3 `.keras` files and legacy H5 format files (`.h5` extension). Note that the legacy SavedModel format is not supported by `load_model()` in Keras 3. In order to reload a TensorFlow SavedModel as an inference-only layer in Keras 3, use `keras.layers.TFSMLayer(/kaggle/input/usppm-bft-ft-v1/, call_endpoint='serving_default')` (note that your `call_endpoint` might have a different name).

## === cell 12
pred = usppm_bfp_ft_model.predict(test_x)

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1855633231.py in <cell line: 0>()
----> 1 pred = usppm_bfp_ft_model.predict(test_x)

NameError: name 'usppm_bfp_ft_model' is not defined

## === cell 14
submission = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/sample_submission.csv")
submission['score'] = pred
submission['score'] = submission.score.apply(lambda x: 0 if x < 0 else x)
submission['score'] = submission.score.apply(lambda x: 1 if x > 1 else x)
submission.to_csv("submission.csv",index=False)
submission

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3030894002.py in <cell line: 0>()
      1 submission = pd.read_csv("/kaggle/input/us-patent-phrase-to-phrase-matching/sample_submission.csv")
----> 2 submission['score'] = pred
      3 submission['score'] = submission.score.apply(lambda x: 0 if x < 0 else x)
      4 submission['score'] = submission.score.apply(lambda x: 1 if x > 1 else x)
      5 submission.to_csv("submission.csv",index=False)

NameError: name 'pred' is not defined
